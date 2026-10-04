"""New-squares transfer with a GENERIC attention readout, on either trunk.

The question: does a trunk whose representation supports the board show a bias
toward GEOMETRICALLY COHERENT new squares, where Othello-GPT shows none
(IL_prob 1.02x, IL_acc 1.28x -- see experiments/new_squares/gpt_cond_00*.json)?

Why a GENERIC attention readout rather than a ray-masked one: square-attention
reaches 95% ray-attention mass unaided (ray_attn_test.py), so the geometry is
LEARNED from the legality task, not hand-coded.  Nothing about lines is built
in, so a coherent/incoherent difference is attributable to the TRUNK -- which
is the whole point of running the same readout on both trunks.

Protocol, matched across trunks:
  phase 1  pretrain the readout on STANDARD legality from real games.  Output
           space is already 72 cells (9 rows x 8 cols); row 8 is never legal
           here, so its embedding stays untrained -- exactly the transfer
           question.
  phase 2  fine-tune on a condition's games, scoring the shared test manifest
           on a log schedule.  IL_prob/IL_acc come from
           new_squares_data.score_manifest, the same function the GPT and MLP
           arms used.

The trunk is frozen and pinned to eval() in both phases: these models carry
dropout 0.1, which would otherwise inject noise into the features under test.
"""
import argparse, json, os, pickle, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
from new_squares_data import (N_CELLS, NEW_SQUARE_IDS, CENTER_CELLS,
                              condition_dir, score_manifest)

PLAYABLE = [c for c in range(N_CELLS) if c not in CENTER_CELLS]   # 68 cells
CELL_TO_TOK = {c: i + 1 for i, c in enumerate(PLAYABLE)}          # 1..68, 0 = pad
TOK_TO_CELL = {t: c for c, t in CELL_TO_TOK.items()}
VOCAB = len(CELL_TO_TOK) + 1                                      # 69


# ---------------------------------------------------------------- trunks
class Trunk(nn.Module):
    """Frozen feature extractor: tokens -> resid_post of block `layer`.

    The readout applies its own LayerNorm, so no ln_f is used here -- matching
    how a linear probe reads the residual stream.
    """

    def __init__(self, gpt, layer):
        super().__init__()
        self.g = gpt; self.layer = layer
        for p in self.g.parameters():
            p.requires_grad = False
        self.g.eval()

    def train(self, mode=True):
        super().train(mode); self.g.eval(); return self

    def forward(self, idx):
        g = self.g
        with torch.no_grad():
            x = g.drop(g.tok_emb(idx) + g.pos_emb[:, :idx.size(1), :])
            for blk in g.blocks[:self.layer]:
                x = blk(x)
            return x


def load_trunk(kind, ckpt, layer, dev):
    sd = torch.load(ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd:          # board-state ckpt
        cfg = GPTConfig(**sd['cfg']); sdw = sd['model']
    else:                                               # raw Othello-GPT state
        sdw = sd
        v, d = sdw['tok_emb.weight'].shape
        nl = 1 + max(int(k.split('.')[1]) for k in sdw if k.startswith('blocks.'))
        cfg = GPTConfig(v, sdw['pos_emb'].shape[1], n_layer=nl, n_head=8, n_embd=d)
    g = GPT(cfg)
    # Drop head.* : GPTBoardState's head is Linear(d, 64*3) where GPT's is
    # Linear(d, vocab), and strict=False forgives missing/unexpected keys but
    # NOT shape mismatches.  Trunk only runs tok_emb + pos_emb + blocks, so the
    # head is never needed.
    sdw = {k: v for k, v in sdw.items() if not k.startswith('head.')}
    g.load_state_dict(sdw, strict=False)
    # expand the token embedding 61 -> 69 so the new-square moves can be INPUT.
    old = g.tok_emb.weight.shape[0]
    if old < VOCAB:
        emb = nn.Embedding(VOCAB, cfg.n_embd)
        emb.weight.data[:old] = g.tok_emb.weight.data
        emb.weight.data[old:].normal_(0.0, 0.02)
        g.tok_emb = emb
        print(f'  expanded token embedding {old} -> {VOCAB}', flush=True)
    print(f'  {kind}: {cfg.n_layer} layers, d{cfg.n_embd}, reading block {layer}',
          flush=True)
    return Trunk(g, layer).to(dev), cfg.n_embd, cfg.block_size


# ---------------------------------------------------------------- readout
class AttnReadout(nn.Module):
    """Generic self-attention over the 72 cells of a 9x8 board.

    Tokens are projected from the trunk state; row/col embeddings are the only
    positional information.  No ray mask, no line structure -- whatever geometry
    appears is learned from the legality task.
    """

    def __init__(self, d_in, dm=128, layers=2, heads=4):
        super().__init__()
        self.dm = dm
        self.ln_in = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, N_CELLS * dm)
        self.row = nn.Embedding(9, dm)                 # 9 rows: the new row is 8
        self.col = nn.Embedding(8, dm)
        enc = nn.TransformerEncoderLayer(dm, heads, 4 * dm, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(dm, 1)
        self.register_buffer('rid', torch.arange(N_CELLS) // 8)
        self.register_buffer('cid', torch.arange(N_CELLS) % 8)

    def forward(self, h):                              # h (B, d_in) -> (B, 72)
        t = self.proj(self.ln_in(h)).view(-1, N_CELLS, self.dm)
        t = t + self.row(self.rid) + self.col(self.cid)
        return self.out(self.tr(t)).squeeze(-1)


def soft_ce(logits, legal):
    q = legal.float(); s = q.sum(1, keepdim=True).clamp_min(1e-9); q = q / s
    return -(q * F.log_softmax(logits, 1)).sum(1).mean()


# ---------------------------------------------------------------- data
def positions_from_real_games(data_dir, n_games, ply_max=54):
    """(prefix_tokens, legal_mask) from REAL games -- phase 1."""
    import glob
    X, L = [], []
    files = sorted(glob.glob(os.path.join(data_dir, '*.pickle')))[:n_games // 90000 + 1]
    got = 0
    for f in files:
        for g in pickle.load(open(f, 'rb')):
            if len(g) < 60:
                continue
            bd = OthelloBoardState()
            for ply in range(ply_max):
                if ply >= 4:
                    lg = bd.get_valid_moves()
                    if lg:
                        m = np.zeros(N_CELLS, bool)
                        for c in lg:
                            m[c] = True
                        X.append([CELL_TO_TOK[c] for c in g[:ply]]); L.append(m)
                bd.update([g[ply]])
            got += 1
            if got >= n_games:
                return X, L
    return X, L


def positions_from_condition(cdir, max_pos):
    """(prefix_tokens, legal_mask) from a condition's train_records -- phase 2."""
    games = pickle.load(open(os.path.join(cdir, 'train_games.pickle'), 'rb'))
    recs = pickle.load(open(os.path.join(cdir, 'train_records.pickle'), 'rb'))
    X, L = [], []
    for g, rs in zip(games, recs):
        for r in rs:
            m = np.zeros(N_CELLS, bool)
            for c in r['all_legal']:
                if 0 <= c < N_CELLS:
                    m[c] = True
            if not m.any():
                continue
            X.append([CELL_TO_TOK[c] for c in g[:r['prefix_len']] if c in CELL_TO_TOK])
            L.append(m)
            if len(X) >= max_pos:
                return X, L
    return X, L


def batches(X, L, bs, block, dev, shuffle=True):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        ln = max(1, min(block, max(len(X[k]) for k in j)))
        t = np.zeros((len(j), ln), np.int64)
        last = np.zeros(len(j), np.int64)
        for a, k in enumerate(j):
            s = X[k][-ln:]
            t[a, :len(s)] = s; last[a] = max(0, len(s) - 1)
        yield (torch.from_numpy(t).to(dev), torch.from_numpy(last).to(dev),
               torch.from_numpy(np.stack([L[k] for k in j])).to(dev))


def step(trunk, head, x, last, y, opt=None):
    h = trunk(x)[torch.arange(len(x), device=x.device), last]
    lg = head(h)
    loss = soft_ce(lg, y)
    if opt is not None:
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return loss.item()


def _scores_batched(trunk, head, block, dev, prefixes, bs=256):
    """Score many prefixes at once.  score_manifest calls score_fn ONE position
    at a time, and the manifest holds 10,000 of them -- 100,000 batch-1 forwards
    per run at 10 eval points, which is most of the runtime and leaves the GPU
    idle.  Precomputing in batches keeps score_manifest untouched (so the
    metrics stay identical to the GPT and MLP arms) while doing the work 256 at
    a time.

    Right-padding is equivalent to the unbatched path: the trunk is causal, so
    position len(toks)-1 depends only on tokens 0..len(toks)-1, and each token
    receives the same pos_emb index either way.
    """
    out = {}
    for i in range(0, len(prefixes), bs):
        chunk = prefixes[i:i + bs]
        toks = [[CELL_TO_TOK[m] for m in pre if m in CELL_TO_TOK][-block:]
                for pre in chunk]
        ln = max(1, max(len(t) for t in toks))
        x = np.zeros((len(toks), ln), np.int64)
        last = np.zeros(len(toks), np.int64)
        for a, t in enumerate(toks):
            if t:
                x[a, :len(t)] = t; last[a] = len(t) - 1
        xt = torch.from_numpy(x).to(dev)
        lt = torch.from_numpy(last).to(dev)
        with torch.no_grad():
            h = trunk(xt)[torch.arange(len(toks), device=dev), lt]
            p = torch.softmax(head(h).float(), -1).cpu().numpy()
        for a, pre in enumerate(chunk):
            out[pre] = p[a] if toks[a] else np.zeros(N_CELLS, np.float32)
    return out


def make_score_fn(trunk, head, block, dev, manifest=None):
    cache = {}
    if manifest is not None:
        pres = [tuple(pos['game_prefix']) for st in ('IL', 'LL')
                for pos in manifest.get(st, [])]
        cache = _scores_batched(trunk, head, block, dev,
                                list(dict.fromkeys(pres)))

    def score_fn(prefix):
        k = tuple(prefix)
        if k in cache:
            return cache[k]
        toks = [CELL_TO_TOK[m] for m in prefix if m in CELL_TO_TOK][-block:]
        out = np.zeros(N_CELLS, np.float32)
        if not toks:
            return out
        x = torch.tensor([toks], dtype=torch.long, device=dev)
        with torch.no_grad():
            h = trunk(x)[:, -1]
            return torch.softmax(head(h)[0].float(), -1).cpu().numpy()
    return score_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trunk', choices=['ogpt', 'board'], required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--condition-id', type=int, required=True)
    ap.add_argument('--exp-dir', default='experiments/new_squares')
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--pre-games', type=int, default=4000)
    ap.add_argument('--pre-epochs', type=int, default=3)
    ap.add_argument('--max-pos', type=int, default=400000)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0,
                    help='seeds readout init and batch order.  Within a seed the '
                         'two conditions share one phase-1 readout, so they '
                         'differ ONLY in fine-tuning data; across seeds both '
                         'init and data order vary, which is what the error bars '
                         'need to capture.')
    ap.add_argument('--readout-cache', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    print(f'device {dev}  seed {a.seed}', flush=True)
    trunk, d_in, block = load_trunk(a.trunk, a.ckpt, a.layer, dev)
    head = AttnReadout(d_in).to(dev)
    print(f'  readout: {sum(p.numel() for p in head.parameters()):,} params',
          flush=True)

    # ---- phase 1: standard legality on real games ----
    t0 = time.time()
    if a.readout_cache and os.path.exists(a.readout_cache):
        head.load_state_dict(torch.load(a.readout_cache, map_location=dev))
        print(f'  loaded pretrained readout {a.readout_cache}', flush=True)
    else:
        Xp, Lp = positions_from_real_games(a.data_dir, a.pre_games)
        print(f'phase 1: {len(Xp):,} positions from {a.pre_games} real games '
              f'({time.time()-t0:.0f}s)', flush=True)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        for ep in range(a.pre_epochs):
            ls = [step(trunk, head, *b, opt) for b in
                  batches(Xp, Lp, a.bs, block, dev)]
            print(f'  pre ep{ep} loss {np.mean(ls[-20:]):.4f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        if a.readout_cache:
            torch.save(head.state_dict(), a.readout_cache)

    # ---- phase 2: fine-tune on the condition, scoring the shared manifest ----
    cdir = condition_dir(a.exp_dir, a.condition_id)
    man = json.load(open(os.path.join(cdir, 'test_manifest.json')))
    Xc, Lc = positions_from_condition(cdir, a.max_pos)
    print(f'\nphase 2: condition {a.condition_id}, {len(Xc):,} positions '
          f'({time.time()-t0:.0f}s)', flush=True)
    sched = [0, 5, 25, 50, 100, 200, 500, 1000, 2000, 5000]
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr)
    res = {'trunk': a.trunk, 'ckpt': a.ckpt, 'layer': a.layer, 'seed': a.seed,
           'condition_id': a.condition_id, 'readout': 'generic_attention',
           'n_positions': len(Xc), 'lr': a.lr, 'bs': a.bs,
           'eval_steps': [], 'IL_prob': [], 'IL_acc': [], 'LL_prob': [], 'LL_acc': [],
           # the two NORMALISED variants score_manifest provides: prob_frac is
           # scale-free (for cross-model comparison) and prob_per_target removes
           # the co-legality artifact -- coherent positions have more
           # simultaneously-legal new squares, which inflates raw IL_prob.
           'IL_prob_frac': [], 'IL_prob_per_target': [], 'IL_n': []}

    def ev(st):
        head.eval()
        r = score_manifest(make_score_fn(trunk, head, block, dev, man), man,
                           per_bucket=False)
        head.train()
        res['eval_steps'].append(st)
        for k in ('IL_prob', 'IL_acc', 'LL_prob', 'LL_acc',
                  'IL_prob_frac', 'IL_prob_per_target', 'IL_n'):
            res[k].append(float(r.get(k, float('nan'))))
        print('  step %6d  IL_prob %.4f  frac %.4f  per_tgt %.4f  IL_acc %.4f  '
              'LL_prob %.4f  (%.0fs)'
              % (st, res['IL_prob'][-1], res['IL_prob_frac'][-1],
                 res['IL_prob_per_target'][-1], res['IL_acc'][-1],
                 res['LL_prob'][-1], time.time() - t0), flush=True)

    ev(0)
    st = 0; done = False
    while not done:
        for b in batches(Xc, Lc, a.bs, block, dev):
            step(trunk, head, *b, opt); st += 1
            if st in sched:
                ev(st)
            if st >= sched[-1]:
                done = True; break
    if st not in res['eval_steps']:
        ev(st)
    res['elapsed_seconds'] = time.time() - t0
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2)
    print(f'\nwrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
