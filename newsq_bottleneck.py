"""New-squares transfer through an EXPLICIT 9x8 board bottleneck, both trunks.

Architecture (identical for both arms; only the trunk/decoder pair differs):

  move tokens -> FROZEN trunk -> h
     -> board decoder, 72x3        <- 64 original rows FROZEN and
        (the bottleneck)              board-supervised; 8 new rows trained
     -> the residual stream stops here: the readout sees ONLY these 216 numbers
     -> attention over 72 cells (9 rows x 8 cols)
     -> 68 move logits (60 original playable + 8 new)

Trained on NEXT-MOVE prediction only, matched to Othello-GPT's objective.  An
auxiliary board loss would break the comparison -- it could teach our arm the
new row's geometry directly rather than having the pretrained representation
generalise -- so new-square board accuracy is reported as a READ-ONLY
diagnostic and never optimised.

The bottleneck stays honest without that loss because the original 64x3 rows
are frozen from board supervision, and because the decoder is LINEAR: a rule's
firing condition is an AND over several cells, and a linear map cannot compute
AND.  So the 8 new rows can carry "is new square i occupied" but cannot
shortcut "these cells all hold value v".

  arm    trunk               frozen board decoder        board acc
  board  board_gpt_L6_20M    its own 64x3 head           99.63%
(both checkpoints give identical probe accuracy, so the TL->mingpt conversion
is faithful and the pairing is free.)
  ogpt   gpt_nanda_synthetic Nanda's probe, MODE 0       98.47%

Next-move prediction IS legality here: the games are random.choice(all_legal),
so the Bayes-optimal next-move distribution is uniform over the legal set.
"""
import argparse, json, os, pickle, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
from train_board_state_gpt import GPTBoardState, GPTBoardGrid
from new_squares_data import (N_CELLS, NEW_SQUARE_IDS, CENTER_CELLS, N_NEW,
                              rule_length, condition_dir, score_manifest)

PLAYABLE = [c for c in range(N_CELLS) if c not in CENTER_CELLS]      # 68
CELL_TO_TOK = {c: i + 1 for i, c in enumerate(PLAYABLE)}             # 1..68
TOK_TO_CELL = {t: c for c, t in CELL_TO_TOK.items()}
VOCAB = len(CELL_TO_TOK) + 1                                         # 69
NEWSET = set(NEW_SQUARE_IDS)


# ----------------------------------------------------------------- replay
def replay_72(moves, rules):
    """Per-ply MOVER-RELATIVE 72-square labels (0 empty, 1 theirs, 2 mine).

    Mirrors generate_game_with_new_squares exactly: a new square is PLACED and
    never flipped, and a standard move with no flips is placed directly with the
    turn handed over.  Returns labels[p] = the board AFTER move p, i.e. the
    target for the position whose prefix is moves[:p+1].
    """
    bd = OthelloBoardState()
    new_state = np.zeros(N_NEW, np.int32)
    out = []
    for mv in moves:
        my = 1 if bd.next_hand_color == 1 else -1
        if mv >= 64:
            new_state[mv - 64] = my
            bd.next_hand_color *= -1
        else:
            if bd.tentative_move(mv) != 0:
                bd.update([mv])
            else:
                bd.state[mv // 8, mv % 8] = my
                bd.next_hand_color *= -1
        mover = bd.next_hand_color
        lab = np.zeros(N_CELLS, np.int8)
        flat = bd.state.flatten()
        lab[:64][flat == -mover] = 1
        lab[:64][flat == mover] = 2
        lab[64:][new_state == -mover] = 1
        lab[64:][new_state == mover] = 2
        out.append(lab)
    return out


# ----------------------------------------------------------------- trunks
class Feat(nn.Module):
    """Frozen trunk -> the vector the board decoder reads.

    `board` reads ln_f(blocks(x)), where its own 64x3 head sits.
    `ogpt`  reads resid_post of block `layer`, where Nanda's probe sits (no
            ln_f -- that is how a residual-stream probe reads).
    The 8 new move tokens get their OWN trainable embedding: they are new
    symbols, and leaving them at random init makes the trunk see noise whenever
    a new square is played.
    """

    def __init__(self, g, kind, layer, n_old):
        super().__init__()
        self.g = g; self.kind = kind; self.layer = layer; self.n_old = n_old
        for p in self.g.parameters():
            p.requires_grad = False
        self.g.eval()
        self.new_emb = nn.Embedding(VOCAB - n_old, g.pos_emb.shape[-1])
        self.new_emb.weight.data.normal_(0.0, 0.02)

    def train(self, mode=True):
        super().train(mode); self.g.eval(); return self

    def forward(self, idx):
        g = self.g
        e = g.tok_emb(idx.clamp(max=self.n_old - 1))
        e = torch.where((idx >= self.n_old).unsqueeze(-1),
                        self.new_emb((idx - self.n_old).clamp(min=0)), e)
        x = g.drop(e + g.pos_emb[:, :idx.size(1), :])
        for blk in g.blocks[:self.layer]:
            x = blk(x)
        return g.ln_f(x) if self.kind == 'board' else x


def load_arm(kind, ckpt, probe_path, layer, dev):
    """-> Feat, frozen decoder W (64,3,d), d_model, block_size"""
    sd = torch.load(ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd:
        cfg = GPTConfig(**sd['cfg']); sdw = sd['model']
        head = sd.get('args', {}).get('head', 'flat')
        tmp = (GPTBoardGrid(cfg, rank=sd['args'].get('head_rank', 64))
               if head == 'grid' else GPTBoardState(cfg))
        tmp.load_state_dict(sdw)
        W = tmp.head.weight.detach().view(64, 3, cfg.n_embd).clone()
        nlayer = cfg.n_layer
    else:
        sdw = sd
        v, d = sdw['tok_emb.weight'].shape
        nlayer = 1 + max(int(k.split('.')[1]) for k in sdw if k.startswith('blocks.'))
        cfg = GPTConfig(v, sdw['pos_emb'].shape[1], n_layer=nlayer, n_head=8, n_embd=d)
        # MODE 0, not mode 2.  Measured per-square board accuracy at block 6
        # over plies 4-53: mode 0 = 98.47%, mode 1 = 46.86%, mode 2 = 75.68%,
        # parity-selected = 71.68%.  Mode 0 works at ALL plies.  Note
        # ogpt_intervention.py and ogpt_legal_mass_shift.py both use mode 2,
        # so the published alpha sweep used a 75.68% decoder when a 98.47% one
        # was available -- worth revisiting there.
        P = torch.load(probe_path, map_location='cpu')[0].detach()   # (d,8,8,3)
        W = P.permute(1, 2, 3, 0).reshape(64, 3, d).clone()          # (64,3,d)
    g = GPT(cfg)
    g.load_state_dict({k: v for k, v in sdw.items() if not k.startswith('head.')},
                      strict=False)
    n_old = g.tok_emb.weight.shape[0]
    L = nlayer if kind == 'board' else layer
    print(f'  {kind}: {nlayer} layers d{cfg.n_embd}, decoder reads '
          f'{"ln_f(final)" if kind == "board" else f"block {L}"}; '
          f'frozen decoder {tuple(W.shape)}', flush=True)
    return (Feat(g, kind, L, n_old).to(dev), W.to(dev), cfg.n_embd,
            cfg.block_size)


# ----------------------------------------------------------------- model
class Bottleneck(nn.Module):
    def __init__(self, Wfrozen, d, dm=128, layers=2, heads=4):
        super().__init__()
        self.register_buffer('W64', Wfrozen)                 # (64,3,d) frozen
        self.Wnew = nn.Parameter(torch.zeros(N_NEW, 3, d))   # 8 new rows
        self.Wnew.data.normal_(0.0, 0.02)
        self.dm = dm
        self.inp = nn.Linear(3, dm)
        self.row = nn.Embedding(9, dm); self.col = nn.Embedding(8, dm)
        enc = nn.TransformerEncoderLayer(dm, heads, 4 * dm, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(dm, 1)
        self.register_buffer('rid', torch.arange(N_CELLS) // 8)
        self.register_buffer('cid', torch.arange(N_CELLS) % 8)
        self.register_buffer('valid', torch.tensor(PLAYABLE))

    def board(self, h):                                      # -> (B,72,3)
        W = torch.cat([self.W64, self.Wnew], 0)
        return torch.einsum('bd,ckd->bck', h, W)

    def forward(self, h):
        b3 = self.board(h)
        # THE BOTTLENECK: only these 216 numbers continue.  h stops here.
        tok = self.inp(b3) + self.row(self.rid) + self.col(self.cid)
        lg = self.out(self.tr(tok)).squeeze(-1)              # (B,72)
        pad = torch.full_like(lg[:, :1], -1e4)
        return torch.cat([pad, lg[:, self.valid]], 1), b3    # (B,69), (B,72,3)


# ----------------------------------------------------------------- data
def collect(cdir, n_new_tgt, n_old_tgt, block, want_labels=True):
    """(prefix tokens, next-move token, new-square labels) from a condition.

    EXPOSURE-MATCHED on the training signal that matters for next-move
    prediction: equal counts of positions whose TARGET is a new square and
    whose target is not.  Incoherent rules fire far more often, so an unmatched
    pool hands it substantially more new-square signal -- which alone produces a
    coherent/incoherent difference (measured: 60.0% vs 44.1% of positions had a
    new square legal under a flat cap).

    Prefixes are capped at `block` tokens and LONGER GAMES ARE TRUNCATED FROM
    THE END, not slid: condition games run to 68 moves but the trunk's
    block_size is 59, and taking the last 59 would shift every position index
    away from the ply it meant during pretraining.
    """
    games = pickle.load(open(os.path.join(cdir, 'train_games.pickle'), 'rb'))
    rules = json.load(open(os.path.join(cdir, 'rules.json')))
    A, B = [], []
    for g in games:
        labs = replay_72(g, rules) if want_labels else None
        for p in range(1, min(len(g), block + 1)):
            tgt = g[p]
            if tgt not in CELL_TO_TOK:
                continue
            item = ([CELL_TO_TOK[c] for c in g[:p]], CELL_TO_TOK[tgt],
                    labs[p - 1][64:].copy() if want_labels else None)
            (A if tgt >= 64 else B).append(item)
        if len(A) >= n_new_tgt and len(B) >= n_old_tgt:
            break
    A, B = A[:n_new_tgt], B[:n_old_tgt]
    print(f'  exposure-matched: {len(A):,} new-square targets + '
          f'{len(B):,} other', flush=True)
    return A + B


def batches(data, bs, dev, shuffle=True):
    idx = np.random.permutation(len(data)) if shuffle else np.arange(len(data))
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        ln = max(1, max(len(data[k][0]) for k in j))
        x = np.zeros((len(j), ln), np.int64); last = np.zeros(len(j), np.int64)
        y = np.zeros(len(j), np.int64); nl = np.zeros((len(j), N_NEW), np.int64)
        for a, k in enumerate(j):
            s, t, lb = data[k]
            x[a, :len(s)] = s; last[a] = len(s) - 1; y[a] = t
            if lb is not None:
                nl[a] = lb
        yield (torch.from_numpy(x).to(dev), torch.from_numpy(last).to(dev),
               torch.from_numpy(y).to(dev), torch.from_numpy(nl).to(dev))


def feats(feat, model, x, last):
    h = feat(x)[torch.arange(len(x), device=x.device), last]
    return model(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['board', 'ogpt'], required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--probe', default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6, help='ogpt only: probe layer')
    ap.add_argument('--condition-id', type=int, required=True)
    ap.add_argument('--exp-dir', default='experiments/new_squares')
    ap.add_argument('--new-tgt', type=int, default=60000)
    ap.add_argument('--old-tgt', type=int, default=60000)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    print(f'device {dev}  seed {a.seed}  arm {a.arm}', flush=True)
    feat, W, d, block = load_arm(a.arm, a.ckpt, a.probe, a.layer, dev)
    model = Bottleneck(W, d).to(dev)
    tp = [p for p in feat.parameters() if p.requires_grad]
    print(f'  trainable: readout+new-decoder {sum(p.numel() for p in model.parameters() if p.requires_grad):,}'
          f' + new-token emb {sum(p.numel() for p in tp):,}', flush=True)

    cdir = condition_dir(a.exp_dir, a.condition_id)
    man = json.load(open(os.path.join(cdir, 'test_manifest.json')))
    t0 = time.time()
    data = collect(cdir, a.new_tgt, a.old_tgt, block)
    print(f'  {len(data):,} positions ({time.time()-t0:.0f}s)', flush=True)

    def score_fn(prefix):
        toks = [CELL_TO_TOK[m] for m in prefix if m in CELL_TO_TOK][:block]
        out = np.zeros(N_CELLS, np.float32)
        if not toks:
            return out
        x = torch.tensor([toks], dtype=torch.long, device=dev)
        with torch.no_grad():
            lg, _ = feats(feat, model, x, torch.tensor([len(toks)-1], device=dev))
            p = torch.softmax(lg[0, 1:].float(), -1).cpu().numpy()
        for t, c in TOK_TO_CELL.items():
            out[c] = p[t - 1]
        return out

    @torch.no_grad()
    def newsq_board_acc():
        """READ-ONLY diagnostic: is the new row actually decodable?  Never
        optimised -- an auxiliary board loss would break the match with
        Othello-GPT's next-move-only objective."""
        feat.eval(); model.eval()
        hit = tot = 0
        for x, last, y, nl in batches(data[:4096], 256, dev, shuffle=False):
            _, b3 = feats(feat, model, x, last)
            pred = b3[:, 64:, :].argmax(-1)
            hit += int((pred == nl).sum()); tot += nl.numel()
        feat.train(); model.train()
        return hit / max(tot, 1)

    opt = torch.optim.AdamW(list(model.parameters()) + tp, lr=a.lr)
    sched = [0, 5, 25, 50, 100, 200, 500, 1000, 2000, 5000]
    res = {'arm': a.arm, 'ckpt': a.ckpt, 'condition_id': a.condition_id,
           'seed': a.seed, 'architecture': 'board_bottleneck_72x3',
           'objective': 'next_move_only', 'n_positions': len(data),
           'new_tgt': a.new_tgt, 'old_tgt': a.old_tgt, 'lr': a.lr, 'bs': a.bs,
           'eval_steps': [], 'IL_prob': [], 'IL_acc': [], 'LL_prob': [],
           'IL_prob_per_target': [], 'newsq_board_acc': []}

    def ev(st):
        feat.eval(); model.eval()
        r = score_manifest(score_fn, man, per_bucket=False)
        feat.train(); model.train()
        ba = newsq_board_acc()
        res['eval_steps'].append(st)
        for k in ('IL_prob', 'IL_acc', 'LL_prob', 'IL_prob_per_target'):
            res[k].append(float(r.get(k, float('nan'))))
        res['newsq_board_acc'].append(float(ba))
        print('  step %6d  IL_prob %.4f  per_tgt %.4f  IL_acc %.4f  '
              'LL_prob %.4f  newsq_board %.4f  (%.0fs)'
              % (st, res['IL_prob'][-1], res['IL_prob_per_target'][-1],
                 res['IL_acc'][-1], res['LL_prob'][-1], ba, time.time()-t0),
              flush=True)

    ev(0)
    st = 0; done = False
    feat.train(); model.train()
    while not done:
        for x, last, y, nl in batches(data, a.bs, dev):
            lg, _ = feats(feat, model, x, last)
            loss = F.cross_entropy(lg, y)          # NEXT-MOVE ONLY
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            st += 1
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
