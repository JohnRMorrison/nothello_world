"""New-squares transfer with a CONVOLUTIONAL legal-move readout, both trunks.

The readout reads the frozen decoded board and convolves along the four line
families in both directions -- 8 directions -- looking for the flanking
pattern [empty, theirs x k, mine].  Kernels are SHARED across positions, which
is the whole point:

  coherent_all    122/122 new-square rules are a contiguous run along one
                  direction -> fully conv-detectable
  incoherent_all   42/122, and those 42 are the along-row block that is
                  IDENTICAL in both conditions by design.  The 80
                  random_into_board rules are 0/80 detectable.

So the along-row block is a BUILT-IN CONTROL: a shared kernel set should
transfer it equally in both arms, and the two arms should diverge only on the
into-board block.  Weight sharing is what forbids the random rules -- not
receptive field.  Bigger kernels or more layers do not help, because each new
square has its OWN arbitrary cell set and a shared kernel applies identically
everywhere.  Expressing that needs position-specific parameters, which is what
--mlp-hidden adds as an optional escape hatch:

  --mlp-hidden 0    pure conv.  Prediction: coherent transfers, incoherent only
                    on its 34% shared block.  Architectural, but a precise
                    quantitative claim with the shared block as control.
  --mlp-hidden H    conv + a per-cell nonlinear path.  Incoherent becomes
                    learnable but must be memorised per square, so the question
                    becomes sample efficiency rather than possible/impossible.

Everything else -- trunk, frozen board decoder, exposure matching, next-move
objective, manifest scoring, the read-only newsq_board diagnostic -- is reused
unchanged from newsq_bottleneck.py so the two readouts are comparable.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from new_squares_data import N_CELLS, N_NEW, condition_dir, score_manifest
from newsq_bottleneck import (PLAYABLE, CELL_TO_TOK, TOK_TO_CELL, VOCAB,
                              load_arm, collect, batches)

ROWS, COLS = 9, 8                      # the 9th row is the new squares
OFFB = 3                               # 4th channel: off-board, distinct from empty


def line_index():
    """Index tensors for the 4 line families on a 9x8 grid.

    Returns a list of (name, idx) where idx is (n_lines, max_len) holding cell
    ids, or -1 off-board.  Off-board must be its own channel: a ray running off
    the edge does not flank, and conflating it with 'empty' breaks every edge
    and corner case.
    """
    fams = []
    rows = [[r * COLS + c for c in range(COLS)] for r in range(ROWS)]
    cols = [[r * COLS + c for r in range(ROWS)] for c in range(COLS)]
    diag, anti = {}, {}
    for r in range(ROWS):
        for c in range(COLS):
            diag.setdefault(r - c, []).append(r * COLS + c)
            anti.setdefault(r + c, []).append(r * COLS + c)
    for name, ls in (('rows', rows), ('cols', cols),
                     ('diag', [diag[k] for k in sorted(diag)]),
                     ('anti', [anti[k] for k in sorted(anti)])):
        m = max(len(l) for l in ls)
        a = np.full((len(ls), m), -1, np.int64)
        for i, l in enumerate(ls):
            a[i, :len(l)] = l
        fams.append((name, torch.from_numpy(a)))
    return fams


class ConvReadout(nn.Module):
    """Convolutions over the trunk's 64x3 board output.  NO residual stream.

    The ONLY path from the trunk to the convolution is the frozen,
    board-supervised 64x3 decoder.  There is no trainable decoder row: an
    earlier version had Wnew (8x3x512 params reading h) for the new squares,
    and that was a LEAK -- 24 free numbers optimised by the next-move loss can
    encode "this square is legal right now", which a length-1 kernel then
    reads locally.  The convolution never had to look at the scattered cells,
    which is how a readout that cannot express 80 of 122 incoherent rules still
    scored better on them.  Evidence it was happening: those channels decoded
    occupancy at only 0.50-0.54 where the frozen 64 decode at ~99%.

    The grid is 9x8.  Rows 0-7 carry the decoded board; row 8 is the 8 new
    squares, which have NO board channels -- they get a marker channel only, so
    nothing about them comes from h.  Rays from (8,i) run UP into the real
    board, which is exactly where coherent rules live: [., theirs, theirs, mine]
    up column i is detectable from the frozen board alone, while incoherent
    rules reference scattered cells no ray reaches.

    Channels: 0 empty, 1 theirs, 2 mine, 3 off-board, 4 new-square,
              5 new-square-occupied (only with --newsq-occupancy occupied).
    Off-board must be distinct from empty or every edge case breaks.
    """

    NCH = 6

    def __init__(self, Wfrozen, d, ch=16, ksize=8, mlp_hidden=0,
                 newsq_occ='none', newsq_rows=False):
        super().__init__()
        self.register_buffer('W64', Wfrozen)        # (64,3,d) always FROZEN
        self.newsq_rows = newsq_rows
        if newsq_rows:
            # 8 trainable decoder rows extending the board to 9x8.  ONLY safe
            # when supervised on new-square occupancy (--board-loss > 0):
            # unsupervised, these 24 numbers are a leak -- the next-move loss
            # makes them encode "this square is legal", which a length-1 kernel
            # reads locally, so the convolution never consults the scattered
            # cells.  Supervised, they are an honest board extension and the
            # rule work has nowhere to go but the shared kernels.
            self.Wnew = nn.Parameter(torch.zeros(N_NEW, 3, d))
            self.Wnew.data.normal_(0.0, 0.02)
        self.ch, self.k, self.newsq_occ = ch, ksize, newsq_occ
        self.fams = []
        for i, (name, idx) in enumerate(line_index()):
            self.register_buffer(f'idx{i}', idx)
            self.fams.append((name, i, idx.shape))
        self.conv = nn.ModuleList([nn.Conv1d(self.NCH, ch, ksize)
                                   for _ in range(len(self.fams) * 2)])
        self.comb = nn.Linear(len(self.fams) * 2 * ch, 1)
        self.mlp = (nn.Sequential(nn.Linear(N_CELLS * self.NCH, mlp_hidden),
                                  nn.ReLU(), nn.Linear(mlp_hidden, N_CELLS))
                    if mlp_hidden else None)
        self.register_buffer('valid', torch.tensor(PLAYABLE))
        self.register_buffer('newtok',
                             torch.tensor([CELL_TO_TOK[64 + i] for i in range(N_NEW)]))

    def board(self, h):
        """64x3 from the frozen decoder, optionally extended to 72x3."""
        W = torch.cat([self.W64, self.Wnew], 0) if self.newsq_rows else self.W64
        return torch.einsum('bd,ckd->bck', h, W)

    def grid(self, h, idx):
        b3 = self.board(h)
        B = b3.shape[0]
        g = torch.zeros(B, N_CELLS, self.NCH, device=b3.device, dtype=b3.dtype)
        g[:, :64, :3] = torch.softmax(b3[:, :64], -1)
        g[:, 64:, 4] = 1.0                                    # new-square marker
        if self.newsq_rows:
            g[:, 64:, :3] = torch.softmax(b3[:, 64:], -1)     # decoded new row
        if self.newsq_occ == 'occupied' and idx is not None:
            # occupancy is a deterministic function of the INPUT tokens -- square
            # 64+i is occupied iff its token appears in the prefix.  Not residual
            # access.  Colour is not recoverable this way (forfeits break parity),
            # so the along-row rules are only partly served.
            occ = (idx.unsqueeze(-1) == self.newtok).any(1).to(g.dtype)
            g[:, 64:, 5] = occ
        return g, b3

    def forward(self, h, idx=None):
        g, b3 = self.grid(h, idx)
        B = g.shape[0]
        feats_ = []
        for name, i, shape in self.fams:
            lidx = getattr(self, f'idx{i}')
            L, M = lidx.shape
            gl = g[:, lidx.clamp(min=0).reshape(-1), :].view(B, L, M, self.NCH)
            offmask = (lidx < 0)
            if offmask.any():
                # One unambiguous assignment.  `gl[:, offmask, 3] = 1.0` mixes a
                # 2-D boolean mask with an integer index, and torch disagrees
                # across versions about which dim the 3 consumes -- it worked
                # locally and raised on the pod's 2.8.0.
                ob = torch.zeros(self.NCH, device=gl.device, dtype=gl.dtype)
                ob[3] = 1.0                                   # off-board channel
                gl = gl.clone()
                gl[:, offmask] = ob
            for d in (0, 1):
                seq = gl if d == 0 else torch.flip(gl, dims=[2])
                x = seq.permute(0, 1, 3, 2).reshape(B * L, self.NCH, M)
                x = F.pad(x, (0, self.k - 1))
                y = self.conv[i * 2 + d](x)
                y = y.view(B, L, self.ch, M).permute(0, 1, 3, 2)
                if d == 1:
                    y = torch.flip(y, dims=[2])
                acc = torch.zeros(B, N_CELLS, self.ch, device=g.device, dtype=y.dtype)
                m = (lidx >= 0)
                acc.index_add_(1, lidx[m], y[:, m])
                feats_.append(acc)
        lg = self.comb(torch.cat(feats_, -1)).squeeze(-1)
        if self.mlp is not None:
            lg = lg + self.mlp(g.reshape(B, -1))
        pad = torch.full_like(lg[:, :1], -1e4)
        return torch.cat([pad, lg[:, self.valid]], 1), b3


def feats(feat, model, x, last):
    h = feat(x)[torch.arange(len(x), device=x.device), last]
    return model(h, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['board', 'ogpt'], required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--probe', default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--condition-id', type=int, required=True)
    ap.add_argument('--exp-dir', default='experiments/new_squares')
    ap.add_argument('--ch', type=int, default=16)
    ap.add_argument('--ksize', type=int, default=8)
    ap.add_argument('--newsq-occupancy', choices=['none','occupied'],
                    default='none',
                    help="'none' is the strict spec: the readout sees ONLY the "
                         "64x3 board plus a new-square marker.  'occupied' also "
                         "supplies new-square occupancy from the input tokens "
                         "(deterministic, not residual access).")
    ap.add_argument('--newsq-decoder', choices=['none', 'learned'], default='none',
                    help="'none': row 8 is a marker only, nothing about it comes "
                         "from h (strictest).  'learned': 8 trainable decoder "
                         "rows extend the board to 9x8 -- use WITH --board-loss, "
                         "or they leak.")
    ap.add_argument('--board-loss', type=float, default=0.0,
                    help='weight on an auxiliary cross-entropy supervising the 8 '
                         'new decoder rows on new-square OCCUPANCY (never on '
                         'legality -- the flanking rule is not taught).  Pins '
                         'those rows to being a real board so the rule work must '
                         'happen in the shared kernels.')
    ap.add_argument('--mlp-hidden', type=int, default=0,
                    help='0 = pure conv (incoherent into-board rules are '
                         'inexpressible); >0 adds a per-cell nonlinear escape '
                         'hatch so they become learnable but must be memorised')
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
    print(f'device {dev}  seed {a.seed}  arm {a.arm}  mlp_hidden {a.mlp_hidden}',
          flush=True)
    feat, W, d, block = load_arm(a.arm, a.ckpt, a.probe, a.layer, dev)
    model = ConvReadout(W, d, a.ch, a.ksize, a.mlp_hidden, a.newsq_occupancy,
                        a.newsq_decoder == 'learned').to(dev)
    if a.newsq_decoder == 'learned' and a.board_loss <= 0:
        print('  WARNING: learned new-square rows with NO board loss -- these '
              '24 numbers can encode legality directly and bypass the kernels',
              flush=True)
    tp = [p for p in feat.parameters() if p.requires_grad]
    print('  trainable: readout %d (conv kernels %d) + new-token emb %d'
          % (sum(p.numel() for p in model.parameters() if p.requires_grad),
             sum(c.weight.numel() + c.bias.numel() for c in model.conv),
             sum(p.numel() for p in tp)), flush=True)

    cdir = condition_dir(a.exp_dir, a.condition_id)
    man = json.load(open(os.path.join(cdir, 'test_manifest.json')))
    t0 = time.time()
    data = collect(cdir, a.new_tgt, a.old_tgt, block)
    print(f'  {len(data):,} positions ({time.time()-t0:.0f}s)', flush=True)

    def _score_many(prefixes, bs=256):
        out = {}
        for i in range(0, len(prefixes), bs):
            chunk = prefixes[i:i + bs]
            toks = [[CELL_TO_TOK[m] for m in pre if m in CELL_TO_TOK][:block]
                    for pre in chunk]
            ln = max(1, max(len(t) for t in toks))
            xa = np.zeros((len(toks), ln), np.int64)
            la = np.zeros(len(toks), np.int64)
            for a_, t in enumerate(toks):
                if t:
                    xa[a_, :len(t)] = t; la[a_] = len(t) - 1
            with torch.no_grad():
                lg, _ = feats(feat, model, torch.from_numpy(xa).to(dev),
                              torch.from_numpy(la).to(dev))
                pr = torch.softmax(lg[:, 1:].float(), -1).cpu().numpy()
            for a_, pre in enumerate(chunk):
                v = np.zeros(N_CELLS, np.float32)
                if toks[a_]:
                    for t, c in TOK_TO_CELL.items():
                        v[c] = pr[a_, t - 1]
                out[pre] = v
        return out

    _cache = {}

    def score_fn(prefix):
        k = tuple(prefix)
        return _cache[k] if k in _cache else _score_many([k])[k]

    @torch.no_grad()
    def newsq_board_acc():
        """READ-ONLY: is the new row decodable?  Never optimised -- a board loss
        would break the match with Othello-GPT's next-move-only objective."""
        feat.eval(); model.eval()
        hit = tot = 0
        for x, last, y, nl in batches(data[:4096], 256, dev, shuffle=False):
            _, b3 = feats(feat, model, x, last)
            if b3.shape[1] <= 64:
                # No new-square decoder rows exist in the board-only
                # architecture -- that is the point: nothing about row 8 comes
                # from the residual stream.  This diagnostic measured Wnew,
                # which has been removed, so it no longer applies.
                feat.train(); model.train()
                return float('nan')
            hit += int((b3[:, 64:, :].argmax(-1) == nl).sum()); tot += nl.numel()
        feat.train(); model.train()
        return hit / max(tot, 1)

    opt = torch.optim.AdamW(list(model.parameters()) + tp, lr=a.lr)
    sched = [0, 5, 25, 50, 100, 200, 500, 1000, 2000, 5000]
    res = {'arm': a.arm, 'ckpt': a.ckpt, 'condition_id': a.condition_id,
           'seed': a.seed, 'architecture': 'conv_readout',
           'ch': a.ch, 'ksize': a.ksize, 'mlp_hidden': a.mlp_hidden,
           'newsq_occupancy': a.newsq_occupancy,
           'newsq_decoder': a.newsq_decoder, 'board_loss': a.board_loss,
           'board_channels': '72x3' if a.newsq_decoder=='learned' else '64x3_only',
           'objective': 'next_move_only', 'n_positions': len(data),
           'lr': a.lr, 'bs': a.bs, 'eval_steps': [], 'IL_prob': [],
           'IL_acc': [], 'LL_prob': [], 'IL_prob_per_target': [],
           'newsq_board_acc': [],
           # DISCRIMINATION metrics.  IL_prob_per_target is NOT one:
           # per_tgt = prob / n_legal_targets, so a model putting mass p on
           # EVERY new square scores p regardless of which are legal.  It
           # measures how much mass goes to new squares, and incoherent
           # rules fire more often (60% vs 44% of positions), so it rewards
           # prior matching.  AUC_within ranks new squares WITHIN a
           # position, so it is immune to that base rate.
           'AUC': [], 'AUC_within': [], 'dprime': [],
           'dprime_within': [], 'hit': [], 'FA': []}

    def ev(st):
        feat.eval(); model.eval()
        _cache.clear()
        pres = [tuple(q['game_prefix']) for s_ in ('IL', 'LL')
                for q in man.get(s_, [])]
        _cache.update(_score_many(list(dict.fromkeys(pres))))
        r = score_manifest(score_fn, man, per_bucket=False)
        feat.train(); model.train()
        ba = newsq_board_acc()
        res['eval_steps'].append(st)
        for k in ('IL_prob', 'IL_acc', 'LL_prob', 'IL_prob_per_target',
                  'AUC', 'AUC_within', 'dprime', 'dprime_within',
                  'hit', 'FA'):
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
            lg, b3 = feats(feat, model, x, last)
            loss = F.cross_entropy(lg, y)                     # next-move
            if a.board_loss > 0 and b3.shape[1] > 64:
                # supervises OCCUPANCY of the 8 new squares, not legality
                loss = loss + a.board_loss * F.cross_entropy(
                    b3[:, 64:, :].reshape(-1, 3), nl.reshape(-1))
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
    torch.save({'readout': model.state_dict()},
               a.out.replace('.json', '_readout.pt'))
    json.dump(res, open(a.out, 'w'), indent=2)
    print(f'\nwrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
