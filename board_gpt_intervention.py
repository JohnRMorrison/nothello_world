"""Calibrated causal interventions on the board-state GPT, matched to the
Othello-GPT alpha sweep in notebooks/talk_data/alpha_sweep_prob_shift_L*.txt.

Othello-GPT needs a FITTED probe to locate the board, so every intervention
result carries the caveat "maybe the direction is wrong".  The board-state GPT's
own head IS that probe, exactly, by construction.  A larger effect here should
therefore be attributed to "exact directions edit more cleanly", not to "this
model has a better world model".

PROTOCOL, copied from ogpt_legal_mass_shift.py so the numbers are comparable:

  direction    d = W[s, target_class] - W[s, current_class],  normalised
  edit         h' = h - s * (h . d_hat) * d_hat
  calibration  20-step binary search for the minimal s in [0, 10] whose edit
               flips the decoded class at the target square to target_class,
               then * 1.1 (safety margin), capped at 10.  s=0.5 if already at
               target, 10.0 if unreachable.
  sweep        s_used = alpha_mult * calibrated_s, for alpha in 1..10x

Interventions run at layer 6 -- the ln_f output, where the board is linearly
decodable and where BOTH the board head and the legal readout read.  That is
the cd=0 condition (intervene at the layer the readout reads, no further
computation in between).

THE BAR.  Othello-GPT shows almost no newly-legal promotion at its own L6
(+0.0000 at 2x) because that edit must still survive two blocks before its move
head.  Comparing against L6 would flatter us.  The honest bar is Othello-GPT's
BEST layer, L4:

  alpha 2x   remove: dP(newly legal) +0.0653   dP(newly illegal) -0.0691
             add_mine   +0.0744 / -0.0959      add_yours +0.0633 / -0.0945
             flip       +0.0441 / -0.0706
  saturated (4x+)      ~+0.09 / ~-0.10

Use the LINEAR readout for the headline: Othello-GPT's move head is a single
linear map, and our `attn` readout is a 2-layer transformer with 4.6M params
that would plausibly amplify an edit.

Classes follow train_board_state_gpt._one: 0 empty, 1 theirs, 2 mine, with the
mover from OthelloBoardState.next_hand_color.  Counterfactual legal sets are
computed by actually writing the new value onto a real board, not predicted.
"""
import argparse, os, sys, time, collections
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mingpt.model import GPTConfig
from data.othello import OthelloBoardState
from train_board_state_gpt import (GPTBoardState, GPTBoardGrid, CELL_TO_TOK,
                                   _autocast)
from train_legal_readout import build_head, held_out_games

CENTER = {27, 28, 35, 36}
VALID60 = [c for c in range(64) if c not in CENTER]
# category -> list of (current_class, target_class)
CATEGORIES = {
    'remove':    [(1, 0), (2, 0)],
    'add_mine':  [(0, 2)],
    'add_yours': [(0, 1)],
    'flip':      [(1, 2), (2, 1)],
}
ALPHAS = [1, 2, 3, 4, 6, 8, 10]


def class_weights(trunk, kind):
    """(64, 3, d): the head's weight on each square/class, so the two-class
    direction is just a difference of two rows."""
    with torch.no_grad():
        if kind == 'grid':
            q = (trunk.row[:, None, :] * trunk.col[None, :, :]).reshape(64, -1)
            # logits[s,k] = (Wk^T h) . q[s] / sqrt(m)  ->  dlogit/dh = Wk q[s]
            return torch.einsum('kij,sj->ski', trunk.Wk, q) / trunk.rank ** 0.5
        return trunk.head.weight.view(64, 3, -1).clone()


def decode_at(W, h, sq):
    """Decoded class at one square per row.  h (N,d), sq (N,) -> (N,)."""
    return torch.einsum('nd,nkd->nk', h, W[sq]).argmax(-1)


def calibrate(W, h, sq, dhat, coeff, target, iters=20):
    """Vectorised version of ogpt_legal_mass_shift.find_min_scale: least s whose
    edit flips the decoded class to `target`, x1.1, capped at 10."""
    def flips(s):
        hp = h - s.unsqueeze(1) * coeff.unsqueeze(1) * dhat
        return decode_at(W, hp, sq) == target
    n = len(h)
    zero = torch.zeros(n, device=h.device)
    already = flips(zero)
    reach = flips(torch.full_like(zero, 10.0))
    lo = torch.zeros(n, device=h.device); hi = torch.full_like(lo, 10.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = flips(mid)
        hi = torch.where(ok, mid, hi); lo = torch.where(ok, lo, mid)
    s = torch.clamp(hi * 1.1, max=10.0)
    s = torch.where(already, torch.full_like(s, 0.5), s)
    return torch.where(reach | already, s, torch.full_like(s, 10.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ckpts/board_gpt_L6_20M.ckpt')
    ap.add_argument('--readout', default='ckpts/legal_readout_L6_20M.ckpt')
    ap.add_argument('--readout-kind', default='linear')
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--n-games', type=int, default=2000)
    ap.add_argument('--ply-min', type=int, default=5)
    ap.add_argument('--ply-max', type=int, default=53)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fp32', action='store_true')
    ap.add_argument('--out-csv', default=None)
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = not a.fp32
    if dev == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(a.ckpt, map_location='cpu')
    cfg = GPTConfig(**ck['cfg'])
    kind = ck.get('args', {}).get('head', 'flat')
    trunk = (GPTBoardGrid(cfg, rank=ck['args'].get('head_rank', 64))
             if kind == 'grid' else GPTBoardState(cfg))
    trunk.load_state_dict(ck['model']); trunk.eval().to(dev)
    W = class_weights(trunk, kind).to(dev).float()
    rck = torch.load(a.readout, map_location='cpu')
    readout = build_head(a.readout_kind, cfg.n_embd)
    readout.load_state_dict(rck['head']); readout.eval().to(dev)
    print(f'{a.ckpt}: {cfg.n_layer}L d{cfg.n_embd} head={kind}', flush=True)
    print(f'readout: {a.readout} ({a.readout_kind})  device {dev}', flush=True)

    games = held_out_games(a.data_dir, a.n_games)
    rng = np.random.default_rng(a.seed)
    print(f'{len(games)} held-out games, ply sampled uniformly in '
          f'[{a.ply_min},{a.ply_max}]', flush=True)

    # one position per game; one target square per category within that position
    acc = collections.defaultdict(lambda: dict(n=0, dl=0.0, di=0.0))
    rowsout = []
    t0 = time.time(); B = 128
    for b0 in range(0, len(games), B):
        bt = games[b0:b0 + B]
        T = a.ply_max
        plies = rng.integers(a.ply_min, a.ply_max + 1, size=len(bt))
        toks = torch.tensor([[CELL_TO_TOK[c] for c in g[:T]] for g in bt],
                            dtype=torch.long, device=dev)
        with torch.no_grad(), _autocast(dev, amp):
            x = trunk.drop(trunk.tok_emb(toks) + trunk.pos_emb[:, :T, :])
            Hall = trunk.ln_f(trunk.blocks(x)).float()            # (B, T, d)
        pidx = torch.tensor(plies - 1, device=dev)
        H = Hall[torch.arange(len(bt), device=dev), pidx]          # layer 6, cd=0

        boards, movers = [], []
        for g, p in zip(bt, plies):
            bd = OthelloBoardState(); bd.update(list(g[:int(p)]))
            boards.append(bd); movers.append(bd.next_hand_color)

        for cat, pairs in CATEGORIES.items():
            sel, sqs, curs, tgts, nl, ni = [], [], [], [], [], []
            for i, bd in enumerate(boards):
                flat = bd.state.flatten(); mv = movers[i]
                cls = np.zeros(64, np.int8)
                cls[flat == -mv] = 1; cls[flat == mv] = 2
                cand = [c for c in VALID60
                        if any(cls[c] == cu for cu, _ in pairs)]
                if not cand:
                    continue
                c = int(rng.choice(cand))
                cur = int(cls[c])
                tg = [t for cu, t in pairs if cu == cur][0]
                before = set(bd.get_valid_moves())
                st = bd.state.reshape(8, 8); saved = st[c // 8, c % 8]
                st[c // 8, c % 8] = {0: 0, 1: -mv, 2: mv}[tg]
                after = set(bd.get_valid_moves())
                st[c // 8, c % 8] = saved
                sel.append(i); sqs.append(c); curs.append(cur); tgts.append(tg)
                nl.append(after - before); ni.append(before - after)
            if not sel:
                continue
            ii = torch.tensor(sel, device=dev)
            sq = torch.tensor(sqs, device=dev)
            cu = torch.tensor(curs, device=dev); tg = torch.tensor(tgts, device=dev)
            h = H[ii]
            d = W[sq, tg, :] - W[sq, cu, :]
            dhat = d / d.norm(dim=1, keepdim=True).clamp_min(1e-9)
            coeff = (h * dhat).sum(1)
            s0 = calibrate(W, h, sq, dhat, coeff, tg)
            with torch.no_grad(), _autocast(dev, amp):
                p0 = torch.softmax(readout(h)[..., 1:].float(), -1)
            for am in ALPHAS:
                hp = h - (am * s0).unsqueeze(1) * coeff.unsqueeze(1) * dhat
                with torch.no_grad(), _autocast(dev, amp):
                    p1 = torch.softmax(readout(hp)[..., 1:].float(), -1)
                for j in range(len(sel)):
                    sub = ('both' if nl[j] and ni[j] else
                           'newly_legal' if nl[j] else
                           'newly_illegal' if ni[j] else None)
                    if sub is None:
                        continue
                    dl = di = 0.0
                    if nl[j]:
                        k = [CELL_TO_TOK[c] - 1 for c in nl[j]]
                        dl = float(p1[j, k].sum() - p0[j, k].sum())
                    if ni[j]:
                        k = [CELL_TO_TOK[c] - 1 for c in ni[j]]
                        di = float(p1[j, k].sum() - p0[j, k].sum())
                    key = (sqs[j] // 8, sqs[j] % 8, cat, sub, am)
                    r = acc[key]; r['n'] += 1; r['dl'] += dl; r['di'] += di
                    if a.out_csv:
                        rowsout.append((sqs[j]//8, sqs[j]%8, cat, sub, am,
                                        int(plies[sel[j]]), float(s0[j]), dl, di))
        if (b0 // B) % 4 == 0:
            print(f'  {b0+len(bt)}/{len(games)} games ({time.time()-t0:.0f}s)',
                  flush=True)

    # aggregate exactly as the Othello-GPT table is aggregated
    print(f'\n=== layer-6 (cd=0) calibrated interventions, head={kind}, '
          f'readout={a.readout_kind}, ply {a.ply_min}-{a.ply_max} ===')
    print('  alpha   dP(newly legal)   dP(newly illegal)      n_legal   n_illegal')
    for am in ALPHAS:
        wl = [(r['dl'], r['n']) for k, r in acc.items()
              if k[4] == am and k[3] in ('both', 'newly_legal')]
        wi = [(r['di'], r['n']) for k, r in acc.items()
              if k[4] == am and k[3] in ('both', 'newly_illegal')]
        f = lambda w: sum(v for v, _ in w) / max(sum(n for _, n in w), 1)
        print(f'  {am:2d}x     {f(wl):+12.4f}      {f(wi):+12.4f}     '
              f'{sum(n for _,n in wl):8d}    {sum(n for _,n in wi):8d}')
    print('\n  by category at 2x:')
    for cat in CATEGORIES:
        wl = [(r['dl'], r['n']) for k, r in acc.items()
              if k[4] == 2 and k[2] == cat and k[3] in ('both', 'newly_legal')]
        wi = [(r['di'], r['n']) for k, r in acc.items()
              if k[4] == 2 and k[2] == cat and k[3] in ('both', 'newly_illegal')]
        f = lambda w: sum(v for v, _ in w) / max(sum(n for _, n in w), 1)
        print(f'    {cat:10s} {f(wl):+8.4f} / {f(wi):+8.4f}')
    print('\n  OGPT L4 (the bar, 2x): remove +0.0653/-0.0691  add_mine +0.0744/-0.0959')
    print('                          add_yours +0.0633/-0.0945  flip +0.0441/-0.0706')
    print('  OGPT L4 saturated (4x+): ~+0.09 / ~-0.10')
    print('  OGPT L6 (NOT the bar -- 2 blocks from its head): +0.0000 at 2x')
    if a.out_csv:
        import csv
        with open(a.out_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['row','col','category','sub_condition','alpha_mult',
                        'ply','calibrated_s','dP_newly_legal','dP_newly_illegal'])
            for r in rowsout:
                w.writerow([r[0], r[1], r[2], r[3], r[4], r[5],
                            f'{r[6]:.6f}', f'{r[7]:.6f}', f'{r[8]:.6f}'])
        print(f'\nwrote {a.out_csv} ({len(rowsout)} rows, 6dp)')


if __name__ == '__main__':
    main()
