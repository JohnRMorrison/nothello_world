"""The intervention CEILING: edit an explicit 8x8 board bottleneck directly.

Today's layer-6 interventions underperformed because editing h along one
direction produces an internally incoherent, off-manifold state -- the board
head reads "empty" while the rest of h still encodes "occupied" -- and because
the counterfactual board is often unreachable by legal play.  None of that
applies here.  The `attnboard` readout's ONLY input is the frozen 64x3 decoded
board, so setting a square sets the world model exactly.

This is therefore effective BY CONSTRUCTION, and that is the point: it is not a
test of whether the model uses the board (we forced it to) but a CEILING.  We
measured Othello-GPT at +0.09 dP(newly legal) with no idea whether that is good
or bad.  This says what a perfect edit achieves.

Two levels:
  decoded  b3 = frozen_head(h)          -- what THIS model's world model gives
  true     b3 = per-class mean vectors  -- a perfectly decoded board
The gap between them is the cost of 87.57% exact-position decoding.

Substituting a class uses the per-class MEAN logit vector rather than a scaled
one-hot: the readout was trained on the head's logits, so a one-hot would be
off-distribution in scale and would measure the wrong thing.
"""
import argparse, collections, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mingpt.model import GPTConfig
from data.othello import OthelloBoardState
from train_board_state_gpt import GPTBoardState, GPTBoardGrid, CELL_TO_TOK, _autocast
from train_legal_readout import build_head, held_out_games, _pad61

TOK_TO_CELL = {t: c for c, t in CELL_TO_TOK.items()}
CENTER = {27, 28, 35, 36}
VALID60 = [c for c in range(64) if c not in CENTER]
CATEGORIES = {'remove': [(1, 0), (2, 0)], 'add_mine': [(0, 2)],
              'add_yours': [(0, 1)], 'flip': [(1, 2), (2, 1)]}


def head_from_b3(head, b3):
    """Replicate AttnHead.forward, but taking the board logits directly so a
    square can be overwritten."""
    tok = head.inp(b3) + head.row(head.rid) + head.col(head.cid)
    tok = head.tr(tok)
    return _pad61(head.out(tok).squeeze(-1)[..., head.valid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ckpts/board_gpt_L6_20M.ckpt')
    ap.add_argument('--readout', default='ckpts/legal_readout_attnboard.ckpt')
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--n-games', type=int, default=4000)
    ap.add_argument('--ply-min', type=int, default=5)
    ap.add_argument('--ply-max', type=int, default=53)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-csv', default=None)
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(a.ckpt, map_location='cpu')
    cfg = GPTConfig(**ck['cfg'])
    kind = ck.get('args', {}).get('head', 'flat')
    trunk = (GPTBoardGrid(cfg, rank=ck['args'].get('head_rank', 64))
             if kind == 'grid' else GPTBoardState(cfg))
    trunk.load_state_dict(ck['model']); trunk.eval().to(dev)
    rck = torch.load(a.readout, map_location='cpu')
    head = build_head('attnboard', cfg.n_embd)
    head.load_state_dict(rck['head']); head.eval().to(dev)
    print(f'{a.ckpt} ({kind}) + {a.readout}  device {dev}', flush=True)

    games = held_out_games(a.data_dir, a.n_games)
    rng = np.random.default_rng(a.seed)
    print(f'{len(games)} held-out games, ply {a.ply_min}-{a.ply_max}', flush=True)

    # ---- per-class mean logit vectors, for faithful class substitution ----
    sums = np.zeros((3, 3)); cnts = np.zeros(3)
    with torch.no_grad(), _autocast(dev, False):
        for g in games[:200]:
            p = int(rng.integers(a.ply_min, a.ply_max + 1))
            toks = torch.tensor([[CELL_TO_TOK[c] for c in g[:p]]],
                                dtype=torch.long, device=dev)
            x = trunk.drop(trunk.tok_emb(toks) + trunk.pos_emb[:, :p, :])
            h = trunk.ln_f(trunk.blocks(x))[:, -1, :]
            b3 = trunk.head(h).view(64, 3).float().cpu().numpy()
            bd = OthelloBoardState(); bd.update(list(g[:p]))
            st = bd.state.flatten(); mv = bd.next_hand_color
            cls = np.zeros(64, int); cls[st == -mv] = 1; cls[st == mv] = 2
            for k in range(3):
                m = cls == k
                sums[k] += b3[m].sum(0); cnts[k] += m.sum()
    MEAN = torch.tensor(sums / np.maximum(cnts[:, None], 1), dtype=torch.float32,
                        device=dev)
    print('per-class mean board logits:\n%s' % MEAN.cpu().numpy().round(3), flush=True)
    # If the class means are not well separated, substituting one for another is
    # a no-op and every dP comes out 0.0000 -- which is what an untrained head
    # does.  Make that visible instead of silent.
    M = MEAN.cpu().numpy()
    sep = min(float(np.linalg.norm(M[i] - M[j])) for i in range(3)
              for j in range(i + 1, 3))
    print('  min pairwise separation between class means: %.4f' % sep, flush=True)
    if sep < 0.5:
        print('  WARNING: class means nearly identical -- the board head carries '
              'little class information, so every dP will be ~0 for that reason '
              'and NOT because interventions are ineffective.', flush=True)

    acc = collections.defaultdict(lambda: dict(n=0, dl=0.0, di=0.0, nb=0))
    rows = []
    t0 = time.time(); B = 128
    for b0 in range(0, len(games), B):
        bt = games[b0:b0 + B]
        T = a.ply_max
        plies = rng.integers(a.ply_min, a.ply_max + 1, size=len(bt))
        toks = torch.tensor([[CELL_TO_TOK[c] for c in g[:T]] for g in bt],
                            dtype=torch.long, device=dev)
        with torch.no_grad():
            x = trunk.drop(trunk.tok_emb(toks) + trunk.pos_emb[:, :T, :])
            Hall = trunk.ln_f(trunk.blocks(x)).float()
            H = Hall[torch.arange(len(bt), device=dev),
                     torch.tensor(plies - 1, device=dev)]
            B3_dec = trunk.head(H).view(len(bt), 64, 3).float()

        boards, movers, truecls = [], [], []
        for g, p in zip(bt, plies):
            bd = OthelloBoardState(); bd.update(list(g[:int(p)]))
            st = bd.state.flatten(); mv = bd.next_hand_color
            cls = np.zeros(64, int); cls[st == -mv] = 1; cls[st == mv] = 2
            boards.append(bd); movers.append(mv); truecls.append(cls)
        B3_true = MEAN[torch.tensor(np.stack(truecls), device=dev)]   # (B,64,3)

        for src, B3 in (('decoded', B3_dec), ('true', B3_true)):
            for cat, pairs in CATEGORIES.items():
                sel, sqs, tgs, nl, ni = [], [], [], [], []
                for i, bd in enumerate(boards):
                    cls = truecls[i]; mv = movers[i]
                    cand = [c for c in VALID60 if any(cls[c] == cu for cu, _ in pairs)]
                    if not cand:
                        continue
                    c = int(rng.choice(cand)); cur = int(cls[c])
                    tg = [t for cu, t in pairs if cu == cur][0]
                    before = set(bd.get_valid_moves())
                    stf = bd.state.reshape(8, 8); saved = stf[c // 8, c % 8]
                    stf[c // 8, c % 8] = {0: 0, 1: -mv, 2: mv}[tg]
                    after = set(bd.get_valid_moves())
                    stf[c // 8, c % 8] = saved
                    sel.append(i); sqs.append(c); tgs.append(tg)
                    nl.append(after - before); ni.append(before - after)
                if not sel:
                    continue
                ii = torch.tensor(sel, device=dev)
                b0_ = B3[ii].clone()
                b1_ = b0_.clone()
                b1_[torch.arange(len(sel)), torch.tensor(sqs, device=dev)] = \
                    MEAN[torch.tensor(tgs, device=dev)]
                with torch.no_grad():
                    p0 = torch.softmax(head_from_b3(head, b0_)[..., 1:].float(), -1)
                    p1 = torch.softmax(head_from_b3(head, b1_)[..., 1:].float(), -1)
                for j in range(len(sel)):
                    sub = ('both' if nl[j] and ni[j] else 'newly_legal' if nl[j]
                           else 'newly_illegal' if ni[j] else None)
                    if sub is None:
                        continue
                    dl = di = 0.0
                    if nl[j]:
                        k = [CELL_TO_TOK[c] - 1 for c in nl[j]]
                        dl = float(p1[j, k].sum() - p0[j, k].sum())
                    if ni[j]:
                        k = [CELL_TO_TOK[c] - 1 for c in ni[j]]
                        di = float(p1[j, k].sum() - p0[j, k].sum())
                    r = acc[(src, cat, sub)]
                    r['n'] += 1; r['dl'] += dl; r['di'] += di
                    if a.out_csv:
                        rows.append((src, cat, sub, sqs[j] // 8, sqs[j] % 8,
                                     int(plies[sel[j]]), dl, di))
        if (b0 // B) % 5 == 0:
            print(f'  {b0+len(bt)}/{len(games)} games ({time.time()-t0:.0f}s)',
                  flush=True)

    print('\n=== INTERVENTION CEILING: exact edit on the board bottleneck ===')
    print('  board source  category     dP(newly legal)  dP(newly illegal)     n')
    for src in ('decoded', 'true'):
        for cat in list(CATEGORIES) + ['ALL']:
            cats = list(CATEGORIES) if cat == 'ALL' else [cat]
            wl = [(acc[(src, c, s)]['dl'], acc[(src, c, s)]['n'])
                  for c in cats for s in ('both', 'newly_legal')]
            wi = [(acc[(src, c, s)]['di'], acc[(src, c, s)]['n'])
                  for c in cats for s in ('both', 'newly_illegal')]
            f = lambda w: sum(v for v, _ in w) / max(sum(n for _, n in w), 1)
            print('  %-13s %-11s %+14.4f   %+14.4f   %7d'
                  % (src, cat, f(wl), f(wi), sum(n for _, n in wl)))
    print('\n  Othello-GPT L4, the best probe-based result: +0.0620 / -0.0826 at 2x,')
    print('  saturating near +0.09 / -0.10.  Our layer-6 h-edit with a matched')
    print('  linear readout: +0.1005 / -0.0606 aggregate, but only `remove` beat')
    print('  OGPT; the other three categories lost.')
    if a.out_csv:
        import csv
        with open(a.out_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['board_source', 'category', 'sub_condition', 'row', 'col',
                        'ply', 'dP_newly_legal', 'dP_newly_illegal'])
            for r in rows:
                w.writerow([*r[:6], f'{r[6]:.6f}', f'{r[7]:.6f}'])
        print(f'\nwrote {a.out_csv} ({len(rows)} rows, 6dp)')


if __name__ == '__main__':
    main()
