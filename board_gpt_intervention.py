"""Causal interventions on the board-state GPT at layer 6 (the final layer).

Othello-GPT needs a PROBE to locate the board, so every intervention result
carries the caveat "maybe the probe direction is wrong."  Here the model's own
head is the probe, exactly, by construction -- so this is a positive control for
the intervention methodology itself.  If editing along a direction the model
demonstrably uses still produces collateral damage, that damage is intrinsic,
not a probe artefact.

Direction, matching ogpt_intervention.py:199 but read off the head instead of a
fitted probe:

  flat head  d = W[s,0] - 0.5*(W[s,1] + W[s,2]),  W = head.weight.view(64,3,512)
  grid head  d = (W0 - 0.5*W1 - 0.5*W2) @ q[r,c], since dlogit/dh = Wk q[r,c]

Edit, matching the Othello-GPT protocol:  h' = h - s * (h . d_hat) * d_hat
so the component along d_hat is scaled by (1 - s).  Note this is NOT the exact
two-class margin edit: d is empty-minus-MEAN-of-occupied, matching
ogpt_intervention.py, so s=1 removes the projection but does not land exactly on
a pairwise decision boundary.

Measured per intervention:
  * efficacy   -- did the TARGET square's decoded class become empty?
  * collateral -- how many OTHER squares changed their decoded class?
  * behaviour  -- with a legal readout attached, how the move distribution moves
                  on squares that the counterfactual board makes newly legal or
                  newly illegal.  The counterfactual legal set is computed by
                  actually emptying the square on a real board, not predicted.

One intervention per game, for independence.
"""
import argparse, glob, os, pickle, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mingpt.model import GPTConfig
from data.othello import OthelloBoardState
from train_board_state_gpt import (GPTBoardState, GPTBoardGrid, BLOCK,
                                   CELL_TO_TOK, _autocast)
from train_legal_readout import build_head, held_out_games

TOK_TO_CELL = {t: c for c, t in CELL_TO_TOK.items()}
CENTER = {27, 28, 35, 36}
VALID60 = [c for c in range(64) if c not in CENTER]


def head_directions(trunk, kind, dev):
    """(64, d) unit directions: 'empty' minus the mean of the two occupied
    classes, for every square.  Read straight off the frozen head."""
    with torch.no_grad():
        if kind == 'grid':
            q = (trunk.row[:, None, :] * trunk.col[None, :, :]).reshape(64, -1)
            # logits[s,k] = (Wk^T h) . q[s] / sqrt(m)  =>  dlogit/dh = Wk q[s]
            Wq = torch.einsum('kij,sj->ski', trunk.Wk, q)      # (64, 3, d)
        else:
            Wq = trunk.head.weight.view(64, 3, -1)             # (64, 3, d)
        d = Wq[:, 0, :] - 0.5 * (Wq[:, 1, :] + Wq[:, 2, :])
        return (d / d.norm(dim=1, keepdim=True).clamp_min(1e-9)).to(dev)


def decode(trunk, kind, h):
    """h (N, d) -> decoded class per square (N, 64)."""
    if kind == 'grid':
        q = (trunk.row[:, None, :] * trunk.col[None, :, :]).reshape(64, -1)
        z = torch.einsum('ni,kij->nkj', h, trunk.Wk)
        lg = torch.einsum('nkj,sj->nsk', z, q) / trunk.rank ** 0.5 + trunk.bk
    else:
        lg = trunk.head(h).view(len(h), 64, 3)
    return lg.argmax(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ckpts/board_gpt_L6_20M.ckpt')
    ap.add_argument('--readout', default=None,
                    help='legal-readout ckpt (e.g. ckpts/legal_readout_attn.ckpt); '
                         'without it only board efficacy/collateral are reported')
    ap.add_argument('--readout-kind', default='attn')
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--n-games', type=int, default=2000)
    ap.add_argument('--ply', type=int, default=25, help='one position per game')
    ap.add_argument('--strengths', type=float, nargs='+',
                    default=[0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0])
    ap.add_argument('--fp32', action='store_true')
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
    print(f'{a.ckpt}: {cfg.n_layer}L d{cfg.n_embd} head={kind}  device {dev}', flush=True)

    readout = None
    if a.readout:
        rck = torch.load(a.readout, map_location='cpu')
        readout = build_head(a.readout_kind, cfg.n_embd)
        readout.load_state_dict(rck['head']); readout.eval().to(dev)
        print(f'legal readout: {a.readout} ({a.readout_kind})', flush=True)

    D = head_directions(trunk, kind, dev)
    games = held_out_games(a.data_dir, a.n_games)
    print(f'{len(games)} held-out games, one intervention each at ply {a.ply}',
          flush=True)

    rows = {s: dict(n=0, flip=0, collat=0.0, dlegal=0.0, dillegal=0.0, nb=0)
            for s in a.strengths}
    t0 = time.time()
    B = 64
    for b0 in range(0, len(games), B):
        bt = games[b0:b0 + B]
        toks = torch.tensor([[CELL_TO_TOK[c] for c in g[:a.ply]] for g in bt],
                            dtype=torch.long, device=dev)
        with torch.no_grad(), _autocast(dev, amp):
            x = trunk.drop(trunk.tok_emb(toks) + trunk.pos_emb[:, :a.ply, :])
            H = trunk.ln_f(trunk.blocks(x))[:, -1, :].float()   # layer 6, last pos
        base_cls = decode(trunk, kind, H)                       # (B, 64)

        # pick ONE occupied, non-centre target square per game
        tgt, keep = [], []
        boards = []
        for i, g in enumerate(bt):
            bd = OthelloBoardState(); bd.update(list(g[:a.ply]))
            flat = bd.state.flatten()
            occ = [c for c in VALID60 if flat[c] != 0]
            if not occ:
                continue
            tgt.append(occ[len(occ) // 2]); keep.append(i); boards.append(bd)
        if not keep:
            continue
        ki = torch.tensor(keep, device=dev)
        ts = torch.tensor(tgt, device=dev)
        Hk = H[ki]; bk = base_cls[ki]
        dh = D[ts]                                              # (K, d)
        proj = (Hk * dh).sum(1, keepdim=True)

        # counterfactual legal sets: actually empty the square on a real board
        newly_legal, newly_illegal = [], []
        for j, bd in enumerate(boards):
            before = set(bd.get_valid_moves())
            saved = bd.state.flatten()[tgt[j]]
            st = bd.state.reshape(8, 8)
            st[tgt[j] // 8, tgt[j] % 8] = 0
            after = set(bd.get_valid_moves())
            st[tgt[j] // 8, tgt[j] % 8] = saved
            newly_legal.append(after - before); newly_illegal.append(before - after)

        for s in a.strengths:
            Hp = Hk - s * proj * dh
            cls = decode(trunk, kind, Hp)
            r = rows[s]
            r['n'] += len(keep)
            r['flip'] += int((cls[torch.arange(len(keep)), ts] == 0).sum())
            diff = (cls != bk); diff[torch.arange(len(keep)), ts] = False
            r['collat'] += float(diff.sum(1).float().sum())
            if readout is not None:
                with torch.no_grad(), _autocast(dev, amp):
                    p0 = torch.softmax(readout(Hk)[..., 1:].float(), -1)
                    p1 = torch.softmax(readout(Hp)[..., 1:].float(), -1)
                for j in range(len(keep)):
                    if newly_legal[j]:
                        idx = [CELL_TO_TOK[c] - 1 for c in newly_legal[j]]
                        r['dlegal'] += float(p1[j, idx].sum() - p0[j, idx].sum())
                        r['nb'] += 1
                    if newly_illegal[j]:
                        idx = [CELL_TO_TOK[c] - 1 for c in newly_illegal[j]]
                        r['dillegal'] += float(p1[j, idx].sum() - p0[j, idx].sum())
        if (b0 // B) % 10 == 0:
            print(f'  {b0 + len(bt)}/{len(games)} games ({time.time()-t0:.0f}s)',
                  flush=True)

    print(f'\n=== layer-6 interventions, head={kind}, ply {a.ply}, '
          f'N={rows[a.strengths[0]]["n"]:,} ===')
    print('  s     target flipped   collateral squares   dP(newly legal)  dP(newly illegal)')
    for s in a.strengths:
        r = rows[s]; n = max(r['n'], 1); nb = max(r['nb'], 1)
        print(f'  {s:4.2f}   {100*r["flip"]/n:8.2f}%   {r["collat"]/n:12.3f}       '
              f'{100*r["dlegal"]/nb:+8.3f}%      {100*r["dillegal"]/nb:+8.3f}%')
    print('\n  s=1 removes the whole component along d_hat.  NOTE: unlike the')
    print('  two-class probe margin, this is NOT an exact landing on the')
    print('  decision boundary -- d is empty-minus-MEAN-of-occupied (matching')
    print('  ogpt_intervention.py), so the three class logits still depend on')
    print('  directions orthogonal to d.  Read s as projection removed, not margin.')
    print('  collateral = OTHER squares whose decoded class changed.')


if __name__ == '__main__':
    main()
