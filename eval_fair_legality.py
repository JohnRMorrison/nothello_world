"""Fair legal-move eval for the moveset/movegrid pattern-detector MLPs
(pattern_simple_direct / DirectMLP, 960-pattern head).

Scores TOP-1 argmax-legality on the SAME held-out chunk, ply range, and metric
as reeval_argmax_legality.py (J1B) -- so these numbers sit fairly next to
J1B's prob-OR=98.53% / max=98.04% and OGPT.

Loads the chunk ONCE and evaluates every --ckpt on it (the chunk decompress +
960-pattern legal-mask recompute is the single-threaded bottleneck, ~15-20 min;
don't pay it per model). Caps to --max-positions (default 500k) after the ply
filter to match the J1B reeval's N. The MLP forward uses all CPU cores via BLAS.

The 960 pattern probabilities are aggregated to 60 per-cell legality scores two
ways, then the top-1 cell is checked against the true legal set:
  * prob-OR : 1 - Prod(1 - p_j)   (higher; accumulates patterns)
  * max     : max_j p_j           (simpler; reads the single strongest pattern)

--ks gives top-K legality: all of the top min(K, n_legal) cells must be legal.
The cap matters -- ~5% of positions have fewer than 5 legal moves, so uncapped
top-5 is unachievable there and understates the model by several points.

Input rep is auto-detected from each checkpoint's input_dim
(120 = played+even = MOVESET; 3600 = MOVEGRID).

Usage (on the pod):
  /usr/bin/python3.13 eval_fair_legality.py \
    --ckpts <CKDIR>/pattern_simple_direct_H{512,4096}_{playedeven,move_grid}.pt \
    --chunk /workspace/feature_chunks/chunk_ext_0004.npz --ply-min 5 --ply-max 54
"""
import argparse, os, sys
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pattern_simple import DirectMLP
from train_next_cell_mlp_chunks import to_move_grid_input, PATTERN_TO_CELL60
from eval_next_cell_legality import load_chunk_with_legal_cells

torch.set_num_threads(os.cpu_count() or 1)


def build_input(feats180, rep):
    if rep == 'move_grid':
        return to_move_grid_input(feats180)                                  # (B, 3600)
    if rep == 'playedeven':
        return torch.cat([feats180[:, :60], feats180[:, 120:180]], dim=1)    # (B, 120)
    raise ValueError(f'unknown rep {rep}')


def cell_group_index():
    by_cell = defaultdict(list)
    for p in range(len(PATTERN_TO_CELL60)):
        by_cell[int(PATTERN_TO_CELL60[p])].append(p)
    maxper = max(len(v) for v in by_cell.values())
    idx = torch.zeros(60, maxper, dtype=torch.long)
    msk = torch.zeros(60, maxper, dtype=torch.bool)
    for c in range(60):
        for k, p in enumerate(by_cell.get(c, [])):
            idx[c, k] = p; msk[c, k] = True
    return idx, msk


@torch.no_grad()
def eval_one(ckpt_path, feats180, cell_legal, positions, kidx, idx, msk, device, batch, rep_override, ks=(1,)):
    try:
        # torch on the cluster (py3.8) predates weights_only; newer torch
        # defaults it to True, which refuses these checkpoints.  Try the
        # explicit form, fall back to the old signature.
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location='cpu')
    input_dim, hidden = ck['input_dim'], ck['hidden_dim']
    rep = rep_override or {120: 'playedeven', 3600: 'move_grid'}.get(input_dim)
    if rep is None:
        raise ValueError(f'{os.path.basename(ckpt_path)}: cannot infer rep from input_dim={input_dim}; pass --rep')

    def head(state):
        m = DirectMLP(input_dim, hidden).to(device); m.load_state_dict(state); m.eval(); return m
    even = head(ck['even'])
    odd = head(ck['odd']) if ck.get('odd') is not None else even

    po_hit = {k: 0 for k in ks}
    mx_hit = {k: 0 for k in ks}
    po_frac = {k: 0.0 for k in ks}
    mx_frac = {k: 0.0 for k in ks}
    n = 0
    for i in range(0, len(kidx), batch):
        b = kidx[i:i + batch]
        x = build_input(torch.from_numpy(feats180[b].astype(np.float32)).to(device), rep)
        pos = positions[b]
        logits = torch.empty(len(b), 960, device=device)
        em = torch.from_numpy(pos % 2 == 0).to(device)
        if em.any():    logits[em] = even(x[em])
        if (~em).any(): logits[~em] = odd(x[~em])
        p = torch.sigmoid(logits)
        g = p[:, idx]
        probor = 1.0 - torch.where(msk, 1.0 - g, torch.ones_like(g)).prod(dim=2)
        maxagg = torch.where(msk, g, torch.zeros_like(g)).max(dim=2).values
        legal = torch.from_numpy(cell_legal[b]).to(device)
        nlegal = legal.sum(1)                       # legal moves at each position
        for agg, acc_all, acc_frac in ((probor, po_hit, po_frac),
                                       (maxagg, mx_hit, mx_frac)):
            order = agg.argsort(dim=1, descending=True)
            for k in ks:
                # cap at n_legal: with fewer than k legal moves neither metric
                # is achievable at full k, and the number would measure the
                # board rather than the model
                ke = torch.clamp(nlegal, max=k)
                picked = legal.gather(1, order[:, :k])          # (B, k) 0/1
                rank = torch.arange(k, device=device).unsqueeze(0)
                within = rank < ke.unsqueeze(1)
                # ALL: every one of the top ke is legal (strict, falls steeply)
                acc_all[k] += int(((picked == 1) | ~within).all(dim=1).sum())
                # FRAC: mean proportion of the top ke that are legal (gentle)
                hits = (picked * within).sum(dim=1).float()
                acc_frac[k] += float((hits / ke.float()).sum())
        n += len(b)
    return rep, hidden, po_hit, mx_hit, po_frac, mx_frac, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--chunk', default=('experiments/mathematical_transformation_experiments/'
                                        'heuristic_probe_results/feature_chunks/chunk_ext_0004.npz'))
    ap.add_argument('--rep', choices=['playedeven', 'move_grid'], default=None)
    ap.add_argument('--ply-min', type=int, default=5)
    ap.add_argument('--ply-max', type=int, default=54)     # half-open [5,54) = moves 5-53
    ap.add_argument('--max-positions', type=int, default=500_000)
    ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 3, 5],
                    help='top-K legality, capped at the number of legal moves')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'threads={torch.get_num_threads()} device={device}', flush=True)

    print(f'loading chunk ONCE: {args.chunk} ...', flush=True)
    feats180, cell_legal, positions = load_chunk_with_legal_cells(args.chunk)
    positions = positions.astype(np.int64)
    keep = ((positions >= args.ply_min) & (positions < args.ply_max)
            & (cell_legal.sum(1) > 0))
    kidx = np.where(keep)[0]
    if len(kidx) > args.max_positions:
        rng = np.random.RandomState(args.seed)
        kidx = np.sort(rng.choice(kidx, args.max_positions, replace=False))
    print(f'{len(kidx):,} eval positions (ply [{args.ply_min},{args.ply_max}), capped {args.max_positions:,})', flush=True)

    idx, msk = cell_group_index(); idx = idx.to(device); msk = msk.to(device)

    rows = []
    for ck in args.ckpts:
        rep, H, po, mx, pof, mxf, n = eval_one(ck, feats180, cell_legal, positions,
                                               kidx, idx, msk, device,
                                               args.batch_size, args.rep, tuple(args.ks))
        rows.append((os.path.basename(ck), rep, H, po, mx, pof, mxf, n))
        fmt = lambda d, sc: '  '.join(f'top{k}={100*d[k]/sc:.2f}%' for k in args.ks)
        print(f'  {os.path.basename(ck):45s} rep={rep:10s} H={H:<5d}', flush=True)
        print(f'      prob-OR  ALL  {fmt(po, n)}', flush=True)
        print(f'      prob-OR  FRAC {fmt(pof, n)}', flush=True)
        print(f'      max      ALL  {fmt(mx, n)}', flush=True)
        print(f'      max      FRAC {fmt(mxf, n)}', flush=True)

    print(f'\n=== top-1 argmax-legality, ply [{args.ply_min},{args.ply_max}), N={len(kidx):,} ===')
    print('ALL  = every one of the top min(K, n_legal) is legal')
    print('FRAC = mean proportion of the top min(K, n_legal) that are legal\n')
    for metric, i_po, i_mx in (('ALL', 3, 4), ('FRAC', 5, 6)):
        print(f'--- {metric} ---')
        hdr = ' '.join(f'{"pOR-top"+str(k):>11}' for k in args.ks) + \
              ' ' + ' '.join(f'{"max-top"+str(k):>11}' for k in args.ks)
        print(f'{"model":45s} {"rep":10s} {"H":>5}' + hdr)
        for row in rows:
            name, rep, H, n = row[0], row[1], row[2], row[7]
            po_d, mx_d = row[i_po], row[i_mx]
            vals = ' '.join(f'{100*po_d[k]/n:>10.2f}%' for k in args.ks) + \
                   ' ' + ' '.join(f'{100*mx_d[k]/n:>10.2f}%' for k in args.ks)
            print(f'{name:45s} {rep:10s} {H:>5}' + vals)
        print()


if __name__ == '__main__':
    main()
