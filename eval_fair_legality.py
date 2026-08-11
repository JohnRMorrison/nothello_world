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
def eval_one(ckpt_path, feats180, cell_legal, positions, kidx, idx, msk, device, batch, rep_override):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    input_dim, hidden = ck['input_dim'], ck['hidden_dim']
    rep = rep_override or {120: 'playedeven', 3600: 'move_grid'}.get(input_dim)
    if rep is None:
        raise ValueError(f'{os.path.basename(ckpt_path)}: cannot infer rep from input_dim={input_dim}; pass --rep')

    def head(state):
        m = DirectMLP(input_dim, hidden).to(device); m.load_state_dict(state); m.eval(); return m
    even = head(ck['even'])
    odd = head(ck['odd']) if ck.get('odd') is not None else even

    po_hit = mx_hit = n = 0
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
        po_hit += int(legal.gather(1, probor.argmax(1, keepdim=True)).sum())
        mx_hit += int(legal.gather(1, maxagg.argmax(1, keepdim=True)).sum())
        n += len(b)
    return rep, hidden, 100 * po_hit / n, 100 * mx_hit / n, n


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
        rep, H, po, mx, n = eval_one(ck, feats180, cell_legal, positions, kidx, idx, msk,
                                     device, args.batch_size, args.rep)
        rows.append((os.path.basename(ck), rep, H, po, mx))
        print(f'  {os.path.basename(ck):45s} rep={rep:10s} H={H:<5d}  prob-OR={po:.2f}%  max={mx:.2f}%', flush=True)

    print(f'\n=== top-1 argmax-legality, ply [{args.ply_min},{args.ply_max}), N={len(kidx):,} ===')
    print(f'{"model":45s} {"rep":10s} {"H":>5} {"prob-OR":>8} {"max":>7}')
    for name, rep, H, po, mx in rows:
        print(f'{name:45s} {rep:10s} {H:>5} {po:>7.2f}% {mx:>6.2f}%')


if __name__ == '__main__':
    main()
