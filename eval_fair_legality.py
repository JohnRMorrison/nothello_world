"""Fair legal-move eval for the moveset/movegrid pattern-detector MLPs
(pattern_simple_direct / DirectMLP, 960-pattern head).

Scores TOP-1 argmax-legality on the SAME held-out chunk, ply range, and metric
as reeval_argmax_legality.py (J1B) -- so these numbers sit fairly next to
J1B's prob-OR=98.53% / max=98.04% and OGPT.

The 960 pattern probabilities are aggregated to 60 per-cell legality scores two
ways, then the top-1 cell is checked against the true legal set:
  * prob-OR : 1 - Prod(1 - p_j)     (higher; accumulates patterns)
  * max     : max_j p_j             (simpler; reads the single strongest pattern
                                      -- avoids the "the readout infers legality"
                                      objection)

Input representation is auto-detected from the checkpoint's input_dim
(120 = played+even = MOVESET; 3600 = MOVEGRID) and built from the chunk's
180-d [played(60), when(60), even(60)] features, identical to training.

Usage (on the cluster):
  python eval_fair_legality.py \
    --ckpt experiments/.../pattern_detector_checkpoints/pattern_simple_direct_H4096_move_grid.pt \
    --chunk experiments/.../feature_chunks/chunk_ext_0004.npz \
    --ply-min 5 --ply-max 54
"""
import argparse, os, sys
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pattern_simple import DirectMLP
from train_next_cell_mlp_chunks import to_move_grid_input, PATTERN_TO_CELL60
from eval_next_cell_legality import load_chunk_with_legal_cells


def build_input(feats180, rep):
    """feats180: (B,180) float tensor [played(60), when(60), even(60)]."""
    if rep == 'move_grid':
        return to_move_grid_input(feats180)                                  # (B, 3600)
    if rep == 'playedeven':
        return torch.cat([feats180[:, :60], feats180[:, 120:180]], dim=1)    # (B, 120)
    raise ValueError(f'unknown rep {rep}')


def cell_group_index():
    """(60, max_per_cell) pattern-index + mask, grouping the 960 patterns by
    their target cell (PATTERN_TO_CELL60)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--chunk', default=('experiments/mathematical_transformation_experiments/'
                                        'heuristic_probe_results/feature_chunks/chunk_ext_0004.npz'))
    ap.add_argument('--rep', choices=['playedeven', 'move_grid'], default=None,
                    help='auto-detected from input_dim if omitted (120->playedeven, 3600->move_grid).')
    ap.add_argument('--ply-min', type=int, default=5)
    ap.add_argument('--ply-max', type=int, default=54)   # half-open [5, 54) = moves 5-53
    ap.add_argument('--batch-size', type=int, default=4096)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    input_dim = ck['input_dim']; hidden = ck['hidden_dim']
    rep = args.rep or ({120: 'playedeven', 3600: 'move_grid'}.get(input_dim))
    if rep is None:
        raise ValueError(f'cannot infer rep from input_dim={input_dim}; pass --rep')
    print(f'{os.path.basename(args.ckpt)}: rep={rep}  input_dim={input_dim}  H={hidden}', flush=True)

    def load_head(state):
        m = DirectMLP(input_dim, hidden).to(device); m.load_state_dict(state); m.eval(); return m
    even = load_head(ck['even'])
    odd = load_head(ck['odd']) if 'odd' in ck and ck['odd'] is not None else even

    idx, msk = cell_group_index(); idx = idx.to(device); msk = msk.to(device)

    print(f'loading chunk {args.chunk} ...', flush=True)
    feats180, cell_legal, positions = load_chunk_with_legal_cells(args.chunk)
    positions = positions.astype(np.int64)
    keep = ((positions >= args.ply_min) & (positions < args.ply_max)
            & (cell_legal.sum(1) > 0))
    kidx = np.where(keep)[0]
    print(f'{len(kidx):,} positions in ply [{args.ply_min},{args.ply_max}) with a legal move', flush=True)

    po_hit = mx_hit = n = 0
    with torch.no_grad():
        for i in range(0, len(kidx), args.batch_size):
            b = kidx[i:i + args.batch_size]
            x180 = torch.from_numpy(feats180[b].astype(np.float32)).to(device)
            x = build_input(x180, rep)
            pos = positions[b]
            logits = torch.empty(len(b), 960, device=device)
            em = torch.from_numpy(pos % 2 == 0).to(device)
            if em.any():  logits[em] = even(x[em])
            if (~em).any(): logits[~em] = odd(x[~em])
            p = torch.sigmoid(logits)                       # (B, 960)
            g = p[:, idx]                                   # (B, 60, maxper)
            probor = 1.0 - torch.where(msk, 1.0 - g, torch.ones_like(g)).prod(dim=2)
            maxagg = torch.where(msk, g, torch.zeros_like(g)).max(dim=2).values
            legal = torch.from_numpy(cell_legal[b]).to(device)   # (B, 60) uint8
            po_hit += int(legal.gather(1, probor.argmax(1, keepdim=True)).sum())
            mx_hit += int(legal.gather(1, maxagg.argmax(1, keepdim=True)).sum())
            n += len(b)

    print(f'\ntop-1 argmax-legality  (ply [{args.ply_min},{args.ply_max}), N={n:,}):')
    print(f'  prob-OR : {100*po_hit/n:.2f}%')
    print(f'  max     : {100*mx_hit/n:.2f}%')


if __name__ == '__main__':
    main()
