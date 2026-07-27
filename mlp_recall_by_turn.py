#!/usr/bin/env python
"""Top-1 legal-move accuracy by MOVE for a tree-MLP legal readout (J1B / J3B).

Matches ogpt_recall_by_turn.py's metric: at each position, rank the 64 cells by
the readout's prob-OR legality score, take the top-1 cell, and check whether it
is actually legal (from the true legal mask).  Binned by move (stream position
T = the move being predicted), plus an overall number.  Saves an npz in the
same format as the OGPT script for a shared by-move comparison figure.

  # J1B (per-pattern, mlp hidden layer):
  python mlp_recall_by_turn.py --bank banks/J1_perpattern.pt \
    --readout stream_out/J1_B.pt --no-flanking --canonicalize-mover \
    --label J1B --out-npz notebooks/talk_data/j1b_top1_by_move.npz

  # J3B (ordinal, leaf-index cache):
  python mlp_recall_by_turn.py --bank banks/J3_ordinal.pt \
    --readout stream_out/J3_B_conv.pt --no-flanking --canonicalize-mover \
    --leaf-index-cache-dir /workspace/leaf_cache \
    --label J3B --out-npz notebooks/talk_data/j3b_top1_by_move.npz
"""
import argparse, glob, os
import numpy as np
import torch

import train_streaming_probe as tsp
from opening_tree_mlp import LinearPatternProbOr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True)
    ap.add_argument('--readout', required=True,
                    help='saved legal readout (LinearPatternProbOr state_dict)')
    ap.add_argument('--no-flanking', action='store_true')
    ap.add_argument('--leaf-index-cache-dir', default=None)
    ap.add_argument('--chunk-dir', default='/workspace/feature_chunks')
    ap.add_argument('--cache-dir', default='/workspace/chunk_cache')
    ap.add_argument('--canonicalize-mover', action='store_true')
    ap.add_argument('--ply-min', type=int, default=0)      # want moves 0..59
    ap.add_argument('--ply-max', type=int, default=60)
    ap.add_argument('--max-positions', type=int, default=500_000)
    ap.add_argument('--flanking-patterns',
                    default='hand_crafted_flanking_patterns.pt')
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--eval-chunk', default=None)
    ap.add_argument('--label', default='mlp')
    ap.add_argument('--out-npz', required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # --- tree hidden layer ---
    W_tree, b_tree, tree_meta = tsp.load_trees(args.bank)
    mlp = tsp.OpeningTreeMLP(W_tree, b_tree, tree_meta, device)
    leaf_build = tsp.load_leaf_build(args.bank)
    needs_ordinal = leaf_build is not None
    leaf_index = None
    if args.leaf_index_cache_dir and needs_ordinal:
        cm = np.load(os.path.join(args.leaf_index_cache_dir, 'colmap.npz'))
        leaf_index = (cm['col_tree_idx'], cm['col_nid'])
    patterns = tsp.load_patterns(args.flanking_patterns)

    # --- legal readout (prob-OR over 960 patterns -> 64 cells) ---
    ck = torch.load(args.readout, map_location=device, weights_only=False)
    state = ck['probe_state'] if 'probe_state' in ck else ck['probe_states'][0]
    hidden_dim = state['linear.weight'].shape[1]
    probe = LinearPatternProbOr(hidden_dim, patterns).to(device)
    probe.load_state_dict(state); probe.eval()
    print(f'readout {args.readout}: hidden_dim={hidden_dim}')

    files = sorted(glob.glob(os.path.join(args.chunk_dir, 'chunk_ext_*.npz')))
    eval_path = args.eval_chunk or files[-1]
    if leaf_index is not None:
        X, S, T, L = tsp.load_leaf_index_chunk(
            args.leaf_index_cache_dir, eval_path, args.ply_min, args.ply_max,
            args.max_positions)
    else:
        X, S, T, L = tsp.load_chunk_cached(
            eval_path, args.ply_min, args.ply_max, args.canonicalize_mover,
            args.max_positions, needs_ordinal, cache_dir=args.cache_dir)
    N = X.shape[0]

    NMOVES = 60
    per_move_n = np.zeros(NMOVES, dtype=np.int64)
    per_move_top1 = np.zeros(NMOVES, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            H = tsp.build_hidden_layer_batch(
                X[i:i + args.batch_size], mlp, patterns, None, False, device,
                no_flanking=args.no_flanking, leaf_build=leaf_build,
                leaf_index=leaf_index)
            if H.dtype != torch.float32:
                H = H.float()
            p = probe(H)                                  # (b, 64) prob-OR legality
            top = p.argmax(1).cpu().numpy()               # (b,) top-1 cell
            Lb = L[i:i + args.batch_size]                 # (b, 64) uint8 legal mask
            Tb = T[i:i + args.batch_size]
            has = Lb.sum(1) > 0                           # skip positions w/ no legal move
            for j in range(p.shape[0]):
                t = int(Tb[j])
                if 0 <= t < NMOVES and has[j]:
                    per_move_n[t] += 1
                    if Lb[j, top[j]] == 1:
                        per_move_top1[t] += 1

    top1_by_move = np.where(per_move_n > 0,
                            per_move_top1 / np.maximum(per_move_n, 1), np.nan)
    overall = per_move_top1.sum() / max(per_move_n.sum(), 1)
    print(f'\n=== {args.label}: top-1 legality ===')
    print(f'OVERALL: {100*overall:.4f}%   (N={int(per_move_n.sum())})')
    for m in range(NMOVES):
        if per_move_n[m] > 0:
            print(f'  move {m:2d}: {100*top1_by_move[m]:7.3f}%  (n={per_move_n[m]})')
    np.savez(args.out_npz, model=args.label, turns=np.arange(NMOVES),
             top1_by_move=top1_by_move, n_by_move=per_move_n,
             overall=float(overall), n_total=int(per_move_n.sum()))
    print(f'saved {args.out_npz}')


if __name__ == '__main__':
    main()
