"""Local confirmation that the 50-leaf multioutput (joint) pattern tree cannot
resolve the 960 flanking patterns.

We reproduce exactly what the joint tree does: a leaf's prediction for pattern p
is the MEAN of pattern p's label over the training positions that land in that
leaf.  With only 50 leaves partitioning board-occupancy space (not pattern-
firing space), every leaf's mean over 960 rare, near-mutually-exclusive targets
is ~0, so at threshold 0.5 the tree predicts nothing on -> ~0% recall.

Method:
  1. self-play fresh positions (same encoding as the fit: canonicalize_mover,
     ply 0-60, playedeven features, no recent-K).
  2. route each position through the 50 leaf decision paths stored in the bank.
  3. leaf_pred[leaf, pattern] = mean pattern label over positions in that leaf.
  4. broadcast to positions and measure recall / precision at threshold 0.5,
     plus the best predicted probability any leaf assigns any pattern.

Usage:
    python confirm_multioutput_recall.py <bank.pt> [--num-games 3000] [--seed 7]
"""
import argparse
import numpy as np
import torch

from midgame_tree_mlp import sample_midgame_positions
from flanking_patterns import load_patterns, true_pattern_activations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bank')
    ap.add_argument('--patterns', default='hand_crafted_flanking_patterns.pt')
    ap.add_argument('--num-games', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=7,   # != fit seed (42): fresh data
                    help='self-play seed; independent of the fit set.')
    args = ap.parse_args()

    d = torch.load(args.bank, map_location='cpu', weights_only=False)
    a = d.get('args', {})
    leaves = [m for m in d['path_info'] if m.get('kind') == 'pattern_multi']
    print(f'bank: {args.bank}')
    print(f'  {len(leaves)} multioutput leaves  '
          f'(canonicalize_mover={a.get("canonicalize_mover")}, depth cap '
          f'{a.get("tree_max_depth")}, min_leaf {a.get("tree_min_samples_leaf")})')

    patterns = load_patterns(args.patterns)
    K = len(patterns)
    print(f'  {K} flanking patterns')

    # 1. fresh positions (identical encoding to the fit) --------------------
    print(f'\nself-playing {args.num_games} games (seed {args.seed})...')
    X, S, T = sample_midgame_positions(
        args.num_games, ply_min=0, ply_max=60, seed=args.seed,
        canonicalize_mover=bool(a.get('canonicalize_mover', True)))
    X = np.asarray(X)
    S = np.asarray(S)
    N = X.shape[0]
    print(f'  {N} positions, feature dim {X.shape[1]}')

    # 2. route each position through the 50 leaf paths ----------------------
    #    (conditions reference base board features f0..f119, all < X.shape[1])
    leaf_of = np.full(N, -1, dtype=np.int64)
    for li, m in enumerate(leaves):
        mask = np.ones(N, dtype=bool)
        for feat, val in m['conditions']:
            mask &= (X[:, feat] == val)
        leaf_of[mask] = li
    matched = leaf_of >= 0
    print(f'  routed: {matched.sum()}/{N} positions matched a leaf '
          f'({100*matched.mean():.1f}%)')
    X, S, leaf_of = X[matched], S[matched], leaf_of[matched]
    N = X.shape[0]

    # 3. ground-truth pattern labels + per-leaf mean (= tree prediction) -----
    Y = true_pattern_activations(patterns, S)          # (N, K) uint8
    base_rate = Y.mean()
    print(f'\npattern firing base rate: {100*base_rate:.3f}%  '
          f'(a position lights up {Y.sum(1).mean():.2f} of {K} patterns on avg)')

    leaf_pred = np.zeros((len(leaves), K), dtype=np.float64)
    for li in range(len(leaves)):
        rows = leaf_of == li
        if rows.any():
            leaf_pred[li] = Y[rows].mean(axis=0)

    # 4. metrics ------------------------------------------------------------
    print('\n=== can any leaf turn any pattern ON? ===')
    print(f'  max predicted prob over all (leaf, pattern): '
          f'{leaf_pred.max():.4f}')
    for thr in (0.5, 0.25, 0.1):
        n_cells = int((leaf_pred >= thr).sum())
        n_pat = int((leaf_pred >= thr).any(axis=0).sum())
        print(f'  (leaf,pattern) cells with pred >= {thr:>4}: {n_cells:5d}  '
              f'-> {n_pat}/{K} patterns reachable')

    # position-level recall/precision at 0.5
    pred_pos = leaf_pred[leaf_of]                       # (N, K) predicted prob
    pred_bin = pred_pos >= 0.5
    tp = int((pred_bin & (Y == 1)).sum())
    fn = int((~pred_bin & (Y == 1)).sum())
    fp = int((pred_bin & (Y == 0)).sum())
    firings = tp + fn
    recall = tp / firings if firings else float('nan')
    print(f'\n=== threshold 0.5 (what the readout would binarize to) ===')
    print(f'  actual pattern firings in eval set: {firings}')
    print(f'  recovered (recall):  {tp}/{firings} = {100*recall:.3f}%')
    print(f'  false positives:     {fp}')

    # best-case: per pattern, the single leaf that predicts it highest
    best_per_pattern = leaf_pred.max(axis=0)
    print(f'\n=== best-case per pattern (max over leaves) ===')
    print(f'  patterns whose best leaf ever exceeds 0.5: '
          f'{int((best_per_pattern >= 0.5).sum())}/{K}')
    print(f'  median best-leaf prob across patterns: '
          f'{np.median(best_per_pattern):.4f}')
    print(f'  mean   best-leaf prob across patterns: '
          f'{best_per_pattern.mean():.4f}')


if __name__ == '__main__':
    main()
