"""Sweep over (A threshold, B fraction) pairs and report how often the two
methods produce identical cell sets per node.

Use this to find threshold pairs where the weights-based heuristic
attribution is robust to method choice.

Usage:
    python sweep_threshold_agreement.py [--ckpt PATH] [--side in|out]
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hand_crafted_flanking import enumerate_flanking_patterns
from compare_node_heuristics import (
    method_a, method_b, aggregate_input_to_cells, aggregate_output_to_cells,
    build_pattern_to_cell, N_MOVES,
)


def collect_cell_vectors(ckpt, side, pat_to_cell):
    """Return a (n_nodes, 60) array of per-cell weight magnitudes, one row per
    hidden node, concatenating both parities."""
    rows = []
    for parity in ('even', 'odd'):
        sd = ckpt[parity]
        W1 = sd['net.0.weight'].detach().cpu().numpy()
        W2 = sd['net.2.weight'].detach().cpu().numpy()
        H = W1.shape[0]
        for j in range(H):
            if side == 'in':
                rows.append(aggregate_input_to_cells(W1[j]))
            else:
                rows.append(aggregate_output_to_cells(W2[:, j], pat_to_cell))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='experiments/mathematical_transformation_experiments/'
                                       'heuristic_probe_results/pattern_detector_checkpoints/'
                                       'pattern_simple_direct_H512_wheneven.pt')
    ap.add_argument('--side', choices=['in', 'out'], default='in')
    args = ap.parse_args()

    print(f"Loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    patterns = enumerate_flanking_patterns()
    pat_to_cell = build_pattern_to_cell(patterns)

    cell_vecs = collect_cell_vectors(ckpt, args.side, pat_to_cell)
    print(f"Side={args.side}: {cell_vecs.shape[0]} nodes x {cell_vecs.shape[1]} cells")

    a_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    b_fractions  = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    n_nodes = cell_vecs.shape[0]
    # Precompute method outputs at each threshold
    a_sets = {a: [frozenset(method_a(v, a)) for v in cell_vecs] for a in a_thresholds}
    b_sets = {b: [frozenset(method_b(v, b)) for v in cell_vecs] for b in b_fractions}

    print(f"\n{'A threshold':>12} " + " ".join(f"{b:6.2f}" for b in b_fractions))
    best = (0.0, None, None)
    for a in a_thresholds:
        row = []
        for b in b_fractions:
            agree = sum(1 for sa, sb in zip(a_sets[a], b_sets[b]) if sa == sb)
            frac = agree / n_nodes
            row.append(frac)
            if frac > best[0]:
                best = (frac, a, b)
        print(f"{a:>12.2f} " + " ".join(f"{r:6.3f}" for r in row))

    print(f"\nBest pair: A={best[1]}, B={best[2]} -> "
          f"{100*best[0]:.1f}% nodes agree")

    # At the best pair, print the count distributions
    best_a, best_b = best[1], best[2]
    sa = a_sets[best_a]
    sb = b_sets[best_b]
    a_counts = np.array([len(s) for s in sa])
    b_counts = np.array([len(s) for s in sb])
    print(f"\nAt best pair:")
    print(f"  A count: median={np.median(a_counts):.1f}, mean={a_counts.mean():.2f}")
    print(f"  B count: median={np.median(b_counts):.1f}, mean={b_counts.mean():.2f}")
    jaccards = np.array([
        len(s1 & s2) / max(len(s1 | s2), 1)
        for s1, s2 in zip(sa, sb)
    ])
    print(f"  Mean Jaccard(A, B): {jaccards.mean():.3f}")


if __name__ == '__main__':
    main()
