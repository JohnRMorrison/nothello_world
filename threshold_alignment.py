"""For each hidden node, compute the cumulative fraction of total weight that
method A (>= 50% of max) actually captures. If this 'A-captured fraction' is
concentrated across nodes, that fraction is the right B value to use for
near-equal cell sets.

Also report agreement at the resulting matched (A, B) pair vs. the original
default (A=0.5, B=0.8).

Usage:
    python threshold_alignment.py [--ckpt PATH] [--side in|out] [--a-thresh 0.5]
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hand_crafted_flanking import enumerate_flanking_patterns
from compare_node_heuristics import (
    method_a, method_b,
    aggregate_input_to_cells, aggregate_output_to_cells,
    build_pattern_to_cell, N_MOVES,
)


def captured_fraction_for_a(v, a_threshold):
    """If method A picks K_A cells (those with v >= a*max), what fraction of
    total |v| does that sum to?"""
    if v.max() == 0 or v.sum() == 0:
        return 1.0
    cutoff = a_threshold * v.max()
    return float(v[v >= cutoff].sum() / v.sum())


def collect_cell_vectors(ckpt, side, pat_to_cell):
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
    ap.add_argument('--a-thresh', type=float, default=0.5)
    args = ap.parse_args()

    print(f"Loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    patterns = enumerate_flanking_patterns()
    pat_to_cell = build_pattern_to_cell(patterns)
    cell_vecs = collect_cell_vectors(ckpt, args.side, pat_to_cell)
    print(f"Side={args.side}, A threshold={args.a_thresh}: "
          f"{cell_vecs.shape[0]} nodes")

    # For each node, compute the fraction of |v| that A captures
    captured = np.array([
        captured_fraction_for_a(v, args.a_thresh) for v in cell_vecs
    ])
    print(f"\nA-captured fraction (per node):")
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    qvals = np.quantile(captured, qs)
    for q, v in zip(qs, qvals):
        print(f"  p{int(q*100):>2d}: {v:.3f}")
    print(f"  mean: {captured.mean():.3f}")

    # If we set B = median captured fraction, how often does B's K match A's K?
    a_lists = [method_a(v, args.a_thresh) for v in cell_vecs]

    print(f"\nUsing B = median A-captured fraction ({np.median(captured):.3f}):")
    b_med = float(np.median(captured))
    b_lists_med = [method_b(v, b_med) for v in cell_vecs]
    agree_med = sum(1 for a, b in zip(a_lists, b_lists_med) if set(a) == set(b))
    print(f"  set equality: {agree_med}/{len(cell_vecs)} "
          f"({100*agree_med/len(cell_vecs):.1f}%)")

    # Tighter: what fraction has |K_B - K_A| <= 1?
    near_med = sum(1 for a, b in zip(a_lists, b_lists_med)
                   if abs(len(set(a)) - len(set(b))) <= 1)
    print(f"  |K_A - K_B| <= 1: {near_med}/{len(cell_vecs)} "
          f"({100*near_med/len(cell_vecs):.1f}%)")

    # Sweep B around the median to find the true optimum
    print(f"\nFine sweep over B (A fixed at {args.a_thresh}):")
    b_values = np.round(np.linspace(0.3, 0.95, 14), 3)
    for b in b_values:
        b_lists = [method_b(v, b) for v in cell_vecs]
        agree = sum(1 for a, bl in zip(a_lists, b_lists) if set(a) == set(bl))
        near = sum(1 for a, bl in zip(a_lists, b_lists)
                   if abs(len(set(a)) - len(set(bl))) <= 1)
        marker = "  <-- recommended" if abs(b - b_med) < 0.05 else ""
        print(f"  B={b:.3f}: equal={agree}/{len(cell_vecs)} "
              f"({100*agree/len(cell_vecs):4.1f}%), "
              f"|dK|<=1: {100*near/len(cell_vecs):4.1f}%{marker}")

    # Compare to the original default (A=0.5, B=0.8)
    print(f"\nFor reference, default (A=0.5, B=0.8):")
    a_lists_def = [method_a(v, 0.5) for v in cell_vecs]
    b_lists_def = [method_b(v, 0.8) for v in cell_vecs]
    agree_def = sum(1 for a, b in zip(a_lists_def, b_lists_def) if set(a) == set(b))
    near_def = sum(1 for a, b in zip(a_lists_def, b_lists_def)
                   if abs(len(set(a)) - len(set(b))) <= 1)
    print(f"  equal: {agree_def}/{len(cell_vecs)} "
          f"({100*agree_def/len(cell_vecs):.1f}%)")
    print(f"  |dK|<=1: {near_def}/{len(cell_vecs)} "
          f"({100*near_def/len(cell_vecs):.1f}%)")


if __name__ == '__main__':
    main()
