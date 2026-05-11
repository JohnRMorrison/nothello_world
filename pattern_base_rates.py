"""Step 0 for the class-imbalance hypothesis: measure how often each of the
960 flanking patterns fires in training data, bucketed by target-cell class
(corner / edge / inner).

If corner-target patterns fire 30-100x less often than inner-target patterns,
class imbalance is a strong candidate for the recall@K gap at edges/corners.
If the skew is only 2-3x, look elsewhere.

Usage:
    python pattern_base_rates.py
"""
import sys, os
sys.path.insert(0, '.')

import numpy as np
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, N_MOVES,
)
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import compute_pattern_labels_batch


CENTER_CELLS_64 = {27, 28, 35, 36}   # d4, e4, d5, e5 -- excluded from MOVE_TO_IDX


def idx60_to_square(i):
    """Map cell index 0..59 (over the 60 non-center squares) to 0..63."""
    j = -1
    for s in range(64):
        if s in CENTER_CELLS_64:
            continue
        j += 1
        if j == i:
            return s
    raise ValueError(i)


def classify_cell(idx60):
    sq = idx60_to_square(idx60)
    row, col = sq // 8, sq % 8
    if row in (0, 7) and col in (0, 7):
        return 'corner'
    if row in (0, 7) or col in (0, 7):
        return 'edge'
    return 'inner'


def idx60_to_alg(i):
    sq = idx60_to_square(i)
    return f"{'abcdefgh'[sq % 8]}{sq // 8 + 1}"


if __name__ == "__main__":
    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = \
        precompute_pattern_arrays(patterns)

    pattern_cell = np.array([MOVE_TO_IDX[p['target']] for p in patterns])
    pattern_class = np.array([classify_cell(c) for c in pattern_cell])
    pattern_length = np.array([len(p['opponents']) for p in patterns])
    print(f"Total patterns: {len(patterns)}")
    print(f"  corner-target patterns: {(pattern_class == 'corner').sum()}")
    print(f"  edge-target patterns:   {(pattern_class == 'edge').sum()}")
    print(f"  inner-target patterns:  {(pattern_class == 'inner').sum()}")

    output_dir = "experiments/mathematical_transformation_experiments/heuristic_probe_results"
    chunk_dir = os.path.join(output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz")
                         and "_patterns" not in f
                         and "_when60" not in f)
    eval_path = chunk_files[-1]
    print(f"\nLoading {eval_path}")

    X, Y, pos = _load_features(eval_path)
    del X     # we don't need the 180-d features here -- save memory
    N = len(Y)
    print(f"Positions in chunk: {N}")

    n = min(N, 49 * 10000)
    rng = np.random.RandomState(0)
    si = np.sort(rng.choice(N, n, replace=False))
    Y, pos = Y[si], pos[si]
    print(f"Sampled to: {n}")

    counts = np.zeros(len(patterns), dtype=np.int64)
    batch = 50_000
    for i in range(0, n, batch):
        yb = Y[i:i + batch].numpy()
        pb = pos[i:i + batch].numpy()
        labels = compute_pattern_labels_batch(
            yb, pb, pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
        counts += labels.sum(axis=0)

    print()
    print("=" * 68)
    print("Per-class aggregate firing rates")
    print("=" * 68)
    print(f"{'class':>8s} {'n_pat':>6s} {'total_fires':>12s} "
          f"{'avg/pat':>10s} {'base_rate':>12s}")
    rates = {}
    for cls in ('corner', 'edge', 'inner'):
        m = pattern_class == cls
        npat = int(m.sum())
        total = int(counts[m].sum())
        per_pat = total / max(npat, 1)
        rate = per_pat / n
        rates[cls] = per_pat
        print(f"{cls:>8s} {npat:>6d} {total:>12d} "
              f"{per_pat:>10.1f} {rate:>12.5f}")

    print()
    print("Imbalance ratios (mean fires per pattern):")
    print(f"  inner / edge:   {rates['inner'] / max(rates['edge'],   1e-9):>7.2f}x")
    print(f"  inner / corner: {rates['inner'] / max(rates['corner'], 1e-9):>7.2f}x")
    print(f"  edge  / corner: {rates['edge']  / max(rates['corner'], 1e-9):>7.2f}x")

    print()
    print("=" * 68)
    print("Per-pattern distribution within each class")
    print("=" * 68)
    print(f"{'class':>8s} {'min':>8s} {'p25':>8s} {'median':>8s} "
          f"{'p75':>8s} {'max':>8s}")
    for cls in ('corner', 'edge', 'inner'):
        sub = counts[pattern_class == cls]
        q = np.percentile(sub, [0, 25, 50, 75, 100])
        print(f"{cls:>8s} {int(q[0]):>8d} {int(q[1]):>8d} "
              f"{int(q[2]):>8d} {int(q[3]):>8d} {int(q[4]):>8d}")

    print()
    print("=" * 68)
    print("Pattern firing rate by pattern length (# opponent cells)")
    print("=" * 68)
    print(f"{'length':>8s} {'n_pat':>6s} {'avg/pat':>10s}")
    for L in sorted(set(pattern_length.tolist())):
        m = pattern_length == L
        npat = int(m.sum())
        total = int(counts[m].sum())
        print(f"{L:>8d} {npat:>6d} {total / max(npat, 1):>10.1f}")

    print()
    print("Bottom 10 patterns by total fires (rare patterns):")
    order = np.argsort(counts)
    for k in range(10):
        i = order[k]
        p = patterns[i]
        print(f"  {idx60_to_alg(int(pattern_cell[i])):>3s}  "
              f"length={pattern_length[i]}  fires={counts[i]:>8d}  "
              f"class={pattern_class[i]:>6s}")

    print("\nTop 10 patterns by total fires (common patterns):")
    for k in range(1, 11):
        i = order[-k]
        p = patterns[i]
        print(f"  {idx60_to_alg(int(pattern_cell[i])):>3s}  "
              f"length={pattern_length[i]}  fires={counts[i]:>8d}  "
              f"class={pattern_class[i]:>6s}")
