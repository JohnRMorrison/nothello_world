"""Test the compressibility hypothesis: maybe the network doesn't need to
represent board state, just the multiset -> legal-move-intersection mapping.
If many multisets share the same intersection set, the model can learn a
small lookup-style "kind detector" instead of full board reasoning.

For each k, sample many multisets, compute each multiset's
legal-move-intersection (= cells legal under ALL consistent boards),
then count how many DISTINCT intersection sets occur.

If 1000 multisets have only 50 distinct intersection sets → 20x compression.
If 1000 multisets have ~1000 distinct intersection sets → no compression.

Usage:
    python analyze_overlap_compressibility.py --positions 15 20 30 40 50 \
        --num-multisets 500 --samples-per-multiset 30
"""
import argparse
import os
import pickle
import random
import sys
from collections import Counter

import numpy as np
from tqdm import tqdm

sys.path.insert(0, '.')
from data.othello import OthelloBoardState


def load_games(data_dir='./data/othello_synthetic', num_files=1):
    files = sorted(os.listdir(data_dir))
    out = []
    for fname in files[-num_files:]:
        with open(os.path.join(data_dir, fname), 'rb') as f:
            batch = pickle.load(f)
        if len(batch) >= 9e4:
            out.extend(batch)
    return out


def sample_one_valid_ordering(p0_cells, p1_cells):
    remaining_p0 = list(p0_cells)
    remaining_p1 = list(p1_cells)
    board = OthelloBoardState()
    step = 0
    while remaining_p0 or remaining_p1:
        parity = step % 2
        cands = remaining_p0 if parity == 0 else remaining_p1
        legal = set(board.get_valid_moves())
        available = [c for c in cands if c in legal]
        if not available:
            return False, None, None
        chosen = random.choice(available)
        board.update([chosen])
        if parity == 0:
            remaining_p0.remove(chosen)
        else:
            remaining_p1.remove(chosen)
        step += 1
    return True, board.state.tobytes(), frozenset(board.get_valid_moves())


def intersection_for_multiset(multiset, n_samples, max_retries=3):
    """Run biased MC, return the intersection of legal-move sets across all
    distinct boards observed.  Returns frozenset (possibly empty)."""
    p0_cells, p1_cells = multiset
    board_to_legal = {}
    for _ in range(n_samples):
        for _ in range(max_retries + 1):
            valid, board, legal = sample_one_valid_ordering(p0_cells, p1_cells)
            if valid:
                if board not in board_to_legal:
                    board_to_legal[board] = legal
                break
    if not board_to_legal:
        return None
    legal_sets = list(board_to_legal.values())
    inter = legal_sets[0]
    for L in legal_sets[1:]:
        inter = inter & L
    return inter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-multisets', type=int, default=500)
    p.add_argument('--samples-per-multiset', type=int, default=30)
    p.add_argument('--positions', type=int, nargs='+',
                   default=[15, 20, 30, 40, 50])
    p.add_argument('--num-pickle-files', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading games...")
    games = load_games(num_files=args.num_pickle_files)
    print(f"  {len(games)} games loaded\n")

    for k in args.positions:
        eligible = [g for g in games if len(g) > k]
        sample_games = random.sample(eligible,
                                      min(args.num_multisets, len(eligible)))
        print(f"k={k}: computing intersection set for {len(sample_games)} multisets...")

        intersection_to_count = Counter()  # intersection set -> # multisets with this set
        intersection_sizes = []
        n_valid = 0
        n_empty_intersection = 0

        for game in tqdm(sample_games, desc=f"k={k}", leave=False):
            prefix = game[:k]
            p0 = tuple(prefix[0::2])
            p1 = tuple(prefix[1::2])
            inter = intersection_for_multiset(
                (p0, p1), n_samples=args.samples_per_multiset)
            if inter is None:
                continue
            n_valid += 1
            intersection_to_count[inter] += 1
            intersection_sizes.append(len(inter))
            if not inter:
                n_empty_intersection += 1

        if n_valid == 0:
            print(f"  (no valid samples at k={k})\n")
            continue

        n_distinct_inter = len(intersection_to_count)
        compression_ratio = n_valid / n_distinct_inter

        # How concentrated is the distribution?
        counts = np.array(list(intersection_to_count.values()))
        counts_sorted = np.sort(counts)[::-1]
        cum = np.cumsum(counts_sorted) / n_valid

        # Find smallest set of intersection patterns that covers X% of multisets
        def kths(threshold):
            idx = np.argmax(cum >= threshold) + 1
            return idx, 100 * counts_sorted[:idx].sum() / n_valid

        k50, p50 = kths(0.50)
        k90, p90 = kths(0.90)
        k99, p99 = kths(0.99)

        print(f"\nk={k} summary (n={n_valid} multisets with valid samples):")
        print(f"  Distinct intersection sets:           {n_distinct_inter:>5}")
        print(f"  Multisets / distinct intersection:    {compression_ratio:>6.2f}  "
              f"(compression ratio)")
        print(f"  Distinct empty intersections:         "
              f"{1 if n_empty_intersection else 0} ({n_empty_intersection} multisets)")
        print(f"  Top intersection pattern coverage:")
        print(f"    Top {k50:>4} patterns cover 50% of multisets")
        print(f"    Top {k90:>4} patterns cover 90% of multisets")
        print(f"    Top {k99:>4} patterns cover 99% of multisets")
        print(f"  Multisets per pattern: "
              f"mean {counts.mean():.2f}  max {counts.max()}")
        print(f"  Most common patterns:")
        for inter, count in intersection_to_count.most_common(5):
            cells = sorted(inter)
            cell_str = ", ".join(str(c) for c in cells[:8])
            if len(cells) > 8:
                cell_str += f", ... ({len(cells)} cells total)"
            elif not cells:
                cell_str = "(empty set)"
            pct = 100 * count / n_valid
            print(f"    [{cell_str}] -> {count} multisets ({pct:.1f}%)")
        print()


if __name__ == '__main__':
    main()
