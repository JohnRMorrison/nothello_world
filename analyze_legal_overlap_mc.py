"""Biased Monte Carlo analysis of LEGAL-MOVE INTERSECTION across the
distinct boards consistent with a parity-tagged multiset.

For each random multiset M at depth k:
  1. Sample many valid Othello orderings of M via biased MC
     (at each step pick uniformly from legal x remaining-of-correct-parity)
  2. For each completed ordering, replay to its final board state and
     record the set of legal moves at that position
  3. Group by board state — each distinct board contributes one legal-moves set
  4. Across distinct boards, compute:
       |intersection| = cells legal under EVERY consistent board
       |union|        = cells legal under AT LEAST ONE consistent board

Why this matters: the order-blind models achieve ~99% top-1 legal-move
accuracy even when the multiset is consistent with 10+ board states.  The
intersection size puts a structural ceiling on this performance — an
order-blind ideal predictor can "safely" pick any cell in the intersection.

If |intersection| is typically >= 1, then ~99% top-1 is achievable from
any order-blind input.  If |intersection| often == 0, the model must be
doing something more sophisticated (Bayesian board reasoning).

Usage:
    python analyze_legal_overlap_mc.py --positions 15 20 30 \
        --num-multisets 50 --samples-per-multiset 100
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
    """Sample a valid Othello ordering of the (p0_cells, p1_cells) multiset.
    Returns (valid, board_state_bytes, frozenset_of_legal_moves_at_final_pos).
    valid=False on dead-end."""
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
    # All multiset moves played; record final board state + legal moves
    legal_at_end = frozenset(board.get_valid_moves())
    return True, board.state.tobytes(), legal_at_end


def measure_multiset_overlap(multiset, n_samples, max_dead_end_retries=5):
    """For a given (p0_cells, p1_cells) multiset, sample valid orderings and
    return per-distinct-board legal-move sets.  Returns dict:
       board_hash -> legal_move_set"""
    p0_cells, p1_cells = multiset
    board_to_legal = {}
    n_valid = 0
    n_dead = 0
    for _ in range(n_samples):
        for _ in range(max_dead_end_retries + 1):
            valid, board, legal = sample_one_valid_ordering(p0_cells, p1_cells)
            if valid:
                # Keep one legal-moves set per distinct board (they're identical
                # for the same board state regardless of which ordering reached it)
                if board not in board_to_legal:
                    board_to_legal[board] = legal
                n_valid += 1
                break
            else:
                n_dead += 1
    return n_valid, n_dead, board_to_legal


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-multisets', type=int, default=50)
    p.add_argument('--samples-per-multiset', type=int, default=100)
    p.add_argument('--positions', type=int, nargs='+',
                   default=[15, 20, 30, 40, 50])
    p.add_argument('--num-pickle-files', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading games...")
    games = load_games(num_files=args.num_pickle_files)
    print(f"  {len(games)} games loaded\n")

    for k in args.positions:
        eligible = [g for g in games if len(g) > k]
        sample_games = random.sample(eligible,
                                      min(args.num_multisets, len(eligible)))
        print(f"k={k}: sampling {args.samples_per_multiset} valid orderings of "
              f"each of {len(sample_games)} multisets...")

        intersection_sizes = []
        union_sizes = []
        n_boards_list = []
        nonempty_intersection = 0
        sole_distinct_legal = 0  # multisets with only 1 distinct legal-moves set across all boards

        for game in tqdm(sample_games, desc=f"k={k}", leave=False):
            prefix = game[:k]
            p0 = tuple(prefix[0::2])
            p1 = tuple(prefix[1::2])
            n_valid, n_dead, board_to_legal = measure_multiset_overlap(
                (p0, p1), n_samples=args.samples_per_multiset)

            if len(board_to_legal) == 0:
                continue  # MC failed to find any valid ordering; skip
            n_boards_list.append(len(board_to_legal))

            legal_sets = list(board_to_legal.values())
            inter = legal_sets[0]
            uni = set(legal_sets[0])
            for L in legal_sets[1:]:
                inter = inter & L
                uni = uni | L
            intersection_sizes.append(len(inter))
            union_sizes.append(len(uni))
            if inter:
                nonempty_intersection += 1

            # Are all distinct boards yielding the SAME legal-move set?
            unique_legal_sets = set(frozenset(L) for L in legal_sets)
            if len(unique_legal_sets) == 1:
                sole_distinct_legal += 1

        n = len(intersection_sizes)
        if n == 0:
            print(f"  (no valid samples found at this k)")
            continue

        inter_arr = np.array(intersection_sizes)
        union_arr = np.array(union_sizes)
        boards_arr = np.array(n_boards_list)
        ratios = inter_arr / np.maximum(union_arr, 1)

        print(f"\nk={k} summary (n={n} multisets, "
              f"each sampled {args.samples_per_multiset} times):")
        print(f"  Distinct boards per multiset:")
        print(f"    mean {boards_arr.mean():.2f}  "
              f"median {np.median(boards_arr):.0f}  "
              f"max {boards_arr.max()}")
        print(f"  |intersection| (legal under ALL consistent boards):")
        print(f"    mean {inter_arr.mean():.2f}  "
              f"median {np.median(inter_arr):.0f}  "
              f"min {inter_arr.min()}  "
              f"max {inter_arr.max()}")
        print(f"  |union| (legal under ANY consistent board):")
        print(f"    mean {union_arr.mean():.2f}  "
              f"median {np.median(union_arr):.0f}  "
              f"max {union_arr.max()}")
        print(f"  intersection/union ratio:  mean {ratios.mean():.3f}")
        print(f"  non-empty intersection:    "
              f"{100*nonempty_intersection/n:.1f}% of multisets")
        print(f"  all boards share legal set: "
              f"{100*sole_distinct_legal/n:.1f}% of multisets")

        print(f"\n  Distribution of |intersection|:")
        cnt = Counter(inter_arr.tolist())
        for sz in sorted(cnt):
            pct = 100 * cnt[sz] / n
            print(f"    |intersection| = {sz:>2}: {cnt[sz]:>3} multisets ({pct:5.1f}%)")
        print()


if __name__ == '__main__':
    main()
