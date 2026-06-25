"""For late-k multisets that the database method labels as "1 board observed"
(because only 1 game in 20M has that multiset), determine whether the
multiset is truly consistent with only 1 board, or whether it COULD admit
multiple boards — we just never saw the alternative orderings.

Approach: biased Monte Carlo.  For each sampled multiset:
  1. Start with empty Othello board
  2. At each step k, current parity = k % 2
  3. Available cells = (parity-k%2 cells in M) ∩ (legal moves at current pos)
  4. Sample uniformly from available, play
  5. Continue until all multiset cells played -> record final board hash
By construction every completed trial is a valid Othello ordering.

Repeat N=1000 trials per multiset, count distinct boards reached.  If the
count is 1, the multiset is truly order-redundant.  If >1, the database's
"1 board" label was an artifact of having only one game in the dataset.

Usage:
    python analyze_late_ambiguity.py --positions 15 20 30 40 50 \\
        --num-multisets 200 --samples-per-multiset 1000
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
    """Sample a valid Othello ordering of the (p0_cells, p1_cells) multiset
    via uniform-random choice at each step among legal x remaining.
    Returns (valid, board_state_bytes).  valid=False on dead-end."""
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
            return False, None
        chosen = random.choice(available)
        board.update([chosen])
        if parity == 0:
            remaining_p0.remove(chosen)
        else:
            remaining_p1.remove(chosen)
        step += 1
    return True, board.state.tobytes()


def measure_multiset(multiset, n_samples, max_dead_end_retries=5):
    """For a given (p0_cells, p1_cells) multiset, return:
        n_valid_trials, n_dead_ends, n_distinct_boards"""
    p0_cells, p1_cells = multiset
    distinct = set()
    n_valid = 0
    n_dead = 0
    for _ in range(n_samples):
        for _ in range(max_dead_end_retries + 1):
            valid, board = sample_one_valid_ordering(p0_cells, p1_cells)
            if valid:
                distinct.add(board)
                n_valid += 1
                break
            else:
                n_dead += 1
    return n_valid, n_dead, len(distinct)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-multisets', type=int, default=200,
                   help='How many random multisets to test per k')
    p.add_argument('--samples-per-multiset', type=int, default=1000)
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

        distinct_counts = []
        valid_trials = []
        dead_end_rates = []
        for game in tqdm(sample_games, desc=f"k={k}", leave=False):
            prefix = game[:k]
            p0 = tuple(prefix[0::2])
            p1 = tuple(prefix[1::2])
            n_valid, n_dead, n_distinct = measure_multiset(
                (p0, p1), n_samples=args.samples_per_multiset)
            distinct_counts.append(n_distinct)
            valid_trials.append(n_valid)
            dead_end_rates.append(n_dead / max(1, n_dead + n_valid))

        arr = np.array(distinct_counts)
        valid_arr = np.array(valid_trials)
        dead_arr = np.array(dead_end_rates)

        print(f"\nk={k} summary (n={len(arr)} multisets, "
              f"{args.samples_per_multiset} sample attempts each):")
        print(f"  trials per multiset that completed validly: "
              f"mean {valid_arr.mean():.0f}  median {np.median(valid_arr):.0f}")
        print(f"  dead-end rate: mean {dead_arr.mean()*100:.1f}%  "
              f"max {dead_arr.max()*100:.1f}%")
        print(f"  distinct boards per multiset:")
        print(f"    mean {arr.mean():.2f}  median {np.median(arr):.0f}  "
              f"p90 {np.percentile(arr, 90):.0f}  max {arr.max()}")
        print(f"  distinct-boards distribution:")
        cnt = Counter(arr.tolist())
        for nb in sorted(cnt):
            n_sets = cnt[nb]
            pct = 100 * n_sets / len(arr)
            print(f"    {nb:>3} boards: {n_sets:>4} multisets ({pct:5.1f}%)")
        print()


if __name__ == '__main__':
    main()
