"""Monte Carlo analysis of how recoverable move-order is from the played
SET of moves (with parities preserved).

For each game and several prefix lengths k, we:
  1. Take the prefix M_1..M_k as a multiset of (cell, parity) pairs
  2. Randomly sample N orderings that interleave the parity-0 cells and
     parity-1 cells (each parity permuted independently)
  3. For each sampled ordering, replay as Othello and check whether it's
     a VALID game sequence (each move must be a legal flank at the time)
  4. For valid orderings, record the resulting board state

This estimates two important quantities:
  - frac_valid : probability that a random ordering of the played set
                 is a valid game.  1.0 = order is fully redundant.
                 ~0   = order is highly constrained.
  - n_distinct_boards : how many distinct board states are reachable
                        from the played set.  1 = order doesn't matter
                        for board state.  >1 = there's irreducible
                        ambiguity that the order-blind predictor faces.

The second metric (distinct boards) is the real upper bound on what an
order-blind predictor can know about the board.

Usage:
    python analyze_order_ambiguity.py --num-games 100 --samples 1000

Output: per-k summary stats plus the empirical distribution of
        distinct-board counts.
"""
import argparse
import copy
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


def replay(moves):
    """Try to play `moves` as an Othello game.

    Returns (valid, board_state_hash).  If any move isn't legal at the
    time of play, returns (False, None).
    """
    board = OthelloBoardState()
    for m in moves:
        if m not in board.get_valid_moves():
            return False, None
        board.update([m])
    return True, board.state.tobytes()


def sample_orderings(prefix, n_samples):
    """Sample n_samples interleavings of `prefix` that respect parity.

    Move at original index 0 (M_1) has parity 0; index 1 (M_2) has parity
    1; etc.  We shuffle the parity-0 cells and parity-1 cells
    independently, then interleave them back in alternating slots.
    """
    p0 = prefix[0::2]  # M_1, M_3, M_5, ...
    p1 = prefix[1::2]  # M_2, M_4, M_6, ...
    k = len(prefix)
    samples = []
    for _ in range(n_samples):
        a = list(p0)
        b = list(p1)
        random.shuffle(a)
        random.shuffle(b)
        seq = []
        for i in range(k):
            if i % 2 == 0:
                seq.append(a[i // 2])
            else:
                seq.append(b[i // 2])
        samples.append(seq)
    return samples


def analyze_position(game, k, n_samples):
    """Return per-position metrics dict for the prefix game[:k]."""
    prefix = game[:k]
    samples = sample_orderings(prefix, n_samples)

    n_valid = 0
    distinct_boards = set()
    for seq in samples:
        valid, board_hash = replay(seq)
        if valid:
            n_valid += 1
            distinct_boards.add(board_hash)

    return {
        'k': k,
        'n_samples': n_samples,
        'n_valid': n_valid,
        'frac_valid': n_valid / n_samples,
        'n_distinct_boards': len(distinct_boards),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-games', type=int, default=100,
                   help='How many val games to analyze')
    p.add_argument('--samples', type=int, default=1000,
                   help='Monte Carlo orderings per (game, k)')
    p.add_argument('--positions', type=int, nargs='+',
                   default=[3, 5, 8, 12, 16, 20, 30, 40, 50])
    p.add_argument('--num-pickle-files', type=int, default=1)
    p.add_argument('--data-dir', default='./data/othello_synthetic')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-csv', default=None,
                   help='Optional CSV path for per-(game, k) results')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading games from {args.data_dir}...")
    games = load_games(data_dir=args.data_dir, num_files=args.num_pickle_files)
    games = games[:args.num_games]
    print(f"  {len(games)} games\n")

    # Run analysis per (game, k)
    per_record = []
    for k in args.positions:
        print(f"k={k} (Monte Carlo, {args.samples} samples/game)...")
        for game in tqdm(games, leave=False):
            if len(game) <= k:
                continue
            r = analyze_position(game, k, args.samples)
            r['game_idx'] = games.index(game) if False else -1  # idx not needed
            per_record.append(r)

    # Summarize per k
    print("\n" + "=" * 80)
    print("Summary (mean across games):")
    print(f"{'k':>4}  {'frac_valid':>10}  {'n_orderings_valid':>20}  "
          f"{'mean distinct boards':>22}  {'median':>8}  {'p90':>6}  {'max':>6}")
    print("-" * 80)
    by_k = {}
    for r in per_record:
        by_k.setdefault(r['k'], []).append(r)
    for k in args.positions:
        rs = by_k.get(k, [])
        if not rs:
            print(f"{k:>4}  (no data)")
            continue
        mean_fv = np.mean([r['frac_valid'] for r in rs])
        mean_nv = np.mean([r['n_valid'] for r in rs])
        boards_arr = np.array([r['n_distinct_boards'] for r in rs])
        # Note: n_distinct_boards is bounded by n_valid which is bounded by samples.
        print(f"{k:>4}  {mean_fv*100:>9.4f}%  {mean_nv:>20.2f}  "
              f"{boards_arr.mean():>22.2f}  "
              f"{np.median(boards_arr):>8.1f}  "
              f"{np.percentile(boards_arr, 90):>6.0f}  "
              f"{int(boards_arr.max()):>6}")

    # Print the distribution of distinct-boards at a representative k
    rep_k = max(args.positions[0],
                min(args.positions[len(args.positions)//2], 20))
    rs = by_k.get(rep_k, [])
    if rs:
        print(f"\nDistribution of n_distinct_boards at k={rep_k}:")
        counter = Counter(r['n_distinct_boards'] for r in rs)
        for nb in sorted(counter):
            pct = 100 * counter[nb] / len(rs)
            print(f"  {nb:>4} board(s):  {counter[nb]:>4} games  ({pct:5.1f}%)")

    if args.out_csv:
        import csv
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['k', 'n_samples', 'n_valid',
                                              'frac_valid',
                                              'n_distinct_boards'])
            w.writeheader()
            for r in per_record:
                w.writerow({k: r[k] for k in w.fieldnames})
        print(f"\nWrote {args.out_csv}")


if __name__ == '__main__':
    main()
