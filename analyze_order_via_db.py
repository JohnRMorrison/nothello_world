"""Empirically measure board-state ambiguity by finding REAL games that
share the same SET of (cell, parity) moves at a given prefix length k.

If two games have the same first-k multiset of (cell, parity) pairs but
were played in DIFFERENT orders, they constitute a natural "ambiguity
example" — same input to an order-blind model, but possibly different
board states.

For each k, we:
  1. Compute a canonical key for each game's prefix:
       (sorted tuple of parity-0 cells, sorted tuple of parity-1 cells)
  2. Group games by this key
  3. Count games per key (= "popularity" of this played-set)
  4. Count DISTINCT board states per key (= "ambiguity" of this set)

If most keys with multiple games still yield 1 distinct board, move order
is empirically redundant for board state.  If they yield many distinct
boards, the order-blind model faces real ambiguity.

Usage:
    python analyze_order_via_db.py --num-games 200000
"""
import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, '.')
from data.othello import OthelloBoardState


def load_games(data_dir='./data/othello_synthetic', num_files=2):
    files = sorted(os.listdir(data_dir))
    out = []
    for fname in files[-num_files:]:
        with open(os.path.join(data_dir, fname), 'rb') as f:
            batch = pickle.load(f)
        if len(batch) >= 9e4:
            out.extend(batch)
    return out


def canonical_key(prefix):
    """Hashable canonical representation of the prefix's parity-tagged
    multiset: (sorted parity-0 cells, sorted parity-1 cells)."""
    p0 = tuple(sorted(prefix[0::2]))
    p1 = tuple(sorted(prefix[1::2]))
    return (p0, p1)


def board_hash(prefix):
    """Replay prefix and return a hashable representation of the
    resulting board state."""
    board = OthelloBoardState()
    for m in prefix:
        board.update([m])
    return board.state.tobytes()


def analyze_at_k(games, k):
    """Group games by canonical key at depth k.  For each key, record the
    set of board states reached.  Returns (count_per_key, boards_per_key)."""
    count_per_key = Counter()
    boards_per_key = defaultdict(set)
    for game in tqdm(games, leave=False):
        if len(game) <= k:
            continue
        prefix = game[:k]
        key = canonical_key(prefix)
        count_per_key[key] += 1
        boards_per_key[key].add(board_hash(prefix))
    return count_per_key, boards_per_key


def summarize(k, count_per_key, boards_per_key):
    keys = list(count_per_key)
    n_keys = len(keys)
    n_games_total = sum(count_per_key.values())
    counts = np.array([count_per_key[k_] for k_ in keys])
    n_boards = np.array([len(boards_per_key[k_]) for k_ in keys])

    # Restrict to keys that appear in at least 2 games (where the
    # ambiguity question is meaningful).
    multi_mask = counts >= 2
    n_multi = int(multi_mask.sum())

    print(f"k={k}:")
    print(f"  total games processed:        {n_games_total:>9d}")
    print(f"  unique played-sets (keys):    {n_keys:>9d}")
    print(f"  keys with >= 2 games:         {n_multi:>9d}  "
          f"({100*n_multi/max(1,n_keys):.1f}%)")
    print(f"  fraction of games whose key is shared with at least one other: "
          f"{100*(counts[multi_mask].sum())/max(1,n_games_total):.1f}%")

    if n_multi == 0:
        print(f"  (no shared keys to analyze)")
        return

    multi_counts = counts[multi_mask]
    multi_boards = n_boards[multi_mask]

    print(f"  among shared-key sets (n={n_multi}):")
    print(f"    games per set:  mean {multi_counts.mean():.2f}  "
          f"median {np.median(multi_counts):.0f}  "
          f"p90 {np.percentile(multi_counts, 90):.0f}  "
          f"max {multi_counts.max()}")
    print(f"    distinct boards per set:  mean {multi_boards.mean():.3f}  "
          f"median {np.median(multi_boards):.0f}  "
          f"p90 {np.percentile(multi_boards, 90):.0f}  "
          f"max {multi_boards.max()}")
    # Distribution of distinct-boards
    board_distribution = Counter(multi_boards.tolist())
    print(f"    distinct-boards distribution among shared-key sets:")
    for nb in sorted(board_distribution):
        n_sets = board_distribution[nb]
        pct = 100 * n_sets / n_multi
        # Also: what fraction of GAMES are in sets of this ambiguity level?
        games_in_these = sum(count_per_key[k_]
                              for k_ in keys
                              if count_per_key[k_] >= 2
                              and len(boards_per_key[k_]) == nb)
        pct_games = 100 * games_in_these / n_games_total
        print(f"      {nb:>3} distinct board(s):  "
              f"{n_sets:>6} sets  ({pct:5.1f}% of sets, "
              f"{pct_games:5.1f}% of games)")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-games', type=int, default=200000,
                   help='How many games to load')
    p.add_argument('--positions', type=int, nargs='+',
                   default=[3, 5, 8, 10, 12, 15, 20, 25, 30, 40])
    p.add_argument('--num-pickle-files', type=int, default=2)
    p.add_argument('--data-dir', default='./data/othello_synthetic')
    args = p.parse_args()

    print(f"Loading games from {args.num_pickle_files} pickle file(s)...")
    games = load_games(args.data_dir, args.num_pickle_files)
    games = games[:args.num_games]
    print(f"  {len(games)} games loaded\n")

    for k in args.positions:
        count_per_key, boards_per_key = analyze_at_k(games, k)
        summarize(k, count_per_key, boards_per_key)


if __name__ == '__main__':
    main()
