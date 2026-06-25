"""For each (cell, parity) multiset that's consistent with MULTIPLE distinct
board states (the "ambiguous" multisets we found in analyze_order_via_db.py),
compute the intersection and union of legal moves across those boards.

If two real games share the same multiset but reach different boards, an
order-blind model sees the same input and has to pick a move that's legal
under both boards.  The legal-move set under each board might be:

    Board A:  {C5, D6, F4, G3}        |A| = 4
    Board B:  {C5, F4, F6, G7}        |B| = 4
    Intersection:  {C5, F4}           — safe under both
    Union:         {C5, D6, F4, F6, G3, G7}   — needs to consider all

The model's top-1 prediction can be a "safe" pick from the intersection.
Top-K (K > |intersection|) starts including cells that are only legal
under SOME of the consistent boards — explaining why top-1 stays high
while top-K falls off.

For each multiset with multiple distinct boards:
  - |intersection|  : legal-move set guaranteed legal regardless of order
  - |union|         : maximum size of a "potentially legal" candidate pool
  - intersection / union : the "shared-legality" ratio

Output: distributions of these metrics across multisets.

Usage:
    python analyze_legal_overlap.py --num-games 20000000 --positions 8 12 15
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
    p0 = tuple(sorted(prefix[0::2]))
    p1 = tuple(sorted(prefix[1::2]))
    return (p0, p1)


def board_and_legal(prefix):
    """Replay prefix; return (board_state_bytes, frozenset_of_legal_moves)."""
    board = OthelloBoardState()
    for m in prefix:
        board.update([m])
    return board.state.tobytes(), frozenset(board.get_valid_moves())


def analyze_at_k(games, k):
    """For each multiset key, record the legal-move set under each distinct
    board reached.  Returns dict: key -> {board_hash: frozenset(legal_moves)}."""
    key_to_boards = defaultdict(dict)  # key -> {board_hash: legal_moves}
    for game in tqdm(games, leave=False):
        if len(game) <= k:
            continue
        prefix = game[:k]
        key = canonical_key(prefix)
        if key in key_to_boards:
            # Possibly the same board we've already seen for this key
            bh, legal = board_and_legal(prefix)
            if bh not in key_to_boards[key]:
                key_to_boards[key][bh] = legal
        else:
            bh, legal = board_and_legal(prefix)
            key_to_boards[key][bh] = legal
    return key_to_boards


def summarize(k, key_to_boards):
    """Compute intersection/union of legal moves across consistent boards
    for each multiset with >=2 distinct boards."""
    multi_board_keys = [k_ for k_, bd in key_to_boards.items() if len(bd) >= 2]
    single_board_keys = [k_ for k_, bd in key_to_boards.items() if len(bd) == 1]

    print(f"\nk={k}:")
    print(f"  total unique keys:                       {len(key_to_boards):>9d}")
    print(f"  single-distinct-board keys:              {len(single_board_keys):>9d}")
    print(f"  multi-distinct-board keys (focus group): {len(multi_board_keys):>9d}")

    if not multi_board_keys:
        print(f"  (no multi-board multisets — nothing to analyze for overlap)")
        return

    # Stats over multi-board multisets
    intersection_sizes = []
    union_sizes = []
    n_boards_list = []
    nonempty_intersection = 0
    for k_ in multi_board_keys:
        boards = list(key_to_boards[k_].values())
        n_boards_list.append(len(boards))
        inter = boards[0]
        uni = set(boards[0])
        for b in boards[1:]:
            inter = inter & b
            uni = uni | b
        intersection_sizes.append(len(inter))
        union_sizes.append(len(uni))
        if inter:
            nonempty_intersection += 1

    intersection_arr = np.array(intersection_sizes)
    union_arr = np.array(union_sizes)
    n_boards_arr = np.array(n_boards_list)
    n = len(multi_board_keys)

    print(f"\n  Stats over {n} multi-board multisets:")
    print(f"    distinct boards per multiset:")
    print(f"      mean {n_boards_arr.mean():.2f}  median {np.median(n_boards_arr):.0f}  "
          f"p90 {np.percentile(n_boards_arr, 90):.0f}  max {n_boards_arr.max()}")
    print(f"    |union| (max plausible legal moves):")
    print(f"      mean {union_arr.mean():.2f}  median {np.median(union_arr):.0f}  "
          f"p90 {np.percentile(union_arr, 90):.0f}  max {union_arr.max()}")
    print(f"    |intersection| (safely-legal under all boards):")
    print(f"      mean {intersection_arr.mean():.2f}  median {np.median(intersection_arr):.0f}  "
          f"p90 {np.percentile(intersection_arr, 90):.0f}  max {intersection_arr.max()}")
    print(f"    non-empty intersection: "
          f"{100*nonempty_intersection/n:.2f}% of multi-board multisets")

    # Ratio intersection/union
    ratios = [i/u for i, u in zip(intersection_sizes, union_sizes) if u > 0]
    print(f"    intersection/union ratio: mean {np.mean(ratios):.3f}")

    # Distribution of intersection sizes
    print(f"\n  Distribution of |intersection|:")
    int_dist = Counter(intersection_sizes)
    for sz in sorted(int_dist):
        pct = 100 * int_dist[sz] / n
        print(f"    |intersection| = {sz:>2}: {int_dist[sz]:>7} multisets ({pct:5.1f}%)")

    # Distribution of union sizes
    print(f"\n  Distribution of |union|:")
    uni_dist = Counter(union_sizes)
    for sz in sorted(uni_dist):
        pct = 100 * uni_dist[sz] / n
        if pct >= 0.1:
            print(f"    |union| = {sz:>2}: {uni_dist[sz]:>7} multisets ({pct:5.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-games', type=int, default=200000)
    p.add_argument('--positions', type=int, nargs='+', default=[8, 10, 12, 15])
    p.add_argument('--num-pickle-files', type=int, default=2)
    p.add_argument('--data-dir', default='./data/othello_synthetic')
    args = p.parse_args()

    print(f"Loading games...")
    games = load_games(args.data_dir, args.num_pickle_files)
    games = games[:args.num_games]
    print(f"  {len(games)} games loaded")

    for k in args.positions:
        key_to_boards = analyze_at_k(games, k)
        summarize(k, key_to_boards)


if __name__ == '__main__':
    main()
