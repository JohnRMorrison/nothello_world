"""
Compute game statistics for variant and corruption games.

Reads games.pickle and legal_moves.pickle from each condition and computes:
  - avg_branching_factor: mean number of legal moves per position
  - avg_game_length: mean number of moves per game
  - move_entropy: entropy of the move frequency distribution (bits)
  - random_top1_acc: expected top-1 accuracy of a random predictor (1/branching)

Usage:
  python compute_game_statistics.py --batch \
      --variant-base experiments/variants/games_2m \
      --corruption-base experiments/corruption_v2/games_2m \
      --output-dir experiments/divergence/game_stats

  python compute_game_statistics.py \
      --games-dir experiments/variants/games_2m/no_diagonal_flips \
      --label no_diagonal_flips
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

VARIANTS = [
    "no_same_quadrant",
    "no_diagonal_flips",
    "no_row_flips",
    "locked_flips",
    "max_three_flips",
    "self_flanking",
    "delayed_flips",
    "skip_empty_flips",
    "capture_any",
    "adjacent_legal",
]

CORRUPTION_ALPHAS = [
    "alpha000", "alpha001", "alpha002", "alpha005",
    "alpha010", "alpha020", "alpha030", "alpha050",
    "alpha070", "alpha100",
]


def compute_stats(games, legal_moves, max_games=100000, seed=42):
    """Compute game statistics from games and legal_moves lists."""
    rng = np.random.RandomState(seed)
    n = min(len(games), max_games)
    indices = rng.choice(len(games), size=n, replace=False)

    branching_factors = []
    game_lengths = []
    move_counts = np.zeros(64, dtype=np.int64)

    for gi in indices:
        game = games[gi]
        legal = legal_moves[gi]
        game_lengths.append(len(game))

        for t, move in enumerate(game):
            if t < len(legal):
                branching_factors.append(len(legal[t]))
            move_counts[move] += 1

    # Move entropy
    probs = move_counts / move_counts.sum()
    probs = probs[probs > 0]
    move_entropy = -np.sum(probs * np.log2(probs))

    bf = np.array(branching_factors)
    gl = np.array(game_lengths)

    return {
        'avg_branching_factor': float(bf.mean()),
        'std_branching_factor': float(bf.std()),
        'median_branching_factor': float(np.median(bf)),
        'avg_game_length': float(gl.mean()),
        'std_game_length': float(gl.std()),
        'move_entropy': float(move_entropy),
        'random_top1_acc': float((1.0 / bf).mean()),
        'n_games': n,
        'n_positions': len(branching_factors),
    }


def load_data(games_dir):
    """Load games and legal_moves from a directory."""
    games_path = os.path.join(games_dir, "games.pickle")
    legal_path = os.path.join(games_dir, "legal_moves.pickle")
    if not os.path.exists(games_path) or not os.path.exists(legal_path):
        return None, None
    with open(games_path, 'rb') as f:
        games = pickle.load(f)
    with open(legal_path, 'rb') as f:
        legal_moves = pickle.load(f)
    return games, legal_moves


def main():
    parser = argparse.ArgumentParser(description="Compute game statistics")
    parser.add_argument("--games-dir", type=str,
                        help="Single games directory")
    parser.add_argument("--label", type=str)
    parser.add_argument("--variant-base", type=str,
                        default="experiments/variants/games_2m")
    parser.add_argument("--corruption-base", type=str,
                        default="experiments/corruption_v2/games_2m")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/divergence/game_stats")
    parser.add_argument("--max-games", type=int, default=100000)
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    if args.batch:
        conditions = []
        # Standard Othello (alpha000)
        conditions.append(("standard", os.path.join(args.corruption_base, "alpha000")))
        # Variants
        for v in VARIANTS:
            conditions.append((v, os.path.join(args.variant_base, v)))
        # Corruption
        for a in CORRUPTION_ALPHAS:
            conditions.append((a, os.path.join(args.corruption_base, a)))
    else:
        label = args.label or os.path.basename(args.games_dir)
        conditions = [(label, args.games_dir)]

    for label, games_dir in conditions:
        games, legal_moves = load_data(games_dir)
        if games is None:
            print(f"Skipping {label}: data not found at {games_dir}")
            continue

        print(f"{label}...", flush=True, end=" ")
        stats = compute_stats(games, legal_moves, max_games=args.max_games)
        stats['label'] = label
        results[label] = stats
        print(f"BF={stats['avg_branching_factor']:.1f}, "
              f"len={stats['avg_game_length']:.1f}, "
              f"entropy={stats['move_entropy']:.2f}, "
              f"random_acc={stats['random_top1_acc']:.3f}")

        out_path = os.path.join(args.output_dir, f"{label}.json")
        with open(out_path, 'w') as f:
            json.dump(stats, f, indent=2)

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Condition':<25s} {'Branching':>10s} {'Game Len':>9s} "
          f"{'Entropy':>8s} {'Random Acc':>11s}")
    print(f"{'-'*25} {'-'*10} {'-'*9} {'-'*8} {'-'*11}")
    for label in results:
        m = results[label]
        print(f"{label:<25s} {m['avg_branching_factor']:>9.1f} "
              f"{m['avg_game_length']:>8.1f} "
              f"{m['move_entropy']:>7.2f} "
              f"{m['random_top1_acc']:>10.3f}")

    # Save combined
    out_path = os.path.join(args.output_dir, "all_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
