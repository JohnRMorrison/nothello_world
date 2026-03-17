"""
Measure trajectory divergence between variant and standard Othello.

Unlike rule divergence (which compares legal move sets on the same board state),
trajectory divergence plays the SAME moves in both standard and variant games,
then compares the resulting legal move sets at each turn. This captures divergence
caused by different flip mechanics (e.g., self_flanking, delayed_flips) that
don't change the legality rule but produce different board states.

Metrics (averaged over positions):
  - jaccard_dist:    1 - |intersection| / |union|
  - newly_illegal:   fraction of standard-legal moves not legal in variant
  - newly_legal:     fraction of variant-legal moves not legal in standard
  - board_diff_frac: fraction of non-empty cells that differ between boards

Usage:
  # All variants:
  python measure_trajectory_divergence.py --batch \
      --std-games-dir experiments/corruption_v2/games_2m/alpha000 \
      --output-dir experiments/divergence/trajectory

  # Single variant:
  python measure_trajectory_divergence.py \
      --variant no_diagonal_flips \
      --std-games-dir experiments/corruption_v2/games_2m/alpha000 \
      --output-dir experiments/divergence/trajectory
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data.othello import OthelloBoardState
from generate_variant_games import VariantBoard


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


def measure_trajectory_divergence(std_games, variant_name, max_games=100000,
                                  seed=42):
    """Play same moves in standard and variant boards, compare legal moves.

    For each standard game, replay the moves on both a standard and variant
    board. At each turn, compare the legal move sets. If the chosen move is
    illegal in the variant, the game stops (trajectories have fully diverged).
    """
    rng = np.random.RandomState(seed)
    n = min(len(std_games), max_games)
    indices = rng.choice(len(std_games), size=n, replace=False)

    jaccard_sum = 0.0
    newly_illegal_sum = 0.0
    newly_legal_sum = 0.0
    board_diff_sum = 0.0
    count = 0
    early_stops = 0
    t0 = time.time()

    for idx_i, gi in enumerate(indices):
        game = std_games[gi]
        std_board = OthelloBoardState()
        var_board = VariantBoard(variant_name)

        for t, move in enumerate(game):
            # Get legal moves from both boards
            std_legal = set(std_board.get_valid_moves())
            var_legal = set(var_board.get_valid_moves())

            # Compare
            union = std_legal | var_legal
            if len(union) > 0:
                jd = 1 - len(std_legal & var_legal) / len(union)
            else:
                jd = 0.0
            ni = len(std_legal - var_legal) / max(len(std_legal), 1)
            nl = len(var_legal - std_legal) / max(len(var_legal), 1)

            # Board state difference
            std_flat = std_board.state.flatten()
            var_flat = var_board.state.ravel().flatten()
            occupied = (std_flat != 0) | (var_flat != 0)
            n_occupied = occupied.sum()
            if n_occupied > 0:
                bd = ((std_flat != var_flat) & occupied).sum() / n_occupied
            else:
                bd = 0.0

            jaccard_sum += jd
            newly_illegal_sum += ni
            newly_legal_sum += nl
            board_diff_sum += bd
            count += 1

            # If the standard game's move is illegal in the variant, stop
            if move not in var_legal:
                early_stops += 1
                break

            # Advance both boards
            try:
                std_board.update([move])
            except (AssertionError, AssertionError):
                break
            var_board.make_move(move)

        if (idx_i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  {idx_i+1}/{n} games, {count} positions, "
                  f"elapsed={elapsed:.0f}s", flush=True)

    return {
        'jaccard_dist': jaccard_sum / max(count, 1),
        'newly_illegal': newly_illegal_sum / max(count, 1),
        'newly_legal': newly_legal_sum / max(count, 1),
        'board_diff_frac': board_diff_sum / max(count, 1),
        'n_games': n,
        'n_positions': count,
        'early_stop_frac': early_stops / max(n, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure trajectory divergence between variant and standard Othello")
    parser.add_argument("--variant", type=str,
                        help="Single variant to measure")
    parser.add_argument("--std-games-dir", type=str,
                        default="experiments/corruption_v2/games_2m/alpha000",
                        help="Dir with standard Othello games.pickle")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/divergence/trajectory")
    parser.add_argument("--max-games", type=int, default=100000)
    parser.add_argument("--batch", action="store_true",
                        help="Run all variants")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load standard games
    print(f"Loading standard games from {args.std_games_dir}...", flush=True)
    games_path = os.path.join(args.std_games_dir, "games.pickle")
    with open(games_path, 'rb') as f:
        std_games = pickle.load(f)
    print(f"Loaded {len(std_games)} standard games")

    variants = VARIANTS if args.batch else [args.variant]

    results = {}
    for variant in variants:
        print(f"\nVariant: {variant}", flush=True)
        metrics = measure_trajectory_divergence(
            std_games, variant, max_games=args.max_games)
        metrics['variant'] = variant
        results[variant] = metrics
        print(f"  jaccard={metrics['jaccard_dist']:.4f}, "
              f"board_diff={metrics['board_diff_frac']:.4f}, "
              f"early_stop={metrics['early_stop_frac']:.2%}")

        # Save per-variant
        out_path = os.path.join(args.output_dir, f"{variant}.json")
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Variant':<25s} {'Jaccard':>8s} {'Board Diff':>11s} {'Early Stop':>11s}")
    print(f"{'-'*25} {'-'*8} {'-'*11} {'-'*11}")
    for variant in variants:
        m = results[variant]
        print(f"{variant:<25s} {m['jaccard_dist']:>7.4f} "
              f"{m['board_diff_frac']:>10.4f} "
              f"{m['early_stop_frac']:>10.2%}")

    # Save combined
    out_path = os.path.join(args.output_dir, "all_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output_dir}/")


if __name__ == "__main__":
    main()
