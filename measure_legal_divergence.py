"""
Measure legal move divergence between experimental conditions and standard Othello.

Two directions:
  1. "on_experimental": Replay experimental (variant/corruption) games, compute
     standard Othello legal moves at each position, compare to saved experimental
     legal moves.
  2. "on_standard": Replay standard Othello games, compute variant/corruption
     legal moves at each position, compare to standard legal moves.

Metrics:
  - newly_illegal: fraction of standard-legal moves removed by the condition
  - newly_legal:   fraction of condition-legal moves that aren't standard-legal
  - jaccard_dist:  1 - |intersection| / |union|  (overall divergence)

Usage:
  # Batch mode (all conditions, both directions):
  python measure_legal_divergence.py --batch --output-dir experiments/divergence/

  # Single condition:
  python measure_legal_divergence.py \\
      --games-dir experiments/corruption_v2/games_2m/alpha050 \\
      --condition-type corruption --condition-label alpha050
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
from generate_variant_games import (
    VariantBoard, _flips_vec, _self_flank_vec, _flips_locked_vec,
    DIR_MASK_ALL, DIR_MASK_NO_DIAG, DIR_MASK_NO_ROW, QUADRANTS,
)


def place_piece_no_flip(board, board_pos):
    """Place a piece on the board without flipping (for corruption replay)."""
    r, c = board_pos // 8, board_pos % 8
    board.state[r, c] = board.next_hand_color
    board.next_hand_color *= -1


def compute_standard_legal(flat, color):
    """Compute standard Othello legal moves from flat board state."""
    flat = flat.astype(np.int8)
    n_flips, _ = _flips_vec(flat, color, DIR_MASK_ALL)
    empty = (flat == 0)
    valid = empty & (n_flips > 0)
    regular = np.where(valid)[0].tolist()
    if regular:
        return regular
    # Forfeit
    opp_flips, _ = _flips_vec(flat, -color, DIR_MASK_ALL)
    forfeit = empty & (opp_flips > 0)
    return np.where(forfeit)[0].tolist()


def compute_variant_legal(flat, color, variant_name, last_move=None,
                          flip_count_flat=None):
    """Compute variant legal moves from flat board state.

    This applies variant legality rules to an arbitrary board state,
    without needing a VariantBoard object. Used for direction 2
    (evaluating variant rules on standard Othello positions).
    """
    flat = flat.astype(np.int8)
    empty = (flat == 0)

    # Direction mask
    if variant_name == "no_diagonal_flips":
        dir_mask = DIR_MASK_NO_DIAG
    elif variant_name == "no_row_flips":
        dir_mask = DIR_MASK_NO_ROW
    else:
        dir_mask = DIR_MASK_ALL

    # Flip counts
    if variant_name == "locked_flips" and flip_count_flat is not None:
        n_flips = _flips_locked_vec(flat, color, dir_mask, flip_count_flat)
    else:
        n_flips, _ = _flips_vec(flat, color, dir_mask)

    # Variant-specific filters
    if variant_name == "no_same_quadrant" and last_move is not None:
        last_q = QUADRANTS[last_move]
        empty = empty & (QUADRANTS != last_q)

    if variant_name == "self_flanking":
        has_sf = _self_flank_vec(flat, color, dir_mask)
        empty = empty & ~has_sf

    if variant_name == "max_three_flips":
        valid_mask = empty & (n_flips >= 1) & (n_flips <= 3)
        return np.where(valid_mask)[0].tolist()

    # Regular moves
    valid_mask = empty & (n_flips > 0)
    regular = np.where(valid_mask)[0].tolist()
    if regular:
        return regular

    # Forfeit
    if variant_name == "locked_flips" and flip_count_flat is not None:
        opp_flips = _flips_locked_vec(flat, -color, dir_mask, flip_count_flat)
    else:
        opp_flips, _ = _flips_vec(flat, -color, dir_mask)
    forfeit_mask = empty & (opp_flips > 0)
    return np.where(forfeit_mask)[0].tolist()


def _compare_sets(std_set, exp_set):
    """Compute divergence metrics between two legal move sets."""
    newly_illegal = 0.0
    newly_legal = 0.0
    jaccard = 0.0

    if len(std_set) > 0:
        newly_illegal = len(std_set - exp_set) / len(std_set)
    if len(exp_set) > 0:
        newly_legal = len(exp_set - std_set) / len(exp_set)
    union = std_set | exp_set
    if len(union) > 0:
        jaccard = 1 - len(std_set & exp_set) / len(union)

    return newly_illegal, newly_legal, jaccard


# ---------------------------------------------------------------------------
# Direction 1: Replay experimental games, compare to standard legal moves
# ---------------------------------------------------------------------------

def measure_on_experimental(games, legal_moves, condition_type, variant_name=None,
                            max_games=100000, seed=42):
    """Replay experimental games, compute standard legal moves, compare."""
    rng = np.random.RandomState(seed)
    n = min(len(games), max_games)
    indices = rng.choice(len(games), size=n, replace=False)

    newly_illegal_sum = 0.0
    newly_legal_sum = 0.0
    jaccard_sum = 0.0
    count = 0
    t0 = time.time()

    for idx_i, gi in enumerate(indices):
        game = games[gi]
        legal = legal_moves[gi]

        if condition_type == "corruption":
            board = OthelloBoardState()
        else:
            board = VariantBoard(variant_name)

        for t, move in enumerate(game):
            if t >= len(legal):
                break

            flat = board.state.flatten()
            color = board.next_hand_color
            std_legal = compute_standard_legal(flat, color)

            exp_set = set(legal[t])
            std_set = set(std_legal)

            ni, nl, jd = _compare_sets(std_set, exp_set)
            newly_illegal_sum += ni
            newly_legal_sum += nl
            jaccard_sum += jd
            count += 1

            # Advance board
            if condition_type == "corruption":
                if board.tentative_move(move) != 0:
                    board.update([move])
                else:
                    place_piece_no_flip(board, move)
            else:
                board.make_move(move)

        if (idx_i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  {idx_i+1}/{n} games, {count} positions, "
                  f"elapsed={elapsed:.0f}s", flush=True)

    return {
        'newly_illegal': newly_illegal_sum / max(count, 1),
        'newly_legal': newly_legal_sum / max(count, 1),
        'jaccard_dist': jaccard_sum / max(count, 1),
        'n_games': n,
        'n_positions': count,
    }


# ---------------------------------------------------------------------------
# Direction 2: Replay standard Othello games, compare to variant legal moves
# ---------------------------------------------------------------------------

def measure_on_standard(std_games, variant_name, max_games=100000, seed=42):
    """Replay standard Othello games, compute variant legal moves, compare.

    For locked_flips, we track flip counts during standard play.
    For delayed_flips, legality rules are the same as standard (the variant
    only changes when flips happen), so divergence comes only from board
    evolution — on standard boards, delayed_flips shows ~0% divergence.
    """
    rng = np.random.RandomState(seed)
    n = min(len(std_games), max_games)
    indices = rng.choice(len(std_games), size=n, replace=False)

    newly_illegal_sum = 0.0
    newly_legal_sum = 0.0
    jaccard_sum = 0.0
    count = 0
    t0 = time.time()

    for idx_i, gi in enumerate(indices):
        game = std_games[gi]
        board = OthelloBoardState()

        # Track flip counts for locked_flips variant
        flip_counts = np.zeros(64, dtype=np.int8) if variant_name == "locked_flips" else None

        for t, move in enumerate(game):
            flat = board.state.flatten().astype(np.int8)
            color = board.next_hand_color

            # Standard legal moves
            std_legal = compute_standard_legal(flat, color)

            # Variant legal moves on same board state
            last_move = game[t - 1] if t > 0 else None
            var_legal = compute_variant_legal(
                flat, color, variant_name,
                last_move=last_move,
                flip_count_flat=flip_counts)

            std_set = set(std_legal)
            var_set = set(var_legal)

            ni, nl, jd = _compare_sets(std_set, var_set)
            newly_illegal_sum += ni
            newly_legal_sum += nl
            jaccard_sum += jd
            count += 1

            # Track flips for locked_flips before advancing
            try:
                if flip_counts is not None:
                    # Count which pieces will flip during this standard move
                    pre_state = board.state.flatten().copy()
                    board.update([move])
                    post_state = board.state.flatten()
                    flipped = (pre_state != 0) & (post_state != 0) & (pre_state != post_state)
                    flip_counts[flipped] += 1
                else:
                    board.update([move])
            except AssertionError:
                # Game ended (no legal moves for either player)
                break

        if (idx_i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  {idx_i+1}/{n} games, {count} positions, "
                  f"elapsed={elapsed:.0f}s", flush=True)

    return {
        'newly_illegal': newly_illegal_sum / max(count, 1),
        'newly_legal': newly_legal_sum / max(count, 1),
        'jaccard_dist': jaccard_sum / max(count, 1),
        'n_games': n,
        'n_positions': count,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_condition(games_dir):
    """Load games and legal_moves from a condition directory."""
    games_path = os.path.join(games_dir, "games.pickle")
    legal_path = os.path.join(games_dir, "legal_moves.pickle")

    with open(games_path, 'rb') as f:
        games = pickle.load(f)
    with open(legal_path, 'rb') as f:
        legal_moves = pickle.load(f)

    return games, legal_moves


def load_standard_games(games_dir):
    """Load standard Othello games (just games.pickle, no legal_moves needed)."""
    games_path = os.path.join(games_dir, "games.pickle")
    with open(games_path, 'rb') as f:
        games = pickle.load(f)
    return games


CORRUPTION_ALPHAS = [
    ("alpha000", 0.0),
    ("alpha001", 0.01),
    ("alpha002", 0.02),
    ("alpha005", 0.05),
    ("alpha010", 0.10),
    ("alpha020", 0.20),
    ("alpha030", 0.30),
    ("alpha050", 0.50),
    ("alpha070", 0.70),
    ("alpha100", 1.00),
]

VARIANTS = [
    "no_same_quadrant",
    "no_diagonal_flips",
    "no_row_flips",
    "locked_flips",
    "max_three_flips",
    "self_flanking",
    "delayed_flips",
]


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_batch(corruption_base, variant_base, std_games_dir, output_dir,
              max_games):
    """Run divergence measurement for all conditions, both directions."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # --- Direction 1: On experimental games ---
    print("\n" + "=" * 70)
    print("DIRECTION 1: On experimental games (variant/corruption board states)")
    print("=" * 70)

    # Corruption conditions
    for label, alpha in CORRUPTION_ALPHAS:
        games_dir = os.path.join(corruption_base, label)
        if not os.path.isdir(games_dir):
            print(f"Skipping {label}: {games_dir} not found")
            continue

        print(f"\nCorruption: {label} (alpha={alpha})", flush=True)
        games, legal_moves = load_condition(games_dir)
        metrics = measure_on_experimental(games, legal_moves, "corruption",
                                          max_games=max_games)
        metrics['alpha'] = alpha
        metrics['condition_type'] = 'corruption'
        metrics['label'] = label
        metrics['direction'] = 'on_experimental'
        results[f"{label}_on_exp"] = metrics
        print(f"  newly_illegal={metrics['newly_illegal']:.4f}, "
              f"newly_legal={metrics['newly_legal']:.4f}, "
              f"jaccard={metrics['jaccard_dist']:.4f}")

    # Variant conditions
    for variant in VARIANTS:
        games_dir = os.path.join(variant_base, variant)
        if not os.path.isdir(games_dir):
            print(f"Skipping {variant}: {games_dir} not found")
            continue

        print(f"\nVariant: {variant}", flush=True)
        games, legal_moves = load_condition(games_dir)
        metrics = measure_on_experimental(games, legal_moves, "variant",
                                          variant_name=variant,
                                          max_games=max_games)
        metrics['condition_type'] = 'variant'
        metrics['label'] = variant
        metrics['direction'] = 'on_experimental'
        results[f"{variant}_on_exp"] = metrics
        print(f"  newly_illegal={metrics['newly_illegal']:.4f}, "
              f"newly_legal={metrics['newly_legal']:.4f}, "
              f"jaccard={metrics['jaccard_dist']:.4f}")

    # --- Direction 2: On standard Othello games ---
    print("\n" + "=" * 70)
    print("DIRECTION 2: On standard Othello games (standard board states)")
    print("=" * 70)

    # Load standard games once (use alpha000 = uncorrupted games)
    if os.path.isdir(std_games_dir):
        print(f"\nLoading standard games from {std_games_dir}...", flush=True)
        std_games = load_standard_games(std_games_dir)
        print(f"Loaded {len(std_games)} standard games")

        # Corruption: on standard boards, corrupted rules define different
        # legality. We need the corrupted rule patterns to evaluate.
        # For now, skip corruption direction 2 (it requires loading the
        # corrupted patterns, which adds complexity). The corruption
        # direction 1 already captures the key information since corruption
        # games use real Othello board evolution for legal moves.

        # Variant conditions on standard boards
        for variant in VARIANTS:
            print(f"\nVariant on standard: {variant}", flush=True)
            metrics = measure_on_standard(std_games, variant,
                                          max_games=max_games)
            metrics['condition_type'] = 'variant'
            metrics['label'] = variant
            metrics['direction'] = 'on_standard'
            results[f"{variant}_on_std"] = metrics
            print(f"  newly_illegal={metrics['newly_illegal']:.4f}, "
                  f"newly_legal={metrics['newly_legal']:.4f}, "
                  f"jaccard={metrics['jaccard_dist']:.4f}")
    else:
        print(f"\nStandard games not found at {std_games_dir}, skipping direction 2")

    # --- Summary tables ---
    print("\n" + "=" * 70)
    print("SUMMARY: Direction 1 (on experimental games)")
    print("=" * 70)
    print(f"{'Condition':<25s} {'Newly Illegal':>14s} {'Newly Legal':>12s} {'Jaccard':>10s}")
    print(f"{'-'*25} {'-'*14} {'-'*12} {'-'*10}")
    for label, _ in CORRUPTION_ALPHAS:
        key = f"{label}_on_exp"
        if key in results:
            m = results[key]
            print(f"{label:<25s} {m['newly_illegal']:>13.2%} "
                  f"{m['newly_legal']:>11.2%} {m['jaccard_dist']:>9.2%}")
    for variant in VARIANTS:
        key = f"{variant}_on_exp"
        if key in results:
            m = results[key]
            print(f"{variant:<25s} {m['newly_illegal']:>13.2%} "
                  f"{m['newly_legal']:>11.2%} {m['jaccard_dist']:>9.2%}")

    print(f"\n{'='*70}")
    print("SUMMARY: Direction 2 (on standard Othello games)")
    print("=" * 70)
    print(f"{'Condition':<25s} {'Newly Illegal':>14s} {'Newly Legal':>12s} {'Jaccard':>10s}")
    print(f"{'-'*25} {'-'*14} {'-'*12} {'-'*10}")
    for variant in VARIANTS:
        key = f"{variant}_on_std"
        if key in results:
            m = results[key]
            print(f"{variant:<25s} {m['newly_illegal']:>13.2%} "
                  f"{m['newly_legal']:>11.2%} {m['jaccard_dist']:>9.2%}")

    # Save
    out_path = os.path.join(output_dir, "divergence_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Measure legal move divergence")
    parser.add_argument("--games-dir", type=str,
                        help="Directory with games.pickle and legal_moves.pickle")
    parser.add_argument("--condition-type", choices=["corruption", "variant"])
    parser.add_argument("--condition-label", type=str)
    parser.add_argument("--variant-name", type=str,
                        help="Variant name (required for variant conditions)")
    parser.add_argument("--max-games", type=int, default=100000)
    parser.add_argument("--output-dir", type=str, default="experiments/divergence")
    parser.add_argument("--batch", action="store_true",
                        help="Run all conditions in batch mode")
    parser.add_argument("--corruption-base", type=str,
                        default="experiments/corruption_v2/games_2m",
                        help="Base dir for corruption games")
    parser.add_argument("--variant-base", type=str,
                        default="experiments/variants/games_2m",
                        help="Base dir for variant games")
    parser.add_argument("--std-games-dir", type=str,
                        default="experiments/corruption_v2/games_2m/alpha000",
                        help="Dir with standard Othello games (for direction 2)")
    parser.add_argument("--direction", choices=["both", "on_experimental", "on_standard"],
                        default="both",
                        help="Which direction(s) to measure")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.corruption_base, args.variant_base,
                  args.std_games_dir, args.output_dir, args.max_games)
    else:
        if not args.games_dir or not args.condition_type:
            parser.error("--games-dir and --condition-type required in single mode")

        label = args.condition_label or os.path.basename(args.games_dir)
        variant_name = args.variant_name
        if args.condition_type == "variant" and not variant_name:
            variant_name = os.path.basename(args.games_dir)

        os.makedirs(args.output_dir, exist_ok=True)

        # Direction 1
        if args.direction in ("both", "on_experimental"):
            games, legal_moves = load_condition(args.games_dir)
            metrics = measure_on_experimental(
                games, legal_moves, args.condition_type,
                variant_name=variant_name, max_games=args.max_games)
            print(f"\nDirection 1 (on experimental) for {label}:")
            print(f"  Newly illegal: {metrics['newly_illegal']:.4f}")
            print(f"  Newly legal:   {metrics['newly_legal']:.4f}")
            print(f"  Jaccard dist:  {metrics['jaccard_dist']:.4f}")

            metrics['label'] = label
            metrics['condition_type'] = args.condition_type
            metrics['direction'] = 'on_experimental'
            out_path = os.path.join(args.output_dir, f"{label}_on_exp.json")
            with open(out_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved to {out_path}")

        # Direction 2 (variants only)
        if args.direction in ("both", "on_standard") and args.condition_type == "variant":
            std_games = load_standard_games(args.std_games_dir)
            metrics = measure_on_standard(
                std_games, variant_name, max_games=args.max_games)
            print(f"\nDirection 2 (on standard) for {label}:")
            print(f"  Newly illegal: {metrics['newly_illegal']:.4f}")
            print(f"  Newly legal:   {metrics['newly_legal']:.4f}")
            print(f"  Jaccard dist:  {metrics['jaccard_dist']:.4f}")

            metrics['label'] = label
            metrics['condition_type'] = args.condition_type
            metrics['direction'] = 'on_standard'
            out_path = os.path.join(args.output_dir, f"{label}_on_std.json")
            with open(out_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
