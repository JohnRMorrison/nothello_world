"""
Generate Othello games with heuristic-based restrictions on legal moves.

Normal Othello game dynamics are preserved (flips work identically). The only
change is that certain moves are removed from the legal set when specific
board-state conditions hold. If all legal moves would be removed, the
restriction is waived for that turn.

Usage:
    python generate_restricted_games.py \\
        --config restrictions_aligned.json \\
        --output-dir ../../data/restricted_aligned \\
        --num-games 500000

    python generate_restricted_games.py \\
        --config restrictions_random.json \\
        --output-dir ../../data/restricted_random \\
        --num-games 500000
"""

import argparse
import json
import multiprocessing
import os
import pickle
import random
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from data.othello import OthelloBoardState

from restriction_utils import (
    evaluate_restriction, get_flipped_squares,
)


# ---------------------------------------------------------------------------
# Shared state for worker processes (set once via initializer)
# ---------------------------------------------------------------------------

_RESTRICTIONS = None
_MIN_LENGTH = None


def _worker_init(restrictions, min_length):
    global _RESTRICTIONS, _MIN_LENGTH
    _RESTRICTIONS = restrictions
    _MIN_LENGTH = min_length


# ---------------------------------------------------------------------------
# Game generation
# ---------------------------------------------------------------------------

def generate_game(seed):
    """Generate one game under the restriction rules.

    Returns (moves_list, stats_dict) or None if game is too short.
    """
    rng = random.Random(seed)
    board = OthelloBoardState()
    moves = []
    last_flipped = set()
    last_move = -1
    restrictions_fired = 0
    moves_removed = 0
    consecutive_skips = 0

    while True:
        standard_legal = board.get_valid_moves()
        if not standard_legal:
            break

        # Apply restrictions if we have a previous move to condition on
        if last_move >= 0:
            forbidden = set()
            for r in _RESTRICTIONS:
                if evaluate_restriction(r, board, last_move, last_flipped):
                    forbidden.add(r["forbidden_position"])
                    restrictions_fired += 1

            filtered = [m for m in standard_legal if m not in forbidden]
            actual_removed = len(standard_legal) - len(filtered)
            moves_removed += actual_removed

            if filtered:
                legal = filtered
            else:
                legal = standard_legal  # fallback: waive restrictions
        else:
            legal = standard_legal

        if not legal:
            # No moves at all — try skip
            consecutive_skips += 1
            if consecutive_skips >= 2:
                break
            board.next_hand_color *= -1
            continue

        move = rng.choice(legal)
        state_before = board.state.copy()
        board.umpire(move)
        last_flipped = get_flipped_squares(state_before, board.state, move)
        last_move = move
        moves.append(move)
        consecutive_skips = 0

    if len(moves) < _MIN_LENGTH:
        return None
    return moves, {
        "length": len(moves),
        "restrictions_fired": restrictions_fired,
        "moves_removed": moves_removed,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_game(game, restrictions):
    """Replay a game and check that no played move violates restrictions."""
    board = OthelloBoardState()
    last_flipped = set()
    last_move = -1

    for mi, move in enumerate(game):
        standard_legal = board.get_valid_moves()
        if not standard_legal:
            return False, f"No legal moves at step {mi}"

        if move not in standard_legal:
            return False, f"Move {move} not standard-legal at step {mi}"

        # Check restrictions
        if last_move >= 0:
            forbidden = set()
            for r in restrictions:
                if evaluate_restriction(r, board, last_move, last_flipped):
                    forbidden.add(r["forbidden_position"])

            filtered = [m for m in standard_legal if m not in forbidden]
            if filtered and move not in filtered:
                return False, (
                    f"Move {move} is restricted at step {mi} "
                    f"(forbidden={forbidden})"
                )

        state_before = board.state.copy()
        board.umpire(move)
        last_flipped = get_flipped_squares(state_before, board.state, move)
        last_move = move

    return True, "OK"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Othello games with heuristic-based restrictions")
    parser.add_argument("--config", type=str, required=True,
                        help="Restriction config JSON from build_restriction_configs.py")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output pickle files")
    parser.add_argument("--num-games", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=100_000,
                        help="Games per pickle file")
    parser.add_argument("--min-game-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true",
                        help="Run verification on a sample of generated games")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: CPU count)")
    args = parser.parse_args()

    # --- Load restrictions ---
    with open(args.config) as f:
        config = json.load(f)
    restrictions = config["restrictions"]
    print(f"Loaded {len(restrictions)} restrictions from {args.config}")
    print(f"  Description: {config.get('description', 'N/A')}")

    os.makedirs(args.output_dir, exist_ok=True)
    num_workers = args.workers or multiprocessing.cpu_count()
    oversample = int(args.num_games * 1.3)

    seeds = list(range(args.seed, args.seed + oversample))

    # --- Generate games ---
    all_games = []
    total_restrictions_fired = 0
    total_moves_removed = 0
    total_positions = 0

    print(f"Generating up to {args.num_games} games with {num_workers} workers...")
    with multiprocessing.Pool(
        num_workers,
        initializer=_worker_init,
        initargs=(restrictions, args.min_game_length),
    ) as pool:
        for result in tqdm(
            pool.imap(generate_game, seeds, chunksize=256),
            total=oversample,
        ):
            if result is not None:
                game, stats = result
                all_games.append(game)
                total_restrictions_fired += stats["restrictions_fired"]
                total_moves_removed += stats["moves_removed"]
                total_positions += stats["length"]
            if len(all_games) >= args.num_games:
                break

    all_games = all_games[: args.num_games]

    # --- Save in batches ---
    file_idx = 0
    for start in range(0, len(all_games), args.batch_size):
        batch = all_games[start : start + args.batch_size]
        t_stamp = time.strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(
            args.output_dir, f"restricted_{file_idx:03d}_{t_stamp}.pickle"
        )
        with open(fname, "wb") as f:
            pickle.dump(batch, protocol=pickle.HIGHEST_PROTOCOL, file=f)
        print(f"Saved {len(batch)} games to {fname}")
        file_idx += 1

    # --- Statistics ---
    lengths = np.array([len(g) for g in all_games])
    print(f"\n--- Statistics ---")
    print(f"Total games: {len(all_games)}")
    print(f"Game length: min={lengths.min()}, max={lengths.max()}, "
          f"mean={lengths.mean():.1f}, std={lengths.std():.1f}")
    if total_positions > 0:
        print(f"Restrictions fired: {total_restrictions_fired} total, "
              f"{total_restrictions_fired / total_positions:.3f} per position")
        print(f"Moves removed: {total_moves_removed} total, "
              f"{total_moves_removed / total_positions:.3f} per position")

    # --- Save metadata ---
    meta = {
        "config_file": args.config,
        "num_games": len(all_games),
        "seed": args.seed,
        "min_game_length": args.min_game_length,
        "game_length_mean": float(lengths.mean()),
        "game_length_std": float(lengths.std()),
        "restrictions_per_position": total_restrictions_fired / max(total_positions, 1),
        "moves_removed_per_position": total_moves_removed / max(total_positions, 1),
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    # --- Verification ---
    if args.verify:
        sample_size = min(5000, len(all_games))
        print(f"\nVerifying {sample_size} games...")
        failures = 0
        for game in all_games[:sample_size]:
            valid, msg = verify_game(game, restrictions)
            if not valid:
                failures += 1
                if failures <= 5:
                    print(f"  FAIL: {msg}")
        print(f"Verification: {failures}/{sample_size} failures")
        if failures == 0:
            print("All checks passed!")


if __name__ == "__main__":
    main()
