"""
Generate Othello games with quadrant restriction:
after a move in quadrant Q, the next player cannot play in Q.

Quadrants: Q0 (rows 0-3, cols 0-3), Q1 (rows 0-3, cols 4-7),
           Q2 (rows 4-7, cols 0-3), Q3 (rows 4-7, cols 4-7).

Games with quadrant-skips are NOT valid standard Othello (the skip changes
whose turn it is without a move being recorded). They are valid under the
quadrant-restricted ruleset and can still be loaded by CharDataset.
"""

import argparse
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


def get_quadrant(move):
    """Return quadrant index (0-3) for a board position (0-63)."""
    row, col = move // 8, move % 8
    return (0 if row < 4 else 2) + (0 if col < 4 else 1)


def generate_game(args):
    """Generate a single quadrant-restricted Othello game.

    Returns a list of moves (ints), or None if game is too short.
    """
    seed, min_length, carry_forward = args
    random.seed(seed)

    board = OthelloBoardState()
    forbidden_quadrant = None
    moves = []
    consecutive_skips = 0

    while True:
        base_valid = board.get_valid_moves()
        if not base_valid:
            break  # standard game over

        # Apply quadrant restriction
        if forbidden_quadrant is not None:
            filtered = [m for m in base_valid if get_quadrant(m) != forbidden_quadrant]
        else:
            filtered = list(base_valid)

        if filtered:
            move = random.choice(filtered)
            board.umpire(move)
            moves.append(move)
            forbidden_quadrant = get_quadrant(move)
            consecutive_skips = 0
        else:
            # All legal moves are in forbidden quadrant -> skip turn
            consecutive_skips += 1
            if consecutive_skips >= 2:
                break  # deadlock
            if not carry_forward:
                forbidden_quadrant = None
            board.next_hand_color *= -1  # skip turn

    if len(moves) < min_length:
        return None
    return moves


def verify_game(game, carry_forward=False):
    """Replay a game using the same quadrant-skip logic as generation.

    Returns (valid, skips, error_msg).
    """
    board = OthelloBoardState()
    forbidden_quadrant = None
    skips = 0
    move_idx = 0

    while move_idx < len(game):
        base_valid = board.get_valid_moves()
        if not base_valid:
            return False, skips, f"No valid moves at step {move_idx}"

        if forbidden_quadrant is not None:
            filtered = [m for m in base_valid if get_quadrant(m) != forbidden_quadrant]
        else:
            filtered = list(base_valid)

        if filtered:
            move = game[move_idx]
            if move not in filtered:
                return False, skips, (
                    f"Move {move} (q={get_quadrant(move)}) not in filtered valid "
                    f"moves at step {move_idx}, forbidden_q={forbidden_quadrant}"
                )
            board.umpire(move)
            forbidden_quadrant = get_quadrant(move)
            move_idx += 1
        else:
            # Skip — mirror generation logic exactly
            skips += 1
            if not carry_forward:
                forbidden_quadrant = None
            board.next_hand_color *= -1

    return True, skips, "OK"


def verify_quadrant_rule(game, carry_forward=False):
    """Check that no two consecutive played moves violate the quadrant restriction.

    A violation: move[i] is in quadrant Q, and move[i+1] is also in Q,
    UNLESS a skip reset the restriction between them.
    Returns (valid, error_msg).
    """
    board = OthelloBoardState()
    forbidden_quadrant = None
    move_idx = 0

    while move_idx < len(game):
        base_valid = board.get_valid_moves()
        if not base_valid:
            break

        if forbidden_quadrant is not None:
            filtered = [m for m in base_valid if get_quadrant(m) != forbidden_quadrant]
        else:
            filtered = list(base_valid)

        if filtered:
            move = game[move_idx]
            if forbidden_quadrant is not None and get_quadrant(move) == forbidden_quadrant:
                return False, f"Quadrant violation at step {move_idx}: q={get_quadrant(move)} == forbidden={forbidden_quadrant}"
            board.umpire(move)
            forbidden_quadrant = get_quadrant(move)
            move_idx += 1
        else:
            if not carry_forward:
                forbidden_quadrant = None
            board.next_hand_color *= -1

    return True, "OK"


def main():
    parser = argparse.ArgumentParser(description="Generate quadrant-restricted Othello games")
    parser.add_argument("--num-games", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=100_000, help="Games per pickle file")
    parser.add_argument("--min-game-length", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="data/quadrant_restricted_synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--carry-forward", action="store_true",
                        help="If set, quadrant restriction persists through skips")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification on generated games")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    num_workers = multiprocessing.cpu_count()
    oversample = int(args.num_games * 1.2)

    task_args = [
        (args.seed + i, args.min_game_length, args.carry_forward)
        for i in range(oversample)
    ]

    all_games = []
    print(f"Generating games with {num_workers} workers...")

    with multiprocessing.Pool(num_workers) as pool:
        for result in tqdm(pool.imap(generate_game, task_args, chunksize=256),
                           total=oversample):
            if result is not None:
                all_games.append(result)
            if len(all_games) >= args.num_games:
                break

    all_games = all_games[:args.num_games]

    # Save in batches
    file_idx = 0
    for start in range(0, len(all_games), args.batch_size):
        batch = all_games[start:start + args.batch_size]
        t_stamp = time.strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(args.output_dir, f"quadrant_{file_idx:03d}_{t_stamp}.pickle")
        with open(fname, "wb") as f:
            pickle.dump(batch, protocol=pickle.HIGHEST_PROTOCOL, file=f)
        print(f"Saved {len(batch)} games to {fname}")
        file_idx += 1

    # Statistics
    lengths = np.array([len(g) for g in all_games])
    print(f"\n--- Statistics ---")
    print(f"Total games: {len(all_games)}")
    print(f"Game length: min={lengths.min()}, max={lengths.max()}, "
          f"mean={lengths.mean():.1f}, std={lengths.std():.1f}")

    # Verification and skip counting
    if args.verify:
        sample_size = min(10000, len(all_games))
        print(f"\nVerifying {sample_size} games...")
        total_skips = 0
        replay_failures = 0
        quadrant_violations = 0

        for game in all_games[:sample_size]:
            valid, skips, msg = verify_game(game, args.carry_forward)
            if not valid:
                replay_failures += 1
                if replay_failures <= 5:
                    print(f"  Replay failure: {msg}")
            total_skips += skips

            qvalid, qmsg = verify_quadrant_rule(game, args.carry_forward)
            if not qvalid:
                quadrant_violations += 1
                if quadrant_violations <= 5:
                    print(f"  Quadrant violation: {qmsg}")

        print(f"Replay failures: {replay_failures}/{sample_size}")
        print(f"Quadrant violations: {quadrant_violations}/{sample_size}")
        print(f"Average quadrant-skips per game: {total_skips / sample_size:.2f}")

        if replay_failures == 0 and quadrant_violations == 0:
            print("All checks passed!")
        else:
            print("ISSUES FOUND")


if __name__ == "__main__":
    main()
