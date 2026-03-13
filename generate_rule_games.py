"""
Generate Othello games using corrupted flanking rules (no neural network).

For fraction alpha of the 960 flanking patterns, one cell reference
(from opponents or terminal) is replaced with a random cell.
The corrupted rules are consistent — same input always gives same output.

Usage:
  python generate_rule_games.py --alpha 0.1 --num-games 2000000 --output-dir experiments/corruption_v2/games/alpha010
"""

import argparse
import os
import pickle
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data.othello import OthelloBoardState
from hand_crafted_flanking import (
    enumerate_flanking_patterns, VALID_MOVES, MOVE_TO_IDX, N_MOVES, CENTER_CELLS
)


def corrupt_patterns(patterns, alpha, rng):
    """Corrupt fraction alpha of patterns by replacing one cell reference.

    For each corrupted pattern, pick one cell from opponents + terminal
    and replace it with a random cell (0-63).
    """
    patterns = [p.copy() for p in patterns]
    n = len(patterns)
    n_corrupt = int(alpha * n)
    to_corrupt = rng.choice(n, size=n_corrupt, replace=False)

    for j in to_corrupt:
        p = patterns[j]
        p['opponents'] = list(p['opponents'])  # ensure mutable copy

        # Collect indices of replaceable cells: opponents + terminal
        n_opp = len(p['opponents'])
        n_cells = n_opp + 1  # opponents + terminal

        pick = rng.randint(n_cells)
        new_cell = rng.randint(64)

        if pick < n_opp:
            p['opponents'][pick] = new_cell
        else:
            p['terminal'] = new_cell

    return patterns


def evaluate_rules(patterns, board_state, is_black_turn):
    """Return set of cells predicted legal by the (possibly corrupted) rules."""
    flat = board_state.flatten()
    my_val = 1 if is_black_turn else -1
    opp_val = -my_val
    legal = set()

    for pat in patterns:
        target = pat['target']
        # Target must be empty
        if flat[target] != 0:
            continue
        # All opponent cells must have opponent piece
        if not all(flat[o] == opp_val for o in pat['opponents']):
            continue
        # Terminal must have friendly piece
        if flat[pat['terminal']] != my_val:
            continue
        legal.add(target)

    return legal


def place_piece_no_flip(board, board_pos):
    """Place a piece on the board without flipping. Used for illegal moves."""
    r, c = board_pos // 8, board_pos % 8
    board.state[r, c] = board.next_hand_color
    board.next_hand_color *= -1


def generate_single_game(patterns, rng):
    """Generate a single game using (corrupted) rules on exact board state."""
    board = OthelloBoardState()
    moves = []

    for turn in range(60):
        is_black_turn = (board.next_hand_color == 1)
        predicted_legal = evaluate_rules(patterns, board.state, is_black_turn)

        if not predicted_legal:
            # No moves predicted — sample from empty cells
            flat = board.state.flatten()
            empty_cells = [i for i in range(64) if flat[i] == 0 and i not in CENTER_CELLS]
            if not empty_cells:
                break
            board_pos = empty_cells[rng.randint(len(empty_cells))]
        else:
            predicted_legal = sorted(predicted_legal)
            board_pos = predicted_legal[rng.randint(len(predicted_legal))]

        # Try legal move (with flips); if illegal, place without flipping
        try:
            board.update([board_pos])
        except:
            place_piece_no_flip(board, board_pos)

        moves.append(board_pos)

    return moves


def generate_games(patterns, num_games, rng, chunk_size=100000):
    """Generate games using (corrupted) rules."""
    all_games = []
    for chunk_start in range(0, num_games, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_games)
        for game_idx in range(chunk_start, chunk_end):
            game = generate_single_game(patterns, rng)
            all_games.append(game)
        print(f"  Generated {len(all_games)}/{num_games} games...")
    return all_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--num-games", type=int, default=2000000)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Alpha: {args.alpha}")
    print(f"Num games: {args.num_games}")
    print(f"Seed: {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build and corrupt patterns
    patterns = enumerate_flanking_patterns()
    print(f"Flanking patterns: {len(patterns)}")

    rng = np.random.RandomState(args.seed)
    if args.alpha > 0:
        patterns = corrupt_patterns(patterns, args.alpha, rng)
        n_corrupt = int(args.alpha * len(patterns))
        print(f"Corrupted {n_corrupt}/{len(patterns)} patterns")

    # Generate games
    game_rng = np.random.RandomState(args.seed + 1000)
    print(f"Generating {args.num_games} games...")
    games = generate_games(patterns, args.num_games, game_rng)

    # Report stats
    lengths = [len(g) for g in games]
    print(f"\nGame stats:")
    print(f"  Total games: {len(games)}")
    print(f"  Mean length: {np.mean(lengths):.1f}")
    print(f"  Min length: {np.min(lengths)}")
    print(f"  Max length: {np.max(lengths)}")
    print(f"  Games with 60 moves: {sum(1 for l in lengths if l == 60)}")

    # Save
    out_path = os.path.join(args.output_dir, "games.pickle")
    with open(out_path, 'wb') as f:
        pickle.dump(games, f)
    print(f"Saved to {out_path}")

    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.txt")
    with open(meta_path, 'w') as f:
        f.write(f"alpha={args.alpha}\n")
        f.write(f"num_games={len(games)}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"mean_length={np.mean(lengths):.1f}\n")
        f.write(f"num_patterns={len(patterns)}\n")
        n_corrupt = int(args.alpha * 960)
        f.write(f"num_corrupted={n_corrupt}\n")
    print(f"Saved metadata to {meta_path}")


if __name__ == '__main__':
    main()
