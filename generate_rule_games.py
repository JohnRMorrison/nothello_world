"""
Generate Othello games using corrupted flanking rules (no neural network).

For fraction alpha of the 960 flanking patterns, one cell reference
(from opponents or terminal) is replaced with a random cell.
The corrupted rules are consistent — same input always gives same output.

Uses numpy vectorization to evaluate all patterns simultaneously.

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


def precompute_pattern_arrays(patterns):
    """Precompute numpy arrays for vectorized pattern evaluation."""
    n = len(patterns)
    max_opp = max(len(p['opponents']) for p in patterns)

    targets = np.array([p['target'] for p in patterns], dtype=np.int32)
    terminals = np.array([p['terminal'] for p in patterns], dtype=np.int32)
    opp_lens = np.array([len(p['opponents']) for p in patterns], dtype=np.int32)

    # Pad opponent arrays — use 0 as filler (masked out later)
    opp_cells = np.zeros((n, max_opp), dtype=np.int32)
    for i, p in enumerate(patterns):
        for j, o in enumerate(p['opponents']):
            opp_cells[i, j] = o

    # Precompute mask: which opponent slots are real
    opp_mask = np.arange(max_opp)[None, :] < opp_lens[:, None]  # (n, max_opp)

    return targets, terminals, opp_cells, opp_mask


def evaluate_rules_vec(flat, is_black_turn, targets, terminals, opp_cells, opp_mask):
    """Vectorized: evaluate all patterns at once, return list of legal cells."""
    my_val = 1 if is_black_turn else -1
    opp_val = -my_val

    # Check target is empty: (n_patterns,)
    target_empty = (flat[targets] == 0)

    # Check terminal has friendly piece: (n_patterns,)
    terminal_friendly = (flat[terminals] == my_val)

    # Check opponent cells: (n_patterns, max_opp)
    opp_vals = flat[opp_cells]
    opp_match = (opp_vals == opp_val) | ~opp_mask  # masked slots always pass
    opp_all = opp_match.all(axis=1)  # (n_patterns,)

    # Pattern fires if all conditions met
    fires = target_empty & opp_all & terminal_friendly

    if not fires.any():
        return None

    # Unique target cells where at least one pattern fires
    legal_cells = np.unique(targets[fires])
    return legal_cells


CENTER_SET = np.array(sorted(CENTER_CELLS), dtype=np.int32)


def place_piece_no_flip(board, board_pos):
    """Place a piece on the board without flipping. Used for illegal moves."""
    r, c = board_pos // 8, board_pos % 8
    board.state[r, c] = board.next_hand_color
    board.next_hand_color *= -1


def generate_single_game(targets, terminals, opp_cells, opp_mask, rng):
    """Generate a single game using (corrupted) rules on exact board state."""
    board = OthelloBoardState()
    moves = []

    for turn in range(60):
        flat = board.state.flatten()
        is_black_turn = (board.next_hand_color == 1)

        legal_cells = evaluate_rules_vec(
            flat, is_black_turn, targets, terminals, opp_cells, opp_mask)

        if legal_cells is None:
            # No moves predicted — sample from empty non-center cells
            empty = np.where(flat == 0)[0]
            empty = np.setdiff1d(empty, CENTER_SET)
            if len(empty) == 0:
                break
            board_pos = int(empty[rng.randint(len(empty))])
        else:
            board_pos = int(legal_cells[rng.randint(len(legal_cells))])

        # Try legal move (with flips); if illegal, place without flipping
        try:
            board.update([board_pos])
        except:
            place_piece_no_flip(board, board_pos)

        moves.append(board_pos)

    return moves


def generate_games(targets, terminals, opp_cells, opp_mask, num_games, rng,
                   chunk_size=100000):
    """Generate games using (corrupted) rules."""
    all_games = []
    for chunk_start in range(0, num_games, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_games)
        for game_idx in range(chunk_start, chunk_end):
            game = generate_single_game(
                targets, terminals, opp_cells, opp_mask, rng)
            all_games.append(game)
        print(f"  Generated {len(all_games)}/{num_games} games...", flush=True)
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

    # Precompute arrays for vectorized evaluation
    targets, terminals, opp_cells, opp_mask = precompute_pattern_arrays(patterns)
    print(f"Pattern arrays: targets={targets.shape}, opp_cells={opp_cells.shape}")

    # Generate games
    game_rng = np.random.RandomState(args.seed + 1000)
    print(f"Generating {args.num_games} games...", flush=True)
    games = generate_games(targets, terminals, opp_cells, opp_mask,
                           args.num_games, game_rng)

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
