"""
Generate Othello games using corrupted hand-crafted flanking networks.

Three corruption types:
  1. Interpolate toward random weights
  2. Randomize a fraction of individual weights
  3. Redirect heuristic outputs to random cells

Usage:
  python generate_corruption_games.py --corruption-type 1 --alpha 0.1 --num-games 2000000 --output-dir experiments/corruption/games/type1_alpha010
"""

import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data.othello import OthelloBoardState
from hand_crafted_flanking import (
    enumerate_flanking_patterns, HandCraftedFlanking, encode_board,
    VALID_MOVES, MOVE_TO_IDX, N_MOVES, CENTER_CELLS
)


def corrupt_type1(net, alpha, rng):
    """Interpolate hidden layer toward random weights."""
    with torch.no_grad():
        w = net.hidden.weight.data
        b = net.hidden.bias.data
        # Xavier init scale
        fan_in, fan_out = w.shape[1], w.shape[0]
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        w_rand = torch.tensor(rng.normal(0, std, size=w.shape), dtype=torch.float32)
        b_rand = torch.tensor(rng.normal(0, std, size=b.shape), dtype=torch.float32)
        net.hidden.weight.data = (1 - alpha) * w + alpha * w_rand
        net.hidden.bias.data = (1 - alpha) * b + alpha * b_rand


def corrupt_type2(net, alpha, rng):
    """Randomize a fraction of individual weights."""
    with torch.no_grad():
        w = net.hidden.weight.data
        b = net.hidden.bias.data
        fan_in, fan_out = w.shape[1], w.shape[0]
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        # Weight matrix
        mask_w = torch.tensor(rng.random(w.shape) < alpha, dtype=torch.bool)
        w_rand = torch.tensor(rng.normal(0, std, size=w.shape), dtype=torch.float32)
        net.hidden.weight.data = torch.where(mask_w, w_rand, w)
        # Bias
        mask_b = torch.tensor(rng.random(b.shape) < alpha, dtype=torch.bool)
        b_rand = torch.tensor(rng.normal(0, std, size=b.shape), dtype=torch.float32)
        net.hidden.bias.data = torch.where(mask_b, b_rand, b)


def corrupt_type3(net, alpha, rng, patterns):
    """Redirect heuristic outputs to random cells."""
    with torch.no_grad():
        num_patterns = len(patterns)
        for j in range(num_patterns):
            if rng.random() < alpha:
                # Zero out old connection
                old_target = patterns[j]['target']
                old_idx = MOVE_TO_IDX[old_target]
                net.output.weight.data[old_idx, j] = 0.0
                # Connect to random playable cell
                new_idx = rng.randint(N_MOVES)
                net.output.weight.data[new_idx, j] = 1.0


def generate_games(even_net, odd_net, num_games, rng, chunk_size=100000):
    """Generate games using corrupted networks."""
    all_games = []
    for chunk_start in range(0, num_games, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_games)
        chunk_games = []
        for game_idx in range(chunk_start, chunk_end):
            game = generate_single_game(even_net, odd_net, rng)
            chunk_games.append(game)
        all_games.extend(chunk_games)
        print(f"  Generated {len(all_games)}/{num_games} games...")
    return all_games


def generate_single_game(even_net, odd_net, rng):
    """Generate a single game using corrupted networks on exact board state."""
    board = OthelloBoardState()
    moves = []
    consecutive_passes = 0

    for turn in range(60):
        is_black_turn = (board.next_hand_color == 1)
        x = encode_board(board.state, is_black_turn).unsqueeze(0)
        net = even_net if is_black_turn else odd_net

        with torch.no_grad():
            probs = net(x).squeeze(0).numpy()

        # Get predicted legal moves (threshold at 0.5)
        predicted_legal = np.where(probs > 0.5)[0]

        if len(predicted_legal) == 0:
            # No moves predicted legal — sample from empty cells
            flat = board.state.flatten()
            empty_cells = [i for i in range(64) if flat[i] == 0 and i not in CENTER_CELLS]
            if not empty_cells:
                break
            # Try each empty cell, pick one that's actually playable
            # If none works, pass
            board_pos = empty_cells[rng.randint(len(empty_cells))]
            try:
                board.update([board_pos])
                moves.append(board_pos)
                consecutive_passes = 0
            except:
                # Invalid move — pass
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break
                board.update([])
        else:
            # Sample from predicted legal moves
            move_idx = predicted_legal[rng.randint(len(predicted_legal))]
            board_pos = VALID_MOVES[move_idx]

            try:
                board.update([board_pos])
                moves.append(board_pos)
                consecutive_passes = 0
            except:
                # Predicted move was actually illegal — pass
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break
                board.update([])

    return moves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corruption-type", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--num-games", type=int, default=2000000)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Corruption type: {args.corruption_type}")
    print(f"Alpha: {args.alpha}")
    print(f"Num games: {args.num_games}")
    print(f"Seed: {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build networks
    patterns = enumerate_flanking_patterns()
    print(f"Flanking patterns: {len(patterns)}")

    even_net = HandCraftedFlanking(patterns)
    odd_net = HandCraftedFlanking(patterns)

    # Apply corruption
    rng = np.random.RandomState(args.seed)
    if args.alpha > 0:
        if args.corruption_type == 1:
            corrupt_type1(even_net, args.alpha, rng)
            corrupt_type1(odd_net, args.alpha, rng)
        elif args.corruption_type == 2:
            corrupt_type2(even_net, args.alpha, rng)
            corrupt_type2(odd_net, args.alpha, rng)
        elif args.corruption_type == 3:
            corrupt_type3(even_net, args.alpha, rng, patterns)
            corrupt_type3(odd_net, args.alpha, rng, patterns)

    even_net.eval()
    odd_net.eval()

    # Generate games
    game_rng = np.random.RandomState(args.seed + 1000)
    print(f"Generating {args.num_games} games...")
    games = generate_games(even_net, odd_net, args.num_games, game_rng)

    # Report stats
    lengths = [len(g) for g in games]
    print(f"\nGame stats:")
    print(f"  Total games: {len(games)}")
    print(f"  Mean length: {np.mean(lengths):.1f}")
    print(f"  Min length: {np.min(lengths)}")
    print(f"  Max length: {np.max(lengths)}")
    print(f"  Games with 60 moves: {sum(1 for l in lengths if l == 60)}")

    # Save in pickle format (same as existing Othello data)
    out_path = os.path.join(args.output_dir, "games.pickle")
    with open(out_path, 'wb') as f:
        pickle.dump(games, f)
    print(f"Saved to {out_path}")

    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.txt")
    with open(meta_path, 'w') as f:
        f.write(f"corruption_type={args.corruption_type}\n")
        f.write(f"alpha={args.alpha}\n")
        f.write(f"num_games={len(games)}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"mean_length={np.mean(lengths):.1f}\n")
    print(f"Saved metadata to {meta_path}")


if __name__ == '__main__':
    main()
