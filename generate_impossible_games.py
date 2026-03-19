"""Generate Othello games using spatially impossible flanking rules.

Uses cross-wired patterns from generate_impossible_patterns.py.
Can either target a specific GER value (hill-climbing search) or
load pre-saved patterns from a JSON file.

Output format matches generate_rule_games.py: games.pickle + legal_moves.pickle.

Usage:
  python generate_impossible_games.py --target-ger 0.2 --num-games 2000000 --output-dir experiments/impossible/games/ger020
  python generate_impossible_games.py --patterns-file experiments/impossible/patterns/patterns_ger0.20.json --num-games 2000000 --output-dir experiments/impossible/games/ger020
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hand_crafted_flanking import enumerate_flanking_patterns
from generate_rule_games import (
    precompute_pattern_arrays, generate_games
)
from generate_impossible_patterns import (
    cross_wire_patterns, search_target_ger, compute_ger, count_dead_patterns,
    compressed_size, patterns_to_bytes
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate games from spatially impossible flanking rules")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-ger", type=float,
                       help="Target GER value (will search for matching patterns)")
    group.add_argument("--patterns-file", type=str,
                       help="Path to pre-saved patterns JSON file")

    parser.add_argument("--num-games", type=int, default=2000000)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=10000,
                        help="Max iterations for GER search")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.patterns_file:
        # Load pre-saved patterns
        print(f"Loading patterns from {args.patterns_file}")
        with open(args.patterns_file) as f:
            raw = json.load(f)
        patterns = []
        for p in raw:
            patterns.append({
                'target': p['target'],
                'opponents': p['opponents'],
                'terminal': p['terminal'],
                'direction': tuple(p['direction']),
                'length': p['length'],
            })
        ger = compute_ger(patterns)
        print(f"Loaded {len(patterns)} patterns, GER={ger:.4f}")
    else:
        # Search for target GER
        print(f"Target GER: {args.target_ger}")
        base_patterns = enumerate_flanking_patterns()
        print(f"Base patterns: {len(base_patterns)}")

        patterns, ger, n_swaps = search_target_ger(
            base_patterns, args.target_ger, seed=args.seed,
            max_iter=args.max_iter)
        print(f"Achieved GER: {ger:.4f} ({n_swaps} swaps)")

    gz = compressed_size(patterns_to_bytes(patterns))
    dead = count_dead_patterns(patterns)
    print(f"Gzip size: {gz} bytes, dead patterns: {dead}/{len(patterns)}")

    # Precompute arrays for vectorized evaluation
    targets, terminals, opp_cells, opp_mask = precompute_pattern_arrays(patterns)
    print(f"Pattern arrays: targets={targets.shape}, opp_cells={opp_cells.shape}")

    # Generate games
    game_rng = np.random.RandomState(args.seed + 1000)
    print(f"Generating {args.num_games} games...", flush=True)
    games, legal_moves = generate_games(
        targets, terminals, opp_cells, opp_mask,
        args.num_games, game_rng, save_legal=True)

    # Stats
    lengths = [len(g) for g in games]
    print(f"\nGame stats:")
    print(f"  Total games: {len(games)}")
    print(f"  Mean length: {np.mean(lengths):.1f}")
    print(f"  Min/Max length: {np.min(lengths)}/{np.max(lengths)}")
    print(f"  Games with 60 moves: {sum(1 for l in lengths if l == 60)}")

    # Save
    with open(os.path.join(args.output_dir, "games.pickle"), 'wb') as f:
        pickle.dump(games, f)
    with open(os.path.join(args.output_dir, "legal_moves.pickle"), 'wb') as f:
        pickle.dump(legal_moves, f)

    # Save metadata
    meta = {
        'ger': ger,
        'gzip_size': gz,
        'n_patterns': len(patterns),
        'n_dead': dead,
        'num_games': len(games),
        'seed': args.seed,
        'mean_length': float(np.mean(lengths)),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {args.output_dir}")


if __name__ == '__main__':
    main()
