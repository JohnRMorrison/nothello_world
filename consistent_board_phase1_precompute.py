"""Phase 1 of the ambiguity analysis: precompute biased-MC consistent boards
for N game positions.  H-independent — no MLPs or probes loaded.

Saves a pickle that phase 2 (per-H) can load and evaluate fast.

Usage:
    python consistent_board_phase1_precompute.py \\
        --k 25 --num-games 300 --n-samples 1000 \\
        --output-pkl consist_boards_k25_N300.pkl
"""
import argparse
import os
import pickle
import random
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from compare_v4_vs_mlp import load_val_games
from data.othello import OthelloBoardState
from consistent_board_saved_probe import enumerate_consistent_boards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=25)
    ap.add_argument('--num-games', type=int, default=300)
    ap.add_argument('--n-samples', type=int, default=1000)
    ap.add_argument('--game-offset', type=int, default=2000)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    ap.add_argument('--output-pkl', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--flush-every', type=int, default=25,
                    help='Re-save the pickle every N processed games.')
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Loading games...")
    games = load_val_games(args.data_dir, args.num_data_files)
    experiment_games = games[args.game_offset:
                              args.game_offset + args.num_games]
    print(f"  experiment: {len(experiment_games)} games at k={args.k}")

    records = []
    n_positions = 0
    n_with_ambiguity = 0
    t0 = time.time()

    def _save():
        tmp = args.output_pkl + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump({
                'k': args.k,
                'n_samples': args.n_samples,
                'game_offset': args.game_offset,
                'records': records,
                'n_positions_seen': n_positions,
                'n_with_ambiguity': n_with_ambiguity,
                'total_games_planned': len(experiment_games),
            }, f)
        os.replace(tmp, args.output_pkl)

    for g_idx, game in enumerate(experiment_games):
        prefix = game[:args.k]

        boards = enumerate_consistent_boards(prefix, args.n_samples)
        n_distinct = len(boards)
        n_positions += 1

        # Training-observed board (replay of original prefix)
        b_actual = OthelloBoardState()
        valid_actual = True
        try:
            for m in prefix:
                b_actual.umpire(m)
        except Exception:
            valid_actual = False
        training_hash = (b_actual.state.tobytes()
                          if valid_actual else None)

        if n_distinct >= 2:
            n_with_ambiguity += 1

        records.append({
            'game_idx': g_idx,
            'prefix': list(prefix),
            'training_hash': training_hash,
            'boards': boards,          # {hash: (state_8x8, next_c, count)}
            'n_distinct_boards': n_distinct,
        })

        if (g_idx + 1) % args.flush_every == 0:
            _save()
            print(f"  {g_idx+1}/{len(experiment_games)}  "
                  f"ambiguous: {n_with_ambiguity}  "
                  f"records: {len(records)}  "
                  f"({int(time.time()-t0)}s)", flush=True)

    _save()
    print()
    print(f"Done.  {n_with_ambiguity}/{n_positions} ambiguous positions.")
    print(f"Output: {args.output_pkl}")


if __name__ == '__main__':
    main()
