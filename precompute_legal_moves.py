"""Precompute legal move vectors for all games and save to disk.

Usage:
    python precompute_legal_moves.py --max-games 6000000
"""
import sys, os
sys.path.insert(0, '.')

import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    N_MOVES, _MOVE_TO_IDX, POS_START, POS_END,
)
from experiments.mathematical_transformation_experiments.probe_variant_boards import (
    load_games, OthelloBoardState,
)

LENGTH = POS_END - POS_START  # 49 positions per game... but we use POS_END-1
POS_END_LEGAL = POS_END - 1   # we predict legal moves AFTER position t


def _compute_chunk(args):
    """Worker: compute legal moves for a chunk of games."""
    games_chunk, pos_start, pos_end, chunk_id = args
    length = pos_end - pos_start
    n = len(games_chunk)
    legal = np.zeros((n, length, N_MOVES), dtype=np.int8)

    for gi, game in enumerate(games_chunk):
        board = OthelloBoardState()
        for s in range(pos_start):
            board.umpire(game[s])
        for ti, t in enumerate(range(pos_start, pos_end)):
            board.umpire(game[t])
            for m in board.get_valid_moves():
                if m in _MOVE_TO_IDX:
                    legal[gi, ti, _MOVE_TO_IDX[m]] = 1

        if (gi + 1) % 50000 == 0:
            print(f"    Worker {chunk_id}: {gi + 1}/{n} games", flush=True)

    return legal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Number of workers (default: min(cpu_count, 16))")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/"
                                "heuristic_probe_results/adversarial")
    parser.add_argument("--chunk-size", type=int, default=500000,
                        help="Games per output file (to avoid huge single files)")
    args = parser.parse_args()

    print("Loading games...")
    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    print(f"Loaded {len(games)} games")

    n_workers = args.n_workers or min(cpu_count(), 16)
    pos_start = POS_START
    pos_end = POS_END_LEGAL
    length = pos_end - pos_start

    os.makedirs(args.output_dir, exist_ok=True)

    # Process in output chunks to avoid huge single files
    chunk_size = args.chunk_size
    n_output_chunks = (len(games) + chunk_size - 1) // chunk_size

    for ci in range(n_output_chunks):
        g_start = ci * chunk_size
        g_end = min(g_start + chunk_size, len(games))
        chunk_games = games[g_start:g_end]
        n_games = len(chunk_games)

        out_path = os.path.join(args.output_dir,
                                f"legal_moves_chunk_{ci:04d}.npz")
        if os.path.exists(out_path):
            print(f"Chunk {ci} already exists ({out_path}), skipping")
            continue

        print(f"\nChunk {ci}/{n_output_chunks}: games {g_start}-{g_end} "
              f"({n_games} games)")

        # Split across workers
        worker_chunk = (n_games + n_workers - 1) // n_workers
        work_items = []
        for wi in range(0, n_games, worker_chunk):
            work_items.append((
                chunk_games[wi:wi + worker_chunk],
                pos_start, pos_end, len(work_items)
            ))

        print(f"  Using {n_workers} workers...", flush=True)
        with Pool(n_workers) as pool:
            results = pool.map(_compute_chunk, work_items)

        # Concatenate: (n_games, length, 60) int8
        legal = np.concatenate(results, axis=0)
        print(f"  Shape: {legal.shape}, saving...")

        np.savez_compressed(out_path, legal=legal)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Saved to {out_path} ({size_mb:.1f} MB)")

    print(f"\nDone! {n_output_chunks} chunk(s) saved to {args.output_dir}")


if __name__ == "__main__":
    main()
