"""Build an eval chunk with features + labels for late-game turns 54-58.

The existing chunks (chunk_NNNN.npz) only go up to turn 53. This produces
a parallel chunk covering turns 54-58, saved as `late_turns_eval.npz`.
Same format as chunk_NNNN.npz so existing analysis scripts can consume it.

Output keys:
  features  (N, 180)  float16  same 180-d move-history features
  labels    (N, 64)   int8     0=empty, 1=white, 2=black
  positions (N,)      int8     turn number (54..58)

Usage:
    python precompute_late_turn_eval.py --n-games 50000
"""
import sys, os, argparse, pickle, time
sys.path.insert(0, '.')

import numpy as np

from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _build_move_features_batch,
)
sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import SYNTHETIC_DIR, GAME_LEN


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-games", type=int, default=20000)
    p.add_argument("--pos-start", type=int, default=54)
    p.add_argument("--pos-end", type=int, default=59,
                   help="Exclusive upper bound; max=59 because GAME_LEN=60.")
    p.add_argument("--max-files", type=int, default=2)
    p.add_argument("--output-dir",
        default="experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks")
    args = p.parse_args()

    t0 = time.time()
    files = sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pickle"))
    files = files[:args.max_files]
    games = []
    for fname in files:
        with open(os.path.join(SYNTHETIC_DIR, fname), "rb") as f:
            batch = pickle.load(f)
        games.extend(g for g in batch if len(g) == GAME_LEN)
    if len(games) > args.n_games:
        games = games[:args.n_games]
    print(f"Loaded {len(games)} games in {time.time()-t0:.0f}s")
    print(f"Building features for turns [{args.pos_start}, {args.pos_end})...")

    feats, labels, positions = _build_move_features_batch(
        games, args.pos_start, args.pos_end, include_pairwise=False
    )
    print(f"  features:  {tuple(feats.shape)}  dtype={feats.dtype}")
    print(f"  labels:    {tuple(labels.shape)}  dtype={labels.dtype}")
    print(f"  positions: {tuple(positions.shape)}  unique={sorted(np.unique(positions.numpy()))}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "late_turns_eval.npz")
    np.savez_compressed(out_path,
        features=feats.numpy().astype(np.float16),
        labels=labels.numpy().astype(np.int8),
        positions=positions.numpy().astype(np.int8),
    )
    print(f"Saved {out_path}  (total time: {time.time()-t0:.0f}s)")
