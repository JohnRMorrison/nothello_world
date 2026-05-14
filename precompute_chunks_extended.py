"""Precompute extended-range chunks (turns 5-58) for fair late-game eval.

Mirrors the existing chunk_NNNN.npz format but extends the turn range
from 5-53 to 5-58. Lets us retrain pattern detectors + probes on the
full distribution OGPT can handle, for fair MLP-vs-OGPT comparison at
end-game turns.

Output: chunk_ext_NNNN.npz with the same keys (features/labels/positions).

Usage (SLURM array, 40 tasks of 6 pickle files each):
    sbatch --array=0-39%20 run_precompute_chunks_extended.sh
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
    p.add_argument("--start-file", type=int, required=True)
    p.add_argument("--end-file", type=int, required=True)
    p.add_argument("--chunk-id", type=int, required=True)
    p.add_argument("--pos-start", type=int, default=5)
    p.add_argument("--pos-end", type=int, default=59,
                   help="Exclusive upper bound; max=59 because block_size=59.")
    p.add_argument("--output-dir",
        default="experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks")
    args = p.parse_args()

    t0 = time.time()
    files = sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pickle"))
    files = files[args.start_file:args.end_file]
    print(f"Loading {len(files)} pickle files: {files[:1]}...{files[-1:]}")
    games = []
    for fname in files:
        with open(os.path.join(SYNTHETIC_DIR, fname), "rb") as f:
            batch = pickle.load(f)
        games.extend(g for g in batch if len(g) == GAME_LEN)
    print(f"Loaded {len(games)} games in {time.time()-t0:.0f}s")

    t1 = time.time()
    feats, labels, positions = _build_move_features_batch(
        games, args.pos_start, args.pos_end, include_pairwise=False
    )
    print(f"  features:  {tuple(feats.shape)}  dtype={feats.dtype}")
    print(f"  labels:    {tuple(labels.shape)}  dtype={labels.dtype}")
    print(f"  positions: {tuple(positions.shape)}  unique={sorted(np.unique(positions.numpy()))}")
    print(f"  built in {time.time()-t1:.0f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir,
                            f"chunk_ext_{args.chunk_id:04d}.npz")
    np.savez_compressed(out_path,
        features=feats.numpy().astype(np.float16),
        labels=labels.numpy().astype(np.int8),
        positions=positions.numpy().astype(np.int8),
    )
    print(f"Saved {out_path}  (total: {time.time()-t0:.0f}s)")
