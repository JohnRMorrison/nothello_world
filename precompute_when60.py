"""Precompute 60-d 'when' feature chunks for faster/smaller training runs.

For each chunk_NNNN.npz, creates chunk_NNNN_when60.npz containing:
  - features: (N, 60) float32 — just the 'when' slice (cols 60-119 of 180-d)
  - labels:   (N, 64) uint8   — board state (0=empty, 1=white, 2=black)
  - positions: (N,) uint8    — move position (always 5-54)

Downstream training loads these smaller files to avoid holding the full
180-d chunk in memory (3.5 GB → 1.15 GB features, 2.5 GB → 307 MB labels).

Usage:
    python precompute_when60.py                    # all chunks
    python precompute_when60.py --chunk-id 5       # single chunk
    sbatch --array=0-39 precompute_when60.sh       # parallel
"""
import sys, os, time
sys.path.insert(0, '.')

import argparse
import numpy as np


N_MOVES = 60  # number of valid move IDs


def precompute_one_chunk(chunk_path, output_path):
    """Extract 60-d 'when' features + compact labels/positions for one chunk."""
    data = np.load(chunk_path)
    features_180 = data['features']     # (N, 180) float32
    labels = data['labels']              # (N, 64) int64
    positions = data['positions']        # (N,) int64
    n = len(features_180)

    print(f"  {os.path.basename(chunk_path)}: {n} samples")
    t0 = time.time()

    # Extract 'when' slice: columns N_MOVES to 2*N_MOVES (60 to 120)
    # Force float32 — source chunks may be float16
    features_when = features_180[:, N_MOVES:2 * N_MOVES].astype(np.float32)

    # Compact labels to uint8 (values are 0, 1, 2)
    labels_u8 = labels.astype(np.uint8)

    # Compact positions to uint8 (values are 5..54)
    positions_u8 = positions.astype(np.uint8)

    np.savez_compressed(
        output_path,
        features=features_when,
        labels=labels_u8,
        positions=positions_u8,
    )
    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"    Done in {elapsed:.0f}s — saved {size_mb:.1f} MB to "
          f"{os.path.basename(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", type=int, default=None,
                        help="Process single chunk (for SLURM array jobs)")
    parser.add_argument("--chunk-dir",
                        default="experiments/mathematical_transformation_experiments/"
                                "heuristic_probe_results/feature_chunks")
    args = parser.parse_args()

    chunk_files = sorted(f for f in os.listdir(args.chunk_dir)
                         if f.endswith(".npz")
                         and "_patterns" not in f
                         and "_when60" not in f)
    print(f"Found {len(chunk_files)} source feature chunks in {args.chunk_dir}")

    if args.chunk_id is not None:
        if args.chunk_id >= len(chunk_files):
            print(f"Chunk {args.chunk_id} out of range (have {len(chunk_files)})")
            sys.exit(1)
        fname = chunk_files[args.chunk_id]
        chunk_path = os.path.join(args.chunk_dir, fname)
        out_name = fname.replace(".npz", "_when60.npz")
        output_path = os.path.join(args.chunk_dir, out_name)

        if os.path.exists(output_path):
            print(f"  Already exists: {output_path}, skipping")
        else:
            precompute_one_chunk(chunk_path, output_path)
    else:
        for i, fname in enumerate(chunk_files):
            chunk_path = os.path.join(args.chunk_dir, fname)
            out_name = fname.replace(".npz", "_when60.npz")
            output_path = os.path.join(args.chunk_dir, out_name)

            if os.path.exists(output_path):
                print(f"  [{i}/{len(chunk_files)}] Already exists: {out_name}, skipping")
                continue

            print(f"  [{i}/{len(chunk_files)}] Processing {fname}...")
            precompute_one_chunk(chunk_path, output_path)

    print("Done.")
