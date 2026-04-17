"""Precompute 960-bit pattern labels for each position in feature chunks.

For each chunk_NNNN.npz, creates chunk_NNNN_patterns.npz containing:
  - pattern_labels: (N, 960) uint8 — binary firing vector per position

Usage:
    python precompute_pattern_labels.py                    # all chunks
    python precompute_pattern_labels.py --chunk-id 5       # single chunk
    sbatch --array=0-39 precompute_pattern_labels.sh       # parallel
"""
import sys, os, time
sys.path.insert(0, '.')

import argparse
import numpy as np

from hand_crafted_flanking import enumerate_flanking_patterns
from generate_rule_games import precompute_pattern_arrays
from train_pattern_detectors import compute_pattern_labels_batch


def precompute_one_chunk(chunk_path, output_path, targets, terminals,
                         opp_cells, opp_mask, block_size=100000):
    """Compute pattern labels for one chunk, in blocks to limit memory."""
    data = np.load(chunk_path)
    labels = data['labels']      # (N, 64) int64
    positions = data['positions']  # (N,) int64
    n = len(labels)

    print(f"  {os.path.basename(chunk_path)}: {n} samples")

    pattern_labels = np.zeros((n, len(targets)), dtype=np.uint8)

    t0 = time.time()
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = compute_pattern_labels_batch(
            labels[start:end], positions[start:end],
            targets, terminals, opp_cells, opp_mask)
        pattern_labels[start:end] = block.astype(np.uint8)

        if (start // block_size) % 10 == 0 and start > 0:
            elapsed = time.time() - t0
            rate = end / elapsed
            eta = (n - end) / rate
            print(f"    {end}/{n} ({100*end/n:.0f}%) — {eta:.0f}s remaining")

    np.savez_compressed(output_path, pattern_labels=pattern_labels)
    elapsed = time.time() - t0
    firing_rate = pattern_labels.mean()
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"    Done in {elapsed:.0f}s — firing_rate={firing_rate:.4f}, "
          f"saved {size_mb:.1f} MB to {os.path.basename(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", type=int, default=None,
                        help="Process single chunk (for SLURM array jobs)")
    parser.add_argument("--chunk-dir",
                        default="experiments/mathematical_transformation_experiments/"
                                "heuristic_probe_results/feature_chunks")
    args = parser.parse_args()

    patterns = enumerate_flanking_patterns()
    targets, terminals, opp_cells, opp_mask = precompute_pattern_arrays(patterns)
    print(f"Patterns: {len(patterns)}")

    chunk_files = sorted(f for f in os.listdir(args.chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    print(f"Found {len(chunk_files)} feature chunks in {args.chunk_dir}")

    if args.chunk_id is not None:
        # Single chunk mode
        if args.chunk_id >= len(chunk_files):
            print(f"Chunk {args.chunk_id} out of range (have {len(chunk_files)})")
            sys.exit(1)
        fname = chunk_files[args.chunk_id]
        chunk_path = os.path.join(args.chunk_dir, fname)
        out_name = fname.replace(".npz", "_patterns.npz")
        output_path = os.path.join(args.chunk_dir, out_name)

        if os.path.exists(output_path):
            print(f"  Already exists: {output_path}, skipping")
        else:
            precompute_one_chunk(chunk_path, output_path,
                                 targets, terminals, opp_cells, opp_mask)
    else:
        # All chunks
        for i, fname in enumerate(chunk_files):
            chunk_path = os.path.join(args.chunk_dir, fname)
            out_name = fname.replace(".npz", "_patterns.npz")
            output_path = os.path.join(args.chunk_dir, out_name)

            if os.path.exists(output_path):
                print(f"  [{i}/{len(chunk_files)}] Already exists: {out_name}, skipping")
                continue

            print(f"  [{i}/{len(chunk_files)}] Processing {fname}...")
            precompute_one_chunk(chunk_path, output_path,
                                 targets, terminals, opp_cells, opp_mask)

    print("Done.")
