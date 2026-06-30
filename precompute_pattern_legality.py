"""One-time precompute of pattern legality for chunk_ext_*.npz.

For each chunk_ext_NNNN.npz, derive the 960-d pattern legality mask from
the 64-d board state via compute_pattern_labels_batch.  Stores as PACKED
BITS (8 patterns per byte) for compactness:
    raw uint8:        n_rows × 960 bytes  (~17 GB / chunk)
    packed bits:      n_rows × 120 bytes  (~2 GB / chunk, before compress)
    after savez_compressed: typically ~1 GB / chunk
    40-chunk total: ~40 GB

Subsequent multi-seed training jobs load this directly (~30 sec/chunk
instead of ~19 min derive time).

Usage (single chunk):
    python precompute_pattern_legality.py --chunk-idx 0

Usage (SLURM array, recommended):
    sbatch --array=0-39%10 precompute_pattern_legality.sh
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from hand_crafted_flanking import enumerate_flanking_patterns
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import compute_pattern_labels_batch


_PATTERNS = enumerate_flanking_patterns()
_PAT_TARGETS, _PAT_TERMINALS, _PAT_OPP_CELLS, _PAT_OPP_MASK = (
    precompute_pattern_arrays(_PATTERNS)
)


def derive_pattern_labels(board_labels, positions, batch_size=200_000):
    """Compute 960-d pattern legality from 64-d board state, store as uint8."""
    n = len(board_labels)
    out = np.zeros((n, 960), dtype=np.uint8)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pat = compute_pattern_labels_batch(
            board_labels[start:end].astype(np.int8),
            positions[start:end].astype(np.int64),
            _PAT_TARGETS, _PAT_TERMINALS, _PAT_OPP_CELLS, _PAT_OPP_MASK,
        )
        out[start:end] = (pat > 0).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk-idx', type=int, default=None,
                    help='Which chunk (0..N-1) to process.  If unset, '
                         'reads SLURM_ARRAY_TASK_ID env var.')
    ap.add_argument('--chunk-dir',
                    default='experiments/mathematical_transformation_experiments/'
                            'heuristic_probe_results/feature_chunks')
    ap.add_argument('--chunk-glob', default='chunk_ext_*.npz')
    args = ap.parse_args()

    chunks = sorted(glob.glob(os.path.join(args.chunk_dir, args.chunk_glob)))
    print(f"Found {len(chunks)} chunks", flush=True)

    if args.chunk_idx is None:
        env = os.environ.get('SLURM_ARRAY_TASK_ID')
        if env is None:
            raise SystemExit("Provide --chunk-idx or run via SLURM array")
        args.chunk_idx = int(env)
    if args.chunk_idx < 0 or args.chunk_idx >= len(chunks):
        raise SystemExit(f"chunk-idx {args.chunk_idx} out of range [0, {len(chunks)})")

    chunk_path = chunks[args.chunk_idx]
    base = os.path.basename(chunk_path).replace('.npz', '_patlegal.npz')
    out_path = os.path.join(args.chunk_dir, base)
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}  (skipping)", flush=True)
        return

    print(f"  Processing {os.path.basename(chunk_path)} -> {base}",
          flush=True)
    t0 = time.time()
    with np.load(chunk_path) as z:
        board_labels = z['labels'].astype(np.int8)
        positions = z['positions'].astype(np.int64)
    print(f"  Loaded {len(board_labels):,} rows in {time.time()-t0:.1f}s",
          flush=True)

    t1 = time.time()
    pat_legal = derive_pattern_labels(board_labels, positions)
    print(f"  Derived 960-d labels in {time.time()-t1:.1f}s "
          f"({pat_legal.shape}, dtype={pat_legal.dtype})",
          flush=True)

    t2 = time.time()
    # Pack 8 patterns per byte: (n, 960) uint8 -> (n, 120) uint8 packed
    pat_packed = np.packbits(pat_legal, axis=1)
    n_rows = pat_legal.shape[0]
    np.savez_compressed(out_path,
                        pat_legal_packed=pat_packed,
                        n_rows=np.int64(n_rows),
                        n_patterns=np.int64(960))
    print(f"  Saved {out_path} ({os.path.getsize(out_path)/1e9:.2f} GB, "
          f"packbits) in {time.time()-t2:.1f}s", flush=True)
    print(f"  TOTAL: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
