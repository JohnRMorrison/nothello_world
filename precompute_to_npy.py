"""Re-save existing _when60.npz and _patterns.npz as uncompressed .npy files.

Why: np.savez_compressed stores data in a compressed zip. On load, numpy
decompresses into page cache that doesn't evict even with POSIX_FADV_DONTNEED.
Uncompressed .npy files can be loaded with mmap_mode='r', which keeps
pages in the page cache but backs them by the file on disk — the kernel
can evict them freely under memory pressure without reloading cost.

Usage:
    python precompute_to_npy.py --chunk-id 5
    sbatch --array=0-39 precompute_to_npy.sh
"""
import sys, os, time, argparse
sys.path.insert(0, '.')

import numpy as np


def convert_one_chunk(chunk_path):
    """For chunk_NNNN.npz, create chunk_NNNN_when60.npy and chunk_NNNN_patterns.npy."""
    when60_npz = chunk_path.replace(".npz", "_when60.npz")
    patterns_npz = chunk_path.replace(".npz", "_patterns.npz")

    when60_features_npy = chunk_path.replace(".npz", "_when60_features.npy")
    when60_labels_npy = chunk_path.replace(".npz", "_when60_labels.npy")
    when60_pos_npy = chunk_path.replace(".npz", "_when60_pos.npy")
    patterns_npy = chunk_path.replace(".npz", "_patterns.npy")

    # when60
    if os.path.exists(when60_npz) and not os.path.exists(when60_features_npy):
        t0 = time.time()
        with np.load(when60_npz) as d:
            feats = d['features']
            labels = d['labels']
            pos = d['positions']
            np.save(when60_features_npy, np.ascontiguousarray(feats))
            np.save(when60_labels_npy, np.ascontiguousarray(labels))
            np.save(when60_pos_npy, np.ascontiguousarray(pos))
        size_mb = (os.path.getsize(when60_features_npy) +
                   os.path.getsize(when60_labels_npy) +
                   os.path.getsize(when60_pos_npy)) / 1e6
        print(f"  when60 → {size_mb:.0f} MB in {time.time()-t0:.0f}s")
    else:
        print(f"  when60: already exists or source missing")

    # patterns
    if os.path.exists(patterns_npz) and not os.path.exists(patterns_npy):
        t0 = time.time()
        with np.load(patterns_npz) as d:
            np.save(patterns_npy, np.ascontiguousarray(d['pattern_labels']))
        size_mb = os.path.getsize(patterns_npy) / 1e6
        print(f"  patterns → {size_mb:.0f} MB in {time.time()-t0:.0f}s")
    else:
        print(f"  patterns: already exists or source missing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", type=int, default=None)
    parser.add_argument("--chunk-dir",
                        default="experiments/mathematical_transformation_experiments/"
                                "heuristic_probe_results/feature_chunks")
    args = parser.parse_args()

    chunk_files = sorted(f for f in os.listdir(args.chunk_dir)
                         if f.endswith(".npz")
                         and "_patterns" not in f
                         and "_when60" not in f)

    if args.chunk_id is not None:
        if args.chunk_id >= len(chunk_files):
            print(f"out of range: have {len(chunk_files)}")
            sys.exit(1)
        fname = chunk_files[args.chunk_id]
        print(f"Converting {fname}...")
        convert_one_chunk(os.path.join(args.chunk_dir, fname))
    else:
        for i, fname in enumerate(chunk_files):
            print(f"[{i}/{len(chunk_files)}] {fname}")
            convert_one_chunk(os.path.join(args.chunk_dir, fname))

    print("Done.")
