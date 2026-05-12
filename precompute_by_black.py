"""Precompute a forfeit-correct 'played-by-black' channel for each feature chunk.

The standard +even feature encodes (played_turn % 2 == 0), which only matches
the actual playing color under strict B-W-B-W alternation. Forfeits break
this -- ~1% of positions in the synthetic training data follow a forfeit
and account for ~35% of MLP top-1 errors.

For each position independently, we:
  1. Reconstruct the move sequence from that position's own 'when' channel
     (when[c] = (played_turn + 1) / 60 if played, else 0).
  2. Replay through OthelloBoardState. At each move, record whether the
     player who actually played that cell was black (handles forfeits).
  3. Save by_black for that position.

Per-position rather than per-game so we don't depend on the chunk being
in game order (which it apparently isn't). Multiprocessed across the
chunk for speed.

Output: alongside each chunk file (chunk_NNNN.npz), writes chunk_NNNN_by_black.npy
with shape (N_chunk, 60), values in {0.0, 1.0}.

Usage:
    python precompute_by_black.py                  # all chunks
    python precompute_by_black.py --n-chunks 10    # first 10
    python precompute_by_black.py --workers 16     # tune parallelism
"""
import sys, os, argparse, glob, time
sys.path.insert(0, '.')

import numpy as np
from data.othello import OthelloBoardState
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import _load_features

N_MOVES = 60
CENTER_64 = {27, 28, 35, 36}
_movable_64 = [c for c in range(64) if c not in CENTER_64]
_b64_to_m60 = {c64: i for i, c64 in enumerate(_movable_64)}
_m60_to_b64 = np.array(_movable_64, dtype=np.int64)


def by_black_for_one_position(when_vec):
    """Replay the game prefix encoded in `when_vec` (60-d) and return a
    60-d float array: by_black[c] = 1.0 if cell c was first played by black,
    0.0 otherwise (including 'not played').
    """
    # Build move list: (played_turn, cell60) for cells with when > 0.
    moves = []
    for c in range(60):
        w = when_vec[c]
        if w > 0:
            t = int(round(float(w) * N_MOVES)) - 1
            if 0 <= t < N_MOVES:
                moves.append((t, c))
    moves.sort()

    out = np.zeros(60, dtype=np.float32)
    b = OthelloBoardState()
    for t, c60 in moves:
        c64 = int(_m60_to_b64[c60])
        try:
            color = b.next_hand_color   # +1 black, -1 white
            b.umpire(c64)
        except Exception:
            break
        if color == 1:
            out[c60] = 1.0
    return out


def process_slice(args):
    """Worker function: process a slice of positions. Returns (n_slice, 60)."""
    when_block = args   # already np.ndarray
    n = when_block.shape[0]
    out = np.zeros((n, 60), dtype=np.float32)
    for i in range(n):
        out[i] = by_black_for_one_position(when_block[i])
    return out


def process_chunk(chunk_path, n_workers=16):
    print(f"Processing {chunk_path}", flush=True)
    t0 = time.time()
    X_t, _, _ = _load_features(chunk_path)
    X = X_t.numpy() if hasattr(X_t, 'numpy') else np.asarray(X_t)
    N = X.shape[0]
    when_ch = X[:, N_MOVES:2 * N_MOVES].astype(np.float32)   # (N, 60)
    print(f"  {N} positions, {n_workers} workers", flush=True)

    if n_workers <= 1:
        by_black = np.zeros((N, 60), dtype=np.float32)
        for i in range(N):
            by_black[i] = by_black_for_one_position(when_ch[i])
            if (i + 1) % 100000 == 0:
                print(f"  {i+1}/{N}", flush=True)
    else:
        from multiprocessing import Pool
        # Split chunk into n_workers contiguous slices.
        slice_idx = np.array_split(np.arange(N), n_workers)
        slices = [when_ch[s] for s in slice_idx]
        with Pool(n_workers) as pool:
            results = pool.map(process_slice, slices)
        by_black = np.concatenate(results, axis=0)

    dt = time.time() - t0
    print(f"  Done in {dt:.0f}s ({N/dt:.0f} positions/sec)", flush=True)

    # Diagnostics
    even_ch = X[:, 2 * N_MOVES:3 * N_MOVES]
    played_mask = (when_ch > 0)
    diff_per_pos = ((by_black != even_ch) & played_mask).any(axis=1)
    print(f"  mean(by_black) over played cells: "
          f"{by_black[played_mask].mean():.4f}", flush=True)
    print(f"  Positions with >=1 forfeit-affected cell: "
          f"{diff_per_pos.sum()} / {N}  ({diff_per_pos.mean():.4%})", flush=True)

    out_path = chunk_path.replace('.npz', '_by_black.npy')
    np.save(out_path, by_black)
    print(f"  Saved {out_path}\n", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-dir",
                   default="experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks")
    p.add_argument("--chunk-glob", default="chunk_*.npz")
    p.add_argument("--n-chunks", type=int, default=None,
                   help="If set, process only the first N chunks (in sorted order).")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.chunk_dir, args.chunk_glob)))
    files = [f for f in files if "_patterns" not in f
                              and "_when60" not in f
                              and "_by_black" not in f]
    if args.n_chunks is not None:
        files = files[:args.n_chunks]
    print(f"Processing {len(files)} chunks with {args.workers} workers", flush=True)
    for f in files:
        process_chunk(f, n_workers=args.workers)
    print("All chunks done.", flush=True)
