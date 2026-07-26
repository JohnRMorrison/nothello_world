#!/usr/bin/env python
"""Process-parallel leaf-index cache for J3 (ordinal) -- the fast path.

Threading tree.apply failed (GIL held + false sharing -> 4x SLOWER, measured).
Separate PROCESSES bypass the GIL, and fork shares the 960 trees copy-on-write
(read-only -> never actually copied), so this parallelises for real.

For each chunk we apply the 960 ordinal trees ONCE and save an (N, 960) int16
matrix of the leaf node-id each row lands in per tree.  Row blocks are written
directly to the output file with os.pwrite at disjoint byte offsets -- no mmap
(NFS-safe), no 19GB IPC.  Training then builds the one-hot hidden layer by a
cheap gather+compare (leaves[:,col_tree_idx]==col_nid) instead of walking 960
trees every batch -- turning ~6.4 hr/chunk of tree.apply into a one-time
~minutes/chunk, reused across all epochs.

  python build_leaf_cache.py --bank banks/J3_ordinal.pt \
    --chunk-dir /workspace/feature_chunks --cache-dir /workspace/chunk_cache \
    --leaf-cache-dir /workspace/leaf_cache --procs 96
"""
import argparse, glob, os, time
import numpy as np
import multiprocessing as mp

import train_streaming_probe as tsp

_SH = {}   # fork-shared read-only state: 'X', 'trees', 'path', 'nt'


def tree_order(leaf_build):
    """Unique trees in stable order + per-H-column (tree_index, node_id)."""
    trees, pos = [], {}
    col_tree_idx = np.empty(len(leaf_build), dtype=np.int32)
    col_nid = np.empty(len(leaf_build), dtype=np.int32)
    for col, (tree, nid) in enumerate(leaf_build):
        tid = id(tree)
        if tid not in pos:
            pos[tid] = len(trees); trees.append(tree)
        col_tree_idx[col] = pos[tid]; col_nid[col] = nid
    return trees, col_tree_idx, col_nid


def _apply_block(rng):
    lo, hi = rng
    X = _SH['X']; trees = _SH['trees']; nt = _SH['nt']
    Xc = np.ascontiguousarray(X[lo:hi])
    out = np.empty((hi - lo, nt), dtype=np.int16)     # node-ids fit in int16
    for j, t in enumerate(trees):
        out[:, j] = t.apply(Xc)
    fd = os.open(_SH['path'], os.O_WRONLY)
    try:
        os.pwrite(fd, out.tobytes(), lo * nt * 2)     # disjoint row block
    finally:
        os.close(fd)
    return hi - lo


def build_leaves_to_file(trees, X241, path, procs, block):
    N, nt = X241.shape[0], len(trees)
    with open(path, 'wb') as f:                       # preallocate full size
        f.truncate(N * nt * 2)
    ranges = [(lo, min(lo + block, N)) for lo in range(0, N, block)]
    _SH.update(X=X241, trees=trees, path=path, nt=nt)
    ctx = mp.get_context('fork')
    done = 0
    with ctx.Pool(procs) as pool:
        for n in pool.imap_unordered(_apply_block, ranges):
            done += n
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='banks/J3_ordinal.pt')
    ap.add_argument('--chunk-dir', default='/workspace/feature_chunks')
    ap.add_argument('--cache-dir', default='/workspace/chunk_cache')
    ap.add_argument('--leaf-cache-dir', default='/workspace/leaf_cache')
    ap.add_argument('--load-cap', type=int, default=10_000_000)
    ap.add_argument('--eval-cap', type=int, default=500_000)
    ap.add_argument('--procs', type=int, default=96)
    ap.add_argument('--block', type=int, default=50_000)
    ap.add_argument('--ply-min', type=int, default=5)
    ap.add_argument('--ply-max', type=int, default=54)
    ap.add_argument('--validate', type=int, default=8192,
                    help='rows checked vs sequential build_H (0=skip)')
    args = ap.parse_args()

    os.makedirs(args.leaf_cache_dir, exist_ok=True)
    leaf_build = tsp.load_leaf_build(args.bank)
    if leaf_build is None:
        raise SystemExit(f'{args.bank} is not a --hidden-from-leaves bank')
    trees, col_tree_idx, col_nid = tree_order(leaf_build)
    nt = len(trees)
    print(f'{nt} trees, {len(leaf_build)} H columns, procs={args.procs}',
          flush=True)
    np.savez(os.path.join(args.leaf_cache_dir, 'colmap.npz'),
             col_tree_idx=col_tree_idx, col_nid=col_nid, n_trees=nt)

    chunks = sorted(glob.glob(os.path.join(args.chunk_dir, 'chunk_ext_*.npz')))
    jobs = [(c, args.load_cap) for c in chunks[:-1]] + [(chunks[-1], args.eval_cap)]

    for chunk, cap in jobs:
        base = os.path.basename(chunk)
        stem = f'{base}.ply{args.ply_min}-{args.ply_max}.cap{cap}.leaves.i16'
        out = os.path.join(args.leaf_cache_dir, stem)
        if os.path.exists(out) and os.path.exists(out + '.meta.npz'):
            print(f'  {base} (cap {cap}): already cached', flush=True)
            continue
        t = time.time()
        X181, S, T, L = tsp.load_chunk_cached(
            chunk, args.ply_min, args.ply_max, True, cap,
            needs_ordinal=True, cache_dir=args.cache_dir)
        if X181 is None:
            print(f'  {base}: no rows, skipped', flush=True); continue
        X241 = tsp.assemble_ordinal_241(X181).astype(np.float32)
        del X181
        N = X241.shape[0]
        tmp = out + '.tmp'
        build_leaves_to_file(trees, X241, tmp, args.procs, args.block)

        if args.validate:
            v = min(args.validate, N)
            H_seq = tsp.build_H_from_leaves_np(leaf_build, X241[:v])
            lv = np.fromfile(tmp, dtype=np.int16, count=v * nt).reshape(v, nt)
            H_fast = (lv[:, col_tree_idx] == col_nid[None, :])
            if not np.array_equal(H_seq, H_fast):
                os.remove(tmp)
                raise SystemExit(f'LEAF CACHE MISMATCH on {base} -- aborting')
            print(f'    validate {v} rows: H exact match', flush=True)

        os.replace(tmp, out)
        np.savez(out + '.meta.npz', N=N, n_trees=nt)
        np.savez(out + '.stl.npz', S=np.asarray(S, np.int8),
                 T=np.asarray(T, np.int16), L=np.asarray(L, np.uint8))
        dt = time.time() - t
        gb = N * nt * 2 / 1e9
        print(f'  {base} (cap {cap}): {N} rows in {dt:.0f}s '
              f'({N/max(dt,1):,.0f} rows/s, {gb:.1f}GB)', flush=True)
        del X241, S, T, L

    print('leaf cache done.', flush=True)


if __name__ == '__main__':
    main()
