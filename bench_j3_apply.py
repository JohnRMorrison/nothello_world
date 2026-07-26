#!/usr/bin/env python
"""Measure the tree.apply fraction of a J3 (ordinal) training batch, and how
well threading the 960-tree apply scales -- so we know the REAL speedup before
committing to threading vs leaf-index caching.

Runs standalone; does NOT touch the live J3 jobs.  Uses the existing chunk
cache for a realistic batch.

  python bench_j3_apply.py \
    --bank banks/J3_ordinal.pt \
    --chunk-dir /workspace/feature_chunks \
    --cache-dir /workspace/chunk_cache
"""
import argparse, glob, os, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

import train_streaming_probe as tsp


def build_H_parallel(leaf_build, Xnp, workers):
    """Threaded build_H_from_leaves_np: one task per tree (tree.apply releases
    the GIL, so threads run truly in parallel).  Each task writes DISTINCT
    columns of H, so no locking needed."""
    from concurrent.futures import ThreadPoolExecutor
    N = Xnp.shape[0]
    H = np.zeros((N, len(leaf_build)), dtype=bool)
    cols_by_tree = defaultdict(list); tref = {}
    for col, (tree, nid) in enumerate(leaf_build):
        cols_by_tree[id(tree)].append((col, nid)); tref[id(tree)] = tree
    Xc = np.ascontiguousarray(Xnp)
    items = list(cols_by_tree.items())

    def work(item):
        tid, colnodes = item
        leaves = tref[tid].apply(Xc)
        for col, nid in colnodes:
            H[:, col] = (leaves == nid)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, items))
    return H


def timed(fn, reps=3):
    fn()                                   # warmup
    best = float('inf')
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='banks/J3_ordinal.pt')
    ap.add_argument('--chunk-dir', default='/workspace/feature_chunks')
    ap.add_argument('--cache-dir', default='/workspace/chunk_cache')
    ap.add_argument('--batch', type=int, default=2048,
                    help='per-batch size (training uses 2048)')
    ap.add_argument('--big', type=int, default=200_000,
                    help='big batch to model the precompute-once (leaf-cache) '
                         'scenario, where scaling is better')
    ap.add_argument('--load-cap', type=int, default=10_000_000,
                    help='cap used when the training cache was BUILT (so we hit '
                         'it instead of rebuilding).  Must match run_stream_grid '
                         "MAXPOS.  We only slice --big rows out of it.")
    ap.add_argument('--workers', default='16,32,64,96,192')
    ap.add_argument('--ply-min', type=int, default=5)
    ap.add_argument('--ply-max', type=int, default=54)
    args = ap.parse_args()

    print(f'torch threads={torch.get_num_threads()}  '
          f'cuda={torch.cuda.is_available()}')
    leaf_build = tsp.load_leaf_build(args.bank)
    if leaf_build is None:
        raise SystemExit(f'{args.bank} is not a --hidden-from-leaves bank')
    n_trees = len({id(t) for t, _ in leaf_build})
    n_leaves = len(leaf_build)
    print(f'bank: {n_trees} trees, {n_leaves} leaf units (hidden_dim)')

    chunk = sorted(glob.glob(os.path.join(args.chunk_dir, 'chunk_ext_*.npz')))[0]
    # Read the EXISTING training cache (cap=load_cap) and slice --big rows out
    # of it -- do NOT request a new cap (that would be a cache miss and rebuild
    # the whole chunk from the NFS volume).
    cpath = tsp._chunk_cache_path(args.cache_dir, chunk, args.ply_min,
                                  args.ply_max, True, args.load_cap)
    if os.path.exists(cpath):
        print(f'reading existing cache {os.path.basename(cpath)} '
              f'(slicing {args.big} rows)...', flush=True)
        z = np.load(cpath)
        X181 = z['X'][:args.big].astype(np.float32)   # slice before float cast
        z.close()
    else:
        raise SystemExit(
            f'no cache at {cpath}\n'
            f'  (expected cap={args.load_cap}; check --cache-dir / --load-cap '
            f'match how run_stream_grid_cached built it)')
    print(f'  got X {X181.shape}', flush=True)
    workers = [int(w) for w in args.workers.split(',')]

    for label, nrows in [('per-batch', args.batch), ('big/precompute', args.big)]:
        n = min(nrows, X181.shape[0])
        X241 = tsp.assemble_ordinal_241(X181[:n]).astype(np.float32)
        print(f'\n===== {label}: {n} rows =====')

        t_seq = timed(lambda: tsp.build_H_from_leaves_np(leaf_build, X241))
        print(f'  tree.apply SEQUENTIAL : {t_seq*1e3:8.1f} ms')

        best_par, best_w = t_seq, 1
        for w in workers:
            tw = timed(lambda: build_H_parallel(leaf_build, X241, w))
            spd = t_seq / tw
            print(f'  tree.apply {w:3d} threads: {tw*1e3:8.1f} ms   ({spd:4.1f}x)')
            if tw < best_par:
                best_par, best_w = tw, w

        # readout cost (the non-apply part of a batch): H->device, linear->960,
        # BCE, backward, step.  Approximates what does NOT speed up.
        H = tsp.build_H_from_leaves_np(leaf_build, X241)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lin = torch.nn.Linear(n_leaves, 960).to(dev)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
        tgt = torch.zeros(n, 960, device=dev)

        def readout():
            Ht = torch.from_numpy(H).to(device=dev, dtype=torch.float32)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(lin(Ht), tgt)
            loss.backward(); opt.step()
            if dev.type == 'cuda':
                torch.cuda.synchronize()
        t_read = timed(readout)
        print(f'  readout (non-apply)   : {t_read*1e3:8.1f} ms  (dev={dev.type})')

        f = t_seq / (t_seq + t_read)
        tot_seq = t_seq + t_read
        tot_par = best_par + t_read
        print(f'  --> tree.apply is {100*f:.1f}% of the batch')
        print(f'  --> best threading: {best_w} threads, '
              f'{t_seq/best_par:.1f}x on apply')
        print(f'  --> PROJECTED total per-batch speedup: '
              f'{tot_seq/tot_par:.1f}x  '
              f'({tot_seq*1e3:.1f}ms -> {tot_par*1e3:.1f}ms)')

    print('\nLeaf-index caching removes tree.apply from the training loop '
          'entirely (apply once per chunk at the "big" rate, reuse across '
          'epochs) -> training batches then cost ~the readout time alone.')


if __name__ == '__main__':
    main()
