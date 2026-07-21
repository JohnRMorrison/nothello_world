"""Prune a pattern-tree checkpoint to top-K leaves per pattern.

The checkpoint stores W (hidden weights), b (biases), path_info (per-leaf
metadata including leaf_counts and pattern id).  This script keeps only
the top-K leaves per pattern (by training-count sum), producing a
narrower checkpoint compatible with the streaming pipeline.

Usage:
    python prune_tree_checkpoint.py \\
        --in-ckpt ckpts_midgame/midgame_leg_pattern_trees_no_recent_canonical_g20000_d15_ml10_p10-50.pt \\
        --top-k 1 \\
        --out ckpts_midgame/midgame_leg_pattern_trees_no_recent_canonical_g20000_d15_ml10_p10-50_topk1.pt
"""
import argparse
from collections import defaultdict

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-ckpt', required=True)
    ap.add_argument('--top-k', type=int, required=True,
                    help='Max leaves to keep per pattern.')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    print(f'Loading {args.in_ckpt}')
    ck = torch.load(args.in_ckpt, map_location='cpu')
    W = ck['W'].numpy() if isinstance(ck['W'], torch.Tensor) else ck['W']
    b = ck['b'].numpy() if isinstance(ck['b'], torch.Tensor) else ck['b']
    meta = ck['path_info']

    # Group leaf indices by pattern id (only 'pattern_path' kind entries
    # are per-pattern; others like tree_path pass through unchanged).
    per_pattern_leaves = defaultdict(list)
    passthrough_idx = []
    for i, m in enumerate(meta):
        if m.get('kind') == 'pattern_path':
            per_pattern_leaves[m['pattern']].append(i)
        else:
            passthrough_idx.append(i)

    # Keep top-K per pattern by sum(leaf_counts).
    keep_idx = list(passthrough_idx)
    kept_per_pat = {}
    for pat, idx_list in per_pattern_leaves.items():
        scored = sorted(idx_list,
                          key=lambda i: -sum(meta[i]['leaf_counts']))
        keep = scored[:args.top_k]
        kept_per_pat[pat] = len(keep)
        keep_idx.extend(keep)
    keep_idx.sort()  # preserve original order

    print(f'Before: {len(meta)} hidden units, '
           f'{len(per_pattern_leaves)} patterns')
    print(f'After:  {len(keep_idx)} hidden units, top-{args.top_k} per pattern '
           f'(mean kept = {np.mean(list(kept_per_pat.values())):.2f})')

    # Subset W, b, meta.
    W_new = W[keep_idx]
    b_new = b[keep_idx]
    meta_new = [meta[i] for i in keep_idx]

    # per_cell_leaf_counts needs updating too.
    per_pat_counts = np.zeros(len(per_pattern_leaves), dtype=int)
    for pat in per_pattern_leaves:
        per_pat_counts[pat] = kept_per_pat.get(pat, 0)

    out = dict(ck)
    out['W'] = torch.from_numpy(W_new)
    out['b'] = torch.from_numpy(b_new)
    out['path_info'] = meta_new
    out['per_cell_leaf_counts'] = per_pat_counts
    # Note: probe_state is stale — not compatible with the new hidden
    # layer.  Drop it so a fresh Linear->ProbOR head is trained.
    out['probe_state'] = None
    # Record the pruning in args for audit.
    orig_args = out.get('args', {})
    if isinstance(orig_args, dict):
        orig_args['pruned_top_k'] = args.top_k
        orig_args['pruned_from'] = args.in_ckpt
        out['args'] = orig_args

    torch.save(out, args.out)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
