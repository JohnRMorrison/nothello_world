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

    # W-backed entries are only those with kind in {tree_path, pattern_path}.
    # 'flanking_pattern' entries are informational — the runtime pipeline
    # computes flanking activations from the patterns list, not from W.
    W_KINDS = ('tree_path', 'pattern_path')
    w_backed_idx = [i for i, m in enumerate(meta)
                    if m.get('kind') in W_KINDS]
    non_w_backed_idx = [i for i, m in enumerate(meta)
                        if m.get('kind') not in W_KINDS]
    if len(w_backed_idx) != len(W):
        raise ValueError(
            f'{len(w_backed_idx)} W-backed meta entries but W has '
            f'{len(W)} rows — mismatch, aborting.')

    # Map meta_index -> W_row for pruning
    meta_to_w = {mi: wi for wi, mi in enumerate(w_backed_idx)}

    # Group by pattern (only pattern_path entries).
    per_pattern_leaves = defaultdict(list)
    passthrough_idx = []
    for i in w_backed_idx:
        m = meta[i]
        if m.get('kind') == 'pattern_path':
            per_pattern_leaves[m['pattern']].append(i)
        else:
            passthrough_idx.append(i)   # e.g. tree_path

    # Keep top-K per pattern by sum(leaf_counts).
    keep_meta_idx = list(passthrough_idx)
    kept_per_pat = {}
    for pat, idx_list in per_pattern_leaves.items():
        scored = sorted(idx_list,
                          key=lambda i: -sum(meta[i]['leaf_counts']))
        keep = scored[:args.top_k]
        kept_per_pat[pat] = len(keep)
        keep_meta_idx.extend(keep)
    keep_meta_idx.sort()

    # Corresponding W-row indices.
    keep_w_idx = [meta_to_w[i] for i in keep_meta_idx]

    print(f'Before: {len(w_backed_idx)} W-backed hidden units '
           f'(+ {len(non_w_backed_idx)} runtime-only meta), '
           f'{len(per_pattern_leaves)} patterns')
    print(f'After:  {len(keep_meta_idx)} W-backed hidden units, '
           f'top-{args.top_k} per pattern '
           f'(mean kept = {np.mean(list(kept_per_pat.values())):.2f})')

    # Subset W, b, meta.  Keep the non-W-backed meta entries as-is (they
    # describe the runtime flanking patterns and are still valid).
    W_new = W[keep_w_idx]
    b_new = b[keep_w_idx]
    meta_new = [meta[i] for i in keep_meta_idx]
    for i in non_w_backed_idx:
        meta_new.append(meta[i])

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
