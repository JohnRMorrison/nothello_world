"""Inspect the leaves of the multioutput (joint) pattern tree, straight from a
saved tree bank / tree-cache checkpoint (the `--load-trees-from` format).

The bank stores, per leaf, the root->leaf decision path (`conditions`), its
`depth`, `leaf_counts` (training samples at the leaf), and `patterns` (which of
the 960 targets that leaf's tree covers).  We don't need the sklearn object to
see how coarse the tree is: count the leaves, show the depth/sample spread, and
print each leaf's splits.

Usage:
    python inspect_multioutput_leaves.py <bank.pt> [--max-leaves 50]
"""
import argparse
import numpy as np
import torch


def _fmt_conditions(conds):
    """conditions = list of (feature_index, required_value in {0,1}) pairs along
    the root->leaf path (binary board features).  '=1' means the feature must be
    present at that leaf, '=0' absent."""
    out = []
    for cond in conds:
        try:
            feat, val = cond
        except (TypeError, ValueError):
            out.append(str(cond)); continue
        out.append(f'f{feat}={val}')
    return '  '.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bank')
    ap.add_argument('--max-leaves', type=int, default=50,
                    help='how many leaves to print in full')
    args = ap.parse_args()

    d = torch.load(args.bank, map_location='cpu', weights_only=False)
    meta = d.get('path_info', [])
    a = d.get('args', {})
    print(f'bank: {args.bank}')
    print(f'  tree-mode={a.get("pattern_tree_mode")}  '
          f'canonicalize_mover={a.get("canonicalize_mover")}  '
          f'depth={a.get("tree_max_depth")}  '
          f'min_samples_leaf={a.get("tree_min_samples_leaf")}  '
          f'games={a.get("num_games")}  ply={a.get("ply_range")}')
    print(f'  include_flanking={a.get("include_flanking_patterns")!r}  '
          f'(these are the shortcut features — should be empty for tree-only)')

    leaves = [m for m in meta if m.get('kind') == 'pattern_multi']
    print(f'\ntotal hidden units in bank: {len(meta)}')
    print(f'multioutput (pattern_multi) leaves: {len(leaves)}')
    if not leaves:
        print('  no pattern_multi leaves — this bank is not a joint tree.')
        return

    depths = np.array([m.get('depth', len(m.get('conditions', []))) for m in leaves])
    counts = np.array([(m['leaf_counts'] if np.isscalar(m.get('leaf_counts'))
                        else np.sum(m.get('leaf_counts', 0))) for m in leaves],
                      dtype=float)
    print(f'depth:  min={depths.min()} max={depths.max()} '
          f'mean={depths.mean():.1f}')
    if counts.sum() > 0:
        tot = counts.sum()
        order = np.argsort(-counts)
        top = counts[order][:5]
        print(f'training samples per leaf: total={int(tot)}  '
              f'top-5 share={100*top.sum()/tot:.1f}%  '
              f'(biggest leaf holds {100*top[0]/tot:.1f}%)')
        # how many patterns does each leaf's tree cover?
        npat = leaves[0].get('patterns')
        if npat is not None:
            print(f'patterns covered per leaf (cols): {len(npat)} '
                  f'(a single joint tree covers all targets)')

    print(f'\n--- leaves (up to {args.max_leaves}), largest first ---')
    order = np.argsort(-counts) if counts.sum() > 0 else range(len(leaves))
    for rank, i in enumerate(list(order)[:args.max_leaves]):
        m = leaves[i]
        n = int(counts[i]) if counts.sum() > 0 else '?'
        conds = m.get('conditions', [])
        print(f'[{rank:2d}] depth={m.get("depth", len(conds)):2d}  '
              f'n={n:>7}  {_fmt_conditions(conds) or "(root/stump)"}')


if __name__ == '__main__':
    main()
