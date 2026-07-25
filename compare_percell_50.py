"""How well does grouped-by-target-cell do at a realistic 50-leaves-per-cell
budget (vs the unlimited 36.7k-leaf version that hit 20.09%)?

Two senses of "50 per cell":
  C50-best : each per-cell tree grown best-first to max_leaf_nodes=50.
  C50-topk : per-cell tree grown fully, then only its 50 largest leaves kept
             (positions in other leaves get NO signal) -- mirrors the pipeline's
             top_k_per_cell=50.
Reference: Cfull = unlimited leaves per cell.
"""
import argparse, time
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier

from midgame_tree_mlp import sample_midgame_positions
from flanking_patterns import load_patterns, true_pattern_activations


def gen(n, seed):
    X, S, T = sample_midgame_positions(n, 0, 60, seed=seed, canonicalize_mover=True)
    return np.asarray(X, np.float32), np.asarray(S)


def fill(clf, Xte, cols, K, keep_leaves=None, leaf_te=None):
    P = np.zeros((Xte.shape[0], K))
    pl = clf.predict_proba(Xte); pl = pl if isinstance(pl, list) else [pl]
    classes = clf.classes_ if isinstance(clf.classes_, list) else [clf.classes_]
    for k, j in enumerate(cols):
        pr = pl[k]
        P[:, j] = pr[:, 1] if pr.shape[1] == 2 else (1.0 if classes[k][0] == 1 else 0.0)
    if keep_leaves is not None:              # zero out positions not in kept leaves
        mask = np.isin(leaf_te, list(keep_leaves))
        P[~mask] = 0.0
    return P


def report(name, P, Yte, K, leaves, secs):
    pred = P >= 0.5
    tp = int((pred & (Yte == 1)).sum()); fn = int((~pred & (Yte == 1)).sum())
    fp = int((pred & (Yte == 0)).sum()); firings = tp + fn
    rec = 100 * tp / firings if firings else 0
    prec = 100 * tp / (tp + fp) if (tp + fp) else 0
    reach = int((P.max(0) >= 0.5).sum())
    print(f'\n=== {name} ===\n  leaves={leaves}  fit={secs:.1f}s  '
          f'recall={rec:.2f}%  precision={prec:.2f}%  reachable={reach}/{K}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-games', type=int, default=3000)
    ap.add_argument('--test-games', type=int, default=1500)
    args = ap.parse_args()
    patterns = load_patterns('hand_crafted_flanking_patterns.pt')
    K = len(patterns)
    by_cell = defaultdict(list)
    for j, p in enumerate(patterns):
        by_cell[p['target']].append(j)

    Xtr, Str = gen(args.train_games, 42)
    Xte, Ste = gen(args.test_games, 7)
    Ytr = true_pattern_activations(patterns, Str)
    Yte = true_pattern_activations(patterns, Ste)
    print(f'train {Xtr.shape} test {Xte.shape}  {K} patterns  {len(by_cell)} cells')

    for tag, mln, topk in [('C50-best (max_leaf_nodes=50)', 50, False),
                            ('C50-topk (keep 50 largest leaves)', None, True),
                            ('Cfull (unlimited)', None, False)]:
        t = time.time()
        P = np.zeros((Xte.shape[0], K)); tot = 0
        for cell, cols in by_cell.items():
            clf = DecisionTreeClassifier(max_depth=15, min_samples_leaf=50,
                                         max_leaf_nodes=mln, random_state=0)
            clf.fit(Xtr, Ytr[:, cols])
            if topk:
                cnt = np.bincount(clf.apply(Xtr), minlength=clf.tree_.node_count)
                keep = set(np.argsort(-cnt)[:50].tolist())
                tot += min(50, clf.get_n_leaves())
                P[:, cols] = fill(clf, Xte, cols, K, keep, clf.apply(Xte))[:, cols]
            else:
                tot += clf.get_n_leaves()
                P[:, cols] = fill(clf, Xte, cols, K)[:, cols]
        report(tag, P, Yte, K, tot, time.time() - t)


if __name__ == '__main__':
    main()
