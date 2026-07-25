"""Can the joint (all-960) tree search be improved?  Compare tree-search
variants locally on the SAME self-play data, measuring how well each resolves
the 960 flanking patterns (recall/precision @0.5 of the tree's own leaf-mean
predictions, total leaves, patterns ever predicted ON).

Variants:
  A. single global multi-output tree (baseline)         -- the current approach
  B. single global, DEEPER / smaller leaves             -- "just make it bigger"
  C. grouped by target cell (one tree per cell)         -- joint but scoped to
                                                            correlated patterns
"""
import argparse, time
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier

from midgame_tree_mlp import sample_midgame_positions
from flanking_patterns import load_patterns, true_pattern_activations


def gen(n_games, seed):
    X, S, T = sample_midgame_positions(n_games, 0, 60, seed=seed,
                                       canonicalize_mover=True)
    return np.asarray(X, np.float32), np.asarray(S)


def proba_matrix(clf, Xte, cols, K):
    """Fill an (N,K) predicted-prob matrix for the given output columns."""
    P = np.zeros((Xte.shape[0], K), dtype=np.float64)
    pl = clf.predict_proba(Xte)
    pl = pl if isinstance(pl, list) else [pl]
    classes = clf.classes_ if isinstance(clf.classes_, list) else [clf.classes_]
    for k, j in enumerate(cols):
        pr = pl[k]
        cls = classes[k]
        if pr.shape[1] == 2:
            P[:, j] = pr[:, 1]
        else:
            P[:, j] = 1.0 if cls[0] == 1 else 0.0
    return P


def report(name, P, Yte, K, leaves, secs):
    pred = P >= 0.5
    tp = int((pred & (Yte == 1)).sum()); fn = int((~pred & (Yte == 1)).sum())
    fp = int((pred & (Yte == 0)).sum())
    firings = tp + fn
    rec = 100 * tp / firings if firings else float('nan')
    prec = 100 * tp / (tp + fp) if (tp + fp) else float('nan')
    reach = int((P.max(0) >= 0.5).sum())
    print(f'\n=== {name} ===')
    print(f'  leaves={leaves}  fit={secs:.1f}s')
    print(f'  recall={rec:.2f}%  precision={prec:.2f}%  '
          f'patterns reachable={reach}/{K}')


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
    print(f'train {Xtr.shape} test {Xte.shape}  {K} patterns  '
          f'{len(by_cell)} target cells  base rate {100*Ytr.mean():.3f}%')

    # A. single global multi-output tree ----------------------------------
    t = time.time()
    a = DecisionTreeClassifier(max_depth=15, min_samples_leaf=50, random_state=0)
    a.fit(Xtr, Ytr)
    report('A. single global (depth15, leaf50)',
           proba_matrix(a, Xte, list(range(K)), K), Yte, K,
           a.get_n_leaves(), time.time() - t)

    # B. single global, deeper / smaller leaves ---------------------------
    t = time.time()
    b = DecisionTreeClassifier(max_depth=30, min_samples_leaf=10, random_state=0)
    b.fit(Xtr, Ytr)
    report('B. single global DEEPER (depth30, leaf10)',
           proba_matrix(b, Xte, list(range(K)), K), Yte, K,
           b.get_n_leaves(), time.time() - t)

    # C. grouped by target cell (one tree per cell) -----------------------
    t = time.time()
    P = np.zeros((Xte.shape[0], K), dtype=np.float64)
    total_leaves = 0
    for cell, cols in by_cell.items():
        clf = DecisionTreeClassifier(max_depth=15, min_samples_leaf=50,
                                     random_state=0)
        clf.fit(Xtr, Ytr[:, cols])
        total_leaves += clf.get_n_leaves()
        P[:, cols] = proba_matrix(clf, Xte, cols, K)[:, cols]
    report('C. grouped by target cell (64 trees, depth15/leaf50)',
           P, Yte, K, total_leaves, time.time() - t)


if __name__ == '__main__':
    main()
