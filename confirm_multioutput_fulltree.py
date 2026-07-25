"""Fit the multioutput (joint) pattern tree locally with NO top_k cap, and
measure how well it resolves the 960 flanking patterns.

This sidesteps the saved bank's `top_k_per_cell=50` pruning (which kept only the
50 largest leaves -> ~15% position coverage).  Here we see the FULL tree: how
many leaves it really has, and its per-pattern recall/precision on held-out
self-play positions.  Same tree params as the cluster fit: max_depth=15,
min_samples_leaf=50, class_weight=None (balanced is fatal for 960 rare joint
targets), canonicalize_mover.

We also report the "top-50 leaves only" view -- what the readout ACTUALLY saw --
so we can separate two questions:
  (a) can the joint tree resolve patterns at all?  (full tree)
  (b) did top_k=50 cripple what the readout received?  (top-50 view)

Usage:
    python confirm_multioutput_fulltree.py [--train-games 3000] [--test-games 1500]
"""
import argparse
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from midgame_tree_mlp import sample_midgame_positions
from flanking_patterns import load_patterns, true_pattern_activations


def gen(n_games, seed):
    X, S, T = sample_midgame_positions(
        n_games, ply_min=0, ply_max=60, seed=seed, canonicalize_mover=True)
    return np.asarray(X), np.asarray(S)


def recall_report(name, proba_pos, Yte, K):
    """proba_pos: (N,K) predicted P(pattern fires).  Yte: (N,K) 0/1 truth."""
    pred = proba_pos >= 0.5
    tp = int((pred & (Yte == 1)).sum())
    fn = int((~pred & (Yte == 1)).sum())
    fp = int((pred & (Yte == 0)).sum())
    firings = tp + fn
    recall = tp / firings if firings else float('nan')
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    reachable = int((proba_pos.max(axis=0) >= 0.5).sum())
    print(f'\n=== {name} ===')
    print(f'  max predicted prob (any pos, any pattern): {proba_pos.max():.4f}')
    print(f'  firings={firings}  recall={100*recall:.2f}%  '
          f'precision={100*prec:.2f}%  false_pos={fp}')
    print(f'  patterns ever predicted ON (>=0.5): {reachable}/{K}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-games', type=int, default=3000)
    ap.add_argument('--test-games', type=int, default=1500)
    ap.add_argument('--patterns', default='hand_crafted_flanking_patterns.pt')
    args = ap.parse_args()

    patterns = load_patterns(args.patterns)
    K = len(patterns)

    print(f'self-play train ({args.train_games} games, seed 42)...')
    Xtr, Str = gen(args.train_games, 42)
    print(f'self-play test  ({args.test_games} games, seed 7)...')
    Xte, Ste = gen(args.test_games, 7)
    Ytr = true_pattern_activations(patterns, Str)
    Yte = true_pattern_activations(patterns, Ste)
    print(f'  train {Xtr.shape}  test {Xte.shape}  {K} patterns  '
          f'base firing rate {100*Ytr.mean():.3f}%')

    print('\nfitting multioutput DecisionTree '
          '(max_depth=15, min_samples_leaf=50, class_weight=None)...')
    clf = DecisionTreeClassifier(max_depth=15, min_samples_leaf=50,
                                 class_weight=None, random_state=0)
    clf.fit(Xtr, Ytr)
    n_leaves = clf.get_n_leaves()
    print(f'  FULL tree leaves: {n_leaves}   (bank stored only top 50)')

    # sklearn multioutput predict_proba -> list of K arrays (n,2) or (n,1)
    proba_list = clf.predict_proba(Xte)
    P = np.zeros((Xte.shape[0], K), dtype=np.float64)
    for j, pr in enumerate(proba_list):
        # column for class "1"; some outputs may be single-class (all 0)
        classes = clf.classes_[j] if isinstance(clf.classes_, list) else clf.classes_
        if pr.shape[1] == 2:
            P[:, j] = pr[:, 1]
        else:                       # only class 0 present -> P(fire)=0
            P[:, j] = 0.0 if classes[0] == 0 else pr[:, 0]

    recall_report(f'FULL tree ({n_leaves} leaves)', P, Yte, K)

    # ---- top-50 leaves only: what the readout actually received -----------
    leaf_id = clf.apply(Xte)                       # (Nte,) leaf per test pos
    leaf_id_tr = clf.apply(Xtr)
    counts = np.bincount(leaf_id_tr, minlength=leaf_id_tr.max() + 1)
    top50 = set(np.argsort(-counts)[:50].tolist())
    in_top = np.array([l in top50 for l in leaf_id])
    print(f'\n  top-50 leaves cover {100*in_top.mean():.1f}% of test positions '
          f'(rest -> all hidden units 0, no tree signal)')
    P_top = P.copy()
    P_top[~in_top] = 0.0
    recall_report('top-50 leaves only (readout view)', P_top, Yte, K)


if __name__ == '__main__':
    main()
