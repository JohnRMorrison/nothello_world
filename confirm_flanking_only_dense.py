"""How well do the 960 flanking rules do BY THEMSELVES, fit densely?

The flanking "features" are the 960 hand-crafted legality rules evaluated on the
MOVE-HISTORY features (played/even/mover-parity) under the placement=current-
color approximation -- NOT oracle true state (that is why they are imperfect).

We measure, with NO tree:
  (a) un-fit rule-OR baseline: legal(cell) = OR of rules targeting that cell.
  (b) dense fit: a single Linear(960 -> 64) trained with BCE on legality
      (the "linpo/dense readout" analog).

Reported as per-cell legality accuracy and argmax-legality (fraction of
positions whose highest-scored cell is actually legal -- the ~98.7% metric).

Usage:
    python confirm_flanking_only_dense.py [--train-games 3000] [--test-games 1500]
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from midgame_tree_mlp import sample_midgame_positions
from flanking_patterns import load_patterns, compute_pattern_activations


def gen(n_games, seed):
    X, S, T, L = sample_midgame_positions(
        n_games, ply_min=0, ply_max=60, seed=seed, canonicalize_mover=True,
        collect_legal_moves=True)
    return np.asarray(X), np.asarray(L)


def flank_feats(X, patterns):
    return compute_pattern_activations(
        patterns, X[:, :60].astype(np.uint8), X[:, 60:120].astype(np.uint8),
        X[:, 120].astype(np.uint8))


def per_cell_acc(pred_bin, L):
    return 100.0 * (pred_bin == L).mean()


def argmax_legal(scores, L):
    """fraction of positions whose top-scored cell is truly legal.
    positions with no legal move are skipped."""
    has = L.sum(1) > 0
    idx = scores[has].argmax(1)
    return 100.0 * L[has][np.arange(has.sum()), idx].mean()


def legal_f1(pred_bin, L):
    tp = (pred_bin & (L == 1)).sum(); fp = (pred_bin & (L == 0)).sum()
    fn = (~pred_bin & (L == 1)).sum()
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return 100 * rec, 100 * prec, 100 * f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-games', type=int, default=3000)
    ap.add_argument('--test-games', type=int, default=1500)
    ap.add_argument('--patterns', default='hand_crafted_flanking_patterns.pt')
    ap.add_argument('--epochs', type=int, default=15)
    args = ap.parse_args()

    patterns = load_patterns(args.patterns)
    targets = np.array([p['target'] for p in patterns])   # cell each rule scores
    K = len(patterns)

    print(f'self-play train ({args.train_games} games) / test ({args.test_games})...')
    Xtr, Ltr = gen(args.train_games, 42)
    Xte, Lte = gen(args.test_games, 7)
    FPtr = flank_feats(Xtr, patterns).astype(np.float32)
    FPte = flank_feats(Xte, patterns).astype(np.float32)
    print(f'  train {FPtr.shape}  test {FPte.shape}  {K} rules  '
          f'rule fire-rate {100*FPtr.mean():.3f}%  '
          f'legal base-rate {100*Ltr.mean():.2f}%')

    # ---- (a) un-fit rule-OR baseline -------------------------------------
    or_te = np.zeros((FPte.shape[0], 64), dtype=np.float32)
    for j in range(K):
        np.maximum(or_te[:, targets[j]], FPte[:, j], out=or_te[:, targets[j]])
    or_bin = or_te >= 0.5
    rec, prec, f1 = legal_f1(or_bin, Lte)
    print('\n=== (a) rule-OR, NO fit ===')
    print(f'  per-cell acc: {per_cell_acc(or_bin, Lte):.2f}%   '
          f'argmax-legal: {argmax_legal(or_te, Lte):.2f}%')
    print(f'  legal recall {rec:.2f}%  precision {prec:.2f}%  F1 {f1:.2f}%')

    # ---- (b) dense Linear(960->64) BCE fit -------------------------------
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xtr = torch.from_numpy(FPtr).to(dev); ytr = torch.from_numpy(Ltr.astype(np.float32)).to(dev)
    xte = torch.from_numpy(FPte).to(dev)
    net = nn.Linear(K, 64).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    net.train()
    for ep in range(args.epochs):
        perm = torch.randperm(xtr.shape[0], device=dev)
        for i in range(0, xtr.shape[0], 8192):
            b = perm[i:i + 8192]
            opt.zero_grad()
            loss = lossf(net(xtr[b]), ytr[b])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        logits = net(xte).cpu().numpy()
    scores = 1 / (1 + np.exp(-logits))
    dense_bin = scores >= 0.5
    rec, prec, f1 = legal_f1(dense_bin, Lte)
    print(f'\n=== (b) dense Linear(960->64) BCE, {args.epochs} epochs ===')
    print(f'  per-cell acc: {per_cell_acc(dense_bin, Lte):.2f}%   '
          f'argmax-legal: {argmax_legal(scores, Lte):.2f}%')
    print(f'  legal recall {rec:.2f}%  precision {prec:.2f}%  F1 {f1:.2f}%')

    # reference: always-illegal per-cell floor
    print(f'\n  (ref) always-predict-illegal per-cell acc: '
          f'{100*(Lte == 0).mean():.2f}%')


if __name__ == '__main__':
    main()
