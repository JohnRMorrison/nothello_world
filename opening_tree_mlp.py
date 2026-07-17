"""Build a single-hidden-layer interpretable MLP for the Othello OPENING
(first 10 ply) by:
  1. Sampling opening positions (played_even features + exact board states).
  2. Training a decision tree per cell to predict cell state from features.
  3. Extracting every root-to-leaf path from every tree.
  4. Encoding each path as a 0/±1 hidden unit (single conjunction rule).
  5. Training a linear probe from the hidden activations → per-cell state.

Each hidden unit is one nameable rule: "if these specific plays happened
(or specifically did not happen), the tree assigned this leaf, which
predicts cell C's state as X."  Nothing continuous, nothing hidden.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier, _tree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.othello import OthelloBoardState


BOARD_CELLS = 64
INPUT_DIM = 120                    # 60 played + 60 even
CENTER_64 = {27, 28, 35, 36}
NON_CENTER_64 = sorted(set(range(64)) - CENTER_64)
C64_TO_C60 = {c: i for i, c in enumerate(NON_CENTER_64)}
C60_TO_C64 = {i: c for i, c in enumerate(NON_CENTER_64)}
STATE_NAMES = ['empty', 'mine', 'opp']


def playedeven_features(prefix):
    feat = np.zeros(INPUT_DIM, dtype=np.float32)
    for t, c in enumerate(prefix):
        if c not in C64_TO_C60:
            continue
        i = C64_TO_C60[c]
        feat[i] = 1.0
        if t % 2 == 0:
            feat[60 + i] = 1.0
    return feat


def feature_name(feat_idx):
    """Return a human-readable name for a played_even feature index."""
    if feat_idx < 60:
        cell = C60_TO_C64[feat_idx]
    else:
        cell = C60_TO_C64[feat_idx - 60]
    col = 'ABCDEFGH'[cell % 8]
    row = str(cell // 8 + 1)
    prefix = 'played' if feat_idx < 60 else 'even'
    return f'{prefix}[{col}{row}]'


# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

def sample_opening_positions(num_games, max_ply=10, seed=42):
    """Play random games; for each game extract positions at plies 0..max_ply-1.
    At each position: features (120,), state labels (64,) as mine/opp/empty
    relative to the current mover, and metadata (parity, ply)."""
    rng = np.random.RandomState(seed)
    Xs, Ss, Ps, Ts = [], [], [], []
    for _ in range(num_games):
        board = OthelloBoardState()
        prefix = []
        for turn in range(max_ply):
            valid = board.get_valid_moves()
            if not valid:
                board.update([])
                valid = board.get_valid_moves()
                if not valid:
                    break
            parity = turn % 2
            mover_color = 1 if parity == 0 else -1
            raw = board.state.flatten().astype(np.int8)
            lbl = np.zeros(BOARD_CELLS, dtype=np.int64)
            lbl[raw == mover_color] = 1
            lbl[raw == -mover_color] = 2
            Xs.append(playedeven_features(prefix))
            Ss.append(lbl)
            Ps.append(parity)
            Ts.append(len(prefix))
            move = valid[rng.randint(len(valid))]
            board.update([move])
            prefix.append(move)
    return (np.stack(Xs), np.stack(Ss),
             np.array(Ps, dtype=np.int8),
             np.array(Ts, dtype=np.int32))


# ------------------------------------------------------------------------------
# Trees → paths → hidden layer
# ------------------------------------------------------------------------------

def _fit_one_tree(X, y, max_depth, min_samples_leaf):
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0,
                                    min_samples_leaf=min_samples_leaf)
    tree.fit(X, y)
    return tree


def train_per_cell_trees(X, S, max_depth, min_samples_leaf=1, n_jobs=1):
    """Train 64 decision trees, optionally in parallel across cells."""
    if n_jobs == 1:
        return [_fit_one_tree(X, S[:, c], max_depth, min_samples_leaf)
                for c in range(BOARD_CELLS)]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_tree)(X, S[:, c], max_depth, min_samples_leaf)
        for c in range(BOARD_CELLS))


def extract_paths(tree):
    """Return list of (conditions, leaf_class, leaf_counts) tuples.
    conditions: list of (feature_idx, required_value 0-or-1).
    Feature values are 0/1; sklearn's threshold ~0.5 splits them.  Left =
    feature ≤ 0.5 (i.e., feature = 0); right = feature > 0.5 (feature = 1).
    """
    tree_ = tree.tree_
    classes = tree.classes_
    paths = []

    def recurse(node, conditions):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            counts = tree_.value[node][0]
            majority = classes[int(np.argmax(counts))]
            paths.append((list(conditions), int(majority),
                           counts.tolist()))
            return
        feat = int(tree_.feature[node])
        recurse(tree_.children_left[node], conditions + [(feat, 0)])
        recurse(tree_.children_right[node], conditions + [(feat, 1)])

    recurse(0, [])
    return paths


def path_to_weight(conditions, input_dim=INPUT_DIM):
    """Encode a path as a 0/±1 weight vector + bias.

    For each (feat, required_value):
      required_value == 1: weight +1 on feat (max contribution 1).
      required_value == 0: weight −1 on feat (max contribution 0).
    Bias = −(K_positive − 0.5), where K_positive counts required=1 tests.
    """
    w = np.zeros(input_dim, dtype=np.float32)
    K_positive = 0
    for feat, val in conditions:
        if val == 1:
            w[feat] += 1.0
            K_positive += 1
        else:
            w[feat] -= 1.0
    bias = -(K_positive - 0.5)
    return w, bias


class OpeningTreeMLP(nn.Module):
    """Frozen hidden layer of 0/±1 conjunction rules; forward gives 0/1
    activations."""

    def __init__(self, weights, biases, path_info, device):
        super().__init__()
        H = weights.shape[0]
        self.hidden_dim = H
        self.register_buffer('W', torch.from_numpy(weights).to(device))
        self.register_buffer('b', torch.from_numpy(biases).to(device))
        self.path_info = path_info

    def forward(self, x, batch=8192):
        H = self.hidden_dim
        N = x.shape[0]
        out = torch.empty(N, H, device=x.device)
        with torch.no_grad():
            for i in range(0, N, batch):
                pre = x[i:i + batch] @ self.W.T + self.b
                out[i:i + batch] = (pre > 0.0).float()
        return out


# ------------------------------------------------------------------------------
# Probe training / evaluation
# ------------------------------------------------------------------------------

def train_probe(H_tr, S_tr, H_te, S_te, epochs=25, lr=0.01, batch=512,
                 weight_decay=1e-4):
    device = H_tr.device
    hidden = H_tr.shape[1]
    probe = nn.Linear(hidden, BOARD_CELLS * 3).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                              weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    N = H_tr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            logits = probe(H_tr[idx]).view(-1, BOARD_CELLS, 3)
            loss = ce(logits.reshape(-1, 3), S_tr[idx].reshape(-1).long())
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def evaluate(probe, H, S, T=None):
    with torch.no_grad():
        preds = probe(H).view(-1, BOARD_CELLS, 3).argmax(dim=-1)
    correct = (preds == S).float()
    acc = correct.mean().item()
    per_cell = correct.mean(dim=0).cpu().numpy()
    by_ply = {}
    if T is not None:
        for ply in range(int(T.max().item()) + 1):
            mask = (T == ply)
            if mask.any():
                by_ply[ply] = (int(mask.sum().item()),
                                correct[mask].mean().item())
    return acc, per_cell, by_ply


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-train-games', type=int, default=5000)
    ap.add_argument('--num-test-games', type=int, default=2000)
    ap.add_argument('--max-ply', type=int, default=10)
    ap.add_argument('--tree-max-depth', type=int, default=8)
    ap.add_argument('--tree-min-samples-leaf', type=int, default=1)
    ap.add_argument('--tree-n-jobs', type=int, default=1)
    ap.add_argument('--probe-epochs', type=int, default=25)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='opening_tree_mlp.pt')
    args = ap.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('warning: CUDA requested but not available; falling back to CPU')
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f'device: {device}')

    print(f'sampling {args.num_train_games} train + '
           f'{args.num_test_games} test games, ply 0..{args.max_ply - 1}...')
    t0 = time.time()
    Xnp_tr, Snp_tr, _, Tnp_tr = sample_opening_positions(
        args.num_train_games, max_ply=args.max_ply, seed=args.seed)
    Xnp_te, Snp_te, _, Tnp_te = sample_opening_positions(
        args.num_test_games, max_ply=args.max_ply,
        seed=args.seed + 1_000_000)
    print(f'  train={Xnp_tr.shape[0]}  test={Xnp_te.shape[0]}  '
           f'({time.time() - t0:.1f}s)')

    # --- Train per-cell decision trees ---
    print(f'\ntraining per-cell trees (max_depth={args.tree_max_depth}, '
           f'min_samples_leaf={args.tree_min_samples_leaf}, '
           f'n_jobs={args.tree_n_jobs})...')
    t0 = time.time()
    trees = train_per_cell_trees(
        Xnp_tr, Snp_tr,
        max_depth=args.tree_max_depth,
        min_samples_leaf=args.tree_min_samples_leaf,
        n_jobs=args.tree_n_jobs)
    print(f'  ({time.time() - t0:.1f}s)')

    # Per-cell tree accuracy on test (before path extraction).
    correct = 0; total = 0
    for c in range(BOARD_CELLS):
        preds = trees[c].predict(Xnp_te)
        correct += (preds == Snp_te[:, c]).sum()
        total += len(preds)
    print(f'  aggregate per-cell tree test acc: {100*correct/total:.4f}%')

    # --- Extract paths, encode as hidden units ---
    print('\nextracting paths → hidden units...')
    all_w, all_b, all_meta = [], [], []
    per_cell_leaf_counts = np.zeros(BOARD_CELLS, dtype=int)
    for c in range(BOARD_CELLS):
        paths = extract_paths(trees[c])
        per_cell_leaf_counts[c] = len(paths)
        for path_idx, (conditions, leaf_class, leaf_counts) in enumerate(paths):
            w, b = path_to_weight(conditions)
            all_w.append(w); all_b.append(b)
            all_meta.append({
                'cell': c, 'path_idx': path_idx,
                'conditions': conditions, 'leaf_class': leaf_class,
                'depth': len(conditions),
                'leaf_counts': leaf_counts,
            })
    W = np.stack(all_w); B = np.array(all_b, dtype=np.float32)
    print(f'  total hidden units: {len(all_meta)}')
    print(f'  leaves per tree: mean={per_cell_leaf_counts.mean():.1f}  '
           f'max={per_cell_leaf_counts.max()}  '
           f'min={per_cell_leaf_counts.min()}')

    depths = np.array([m['depth'] for m in all_meta])
    print(f'  path depths: mean={depths.mean():.2f}  max={depths.max()}  '
           f'min={depths.min()}')

    mlp = OpeningTreeMLP(W, B, all_meta, device)

    X_tr = torch.from_numpy(Xnp_tr).to(device)
    X_te = torch.from_numpy(Xnp_te).to(device)
    S_tr = torch.from_numpy(Snp_tr).to(device)
    S_te = torch.from_numpy(Snp_te).to(device)
    T_te = torch.from_numpy(Tnp_te).to(device)

    print('\ncomputing hidden activations...')
    H_tr = mlp(X_tr); H_te = mlp(X_te)

    fire_rate = H_tr.mean(dim=0)
    print(f'  per-unit firing rate on train: mean={fire_rate.mean().item()*100:.2f}%  '
           f'min={fire_rate.min().item()*100:.4f}%  '
           f'max={fire_rate.max().item()*100:.2f}%')
    print(f'  dead units: {int((fire_rate == 0).sum().item())} / {len(all_meta)}')

    print('\ntraining linear probe on hidden layer...')
    probe = train_probe(H_tr, S_tr, H_te, S_te,
                          epochs=args.probe_epochs)

    acc_tr, _, _ = evaluate(probe, H_tr, S_tr)
    acc_te, per_cell_te, by_ply = evaluate(probe, H_te, S_te, T_te)
    print(f'\nresults:')
    print(f'  hidden dim H = {mlp.hidden_dim}')
    print(f'  train per-cell acc: {100*acc_tr:.4f}%')
    print(f'  test  per-cell acc: {100*acc_te:.4f}%')

    print(f'  test acc by ply:')
    for ply, (n, acc) in sorted(by_ply.items()):
        print(f'    ply {ply:2d}:  n={n:6d}  acc={100*acc:.4f}%')

    torch.save({
        'W': mlp.W.cpu(), 'b': mlp.b.cpu(),
        'probe_state': probe.state_dict(),
        'path_info': all_meta,
        'per_cell_leaf_counts': per_cell_leaf_counts,
        'args': vars(args),
        'test_acc': acc_te, 'train_acc': acc_tr,
        'by_ply': by_ply,
    }, args.out)
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
