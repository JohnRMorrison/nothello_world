"""Minimal test of the 'add conjunction features' idea (collaborator's
suggestion): does adding h_i AND h_j features close the pattern-tree
probe → MLP BCE gap?

Option (a): target ground-truth cell legality.  Fits K small decision
trees on the leaves (48K features), each targeting one cell's legality.
The leaves of those small trees are per-position binary indicators of
specific conjunctions of h_i's.  Concatenate to the original features
and refit a plain linear+BCE probe.

Compare test BCE against a baseline linear+BCE probe on the leaves alone.

Reminder (per user): try option (b) next — target the trained MLP's
per-cell output instead of ground-truth legality.

Usage:
    python test_conjunction_features.py \\
        --load-trees-from ckpts_midgame/midgame_leg_pattern_trees_no_recent_canonical_g20000_d15_ml10_p10-50.pt \\
        --chunk-path experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks/chunk_ext_0000.npz \\
        --canonicalize-mover
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from opening_tree_mlp import OpeningTreeMLP, BOARD_CELLS  # noqa: E402
from train_streaming_probe import (                       # noqa: E402
    load_trees, process_chunk_ext_file,
)


# --------------------------------------------------------------------------

def compute_leaves(mlp, X, batch=4096, device=None):
    """Apply mlp to X and return leaf activations as (N, D_leaves) uint8.

    Uses bool step activation, materialised as uint8 on CPU (~1 byte per
    hidden unit).
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available()
                          else torch.device('cpu'))
    N = X.shape[0]
    D = mlp.W.shape[0]
    tree_in_dim = mlp.W.shape[1]
    out = np.zeros((N, D), dtype=np.uint8)
    t0 = time.time()
    for i in range(0, N, batch):
        xb_np = np.ascontiguousarray(X[i:i + batch, :tree_in_dim])
        xb = torch.from_numpy(xb_np).to(device)
        h = mlp(xb, out_device='cpu', out_dtype=torch.bool, use_relu=False)
        out[i:i + batch] = h.numpy().astype(np.uint8)
        if (i // batch) % 20 == 0:
            print(f'  leaves batch {i//batch}: '
                   f'{time.time() - t0:.1f}s so far', flush=True)
    return out


# --------------------------------------------------------------------------

def fit_small_trees(H, L, n_trees, depth, min_leaf, rng):
    """One small decision tree per cell (targeted at cell legality).

    We fit min(n_trees, 64) trees — one per cell whose legality has any
    variance in the training set.  `class_weight='balanced'` and
    `max_features='sqrt'` for speed on ~48K binary features.
    Returns list of (tree, cell) tuples.
    """
    n_cells = min(n_trees, BOARD_CELLS)
    cells = list(range(n_cells))
    trees = []
    t0 = time.time()
    for i, c in enumerate(cells):
        y = L[:, c].astype(np.int64)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        t = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=min_leaf,
            max_features='sqrt',
            class_weight='balanced',
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        t.fit(H, y)
        trees.append((t, c))
        if (i + 1) % 8 == 0:
            print(f'  fit tree {i + 1}/{n_cells}: '
                   f'{time.time() - t0:.1f}s so far', flush=True)
    return trees


def extract_conjunctions(trees, H):
    """For each small tree, compute the leaf id per position and one-hot
    it into a set of new binary columns.  Returns (N, K) uint8.

    Each new column represents a specific conjunction of leaf features
    (a decision path through the small tree).
    """
    parts = []
    for tree, _ in trees:
        leaf_ids = tree.apply(H)   # (N,)
        for u in np.unique(leaf_ids):
            parts.append((leaf_ids == u).astype(np.uint8))
    if not parts:
        return np.zeros((H.shape[0], 0), dtype=np.uint8)
    return np.stack(parts, axis=1)


# --------------------------------------------------------------------------

def train_bce_probe(features, labels, epochs, batch, lr, weight_decay,
                     device, seed=0):
    """Linear + sigmoid, per-cell BCE.  Features on CPU, batched to GPU."""
    torch.manual_seed(seed)
    N, D = features.shape
    probe = torch.nn.Linear(D, BOARD_CELLS).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                              weight_decay=weight_decay)
    features_t = torch.from_numpy(features)          # CPU, uint8
    labels_t = torch.from_numpy(labels).float()      # CPU
    print(f'  training probe D={D}  epochs={epochs} batch={batch} lr={lr}',
           flush=True)
    for ep in range(epochs):
        rng = torch.Generator().manual_seed(seed * 1000 + ep)
        perm = torch.randperm(N, generator=rng)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            x = features_t[idx].to(device=device, dtype=torch.float32)
            y = labels_t[idx].to(device)
            logits = probe(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f'    epoch {ep + 1}/{epochs}: avg loss = '
               f'{epoch_loss / max(n_batches, 1):.4f}', flush=True)
    return probe


@torch.no_grad()
def eval_bce(probe, features, labels, batch, device):
    features_t = torch.from_numpy(features)
    labels_t = torch.from_numpy(labels).float()
    total_loss = 0.0
    total_count = 0
    for i in range(0, features.shape[0], batch):
        x = features_t[i:i + batch].to(device=device, dtype=torch.float32)
        y = labels_t[i:i + batch].to(device)
        logits = probe(x)
        total_loss += F.binary_cross_entropy_with_logits(
            logits, y, reduction='sum').item()
        total_count += y.numel()
    return total_loss / total_count


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load-trees-from', required=True)
    ap.add_argument('--chunk-path', required=True)
    ap.add_argument('--n-positions', type=int, default=100_000)
    ap.add_argument('--train-frac', type=float, default=0.7)
    ap.add_argument('--n-small-trees', type=int, default=64)
    ap.add_argument('--small-tree-depth', type=int, default=5)
    ap.add_argument('--small-tree-min-leaf', type=int, default=100)
    ap.add_argument('--canonicalize-mover', action='store_true')
    ap.add_argument('--ply-min', type=int, default=10)
    ap.add_argument('--ply-max', type=int, default=50)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # -------- load checkpoint + build MLP --------
    print(f'Loading trees from {args.load_trees_from}...', flush=True)
    W, b, meta = load_trees(args.load_trees_from)
    mlp = OpeningTreeMLP(W, b, meta, device)

    # -------- load chunk positions --------
    print(f'Loading chunk {args.chunk_path}...', flush=True)
    t0 = time.time()
    X, _, T, L = process_chunk_ext_file(
        args.chunk_path, args.ply_min, args.ply_max,
        canonicalize_mover=args.canonicalize_mover,
        max_positions=args.n_positions,
    )
    print(f'  loaded {X.shape[0]} positions in {time.time() - t0:.1f}s',
           flush=True)

    # -------- compute leaves --------
    print('Computing leaf activations...', flush=True)
    H = compute_leaves(mlp, X, device=device)
    print(f'  H shape={H.shape}  dtype={H.dtype}  '
           f'mem={H.nbytes/1e9:.2f} GB', flush=True)

    # -------- train/test split --------
    N = H.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_train = int(args.train_frac * N)
    idx_train, idx_test = perm[:n_train], perm[n_train:]
    H_train, H_test = H[idx_train], H[idx_test]
    L_train, L_test = L[idx_train], L[idx_test]
    print(f'  train={len(idx_train)}  test={len(idx_test)}')

    # -------- fit small trees + extract conjunctions --------
    print(f'Fitting small trees (depth={args.small_tree_depth}, '
           f'min_leaf={args.small_tree_min_leaf})...', flush=True)
    trees = fit_small_trees(H_train, L_train, args.n_small_trees,
                              args.small_tree_depth,
                              args.small_tree_min_leaf, rng)
    print(f'  fitted {len(trees)} trees')

    print('Extracting conjunctions...', flush=True)
    C_train = extract_conjunctions(trees, H_train)
    C_test = extract_conjunctions(trees, H_test)
    print(f'  conjunctions: train {C_train.shape}, test {C_test.shape}',
           flush=True)

    # -------- baseline probe on leaves --------
    print('\n=== Baseline: linear + sigmoid + BCE on leaves ===')
    probe_b = train_bce_probe(H_train, L_train, args.epochs, args.batch,
                                 args.lr, args.weight_decay, device,
                                 seed=args.seed)
    base_bce = eval_bce(probe_b, H_test, L_test, args.batch, device)
    print(f'  Baseline test BCE = {base_bce:.4f}')

    # -------- enhanced probe on leaves + conjunctions --------
    print('\n=== Enhanced: baseline + conjunction features ===')
    Enh_train = np.concatenate([H_train, C_train], axis=1)
    Enh_test = np.concatenate([H_test, C_test], axis=1)
    print(f'  Enh feature dim = {Enh_train.shape[1]} '
           f'(leaves={H_train.shape[1]} + conj={C_train.shape[1]})',
           flush=True)
    probe_e = train_bce_probe(Enh_train, L_train, args.epochs, args.batch,
                                 args.lr, args.weight_decay, device,
                                 seed=args.seed)
    enh_bce = eval_bce(probe_e, Enh_test, L_test, args.batch, device)
    print(f'  Enhanced test BCE = {enh_bce:.4f}')

    # -------- summary --------
    print('\n=== Summary ===')
    print(f'  Baseline  BCE = {base_bce:.4f}')
    print(f'  Enhanced  BCE = {enh_bce:.4f}')
    print(f'  Delta         = {enh_bce - base_bce:+.4f}  '
           f'({(enh_bce - base_bce) / base_bce * 100:+.1f}%)')
    if enh_bce < base_bce - 0.005:
        print('  -> Conjunctions help meaningfully; worth integrating.')
    elif enh_bce > base_bce + 0.005:
        print('  -> Conjunctions hurt; probably overfitting.')
    else:
        print('  -> No clear signal.  Try more/deeper small trees, or '
               'option (b) targeting the MLP output instead.')


if __name__ == '__main__':
    main()
