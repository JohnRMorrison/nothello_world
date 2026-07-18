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
INPUT_DIM_BASE = 120               # 60 played + 60 even
INPUT_DIM = 121                    # 60 played + 60 even + 1 mover_parity
CENTER_64 = {27, 28, 35, 36}
NON_CENTER_64 = sorted(set(range(64)) - CENTER_64)
C64_TO_C60 = {c: i for i, c in enumerate(NON_CENTER_64)}
C60_TO_C64 = {i: c for i, c in enumerate(NON_CENTER_64)}
STATE_NAMES = ['empty', 'mine', 'opp']


def playedeven_features(prefix):
    """Return 121-d input: 60 played + 60 even + 1 mover_parity.

    mover_parity = 0 iff it is black's turn to move (i.e., an even number of
    moves have been made so far); = 1 iff white's turn.  This is directly
    inferable from `len(prefix)` at extraction time.

    The mover-parity bit lets a decision tree split on it at depth 1 and
    disambiguate mine/opp labels without having to compute an XOR of the
    60 played bits — which is what depth 15 could not do at high ply.
    """
    feat = np.zeros(INPUT_DIM, dtype=np.float32)
    for t, c in enumerate(prefix):
        if c not in C64_TO_C60:
            continue
        i = C64_TO_C60[c]
        feat[i] = 1.0
        if t % 2 == 0:
            feat[60 + i] = 1.0
    feat[120] = 1.0 if len(prefix) % 2 == 1 else 0.0
    return feat


def feature_name(feat_idx):
    """Return a human-readable name for a played_even feature index."""
    if feat_idx == 120:
        return 'mover_parity'
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

def load_or_sample(cache_path, sampler_fn, *args, **kwargs):
    """Load sampled arrays from an .npz cache if it exists, else run
    `sampler_fn(*args, **kwargs)`, save the resulting tuple of arrays to
    the cache, and return them.

    Sampling takes 12-15 minutes at cluster scale.  With a cache path, an
    interrupted or reconfigured run can skip re-sampling and go straight
    to tree fit / probe.
    """
    if cache_path and os.path.exists(cache_path):
        print(f'  loading cached dataset from {cache_path}...')
        with np.load(cache_path) as d:
            result = tuple(d[k] for k in sorted(d.files,
                                                  key=lambda x: int(x.split('_')[1])))
        return result
    result = sampler_fn(*args, **kwargs)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        print(f'  saving dataset cache to {cache_path}...')
        np.savez_compressed(
            cache_path,
            **{f'arr_{i}': r for i, r in enumerate(result)})
    return result


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


def enumerate_opening_positions(max_ply=10, verbose=True):
    """BFS all distinct board positions reachable at plies 0..max_ply-1.

    Dedup key is (board_state_bytes, side_to_move) — one entry per distinct
    board position, regardless of how many move orderings reach it.  For
    each unique position we keep ONE canonical move history (the first one
    the BFS encountered) and use it to compute the played_even features.

    Note: distinct move orderings that reach the same board state can have
    different played_even features (parity of specific cells' placements
    can differ).  Using the canonical prefix means the training set covers
    every reachable board state exactly once, with one specific
    played_even encoding.  This is an approximation vs. exhaustive
    (state × history) enumeration, but it captures every state.
    """
    initial = OthelloBoardState()
    key0 = (initial.state.tobytes(), int(initial.next_hand_color))
    frontier = {key0: (initial.state.copy(),
                        int(initial.next_hand_color), [])}

    Xs, Ss, Ts = [], [], []

    for ply in range(max_ply):
        t0 = time.time()
        for _key, (state_arr, side, prefix) in frontier.items():
            mover_color = side     # +1 for black to move, -1 for white
            raw = state_arr.flatten().astype(np.int8)
            lbl = np.zeros(BOARD_CELLS, dtype=np.int64)
            lbl[raw == mover_color] = 1
            lbl[raw == -mover_color] = 2
            Xs.append(playedeven_features(prefix))
            Ss.append(lbl)
            Ts.append(ply)

        if verbose:
            print(f'  ply {ply:2d}:  {len(frontier):>10d} positions  '
                   f'(+{time.time() - t0:.1f}s extract)')

        if ply == max_ply - 1:
            break

        # Expand to the next ply.
        t0 = time.time()
        next_frontier = {}
        for _key, (state_arr, side, prefix) in frontier.items():
            b = OthelloBoardState()
            b.state = state_arr.copy()
            b.next_hand_color = side
            valid = b.get_valid_moves()
            if not valid:
                # Pass.
                b.update([])
                valid = b.get_valid_moves()
                if not valid:
                    continue   # terminal
                # After a pass we don't add moves to prefix (no move played);
                # BUT we still want to enumerate the post-pass position at
                # the next ply.  For simplicity, treat pass as if it skipped
                # to the opponent's move.
            for mv in valid:
                nb = OthelloBoardState()
                nb.state = state_arr.copy()
                nb.next_hand_color = side
                nb.update([mv])
                new_key = (nb.state.tobytes(), int(nb.next_hand_color))
                if new_key not in next_frontier:
                    next_frontier[new_key] = (nb.state.copy(),
                                                int(nb.next_hand_color),
                                                prefix + [mv])
        if verbose:
            print(f'    expand → {len(next_frontier):>10d} '
                   f'({time.time() - t0:.1f}s)')
        frontier = next_frontier

    return (np.stack(Xs), np.stack(Ss),
             np.array([], dtype=np.int8),        # parity unused
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


def prune_paths_by_count(paths, top_k):
    """Sort paths by total training count (sum of leaf class distribution)
    and return only the top_k.  Rare paths — those that cover very few
    training examples — are typically noise-fit; keeping only the
    frequent ones reduces H, cuts overfit, and makes the resulting rule
    set inspectable.

    Args:
      paths: list of (conditions, leaf_class, leaf_counts) as returned by
             extract_paths().
      top_k: if None or >= len(paths), returns paths unchanged.
    Returns:
      list of paths sorted DESCENDING by training count, length min(top_k,
      len(paths)).  Each path retains its original (conditions,
      leaf_class, leaf_counts) tuple.
    """
    if top_k is None or top_k >= len(paths):
        return paths
    scored = sorted(paths, key=lambda p: -sum(p[2]))
    return scored[:top_k]


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

    def forward(self, x, batch=256, out_device='cpu',
                 out_dtype=torch.bool):
        """Compute hidden activations in chunks.

        Default batch is small (256) because H can grow into the 100k+
        range for endgame/midgame, and the per-batch matmul on GPU costs
        `batch × H × 4` bytes.  batch=256 with H=200k = 200 MB — safe on
        a 10 GB GPU.
        """
        H = self.hidden_dim
        N = x.shape[0]
        out = torch.empty(N, H, device=out_device, dtype=out_dtype)
        with torch.no_grad():
            for i in range(0, N, batch):
                pre = x[i:i + batch] @ self.W.T + self.b
                act = (pre > 0.0)
                out[i:i + batch] = act.to(device=out_device, dtype=out_dtype)
        return out


# ------------------------------------------------------------------------------
# Probe training / evaluation
# ------------------------------------------------------------------------------

def train_probe(H_tr, S_tr, H_te, S_te, epochs=25, lr=0.01, batch=512,
                 weight_decay=1e-4, device=None):
    """Train linear probe.  H_* may be on CPU (as bool) and S_* on CPU (as
    int64) — batches are moved to `device` (or the probe's device) and cast
    to float / long as needed.
    """
    hidden = H_tr.shape[1]
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    probe = nn.Linear(hidden, BOARD_CELLS * 3).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                              weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    N = H_tr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            h = H_tr[idx].to(device=device, dtype=torch.float32)
            y = S_tr[idx].to(device=device, dtype=torch.long)
            logits = probe(h).view(-1, BOARD_CELLS, 3)
            loss = ce(logits.reshape(-1, 3), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def evaluate(probe, H, S, T=None, batch=4096):
    device = next(probe.parameters()).device
    N = H.shape[0]
    preds_all = torch.empty(N, BOARD_CELLS, dtype=torch.long, device='cpu')
    with torch.no_grad():
        for i in range(0, N, batch):
            h = H[i:i + batch].to(device=device, dtype=torch.float32)
            preds = probe(h).view(-1, BOARD_CELLS, 3).argmax(dim=-1)
            preds_all[i:i + batch] = preds.cpu()
    S_cpu = S.cpu() if S.is_cuda else S
    correct = (preds_all == S_cpu).float()
    acc = correct.mean().item()
    per_cell = correct.mean(dim=0).numpy()
    by_ply = {}
    if T is not None:
        T_cpu = T.cpu() if T.is_cuda else T
        for ply in range(int(T_cpu.max().item()) + 1):
            mask = (T_cpu == ply)
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
    ap.add_argument('--enumerate', dest='do_enumerate', action='store_true',
                    help='Use BFS to enumerate every reachable board position '
                          'at plies 0..max_ply-1 instead of sampling random '
                          'games for the training set.  The test set is '
                          'unchanged (random games).')
    ap.add_argument('--tree-max-depth', type=int, default=8)
    ap.add_argument('--tree-min-samples-leaf', type=int, default=1)
    ap.add_argument('--tree-n-jobs', type=int, default=1)
    ap.add_argument('--probe-epochs', type=int, default=25)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='opening_tree_mlp.pt')
    ap.add_argument('--top-k-per-cell', type=int, default=None,
                    help='If set, keep only the top-K most frequently '
                          'trained-on paths per cell (frequency-based rule '
                          'pruning).  Total H becomes at most 64 * K.  '
                          'Reduces overfit + memory; makes the rule set '
                          'inspectable.')
    ap.add_argument('--cache-tr', default=None,
                    help='Path to .npz cache for the sampled TRAIN set. '
                          'If exists, skip sampling.  If not, sample + save.')
    ap.add_argument('--cache-te', default=None,
                    help='Same but for the TEST set.')
    args = ap.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('warning: CUDA requested but not available; falling back to CPU')
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f'device: {device}')

    t0 = time.time()
    if args.do_enumerate:
        print(f'ENUMERATING every reachable position at plies '
               f'0..{args.max_ply - 1} via BFS...')
        Xnp_tr, Snp_tr, _, Tnp_tr = load_or_sample(
            args.cache_tr, enumerate_opening_positions,
            max_ply=args.max_ply, verbose=True)
        print(f'  enumeration done, {Xnp_tr.shape[0]} training positions '
               f'({time.time() - t0:.1f}s total)')
    else:
        print(f'sampling {args.num_train_games} train games, ply '
               f'0..{args.max_ply - 1}...')
        Xnp_tr, Snp_tr, _, Tnp_tr = load_or_sample(
            args.cache_tr, sample_opening_positions,
            args.num_train_games, max_ply=args.max_ply, seed=args.seed)
        print(f'  train={Xnp_tr.shape[0]}  ({time.time() - t0:.1f}s)')

    t0 = time.time()
    print(f'sampling {args.num_test_games} test games...')
    Xnp_te, Snp_te, _, Tnp_te = load_or_sample(
        args.cache_te, sample_opening_positions,
        args.num_test_games, max_ply=args.max_ply,
        seed=args.seed + 1_000_000)
    print(f'  test={Xnp_te.shape[0]}  ({time.time() - t0:.1f}s)')

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
        n_all = len(paths)
        paths = prune_paths_by_count(paths, args.top_k_per_cell)
        per_cell_leaf_counts[c] = len(paths)
        for path_idx, (conditions, leaf_class, leaf_counts) in enumerate(paths):
            w, b = path_to_weight(conditions)
            all_w.append(w); all_b.append(b)
            all_meta.append({
                'cell': c, 'path_idx': path_idx,
                'conditions': conditions, 'leaf_class': leaf_class,
                'depth': len(conditions),
                'leaf_counts': leaf_counts,
                'train_count': sum(leaf_counts),
            })
    W = np.stack(all_w); B = np.array(all_b, dtype=np.float32)
    print(f'  total hidden units: {len(all_meta)}'
           + (f'  (top-{args.top_k_per_cell} per cell)'
              if args.top_k_per_cell else ''))
    print(f'  leaves per tree: mean={per_cell_leaf_counts.mean():.1f}  '
           f'max={per_cell_leaf_counts.max()}  '
           f'min={per_cell_leaf_counts.min()}')

    depths = np.array([m['depth'] for m in all_meta])
    print(f'  path depths: mean={depths.mean():.2f}  max={depths.max()}  '
           f'min={depths.min()}')

    mlp = OpeningTreeMLP(W, B, all_meta, device)

    X_tr = torch.from_numpy(Xnp_tr).to(device)
    X_te = torch.from_numpy(Xnp_te).to(device)
    # Labels + turn indices live on CPU (bool/int8 activations too);
    # batches move to compute device inside train/eval.
    S_tr = torch.from_numpy(Snp_tr)
    S_te = torch.from_numpy(Snp_te)
    T_te = torch.from_numpy(Tnp_te)

    print('\ncomputing hidden activations (bool on CPU)...')
    t0 = time.time()
    H_tr = mlp(X_tr, out_device='cpu', out_dtype=torch.bool)
    H_te = mlp(X_te, out_device='cpu', out_dtype=torch.bool)
    print(f'  H_tr {tuple(H_tr.shape)} ({H_tr.element_size() * H_tr.nelement() / 1e9:.2f} GB)  '
           f'H_te {tuple(H_te.shape)} ({H_te.element_size() * H_te.nelement() / 1e9:.2f} GB)  '
           f'({time.time() - t0:.1f}s)')
    # Free the big input tensor on GPU; we don't need X after activations.
    del X_tr, X_te
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # H_tr is bool; take the sum along dim 0 (int64, ~1 MB), then divide.
    # Avoids materializing an (N, H) float32 tensor of ~540 GB.
    fire_rate = H_tr.sum(dim=0).float() / H_tr.shape[0]
    print(f'  per-unit firing rate on train: mean={fire_rate.mean().item()*100:.2f}%  '
           f'min={fire_rate.min().item()*100:.4f}%  '
           f'max={fire_rate.max().item()*100:.2f}%')
    print(f'  dead units: {int((fire_rate == 0).sum().item())} / {len(all_meta)}')

    print('\ntraining linear probe on hidden layer...')
    probe = train_probe(H_tr, S_tr, H_te, S_te,
                          epochs=args.probe_epochs, device=device)

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
