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
import torch.nn.functional as F
from sklearn.tree import DecisionTreeClassifier, _tree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.othello import OthelloBoardState


BOARD_CELLS = 64
TURNS = 60
INPUT_DIM_BASE = 120               # 60 played + 60 even
INPUT_DIM = 121                    # 60 played + 60 even + 1 mover_parity
CENTER_64 = {27, 28, 35, 36}
NON_CENTER_64 = sorted(set(range(64)) - CENTER_64)
C64_TO_C60 = {c: i for i, c in enumerate(NON_CENTER_64)}
C60_TO_C64 = {i: c for i, c in enumerate(NON_CENTER_64)}
STATE_NAMES = ['empty', 'mine', 'opp']


def compute_input_dim(when_bucket_size=None, use_move_grid=False,
                        recent_Ks=None):
    """Base 121 (played + even + mover_parity), plus optionally:
      - when-buckets: 60 * n_buckets features (mutex with move grid).
      - move grid: 60 * 60 = 3600 features (mutex with when-buckets).
      - recent_Ks: 60 * len(recent_Ks) features — for each K in the list,
        one per-cell bit indicating whether that cell was played in the
        last K turns of the prefix.  Composes with either mutex block.
    """
    dim = 121
    if use_move_grid:
        dim += TURNS * 60           # 60 turns × 60 non-center cells
    elif when_bucket_size:
        n_buckets = (TURNS + when_bucket_size - 1) // when_bucket_size
        dim += 60 * n_buckets
    if recent_Ks:
        dim += 60 * len(recent_Ks)
    return dim


def playedeven_features(prefix, when_bucket_size=None, use_move_grid=False,
                          recent_Ks=None, canonicalize_mover=False):
    """Return input features for a game prefix.

    Base 121-d: 60 played + 60 even + 1 mover_parity.
    Optional temporal blocks (compose):
      when-buckets: for each non-center cell C and each bucket k, one bit
                    that fires iff C was played at some turn in
                    [k*bucket_size, (k+1)*bucket_size).  Mutex with movegrid.
      move grid:    for each turn t (0..59) and each non-center cell C, one
                    bit that fires iff the move at turn t was cell C.  At
                    most 60 bits set out of 3600.  Mutex with when-buckets.
      recent_Ks:    for each K in the list, one per-cell bit indicating
                    whether that cell was played in prefix[T-K:T].  Composes
                    with either mutex block above.

    canonicalize_mover: if True, replace the absolute-color parity encoding
      (even + mover_parity) with a mover-relative encoding:
        feat[60 + i] = placed_as_mover[c]
                     = 1 iff cell c was placed by whichever side is now
                       to move next.
      feat[120] (mover_parity) is set to 0, since parity is now baked into
      the per-cell bit.  Total dim unchanged at 121-d (constant last bit).
      This lets structurally-identical color-swapped positions have IDENTICAL
      feature vectors, and lets each tree see 800k rows in one representation
      rather than partitioning capacity by mover.
    """
    input_dim = compute_input_dim(when_bucket_size, use_move_grid, recent_Ks)
    feat = np.zeros(input_dim, dtype=np.float32)
    T = len(prefix)
    mp = T % 2                             # 0 if BLACK to move, 1 if WHITE
    for t, c in enumerate(prefix):
        if c not in C64_TO_C60:
            continue
        i = C64_TO_C60[c]
        feat[i] = 1.0
        parity = t % 2                     # 0 for even (BLACK) turn
        if canonicalize_mover:
            # placed_as_mover: 1 iff this move belongs to the current mover.
            # BLACK plays even turns (parity=0) and BLACK-to-move is mp=0;
            # WHITE plays odd turns (parity=1) and WHITE-to-move is mp=1.
            # So cell was placed by mover iff parity == mp.
            if parity == mp:
                feat[60 + i] = 1.0
        else:
            if parity == 0:
                feat[60 + i] = 1.0
        if use_move_grid:
            feat[121 + t * 60 + i] = 1.0
        elif when_bucket_size:
            bucket = t // when_bucket_size
            feat[121 + bucket * 60 + i] = 1.0
    if not canonicalize_mover:
        feat[120] = 1.0 if T % 2 == 1 else 0.0
    # else: feat[120] stays 0 — mover_parity now baked into feat[60:120].
    if recent_Ks:
        # Offset of the recent block: past any mutex temporal block.
        if use_move_grid:
            base = 121 + TURNS * 60
        elif when_bucket_size:
            n_buckets = (TURNS + when_bucket_size - 1) // when_bucket_size
            base = 121 + 60 * n_buckets
        else:
            base = 121
        T = len(prefix)
        for k_idx, K in enumerate(recent_Ks):
            t_lo = max(0, T - K)
            for t in range(t_lo, T):
                c = prefix[t]
                if c in C64_TO_C60:
                    feat[base + k_idx * 60 + C64_TO_C60[c]] = 1.0
    return feat


def feature_name(feat_idx, recent_Ks=None):
    """Return a human-readable name for a played_even feature index."""
    if feat_idx == 120:
        return 'mover_parity'
    if feat_idx < 60:
        cell = C60_TO_C64[feat_idx]
    elif feat_idx < 120:
        cell = C60_TO_C64[feat_idx - 60]
    else:
        cell = None
    if cell is not None:
        col = 'ABCDEFGH'[cell % 8]
        row = str(cell // 8 + 1)
        prefix = 'played' if feat_idx < 60 else 'even'
        return f'{prefix}[{col}{row}]'
    if recent_Ks:
        rel = feat_idx - 121
        k_idx, i = divmod(rel, 60)
        if 0 <= k_idx < len(recent_Ks):
            K = recent_Ks[k_idx]
            cell = C60_TO_C64[i]
            col = 'ABCDEFGH'[cell % 8]
            row = str(cell // 8 + 1)
            return f'recent{K}[{col}{row}]'
    return f'feat[{feat_idx}]'


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


def sample_opening_positions(num_games, max_ply=10, seed=42,
                              when_bucket_size=None, use_move_grid=False):
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
            Xs.append(playedeven_features(prefix, when_bucket_size,
                                            use_move_grid))
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

def _fit_one_tree(X, y, max_depth, min_samples_leaf, max_features=None):
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0,
                                    min_samples_leaf=min_samples_leaf,
                                    max_features=max_features)
    tree.fit(X, y)
    return tree


def train_per_cell_trees(X, S, max_depth, min_samples_leaf=1, n_jobs=1,
                          max_features=None):
    """Train 64 decision trees, optionally in parallel across cells."""
    if n_jobs == 1:
        return [_fit_one_tree(X, S[:, c], max_depth, min_samples_leaf,
                              max_features)
                for c in range(BOARD_CELLS)]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_tree)(X, S[:, c], max_depth, min_samples_leaf,
                                max_features)
        for c in range(BOARD_CELLS))


def _fit_one_pattern_tree(X, y, max_depth, min_samples_leaf,
                             max_features, class_weight, seed, bootstrap):
    """Fit a single (optionally bagged) tree targeting one pattern's
    activation.  When bootstrap is True the training rows are resampled
    with replacement — this is what makes ensembles diverse."""
    if bootstrap:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], X.shape[0], replace=True)
        X = X[idx]; y = y[idx]
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=seed)
    tree.fit(X, y)
    return tree


_SKOPE_PARSE_RE = None   # compiled lazily inside _skope_parse_rule


def _skope_parse_rule(rule_str):
    """Parse a SkopeRules rule string into our [(feat_idx, val), ...]
    conditions format.  Expects features named X[:, i] with threshold 0.5
    (binary features)."""
    import re
    global _SKOPE_PARSE_RE
    if _SKOPE_PARSE_RE is None:
        _SKOPE_PARSE_RE = re.compile(r'X\[:,\s*(\d+)\]\s*([<>=]+)\s*([\d.]+)')
    conds = []
    for conj in rule_str.split(' and '):
        m = _SKOPE_PARSE_RE.search(conj.strip())
        if not m:
            continue
        feat = int(m.group(1))
        op = m.group(2)
        # Binary features + threshold 0.5:
        #   ">" → feat=1;  "<=" (or "<") → feat=0
        val = 1 if op == '>' else 0
        conds.append((feat, val))
    return conds


def _skope_fit_one(X, y, n_estimators, max_depth, precision_min,
                     recall_min, feature_names, random_state):
    """Fit SkopeRules for one binary target; return PreExtractedPaths."""
    # SkopeRules imports sklearn.externals.six -- monkey-patch for modern sklearn.
    import sys as _sys
    if 'sklearn.externals.six' not in _sys.modules:
        try:
            import six as _six
            _sys.modules['sklearn.externals.six'] = _six
        except ImportError:
            pass
    from skrules import SkopeRules
    sr = SkopeRules(
        feature_names=feature_names,
        n_estimators=n_estimators,
        max_depth=max_depth,
        precision_min=precision_min,
        recall_min=recall_min,
        random_state=random_state,
    )
    sr.fit(X, y)
    paths = []
    for rule_tuple in sr.rules_:
        rule_str = rule_tuple[0]
        # rule_tuple[1] = (precision, recall, ...); use it for leaf_counts proxy.
        scores = rule_tuple[1]
        precision = float(scores[0]) if len(scores) > 0 else 0.0
        recall = float(scores[1]) if len(scores) > 1 else 0.0
        conds = _skope_parse_rule(rule_str)
        if not conds:
            continue
        # Fake leaf_counts so prune_paths_by_count still works: put the
        # coverage estimate in the positive-class bin.
        n_pos = max(int(recall * y.sum()), 1)
        n_neg = max(int(n_pos * (1 - precision) / max(precision, 1e-6)), 0)
        paths.append((conds, 1, [n_neg, n_pos]))
    return PreExtractedPaths(paths)


def train_pattern_trees(X, target, n_trees_per_pattern=1, max_depth=10,
                          min_samples_leaf=50, n_jobs=1, max_features=None,
                          class_weight='balanced', use_random_forest=False,
                          algorithm='dt', gb_learning_rate=0.1,
                          skope_precision_min=0.5, skope_recall_min=0.01):
    """Fit `n_trees_per_pattern` trees per pattern.

    algorithm ∈ {'dt', 'rf', 'et', 'gbm', 'skope'}:
      dt    — one or more DecisionTreeClassifier(s), bootstrapped when >1
      rf    — RandomForestClassifier (also triggered by use_random_forest=True)
      et    — ExtraTreesClassifier
      gbm   — GradientBoostingClassifier; shallow trees, additive
      skope — SkopeRules (rare-positive rule mining; filters by precision/recall)

    Returns: list of length K, each entry a list of tree-like objects
    (sklearn trees for dt/rf/et/gbm; PreExtractedPaths for skope).
    """
    # Back-compat: use_random_forest overrides algorithm to 'rf'.
    if use_random_forest and algorithm == 'dt':
        algorithm = 'rf'

    K = target.shape[1]
    bootstrap = (n_trees_per_pattern > 1)

    def one_pattern(j):
        y = target[:, j]
        if algorithm == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(
                n_estimators=n_trees_per_pattern,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
                bootstrap=True,
                n_jobs=1,
                random_state=j * 1000)
            rf.fit(X, y)
            return list(rf.estimators_)
        if algorithm == 'et':
            from sklearn.ensemble import ExtraTreesClassifier
            et = ExtraTreesClassifier(
                n_estimators=n_trees_per_pattern,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
                bootstrap=False,   # ExtraTrees default
                n_jobs=1,
                random_state=j * 1000)
            et.fit(X, y)
            return list(et.estimators_)
        if algorithm == 'skope':
            return [_skope_fit_one(
                X, y,
                n_estimators=n_trees_per_pattern,
                max_depth=max_depth,
                precision_min=skope_precision_min,
                recall_min=skope_recall_min,
                feature_names=None,   # SkopeRules default 'X[:, i]' format
                random_state=j * 1000,
            )]
        if algorithm == 'gbm':
            from sklearn.ensemble import GradientBoostingClassifier
            # For a rare-positive binary target, class_weight isn't native
            # in GBM; use sample_weight balancing manually.
            if class_weight == 'balanced':
                n_pos = int(y.sum())
                n_neg = len(y) - n_pos
                w_pos = 0.5 / max(n_pos, 1)
                w_neg = 0.5 / max(n_neg, 1)
                sample_weight = np.where(y == 1, w_pos, w_neg)
            else:
                sample_weight = None
            gb = GradientBoostingClassifier(
                n_estimators=n_trees_per_pattern,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                learning_rate=gb_learning_rate,
                random_state=j * 1000)
            gb.fit(X, y, sample_weight=sample_weight)
            # gb.estimators_ shape: (n_iterations, 1) DecisionTreeRegressor
            return [est[0] for est in gb.estimators_]
        # default: dt
        return [_fit_one_pattern_tree(X, y, max_depth,
                                          min_samples_leaf, max_features,
                                          class_weight,
                                          seed=j * 1000 + k,
                                          bootstrap=bootstrap)
                for k in range(n_trees_per_pattern)]

    if n_jobs == 1:
        return [one_pattern(j) for j in range(K)]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(delayed(one_pattern)(j)
                                       for j in range(K))


class PreExtractedPaths:
    """Lightweight wrapper for algorithms (like SkopeRules) that produce
    rules directly rather than sklearn trees.  extract_paths() unwraps
    these transparently."""
    def __init__(self, paths):
        self.paths = paths  # list of (conditions, leaf_class, leaf_counts)


def extract_paths(tree):
    """Return list of (conditions, leaf_class, leaf_counts) tuples.
    conditions: list of (feature_idx, required_value 0-or-1).
    Feature values are 0/1; sklearn's threshold ~0.5 splits them.  Left =
    feature ≤ 0.5 (i.e., feature = 0); right = feature > 0.5 (feature = 1).

    Accepts sklearn trees OR PreExtractedPaths.
    """
    if isinstance(tree, PreExtractedPaths):
        return tree.paths
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
                 out_dtype=torch.bool, use_relu=False):
        """Compute hidden activations in chunks.

        Default batch is small (256) because H can grow into the 100k+
        range for endgame/midgame, and the per-batch matmul on GPU costs
        `batch × H × 4` bytes.  batch=256 with H=200k = 200 MB — safe on
        a 10 GB GPU.

        use_relu=False (default): step activation, output is (bool)
          1 iff pre-activation > 0, else 0.
        use_relu=True: ReLU activation, output is float32 = max(0, pre).
          Each unit gives "excess magnitude above threshold" — preserves
          count-like information the linear probe can then weight
          continuously.  out_dtype should be torch.float32 in this case.
        """
        H = self.hidden_dim
        N = x.shape[0]
        if use_relu and out_dtype == torch.bool:
            out_dtype = torch.float32
        out = torch.empty(N, H, device=out_device, dtype=out_dtype)
        with torch.no_grad():
            for i in range(0, N, batch):
                pre = x[i:i + batch] @ self.W.T + self.b
                if use_relu:
                    act = torch.relu(pre)
                else:
                    act = (pre > 0.0)
                out[i:i + batch] = act.to(device=out_device, dtype=out_dtype)
        return out


# ------------------------------------------------------------------------------
# Probe training / evaluation
# ------------------------------------------------------------------------------

def train_probe(H_tr, S_tr, H_te, S_te, epochs=25, lr=0.01, batch=512,
                 weight_decay=1e-4, device=None, l1_lambda=0.0, seed=0):
    """Train linear probe.  H_* may be on CPU (as bool) and S_* on CPU (as
    int64) — batches are moved to `device` (or the probe's device) and cast
    to float / long as needed.

    If l1_lambda > 0, adds L1 penalty to probe.weight, encouraging a sparse
    subset of hidden units.  Post-training, weights below 1e-4 can be
    treated as "not selected".

    seed controls probe initialization + batch order.  For a multi-seed
    ensemble, wrap this in a loop and average softmax(probe(H)) across seeds.
    """
    hidden = H_tr.shape[1]
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    probe = nn.Linear(hidden, BOARD_CELLS * 3).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                              weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    N = H_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            h = H_tr[idx].to(device=device, dtype=torch.float32)
            y = S_tr[idx].to(device=device, dtype=torch.long)
            logits = probe(h).view(-1, BOARD_CELLS, 3)
            loss = ce(logits.reshape(-1, 3), y.reshape(-1))
            if l1_lambda > 0:
                loss = loss + l1_lambda * probe.weight.abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def train_probe_sklearn(H_tr, S_tr, H_te, S_te, C=1.0, n_jobs=1,
                           max_iter=200, verbose=True,
                           subsample_train=None, solver='lbfgs'):
    """Fit sklearn LogisticRegression per cell using the LBFGS solver.

    LBFGS provably converges to the global optimum for the convex logistic
    loss.  If AdamW-based probes drop when features are added but LBFGS
    doesn't, the AdamW is undertrained.

    Returns list of 64 fitted LR models.

    Args:
      C: inverse L2 regularization (larger = less regularization).
      n_jobs: parallelize the 64 per-cell fits.
    """
    from sklearn.linear_model import LogisticRegression
    from joblib import Parallel, delayed

    def _to_np(t):
        if hasattr(t, 'numpy'):
            arr = t.numpy()
        else:
            arr = t
        if arr.dtype == np.bool_:
            arr = arr.astype(np.float32)
        return arr

    H_tr_np = _to_np(H_tr)
    S_tr_np = _to_np(S_tr)

    # Optional subsample to bound memory + fit time.  Each joblib worker
    # copies the whole H_tr_np, so at scale we can OOM.
    if subsample_train is not None and subsample_train < H_tr_np.shape[0]:
        rng = np.random.RandomState(0)
        idx = rng.choice(H_tr_np.shape[0], subsample_train, replace=False)
        H_tr_np = H_tr_np[idx]
        S_tr_np = S_tr_np[idx]
        if verbose:
            print(f'  subsampled train to {H_tr_np.shape[0]} rows')

    def fit_one(c):
        lr = LogisticRegression(solver=solver, C=C, max_iter=max_iter,
                                  tol=1e-3)
        lr.fit(H_tr_np, S_tr_np[:, c])
        return lr

    t0 = time.time()
    if n_jobs == 1:
        models = [fit_one(c) for c in range(BOARD_CELLS)]
    else:
        models = Parallel(n_jobs=n_jobs)(
            delayed(fit_one)(c) for c in range(BOARD_CELLS))
    if verbose:
        print(f'  sklearn LR (LBFGS) fit {BOARD_CELLS} cells in '
               f'{time.time() - t0:.1f}s')
    return models


def evaluate_sklearn(models, H, S, T=None, batch=8192):
    """Evaluate 64 sklearn per-cell LR models.  Returns (mean_acc,
    per_cell_acc, by_ply)."""
    if hasattr(H, 'numpy'):
        H_np = H.numpy()
    else:
        H_np = H
    if H_np.dtype == np.bool_:
        H_np = H_np.astype(np.float32)
    if hasattr(S, 'numpy'):
        S_np = S.numpy()
    else:
        S_np = S
    N = H_np.shape[0]
    preds = np.zeros((N, BOARD_CELLS), dtype=np.int64)
    # Predictions per cell — chunk to avoid huge memory allocations.
    for c in range(BOARD_CELLS):
        for i in range(0, N, batch):
            preds[i:i + batch, c] = models[c].predict(H_np[i:i + batch])
    correct = (preds == S_np).astype(np.float32)
    acc = float(correct.mean())
    per_cell = correct.mean(axis=0)
    by_ply = {}
    if T is not None:
        if hasattr(T, 'numpy'):
            T_np = T.numpy()
        else:
            T_np = T
        for ply in range(int(T_np.max()) + 1):
            mask = (T_np == ply)
            if mask.any():
                by_ply[ply] = (int(mask.sum()),
                                float(correct[mask].mean()))
    return acc, per_cell, by_ply


def train_probe_ensemble(H_tr, S_tr, H_te, S_te, n_seeds=1, **kwargs):
    """Train `n_seeds` probes with different seeds.  Returns list of probes.

    Callers can then average softmax probabilities across probes for
    ensemble prediction, or evaluate each probe individually.
    """
    if n_seeds <= 1:
        return [train_probe(H_tr, S_tr, H_te, S_te, seed=0, **kwargs)]
    probes = []
    for s in range(n_seeds):
        probes.append(train_probe(H_tr, S_tr, H_te, S_te, seed=s, **kwargs))
    return probes


def evaluate_ensemble(probes, H, S, T=None, batch=512):
    """Evaluate an ensemble of probes by averaging softmax probabilities.
    Returns (mean_acc, per_cell_acc, by_ply_dict, per_probe_accs)."""
    device = next(probes[0].parameters()).device
    N = H.shape[0]
    accum_probs = torch.zeros(N, BOARD_CELLS, 3, dtype=torch.float32)
    per_probe_correct = [0 for _ in probes]
    per_probe_total = 0
    S_cpu = S.cpu() if S.is_cuda else S
    for i in range(0, N, batch):
        h = H[i:i + batch].to(device=device, dtype=torch.float32)
        s_batch = S_cpu[i:i + batch].long()
        for pi, probe in enumerate(probes):
            with torch.no_grad():
                logits = probe(h).view(-1, BOARD_CELLS, 3)
                probs = torch.softmax(logits, dim=-1).cpu()
                accum_probs[i:i + batch] += probs
                per_probe_correct[pi] += (probs.argmax(dim=-1)
                                            == s_batch).sum().item()
        per_probe_total += s_batch.numel()
    accum_probs /= len(probes)
    preds = accum_probs.argmax(dim=-1)
    correct = (preds == S_cpu).float()
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
    per_probe_accs = [c / per_probe_total for c in per_probe_correct]
    return acc, per_cell, by_ply, per_probe_accs


def evaluate(probe, H, S, T=None, batch=512):
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
# Legal-move prediction:
#   - Option A: linear head + BCE loss.
#   - Option B: noisy-OR head — 1 - exp(-Σ w_ic h_i) — with BCE loss.
#   - Option C: derive from state predictions by applying flanking rules.
# ------------------------------------------------------------------------------


def _by_ply_10(T_np, correct):
    """Bucket per-cell correctness by 10-ply intervals; return {(lo, hi): (n, acc)}."""
    if T_np is None or correct is None:
        return {}
    out = {}
    for lo in range(int(T_np.min()) // 10 * 10,
                     int(T_np.max()) + 1, 10):
        mask = (T_np >= lo) & (T_np < lo + 10)
        if mask.any():
            out[(lo, lo + 10)] = (int(mask.sum()),
                                    float(correct[mask].mean()))
    return out


def train_probe_legal_bce(H_tr, L_tr, epochs=25, lr=0.01, batch=512,
                             weight_decay=1e-4, device=None, seed=0):
    """Linear H → 64 logits with BCE-with-logits loss.
    L_tr: (N, 64) uint8 legal-move mask."""
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    probe = nn.Linear(H_tr.shape[1], BOARD_CELLS).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                              weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    N = H_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            h = H_tr[idx].to(device=device, dtype=torch.float32)
            l = L_tr[idx].to(device=device, dtype=torch.float32)
            logits = probe(h)
            loss = bce(logits, l)
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def train_probe_legal_bce_ensemble(H_tr, L_tr, n_seeds=1, **kwargs):
    if n_seeds <= 1:
        return [train_probe_legal_bce(H_tr, L_tr, seed=0, **kwargs)]
    return [train_probe_legal_bce(H_tr, L_tr, seed=s, **kwargs)
            for s in range(n_seeds)]


class PatternProbOrHead(nn.Module):
    """Structured prob-OR head for legal-move prediction over pattern trees.

    Groups the hidden units by which pattern's tree they came from.  For
    each pattern j:
        p_j(h) = sigmoid( w_j . h_j + b_j )
    where h_j is the subvector of h containing only pattern j's leaves.
    Then per target cell C:
        P(cell C legal | h) = 1 - Π_{j : target(j) = C} (1 - p_j(h))

    The linear layer per pattern learns to weight "definitively fire"
    leaves higher.  Prob-OR grouped by target cell captures the "any
    pattern for C fires" logic natively.

    Parameter count = (# pattern-path hidden units) + (# patterns) biases.
    Much smaller than a flat BCE probe on the same features.
    """
    def __init__(self, hidden_meta, patterns_list, use_pattern_bias=False):
        super().__init__()
        from collections import defaultdict
        self.use_pattern_bias = use_pattern_bias
        idx_by_pat = defaultdict(list)
        for i, m in enumerate(hidden_meta):
            if m.get('kind') == 'pattern_path':
                idx_by_pat[m['pattern']].append(i)
        # Sort for reproducibility.
        self.pattern_ids = sorted(idx_by_pat.keys())
        # Flat concatenation of leaf-indices + per-pattern boundaries.
        flat = []
        boundaries = [0]
        for j in self.pattern_ids:
            flat.extend(idx_by_pat[j])
            boundaries.append(len(flat))
        self.register_buffer('idx_flat',
                                torch.tensor(flat, dtype=torch.long))
        self.register_buffer('boundaries',
                                torch.tensor(boundaries, dtype=torch.long))
        # Per-leaf pattern index — enables the vectorized scatter_add
        # segment sum instead of a Python loop over patterns.
        n_patterns = len(self.pattern_ids)
        leaf_pat_ids = torch.zeros(len(flat), dtype=torch.long)
        for j in range(n_patterns):
            leaf_pat_ids[boundaries[j]:boundaries[j + 1]] = j
        self.register_buffer('leaf_pat_ids', leaf_pat_ids)
        # One weight per leaf.  Per-pattern bias is only a Parameter when
        # use_pattern_bias=True; otherwise a fixed zero buffer (no learned
        # bias).  This lets us measure how much of the accuracy is
        # attributable to the learned bias vs. the per-leaf weights alone.
        self.leaf_weights = nn.Parameter(
            torch.randn(len(flat)) * 0.05)
        if use_pattern_bias:
            self.pattern_biases = nn.Parameter(
                torch.full((n_patterns,), -3.0))
        else:
            self.register_buffer(
                'pattern_biases',
                torch.zeros(n_patterns, dtype=torch.float32))
        # Precompute per-cell gather + mask for vectorized prob-OR.
        pat_id_to_pos = {j: pos for pos, j in enumerate(self.pattern_ids)}
        by_tgt = {}
        for j, pat in enumerate(patterns_list):
            by_tgt.setdefault(pat['target'], []).append(j)
        max_per_cell = max((len(pids) for pids in by_tgt.values()),
                             default=0)
        cell_pat_indices = torch.zeros(BOARD_CELLS, max_per_cell,
                                          dtype=torch.long)
        cell_pat_mask = torch.zeros(BOARD_CELLS, max_per_cell,
                                        dtype=torch.bool)
        for cell in range(BOARD_CELLS):
            positions = [pat_id_to_pos[j] for j in by_tgt.get(cell, [])
                          if j in pat_id_to_pos]
            for k, pos in enumerate(positions):
                cell_pat_indices[cell, k] = pos
                cell_pat_mask[cell, k] = True
        self.register_buffer('cell_pat_indices', cell_pat_indices)
        self.register_buffer('cell_pat_mask', cell_pat_mask)

    def forward(self, h):
        N = h.shape[0]
        n_patterns = len(self.boundaries) - 1
        h_flat = h[:, self.idx_flat]              # (N, total_leaves)
        contrib = h_flat * self.leaf_weights      # (N, total_leaves)
        # Vectorized segment sum via scatter_add — no Python loop.
        pat_logits = torch.zeros(N, n_patterns,
                                    dtype=contrib.dtype, device=h.device)
        pat_ids_expanded = self.leaf_pat_ids.unsqueeze(0).expand(N, -1)
        pat_logits.scatter_add_(dim=1, index=pat_ids_expanded, src=contrib)
        pat_logits = pat_logits + self.pattern_biases
        pat_probs = torch.sigmoid(pat_logits)     # (N, n_patterns)
        gathered = pat_probs[:, self.cell_pat_indices]        # (N, C, max)
        one_minus = torch.where(
            self.cell_pat_mask.unsqueeze(0), 1.0 - gathered,
            torch.ones_like(gathered))
        return 1.0 - one_minus.prod(dim=2)                     # (N, C)


def train_probe_legal_patterns_structured(H_tr, L_tr, meta, patterns_list,
                                             epochs=25, lr=1e-3, batch=2048,
                                             weight_decay=1e-4, device=None,
                                             seed=0):
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    probe = PatternProbOrHead(meta, patterns_list).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                              weight_decay=weight_decay)
    N = H_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            h = H_tr[idx].to(device=device, dtype=torch.float32)
            l = L_tr[idx].to(device=device, dtype=torch.float32)
            p = probe(h).clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(p, l)
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def train_probe_legal_patterns_structured_ensemble(
        H_tr, L_tr, meta, patterns_list, n_seeds=1, **kwargs):
    if n_seeds <= 1:
        return [train_probe_legal_patterns_structured(
            H_tr, L_tr, meta, patterns_list, seed=0, **kwargs)]
    return [train_probe_legal_patterns_structured(
        H_tr, L_tr, meta, patterns_list, seed=s, **kwargs)
        for s in range(n_seeds)]


class CellProbOrHead(nn.Module):
    """Per-cell linear + sigmoid head for legal-move prediction.

    Groups the hidden units by which cell's tree produced them.  For each
    output cell C:
        p_C(h) = sigmoid( w_C . h_C + b_C )
    Independent per-cell readout, no cross-cell weight sharing.

    Total params = sum over cells of (leaves for cell) + 64 biases;
    typically ~3200 for legaltrees, vs 200k+ for a flat linear readout
    on the same hidden layer.  Cleaner interpretation: w_C says which of
    cell C's tree leaves matter for confirming C is legal.
    """
    def __init__(self, hidden_meta):
        super().__init__()
        from collections import defaultdict
        idx_by_cell = defaultdict(list)
        for i, m in enumerate(hidden_meta):
            if m.get('kind') == 'tree_path' and 'cell' in m:
                idx_by_cell[m['cell']].append(i)
        self.cell_ids = sorted(idx_by_cell.keys())
        flat = []
        boundaries = [0]
        for c in self.cell_ids:
            flat.extend(idx_by_cell[c])
            boundaries.append(len(flat))
        self.register_buffer('idx_flat',
                                torch.tensor(flat, dtype=torch.long))
        self.register_buffer('boundaries',
                                torch.tensor(boundaries, dtype=torch.long))
        self.leaf_weights = nn.Parameter(
            torch.randn(len(flat)) * 0.05)
        self.cell_biases = nn.Parameter(torch.zeros(len(self.cell_ids)))

    def forward(self, h):
        N = h.shape[0]
        n_cells = len(self.boundaries) - 1
        h_flat = h[:, self.idx_flat]
        contrib = h_flat * self.leaf_weights
        cell_logits = torch.empty(N, n_cells, dtype=torch.float32,
                                     device=h.device)
        for c in range(n_cells):
            s, e = int(self.boundaries[c].item()), \
                int(self.boundaries[c + 1].item())
            cell_logits[:, c] = contrib[:, s:e].sum(dim=1)
        cell_logits = cell_logits + self.cell_biases
        # Rearrange into full 64-cell output.  If some cells missing
        # (no leaves), their output stays at sigmoid(0) = 0.5.
        if n_cells == BOARD_CELLS and self.cell_ids == list(range(BOARD_CELLS)):
            return torch.sigmoid(cell_logits)
        full = torch.zeros(N, BOARD_CELLS, dtype=torch.float32,
                              device=h.device)
        for pos, cell in enumerate(self.cell_ids):
            full[:, cell] = cell_logits[:, pos]
        return torch.sigmoid(full)


def train_probe_legal_cells_structured(H_tr, L_tr, meta, epochs=25,
                                             lr=1e-3, batch=512,
                                             weight_decay=1e-4, device=None,
                                             seed=0):
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    probe = CellProbOrHead(meta).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                              weight_decay=weight_decay)
    N = H_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            h = H_tr[idx].to(device=device, dtype=torch.float32)
            l = L_tr[idx].to(device=device, dtype=torch.float32)
            p = probe(h).clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(p, l)
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def train_probe_legal_cells_structured_ensemble(H_tr, L_tr, meta,
                                                    n_seeds=1, **kwargs):
    if n_seeds <= 1:
        return [train_probe_legal_cells_structured(
            H_tr, L_tr, meta, seed=0, **kwargs)]
    return [train_probe_legal_cells_structured(H_tr, L_tr, meta,
                                                    seed=s, **kwargs)
            for s in range(n_seeds)]


class LinearPatternProbOr(nn.Module):
    """Simplest legal-move readout via 960 patterns:

      Linear(H -> 960) -> sigmoid -> prob-OR per target cell

    Matches the "MLP -> 960 rules -> prob-OR" pattern used in the original
    MLP baseline.  Fully vectorized: one matrix multiply, one sigmoid, one
    gather + product.  No Python loops in the forward pass, so runs
    ~10-20x faster than the per-pattern-loop StruPO head.
    """
    def __init__(self, hidden_dim, patterns_list):
        super().__init__()
        K = len(patterns_list)
        self.linear = nn.Linear(hidden_dim, K)
        from collections import defaultdict
        by_tgt = defaultdict(list)
        for j, pat in enumerate(patterns_list):
            by_tgt[pat['target']].append(j)
        max_per_cell = max(len(pids) for pids in by_tgt.values())
        cell_pat_indices = torch.zeros(BOARD_CELLS, max_per_cell,
                                          dtype=torch.long)
        cell_pat_mask = torch.zeros(BOARD_CELLS, max_per_cell,
                                        dtype=torch.bool)
        for cell in range(BOARD_CELLS):
            positions = by_tgt.get(cell, [])
            for k, pos in enumerate(positions):
                cell_pat_indices[cell, k] = pos
                cell_pat_mask[cell, k] = True
        self.register_buffer('cell_pat_indices', cell_pat_indices)
        self.register_buffer('cell_pat_mask', cell_pat_mask)

    def forward(self, h):
        pat_probs = torch.sigmoid(self.linear(h))      # (N, K)
        gathered = pat_probs[:, self.cell_pat_indices] # (N, C, max)
        one_minus = torch.where(
            self.cell_pat_mask.unsqueeze(0), 1.0 - gathered,
            torch.ones_like(gathered))
        return 1.0 - one_minus.prod(dim=2)             # (N, C)


def train_probe_legal_linear_pat_probor(H_tr, L_tr, patterns_list,
                                            epochs=25, lr=0.01,
                                            batch=1024, weight_decay=1e-4,
                                            device=None, seed=0):
    """Larger default batch (1024) since forward is a single matmul —
    exploit GPU parallelism."""
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    probe = LinearPatternProbOr(H_tr.shape[1], patterns_list).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                              weight_decay=weight_decay)
    N = H_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            h = H_tr[idx].to(device=device, dtype=torch.float32)
            l = L_tr[idx].to(device=device, dtype=torch.float32)
            p = probe(h).clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(p, l)
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def train_probe_legal_linear_pat_probor_ensemble(H_tr, L_tr, patterns_list,
                                                       n_seeds=1, **kwargs):
    if n_seeds <= 1:
        return [train_probe_legal_linear_pat_probor(
            H_tr, L_tr, patterns_list, seed=0, **kwargs)]
    return [train_probe_legal_linear_pat_probor(
        H_tr, L_tr, patterns_list, seed=s, **kwargs)
        for s in range(n_seeds)]


class NoisyOrHead(nn.Module):
    """Noisy-OR / prob-OR combination of per-(unit, output-cell) votes.

    Each unit i with activation h_i ≥ 0 casts a per-cell "vote":
        p_{i,c} = 1 - exp(-w_{i,c} h_i),   w_{i,c} ≥ 0
    Combined by prob-OR ("legal iff any unit fires FOR it"):
        P(legal_c) = 1 - Π_i (1 - p_{i,c}) = 1 - exp(-(h @ w)_c)
    Trained end-to-end with BCE.  w = softplus(raw) keeps rates non-negative.
    """
    def __init__(self, H, C=BOARD_CELLS):
        super().__init__()
        # Start with lambda = H * w * h ≈ 0 so P(legal) ≈ 0 across cells.
        # For H ~ 4000 and h ≈ 0.5, we want w * H * h ≈ small ⇒ w ~ 1e-4 ⇒
        # softplus(raw) ≈ 1e-4 ⇒ raw ≈ ln(exp(1e-4) - 1) ≈ -9.2.
        self.raw = nn.Parameter(torch.randn(H, C) * 0.01 - 9.0)

    def forward(self, h):
        w = F.softplus(self.raw)
        return 1.0 - torch.exp(-(h @ w))


class StateNoisyOrHead(nn.Module):
    """3-way per-cell noisy-OR state readout.

    Each hidden unit i has a non-negative rate w_{i,c,k} for each cell C
    and class k (empty / mine / opp).  Per-class scores combine by prob-OR:
        P_k(C, h) = 1 - exp(-Sigma_i w_{i,c,k} * h_i)
    Then normalized across k for a valid distribution:
        P(C = k | h) = P_k / (P_0 + P_1 + P_2)
    Trained with NLL loss.

    Parameter count matches a linear probe: H x C x 3 = H x 192.  The
    non-negativity constraint + noisy-OR combination means each unit i
    can only VOTE FOR classes (with rate w >= 0) — no negative weights,
    no cancellation.  Interpretable as "unit i contributes to cell C
    being class k at rate w_{i,c,k}".
    """
    def __init__(self, H, C=BOARD_CELLS, n_classes=3):
        super().__init__()
        self.C = C
        self.n_classes = n_classes
        # softplus(-9) ~ 1.2e-4 → per-class score starts near 0, so all
        # classes start ~equiprobable after normalization.
        self.raw = nn.Parameter(
            torch.randn(H, C * n_classes) * 0.01 - 9.0)

    def forward(self, h):
        w = F.softplus(self.raw)           # (H, C * K)
        lam = h @ w                        # (N, C * K)  all >= 0
        P = 1.0 - torch.exp(-lam)          # (N, C * K)
        P = P.view(-1, self.C, self.n_classes)
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-9)
        return P


def train_probe_state_probor(H_tr, S_tr, epochs=25, lr=0.05, batch=512,
                                 weight_decay=0.0, device=None, seed=0):
    """Train the StateNoisyOrHead with NLL on the normalized class
    probabilities."""
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    probe = StateNoisyOrHead(H_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                              weight_decay=weight_decay)
    N = H_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            h = H_tr[idx].to(device=device, dtype=torch.float32)
            y = S_tr[idx].to(device=device, dtype=torch.long)   # (b, 64)
            P = probe(h).clamp(1e-9, 1 - 1e-9)                   # (b, 64, 3)
            loss = F.nll_loss(torch.log(P).view(-1, 3), y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def train_probe_state_probor_ensemble(H_tr, S_tr, n_seeds=1, **kwargs):
    if n_seeds <= 1:
        return [train_probe_state_probor(
            H_tr, S_tr, seed=0, **kwargs)]
    return [train_probe_state_probor(H_tr, S_tr, seed=s, **kwargs)
            for s in range(n_seeds)]


def evaluate_state_probor_ensemble(probes, H, S, T=None, batch=512):
    """Average per-cell class probabilities across seeds; argmax.

    Returns (mean_acc, per_cell_acc, by_ply, per_seed_accs).
    Matches the signature of evaluate_ensemble.
    """
    device = next(probes[0].parameters()).device
    N = H.shape[0]
    accum = torch.zeros(N, BOARD_CELLS, 3, dtype=torch.float32)
    S_cpu = S.cpu() if S.is_cuda else S
    for i in range(0, N, batch):
        h = H[i:i + batch].to(device=device, dtype=torch.float32)
        P = torch.zeros(h.shape[0], BOARD_CELLS, 3,
                          dtype=torch.float32, device=device)
        for probe in probes:
            with torch.no_grad():
                P = P + probe(h)
        P = P / len(probes)
        accum[i:i + batch] = P.cpu()
    preds = accum.argmax(dim=-1)
    correct = (preds == S_cpu).float()
    per_cell = correct.mean(dim=0).numpy()
    mean_acc = correct.mean().item()
    by_ply = {}
    if T is not None:
        T_cpu = T.cpu() if T.is_cuda else T
        for ply in range(int(T_cpu.min().item()),
                            int(T_cpu.max().item()) + 1):
            mask = (T_cpu == ply)
            if mask.any():
                by_ply[ply] = (int(mask.sum().item()),
                                correct[mask].mean().item())
    # per-seed accuracies (independent argmax per probe)
    per_seed = []
    for probe in probes:
        preds_s = torch.empty(N, BOARD_CELLS, dtype=torch.long)
        with torch.no_grad():
            for i in range(0, N, batch):
                h = H[i:i + batch].to(device=device, dtype=torch.float32)
                preds_s[i:i + batch] = probe(h).argmax(dim=-1).cpu()
        per_seed.append((preds_s == S_cpu).float().mean().item())
    return mean_acc, per_cell, by_ply, per_seed


def train_probe_legal_probor(H_tr, L_tr, epochs=25, lr=0.05, batch=512,
                                 weight_decay=0.0, device=None, seed=0):
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    probe = NoisyOrHead(H_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                              weight_decay=weight_decay)
    bce = nn.BCELoss()
    N = H_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            h = H_tr[idx].to(device=device, dtype=torch.float32)
            l = L_tr[idx].to(device=device, dtype=torch.float32)
            probs = probe(h).clamp(1e-6, 1 - 1e-6)
            loss = bce(probs, l)
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def train_probe_legal_probor_ensemble(H_tr, L_tr, n_seeds=1, **kwargs):
    if n_seeds <= 1:
        return [train_probe_legal_probor(H_tr, L_tr, seed=0, **kwargs)]
    return [train_probe_legal_probor(H_tr, L_tr, seed=s, **kwargs)
            for s in range(n_seeds)]


def evaluate_legal_ensemble(probes, H, L, T=None, batch=512, kind='bce'):
    """Ensemble by averaging per-cell probability across seeds; threshold 0.5.
    kind='bce' → sigmoid(logits); kind='probor' → probe(h) already probs.

    Returns (mean_acc, per_cell_acc, {by_ply: ..., pos_perfect: ...})."""
    device = next(probes[0].parameters()).device
    N = H.shape[0]
    accum = torch.zeros(N, BOARD_CELLS, dtype=torch.float32)
    for i in range(0, N, batch):
        h = H[i:i + batch].to(device=device, dtype=torch.float32)
        probs = torch.zeros(h.shape[0], BOARD_CELLS,
                              dtype=torch.float32, device=device)
        for probe in probes:
            with torch.no_grad():
                out = probe(h)
                if kind == 'bce':
                    out = torch.sigmoid(out)
                probs = probs + out
        probs = probs / len(probes)
        accum[i:i + batch] = probs.cpu()
    preds = (accum > 0.5).to(torch.uint8)
    L_cpu = torch.from_numpy(L) if isinstance(L, np.ndarray) else L.cpu()
    correct = (preds == L_cpu).float()               # (N, 64)
    per_cell_acc = correct.mean(dim=0).numpy()
    mean_acc = correct.mean().item()
    aux = {'position_perfect':
             correct.all(dim=1).float().mean().item()}
    if T is not None:
        T_np = T.numpy() if hasattr(T, 'numpy') else T
        aux['by_ply'] = _by_ply_10(T_np, correct.mean(dim=1).numpy())
    return mean_acc, per_cell_acc, aux


_DIRS_LEGAL = ((-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1))


def state_to_legal(state_argmax):
    """Given (N, 64) state predictions in {0=empty, 1=mine, 2=opp}, return
    (N, 64) uint8 mask of legal next moves for the mover.

    Cell C is legal iff state[C] == empty AND for some direction there is
    ≥1 opp cell immediately in that direction, followed by a mine cell
    (within board bounds)."""
    N = state_argmax.shape[0]
    grid = state_argmax.reshape(N, 8, 8)
    L = np.zeros((N, 64), dtype=np.uint8)
    for c in range(64):
        r0, c0 = c // 8, c % 8
        empty = grid[:, r0, c0] == 0
        cell_flank = np.zeros(N, dtype=bool)
        for dr, dc in _DIRS_LEGAL:
            r1, c1 = r0 + dr, c0 + dc
            if not (0 <= r1 < 8 and 0 <= c1 < 8):
                continue
            opp_streak = (grid[:, r1, c1] == 2)
            found = np.zeros(N, dtype=bool)
            for step in range(2, 8):
                rs, cs = r0 + step * dr, c0 + step * dc
                if not (0 <= rs < 8 and 0 <= cs < 8):
                    break
                cell = grid[:, rs, cs]
                found |= opp_streak & (cell == 1)
                opp_streak = opp_streak & (cell == 2)
            cell_flank |= found
        L[:, c] = (empty & cell_flank).astype(np.uint8)
    return L


def legal_accuracy_from_state(state_preds, L, T=None):
    """Compute Option-C legal-move accuracy: derive legal moves from
    state argmax preds and compare to ground-truth L."""
    S_np = state_preds if isinstance(state_preds, np.ndarray) \
        else state_preds.cpu().numpy()
    L_np = L if isinstance(L, np.ndarray) else L.cpu().numpy()
    L_pred = state_to_legal(S_np)
    correct = (L_pred == L_np)                       # (N, 64) bool
    per_cell = correct.mean(axis=0)
    mean = float(correct.mean())
    aux = {'position_perfect': float(correct.all(axis=1).mean())}
    if T is not None:
        T_np = T.numpy() if hasattr(T, 'numpy') else T
        aux['by_ply'] = _by_ply_10(T_np, correct.mean(axis=1))
    return mean, per_cell, aux


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
    ap.add_argument('--when-bucket-size', type=int, default=None,
                    help='If set, add turn-range bucket features to input.  '
                          'For each non-center cell C and each bucket k '
                          '(covering turns [k*B, (k+1)*B)), one bit that '
                          'is 1 iff C was placed in that bucket.  Input '
                          'dim becomes 121 + 60 * ceil(60/B).')
    ap.add_argument('--use-move-grid', action='store_true',
                    help='If set, add 3600 move-grid features to input '
                          '(bit per (turn, non-center-cell)).  Mutually '
                          'exclusive with --when-bucket-size.  Direct '
                          'temporal encoding, one bit per specific move '
                          'event.  Input dim becomes 3721.')
    ap.add_argument('--tree-max-features', default=None,
                    help='Passed to sklearn DecisionTreeClassifier.  '
                          "'sqrt' considers only sqrt(D) features per "
                          "split (huge speedup at high D).  'log2', an "
                          "integer, or a float in (0,1] also accepted.  "
                          'Default None = all features (slow at D>500).')
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
            args.num_train_games, max_ply=args.max_ply, seed=args.seed,
            when_bucket_size=args.when_bucket_size,
            use_move_grid=args.use_move_grid)
        print(f'  train={Xnp_tr.shape[0]}  ({time.time() - t0:.1f}s)')

    t0 = time.time()
    print(f'sampling {args.num_test_games} test games...')
    Xnp_te, Snp_te, _, Tnp_te = load_or_sample(
        args.cache_te, sample_opening_positions,
        args.num_test_games, max_ply=args.max_ply,
        seed=args.seed + 1_000_000,
        when_bucket_size=args.when_bucket_size,
        use_move_grid=args.use_move_grid)
    print(f'  test={Xnp_te.shape[0]}  ({time.time() - t0:.1f}s)')

    # --- Train per-cell decision trees ---
    print(f'\ntraining per-cell trees (max_depth={args.tree_max_depth}, '
           f'min_samples_leaf={args.tree_min_samples_leaf}, '
           f'n_jobs={args.tree_n_jobs})...')
    t0 = time.time()
    # Parse --tree-max-features: 'sqrt', 'log2', int, or float
    mf = args.tree_max_features
    if mf is not None and mf not in ('sqrt', 'log2', 'auto'):
        try:
            mf = int(mf)
        except ValueError:
            try:
                mf = float(mf)
            except ValueError:
                pass
    trees = train_per_cell_trees(
        Xnp_tr, Snp_tr,
        max_depth=args.tree_max_depth,
        min_samples_leaf=args.tree_min_samples_leaf,
        n_jobs=args.tree_n_jobs,
        max_features=mf)
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
    input_dim = Xnp_tr.shape[1]
    for c in range(BOARD_CELLS):
        paths = extract_paths(trees[c])
        n_all = len(paths)
        paths = prune_paths_by_count(paths, args.top_k_per_cell)
        per_cell_leaf_counts[c] = len(paths)
        for path_idx, (conditions, leaf_class, leaf_counts) in enumerate(paths):
            w, b = path_to_weight(conditions, input_dim=input_dim)
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

    # Compute per-unit firing rate WITHOUT materializing an (N, H) float or
    # int64 copy of H_tr — chunk over the row axis with int64 accumulator.
    counts = torch.zeros(H_tr.shape[1], dtype=torch.int64)
    chunk_rows = 512
    for i in range(0, H_tr.shape[0], chunk_rows):
        # Convert only the small chunk to int64 (chunk_rows × H × 8 bytes).
        counts += H_tr[i:i + chunk_rows].to(torch.int64).sum(dim=0)
    fire_rate = counts.float() / H_tr.shape[0]
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
