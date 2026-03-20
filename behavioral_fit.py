"""Stage 3: Fit regression trees and extract heuristics.

Three methods for each cell:
  1. two_level: L1 regression tree on random data, L2 classification trees
     per promoting leaf with beam adversarial + 100x weight
  2. beam_weighted: single regression tree on random + beam adversarial at 100x
  3. natural_weighted: single regression tree on random + natural adversarial at 100x

Usage:
    python behavioral_fit.py --cell 0 --data-dir behavioral_data
    python behavioral_fit.py --cell 0 --data-dir behavioral_data --subsample 1000  # test
"""

import argparse
import json
import os
import pickle
import sys
import time
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from behavioral_utils import (
    N_MOVES, VALID_MOVES, MOVE_TO_IDX, IDX_TO_MOVE
)


# =============================================================================
# Data loading
# =============================================================================

def load_subsample_from_shards(data_dir, cell, subsample_per_shard=500000,
                               natural_adv_threshold=0.005):
    """Load subsampled data from Stage 1 shards.

    Returns:
        X: (n_subsample, 120) float32 features
        y: (n_subsample,) float32 probabilities for target cell
        legal_flags: (n_subsample,) bool - is target cell legal?
        natural_adv_X: (n_nat, 120) float32 natural adversarial features
        natural_adv_y: (n_nat,) float32 natural adversarial probabilities
        natural_adv_legal: (n_nat,) bool
    """
    X_list, y_list, legal_list = [], [], []
    nat_X_list, nat_y_list, nat_legal_list = [], [], []

    shard_files = sorted([f for f in os.listdir(data_dir)
                          if f.startswith("shard_") and f.endswith(".npz")])

    for shard_file in shard_files:
        path = os.path.join(data_dir, shard_file)
        data = np.load(path)
        features = data['features'].astype(np.float32)  # (n, 120)
        probs = data['probs'].astype(np.float32)  # (n, 60)
        legal = data['legal']  # (n, 60)

        cell_prob = probs[:, cell]  # (n,)
        cell_legal = legal[:, cell].astype(bool)  # (n,)

        n = len(features)

        # Subsample
        if subsample_per_shard < n:
            idx = np.random.choice(n, subsample_per_shard, replace=False)
            X_list.append(features[idx])
            y_list.append(cell_prob[idx])
            legal_list.append(cell_legal[idx])
        else:
            X_list.append(features)
            y_list.append(cell_prob)
            legal_list.append(cell_legal)

        # Collect natural adversarial: illegal but model assigns > threshold
        nat_mask = (~cell_legal) & (cell_prob > natural_adv_threshold)
        if nat_mask.any():
            nat_X_list.append(features[nat_mask])
            nat_y_list.append(cell_prob[nat_mask])
            nat_legal_list.append(cell_legal[nat_mask])

        del data, features, probs, legal

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    legal_flags = np.concatenate(legal_list)

    if nat_X_list:
        nat_X = np.concatenate(nat_X_list)
        nat_y = np.concatenate(nat_y_list)
        nat_legal = np.concatenate(nat_legal_list)
    else:
        nat_X = np.zeros((0, 120), dtype=np.float32)
        nat_y = np.zeros(0, dtype=np.float32)
        nat_legal = np.zeros(0, dtype=bool)

    return X, y, legal_flags, nat_X, nat_y, nat_legal


def load_beam_adversarial(data_dir, cell):
    """Load beam search adversarial positions from Stage 2."""
    path = os.path.join(data_dir, "adversarial", f"cell_{cell:02d}.npz")
    if not os.path.exists(path):
        print(f"  Warning: no beam adversarial data at {path}")
        return (np.zeros((0, 120), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=bool))

    data = np.load(path)
    features = data['features'].astype(np.float32)
    probs_full = data['probs'].astype(np.float32)  # (n, 60)
    legal = data['legal']  # (n, 60)

    beam_y = probs_full[:, cell]
    beam_legal = legal[:, cell].astype(bool)

    return features, beam_y, beam_legal


# =============================================================================
# Tree fitting
# =============================================================================

def fit_tree_regressor(X, y, sample_weight=None, max_depth=6,
                       min_samples_leaf=500):
    """Fit a regression tree predicting model probability."""
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=0.0001,
    )
    tree.fit(X, y, sample_weight=sample_weight)
    return tree


def fit_tree_classifier(X, y_legal, sample_weight=None, max_depth=4,
                        min_samples_leaf=100):
    """Fit a classification tree predicting legality."""
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=0.001,
    )
    tree.fit(X, y_legal.astype(int), sample_weight=sample_weight)
    return tree


# =============================================================================
# Stats computation (streaming over shards)
# =============================================================================

def compute_leaf_stats_streaming(tree, data_dir, cell):
    """Stream all shards through a fitted tree, accumulate per-leaf stats.

    Returns dict: leaf_id -> {count, sum_prob, sum_legal}
    """
    stats = {}
    shard_files = sorted([f for f in os.listdir(data_dir)
                          if f.startswith("shard_") and f.endswith(".npz")])

    for shard_file in shard_files:
        path = os.path.join(data_dir, shard_file)
        data = np.load(path)
        features = data['features'].astype(np.float32)
        cell_prob = data['probs'].astype(np.float32)[:, cell]
        cell_legal = data['legal'][:, cell].astype(bool)

        leaf_ids = tree.apply(features)

        for lid in np.unique(leaf_ids):
            mask = leaf_ids == lid
            if lid not in stats:
                stats[lid] = {'count': 0, 'sum_prob': 0.0, 'sum_legal': 0}
            stats[lid]['count'] += int(mask.sum())
            stats[lid]['sum_prob'] += float(cell_prob[mask].sum())
            stats[lid]['sum_legal'] += int(cell_legal[mask].sum())

        del data, features

    return stats


# =============================================================================
# Heuristic extraction
# =============================================================================

def feature_name(idx):
    """Convert feature index to human-readable name."""
    if idx < N_MOVES:
        board_pos = VALID_MOVES[idx]
        r, c = board_pos // 8, board_pos % 8
        rows = "ABCDEFGH"
        return f"played_{rows[r]}{c+1}"
    else:
        move_idx = idx - N_MOVES
        board_pos = VALID_MOVES[move_idx]
        r, c = board_pos // 8, board_pos % 8
        rows = "ABCDEFGH"
        return f"even_{rows[r]}{c+1}"


def extract_tree_path(tree, leaf_id):
    """Extract the conjunction of conditions from root to a leaf."""
    from sklearn.tree import _tree

    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right

    # Find path from root to leaf
    def _find_path(node_id, target_leaf, path):
        if node_id == target_leaf:
            return True
        if children_left[node_id] == _tree.TREE_LEAF:
            return False
        # Try left child
        if _find_path(children_left[node_id], target_leaf,
                      path + [(feature[node_id], '<=', threshold[node_id])]):
            path.extend([(feature[node_id], '<=', threshold[node_id])])
            return True
        # Try right child
        if _find_path(children_right[node_id], target_leaf,
                      path + [(feature[node_id], '>', threshold[node_id])]):
            path.extend([(feature[node_id], '>', threshold[node_id])])
            return True
        return False

    # Simpler approach: walk the tree tracking decisions
    path = []
    node = 0
    while node != leaf_id:
        left = children_left[node]
        right = children_right[node]
        feat = feature[node]
        thresh = threshold[node]

        # Check which subtree contains the leaf
        if _subtree_contains(children_left, children_right, left, leaf_id):
            path.append({
                'feature': feature_name(feat),
                'feature_idx': int(feat),
                'direction': '<=',
                'threshold': float(thresh),
            })
            node = left
        elif _subtree_contains(children_left, children_right, right, leaf_id):
            path.append({
                'feature': feature_name(feat),
                'feature_idx': int(feat),
                'direction': '>',
                'threshold': float(thresh),
            })
            node = right
        else:
            break

    return path


def _subtree_contains(children_left, children_right, root, target):
    """Check if target leaf is in the subtree rooted at root."""
    from sklearn.tree import _tree
    if root == target:
        return True
    if children_left[root] == _tree.TREE_LEAF:
        return False
    return (_subtree_contains(children_left, children_right,
                              children_left[root], target) or
            _subtree_contains(children_left, children_right,
                              children_right[root], target))


def make_conjunction_readable(path):
    """Convert path conditions to human-readable string."""
    parts = []
    for cond in path:
        name = cond['feature']
        if cond['direction'] == '<=' and cond['threshold'] == 0.5:
            # Binary feature <= 0.5 means NOT played/even
            parts.append(f"NOT {name}")
        elif cond['direction'] == '>' and cond['threshold'] == 0.5:
            parts.append(name)
        else:
            parts.append(f"{name} {cond['direction']} {cond['threshold']:.2f}")
    return " AND ".join(parts)


# =============================================================================
# Main fitting logic
# =============================================================================

def fit_method_two_level(X_random, y_random, legal_random,
                         beam_X, beam_y, beam_legal,
                         data_dir, cell):
    """Method 1: Two-level tree approach."""
    # L1: fit on random data only
    tree_L1 = fit_tree_regressor(X_random, y_random)

    # Compute stats by streaming all shards
    stats = compute_leaf_stats_streaming(tree_L1, data_dir, cell)

    # Get leaf IDs
    from sklearn.tree import _tree
    leaves = []
    def _find_leaves(node):
        if tree_L1.tree_.children_left[node] == _tree.TREE_LEAF:
            leaves.append(node)
        else:
            _find_leaves(tree_L1.tree_.children_left[node])
            _find_leaves(tree_L1.tree_.children_right[node])
    _find_leaves(0)

    heuristics = []
    l2_trees = {}

    for leaf_id in leaves:
        if leaf_id not in stats:
            continue
        s = stats[leaf_id]
        count = s['count']
        if count == 0:
            continue

        avg_prob = s['sum_prob'] / count
        precision = s['sum_legal'] / count
        path = extract_tree_path(tree_L1, leaf_id)

        heuristic = {
            'leaf_id': int(leaf_id),
            'conjunction': path,
            'conjunction_readable': make_conjunction_readable(path),
            'avg_model_prob': float(avg_prob),
            'support': count,
            'precision': float(precision),
            'error_rate': float(1 - precision),
            'num_errors': count - s['sum_legal'],
        }

        # Classify
        if avg_prob > 0.02 and count > 500:
            heuristic['type'] = 'promoting'
            if precision > 0.90:
                heuristic['reliability'] = 'reliable'
            elif precision > 0.60:
                heuristic['reliability'] = 'noisy'
            else:
                heuristic['reliability'] = 'broken'

            # Fit L2 tree if there are errors
            if precision < 1.0 and len(beam_X) > 0:
                # Get random positions in this leaf
                leaf_mask_random = tree_L1.apply(X_random) == leaf_id
                X_leaf_r = X_random[leaf_mask_random]
                legal_leaf_r = legal_random[leaf_mask_random]

                # Get beam adversarial positions in this leaf
                if len(beam_X) > 0:
                    leaf_mask_beam = tree_L1.apply(beam_X) == leaf_id
                    X_leaf_b = beam_X[leaf_mask_beam]
                    legal_leaf_b = beam_legal[leaf_mask_beam]
                else:
                    X_leaf_b = np.zeros((0, 120), dtype=np.float32)
                    legal_leaf_b = np.zeros(0, dtype=bool)

                if len(X_leaf_r) > 100 and (len(X_leaf_b) > 0 or
                                             (~legal_leaf_r).sum() > 10):
                    X_l2 = np.concatenate([X_leaf_r, X_leaf_b])
                    y_l2 = np.concatenate([legal_leaf_r, legal_leaf_b])
                    weights = np.ones(len(X_l2))
                    weights[len(X_leaf_r):] = 100.0

                    tree_L2 = fit_tree_classifier(X_l2, y_l2,
                                                  sample_weight=weights)
                    l2_trees[leaf_id] = tree_L2

                    # Extract unless conditions from L2 leaves
                    unless = []
                    l2_leaves = []
                    def _find_l2_leaves(node):
                        if tree_L2.tree_.children_left[node] == -1:
                            l2_leaves.append(node)
                        else:
                            _find_l2_leaves(tree_L2.tree_.children_left[node])
                            _find_l2_leaves(tree_L2.tree_.children_right[node])
                    _find_l2_leaves(0)

                    for l2_leaf in l2_leaves:
                        l2_ids = tree_L2.apply(X_l2)
                        in_leaf = l2_ids == l2_leaf
                        if in_leaf.sum() == 0:
                            continue
                        sub_precision = float(y_l2[in_leaf].mean())
                        # Only report sub-leaves with lower precision
                        if sub_precision < precision - 0.05:
                            l2_path = extract_tree_path(tree_L2, l2_leaf)
                            adv_count = int(in_leaf[len(X_leaf_r):].sum())
                            unless.append({
                                'conditions': l2_path,
                                'conditions_readable': make_conjunction_readable(l2_path),
                                'sub_precision': sub_precision,
                                'sub_support': int(in_leaf.sum()),
                                'adversarial_count': adv_count,
                            })

                    if unless:
                        heuristic['unless_conditions'] = unless

        elif avg_prob < 0.005 and count > 500 and precision > 0.30:
            heuristic['type'] = 'suppressing'
            heuristic['reliability'] = 'suppressing'
        else:
            heuristic['type'] = 'neutral'
            heuristic['reliability'] = 'neutral'

        heuristics.append(heuristic)

    return tree_L1, l2_trees, heuristics


def fit_method_weighted(X_random, y_random, legal_random,
                        adv_X, adv_y, adv_legal,
                        data_dir, cell, method_name):
    """Methods 2 & 3: Single weighted tree."""
    if len(adv_X) > 0:
        X = np.concatenate([X_random, adv_X])
        y = np.concatenate([y_random, adv_y])
        legal = np.concatenate([legal_random, adv_legal])
        weights = np.ones(len(X))
        weights[len(X_random):] = 100.0
    else:
        X, y, legal, weights = X_random, y_random, legal_random, None

    tree = fit_tree_regressor(X, y, sample_weight=weights)

    # Compute stats by streaming (on random data only)
    stats = compute_leaf_stats_streaming(tree, data_dir, cell)

    # Extract heuristics
    from sklearn.tree import _tree
    leaves = []
    def _find_leaves(node):
        if tree.tree_.children_left[node] == _tree.TREE_LEAF:
            leaves.append(node)
        else:
            _find_leaves(tree.tree_.children_left[node])
            _find_leaves(tree.tree_.children_right[node])
    _find_leaves(0)

    heuristics = []
    for leaf_id in leaves:
        if leaf_id not in stats:
            continue
        s = stats[leaf_id]
        count = s['count']
        if count == 0:
            continue

        avg_prob = s['sum_prob'] / count
        precision = s['sum_legal'] / count
        path = extract_tree_path(tree, leaf_id)

        heuristic = {
            'leaf_id': int(leaf_id),
            'conjunction': path,
            'conjunction_readable': make_conjunction_readable(path),
            'avg_model_prob': float(avg_prob),
            'support': count,
            'precision': float(precision),
            'error_rate': float(1 - precision),
            'num_errors': count - s['sum_legal'],
        }

        if avg_prob > 0.02 and count > 500:
            heuristic['type'] = 'promoting'
            if precision > 0.90:
                heuristic['reliability'] = 'reliable'
            elif precision > 0.60:
                heuristic['reliability'] = 'noisy'
            else:
                heuristic['reliability'] = 'broken'
        elif avg_prob < 0.005 and count > 500 and precision > 0.30:
            heuristic['type'] = 'suppressing'
            heuristic['reliability'] = 'suppressing'
        else:
            heuristic['type'] = 'neutral'
            heuristic['reliability'] = 'neutral'

        heuristics.append(heuristic)

    return tree, heuristics


def shuffle_sanity_check(X, y, max_depth=6, min_samples_leaf=500):
    """Fit tree on shuffled labels, return max leaf precision."""
    y_shuffled = y.copy()
    np.random.shuffle(y_shuffled)
    tree = fit_tree_regressor(X, y_shuffled, max_depth=max_depth,
                              min_samples_leaf=min_samples_leaf)
    # Get leaf predictions (mean y in each leaf)
    leaf_values = tree.tree_.value.flatten()
    return float(np.max(leaf_values)), float(np.mean(leaf_values))


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Fit trees + extract heuristics")
    parser.add_argument("--cell", type=int, required=True, help="Cell index (0-59)")
    parser.add_argument("--data-dir", type=str, default="behavioral_data")
    parser.add_argument("--output-dir", type=str, default="behavioral_data")
    parser.add_argument("--subsample", type=int, default=500000,
                        help="Rows to subsample per shard (default 500K)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed + args.cell)
    t0 = time.time()

    cell_name = f"{VALID_MOVES[args.cell]}"
    print(f"Cell {args.cell} (board pos {cell_name})", flush=True)

    # Load subsampled random data + natural adversarial
    print("Loading data from shards...", flush=True)
    X_random, y_random, legal_random, nat_X, nat_y, nat_legal = \
        load_subsample_from_shards(args.data_dir, args.cell, args.subsample)
    print(f"  Random: {len(X_random)} rows", flush=True)
    print(f"  Natural adversarial: {len(nat_X)} rows", flush=True)

    # Load beam adversarial
    beam_X, beam_y, beam_legal = load_beam_adversarial(args.data_dir, args.cell)
    print(f"  Beam adversarial: {len(beam_X)} rows", flush=True)

    # Shuffle sanity check
    print("Shuffle sanity check...", flush=True)
    shuf_max, shuf_mean = shuffle_sanity_check(X_random, y_random)
    print(f"  Shuffled tree: max leaf value={shuf_max:.4f}, mean={shuf_mean:.4f}")

    # Method 1: two_level
    print("\n=== Method 1: two_level ===", flush=True)
    tree_tl, l2_trees_tl, heur_tl = fit_method_two_level(
        X_random, y_random, legal_random,
        beam_X, beam_y, beam_legal,
        args.data_dir, args.cell
    )
    promoting_tl = [h for h in heur_tl if h['type'] == 'promoting']
    print(f"  {len(promoting_tl)} promoting heuristics", flush=True)

    # Method 2: beam_weighted
    print("\n=== Method 2: beam_weighted ===", flush=True)
    tree_bw, heur_bw = fit_method_weighted(
        X_random, y_random, legal_random,
        beam_X, beam_y, beam_legal,
        args.data_dir, args.cell, "beam_weighted"
    )
    promoting_bw = [h for h in heur_bw if h['type'] == 'promoting']
    print(f"  {len(promoting_bw)} promoting heuristics", flush=True)

    # Method 3: natural_weighted
    print("\n=== Method 3: natural_weighted ===", flush=True)
    if len(nat_X) < 100:
        print(f"  WARNING: only {len(nat_X)} natural adversarial positions "
              f"(<100). Method 3 may not be meaningful for this cell.", flush=True)
    tree_nw, heur_nw = fit_method_weighted(
        X_random, y_random, legal_random,
        nat_X, nat_y, nat_legal,
        args.data_dir, args.cell, "natural_weighted"
    )
    promoting_nw = [h for h in heur_nw if h['type'] == 'promoting']
    print(f"  {len(promoting_nw)} promoting heuristics", flush=True)

    # Save trees
    for method, tree in [("two_level", tree_tl), ("beam_weighted", tree_bw),
                         ("natural_weighted", tree_nw)]:
        tree_dir = os.path.join(args.output_dir, "trees", method)
        os.makedirs(tree_dir, exist_ok=True)
        with open(os.path.join(tree_dir, f"cell_{args.cell:02d}.pkl"), 'wb') as f:
            pickle.dump(tree, f)

    # Save L2 trees
    if l2_trees_tl:
        l2_dir = os.path.join(args.output_dir, "trees", "two_level",
                              f"cell_{args.cell:02d}_L2")
        os.makedirs(l2_dir, exist_ok=True)
        for leaf_id, l2_tree in l2_trees_tl.items():
            with open(os.path.join(l2_dir, f"leaf_{leaf_id}.pkl"), 'wb') as f:
                pickle.dump(l2_tree, f)

    # Save heuristics JSON (per cell)
    for method, heuristics in [("two_level", heur_tl),
                               ("beam_weighted", heur_bw),
                               ("natural_weighted", heur_nw)]:
        heur_dir = os.path.join(args.output_dir, f"heuristics_{method}")
        os.makedirs(heur_dir, exist_ok=True)

        cell_data = {
            'target_cell': args.cell,
            'target_cell_name': cell_name,
            'board_position': int(VALID_MOVES[args.cell]),
            'n_random': len(X_random),
            'n_beam_adversarial': len(beam_X),
            'n_natural_adversarial': len(nat_X),
            'shuffle_max_leaf': shuf_max,
            'shuffle_mean_leaf': shuf_mean,
            'heuristics': heuristics,
        }

        with open(os.path.join(heur_dir, f"cell_{args.cell:02d}.json"), 'w') as f:
            json.dump(cell_data, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed:.0f}s", flush=True)
    print(f"  two_level: {len(promoting_tl)} promoting", flush=True)
    print(f"  beam_weighted: {len(promoting_bw)} promoting", flush=True)
    print(f"  natural_weighted: {len(promoting_nw)} promoting", flush=True)


if __name__ == "__main__":
    main()
