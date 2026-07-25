"""Single-hidden-layer interpretable MLP for the Othello MID-GAME
(plies 10-49 by default) using per-cell decision trees, matching the
opening and endgame pipelines.

The middle game is the hard regime: state depends on the full move
history through complex chains of captures.  We use the same recipe as
opening/endgame — per-cell decision trees, paths → 0/±1 hidden units,
linear probe on the hidden layer — and measure per-ply accuracy to see
where the residual error lives.

Optional stability features can be enabled with --add-stability-features:
adds a bank of interpretable rules capturing local-homogeneity patterns
that predict when a cell hasn't been flipped recently.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.othello import OthelloBoardState
from opening_tree_mlp import (
    playedeven_features, feature_name, path_to_weight, extract_paths,
    leaf_node_ids,
    train_per_cell_trees, train_pattern_trees, train_probe, train_probe_ensemble,
    train_probe_sklearn, evaluate, evaluate_ensemble, evaluate_sklearn,
    train_probe_legal_bce_ensemble, train_probe_legal_probor_ensemble,
    train_probe_legal_patterns_structured_ensemble,
    train_probe_legal_cells_structured_ensemble,
    train_probe_legal_linear_pat_probor_ensemble,
    evaluate_legal_ensemble, legal_accuracy_from_state,
    train_probe_state_probor_ensemble, evaluate_state_probor_ensemble,
    OpeningTreeMLP,
    load_or_sample, prune_paths_by_count,
    BOARD_CELLS, INPUT_DIM, CENTER_64, NON_CENTER_64,
    C64_TO_C60, C60_TO_C64, STATE_NAMES,
)
from endgame_tree_mlp import (
    CORNERS_64, cell_class, CELL_CLASS, per_class_accuracy,
)
from count_nodes import (
    build_structured_count_nodes, build_random_count_nodes,
    build_tree_derived_count_nodes,
    build_neighborhood_count_nodes, build_ray_count_nodes,
    compute_count_activations,
)
from order_nodes import (
    movegrid_from_flat,
    build_turn_bucket, build_recency, build_ordinal,
    build_pairwise_order, build_streak,
)
from flanking_patterns import (
    load_patterns, compute_pattern_activations, patterns_by_target,
    legal_from_state_probs_via_patterns, true_pattern_activations,
)


# ------------------------------------------------------------------------------
# Mid-game position sampling
# ------------------------------------------------------------------------------

def load_games_from_pickles(num_games, pickle_dir, seed=42):
    """Load `num_games` game prefixes from the pre-generated synthetic
    pickle files.  Much faster than playing games from scratch (roughly
    100K games per pickle file, ~10 sec to load one file vs. hours to
    play 100K games)."""
    import glob
    import pickle
    files = sorted(glob.glob(os.path.join(pickle_dir, '*.pickle')))
    if not files:
        raise ValueError(f'no .pickle files in {pickle_dir}')
    print(f'  loading games from {len(files)} pickle files in {pickle_dir}')
    rng = np.random.RandomState(seed)
    # Shuffle files so the seed selection is diverse.
    files_shuffled = rng.permutation(files)
    games = []
    for path in files_shuffled:
        if len(games) >= num_games:
            break
        with open(path, 'rb') as f:
            file_games = pickle.load(f)
        games.extend(file_games)
        if len(games) % 500_000 == 0 or len(games) >= num_games:
            print(f'    loaded {len(games)} games so far...')
    games = games[:num_games]
    print(f'  loaded {len(games)} games total')
    return games


def sample_midgame_positions(num_games, ply_min=0, ply_max=60, seed=42,
                                when_bucket_size=None,
                                use_move_grid=False,
                                recent_Ks=None,
                                collect_legal_moves=False,
                                canonicalize_mover=False,
                                time_ordinal=None,
                                pickle_dir=None):
    """Play random games; extract positions with ply in [ply_min, ply_max).
    Returns (X, S, T) — or (X, S, T, L) if collect_legal_moves — with:
      X: (N, ...) played_even features
      S: (N, 64) int64 state labels (0 empty, 1 mine, 2 opp)
      T: (N,) int32 — ply index within the game
      L: (N, 64) uint8 legal-move mask (1 iff cell is a legal next move)
    """
    rng = np.random.RandomState(seed)
    Xs, Ss, Ts, Ls = [], [], [], []
    # Choose game source: pre-generated pickles (fast, real-play) OR
    # live random-play (slow, deterministic given seed).
    if pickle_dir is not None:
        pregen_games = load_games_from_pickles(num_games, pickle_dir,
                                                    seed=seed)
    else:
        pregen_games = None
    for g_idx in range(num_games):
        board = OthelloBoardState()
        prefix = []
        if pregen_games is not None:
            # Replay the loaded game move-by-move.
            game_moves = pregen_games[g_idx]
            for move in game_moves:
                valid = board.get_valid_moves()
                if not valid:
                    board.update([])
                    valid = board.get_valid_moves()
                    if not valid:
                        break
                ply = len(prefix)
                if ply_min <= ply < ply_max:
                    parity = ply % 2
                    mover_color = 1 if parity == 0 else -1
                    raw = board.state.flatten().astype(np.int8)
                    lbl = np.zeros(BOARD_CELLS, dtype=np.int64)
                    lbl[raw == mover_color] = 1
                    lbl[raw == -mover_color] = 2
                    Xs.append(playedeven_features(
                        prefix, when_bucket_size,
                        use_move_grid,
                        recent_Ks=recent_Ks,
                        canonicalize_mover=canonicalize_mover,
                        time_ordinal=time_ordinal))
                    Ss.append(lbl)
                    Ts.append(ply)
                    if collect_legal_moves:
                        lmask = np.zeros(BOARD_CELLS, dtype=np.uint8)
                        for m in valid:
                            lmask[m] = 1
                        Ls.append(lmask)
                if move not in valid:
                    # The pre-generated game may include a move that is
                    # not valid in this board — should not happen for
                    # well-formed synthetic data, but bail defensively.
                    break
                board.update([move])
                prefix.append(move)
        else:
            for turn in range(60):
                valid = board.get_valid_moves()
                if not valid:
                    board.update([])
                    valid = board.get_valid_moves()
                    if not valid:
                        break
                ply = len(prefix)
                if ply_min <= ply < ply_max:
                    parity = turn % 2
                    mover_color = 1 if parity == 0 else -1
                    raw = board.state.flatten().astype(np.int8)
                    lbl = np.zeros(BOARD_CELLS, dtype=np.int64)
                    lbl[raw == mover_color] = 1
                    lbl[raw == -mover_color] = 2
                    Xs.append(playedeven_features(
                        prefix, when_bucket_size,
                        use_move_grid,
                        recent_Ks=recent_Ks,
                        canonicalize_mover=canonicalize_mover,
                        time_ordinal=time_ordinal))
                    Ss.append(lbl)
                    Ts.append(ply)
                    if collect_legal_moves:
                        lmask = np.zeros(BOARD_CELLS, dtype=np.uint8)
                        for m in valid:
                            lmask[m] = 1
                        Ls.append(lmask)
                move = valid[rng.randint(len(valid))]
                board.update([move])
                prefix.append(move)
    if not Xs:
        raise RuntimeError(
            f'no positions extracted; ply range [{ply_min},{ply_max}) '
            f'may not overlap game play')
    X = np.stack(Xs); S = np.stack(Ss); T = np.array(Ts, dtype=np.int32)
    if collect_legal_moves:
        return X, S, T, np.stack(Ls)
    return X, S, T


# ------------------------------------------------------------------------------
# Stability feature bank (optional, approach 2)
# ------------------------------------------------------------------------------

def _neighbors_8(c64):
    """Return the up-to-8 orthogonal + diagonal neighbors of a 64-cell index."""
    r, c = c64 // 8, c64 % 8
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                out.append(nr * 8 + nc)
    return out


def build_stability_features(device):
    """Build a bank of interpretable "local-stability" hidden units.

    For each non-center cell C and each placement parity P ∈ {0, 1}, one
    unit fires iff:
      - C was played at parity P (placed at that color)
      - AND every non-center neighbor of C that has been played was
        placed at parity P as well  (no opposite-parity neighbor)

    Interpretation: cell C has been placed and no neighbor is a candidate
    flanking-capture initiator against C from that side.  Not a proof of
    "hasn't been flipped" (longer capture chains exist), but a useful
    stability hint the tree pipeline lacks.

    Encoding is only over played_even (120-d) inputs, so this is a strict
    conjunction over primitives already available.  Each unit is one rule
    with 0/±1 weights and a scalar bias.
    """
    all_w, all_b, meta = [], [], []
    for c60 in range(60):
        c64 = C60_TO_C64[c60]
        played_idx = c60
        even_idx = 60 + c60

        for p in (0, 1):
            w = np.zeros(INPUT_DIM, dtype=np.float32)
            # "cell C was played at parity P":
            #   played=1, even=P
            #   score = played + (1 if P==1 else -1) * even
            # Target contribution 2 (P=1) or 1 (P=0)... let's normalize so
            # any single deviation reduces target by ≥ 1.
            if p == 1:
                w[played_idx] += 1
                w[even_idx] += 1
                target = 2
            else:
                w[played_idx] += 1
                w[even_idx] -= 1
                target = 1

            # For each neighbor N that IS non-center: contribute a term that
            # is 0 iff (not played) or (played at parity P), and negative
            # otherwise (played at parity ≠ P → potential flanker).
            #   played=0             → played=0, even=0 → contribution = 0
            #   played=1, even=P     → contribution = 0
            #   played=1, even≠P     → contribution = -1
            for n64 in _neighbors_8(c64):
                if n64 in CENTER_64:
                    continue
                n60 = C64_TO_C60[n64]
                n_played = n60
                n_even = 60 + n60
                if p == 1:
                    # "neighbor NOT (played AND even=0)":
                    # penalize played=1 AND even=0 → -played + even (max 0)
                    #   played=0, even=0 → 0
                    #   played=1, even=1 → 0
                    #   played=1, even=0 → -1
                    w[n_played] -= 1
                    w[n_even] += 1
                else:  # p == 0
                    # penalize played=1 AND even=1
                    #   contribution: -played -even + played*(1 if even==0 else 0) …
                    # Simplest: -even (played=1, even=1 → -1; else ≤ 0)
                    #   played=0, even=0 → 0
                    #   played=1, even=0 → 0
                    #   played=1, even=1 → -1
                    # even=1 without played=1 is impossible (played_even
                    # invariant), so this is fine.
                    w[n_even] -= 1
            bias = -(target - 0.5)
            all_w.append(w)
            all_b.append(bias)
            meta.append({
                'kind': 'stability',
                'cell': c64,
                'parity': p,
                'neighbors': [n for n in _neighbors_8(c64)
                              if n not in CENTER_64],
            })
    W = np.stack(all_w).astype(np.float32)
    B = np.array(all_b, dtype=np.float32)
    return W, B, meta


def fit_multioutput_pattern_trees(Xnp_tr, pt_tr, Xnp_te, pt_te, patterns_list,
                                    args, max_features, class_weight):
    """Fit multi-output pattern tree(s) jointly over GROUPS of patterns.

    mode=multioutput → one tree over all 960 targets; mode=grouped → one tree
    per target cell (over that cell's ~15 patterns).  A joint tree's splits are
    shared across its targets, so its leaves become board-state regions
    informative about many patterns at once (a decorrelated basis for the
    readout), at the cost of per-pattern sharpness.

    Returns (multi_trees, tree_correct_per_pattern):
      multi_trees: list of (sklearn_tree, group_label, pattern_col_ids)
      tree_correct_per_pattern: (K,) per-pattern test accuracy.
    """
    from sklearn.tree import DecisionTreeClassifier
    K = len(patterns_list)
    if args.pattern_tree_mode == 'multioutput':
        groups = [('all', list(range(K)))]
    else:  # grouped by target cell
        from collections import defaultdict
        by_cell = defaultdict(list)
        for j, p in enumerate(patterns_list):
            by_cell[p['target']].append(j)
        groups = [(f'cell{c}', cols) for c, cols in sorted(by_cell.items())]
    # class_weight='balanced' is unsafe for many-output trees: sklearn takes the
    # PRODUCT of per-output balanced weights, so with hundreds of ~1.5%-positive
    # targets the sample weights underflow to 0 and the tree degenerates to a
    # single leaf.  Disable it for joint trees (rare-pattern upweighting is
    # inherently lost when targets are shared).
    if class_weight is not None:
        print('  NOTE: class_weight disabled for multi-output '
               '(product-of-weights underflows across many targets)')
    print(f'  fitting {len(groups)} multi-output tree(s) '
           f'(mode={args.pattern_tree_mode}; '
           f'{"all 960 targets" if len(groups) == 1 else f"~{K // len(groups)} targets/tree"})...')
    multi_trees = []
    tree_correct_per_pattern = np.zeros(K)
    for label, cols in groups:
        t = DecisionTreeClassifier(
            max_depth=args.tree_max_depth,
            min_samples_leaf=args.tree_min_samples_leaf,
            max_leaf_nodes=args.max_leaf_nodes,
            max_features=max_features,
            class_weight=None,
            random_state=0)
        y = pt_tr[:, cols]
        t.fit(Xnp_tr, y.ravel() if len(cols) == 1 else y)
        pred = t.predict(Xnp_te)
        if pred.ndim == 1:
            pred = pred[:, None]
        for k, col in enumerate(cols):
            tree_correct_per_pattern[col] = (pred[:, k] == pt_te[:, col]).mean()
        multi_trees.append((t, label, cols))
    print(f'  multi-output per-pattern test acc: '
           f'{100 * tree_correct_per_pattern.mean():.4f}% (mean over {K})')
    return multi_trees, tree_correct_per_pattern


def build_H_from_leaves(leaf_build, Xnp, dtype=torch.bool):
    """Hidden layer as TRUE leaf one-hot (honors real/numeric thresholds).

    leaf_build[c] = (sklearn_tree, leaf_node_id).  Column c fires iff the
    sample's tree.apply() lands on that leaf.  Pruned leaves aren't columns,
    so a sample whose leaf was pruned is all-zero across that tree's columns.
    Unlike the ±1 W·x linearization (exact only for binary features), this is
    correct for numeric splits (e.g. --time-ordinal bands).  Calls tree.apply
    once per distinct tree (grouped by id())."""
    from collections import defaultdict
    N = Xnp.shape[0]
    H = np.zeros((N, len(leaf_build)), dtype=bool)
    cols_by_tree = defaultdict(list)
    tref = {}
    for col, (tree, nid) in enumerate(leaf_build):
        if nid is None:
            raise ValueError('--hidden-from-leaves needs tree-based algorithms '
                              '(dt/rf/et/gbm) with sklearn node ids; got a '
                              'rule-based (skope) tree with no leaf nodes.')
        cols_by_tree[id(tree)].append((col, nid))
        tref[id(tree)] = tree
    for tid, colnodes in cols_by_tree.items():
        leaves = tref[tid].apply(Xnp)          # (N,) leaf node id per sample
        for col, nid in colnodes:
            H[:, col] = (leaves == nid)
    return torch.from_numpy(H).to(dtype)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-train-games', type=int, default=20000)
    ap.add_argument('--num-test-games', type=int, default=5000)
    ap.add_argument('--ply-min', type=int, default=0)
    ap.add_argument('--ply-max', type=int, default=60)
    ap.add_argument('--tree-max-depth', type=int, default=15)
    ap.add_argument('--tree-min-samples-leaf', type=int, default=5)
    ap.add_argument('--max-leaf-nodes', type=int, default=None,
                    help='Cap leaves per tree via BEST-FIRST growth (sklearn '
                          'max_leaf_nodes) — keeps the highest-impurity-gain '
                          'leaves.  Strongly preferred over --top-k-per-cell, '
                          'which prunes by sample count and discards the small '
                          'leaves that isolate rare pattern firings.  Applies '
                          'to per-pattern, grouped, and multioutput trees.')
    ap.add_argument('--tree-n-jobs', type=int, default=1)
    ap.add_argument('--probe-epochs', type=int, default=25)
    ap.add_argument('--add-stability-features',
                    dest='add_stability', action='store_true',
                    help='Concatenate a bank of local-stability rules '
                          '(approach 2) to the tree-path hidden units.')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='midgame_tree_mlp.pt')
    ap.add_argument('--when-bucket-size', type=int, default=None)
    ap.add_argument('--use-move-grid', action='store_true',
                    help='Add 3600 move-grid features (bit per (turn, cell)).')
    ap.add_argument('--canonicalize-mover', action='store_true',
                    help='Use mover-relative parity encoding: replace even '
                          'bit with placed_as_mover.  mover_parity bit is '
                          'zeroed since parity is baked into the per-cell '
                          'bit.  Structurally-identical color-swapped '
                          'positions get IDENTICAL feature vectors, so each '
                          'tree sees all 800k rows in one representation '
                          'rather than partitioning capacity by mover.  '
                          'Input-side analog of Nanda-style parity-split '
                          'probes.')
    ap.add_argument('--input-recent-Ks', default='',
                    help='Comma-separated K values.  For each K, appends 60 '
                          'bits (one per non-center cell) that fire iff the '
                          'cell was played in the last K turns.  Trees fit '
                          'on the enlarged input directly — no separate '
                          'order-node hidden bank needed.  E.g., "5" gives '
                          'played+even+recent = 60x3 input (+ mover_parity).')
    ap.add_argument('--time-ordinal', default='none',
                    choices=['none', 'turn', 'movesago'],
                    help='Append a 60-d ORDINAL turn-of-play block to the tree '
                          'input (one numeric feature per non-center cell; -1 if '
                          'unplayed).  "turn"=absolute ply the cell was placed; '
                          '"movesago"=T-ply (phase-invariant, pairs with '
                          '--canonicalize-mover).  Because it is numeric, a '
                          'DecisionTree splits on thresholds and learns '
                          'contiguous, data-adaptive time RANGES ("cell C played '
                          'in turns [12,13]") instead of one-hot point leaves — '
                          'min_samples_leaf/max_depth act as a time-granularity '
                          'dial.  Composes with either recency block.')
    ap.add_argument('--hidden-from-leaves', action='store_true',
                    help='Compute the hidden layer as TRUE leaf one-hot '
                          '(tree.apply) instead of the ±1 path linearization. '
                          'The linearization is only exact for binary 0/1 '
                          'features; numeric splits (e.g. --time-ordinal) '
                          'collapse thresholds to ~0.5 and cancel same-feature '
                          'bands.  Leaf one-hot honors the real thresholds, so '
                          'the tree can learn continuous, adaptive time-bands. '
                          'In-job only (needs the sklearn trees in memory): '
                          'incompatible with --load-trees-from / --add-stability, '
                          'and the resulting checkpoint is NOT usable by the '
                          'W-based streaming / reeval loaders.')
    ap.add_argument('--time-ordinal-split-color', action='store_true',
                    help='Split the --time-ordinal block into two channels per '
                          'cell (mover / opponent): each holds the play-time '
                          'only for cells of that color, -1 otherwise (120 cols '
                          'instead of 60).  Because a player moves on alternate '
                          'turns, an un-split ordinal band straddles both colors '
                          '(turns 3,4,5 = me/opp/me); splitting makes each '
                          "channel same-parity so a threshold band is color-pure "
                          'by construction — no co-gating with placed_as_mover, '
                          'no leaf fragmentation.  Requires --canonicalize-mover.')
    ap.add_argument('--include-flanking-patterns', default='',
                    help='Path to a .pt file with 960 hand-crafted flanking '
                          'patterns.  Each pattern encodes an Othello '
                          'legality rule as a conjunction on moveset+parity; '
                          'activations are concatenated to the hidden layer '
                          'as 960 extra binary units.  Combined with '
                          '--task legal or --task both to train legal-move '
                          'probes over these + tree paths.')
    ap.add_argument('--recent-Ks-as-hidden', default='',
                    help='Comma-separated K values.  Same recent bits as '
                          '--input-recent-Ks, but excluded from tree input '
                          'and instead concatenated to the hidden layer as '
                          'extra units.  Trees fit on played+even only; the '
                          'linear probe reads recency directly, bypassing '
                          'top-K pruning.  Mutually exclusive with '
                          '--input-recent-Ks.')
    ap.add_argument('--recent-split-color', action='store_true',
                    help='For --recent-Ks-as-hidden (canonical only): split '
                          'each recency bit into mover/opponent by AND-ing with '
                          'placed_as_mover, so a node means "cell c played by ME '
                          '/ by YOU within the last K moves" (2x the recency '
                          'units).  Directly tests the flip-reliability idea: a '
                          'recently-placed square is unlikely to have flipped, so '
                          'its placement color approximates its current color.')
    ap.add_argument('--tree-max-features', default=None,
                    help='Passed to sklearn: sqrt/log2/int/float.  Use with '
                          '--use-move-grid to keep tree fit tractable.')
    ap.add_argument('--hidden-activation', default='step',
                    choices=['step', 'relu'],
                    help='step (default): hidden units output bool 0/1.  '
                          'relu: hidden units output continuous excess '
                          'above threshold (max(0, count - K + 0.5)).  '
                          'ReLU preserves count magnitude the probe can '
                          'weight, closing much of the MLP gap.')
    ap.add_argument('--include-count-nodes', action='store_true',
                    help='Append structured count-node bank (~1800 units) '
                          'to the hidden layer.  Each node is "at least K '
                          'cells in region R are played at parity P".')
    ap.add_argument('--include-tree-derived-count-nodes',
                    action='store_true',
                    help='Append count-node bank derived from the tree '
                          'paths: for each output cell C, count features '
                          'over the union of cells appearing in C\'s top-K '
                          'tree paths.  Each node ties directly to a '
                          'specific decoding decision.')
    ap.add_argument('--include-neighborhood-count-nodes', action='store_true',
                    help='Append compact per-cell 8-neighborhood bank '
                          '(60 regions × 3 parity variants = 180 K=1 units).')
    ap.add_argument('--include-ray-count-nodes', action='store_true',
                    help='Append line-ray bank (rows + cols + diagonals + '
                          'anti-diagonals; 42 rays × 3 parity variants = '
                          '126 K=1 units).  Captures directional structure.')
    # ----- Order-aware hidden-unit banks (require --use-move-grid) -----
    ap.add_argument('--include-turn-bucket-nodes', action='store_true',
                    help='cell c played in turn window at parity P. '
                          '60 × #buckets × 3 parities units.')
    ap.add_argument('--turn-bucket-size', type=int, default=10)
    ap.add_argument('--include-recency-nodes', action='store_true',
                    help='cell c played within last K turns of T. '
                          '60 × #Ks units.')
    ap.add_argument('--recency-Ks', default='1,2,5,10,20',
                    help='Comma-separated list of K values for recency.')
    ap.add_argument('--include-ordinal-nodes', action='store_true',
                    help='cell c was the K-th move (== raw movegrid). '
                          '3600 units.')
    ap.add_argument('--include-pairwise-order-nodes', action='store_true',
                    help='cell A played before cell B, restricted to '
                          'spatially-close pairs.')
    ap.add_argument('--pairwise-max-chebyshev', type=int, default=2,
                    help='Chebyshev-distance cutoff for pairwise-order + '
                          'streak pairs.')
    ap.add_argument('--include-streak-nodes', action='store_true',
                    help='cell A and B played within N turns, restricted '
                          'to spatially-close pairs.')
    ap.add_argument('--streak-N-gap', type=int, default=3)
    ap.add_argument('--random-count-nodes', type=int, default=0,
                    help='Number of random-subset count nodes to add.')
    ap.add_argument('--random-count-seed', type=int, default=42)
    ap.add_argument('--probe-l1', type=float, default=0.0,
                    help='L1 regularization strength for probe.  If > 0, '
                          'penalizes |probe.weight| to select a sparse '
                          'subset of hidden units.')
    ap.add_argument('--boost-count-rounds', type=int, default=0,
                    help='If > 0, iteratively add residual-correlated count '
                          'candidates to hidden layer.  Requires a candidate '
                          'pool (--include-count-nodes / --random-count-nodes).')
    ap.add_argument('--boost-count-per-round', type=int, default=200,
                    help='Number of top-scoring candidates added per '
                          'boosting round.')
    ap.add_argument('--boost-candidate-pool', type=int, default=10000,
                    help='If boosting, size of the random candidate pool from '
                          'which residual-correlated features are selected. '
                          'Structured pool is always included.')
    ap.add_argument('--probe-seeds', type=int, default=1,
                    help='Train this many probes with different seeds and '
                          'ensemble their softmax outputs.  Averages out '
                          'noise-feature-fitting from any single init.')
    ap.add_argument('--state-readout', default='linear',
                    choices=['linear', 'probor'],
                    help='linear (default): AdamW linear probe with softmax '
                          '+ cross-entropy.  probor: StateNoisyOrHead — '
                          'per-(unit, cell, class) non-negative rates, '
                          'combined by prob-OR, normalized across 3 '
                          'classes, trained with NLL.  probor forces each '
                          'unit to VOTE FOR classes only, no cancellation.')
    ap.add_argument('--probor-lr', type=float, default=0.05)
    ap.add_argument('--probe-solver', default='adamw',
                    choices=['adamw', 'sklearn'],
                    help='adamw = current AdamW probe.  sklearn = 64 '
                          'per-cell sklearn LogisticRegression (LBFGS) '
                          'fits.  sklearn is convex and provably converges '
                          'to global optimum; use to check whether adamw '
                          'is undertraining.')
    ap.add_argument('--sklearn-C', type=float, default=1.0,
                    help='sklearn LogisticRegression C (inverse L2).')
    ap.add_argument('--sklearn-n-jobs', type=int, default=2,
                    help='Parallelize the 64 per-cell LR fits.  Each worker '
                          'copies H_tr, so keep this small at high H.')
    ap.add_argument('--sklearn-subsample-train', type=int, default=200000,
                    help='If set, subsample this many training rows before '
                          'fitting sklearn LR to bound memory/time.')
    ap.add_argument('--sklearn-solver', default='lbfgs',
                    choices=['lbfgs', 'saga', 'liblinear'],
                    help='sklearn LR solver.  saga is much faster on large '
                          'multi-class problems.')
    ap.add_argument('--sklearn-max-iter', type=int, default=200)
    ap.add_argument('--count-features-as-input', action='store_true',
                    help='Add count-node activations to the raw input X '
                          '(before tree fitting) instead of appending to '
                          'the hidden layer.  Trees can then split on them '
                          'and choose which are useful.  Uses the structured '
                          'pool + --random-count-nodes count.')
    ap.add_argument('--top-k-per-cell', type=int, default=None,
                    help='Keep only the top-K most frequently trained-on '
                          'paths per cell.  Total H becomes at most 64 * K.')
    ap.add_argument('--tree-target', default='state',
                    choices=['state', 'legal', 'patterns'],
                    help='What trees predict.  state (default): 64 per-cell '
                          'trees, 3-class {empty, mine, opp}.  legal: 64 '
                          'per-cell trees, 2-class {illegal, legal}.  '
                          'patterns: 960 per-pattern trees, each predicting '
                          'whether that flanking pattern is currently active '
                          'in the true board state.  patterns requires '
                          '--include-flanking-patterns.')
    ap.add_argument('--pattern-n-trees', type=int, default=1,
                    help='For --tree-target patterns: how many bagged trees '
                          'to fit per pattern.  1 = single tree per pattern '
                          '(deterministic).  >1 = bagged ensemble with '
                          'bootstrap sampling.')
    ap.add_argument('--pattern-tree-mode', default='per_pattern',
                    choices=['per_pattern', 'multioutput', 'grouped'],
                    help='How pattern trees partition the 960 targets. '
                          'per_pattern (default): one independent tree per '
                          'pattern — maximally sharp per pattern, but leaves '
                          'are pattern-specific and the readout sees a '
                          'fragmented, redundant basis.  multioutput: ONE '
                          'multi-output tree jointly fit on all 960 (sklearn '
                          'native) — splits are shared, so leaves are '
                          'board-state regions informative about many patterns '
                          'at once (fewer, decorrelated hidden units), at the '
                          'cost of per-pattern sharpness (rare patterns get '
                          'averaged out).  grouped: one multi-output tree per '
                          'target cell (~64 trees over the ~15 patterns each) — '
                          'related patterns share a tree, unrelated ones do '
                          'not over-compromise.  multioutput/grouped pair with '
                          'the dense (bce/linpo) readout, not strupo.')
    ap.add_argument('--pattern-use-random-forest', action='store_true',
                    help='Fit one sklearn RandomForestClassifier per '
                          'pattern (with n_estimators = --pattern-n-trees) '
                          'instead of hand-bagged DecisionTreeClassifiers. '
                          'RF adds per-split feature subsampling for more '
                          'diverse trees within each pattern ensemble.')
    ap.add_argument('--pattern-algorithm', default='dt',
                    choices=['dt', 'rf', 'et', 'gbm', 'skope'],
                    help='Per-pattern algorithm.  dt: DecisionTreeClassifier. '
                          'rf: RandomForestClassifier.  et: ExtraTreesClassifier. '
                          'gbm: sklearn GradientBoostingClassifier. '
                          'skope: SkopeRules (rare-positive rule mining).')
    ap.add_argument('--pattern-gb-learning-rate', type=float, default=0.1,
                    help='GBM only: learning rate.')
    ap.add_argument('--pattern-skope-precision-min', type=float, default=0.5,
                    help='SkopeRules only: minimum precision for rules '
                          'to be kept.  Rare-positive patterns may need '
                          'lower values (e.g., 0.2-0.3).')
    ap.add_argument('--pattern-skope-recall-min', type=float, default=0.01,
                    help='SkopeRules only: minimum recall for rules.')
    ap.add_argument('--pattern-class-weight', default='balanced',
                    choices=['balanced', 'none'],
                    help='For --tree-target patterns: class weighting.  '
                          'balanced (default): weight inversely to class '
                          'frequency — helps trees notice rare positive '
                          'examples but risks overpredicting under prob-OR '
                          'combination.  none: standard weighting; each '
                          'example equally weighted, so trees may collapse '
                          'to always-negative on very rare patterns but '
                          'output calibrated probabilities.')
    ap.add_argument('--task', default='state',
                    choices=['state', 'legal', 'both'],
                    help='state (default): train state-decoding probe only. '
                          'legal: only legal-move probes.  both: state + '
                          'all three legal-move predictors on the same H.')
    ap.add_argument('--legal-modes', default='bce,probor,derived',
                    help='For task in {legal,both}: comma-separated subset '
                          'of {bce, probor, derived, state_probor} to run.  '
                          'state_probor: prob-OR over 960 flanking patterns '
                          'applied to SOFT state predictions '
                          '(requires --include-flanking-patterns AND a '
                          'trained state probe).')
    ap.add_argument('--legal-probe-epochs', type=int, default=50)
    ap.add_argument('--skip-tree-fit', action='store_true',
                    help='Skip per-cell tree fit + path extraction entirely. '
                          'Hidden layer starts empty (N, 0); only count / '
                          'order banks contribute.  Ablation: how much do '
                          'the tree paths add on top of order features?')
    ap.add_argument('--pickle-dir', default=None,
                    help='Directory of pre-generated .pickle files with '
                          'synthetic games (used by the MLP baseline).  '
                          'Load ~100K games per pickle from disk instead '
                          'of playing games from scratch (which is ~44h '
                          'for 6M games).  With this, 6M games loads in '
                          '~15 min.  Recommended: '
                          'data/othello_synthetic/')
    ap.add_argument('--load-trees-from', default=None,
                    help='Path to a previously-saved checkpoint (.pt).  '
                          'Reuses W, b, and tree_path metadata from the '
                          'checkpoint instead of re-fitting trees.  Saves '
                          '~40 min on 100k-game runs.  Requires the input '
                          'featurization to match the checkpoints (same '
                          'played+even+mover_parity; recent bits and '
                          'flanking patterns are re-computed downstream '
                          'so those can differ).')
    ap.add_argument('--skip-state-probe', action='store_true',
                    help='Skip the state probe training + evaluation '
                          'entirely.  Useful when only running legal-move '
                          'probes and the state probe would be wasted '
                          'compute.')
    ap.add_argument('--skip-inline-legal-probe', action='store_true',
                    help='Skip the in-job legal-move probe(s).  Trees are '
                          'still fit and saved.  Use this when the real '
                          'legal probe will be trained afterward via the '
                          'streaming pipeline on more games.')
    ap.add_argument('--tree-fit-only', action='store_true',
                    help='Fit trees + extract paths, save the tree '
                          'checkpoint, then EXIT.  Skips H_tr computation '
                          'entirely -- necessary for large NUM_TRAIN where '
                          'H_tr would blow memory (100K games x 48K units '
                          '= 200+ GB).  The saved checkpoint is streaming-'
                          'compatible.')
    ap.add_argument('--cache-tr', default=None,
                    help='Path to .npz cache for the sampled TRAIN set.')
    ap.add_argument('--cache-te', default=None,
                    help='Path to .npz cache for the sampled TEST set.')
    ap.add_argument('--tree-cache', default=None,
                    help='Restart aid: path to save the fitted trees to '
                         'IMMEDIATELY after fitting, and auto-load from on a '
                         'rerun (skips the ~40-min tree fit).  Combined with '
                         '--cache-tr/--cache-te (sampling cache), a '
                         'timed-out job that is resubmitted skips both the '
                         'sampling and the tree fit and goes straight to the '
                         'probe stage.  Independent of --load-trees-from '
                         '(which takes precedence if given).')
    args = ap.parse_args()

    # Restart aid: if a tree cache exists and no explicit --load-trees-from
    # was given, load the cached trees instead of re-fitting.  Skipped under
    # --hidden-from-leaves: that mode needs the live sklearn trees (the cache
    # only holds the W-based bank), so it must always re-fit fresh.
    if not args.load_trees_from and args.tree_cache and \
            os.path.exists(args.tree_cache) and not args.hidden_from_leaves:
        print(f'--tree-cache {args.tree_cache} exists: auto-loading fitted '
               f'trees (skipping re-fit)', flush=True)
        args.load_trees_from = args.tree_cache

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('warning: CUDA requested but not available; falling back to CPU')
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f'device: {device}')
    print(f'ply range: [{args.ply_min}, {args.ply_max})')

    recent_Ks = tuple(int(k) for k in args.input_recent_Ks.split(',')
                        if k.strip()) or None
    hidden_recent_Ks = tuple(int(k) for k in
                                args.recent_Ks_as_hidden.split(',')
                                if k.strip()) or None
    if recent_Ks and hidden_recent_Ks:
        raise ValueError('--input-recent-Ks and --recent-Ks-as-hidden '
                          'are mutually exclusive')
    time_ordinal = args.time_ordinal if args.time_ordinal != 'none' else None
    if time_ordinal:
        print(f'time-ordinal block: {time_ordinal} → 60 numeric cols appended '
               f'to tree input (trees learn adaptive turn ranges)')
    if args.hidden_from_leaves:
        if args.load_trees_from:
            raise ValueError('--hidden-from-leaves needs the sklearn trees in '
                              'memory; incompatible with --load-trees-from '
                              '(and the fit/readout STAGE split, which reloads '
                              'the W-based bank).  Run as one full job.')
        if args.add_stability:
            raise ValueError('--hidden-from-leaves is incompatible with '
                              '--add-stability (stability units have no tree).')
        print('hidden layer: TRUE leaf one-hot (tree.apply) — honors numeric '
               'thresholds; ±1 linearization bypassed for H (W still saved '
               'but NOT valid for streaming/reeval loaders).')
    if time_ordinal and not args.hidden_from_leaves:
        print('WARNING: --time-ordinal without --hidden-from-leaves — numeric '
               'splits will be CORRUPTED by the ±1 linearization.  Add '
               '--hidden-from-leaves.')
    # Both flags cause recent bits to be materialized during sampling; the
    # difference is only whether they're passed to the tree fit or kept
    # aside for direct concat to the hidden layer.
    sampling_Ks = recent_Ks or hidden_recent_Ks
    if sampling_Ks:
        role = 'tree input' if recent_Ks else 'hidden layer'
        print(f'sampling recent bits for K in {sampling_Ks} '
               f'({60 * len(sampling_Ks)} bits) → {role}')

    collect_legal = args.task != 'state'
    if collect_legal:
        print(f'task={args.task}: collecting per-position legal-move masks')

    print(f'sampling {args.num_train_games} train + '
           f'{args.num_test_games} test games...')
    t0 = time.time()
    tr = load_or_sample(
        args.cache_tr, sample_midgame_positions,
        args.num_train_games, ply_min=args.ply_min,
        ply_max=args.ply_max, seed=args.seed,
        when_bucket_size=args.when_bucket_size,
        use_move_grid=args.use_move_grid,
        recent_Ks=sampling_Ks,
        collect_legal_moves=collect_legal,
        canonicalize_mover=args.canonicalize_mover,
        time_ordinal=time_ordinal,
        pickle_dir=args.pickle_dir)
    te = load_or_sample(
        args.cache_te, sample_midgame_positions,
        args.num_test_games, ply_min=args.ply_min,
        ply_max=args.ply_max, seed=args.seed + 1_000_000,
        when_bucket_size=args.when_bucket_size,
        use_move_grid=args.use_move_grid,
        recent_Ks=sampling_Ks,
        collect_legal_moves=collect_legal,
        canonicalize_mover=args.canonicalize_mover,
        time_ordinal=time_ordinal,
        pickle_dir=args.pickle_dir)
    if collect_legal:
        Xnp_tr, Snp_tr, Tnp_tr, Lnp_tr = tr
        Xnp_te, Snp_te, Tnp_te, Lnp_te = te
    else:
        Xnp_tr, Snp_tr, Tnp_tr = tr
        Xnp_te, Snp_te, Tnp_te = te
        Lnp_tr = Lnp_te = None

    # If a cache from a wider ply range was loaded, narrow it to the current
    # args range.  Lets a single (10, 50) cache serve any [a, b) subwindow.
    def _narrow(X, S, T, L=None):
        mask = (T >= args.ply_min) & (T < args.ply_max)
        if mask.all():
            return X, S, T, L
        n_before = X.shape[0]
        X = X[mask]; S = S[mask]; T = T[mask]
        if L is not None:
            L = L[mask]
        print(f'  filtered cache: {n_before} → {X.shape[0]} positions '
               f'in ply [{args.ply_min}, {args.ply_max})')
        return X, S, T, L

    Xnp_tr, Snp_tr, Tnp_tr, Lnp_tr = _narrow(Xnp_tr, Snp_tr, Tnp_tr, Lnp_tr)
    Xnp_te, Snp_te, Tnp_te, Lnp_te = _narrow(Xnp_te, Snp_te, Tnp_te, Lnp_te)
    print(f'  train={Xnp_tr.shape[0]}  test={Xnp_te.shape[0]}  '
           f'({time.time() - t0:.1f}s)')

    # ---- Set aside the ordinal turn-of-play block ----
    # playedeven_features appends it as the trailing 60 cols.  Carry it aside
    # so the movegrid / recent-as-hidden slicing below (which hard-slices Xnp
    # back to played_even) doesn't drop it; re-append just before the tree fit
    # so the trees DO see it and can split on turn-of-play thresholds.
    ordinal_tr = ordinal_te = None
    if time_ordinal:
        ordinal_tr = np.ascontiguousarray(Xnp_tr[:, -60:])
        ordinal_te = np.ascontiguousarray(Xnp_te[:, -60:])
        Xnp_tr = np.ascontiguousarray(Xnp_tr[:, :-60])
        Xnp_te = np.ascontiguousarray(Xnp_te[:, :-60])
        if args.time_ordinal_split_color:
            # A player moves on alternate turns, so a single ordinal band mixes
            # colors.  Split into mover/opp channels using placed_as_mover
            # (Xnp cols 60:120, still present after the ordinal slice above):
            # each channel keeps the play-time only for its color, -1 else.
            if not args.canonicalize_mover:
                raise ValueError('--time-ordinal-split-color requires '
                                  '--canonicalize-mover (needs placed_as_mover '
                                  'in Xnp cols 60:120)')
            def _split_ordinal(ordinal, X):
                pam = X[:, 60:120] > 0                     # placed_as_mover
                mover = np.where(pam, ordinal, -1.0).astype(np.float32)
                opp = np.where(pam, -1.0, ordinal).astype(np.float32)
                return np.ascontiguousarray(
                    np.concatenate([mover, opp], axis=1))
            ordinal_tr = _split_ordinal(ordinal_tr, Xnp_tr)
            ordinal_te = _split_ordinal(ordinal_te, Xnp_te)
            print(f'  --time-ordinal-split-color: ordinal split into mover/opp '
                   f'→ {ordinal_tr.shape[1]} cols')
        print(f'  set aside ordinal block {ordinal_tr.shape} '
               f'(Xnp_tr → {Xnp_tr.shape})')

    # ---- Split off movegrid for order-aware hidden banks ----
    # Order-feature builders need the (N, 60, 60) movegrid; trees still fit
    # on the 121-d played_even part so the tree-path baseline is preserved.
    order_flags_on = (args.include_turn_bucket_nodes
                      or args.include_recency_nodes
                      or args.include_ordinal_nodes
                      or args.include_pairwise_order_nodes
                      or args.include_streak_nodes)
    movegrid_tr = movegrid_te = None
    if order_flags_on:
        if not args.use_move_grid:
            raise ValueError('--include-*-order-nodes requires '
                              '--use-move-grid to be set at sampling time.')
        if args.when_bucket_size:
            raise ValueError('--include-*-order-nodes is incompatible with '
                              '--when-bucket-size (both consume the movegrid '
                              'slot).')
        print('extracting movegrid + slicing Xnp to played_even for tree fit')
        movegrid_tr = movegrid_from_flat(Xnp_tr).astype(np.uint8)
        movegrid_te = movegrid_from_flat(Xnp_te).astype(np.uint8)
        Xnp_tr = np.ascontiguousarray(Xnp_tr[:, :121])
        Xnp_te = np.ascontiguousarray(Xnp_te[:, :121])
        print(f'  movegrid_tr {movegrid_tr.shape}  '
               f'Xnp_tr sliced → {Xnp_tr.shape}')

    # ---- Split off recent bits for hidden-layer concat ----
    # If --recent-Ks-as-hidden was used, recent bits were materialized in
    # Xnp cols [121, 121 + 60*#Ks) at sample time.  Slice them out so trees
    # fit on played_even only; keep the recent slab for later concat.
    recent_hidden_tr = recent_hidden_te = None
    if hidden_recent_Ks:
        n_recent = 60 * len(hidden_recent_Ks)
        print(f'slicing recent bits (K={hidden_recent_Ks}, {n_recent} cols) '
               f'from Xnp for direct hidden-layer concat')
        recent_hidden_tr = Xnp_tr[:, 121:121 + n_recent].astype(np.uint8)
        recent_hidden_te = Xnp_te[:, 121:121 + n_recent].astype(np.uint8)
        if args.recent_split_color:
            if not args.canonicalize_mover:
                raise ValueError('--recent-split-color requires '
                                  '--canonicalize-mover (needs placed_as_mover '
                                  'in Xnp cols 60:120)')
            # recent_mover = recent AND placed_as_mover;  recent_opp = recent
            # AND NOT placed_as_mover.  placed_as_mover is Xnp[:, 60:120].
            nK = len(hidden_recent_Ks)
            pam_tr = np.tile(Xnp_tr[:, 60:120].astype(np.uint8), nK)
            pam_te = np.tile(Xnp_te[:, 60:120].astype(np.uint8), nK)
            recent_hidden_tr = np.concatenate(
                [recent_hidden_tr & pam_tr, recent_hidden_tr & (1 - pam_tr)],
                axis=1)
            recent_hidden_te = np.concatenate(
                [recent_hidden_te & pam_te, recent_hidden_te & (1 - pam_te)],
                axis=1)
            print(f'  --recent-split-color: recency split into mover/opp → '
                   f'{recent_hidden_tr.shape[1]} bits')
        Xnp_tr = np.ascontiguousarray(Xnp_tr[:, :121])
        Xnp_te = np.ascontiguousarray(Xnp_te[:, :121])
        print(f'  recent_hidden_tr {recent_hidden_tr.shape}  '
               f'Xnp_tr sliced → {Xnp_tr.shape}')

    # --- Optionally append count features to raw input (Option B) ---
    if args.count_features_as_input:
        cf_nodes = build_structured_count_nodes()
        if args.random_count_nodes > 0:
            rn = build_random_count_nodes(args.random_count_nodes,
                                            seed=args.random_count_seed)
            cf_nodes.extend(rn)
        print(f'\ncomputing {len(cf_nodes)} count features to append to '
               f'raw input...')
        t0 = time.time()
        played_tr = Xnp_tr[:, :60]; even_tr = Xnp_tr[:, 60:120]
        played_te = Xnp_te[:, :60]; even_te = Xnp_te[:, 60:120]
        CF_tr = compute_count_activations(cf_nodes, played_tr, even_tr)
        CF_te = compute_count_activations(cf_nodes, played_te, even_te)
        print(f'  CF_tr {CF_tr.shape} '
               f'({CF_tr.nbytes / 1e9:.2f} GB)  '
               f'({time.time() - t0:.1f}s)')
        Xnp_tr = np.concatenate(
            [Xnp_tr, CF_tr.astype(np.float32)], axis=1)
        Xnp_te = np.concatenate(
            [Xnp_te, CF_te.astype(np.float32)], axis=1)
        print(f'  input dim now: {Xnp_tr.shape[1]}')
        del CF_tr, CF_te

    # ---- Re-append the ordinal turn-of-play block to the tree input ----
    # After all played_even hard-slicing, restore the trailing ordinal cols so
    # the trees fit on [played_even (+ input-recency) + ordinal].
    if time_ordinal:
        Xnp_tr = np.ascontiguousarray(
            np.concatenate([Xnp_tr, ordinal_tr.astype(np.float32)], axis=1))
        Xnp_te = np.ascontiguousarray(
            np.concatenate([Xnp_te, ordinal_te.astype(np.float32)], axis=1))
        print(f'  re-appended ordinal block → tree input dim {Xnp_tr.shape[1]}')

    S_tr = torch.from_numpy(Snp_tr)
    S_te = torch.from_numpy(Snp_te)
    T_te = torch.from_numpy(Tnp_te)
    use_relu = args.hidden_activation == 'relu'
    if args.hidden_from_leaves and use_relu:
        # Leaf one-hot is inherently 0/1 — ReLU float32 wastes 4x memory
        # (e.g. ~48k leaves x 800k positions = 154GB f32 vs 38GB bool) and
        # OOMs the node.  Force step/bool for leaf mode.
        print('  --hidden-from-leaves: forcing step/bool activation '
               '(leaf membership is binary; ReLU float32 would OOM)')
        use_relu = False
    act_dtype = torch.float32 if use_relu else torch.bool

    if args.load_trees_from:
        print(f'\n--load-trees-from {args.load_trees_from}: '
               f'reusing pre-fit trees')
        ck = torch.load(args.load_trees_from, map_location='cpu')
        W_saved = ck['W']
        b_saved = ck['b']
        meta_saved = ck['path_info']
        # Accept both tree_path (state/legal target) and pattern_path
        # (patterns target) entries — depending on what target the
        # checkpoint was trained with.
        tree_kinds = ('tree_path', 'pattern_path', 'pattern_multi')
        tree_meta = [m for m in meta_saved
                      if m.get('kind') in tree_kinds]
        n_saved = len(meta_saved)
        n_tree = len(tree_meta)
        kinds_found = {m.get('kind') for m in tree_meta}
        print(f'  loaded {n_saved} hidden units total; '
               f'{n_tree} are {kinds_found} entries (rest re-computed)')
        # Filter W, b to just the tree rows so the mlp we build
        # only produces those.
        tree_row_idx = [i for i, m in enumerate(meta_saved)
                          if m.get('kind') in tree_kinds]
        if isinstance(W_saved, torch.Tensor):
            W_saved = W_saved.numpy()
        if isinstance(b_saved, torch.Tensor):
            b_saved = b_saved.numpy()
        W = W_saved[tree_row_idx]
        B = b_saved[tree_row_idx]
        all_meta = tree_meta
        n_tree_units = n_tree
        # Verify input_dim compatibility.
        if W.shape[1] > Xnp_tr.shape[1]:
            raise ValueError(
                f'checkpoint tree weights expect input_dim '
                f'{W.shape[1]} but current Xnp has only {Xnp_tr.shape[1]} '
                f'columns; input featurization must match')
        # If Xnp has MORE columns than expected (e.g., extra recent bits
        # baked in), pad W with zeros so it ignores those columns.
        if W.shape[1] < Xnp_tr.shape[1]:
            pad = np.zeros((W.shape[0], Xnp_tr.shape[1] - W.shape[1]),
                              dtype=W.dtype)
            W = np.concatenate([W, pad], axis=1)
        mlp = OpeningTreeMLP(W, B, all_meta, device)
        per_cell_leaf_counts = np.array(
            ck.get('per_cell_leaf_counts',
                    [0] * BOARD_CELLS), dtype=int)
        tree_correct_per_cell = np.array(
            ck.get('per_cell_tree_acc',
                    [0.0] * BOARD_CELLS), dtype=float)
        # If we're targeting patterns, we need patterns_list for the
        # StruPO / patterns_probor readouts.  sklearn tree objects are
        # NOT loaded (only their leaves), so patterns_probor mode (which
        # calls tree.predict_proba) is unavailable — StruPO works.
        pattern_trees = None
        patterns_list = None
        if 'pattern_path' in kinds_found:
            if not args.include_flanking_patterns:
                raise ValueError('--load-trees-from with pattern-path '
                                  'entries requires --include-flanking-'
                                  'patterns to reload the pattern list.')
            patterns_list = load_patterns(args.include_flanking_patterns)
        X_tr = torch.from_numpy(Xnp_tr).to(device)
        X_te = torch.from_numpy(Xnp_te).to(device)
        if use_relu:
            print('computing hidden activations (ReLU) from loaded trees...')
        else:
            print('computing hidden activations (step) from loaded trees...')
        t0 = time.time()
        H_tr = mlp(X_tr, out_device='cpu', out_dtype=act_dtype,
                     use_relu=use_relu)
        H_te = mlp(X_te, out_device='cpu', out_dtype=act_dtype,
                     use_relu=use_relu)
        print(f'  H_tr {tuple(H_tr.shape)}  H_te {tuple(H_te.shape)}  '
               f'({time.time() - t0:.1f}s)')
        del X_tr, X_te
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    elif args.skip_tree_fit:
        print('\n--skip-tree-fit: no tree pipeline; H starts empty (N, 0)')
        all_meta = []
        H_tr = torch.zeros(Xnp_tr.shape[0], 0, dtype=act_dtype)
        H_te = torch.zeros(Xnp_te.shape[0], 0, dtype=act_dtype)
    else:
        # --- Train per-cell trees ---
        print(f'\ntraining per-cell trees '
               f'(max_depth={args.tree_max_depth}, '
               f'min_samples_leaf={args.tree_min_samples_leaf}, '
               f'n_jobs={args.tree_n_jobs})...')
        t0 = time.time()
        mf = args.tree_max_features
        if mf is not None and mf not in ('sqrt', 'log2', 'auto'):
            try:
                mf = int(mf)
            except ValueError:
                try:
                    mf = float(mf)
                except ValueError:
                    pass
        pattern_trees = None
        patterns_list = None
        multi_trees = None
        if args.tree_target == 'patterns':
            if not args.include_flanking_patterns:
                raise ValueError('--tree-target patterns requires '
                                  '--include-flanking-patterns.')
            patterns_list = load_patterns(args.include_flanking_patterns)
            print(f'  loaded {len(patterns_list)} patterns')
            print(f'  computing true pattern activations from state '
                   f'labels ...')
            pt_tr = true_pattern_activations(patterns_list, Snp_tr)
            pt_te = true_pattern_activations(patterns_list, Snp_te)
            fire_rate_tr = 100 * pt_tr.mean()
            print(f'    activation fire rate on train: '
                   f'{fire_rate_tr:.3f}%   '
                   f'per-pattern min: {100 * pt_tr.mean(0).min():.3f}%   '
                   f'max: {100 * pt_tr.mean(0).max():.3f}%')
            cw = (None if args.pattern_class_weight == 'none'
                    else 'balanced')
            print(f'  class_weight = {cw}')
            K = len(patterns_list)
            multi_trees = None
            if args.pattern_tree_mode != 'per_pattern':
                pattern_trees = None
                multi_trees, tree_correct_per_pattern = \
                    fit_multioutput_pattern_trees(
                        Xnp_tr, pt_tr, Xnp_te, pt_te, patterns_list,
                        args, mf, cw)
                print(f'  ({time.time() - t0:.1f}s)')
            else:
                print(f'  fitting {args.pattern_n_trees} tree(s) per pattern '
                       f'({len(patterns_list)} patterns × '
                       f'{args.pattern_n_trees} = '
                       f'{len(patterns_list) * args.pattern_n_trees} trees)...')
                pattern_trees = train_pattern_trees(
                    Xnp_tr, pt_tr,
                    n_trees_per_pattern=args.pattern_n_trees,
                    max_depth=args.tree_max_depth,
                    min_samples_leaf=args.tree_min_samples_leaf,
                    max_leaf_nodes=args.max_leaf_nodes,
                    n_jobs=args.tree_n_jobs,
                    max_features=mf,
                    class_weight=cw,
                    use_random_forest=args.pattern_use_random_forest,
                    algorithm=args.pattern_algorithm,
                    gb_learning_rate=args.pattern_gb_learning_rate,
                    skope_precision_min=args.pattern_skope_precision_min,
                    skope_recall_min=args.pattern_skope_recall_min)
                print(f'  ({time.time() - t0:.1f}s)')

                # Aggregate-per-pattern tree accuracy (majority vote across
                # the pattern's trees).
                tree_correct_per_pattern = np.zeros(K)
                from opening_tree_mlp import PreExtractedPaths
                for j in range(K):
                    # Skip aggregate accuracy for PreExtractedPaths (skope) --
                    # rule-based algorithms don't have a per-tree predict().
                    if any(isinstance(t, PreExtractedPaths)
                            for t in pattern_trees[j]):
                        tree_correct_per_pattern[j] = np.nan
                        continue
                    votes = np.zeros(Xnp_te.shape[0], dtype=np.float32)
                    for tree in pattern_trees[j]:
                        votes += tree.predict(Xnp_te).astype(np.float32)
                    pred = (votes / len(pattern_trees[j]) > 0.5).astype(np.uint8)
                    tree_correct_per_pattern[j] = (pred == pt_te[:, j]).mean()
                valid = ~np.isnan(tree_correct_per_pattern)
                if valid.any():
                    print(f'  aggregate per-pattern tree test acc: '
                           f'{100 * tree_correct_per_pattern[valid].mean():.4f}% '
                           f'({int(valid.sum())}/{K} patterns)')
                else:
                    print(f'  aggregate per-pattern tree test acc: n/a '
                           f'(algorithm produces rule-based outputs)')
        elif args.tree_target == 'legal':
            if Lnp_tr is None:
                raise ValueError('--tree-target legal requires --task legal '
                                  'or --task both to collect legal masks.')
            tree_target_tr = Lnp_tr.astype(np.int64)
            tree_target_te = Lnp_te.astype(np.int64)
            print(f'  fitting trees for LEGAL-MOVE target (binary per cell)')
        else:
            tree_target_tr = Snp_tr
            tree_target_te = Snp_te

        if args.tree_target != 'patterns':
            trees = train_per_cell_trees(
                Xnp_tr, tree_target_tr,
                max_depth=args.tree_max_depth,
                min_samples_leaf=args.tree_min_samples_leaf,
                n_jobs=args.tree_n_jobs,
                max_features=mf)
            print(f'  ({time.time() - t0:.1f}s)')

            tree_correct_per_cell = np.zeros(BOARD_CELLS)
            for c in range(BOARD_CELLS):
                preds = trees[c].predict(Xnp_te)
                tree_correct_per_cell[c] = (
                    preds == tree_target_te[:, c]).mean()
            target_label = ('legal-move'
                            if args.tree_target == 'legal' else 'state')
            print(f'  aggregate per-cell tree test acc ({target_label}): '
                   f'{100*tree_correct_per_cell.mean():.4f}%')
            for cls in ('corner', 'edge', 'inner'):
                cell_ids = [c for c in range(64) if CELL_CLASS[c] == cls]
                print(f'    {cls:6s} ({len(cell_ids)} cells):  '
                       f'{100*tree_correct_per_cell[cell_ids].mean():.4f}%')

        # --- Extract paths ---
        print('\nextracting paths → hidden units...')
        all_w, all_b, all_meta = [], [], []
        # Parallel to all_meta's tree units: (sklearn_tree, leaf_node_id) per
        # hidden column, so --hidden-from-leaves can compute H as true leaf
        # one-hot (honoring numeric thresholds) instead of the ±1 W·x.
        leaf_build = []

        def _pair_and_prune(tree):
            """Pair extract_paths(tree) with its leaf node ids and apply the
            SAME top-k-by-count pruning, so meta/W match the non-leaf path."""
            paths = extract_paths(tree)
            nids = leaf_node_ids(tree)
            pairs = (list(zip(paths, nids)) if nids is not None
                     else [(p, None) for p in paths])
            tk = args.top_k_per_cell
            if tk is not None and len(pairs) > tk:
                pairs = sorted(pairs, key=lambda pn: -sum(pn[0][2]))[:tk]
            return pairs

        input_dim = Xnp_tr.shape[1]
        if args.tree_target == 'patterns' and multi_trees is not None:
            # multioutput / grouped: leaves of the joint tree(s) → hidden units.
            # Each leaf covers many patterns, so meta is per-leaf (not
            # per-pattern); pair with the dense (bce/linpo) readout.
            per_cell_leaf_counts = np.zeros(len(multi_trees), dtype=int)
            for g_idx, (tree, label, cols) in enumerate(multi_trees):
                pairs = _pair_and_prune(tree)
                per_cell_leaf_counts[g_idx] = len(pairs)
                for path_idx, ((conditions, leaf_class,
                                 leaf_counts), nid) in enumerate(pairs):
                    w, b = path_to_weight(conditions, input_dim=input_dim)
                    all_w.append(w); all_b.append(b)
                    all_meta.append({
                        'kind': 'pattern_multi',
                        'group': label,
                        'patterns': cols,
                        'path_idx': path_idx,
                        'conditions': conditions,
                        'depth': len(conditions),
                        'leaf_counts': leaf_counts,
                    })
                    leaf_build.append((tree, nid))
        elif args.tree_target == 'patterns':
            per_pattern_leaf_counts = np.zeros(len(patterns_list), dtype=int)
            for j, tree_list in enumerate(pattern_trees):
                for tree_idx, tree in enumerate(tree_list):
                    pairs = _pair_and_prune(tree)
                    per_pattern_leaf_counts[j] += len(pairs)
                    for path_idx, ((conditions, leaf_class,
                                     leaf_counts), nid) in enumerate(pairs):
                        w, b = path_to_weight(conditions,
                                                 input_dim=input_dim)
                        all_w.append(w); all_b.append(b)
                        all_meta.append({
                            'kind': 'pattern_path',
                            'pattern': j,
                            'tree_idx': tree_idx,
                            'path_idx': path_idx,
                            'target_cell': patterns_list[j]['target'],
                            'conditions': conditions,
                            'leaf_class': leaf_class,
                            'depth': len(conditions),
                            'leaf_counts': leaf_counts,
                        })
                        leaf_build.append((tree, nid))
            per_cell_leaf_counts = per_pattern_leaf_counts
        else:
            per_cell_leaf_counts = np.zeros(BOARD_CELLS, dtype=int)
            for c in range(BOARD_CELLS):
                pairs = _pair_and_prune(trees[c])
                per_cell_leaf_counts[c] = len(pairs)
                for path_idx, ((conditions, leaf_class,
                                 leaf_counts), nid) in enumerate(pairs):
                    w, b = path_to_weight(conditions, input_dim=input_dim)
                    all_w.append(w); all_b.append(b)
                    all_meta.append({
                        'kind': 'tree_path',
                        'cell': c, 'path_idx': path_idx,
                        'conditions': conditions, 'leaf_class': leaf_class,
                        'depth': len(conditions),
                        'leaf_counts': leaf_counts,
                        'cell_class': CELL_CLASS[c],
                    })
                    leaf_build.append((trees[c], nid))
        n_tree_units = len(all_meta)

        if args.add_stability:
            print('  adding stability feature bank...')
            Ws, Bs, meta_s = build_stability_features(device)
            all_w.extend(Ws)
            all_b.extend(Bs.tolist())
            all_meta.extend(meta_s)
            print(f'  stability units added: {Ws.shape[0]}')

        if not all_w:
            raise RuntimeError(
                'No paths extracted from any tree.  Common causes: '
                '(a) SkopeRules produced no qualifying rules under the '
                'given precision/recall thresholds — try lowering '
                '--pattern-skope-precision-min. '
                '(b) all trees produced empty leaves.  Aborting before '
                'np.stack([]) would crash.')
        W = np.stack(all_w); B = np.array(all_b, dtype=np.float32)
        print(f'  total hidden units: {len(all_meta)}   '
               f'(tree={n_tree_units}, stability='
               f'{len(all_meta) - n_tree_units})')
        print(f'  leaves per tree: mean={per_cell_leaf_counts.mean():.1f}  '
               f'max={per_cell_leaf_counts.max()}  '
               f'min={per_cell_leaf_counts.min()}')

        depths = np.array([m['depth'] for m in all_meta
                            if m.get('kind') in ('tree_path',
                                                     'pattern_path',
                                                     'pattern_multi')])
        if depths.size > 0:
            print(f'  tree-path depths: mean={depths.mean():.2f}  '
                   f'max={depths.max()}  min={depths.min()}')
        else:
            print(f'  tree-path depths: (no paths extracted)')

        mlp = OpeningTreeMLP(W, B, all_meta, device)

        # Restart aid: persist the freshly-fit trees before the (expensive,
        # OOM-prone) H_tr computation and probe stage, so a timeout after
        # this point never re-fits.  Same format as --tree-fit-only / the
        # streaming --load-trees-from loader.
        if args.tree_cache and not os.path.exists(args.tree_cache):
            tc_tmp = args.tree_cache + '.tmp'
            torch.save({
                'W': mlp.W.cpu(),
                'b': mlp.b.cpu(),
                'path_info': all_meta,
                'per_cell_leaf_counts': per_cell_leaf_counts,
                'per_cell_tree_acc': (tree_correct_per_cell.tolist()
                                        if args.tree_target != 'patterns'
                                        else tree_correct_per_pattern.tolist()),
                'args': vars(args),
            }, tc_tmp)
            os.replace(tc_tmp, args.tree_cache)      # atomic
            print(f'  [tree-cache] saved fitted trees to {args.tree_cache}',
                    flush=True)

        if args.tree_fit_only:
            print('\n--tree-fit-only: saving tree checkpoint and exiting '
                   'before H_tr computation (skips OOM risk on large data).')
            torch.save({
                'W': mlp.W.cpu(),
                'b': mlp.b.cpu(),
                'path_info': all_meta,
                'per_cell_leaf_counts': per_cell_leaf_counts,
                'per_cell_tree_acc': (tree_correct_per_cell.tolist()
                                        if args.tree_target != 'patterns'
                                        else tree_correct_per_pattern.tolist()),
                'args': vars(args),
            }, args.out)
            print(f'saved {args.out}')
            return

        if args.hidden_from_leaves:
            print('\ncomputing hidden activations (TRUE leaf one-hot via '
                   'tree.apply — honors numeric thresholds)...')
            t0 = time.time()
            H_tr = build_H_from_leaves(leaf_build, Xnp_tr, act_dtype)
            H_te = build_H_from_leaves(leaf_build, Xnp_te, act_dtype)
        else:
            X_tr = torch.from_numpy(Xnp_tr).to(device)
            X_te = torch.from_numpy(Xnp_te).to(device)
            if use_relu:
                print('\ncomputing hidden activations (ReLU, float32 on CPU)...')
            else:
                print('\ncomputing hidden activations (step, bool on CPU)...')
            t0 = time.time()
            H_tr = mlp(X_tr, out_device='cpu', out_dtype=act_dtype,
                         use_relu=use_relu)
            H_te = mlp(X_te, out_device='cpu', out_dtype=act_dtype,
                         use_relu=use_relu)
    print(f'  H_tr {tuple(H_tr.shape)} '
           f'({H_tr.element_size() * H_tr.nelement() / 1e9:.2f} GB)  '
           f'H_te {tuple(H_te.shape)} '
           f'({H_te.element_size() * H_te.nelement() / 1e9:.2f} GB)')
    if not args.skip_tree_fit:
        # The --load-trees-from branch (used by --tree-cache resumes) already
        # frees X_tr/X_te internally, so guard against a double-delete here.
        try:
            del X_tr, X_te
        except UnboundLocalError:
            pass
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ---- Optionally append recent-bits hidden bank ----
    # Direct probe access to "cell c played in last K turns" bits, bypassing
    # the tree-path top-K pruning that discards fine-grained recency leaves.
    if hidden_recent_Ks:
        print(f'\nappending {recent_hidden_tr.shape[1]} recent-bit hidden '
               f'units (K={hidden_recent_Ks})...')
        rb_tr = torch.from_numpy(recent_hidden_tr).to(act_dtype)
        rb_te = torch.from_numpy(recent_hidden_te).to(act_dtype)
        H_tr = torch.cat([H_tr, rb_tr], dim=1)
        H_te = torch.cat([H_te, rb_te], dim=1)
        roles = ['mover', 'opp'] if args.recent_split_color else ['']
        for role in roles:
            for k_idx, K in enumerate(hidden_recent_Ks):
                for cell60 in range(60):
                    cell64 = C60_TO_C64[cell60]
                    alg = 'ABCDEFGH'[cell64 % 8] + str(cell64 // 8 + 1)
                    nm = (f'recent_{role}{K}[{alg}]' if role
                          else f'recent{K}[{alg}]')
                    all_meta.append({
                        'kind': 'recent_bit',
                        'name': nm,
                        'K': K, 'cell60': cell60, 'role': role or 'any',
                    })
        print(f'  combined H_tr {tuple(H_tr.shape)}  '
               f'H_te {tuple(H_te.shape)}')
        del rb_tr, rb_te, recent_hidden_tr, recent_hidden_te

    # ---- Optionally append flanking-pattern hidden bank ----
    # 960 hand-crafted Othello legality rules as {0/1} moveset+parity
    # conjunctions.  Each unit fires iff the pattern's conjunction is
    # satisfied under placement=current-color approximation.
    if args.include_flanking_patterns:
        print(f'\nloading flanking patterns from '
               f'{args.include_flanking_patterns}...')
        patterns = load_patterns(args.include_flanking_patterns)
        print(f'  {len(patterns)} patterns loaded')
        # played/even/mp come straight from the first 121 cols of Xnp.
        played_tr = Xnp_tr[:, :60].astype(np.uint8)
        even_tr = Xnp_tr[:, 60:120].astype(np.uint8)
        mp_tr = Xnp_tr[:, 120].astype(np.uint8)
        played_te = Xnp_te[:, :60].astype(np.uint8)
        even_te = Xnp_te[:, 60:120].astype(np.uint8)
        mp_te = Xnp_te[:, 120].astype(np.uint8)
        t0 = time.time()
        FP_tr = compute_pattern_activations(patterns, played_tr, even_tr,
                                                 mp_tr)
        FP_te = compute_pattern_activations(patterns, played_te, even_te,
                                                 mp_te)
        print(f'  activations: FP_tr {FP_tr.shape} '
               f'({FP_tr.nbytes / 1e9:.2f} GB)  '
               f'fire={100*FP_tr.mean():.3f}%  '
               f'({time.time() - t0:.1f}s)')
        H_tr = torch.cat([H_tr, torch.from_numpy(FP_tr).to(act_dtype)],
                             dim=1)
        H_te = torch.cat([H_te, torch.from_numpy(FP_te).to(act_dtype)],
                             dim=1)
        for j, pat in enumerate(patterns):
            all_meta.append({
                'kind': 'flanking_pattern',
                'name': (f'flank_t{pat["target"]}_o{pat["opponents"]}'
                          f'_x{pat["terminal"]}_d{pat["direction"]}'),
                'target': pat['target'],
                'opponents': pat['opponents'],
                'terminal': pat['terminal'],
                'direction': pat['direction'],
                'length': pat['length'],
            })
        print(f'  combined H_tr {tuple(H_tr.shape)}  '
               f'H_te {tuple(H_te.shape)}')
        del FP_tr, FP_te

    # ---- Optionally append count-node bank ----
    count_nodes_used = []
    if (args.include_count_nodes or args.random_count_nodes > 0
            or args.include_tree_derived_count_nodes
            or args.include_neighborhood_count_nodes
            or args.include_ray_count_nodes):
        print('\nbuilding count-node bank...')
        if args.include_count_nodes:
            count_nodes_used.extend(build_structured_count_nodes())
            print(f'  structured count nodes: '
                   f'{len(count_nodes_used)}')
        if args.include_neighborhood_count_nodes:
            nb = build_neighborhood_count_nodes()
            count_nodes_used.extend(nb)
            print(f'  neighborhood count nodes: {len(nb)}')
        if args.include_ray_count_nodes:
            ry = build_ray_count_nodes()
            count_nodes_used.extend(ry)
            print(f'  ray count nodes:         {len(ry)}')
        if args.include_tree_derived_count_nodes:
            tree_nodes = build_tree_derived_count_nodes(all_meta)
            count_nodes_used.extend(tree_nodes)
            print(f'  tree-derived count nodes: {len(tree_nodes)}')
        if args.random_count_nodes > 0:
            rn = build_random_count_nodes(args.random_count_nodes,
                                            seed=args.random_count_seed)
            count_nodes_used.extend(rn)
            print(f'  random count nodes:     {len(rn)}')

        # Extract played + even from the raw played_even features (positions
        # 0..60 and 60..120).
        played_tr = Xnp_tr[:, :60]
        even_tr = Xnp_tr[:, 60:120]
        played_te = Xnp_te[:, :60]
        even_te = Xnp_te[:, 60:120]

        t0 = time.time()
        CN_tr = compute_count_activations(count_nodes_used, played_tr,
                                            even_tr, use_relu=use_relu)
        CN_te = compute_count_activations(count_nodes_used, played_te,
                                            even_te, use_relu=use_relu)
        print(f'  count activations: '
               f'CN_tr {CN_tr.shape} '
               f'({CN_tr.nbytes / 1e9:.2f} GB)  '
               f'CN_te {CN_te.shape}  '
               f'({time.time() - t0:.1f}s)')

        CN_tr_t = torch.from_numpy(CN_tr)
        CN_te_t = torch.from_numpy(CN_te)
        H_tr = torch.cat([H_tr, CN_tr_t], dim=1)
        H_te = torch.cat([H_te, CN_te_t], dim=1)
        print(f'  combined H_tr {tuple(H_tr.shape)}  '
               f'H_te {tuple(H_te.shape)}')
        del CN_tr_t, CN_te_t, CN_tr, CN_te
        # Also record count-node metadata in path_info.
        for n in count_nodes_used:
            all_meta.append({'kind': 'count_node', 'name': n[0],
                              'parity': n[2], 'threshold': n[3]})

    # ---- Optionally append order-aware banks (movegrid-based) ----
    if order_flags_on:
        print('\nbuilding order-aware banks...')
        recency_Ks = tuple(int(k) for k in args.recency_Ks.split(','))
        order_builders = []
        if args.include_turn_bucket_nodes:
            order_builders.append(('turn_bucket',
                lambda mg, cur: build_turn_bucket(
                    mg, bucket_size=args.turn_bucket_size, use_relu=use_relu)))
        if args.include_recency_nodes:
            order_builders.append(('recency',
                lambda mg, cur: build_recency(
                    mg, current_turns=cur, Ks=recency_Ks, use_relu=use_relu)))
        if args.include_ordinal_nodes:
            order_builders.append(('ordinal',
                lambda mg, cur: build_ordinal(mg, use_relu=use_relu)))
        if args.include_pairwise_order_nodes:
            order_builders.append(('pairwise_order',
                lambda mg, cur: build_pairwise_order(
                    mg, max_chebyshev=args.pairwise_max_chebyshev,
                    use_relu=use_relu)))
        if args.include_streak_nodes:
            order_builders.append(('streak',
                lambda mg, cur: build_streak(
                    mg, max_chebyshev=args.pairwise_max_chebyshev,
                    N_gap=args.streak_N_gap, use_relu=use_relu)))
        for label, fn in order_builders:
            t0 = time.time()
            Otr, meta_o = fn(movegrid_tr, Tnp_tr)
            Ote, _ = fn(movegrid_te, Tnp_te)
            print(f'  {label:18s} {Otr.shape[1]:5d} units  '
                   f'{Otr.nbytes / 1e9:.2f} GB tr  '
                   f'fire={(Otr > 0).mean() * 100:.2f}%  '
                   f'({time.time() - t0:.1f}s)')
            H_tr = torch.cat([H_tr, torch.from_numpy(Otr)], dim=1)
            H_te = torch.cat([H_te, torch.from_numpy(Ote)], dim=1)
            all_meta.extend(meta_o)
            del Otr, Ote
        print(f'  combined H_tr {tuple(H_tr.shape)}  '
               f'H_te {tuple(H_te.shape)}')

    # Diagnostic: "firing rate" (fraction >0) works for both bool and float.
    fire_counts = torch.zeros(H_tr.shape[1], dtype=torch.int64)
    for i in range(0, H_tr.shape[0], 512):
        chunk = H_tr[i:i + 512]
        if chunk.dtype == torch.bool:
            fire_counts += chunk.to(torch.int64).sum(dim=0)
        else:
            fire_counts += (chunk > 0).to(torch.int64).sum(dim=0)
    fire_rate = fire_counts.float() / H_tr.shape[0]
    print(f'  per-unit firing rate on train: '
           f'mean={fire_rate.mean().item()*100:.2f}%  '
           f'min={fire_rate.min().item()*100:.4f}%  '
           f'max={fire_rate.max().item()*100:.2f}%')
    print(f'  dead units: {int((fire_rate == 0).sum().item())} / '
           f'{len(all_meta)}')

    # ---- Optional: greedy residual boosting ----
    if args.boost_count_rounds > 0:
        print(f'\ngreedy residual-boosting: {args.boost_count_rounds} rounds '
               f'× {args.boost_count_per_round} candidates/round...')
        # Build candidate pool: structured + random.
        cand_pool = build_structured_count_nodes()
        if args.boost_candidate_pool > 0:
            rn = build_random_count_nodes(args.boost_candidate_pool,
                                           seed=args.random_count_seed + 1)
            cand_pool.extend(rn)
        print(f'  candidate pool: {len(cand_pool)} nodes')

        # Skip candidates already added (via --include-count-nodes /
        # --random-count-nodes).  Use names.
        used_names = {n[0] for n in count_nodes_used}
        cand_pool = [c for c in cand_pool if c[0] not in used_names]
        print(f'  after removing already-included: {len(cand_pool)}')

        # Compute candidate activations once.
        played_tr = Xnp_tr[:, :60]; even_tr = Xnp_tr[:, 60:120]
        played_te = Xnp_te[:, :60]; even_te = Xnp_te[:, 60:120]
        print(f'  computing candidate activations...')
        t0 = time.time()
        CA_tr = compute_count_activations(cand_pool, played_tr, even_tr)
        CA_te = compute_count_activations(cand_pool, played_te, even_te)
        print(f'    CA_tr {CA_tr.shape} '
               f'({CA_tr.nbytes / 1e9:.2f} GB)  '
               f'({time.time() - t0:.1f}s)')

        added_mask = np.zeros(len(cand_pool), dtype=bool)

        for round_i in range(args.boost_count_rounds):
            print(f'\n  round {round_i + 1}/{args.boost_count_rounds}')
            # Fit probe on current H.
            probe = train_probe(H_tr, S_tr, H_te, S_te,
                                  epochs=max(args.probe_epochs // 2, 10),
                                  device=device, l1_lambda=args.probe_l1)
            # Compute residuals on training set (chunked, CPU-safe).
            device_probe = next(probe.parameters()).device
            residuals = np.zeros(
                (H_tr.shape[0], BOARD_CELLS * 3), dtype=np.float32)
            S_onehot = np.zeros(residuals.shape, dtype=np.float32)
            for i in range(0, H_tr.shape[0], 4096):
                h = H_tr[i:i + 4096].to(device=device_probe,
                                          dtype=torch.float32)
                with torch.no_grad():
                    logits = probe(h).view(-1, BOARD_CELLS, 3)
                    probs = torch.softmax(logits, dim=-1).view(
                        -1, BOARD_CELLS * 3).cpu().numpy()
                residuals[i:i + 4096] = probs
                # onehot
                s = S_tr[i:i + 4096].numpy()
                for k, s_arr in enumerate(s):
                    for c, cls in enumerate(s_arr):
                        S_onehot[i + k, c * 3 + int(cls)] = 1.0
            # residuals = onehot - probs (positive = probe underpredicts)
            residuals = S_onehot - residuals

            # Score each unused candidate: |cand_act.T @ residuals| summed
            # across output.
            scores = np.zeros(len(cand_pool), dtype=np.float64)
            for i in range(0, H_tr.shape[0], 4096):
                r_chunk = residuals[i:i + 4096]           # (b, 192)
                a_chunk = CA_tr[i:i + 4096].astype(np.float32)  # (b, M)
                # (M, 192) += a_chunk.T @ r_chunk
                scores += np.abs(a_chunk.T @ r_chunk).sum(axis=1)
            # Mask already-added candidates.
            scores[added_mask] = -1
            top_j = np.argsort(-scores)[:args.boost_count_per_round]
            top_j = top_j[scores[top_j] > 0]      # skip if all scored 0
            added_mask[top_j] = True
            print(f'    adding {len(top_j)} candidates (top score '
                   f'{scores[top_j[0]]:.1f})')

            # Append to H_tr, H_te.
            new_tr = torch.from_numpy(CA_tr[:, top_j])
            new_te = torch.from_numpy(CA_te[:, top_j])
            H_tr = torch.cat([H_tr, new_tr], dim=1)
            H_te = torch.cat([H_te, new_te], dim=1)
            for j in top_j:
                all_meta.append({
                    'kind': 'count_node_boosted',
                    'name': cand_pool[j][0],
                    'parity': cand_pool[j][2],
                    'threshold': cand_pool[j][3],
                    'round': round_i,
                })
            print(f'    H_tr now {tuple(H_tr.shape)}  '
                   f'H_te now {tuple(H_te.shape)}')

        # Free candidate activation storage.
        del CA_tr, CA_te, residuals, S_onehot

    skip_state_probe = (args.tree_target in ('legal', 'patterns')
                          or args.skip_state_probe)
    # Downstream save code expects tree_correct_per_cell — for patterns mode,
    # define it from the per-pattern accuracies.
    if (args.tree_target == 'patterns' and not args.skip_tree_fit
            and not args.load_trees_from):
        # In the --load-trees-from path tree_correct_per_cell is already loaded
        # from the checkpoint; tree_correct_per_pattern isn't computed (no re-fit).
        tree_correct_per_cell = tree_correct_per_pattern
    if skip_state_probe:
        print(f'\ntree-target=legal → skipping state probe (tree paths do '
               f'not distinguish state).')
        probes = None
        probe = None
        sk_models = None
        acc_tr = acc_te = 0.0
        per_cell_te = np.zeros(BOARD_CELLS)
        by_ply = {}
        per_seed_te = []
    else:
        print('\ntraining linear probe on hidden layer...')
        if args.probe_solver == 'sklearn':
            print(f'  solver: sklearn LR (LBFGS)   '
                   f'C={args.sklearn_C}   n_jobs={args.sklearn_n_jobs}')
            sk_models = train_probe_sklearn(
                H_tr, S_tr, H_te, S_te,
                C=args.sklearn_C, n_jobs=args.sklearn_n_jobs,
                subsample_train=args.sklearn_subsample_train,
                solver=args.sklearn_solver,
                max_iter=args.sklearn_max_iter)
            probes = None
            probe = None
        else:
            if args.probe_l1 > 0:
                print(f'  using L1 regularization: '
                       f'lambda={args.probe_l1}')
            print(f'  epochs: {args.probe_epochs}   '
                   f'seeds: {args.probe_seeds}   '
                   f'readout: {args.state_readout}')
            if args.state_readout == 'probor':
                probes = train_probe_state_probor_ensemble(
                    H_tr, S_tr,
                    n_seeds=args.probe_seeds,
                    epochs=args.probe_epochs, lr=args.probor_lr,
                    device=device)
            else:
                probes = train_probe_ensemble(
                    H_tr, S_tr, H_te, S_te,
                    n_seeds=args.probe_seeds,
                    epochs=args.probe_epochs, device=device,
                    l1_lambda=args.probe_l1)
            probe = probes[0]
            sk_models = None

    # Evaluate.
    if skip_state_probe:
        pass
    elif sk_models is not None:
        acc_tr, _, _ = evaluate_sklearn(sk_models, H_tr, S_tr)
        acc_te, per_cell_te, by_ply = evaluate_sklearn(
            sk_models, H_te, S_te, T_te)
        print(f'\nresults:')
        print(f'  hidden dim H = {H_tr.shape[1]} (sklearn LR probe)')
        print(f'  train per-cell acc: {100*acc_tr:.4f}%')
        print(f'  test  per-cell acc: {100*acc_te:.4f}%')
    else:
        if args.state_readout == 'probor':
            acc_tr, _, _, per_seed_tr = evaluate_state_probor_ensemble(
                probes, H_tr, S_tr)
            acc_te, per_cell_te, by_ply, per_seed_te = (
                evaluate_state_probor_ensemble(
                    probes, H_te, S_te, T_te))
        else:
            acc_tr, _, _, per_seed_tr = evaluate_ensemble(
                probes, H_tr, S_tr)
            acc_te, per_cell_te, by_ply, per_seed_te = evaluate_ensemble(
                probes, H_te, S_te, T_te)
        print(f'\nresults:')
        print(f'  hidden dim H = {H_tr.shape[1]} (tree paths + added units, '
               f'readout={args.state_readout})')
        if args.probe_seeds > 1:
            print(f'  per-seed test acc: '
                   f'{[f"{100*a:.2f}%" for a in per_seed_te]}')
            print(f'  per-seed range: '
                   f'[{100*min(per_seed_te):.2f}%, '
                   f'{100*max(per_seed_te):.2f}%]')
            print(f'  ensemble train acc: {100*acc_tr:.4f}%')
            print(f'  ensemble test  acc: {100*acc_te:.4f}%')
        else:
            print(f'  train per-cell acc: {100*acc_tr:.4f}%')
            print(f'  test  per-cell acc: {100*acc_te:.4f}%')

    # Per-cell class predictions for the corner/edge/inner breakdown.
    if skip_state_probe:
        preds_te = torch.zeros(H_te.shape[0], BOARD_CELLS, dtype=torch.long)
    elif sk_models is not None:
        H_te_np = H_te.numpy()
        if H_te_np.dtype == np.bool_:
            H_te_np = H_te_np.astype(np.float32)
        preds_te_np = np.zeros((H_te.shape[0], BOARD_CELLS),
                                 dtype=np.int64)
        for c in range(BOARD_CELLS):
            preds_te_np[:, c] = sk_models[c].predict(H_te_np)
        preds_te = torch.from_numpy(preds_te_np)
    else:
        device_probe = next(probes[0].parameters()).device
        preds_te = torch.empty(H_te.shape[0], BOARD_CELLS, dtype=torch.long,
                                 device='cpu')
        with torch.no_grad():
            for i in range(0, H_te.shape[0], 512):
                h = H_te[i:i + 512].to(device=device_probe,
                                         dtype=torch.float32)
                probs_acc = torch.zeros(h.shape[0], BOARD_CELLS, 3,
                                          dtype=torch.float32)
                for p in probes:
                    out = p(h)
                    if args.state_readout == 'probor':
                        # NoisyOrHead output is already normalized (b, 64, 3).
                        probs_acc += out.cpu()
                    else:
                        logits = out.view(-1, BOARD_CELLS, 3)
                        probs_acc += torch.softmax(logits, dim=-1).cpu()
                preds_te[i:i + 512] = probs_acc.argmax(dim=-1)
    if skip_state_probe:
        cls_break = {}
    else:
        cls_break = per_class_accuracy(preds_te, S_te)
        print(f'\n  test acc by cell class:')
        for cls, (n_cells, acc) in cls_break.items():
            print(f'    {cls:6s} ({n_cells} cells):  {100*acc:.4f}%')

        # Ply-bucket breakdown (bucket by 10).
        print(f'\n  test acc by ply bucket:')
        ply_arr = T_te.numpy()
        for lo in range(args.ply_min, args.ply_max, 10):
            hi = lo + 10
            mask = (ply_arr >= lo) & (ply_arr < hi)
            if not mask.any():
                continue
            acc_b = (preds_te[mask] == S_te[mask]).float().mean().item()
            print(f'    [{lo:2d},{hi:2d})  n={int(mask.sum()):6d}  '
                   f'acc={100*acc_b:.4f}%')

    # ------------------------------------------------------------------
    # Legal-move task: three predictors.
    # ------------------------------------------------------------------
    legal_results = {}

    def _save_checkpoint():
        torch.save({
            'W': mlp.W.cpu() if not args.skip_tree_fit else None,
            'b': mlp.b.cpu() if not args.skip_tree_fit else None,
            'probe_state': probe.state_dict() if probe is not None else None,
            'path_info': all_meta,
            'per_cell_leaf_counts': per_cell_leaf_counts
                if not args.skip_tree_fit else None,
            'per_cell_tree_acc': tree_correct_per_cell.tolist()
                if not args.skip_tree_fit else None,
            'per_class_probe_acc': cls_break,
            'args': vars(args),
            'test_acc': acc_te, 'train_acc': acc_tr,
            'by_ply': by_ply,
            'legal_results': legal_results,
        }, args.out)

    # Save immediately after tree fit + state probe, BEFORE the inline
    # legal probe.  Guarantees the trees survive even if the legal
    # probe hits the SLURM wall clock.
    _save_checkpoint()
    print(f'\nsaved (pre-legal) {args.out}')

    if args.skip_inline_legal_probe:
        print('--skip-inline-legal-probe set; exiting without training '
               'the legal probe.  Use the streaming pipeline instead.')
        return

    if args.task != 'state':
        L_tr = torch.from_numpy(Lnp_tr)
        L_te = torch.from_numpy(Lnp_te)
        modes = [m.strip() for m in args.legal_modes.split(',') if m.strip()]
        avg_legal = L_te.float().sum(dim=1).mean().item()
        legal_rate = L_te.float().mean().item()
        illegal_rate = 1.0 - legal_rate
        print(f'\n============================================')
        print(f'LEGAL-MOVE TASK  (modes={modes})')
        print(f'============================================')
        print(f'  avg legal-cells per position: {avg_legal:.2f} / 64')
        print(f'  chance-level per-cell acc (predict all illegal): '
               f'{100 * illegal_rate:.2f}%')

        def _print_legal_report(tag, acc, per_cell, aux):
            print(f'  {tag} ensemble test per-cell acc: {100*acc:.4f}%')
            print(f'  {tag} position-perfect: '
                   f'{100*aux["position_perfect"]:.4f}%')
            for (lo, hi), (n, a) in sorted(aux.get('by_ply', {}).items()):
                print(f'    ply [{lo:2d},{hi:2d})  n={n:6d}  '
                       f'acc={100*a:.4f}%')
            legal_results[tag] = {
                'test_acc': acc,
                'position_perfect': aux['position_perfect'],
                'by_ply': aux.get('by_ply', {}),
                'per_cell_acc': per_cell.tolist()
                    if hasattr(per_cell, 'tolist') else list(per_cell),
            }

        if 'bce' in modes:
            print(f'\ntraining legal-BCE probe ('
                   f'{args.probe_seeds} seed(s), '
                   f'{args.legal_probe_epochs} epochs)...')
            t0 = time.time()
            bce_probes = train_probe_legal_bce_ensemble(
                H_tr, L_tr,
                n_seeds=args.probe_seeds,
                epochs=args.legal_probe_epochs,
                device=device)
            print(f'  ({time.time() - t0:.1f}s)')
            acc, per_cell, aux = evaluate_legal_ensemble(
                bce_probes, H_te, L_te, T=T_te, kind='bce')
            _print_legal_report('BCE   ', acc, per_cell, aux)

        if 'probor' in modes:
            print(f'\ntraining legal-probOR probe ('
                   f'{args.probe_seeds} seed(s), '
                   f'{args.legal_probe_epochs} epochs)...')
            t0 = time.time()
            po_probes = train_probe_legal_probor_ensemble(
                H_tr, L_tr,
                n_seeds=args.probe_seeds,
                epochs=args.legal_probe_epochs,
                device=device)
            print(f'  ({time.time() - t0:.1f}s)')
            acc, per_cell, aux = evaluate_legal_ensemble(
                po_probes, H_te, L_te, T=T_te, kind='probor')
            _print_legal_report('probOR', acc, per_cell, aux)

        if 'derived' in modes and not skip_state_probe:
            print(f'\ndriving legal moves from state predictions...')
            acc, per_cell, aux = legal_accuracy_from_state(
                preds_te, L_te, T=T_te)
            _print_legal_report('DerivS', acc, per_cell, aux)
        elif 'derived' in modes:
            print(f'\nskipping derived legal (no state predictions '
                   f'available under tree-target=legal)')

        if ('patterns_probor' in modes and args.tree_target == 'patterns'
                and pattern_trees is not None):
            print(f'\ncomputing legal-move via prob-OR over '
                   f'per-pattern tree probabilities...')
            # For each pattern j: average over the pattern's trees'
            # predict_proba to get P(pattern fires).
            K = len(patterns_list)
            pat_probs = np.zeros((H_te.shape[0], K), dtype=np.float32)
            for j, tree_list in enumerate(pattern_trees):
                for tree in tree_list:
                    proba = tree.predict_proba(Xnp_te)
                    # class ordering may be [0] or [0, 1]; take the "1"
                    # column when present.
                    if proba.shape[1] == 2:
                        pat_probs[:, j] += proba[:, 1]
                    else:
                        # single-class tree — its prediction is always
                        # that class.  Predict = tree.classes_[0].
                        if int(tree.classes_[0]) == 1:
                            pat_probs[:, j] += 1.0
                pat_probs[:, j] /= len(tree_list)
            # Prob-OR combine per target cell.
            by_tgt = patterns_by_target(patterns_list)
            per_cell_legal = np.zeros((H_te.shape[0], BOARD_CELLS),
                                          dtype=np.float32)
            for cell, pattern_ids in by_tgt.items():
                per_cell_legal[:, cell] = (
                    1.0 - np.prod(1.0 - pat_probs[:, pattern_ids],
                                      axis=1))
            preds_legal = (per_cell_legal > 0.5).astype(np.uint8)
            L_te_np = L_te.numpy() if hasattr(L_te, 'numpy') else L_te
            correct = (preds_legal == L_te_np).astype(np.float32)
            per_cell = correct.mean(axis=0)
            mean_acc = float(correct.mean())
            aux = {'position_perfect':
                     float(correct.all(axis=1).mean())}
            T_np = T_te.numpy() if hasattr(T_te, 'numpy') else T_te
            by_ply = {}
            per_pos = correct.mean(axis=1)
            for lo in range(int(T_np.min()) // 10 * 10,
                              int(T_np.max()) + 1, 10):
                mask = (T_np >= lo) & (T_np < lo + 10)
                if mask.any():
                    by_ply[(lo, lo + 10)] = (int(mask.sum()),
                                                float(per_pos[mask].mean()))
            aux['by_ply'] = by_ply
            _print_legal_report('PatPO ', mean_acc, per_cell, aux)

        if ('patterns_structured_probor' in modes
                and args.tree_target == 'patterns'):
            print(f'\ntraining structured pattern-probor probe (per-'
                   f'pattern linear over leaves → sigmoid → prob-OR per '
                   f'cell)...')
            t0 = time.time()
            struct_probes = (
                train_probe_legal_patterns_structured_ensemble(
                    H_tr, L_tr, all_meta, patterns_list,
                    n_seeds=args.probe_seeds,
                    epochs=args.legal_probe_epochs,
                    device=device))
            print(f'  ({time.time() - t0:.1f}s)')
            # Ensemble by averaging per-cell probabilities.
            device_probe = next(struct_probes[0].parameters()).device
            N_te = H_te.shape[0]
            accum = torch.zeros(N_te, BOARD_CELLS, dtype=torch.float32)
            for i in range(0, N_te, 512):
                h = H_te[i:i + 512].to(device=device_probe,
                                         dtype=torch.float32)
                probs = torch.zeros(h.shape[0], BOARD_CELLS,
                                      dtype=torch.float32,
                                      device=device_probe)
                for p in struct_probes:
                    with torch.no_grad():
                        probs = probs + p(h)
                probs = probs / len(struct_probes)
                accum[i:i + 512] = probs.cpu()
            preds_struct = (accum > 0.5).to(torch.uint8)
            L_te_np = L_te.numpy() if hasattr(L_te, 'numpy') else L_te
            correct = (preds_struct.numpy() == L_te_np).astype(np.float32)
            per_cell = correct.mean(axis=0)
            mean_acc = float(correct.mean())
            aux = {'position_perfect':
                     float(correct.all(axis=1).mean())}
            T_np = T_te.numpy() if hasattr(T_te, 'numpy') else T_te
            by_ply = {}
            per_pos = correct.mean(axis=1)
            for lo in range(int(T_np.min()) // 10 * 10,
                              int(T_np.max()) + 1, 10):
                mask = (T_np >= lo) & (T_np < lo + 10)
                if mask.any():
                    by_ply[(lo, lo + 10)] = (int(mask.sum()),
                                                float(per_pos[mask].mean()))
            aux['by_ply'] = by_ply
            _print_legal_report('Linear->ProbOR', mean_acc, per_cell, aux)

        if ('cells_structured_probor' in modes
                and args.tree_target == 'legal'):
            print(f'\ntraining structured per-cell legal probe (per-cell '
                   f'linear over legaltree leaves → sigmoid)...')
            t0 = time.time()
            cell_probes = train_probe_legal_cells_structured_ensemble(
                H_tr, L_tr, all_meta,
                n_seeds=args.probe_seeds,
                epochs=args.legal_probe_epochs,
                device=device)
            print(f'  ({time.time() - t0:.1f}s)')
            device_probe = next(cell_probes[0].parameters()).device
            N_te = H_te.shape[0]
            accum = torch.zeros(N_te, BOARD_CELLS, dtype=torch.float32)
            for i in range(0, N_te, 512):
                h = H_te[i:i + 512].to(device=device_probe,
                                         dtype=torch.float32)
                probs = torch.zeros(h.shape[0], BOARD_CELLS,
                                      dtype=torch.float32,
                                      device=device_probe)
                for p in cell_probes:
                    with torch.no_grad():
                        probs = probs + p(h)
                probs = probs / len(cell_probes)
                accum[i:i + 512] = probs.cpu()
            preds_cell = (accum > 0.5).to(torch.uint8)
            L_te_np = L_te.numpy() if hasattr(L_te, 'numpy') else L_te
            correct = (preds_cell.numpy() == L_te_np).astype(np.float32)
            per_cell_arr = correct.mean(axis=0)
            mean_acc = float(correct.mean())
            aux = {'position_perfect':
                     float(correct.all(axis=1).mean())}
            T_np = T_te.numpy() if hasattr(T_te, 'numpy') else T_te
            by_ply = {}
            per_pos = correct.mean(axis=1)
            for lo in range(int(T_np.min()) // 10 * 10,
                              int(T_np.max()) + 1, 10):
                mask = (T_np >= lo) & (T_np < lo + 10)
                if mask.any():
                    by_ply[(lo, lo + 10)] = (int(mask.sum()),
                                                float(per_pos[mask].mean()))
            aux['by_ply'] = by_ply
            _print_legal_report('CellPO', mean_acc, per_cell_arr, aux)

        if ('patterns_linear_probor' in modes
                and args.include_flanking_patterns):
            # Load patterns (may already be loaded in tree-target=patterns,
            # but load again here for safety).
            if 'patterns_list' in dir() and patterns_list is not None:
                pl = patterns_list
            else:
                pl = load_patterns(args.include_flanking_patterns)
            print(f'\ntraining linear->960 patterns->prob-OR readout '
                   f'({len(pl)} patterns; batch=1024, faster than StruPO)...')
            t0 = time.time()
            lin_probes = train_probe_legal_linear_pat_probor_ensemble(
                H_tr, L_tr, pl,
                n_seeds=args.probe_seeds,
                epochs=args.legal_probe_epochs,
                device=device)
            print(f'  ({time.time() - t0:.1f}s)')
            device_probe = next(lin_probes[0].parameters()).device
            N_te = H_te.shape[0]
            accum = torch.zeros(N_te, BOARD_CELLS, dtype=torch.float32)
            for i in range(0, N_te, 1024):
                h = H_te[i:i + 1024].to(device=device_probe,
                                          dtype=torch.float32)
                probs = torch.zeros(h.shape[0], BOARD_CELLS,
                                      dtype=torch.float32,
                                      device=device_probe)
                for p in lin_probes:
                    with torch.no_grad():
                        probs = probs + p(h)
                probs = probs / len(lin_probes)
                accum[i:i + 1024] = probs.cpu()
            preds_lin = (accum > 0.5).to(torch.uint8)
            L_te_np = L_te.numpy() if hasattr(L_te, 'numpy') else L_te
            correct = (preds_lin.numpy() == L_te_np).astype(np.float32)
            per_cell_arr = correct.mean(axis=0)
            mean_acc = float(correct.mean())
            aux = {'position_perfect':
                     float(correct.all(axis=1).mean())}
            T_np = T_te.numpy() if hasattr(T_te, 'numpy') else T_te
            by_ply = {}
            per_pos = correct.mean(axis=1)
            for lo in range(int(T_np.min()) // 10 * 10,
                              int(T_np.max()) + 1, 10):
                mask = (T_np >= lo) & (T_np < lo + 10)
                if mask.any():
                    by_ply[(lo, lo + 10)] = (int(mask.sum()),
                                                float(per_pos[mask].mean()))
            aux['by_ply'] = by_ply
            _print_legal_report('LinPO ', mean_acc, per_cell_arr, aux)

        if ('state_probor' in modes and not skip_state_probe
                and args.include_flanking_patterns):
            print(f'\ncomputing legal-move via prob-OR of 960 flanking '
                   f'patterns on SOFT state predictions...')
            # Reuse `patterns` from the include-flanking-patterns branch.
            # Compute ensemble state probabilities on H_te (already done
            # implicitly for preds_te, but we need probabilities not argmax).
            device_probe = next(probes[0].parameters()).device
            N_te = H_te.shape[0]
            state_probs = torch.zeros(N_te, BOARD_CELLS, 3,
                                          dtype=torch.float32)
            with torch.no_grad():
                for i in range(0, N_te, 512):
                    h = H_te[i:i + 512].to(device=device_probe,
                                             dtype=torch.float32)
                    accum = torch.zeros(h.shape[0], BOARD_CELLS, 3,
                                          dtype=torch.float32,
                                          device=device_probe)
                    for p in probes:
                        out = p(h)
                        if args.state_readout == 'probor':
                            accum += out
                        else:
                            accum += torch.softmax(
                                out.view(-1, BOARD_CELLS, 3), dim=-1)
                    accum = accum / len(probes)
                    state_probs[i:i + 512] = accum.cpu()
            legal_prob_np = legal_from_state_probs_via_patterns(
                patterns, state_probs.numpy())
            preds_legal = (legal_prob_np > 0.5).astype(np.uint8)
            L_te_np = L_te.numpy() if hasattr(L_te, 'numpy') else L_te
            correct = (preds_legal == L_te_np).astype(np.float32)
            per_cell = correct.mean(axis=0)
            mean_acc = float(correct.mean())
            aux = {'position_perfect':
                     float(correct.all(axis=1).mean())}
            T_np = T_te.numpy() if hasattr(T_te, 'numpy') else T_te
            by_ply = {}
            per_pos = correct.mean(axis=1)
            for lo in range(int(T_np.min()) // 10 * 10,
                              int(T_np.max()) + 1, 10):
                mask = (T_np >= lo) & (T_np < lo + 10)
                if mask.any():
                    by_ply[(lo, lo + 10)] = (int(mask.sum()),
                                                 float(per_pos[mask].mean()))
            aux['by_ply'] = by_ply
            _print_legal_report('StatPO', mean_acc, per_cell, aux)
        elif 'state_probor' in modes:
            print(f'\nskipping state_probor legal (requires state probe + '
                   f'--include-flanking-patterns)')

    _save_checkpoint()
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
