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
    train_per_cell_trees, train_probe, train_probe_ensemble,
    train_probe_sklearn, evaluate, evaluate_ensemble, evaluate_sklearn,
    train_probe_legal_bce_ensemble, train_probe_legal_probor_ensemble,
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
)


# ------------------------------------------------------------------------------
# Mid-game position sampling
# ------------------------------------------------------------------------------

def sample_midgame_positions(num_games, ply_min=10, ply_max=50, seed=42,
                                when_bucket_size=None,
                                use_move_grid=False,
                                recent_Ks=None,
                                collect_legal_moves=False):
    """Play random games; extract positions with ply in [ply_min, ply_max).
    Returns (X, S, T) — or (X, S, T, L) if collect_legal_moves — with:
      X: (N, ...) played_even features
      S: (N, 64) int64 state labels (0 empty, 1 mine, 2 opp)
      T: (N,) int32 — ply index within the game
      L: (N, 64) uint8 legal-move mask (1 iff cell is a legal next move)
    """
    rng = np.random.RandomState(seed)
    Xs, Ss, Ts, Ls = [], [], [], []
    for _ in range(num_games):
        board = OthelloBoardState()
        prefix = []
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
                Xs.append(playedeven_features(prefix, when_bucket_size,
                                                use_move_grid,
                                                recent_Ks=recent_Ks))
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


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-train-games', type=int, default=20000)
    ap.add_argument('--num-test-games', type=int, default=5000)
    ap.add_argument('--ply-min', type=int, default=10)
    ap.add_argument('--ply-max', type=int, default=50)
    ap.add_argument('--tree-max-depth', type=int, default=15)
    ap.add_argument('--tree-min-samples-leaf', type=int, default=5)
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
    ap.add_argument('--input-recent-Ks', default='',
                    help='Comma-separated K values.  For each K, appends 60 '
                          'bits (one per non-center cell) that fire iff the '
                          'cell was played in the last K turns.  Trees fit '
                          'on the enlarged input directly — no separate '
                          'order-node hidden bank needed.  E.g., "5" gives '
                          'played+even+recent = 60x3 input (+ mover_parity).')
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
                    choices=['state', 'legal'],
                    help='What per-cell trees predict.  state (default): '
                          '3-class {empty, mine, opp} — tree paths become '
                          'state-decoding features.  legal: 2-class '
                          '{illegal, legal} — tree paths encode flanking-'
                          'like conjunctions predictive of legality.  Only '
                          'meaningful with --task legal or --task both.')
    ap.add_argument('--task', default='state',
                    choices=['state', 'legal', 'both'],
                    help='state (default): train state-decoding probe only. '
                          'legal: only legal-move probes.  both: state + '
                          'all three legal-move predictors on the same H.')
    ap.add_argument('--legal-modes', default='bce,probor,derived',
                    help='For task in {legal,both}: comma-separated subset '
                          'of {bce, probor, derived} to run.')
    ap.add_argument('--legal-probe-epochs', type=int, default=50)
    ap.add_argument('--skip-tree-fit', action='store_true',
                    help='Skip per-cell tree fit + path extraction entirely. '
                          'Hidden layer starts empty (N, 0); only count / '
                          'order banks contribute.  Ablation: how much do '
                          'the tree paths add on top of order features?')
    ap.add_argument('--cache-tr', default=None,
                    help='Path to .npz cache for the sampled TRAIN set.')
    ap.add_argument('--cache-te', default=None,
                    help='Path to .npz cache for the sampled TEST set.')
    args = ap.parse_args()

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
        collect_legal_moves=collect_legal)
    te = load_or_sample(
        args.cache_te, sample_midgame_positions,
        args.num_test_games, ply_min=args.ply_min,
        ply_max=args.ply_max, seed=args.seed + 1_000_000,
        when_bucket_size=args.when_bucket_size,
        use_move_grid=args.use_move_grid,
        recent_Ks=sampling_Ks,
        collect_legal_moves=collect_legal)
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

    S_tr = torch.from_numpy(Snp_tr)
    S_te = torch.from_numpy(Snp_te)
    T_te = torch.from_numpy(Tnp_te)
    use_relu = args.hidden_activation == 'relu'
    act_dtype = torch.float32 if use_relu else torch.bool

    if args.skip_tree_fit:
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
        if args.tree_target == 'legal':
            if Lnp_tr is None:
                raise ValueError('--tree-target legal requires --task legal '
                                  'or --task both to collect legal masks.')
            tree_target_tr = Lnp_tr.astype(np.int64)
            tree_target_te = Lnp_te.astype(np.int64)
            print(f'  fitting trees for LEGAL-MOVE target (binary per cell)')
        else:
            tree_target_tr = Snp_tr
            tree_target_te = Snp_te
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
            tree_correct_per_cell[c] = (preds == tree_target_te[:, c]).mean()
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
        per_cell_leaf_counts = np.zeros(BOARD_CELLS, dtype=int)
        input_dim = Xnp_tr.shape[1]
        for c in range(BOARD_CELLS):
            paths = extract_paths(trees[c])
            paths = prune_paths_by_count(paths, args.top_k_per_cell)
            per_cell_leaf_counts[c] = len(paths)
            for path_idx, (conditions, leaf_class, leaf_counts) in enumerate(paths):
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
        n_tree_units = len(all_meta)

        if args.add_stability:
            print('  adding stability feature bank...')
            Ws, Bs, meta_s = build_stability_features(device)
            all_w.extend(Ws)
            all_b.extend(Bs.tolist())
            all_meta.extend(meta_s)
            print(f'  stability units added: {Ws.shape[0]}')

        W = np.stack(all_w); B = np.array(all_b, dtype=np.float32)
        print(f'  total hidden units: {len(all_meta)}   '
               f'(tree={n_tree_units}, stability='
               f'{len(all_meta) - n_tree_units})')
        print(f'  leaves per tree: mean={per_cell_leaf_counts.mean():.1f}  '
               f'max={per_cell_leaf_counts.max()}  '
               f'min={per_cell_leaf_counts.min()}')

        depths = np.array([m['depth'] for m in all_meta
                            if m.get('kind') == 'tree_path'])
        print(f'  tree-path depths: mean={depths.mean():.2f}  '
               f'max={depths.max()}  min={depths.min()}')

        mlp = OpeningTreeMLP(W, B, all_meta, device)

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
        del X_tr, X_te
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
        for k_idx, K in enumerate(hidden_recent_Ks):
            for cell60 in range(60):
                cell64 = C60_TO_C64[cell60]
                alg = 'ABCDEFGH'[cell64 % 8] + str(cell64 // 8 + 1)
                all_meta.append({
                    'kind': 'recent_bit',
                    'name': f'recent{K}[{alg}]',
                    'K': K, 'cell60': cell60,
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

    skip_state_probe = (args.tree_target == 'legal')
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
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
