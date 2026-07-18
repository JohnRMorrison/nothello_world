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


# ------------------------------------------------------------------------------
# Mid-game position sampling
# ------------------------------------------------------------------------------

def sample_midgame_positions(num_games, ply_min=10, ply_max=50, seed=42,
                                when_bucket_size=None,
                                use_move_grid=False):
    """Play random games; extract positions with ply in [ply_min, ply_max).
    Returns (X, S, T) with:
      X: (N, 120) played_even features
      S: (N, 64) int64 class labels (0 empty, 1 mine, 2 opp)
      T: (N,) int32 — ply index within the game
    """
    rng = np.random.RandomState(seed)
    Xs, Ss, Ts = [], [], []
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
                                                use_move_grid))
                Ss.append(lbl)
                Ts.append(ply)
            move = valid[rng.randint(len(valid))]
            board.update([move])
            prefix.append(move)
    if not Xs:
        raise RuntimeError(
            f'no positions extracted; ply range [{ply_min},{ply_max}) '
            f'may not overlap game play')
    return (np.stack(Xs), np.stack(Ss),
             np.array(Ts, dtype=np.int32))


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

    print(f'sampling {args.num_train_games} train + '
           f'{args.num_test_games} test games...')
    t0 = time.time()
    Xnp_tr, Snp_tr, Tnp_tr = load_or_sample(
        args.cache_tr, sample_midgame_positions,
        args.num_train_games, ply_min=args.ply_min,
        ply_max=args.ply_max, seed=args.seed,
        when_bucket_size=args.when_bucket_size,
        use_move_grid=args.use_move_grid)
    Xnp_te, Snp_te, Tnp_te = load_or_sample(
        args.cache_te, sample_midgame_positions,
        args.num_test_games, ply_min=args.ply_min,
        ply_max=args.ply_max, seed=args.seed + 1_000_000,
        when_bucket_size=args.when_bucket_size,
        use_move_grid=args.use_move_grid)

    # If a cache from a wider ply range was loaded, narrow it to the current
    # args range.  Lets a single (10, 50) cache serve any [a, b) subwindow.
    def _narrow(X, S, T):
        mask = (T >= args.ply_min) & (T < args.ply_max)
        if mask.all():
            return X, S, T
        n_before = X.shape[0]
        X = X[mask]; S = S[mask]; T = T[mask]
        print(f'  filtered cache: {n_before} → {X.shape[0]} positions '
               f'in ply [{args.ply_min}, {args.ply_max})')
        return X, S, T

    Xnp_tr, Snp_tr, Tnp_tr = _narrow(Xnp_tr, Snp_tr, Tnp_tr)
    Xnp_te, Snp_te, Tnp_te = _narrow(Xnp_te, Snp_te, Tnp_te)
    print(f'  train={Xnp_tr.shape[0]}  test={Xnp_te.shape[0]}  '
           f'({time.time() - t0:.1f}s)')

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

    # --- Train per-cell trees ---
    print(f'\ntraining per-cell trees (max_depth={args.tree_max_depth}, '
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
    trees = train_per_cell_trees(
        Xnp_tr, Snp_tr,
        max_depth=args.tree_max_depth,
        min_samples_leaf=args.tree_min_samples_leaf,
        n_jobs=args.tree_n_jobs,
        max_features=mf)
    print(f'  ({time.time() - t0:.1f}s)')

    tree_correct_per_cell = np.zeros(BOARD_CELLS)
    for c in range(BOARD_CELLS):
        preds = trees[c].predict(Xnp_te)
        tree_correct_per_cell[c] = (preds == Snp_te[:, c]).mean()
    print(f'  aggregate per-cell tree test acc: '
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
    S_tr = torch.from_numpy(Snp_tr)
    S_te = torch.from_numpy(Snp_te)
    T_te = torch.from_numpy(Tnp_te)

    use_relu = args.hidden_activation == 'relu'
    if use_relu:
        act_dtype = torch.float32
        print('\ncomputing hidden activations (ReLU, float32 on CPU)...')
    else:
        act_dtype = torch.bool
        print('\ncomputing hidden activations (step, bool on CPU)...')
    t0 = time.time()
    H_tr = mlp(X_tr, out_device='cpu', out_dtype=act_dtype,
                 use_relu=use_relu)
    H_te = mlp(X_te, out_device='cpu', out_dtype=act_dtype,
                 use_relu=use_relu)
    print(f'  H_tr {tuple(H_tr.shape)} '
           f'({H_tr.element_size() * H_tr.nelement() / 1e9:.2f} GB)  '
           f'H_te {tuple(H_te.shape)} '
           f'({H_te.element_size() * H_te.nelement() / 1e9:.2f} GB)  '
           f'({time.time() - t0:.1f}s)')
    del X_tr, X_te
    if device.type == 'cuda':
        torch.cuda.empty_cache()

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
        # For consistency with downstream code we use ensemble containers
        # but fill with the sklearn models.
        probes = None      # legacy variable, unused with sklearn path
        probe = None
    else:
        if args.probe_l1 > 0:
            print(f'  using L1 regularization: lambda={args.probe_l1}')
        print(f'  epochs: {args.probe_epochs}   seeds: {args.probe_seeds}')
        probes = train_probe_ensemble(
            H_tr, S_tr, H_te, S_te,
            n_seeds=args.probe_seeds,
            epochs=args.probe_epochs, device=device,
            l1_lambda=args.probe_l1)
        probe = probes[0]     # for legacy references
        sk_models = None

    # Evaluate.
    if sk_models is not None:
        acc_tr, _, _ = evaluate_sklearn(sk_models, H_tr, S_tr)
        acc_te, per_cell_te, by_ply = evaluate_sklearn(
            sk_models, H_te, S_te, T_te)
        print(f'\nresults:')
        print(f'  hidden dim H = {H_tr.shape[1]} (sklearn LR probe)')
        print(f'  train per-cell acc: {100*acc_tr:.4f}%')
        print(f'  test  per-cell acc: {100*acc_te:.4f}%')
    else:
        acc_tr, _, _, per_seed_tr = evaluate_ensemble(probes, H_tr, S_tr)
        acc_te, per_cell_te, by_ply, per_seed_te = evaluate_ensemble(
            probes, H_te, S_te, T_te)
        print(f'\nresults:')
        print(f'  hidden dim H = {H_tr.shape[1]} (tree paths + added units)')
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
    if sk_models is not None:
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
                    logits = p(h).view(-1, BOARD_CELLS, 3)
                    probs_acc += torch.softmax(logits, dim=-1).cpu()
                preds_te[i:i + 512] = probs_acc.argmax(dim=-1)
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

    torch.save({
        'W': mlp.W.cpu(), 'b': mlp.b.cpu(),
        'probe_state': probe.state_dict() if probe is not None else None,
        'path_info': all_meta,
        'per_cell_leaf_counts': per_cell_leaf_counts,
        'per_cell_tree_acc': tree_correct_per_cell.tolist(),
        'per_class_probe_acc': cls_break,
        'args': vars(args),
        'test_acc': acc_te, 'train_acc': acc_tr,
        'by_ply': by_ply,
    }, args.out)
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
