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
    train_per_cell_trees, train_probe, evaluate, OpeningTreeMLP,
    BOARD_CELLS, INPUT_DIM, CENTER_64, NON_CENTER_64,
    C64_TO_C60, C60_TO_C64, STATE_NAMES,
)
from endgame_tree_mlp import (
    CORNERS_64, cell_class, CELL_CLASS, per_class_accuracy,
)


# ------------------------------------------------------------------------------
# Mid-game position sampling
# ------------------------------------------------------------------------------

def sample_midgame_positions(num_games, ply_min=10, ply_max=50, seed=42):
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
                Xs.append(playedeven_features(prefix))
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
    Xnp_tr, Snp_tr, Tnp_tr = sample_midgame_positions(
        args.num_train_games, ply_min=args.ply_min,
        ply_max=args.ply_max, seed=args.seed)
    Xnp_te, Snp_te, Tnp_te = sample_midgame_positions(
        args.num_test_games, ply_min=args.ply_min,
        ply_max=args.ply_max, seed=args.seed + 1_000_000)
    print(f'  train={Xnp_tr.shape[0]}  test={Xnp_te.shape[0]}  '
           f'({time.time() - t0:.1f}s)')

    # --- Train per-cell trees ---
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
    for c in range(BOARD_CELLS):
        paths = extract_paths(trees[c])
        per_cell_leaf_counts[c] = len(paths)
        for path_idx, (conditions, leaf_class, leaf_counts) in enumerate(paths):
            w, b = path_to_weight(conditions)
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

    print('\ncomputing hidden activations (bool on CPU)...')
    t0 = time.time()
    H_tr = mlp(X_tr, out_device='cpu', out_dtype=torch.bool)
    H_te = mlp(X_te, out_device='cpu', out_dtype=torch.bool)
    print(f'  H_tr {tuple(H_tr.shape)} '
           f'({H_tr.element_size() * H_tr.nelement() / 1e9:.2f} GB)  '
           f'H_te {tuple(H_te.shape)} '
           f'({H_te.element_size() * H_te.nelement() / 1e9:.2f} GB)  '
           f'({time.time() - t0:.1f}s)')
    del X_tr, X_te
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    fire_rate = H_tr.float().mean(dim=0)
    print(f'  per-unit firing rate on train: '
           f'mean={fire_rate.mean().item()*100:.2f}%  '
           f'min={fire_rate.min().item()*100:.4f}%  '
           f'max={fire_rate.max().item()*100:.2f}%')
    print(f'  dead units: {int((fire_rate == 0).sum().item())} / '
           f'{len(all_meta)}')

    print('\ntraining linear probe on hidden layer...')
    probe = train_probe(H_tr, S_tr, H_te, S_te,
                          epochs=args.probe_epochs, device=device)

    acc_tr, _, _ = evaluate(probe, H_tr, S_tr)
    acc_te, per_cell_te, by_ply = evaluate(probe, H_te, S_te, T_te)
    print(f'\nresults:')
    print(f'  hidden dim H = {mlp.hidden_dim}')
    print(f'  train per-cell acc: {100*acc_tr:.4f}%')
    print(f'  test  per-cell acc: {100*acc_te:.4f}%')

    device_probe = next(probe.parameters()).device
    preds_te = torch.empty(H_te.shape[0], BOARD_CELLS, dtype=torch.long,
                             device='cpu')
    with torch.no_grad():
        for i in range(0, H_te.shape[0], 4096):
            h = H_te[i:i + 4096].to(device=device_probe, dtype=torch.float32)
            preds_te[i:i + 4096] = probe(h).view(-1, BOARD_CELLS, 3
                                                    ).argmax(dim=-1).cpu()
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
        'probe_state': probe.state_dict(),
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
