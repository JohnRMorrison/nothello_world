"""Build a single-hidden-layer interpretable MLP for the Othello ENDGAME
(last 10 ply of each game) using the same per-cell-decision-tree recipe
as `opening_tree_mlp.py`, but with:

  - endgame position sampling (play games to completion, take the last 10 ply),
  - cell-class breakdown (corner / edge / inner) in the results.

Each hidden unit is one root-to-leaf path from a per-cell decision tree,
encoded as a 0/±1 conjunction rule.  A linear probe on the hidden layer
predicts per-cell board state.

The endgame hypothesis: corners never flip (accuracy → 100% with trivial
rules), edges rarely flip (small trees), interior cells are the hard part
even in the endgame.
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
    load_or_sample, prune_paths_by_count,
    BOARD_CELLS, INPUT_DIM, CENTER_64, NON_CENTER_64,
    C64_TO_C60, C60_TO_C64, STATE_NAMES,
)


# ------------------------------------------------------------------------------
# Cell-class classification
# ------------------------------------------------------------------------------

CORNERS_64 = {0, 7, 56, 63}   # A1, H1, A8, H8


def cell_class(c64):
    r, c = c64 // 8, c64 % 8
    if c64 in CORNERS_64:
        return 'corner'
    if r in (0, 7) or c in (0, 7):
        return 'edge'
    return 'inner'


CELL_CLASS = {c: cell_class(c) for c in range(64)}


# ------------------------------------------------------------------------------
# Endgame position sampling
# ------------------------------------------------------------------------------

def sample_endgame_positions(num_games, endgame_ply=10, seed=42):
    """Play random games to completion, extract the LAST `endgame_ply`
    positions of each game.  Returns (X, S, plies) with:
      X: (N, 120) played_even features
      S: (N, 64) int64 class labels (0 empty, 1 mine, 2 opp)
      plies: (N,) int32 — the actual ply index within the game (0..~59)
    """
    rng = np.random.RandomState(seed)
    all_Xs, all_Ss, all_Ts = [], [], []
    for _ in range(num_games):
        board = OthelloBoardState()
        prefix = []
        game_positions = []
        for turn in range(60):
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
            game_positions.append((
                playedeven_features(prefix), lbl, len(prefix)))
            move = valid[rng.randint(len(valid))]
            board.update([move])
            prefix.append(move)
        # Take the last endgame_ply positions from this game.
        for feat, lbl, ply in game_positions[-endgame_ply:]:
            all_Xs.append(feat)
            all_Ss.append(lbl)
            all_Ts.append(ply)
    return (np.stack(all_Xs), np.stack(all_Ss),
             np.array(all_Ts, dtype=np.int32))


# ------------------------------------------------------------------------------
# Per-class accuracy breakdown
# ------------------------------------------------------------------------------

def per_class_accuracy(preds, labels):
    """preds, labels: (N, 64) tensors.  Returns dict class -> mean accuracy."""
    correct = (preds == labels).float()   # (N, 64)
    out = {}
    for cls in ('corner', 'edge', 'inner'):
        cell_ids = [c for c in range(64) if CELL_CLASS[c] == cls]
        acc = correct[:, cell_ids].mean().item()
        n_cells = len(cell_ids)
        out[cls] = (n_cells, acc)
    return out


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-train-games', type=int, default=20000)
    ap.add_argument('--num-test-games', type=int, default=5000)
    ap.add_argument('--endgame-ply', type=int, default=10,
                    help='Number of final ply of each game to include.')
    ap.add_argument('--tree-max-depth', type=int, default=15)
    ap.add_argument('--tree-min-samples-leaf', type=int, default=5)
    ap.add_argument('--tree-n-jobs', type=int, default=1)
    ap.add_argument('--probe-epochs', type=int, default=25)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='endgame_tree_mlp.pt')
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

    print(f'sampling {args.num_train_games} train + '
           f'{args.num_test_games} test games, last {args.endgame_ply} ply...')
    t0 = time.time()
    Xnp_tr, Snp_tr, Tnp_tr = load_or_sample(
        args.cache_tr, sample_endgame_positions,
        args.num_train_games, endgame_ply=args.endgame_ply, seed=args.seed)
    Xnp_te, Snp_te, Tnp_te = load_or_sample(
        args.cache_te, sample_endgame_positions,
        args.num_test_games, endgame_ply=args.endgame_ply,
        seed=args.seed + 1_000_000)
    print(f'  train={Xnp_tr.shape[0]}  test={Xnp_te.shape[0]}  '
           f'({time.time() - t0:.1f}s)')
    print(f'  training ply range: [{Tnp_tr.min()}, {Tnp_tr.max()}]  '
           f'mean={Tnp_tr.mean():.1f}')

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

    # Per-cell tree accuracy on test + break down by cell class.
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
        paths = prune_paths_by_count(paths, args.top_k_per_cell)
        per_cell_leaf_counts[c] = len(paths)
        for path_idx, (conditions, leaf_class, leaf_counts) in enumerate(paths):
            w, b = path_to_weight(conditions)
            all_w.append(w); all_b.append(b)
            all_meta.append({
                'cell': c, 'path_idx': path_idx,
                'conditions': conditions, 'leaf_class': leaf_class,
                'depth': len(conditions), 'leaf_counts': leaf_counts,
                'train_count': sum(leaf_counts),
                'cell_class': CELL_CLASS[c],
            })
    W = np.stack(all_w); B = np.array(all_b, dtype=np.float32)
    print(f'  total hidden units: {len(all_meta)}')
    print(f'  leaves per tree: mean={per_cell_leaf_counts.mean():.1f}  '
           f'max={per_cell_leaf_counts.max()}  '
           f'min={per_cell_leaf_counts.min()}')
    print(f'  leaves per tree by class:')
    for cls in ('corner', 'edge', 'inner'):
        cell_ids = [c for c in range(64) if CELL_CLASS[c] == cls]
        v = per_cell_leaf_counts[cell_ids]
        print(f'    {cls:6s}: mean={v.mean():.1f}  '
               f'min={v.min()}  max={v.max()}')

    depths = np.array([m['depth'] for m in all_meta])
    print(f'  path depths: mean={depths.mean():.2f}  '
           f'max={depths.max()}  min={depths.min()}')

    mlp = OpeningTreeMLP(W, B, all_meta, device)

    X_tr = torch.from_numpy(Xnp_tr).to(device)
    X_te = torch.from_numpy(Xnp_te).to(device)
    # Labels + hidden activations kept on CPU; batches move to device.
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
    del X_tr, X_te
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    counts = torch.zeros(H_tr.shape[1], dtype=torch.int64)
    for i in range(0, H_tr.shape[0], 512):
        counts += H_tr[i:i + 512].to(torch.int64).sum(dim=0)
    fire_rate = counts.float() / H_tr.shape[0]
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

    # Per-class accuracy: run probe batchwise on H_te.
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

    print(f'\n  test acc by actual ply:')
    for ply in sorted(by_ply.keys()):
        n, acc = by_ply[ply]
        print(f'    ply {ply:2d}:  n={n:6d}  acc={100*acc:.4f}%')

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
