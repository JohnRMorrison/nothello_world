"""Evaluate the N independently-trained MLPs from a multi-seed checkpoint.

Reports:
  - Per-seed top-1 legal accuracy (histogram)
  - % positions where ALL N seeds are wrong (the irreducible floor)
  - Distribution of "# seeds correct" per position

Usage:
    python eval_multi_seed_ensemble.py \\
        --multi-ckpt experiments/.../multi_seed_N50_H4096_playedeven.pt \\
        --num-games 500
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from train_multi_seed_mlp import VectorizedMLP
from train_pattern_simple import _get_cell_pat_index
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from compare_v4_vs_mlp import (
    load_val_games, played_even_features, C64_TO_C60,
)
from data.othello import OthelloBoardState


def legal_cells_60(game, k):
    board = OthelloBoardState()
    for c in game[:k]:
        try:
            board.umpire(c)
        except Exception:
            return None
    legal_64 = board.get_valid_moves()
    return {C64_TO_C60[c] for c in legal_64 if c in C64_TO_C60}


def load_vectorized_from_multi(multi_path, device):
    """Pack N saved seed-states back into two VectorizedMLPs (even, odd)."""
    ckpt = torch.load(multi_path, map_location='cpu')
    N = ckpt['num_seeds']
    input_dim = ckpt['input_dim']
    hidden = ckpt['hidden_dim']
    n_patterns = ckpt['n_patterns']
    me = VectorizedMLP(N, input_dim, hidden, n_patterns).to(device)
    mo = VectorizedMLP(N, input_dim, hidden, n_patterns).to(device)
    # state_per_seed wrote net.0.weight as (hidden, input) — transpose back
    # to VectorizedMLP's (N, input, hidden) layout.
    with torch.no_grad():
        for s in range(N):
            ev = ckpt['all_seeds'][s]['even']
            me.W1.data[s] = ev['net.0.weight'].t().to(device)
            me.b1.data[s, 0] = ev['net.0.bias'].to(device)
            me.W2.data[s] = ev['net.2.weight'].t().to(device)
            me.b2.data[s, 0] = ev['net.2.bias'].to(device)
            od = ckpt['all_seeds'][s]['odd']
            mo.W1.data[s] = od['net.0.weight'].t().to(device)
            mo.b1.data[s, 0] = od['net.0.bias'].to(device)
            mo.W2.data[s] = od['net.2.weight'].t().to(device)
            mo.b2.data[s, 0] = od['net.2.bias'].to(device)
    me.eval()
    mo.eval()
    return me, mo, N, hidden, input_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi-ckpt', required=True)
    ap.add_argument('--num-games', type=int, default=500)
    ap.add_argument('--k-min', type=int, default=5)
    ap.add_argument('--k-max', type=int, default=53)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=1024)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading {args.multi_ckpt}")
    me, mo, N, hidden, input_dim = load_vectorized_from_multi(
        args.multi_ckpt, device)
    print(f"  N={N} seeds, H={hidden}, input_dim={input_dim}")

    # Pattern -> cell mapping for prob_or aggregation
    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    games = load_val_games(args.data_dir, args.num_data_files)
    games = games[:args.num_games]
    print(f"Building test set from {len(games)} games × "
          f"{args.k_max - args.k_min + 1} positions/game")

    # Buffer all (feats, k, legal_set) — vectorize legal_mask afterwards
    feats_list, ks_list, legal_list = [], [], []
    for game in games:
        for k in range(args.k_min, args.k_max + 1):
            legal = legal_cells_60(game, k)
            if legal is None or not legal:
                continue
            feats_list.append(played_even_features(game[:k]))
            ks_list.append(k)
            legal_list.append(legal)
    n_total = len(feats_list)
    print(f"  {n_total} valid positions")

    # Dense legal mask (n_total, 60) for fast lookup later
    legal_mask = np.zeros((n_total, 60), dtype=bool)
    for i, legal in enumerate(legal_list):
        for c in legal:
            legal_mask[i, c] = True
    del legal_list

    # Predictions (N, n_total)
    preds = np.zeros((N, n_total), dtype=np.int32)

    print(f"Running batched forward pass...")
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_total, args.batch_size):
            end = min(i + args.batch_size, n_total)
            x = torch.stack(feats_list[i:end]).to(device)
            ks = torch.tensor(ks_list[i:end], device=device)
            use_me = (ks % 2 == 1)
            use_mo = ~use_me
            B = end - i
            logits = torch.zeros(N, B, 960, device=device)
            if use_me.any():
                logits[:, use_me] = me(x[use_me])
            if use_mo.any():
                logits[:, use_mo] = mo(x[use_mo])
            log1m = -F.softplus(logits)                          # (N, B, 960)
            gathered = log1m[:, :, idx]                          # (N, B, 60, K)
            gathered = gathered.masked_fill(~mask[None, None], 0.0)
            cell_scores = -gathered.sum(dim=-1)                  # (N, B, 60)
            preds[:, i:end] = cell_scores.argmax(dim=-1).cpu().numpy()
            if (i // args.batch_size) % 10 == 0:
                print(f"  {end}/{n_total}  ({int(time.time()-t0)}s)",
                      flush=True)

    # Vectorized legality check
    print(f"Computing legality stats...")
    # predicted_legal[s, b] = legal_mask[b, preds[s, b]]
    row_idx = np.arange(n_total)
    predicted_legal = legal_mask[row_idx[None, :], preds]        # (N, n_total)

    # Per-seed individual accuracy
    correct_per_seed = predicted_legal.sum(axis=1)               # (N,)
    individual_acc = correct_per_seed / n_total

    # Per-position: how many seeds got it right
    correct_count_per_position = predicted_legal.sum(axis=0)     # (n_total,)

    print()
    print(f"=== {N}-seed ensemble agreement on {n_total} test positions ===")
    print()
    print(f"Individual seed top-1 legal accuracy:")
    print(f"  mean: {individual_acc.mean():.4f}")
    print(f"  std:  {individual_acc.std():.4f}")
    print(f"  min:  {individual_acc.min():.4f}")
    print(f"  max:  {individual_acc.max():.4f}")
    print()

    n_all_wrong = int((correct_count_per_position == 0).sum())
    n_all_correct = int((correct_count_per_position == N).sum())
    pct_all_wrong = n_all_wrong / n_total * 100
    pct_all_correct = n_all_correct / n_total * 100
    print(f"All {N} seeds CORRECT (full agreement, legal):   "
          f"{n_all_correct:,} positions  ({pct_all_correct:.2f}%)")
    print(f"All {N} seeds WRONG   (full agreement, illegal): "
          f"{n_all_wrong:,} positions  ({pct_all_wrong:.4f}%)")
    print()

    # Distribution of agreement counts
    print(f"Distribution of '# of {N} seeds correct' per position:")
    bins = [0, 1, 5, 10, 20, 30, 40, N, N + 1]
    bin_labels = [f"{lo}-{hi-1}" if hi - 1 > lo else f"={lo}"
                  for lo, hi in zip(bins[:-1], bins[1:])]
    counts, _ = np.histogram(correct_count_per_position, bins=bins)
    for lab, c in zip(bin_labels, counts):
        print(f"  {lab:>8}: {c:>7,} positions  ({c/n_total*100:.2f}%)")
    print()

    # Asymptotic ceiling implications
    print(f"Asymptotic order-blind ceiling (estimated from this {N}-seed data):")
    print(f"  P(all {N} wrong) = {pct_all_wrong/100:.6f}")
    print(f"  Implied ceiling = {(1 - pct_all_wrong/100)*100:.4f}%")
    print(f"  (This estimate tightens as N grows; with N={N} it's an upper bound)")


if __name__ == '__main__':
    main()
