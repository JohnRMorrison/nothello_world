"""Forfeit diagnostic for the multi-seed ensemble.

For each game in the test set, detect whether a forfeit occurred and at
which step. Then partition test positions into:
  - "clean" positions: feature encoding is correctly labeled
                       (no forfeit has happened in this game prior to k)
  - "forfeit-corrupted" positions: at least one forfeit happened earlier in
                                   the game, so the parity feature labels
                                   subsequent cells with the wrong mover

Reports:
  - P(all N wrong | clean)
  - P(all N wrong | forfeit-corrupted)
  - Implied "true" ceiling using only clean positions

Usage:
    python eval_multi_seed_forfeit_diag.py \\
        --multi-ckpt experiments/.../multi_seed_N50_H4096_playedeven.pt \\
        --num-games 10000
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
from eval_multi_seed_ensemble import (
    load_vectorized_from_multi, legal_cells_60,
)


def first_forfeit_step(game, max_k):
    """Return the first step k at which a forfeit had already occurred in
    this game, or None if no forfeit happened in [0, max_k).

    Detection: under strict alternation, board.next_hand_color before step
    t should equal (+1 if t is even else -1).  If they diverge, a forfeit
    happened at some earlier step.
    """
    board = OthelloBoardState()
    expected = 1  # OthelloBoardState starts with black = +1
    for t in range(min(max_k, len(game))):
        if board.next_hand_color != expected:
            return t
        try:
            board.umpire(game[t])
        except Exception:
            return None
        expected *= -1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi-ckpt', required=True)
    ap.add_argument('--num-games', type=int, default=10_000)
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

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    print(f"Loading {args.num_games} games...")
    games = load_val_games(args.data_dir, args.num_data_files)
    games = games[:args.num_games]

    # Detect forfeit per game
    print("Detecting forfeits per game...")
    t0 = time.time()
    forfeit_step = np.full(len(games), -1, dtype=np.int32)  # -1 = no forfeit
    for gi, game in enumerate(games):
        step = first_forfeit_step(game, args.k_max + 2)
        if step is not None:
            forfeit_step[gi] = step
        if (gi + 1) % 2000 == 0:
            print(f"  {gi+1}/{len(games)} ({int(time.time()-t0)}s)",
                  flush=True)
    n_games_with_forfeit = int((forfeit_step >= 0).sum())
    print(f"Games with at least one forfeit (in steps 0..{args.k_max+1}): "
          f"{n_games_with_forfeit} "
          f"({n_games_with_forfeit/len(games)*100:.2f}%)")

    # Build test set
    print(f"Building test set (k in [{args.k_min}, {args.k_max}])...")
    feats_list, ks_list, legal_list = [], [], []
    pos_corrupted = []  # bool: was there a forfeit at some t < k in this game?
    for gi, game in enumerate(games):
        ff_step = forfeit_step[gi]
        for k in range(args.k_min, args.k_max + 1):
            legal = legal_cells_60(game, k)
            if legal is None or not legal:
                continue
            feats_list.append(played_even_features(game[:k]))
            ks_list.append(k)
            legal_list.append(legal)
            pos_corrupted.append(0 <= ff_step <= k)
    n_total = len(feats_list)
    pos_corrupted = np.array(pos_corrupted, dtype=bool)
    print(f"  {n_total} valid positions")
    print(f"  Clean (no prior forfeit):       "
          f"{(~pos_corrupted).sum()} ({(~pos_corrupted).mean()*100:.2f}%)")
    print(f"  Forfeit-corrupted (prior forfeit): "
          f"{pos_corrupted.sum()} ({pos_corrupted.mean()*100:.2f}%)")

    # Dense legal mask
    legal_mask = np.zeros((n_total, 60), dtype=bool)
    for i, legal in enumerate(legal_list):
        for c in legal:
            legal_mask[i, c] = True
    del legal_list

    # Predictions (N, n_total)
    preds = np.zeros((N, n_total), dtype=np.int32)
    print("Batched forward pass through all 50 MLPs...")
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
            log1m = -F.softplus(logits)
            gathered = log1m[:, :, idx]
            gathered = gathered.masked_fill(~mask[None, None], 0.0)
            cell_scores = -gathered.sum(dim=-1)
            preds[:, i:end] = cell_scores.argmax(dim=-1).cpu().numpy()
            if (i // args.batch_size) % 50 == 0:
                print(f"  {end}/{n_total} ({int(time.time()-t0)}s)", flush=True)

    # Vectorized legality
    row_idx = np.arange(n_total)
    predicted_legal = legal_mask[row_idx[None, :], preds]  # (N, n_total)
    correct_count = predicted_legal.sum(axis=0)
    is_all_wrong = (correct_count == 0)

    n_all_wrong = int(is_all_wrong.sum())
    n_clean = int((~pos_corrupted).sum())
    n_corrupted = int(pos_corrupted.sum())
    n_aw_clean = int((is_all_wrong & ~pos_corrupted).sum())
    n_aw_corrupted = int((is_all_wrong & pos_corrupted).sum())

    print()
    print(f"=== Forfeit attribution of all-{N}-wrong positions ===")
    print(f"Total positions:              {n_total:,}")
    print(f"All-{N}-wrong positions:        {n_all_wrong:,}  "
          f"({n_all_wrong/n_total*100:.4f}%)")
    print()
    print(f"Of the {n_all_wrong} all-wrong positions:")
    print(f"  From forfeit-corrupted: {n_aw_corrupted:,}  "
          f"({n_aw_corrupted/max(1,n_all_wrong)*100:.1f}%)")
    print(f"  From clean positions:   {n_aw_clean:,}  "
          f"({n_aw_clean/max(1,n_all_wrong)*100:.1f}%)")
    print()
    print(f"Conditional all-wrong rates:")
    if n_clean > 0:
        rate_clean = n_aw_clean / n_clean
        print(f"  P(all {N} wrong | clean)              = {rate_clean*100:.4f}%")
    if n_corrupted > 0:
        rate_corr = n_aw_corrupted / n_corrupted
        print(f"  P(all {N} wrong | forfeit-corrupted)  = {rate_corr*100:.4f}%")
    if n_clean > 0 and n_corrupted > 0:
        enrich = rate_corr / max(rate_clean, 1e-12)
        print(f"  Enrichment in forfeit positions     = {enrich:.2f}x")
    print()
    if n_clean > 0:
        ceiling_clean = (1 - rate_clean) * 100
        print(f"Implied asymptotic ceiling on CLEAN positions only: "
              f"{ceiling_clean:.4f}%")
        ceiling_overall = (1 - n_all_wrong / n_total) * 100
        print(f"Same ceiling on ALL positions (mixed clean + corrupted): "
              f"{ceiling_overall:.4f}%")


if __name__ == '__main__':
    main()
