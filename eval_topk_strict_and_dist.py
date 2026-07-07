"""Refined top-K legality + legal-move-count distribution.

For each K in {1, 3, 5, 10} report TWO metrics:

  achievability-aware  =  hits / min(K, n_legal)       [current metric]
                          across all positions with >=1 legal move

  strict               =  hits / K                     [new metric]
                          restricted to positions with n_legal >= K

Also reports:
  - The full distribution of n_legal values in the evaluated set
  - Fraction of positions where the current player has no legal moves
    (forfeit / must-pass positions), currently excluded from all
    metrics in eval_single_mlp_val_games.py

Usage:
    python eval_topk_strict_and_dist.py \\
        --ckpt experiments/.../pattern_simple_direct_H512_playedeven.pt \\
        --hidden 512 --num-games 100000 --num-data-files 3
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from train_pattern_simple import DirectMLP, _get_cell_pat_index
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from compare_v4_vs_mlp import (
    load_val_games, played_even_features, C64_TO_C60,
)
from eval_multi_seed_ensemble import legal_cells_60

KS = [1, 3, 5, 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--hidden', type=int, required=True)
    ap.add_argument('--num-games', type=int, default=100000)
    ap.add_argument('--k-min', type=int, default=5)
    ap.add_argument('--k-max', type=int, default=53)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=3)
    ap.add_argument('--file-start', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=512)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device)
    input_dim = ckpt.get('input_dim', 120)
    n_patterns = ckpt.get('n_patterns', 960)
    me = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    mo = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even'])
    mo.load_state_dict(ckpt['odd'])
    me.eval(); mo.eval()
    print(f"  H={args.hidden}, input_dim={input_dim}")

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    print(f"Loading games...")
    if args.file_start is None:
        games = load_val_games(args.data_dir, args.num_data_files)
    else:
        import pickle as _pickle
        all_files = sorted(os.listdir(args.data_dir))
        picked = all_files[args.file_start:
                            args.file_start + args.num_data_files]
        games = []
        for fname in picked:
            with open(os.path.join(args.data_dir, fname), 'rb') as f:
                batch = _pickle.load(f)
            if len(batch) >= 9e4:
                games.extend(batch)
    games = games[:args.num_games]
    print(f"  {len(games)} games")

    # Build test set — track n_legal for EVERY position, including forfeits
    print(f"Building test set (turns {args.k_min}..{args.k_max})...")
    feats_list, ks_list, legal_list = [], [], []
    n_forfeits = 0
    n_total_positions = 0
    for game in games:
        for k in range(args.k_min, args.k_max + 1):
            n_total_positions += 1
            legal = legal_cells_60(game, k)
            if legal is None or not legal:
                n_forfeits += 1
                continue
            feats_list.append(played_even_features(game[:k]))
            ks_list.append(k)
            legal_list.append(legal)
    n_scored = len(feats_list)
    print(f"  {n_total_positions:,} positions total")
    print(f"  {n_forfeits:,} forfeits (no legal moves) — "
          f"{n_forfeits/max(n_total_positions,1):.3%}")
    print(f"  {n_scored:,} scored positions")

    # Legal-move-count distribution
    n_legal_per_pos = np.array([len(l) for l in legal_list], dtype=np.int32)
    print()
    print("=== n_legal distribution across scored positions ===")
    hist_bins = list(range(1, 16)) + [20, 25, 30]
    total = len(n_legal_per_pos)
    for i, b in enumerate(hist_bins):
        if i + 1 < len(hist_bins):
            hi = hist_bins[i + 1]
            c = int(((n_legal_per_pos >= b) & (n_legal_per_pos < hi)).sum())
            label = f"[{b}, {hi})"
        else:
            c = int((n_legal_per_pos >= b).sum())
            label = f">= {b}"
        print(f"  {label:>10}  {c:>10,}  ({c/total:>7.3%})")
    print(f"  mean n_legal:   {n_legal_per_pos.mean():.2f}")
    print(f"  median n_legal: {int(np.median(n_legal_per_pos))}")
    print(f"  max n_legal:    {int(n_legal_per_pos.max())}")
    print()
    print("=== Fraction of scored positions with >= K legal moves ===")
    for K in KS:
        frac = (n_legal_per_pos >= K).mean()
        print(f"  n_legal >= {K:>2}:  {frac:.4f}  "
              f"({int((n_legal_per_pos >= K).sum()):,}/{total:,})")

    # MLP forward pass
    print()
    print(f"Running MLP forward pass...")
    legal_mask = np.zeros((n_scored, 60), dtype=bool)
    for i, lg in enumerate(legal_list):
        for c in lg:
            legal_mask[i, c] = True

    hits_achievability = {K: 0.0 for K in KS}
    hits_strict = {K: 0 for K in KS}
    counts_strict = {K: 0 for K in KS}
    max_K = max(KS)

    t0 = time.time()
    with torch.no_grad():
        for bstart in range(0, n_scored, args.batch_size):
            bend = min(bstart + args.batch_size, n_scored)
            x = torch.stack(feats_list[bstart:bend]).to(device)
            ks = torch.tensor(ks_list[bstart:bend], device=device)
            use_me = (ks % 2 == 1); use_mo = ~use_me
            B = bend - bstart
            logits = torch.zeros(B, 960, device=device)
            if use_me.any():
                logits[use_me] = me(x[use_me])
            if use_mo.any():
                logits[use_mo] = mo(x[use_mo])
            log1m = -F.softplus(logits)
            gathered = log1m[:, idx]
            gathered = gathered.masked_fill(~mask[None], 0.0)
            cell_scores = -gathered.sum(dim=-1)                        # (B, 60)

            legal_batch = torch.from_numpy(
                legal_mask[bstart:bend]).to(device)
            n_legal_batch = legal_batch.sum(dim=1).clamp(min=1).float()

            topk_idx = cell_scores.topk(max_K, dim=1).indices
            legal_at_topk = legal_batch.gather(
                1, topk_idx.to(torch.long))                             # (B, max_K)

            n_legal_np = n_legal_per_pos[bstart:bend]                   # (B,)

            for K in KS:
                got = legal_at_topk[:, :K].sum(dim=1).float()           # (B,)
                # (a) achievability-aware
                denom_a = torch.minimum(
                    torch.full_like(n_legal_batch, K), n_legal_batch)
                hits_achievability[K] += (got / denom_a).sum().item()
                # (b) strict: only positions with n_legal >= K
                strict_mask = (n_legal_np >= K)
                if strict_mask.any():
                    got_np = got.cpu().numpy()
                    hits_strict[K] += int(got_np[strict_mask].sum())
                    counts_strict[K] += int(strict_mask.sum())

            if (bstart // args.batch_size) % 20 == 0:
                print(f"  {bend}/{n_scored}  ({int(time.time()-t0)}s)",
                      flush=True)

    # Print results
    print()
    print(f"=== Legality metrics on {n_scored:,} scored positions ===")
    print(f"  {'K':<5}  "
          f"{'achievability-aware':>20}  "
          f"{'strict (n_legal>=K)':>24}")
    print("  " + "-" * 55)
    for K in KS:
        a = hits_achievability[K] / n_scored
        if counts_strict[K] > 0:
            s = hits_strict[K] / (counts_strict[K] * K)
            s_lbl = f"{s:.4f}  (n={counts_strict[K]:,})"
        else:
            s_lbl = "n/a (no positions)"
        print(f"  top-{K:<2}  {a:>20.4f}  {s_lbl:>24}")


if __name__ == '__main__':
    main()
