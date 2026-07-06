"""Per-seed top-1 legal accuracy for a multi-seed MLP checkpoint.

Loads a multi-seed checkpoint (vectorized N-seed model), runs each seed
independently on the val-games test set, and reports per-seed top-1
legal accuracy.  No ensemble aggregation.

Usage:
    python eval_multi_seed_per_seed.py \\
        --multi-ckpt experiments/.../multi_seed_N100_H512_playedeven.pt \\
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
from eval_multi_seed_ensemble import (
    load_vectorized_from_multi, legal_cells_60,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi-ckpt', required=True)
    ap.add_argument('--num-games', type=int, default=500)
    ap.add_argument('--k-min', type=int, default=5)
    ap.add_argument('--k-max', type=int, default=53)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--output-csv', default=None,
                    help='If set, save per-seed (seed_id, top1_acc) rows as CSV.')
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

    print(f"Building test set from {args.num_games} games...")
    games = load_val_games(args.data_dir, args.num_data_files)[:args.num_games]
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
    print(f"  {n_total} positions")

    legal_mask = np.zeros((n_total, 60), dtype=bool)
    for i, lg in enumerate(legal_list):
        for c in lg:
            legal_mask[i, c] = True

    correct_per_seed = np.zeros(N, dtype=np.int64)

    t0 = time.time()
    with torch.no_grad():
        for bstart in range(0, n_total, args.batch_size):
            bend = min(bstart + args.batch_size, n_total)
            x = torch.stack(feats_list[bstart:bend]).to(device)
            ks = torch.tensor(ks_list[bstart:bend], device=device)
            use_me = (ks % 2 == 1); use_mo = ~use_me
            B = bend - bstart
            logits = torch.zeros(N, B, 960, device=device)
            if use_me.any():
                logits[:, use_me] = me(x[use_me])
            if use_mo.any():
                logits[:, use_mo] = mo(x[use_mo])
            log1m = -F.softplus(logits)
            gathered = log1m[:, :, idx]
            gathered = gathered.masked_fill(~mask[None, None], 0.0)
            cell_scores = -gathered.sum(dim=-1)                    # (N, B, 60)
            top1 = cell_scores.argmax(dim=-1)                        # (N, B)
            legal_batch = torch.from_numpy(
                legal_mask[bstart:bend]).to(device)                  # (B, 60)
            hit = legal_batch.gather(1, top1.t()).t()                # (N, B)
            correct_per_seed += hit.sum(dim=1).cpu().numpy()
            if (bstart // args.batch_size) % 10 == 0:
                print(f"  {bend}/{n_total}  ({int(time.time()-t0)}s)",
                      flush=True)

    acc_per_seed = correct_per_seed / n_total

    print()
    print(f"=== Per-seed top-1 legality ({n_total:,} positions) ===")
    print(f"  {'seed':>4}  {'top-1 acc':>10}")
    print("  " + "-" * 20)
    for i in range(N):
        print(f"  {i:>4}  {acc_per_seed[i]:>10.4f}")

    print()
    print(f"Best  seed {int(acc_per_seed.argmax()):>3}  "
          f"acc={acc_per_seed.max():.4f}")
    print(f"Worst seed {int(acc_per_seed.argmin()):>3}  "
          f"acc={acc_per_seed.min():.4f}")
    print(f"Mean  acc = {acc_per_seed.mean():.4f}  "
          f"(std={acc_per_seed.std():.4f})")

    if args.output_csv:
        with open(args.output_csv, 'w') as f:
            f.write('seed,top1_acc\n')
            for i in range(N):
                f.write(f"{i},{acc_per_seed[i]:.6f}\n")
        print(f"\nSaved CSV: {args.output_csv}")


if __name__ == '__main__':
    main()
