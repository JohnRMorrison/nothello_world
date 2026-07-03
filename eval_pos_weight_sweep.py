"""Evaluate a pos_weight sweep checkpoint: pair per-seed top-1 legality with
each seed's training-time pos_weight, produced by --pos-weight-linspace.

Prints an ordered table and a summary of the best pos_weight range.
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
                    help='If set, also save (pos_weight, top1_acc) rows as CSV.')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading {args.multi_ckpt}")

    # Read the pos_weight metadata
    raw = torch.load(args.multi_ckpt, map_location='cpu')
    pw_values = raw.get('pos_weight_values')
    pw_linspace = raw.get('pos_weight_linspace')
    if pw_values is None:
        raise SystemExit(
            "This checkpoint has no pos_weight_values metadata. "
            "Only checkpoints written by --pos-weight-linspace can be evaluated "
            "with this script.")
    pw_values = np.array(pw_values, dtype=np.float32)
    print(f"  pos_weight sweep: linspace{tuple(pw_linspace)} across {len(pw_values)} seeds")
    print(f"  pw range: [{pw_values.min():.3f}, {pw_values.max():.3f}]")

    me, mo, N, hidden, _ = load_vectorized_from_multi(args.multi_ckpt, device)
    assert N == len(pw_values), \
        f"N seeds ({N}) does not match pos_weight_values length ({len(pw_values)})"

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

    # Per-seed top-1 correct counts across all positions
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
            cell_scores = -gathered.sum(dim=-1)                # (N, B, 60)
            top1 = cell_scores.argmax(dim=-1)                    # (N, B)
            legal_batch = torch.from_numpy(
                legal_mask[bstart:bend]).to(device)              # (B, 60)
            # per-seed: is top-1 legal?
            hit = legal_batch.gather(1, top1.t()).t()            # (N, B)
            correct_per_seed += hit.sum(dim=1).cpu().numpy()
            if (bstart // args.batch_size) % 10 == 0:
                print(f"  {bend}/{n_total}  ({int(time.time()-t0)}s)",
                      flush=True)

    acc_per_seed = correct_per_seed / n_total

    # Print ordered table
    print()
    print(f"=== Per-seed top-1 legality vs pos_weight ({n_total:,} positions) ===")
    print(f"  {'seed':>4}  {'pos_weight':>10}  {'top-1 acc':>10}")
    print("  " + "-" * 30)
    for i in range(N):
        print(f"  {i:>4}  {pw_values[i]:>10.3f}  {acc_per_seed[i]:>10.4f}")

    # Summary
    best_idx = int(acc_per_seed.argmax())
    worst_idx = int(acc_per_seed.argmin())
    print()
    print(f"Best  seed {best_idx:>3}  pos_weight={pw_values[best_idx]:>7.3f}  "
          f"acc={acc_per_seed[best_idx]:.4f}")
    print(f"Worst seed {worst_idx:>3}  pos_weight={pw_values[worst_idx]:>7.3f}  "
          f"acc={acc_per_seed[worst_idx]:.4f}")

    # Bin the accuracy by pos_weight decile
    print()
    print(f"Mean top-1 accuracy by pos_weight decile:")
    order = np.argsort(pw_values)
    for d in range(10):
        lo = int(N * d / 10)
        hi = int(N * (d + 1) / 10)
        dec_seeds = order[lo:hi]
        dec_pw = pw_values[dec_seeds]
        dec_acc = acc_per_seed[dec_seeds]
        print(f"  decile {d+1}  pw [{dec_pw.min():.2f}, {dec_pw.max():.2f}]  "
              f"mean acc {dec_acc.mean():.4f}")

    if args.output_csv:
        with open(args.output_csv, 'w') as f:
            f.write('seed,pos_weight,top1_acc\n')
            for i in range(N):
                f.write(f"{i},{pw_values[i]:.6f},{acc_per_seed[i]:.6f}\n")
        print(f"\nSaved CSV: {args.output_csv}")


if __name__ == '__main__':
    main()
