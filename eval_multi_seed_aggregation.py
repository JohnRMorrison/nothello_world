"""Aggregation evaluation for N-seed multi-seed MLP ensemble.

Reports top-K legality (K = 1, 3, 5, 10) for four aggregators across
N = 1, 5, 10, 20, 50, 100 (or however many seeds are in the checkpoint):

  - sum_log_prob_or : sum of per-model cell_scores across models
  - mean_prob_or    : mean of (1 - exp(-cell_scores)) across models
  - majority_vote   : per-cell vote count (tie-break: sum_log_prob_or)
  - sum_raw_logits  : sum(pattern_logits) across models, then prob_or per cell

Metric: top-K legality = mean fraction of top-K predicted cells that are
legal at the position.

Memory-efficient: aggregations computed per batch on GPU without
materializing per-model pattern_logits for the whole test set.

Usage:
    python eval_multi_seed_aggregation.py \\
        --multi-ckpt experiments/.../multi_seed_N100_H512_playedeven.pt \\
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
from eval_multi_seed_ensemble import (
    load_vectorized_from_multi, legal_cells_60,
)


N_SUBSETS_DEFAULT = [1, 5, 10, 20, 50, 100]
TOP_KS = [1, 3, 5, 10]
AGG_NAMES = [
    "sum_log_prob_or",
    "mean_prob_or",
    "majority_vote",
    "sum_raw_logits",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi-ckpt', required=True)
    ap.add_argument('--num-games', type=int, default=500)
    ap.add_argument('--k-min', type=int, default=5)
    ap.add_argument('--k-max', type=int, default=53)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=512)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading {args.multi_ckpt}")
    me, mo, N, hidden, input_dim = load_vectorized_from_multi(
        args.multi_ckpt, device)
    print(f"  N={N} seeds, H={hidden}")

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    n_subsets = [n for n in N_SUBSETS_DEFAULT if n <= N]
    if N not in n_subsets:
        n_subsets.append(N)
    print(f"N subset sizes: {n_subsets}")

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
    for i, legal in enumerate(legal_list):
        for c in legal:
            legal_mask[i, c] = True
    del legal_list

    # Running total of top-K hits per (aggregator, n_seeds, K).
    hits = {(a, n, K): 0
            for a in AGG_NAMES for n in n_subsets for K in TOP_KS}

    print("Batched forward + aggregation...")
    t0 = time.time()
    max_K = max(TOP_KS)

    with torch.no_grad():
        for bstart in range(0, n_total, args.batch_size):
            bend = min(bstart + args.batch_size, n_total)
            B = bend - bstart
            x = torch.stack(feats_list[bstart:bend]).to(device)
            ks = torch.tensor(ks_list[bstart:bend], device=device)
            use_me = (ks % 2 == 1)
            use_mo = ~use_me

            # Full logits from all N models
            logits = torch.zeros(N, B, 960, device=device)     # (N, B, 960)
            if use_me.any():
                logits[:, use_me] = me(x[use_me])
            if use_mo.any():
                logits[:, use_mo] = mo(x[use_mo])
            log1m = -F.softplus(logits)                          # (N, B, 960)
            gathered = log1m[:, :, idx]                          # (N, B, 60, Kp)
            gathered = gathered.masked_fill(~mask[None, None], 0.0)
            cell_scores = -gathered.sum(dim=-1)                  # (N, B, 60)
            prob_or = 1.0 - torch.exp(-cell_scores.clamp(min=0))  # (N, B, 60)
            per_model_argmax = cell_scores.argmax(dim=-1)         # (N, B)

            legal_batch = torch.from_numpy(
                legal_mask[bstart:bend]).to(device)              # (B, 60)

            for n_seeds in n_subsets:
                sub_scores  = cell_scores[:n_seeds]              # (n, B, 60)
                sub_prob    = prob_or[:n_seeds]
                sub_argmax  = per_model_argmax[:n_seeds]

                # Aggregator A: sum log_prob_or
                agg_a = sub_scores.sum(dim=0)                    # (B, 60)

                # Aggregator B: mean prob_or
                agg_b = sub_prob.mean(dim=0)                     # (B, 60)

                # Aggregator C: majority vote (tie-break: agg_a)
                # Vectorized vote count: scatter-add
                votes = torch.zeros(B, 60, device=device)
                votes.scatter_add_(
                    1, sub_argmax.t(),
                    torch.ones_like(sub_argmax.t(), dtype=torch.float32),
                )
                agg_c = votes * 1e6 + agg_a

                # Aggregator D: sum raw logits, then prob_or
                sub_logits = logits[:n_seeds]                    # (n, B, 960)
                agg_logits = sub_logits.sum(dim=0)               # (B, 960)
                log1m_d = -F.softplus(agg_logits)                # (B, 960)
                gathered_d = log1m_d[:, idx]                     # (B, 60, Kp)
                gathered_d = gathered_d.masked_fill(~mask[None], 0.0)
                agg_d = -gathered_d.sum(dim=-1)                  # (B, 60)

                # Top-K legality per aggregator
                for a_name, agg in [
                    ("sum_log_prob_or", agg_a),
                    ("mean_prob_or",    agg_b),
                    ("majority_vote",   agg_c),
                    ("sum_raw_logits",  agg_d),
                ]:
                    topk_idx = agg.topk(max_K, dim=1).indices    # (B, max_K)
                    # Gather legality at those indices
                    legal_at_topk = legal_batch.gather(
                        1, topk_idx.to(torch.long))              # (B, max_K)
                    for K in TOP_KS:
                        hits[(a_name, n_seeds, K)] += int(
                            legal_at_topk[:, :K].sum())

            if (bstart // args.batch_size) % 10 == 0:
                print(f"  {bend}/{n_total}  ({int(time.time()-t0)}s)",
                      flush=True)

    # Compute rates
    results = {}
    for (a, n, K), h in hits.items():
        results[(a, n, K)] = h / (n_total * K)

    # Print table
    print()
    print(f"=== Aggregation results ({n_total:,} test positions) ===")
    for K in TOP_KS:
        print()
        print(f"Top-{K} legality:")
        header = f"  {'Aggregator':<20}" + \
                 "".join(f"  N={n:>4}" for n in n_subsets)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for a_name in AGG_NAMES:
            row = f"  {a_name:<20}"
            for n_seeds in n_subsets:
                v = results[(a_name, n_seeds, K)]
                row += f"  {v:>6.4f}"
            print(row)


if __name__ == '__main__':
    main()
