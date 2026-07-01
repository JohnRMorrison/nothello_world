"""Aggregation evaluation for N-seed multi-seed MLP ensemble.

Reports top-K legality (K = 1, 3, 5, 10) for four aggregators across
N = 1, 5, 10, 20, 50, 100 (or however many seeds are in the checkpoint):

  - Sum log_prob_or : sum(cell_scores) across models (current default)
  - Mean prob_or    : mean(1 - exp(-cell_scores)) across models
  - Majority vote   : mode(argmax cell) across models (ties broken by
                       sum log_prob_or on tied cells)
  - Sum raw logits  : sum(960-d pattern logits) across models, then
                       apply prob_or aggregator within each cell

Metric: top-K legality = mean fraction of top-K predicted cells that are
legal at the position.

Output: printed table (no PNG).

Usage:
    python eval_multi_seed_aggregation.py \\
        --multi-ckpt experiments/.../multi_seed_N100_H512_playedeven.pt \\
        --num-games 500
"""
import argparse
import os
import sys
import time
from collections import Counter

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


N_SUBSETS = [1, 5, 10, 20, 50, 100]
TOP_KS = [1, 3, 5, 10]
AGG_NAMES = [
    "sum_log_prob_or",
    "mean_prob_or",
    "majority_vote",
    "sum_raw_logits",
]


def topk_legality(topk_indices, legal_mask):
    """topk_indices: (n_pos, K).  legal_mask: (n_pos, 60).
    Returns mean fraction of top-K cells that are legal, per position.
    """
    n_pos, K = topk_indices.shape
    row_idx = np.arange(n_pos)[:, None]
    hits = legal_mask[row_idx, topk_indices]  # (n_pos, K) bool
    return hits.mean()


def aggregate_and_topk(pattern_logits, cell_scores, argmax_preds,
                       n_seeds, agg_name, ks, seed_mask):
    """Apply the aggregator to the first `n_seeds` models (via seed_mask)
    and return a dict K -> mean top-K legality preparation (indices only).

    Returns top-max(ks) predicted cell indices per position: (n_pos, max_k).
    """
    max_k = max(ks)

    if agg_name == "sum_log_prob_or":
        # cell_scores: (N, n_pos, 60)
        agg = cell_scores[seed_mask].sum(axis=0)   # (n_pos, 60)
        topk = np.argsort(-agg, axis=1)[:, :max_k]

    elif agg_name == "mean_prob_or":
        prob_or = 1.0 - np.exp(-np.clip(cell_scores[seed_mask], 0, None))
        agg = prob_or.mean(axis=0)                  # (n_pos, 60)
        topk = np.argsort(-agg, axis=1)[:, :max_k]

    elif agg_name == "majority_vote":
        # For each position, count votes among the n_seeds models
        preds = argmax_preds[seed_mask]             # (n_seeds, n_pos)
        n_pos = preds.shape[1]
        # Vectorized voting via bincount per column
        # We only need top-max_k cells by vote count.
        # tie-breaker: sum_log_prob_or
        agg = cell_scores[seed_mask].sum(axis=0)    # (n_pos, 60)
        # Add vote count as a large weight
        vote_counts = np.zeros((n_pos, 60), dtype=np.int64)
        for c in range(60):
            vote_counts[:, c] = (preds == c).sum(axis=0)
        # Score: primary = vote count (large), secondary = agg
        combined = vote_counts.astype(np.float64) * 1e6 + agg
        topk = np.argsort(-combined, axis=1)[:, :max_k]

    elif agg_name == "sum_raw_logits":
        # pattern_logits: (N, n_pos, 960)
        agg_logits = pattern_logits[seed_mask].sum(axis=0)  # (n_pos, 960)
        # Apply prob_or aggregator within cells
        log1m = -np.log1p(np.exp(agg_logits))       # log(1 - sigmoid(x))
        # Aggregate to cells via _get_cell_pat_index-style scatter
        # Using precomputed AGG_IDX and AGG_MASK
        n_pos = agg_logits.shape[0]
        gathered = log1m[:, AGG_IDX]                # (n_pos, 60, K_max_p)
        gathered = np.where(AGG_MASK[None, :, :], gathered, 0.0)
        cell_scores_agg = -gathered.sum(axis=-1)    # (n_pos, 60)
        topk = np.argsort(-cell_scores_agg, axis=1)[:, :max_k]

    else:
        raise ValueError(f"Unknown aggregator: {agg_name}")

    return topk


# Global pattern-to-cell mapping (filled in main)
AGG_IDX = None
AGG_MASK = None


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
    print(f"  N={N} seeds, H={hidden}")

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx_t, mask_t = _get_cell_pat_index(pattern_to_cell, 60)

    # Numpy versions for sum_raw_logits aggregator
    global AGG_IDX, AGG_MASK
    AGG_IDX = idx_t.cpu().numpy()   # (60, K_max_p)
    AGG_MASK = mask_t.cpu().numpy() # (60, K_max_p) bool

    # Filter N_SUBSETS to values <= actual N
    n_subsets = [n for n in N_SUBSETS if n <= N]
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

    # Forward pass through all N MLPs; collect pattern_logits, cell_scores,
    # argmax_preds for each position and model.
    print(f"Batched forward pass through all {N} MLPs...")
    pattern_logits = np.zeros((N, n_total, 960), dtype=np.float16)  # ~7GB @ N=100, 24K
    cell_scores    = np.zeros((N, n_total, 60), dtype=np.float32)
    argmax_preds   = np.zeros((N, n_total), dtype=np.int32)

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
            log1m = -F.softplus(logits)                       # (N, B, 960)
            gathered = log1m[:, :, idx_t]                     # (N, B, 60, K)
            gathered = gathered.masked_fill(~mask_t[None, None], 0.0)
            scores = -gathered.sum(dim=-1)                    # (N, B, 60)

            pattern_logits[:, i:end, :] = logits.cpu().numpy().astype(np.float16)
            cell_scores[:, i:end, :]    = scores.cpu().numpy()
            argmax_preds[:, i:end]      = scores.argmax(dim=-1).cpu().numpy()

            if (i // args.batch_size) % 10 == 0:
                print(f"  {end}/{n_total}  ({int(time.time()-t0)}s)",
                      flush=True)

    # Convert pattern_logits back to float32 for numpy math
    print("Running aggregators × N subsets...")
    t0 = time.time()

    # For each (aggregator, N), compute top-K legality
    results = {}   # (agg_name, n_seeds, K) -> mean top-K legality
    for n_seeds in n_subsets:
        seed_mask = np.zeros(N, dtype=bool)
        seed_mask[:n_seeds] = True
        for agg_name in AGG_NAMES:
            if agg_name == "sum_raw_logits":
                # Need float32 for accurate logsumexp; convert view
                pattern_logits_f32 = pattern_logits.astype(np.float32)
                topk = aggregate_and_topk_float(
                    pattern_logits_f32, cell_scores, argmax_preds,
                    n_seeds, agg_name, TOP_KS, seed_mask,
                )
            else:
                topk = aggregate_and_topk(
                    pattern_logits, cell_scores, argmax_preds,
                    n_seeds, agg_name, TOP_KS, seed_mask,
                )
            for K in TOP_KS:
                results[(agg_name, n_seeds, K)] = topk_legality(
                    topk[:, :K], legal_mask,
                )
            print(f"  {agg_name} @ N={n_seeds}  "
                  f"({int(time.time()-t0)}s)", flush=True)

    # Print table
    print()
    print(f"=== Aggregation results ({n_total:,} test positions) ===")
    for K in TOP_KS:
        print()
        print(f"Top-{K} legality:")
        header = f"  {'Aggregator':<20}" + "".join(f"  N={n:>4}" for n in n_subsets)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for agg_name in AGG_NAMES:
            row = f"  {agg_name:<20}"
            for n_seeds in n_subsets:
                v = results[(agg_name, n_seeds, K)]
                row += f"  {v:>6.4f}"
            print(row)


def aggregate_and_topk_float(pattern_logits_f32, cell_scores, argmax_preds,
                              n_seeds, agg_name, ks, seed_mask):
    """Same as aggregate_and_topk but for float32 pattern_logits."""
    max_k = max(ks)
    # sum raw logits: sum across models, then prob_or
    agg_logits = pattern_logits_f32[seed_mask].sum(axis=0)   # (n_pos, 960)
    # log(1 - sigmoid(x)) = -softplus(x)
    log1m = -np.log1p(np.exp(agg_logits))
    n_pos = agg_logits.shape[0]
    gathered = log1m[:, AGG_IDX]                             # (n_pos, 60, K_max_p)
    gathered = np.where(AGG_MASK[None, :, :], gathered, 0.0)
    cell_scores_agg = -gathered.sum(axis=-1)                 # (n_pos, 60)
    topk = np.argsort(-cell_scores_agg, axis=1)[:, :max_k]
    return topk


if __name__ == '__main__':
    main()
