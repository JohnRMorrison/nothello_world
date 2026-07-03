"""Pairwise seed agreement across two multi-seed ensembles.

For each position, each seed casts a top-1 vote for a cell.  We measure
how often pairs of seeds agree on the same top-1:

  Intra-A:  average over pairs (i, i') both in A
  Intra-B:  average over pairs (j, j') both in B
  Inter-AB: average over pairs (i, j) with i in A and j in B

If Inter-AB ~= Intra-A ~= Intra-B, then disjoint training data produces
no more seed diversity than random init.  If Inter-AB is meaningfully
LOWER than intra, then data diversity is real.

Also reports the equivalent "correct-agreement" metric: among positions
where both seeds got it right, are they picking the SAME legal cell (which
would matter if the position has multiple legal moves).

Usage:
    python pairwise_seed_agreement.py \\
        --ckpt-a A.pt  --ckpt-b B.pt  --num-games 500
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


@torch.no_grad()
def per_seed_top1(feats_batch, ks_batch, me, mo, idx, mask, N, device):
    """Return per-seed argmax cell (N, B) using eval convention."""
    x = feats_batch.to(device)
    ks_t = ks_batch.to(device)
    use_me = (ks_t % 2 == 1); use_mo = ~use_me
    B = x.shape[0]
    logits = torch.zeros(N, B, 960, device=device)
    if use_me.any():
        logits[:, use_me] = me(x[use_me])
    if use_mo.any():
        logits[:, use_mo] = mo(x[use_mo])
    log1m = -F.softplus(logits)
    gathered = log1m[:, :, idx]
    gathered = gathered.masked_fill(~mask[None, None], 0.0)
    cell_scores = -gathered.sum(dim=-1)                       # (N, B, 60)
    return cell_scores.argmax(dim=-1)                          # (N, B)


def agreement_from_votes(votes_1, votes_2, N1, N2, same_set):
    """Given (N1, B) and (N2, B) argmax votes per position, return the mean
    fraction of (seed_i, seed_j) pairs that agree.

    If same_set is True, exclude i == j pairs (i.e., diagonal in a within-set
    comparison) so we don't count seeds agreeing with themselves.
    Note: same_set requires N1 == N2 and votes_1 is votes_2.
    """
    B = votes_1.shape[1]
    # For each position, count cell -> #seeds in set 1 voting for it
    counts_1 = torch.zeros(B, 60, device=votes_1.device)
    counts_1.scatter_add_(
        1, votes_1.t(),
        torch.ones_like(votes_1.t(), dtype=torch.float32))
    counts_2 = counts_1 if (same_set and votes_1 is votes_2) else torch.zeros(
        B, 60, device=votes_2.device)
    if not (same_set and votes_1 is votes_2):
        counts_2.scatter_add_(
            1, votes_2.t(),
            torch.ones_like(votes_2.t(), dtype=torch.float32))

    if same_set:
        # Agreement fraction over unordered pairs (i, i'), i != i'
        # sum_c C_c * (C_c - 1) / (N * (N - 1))
        # equivalent to: sum_c C_c^2 - N, divided by N * (N-1)
        n_pairs = N1 * (N1 - 1)
        agree = (counts_1 * counts_1).sum(dim=1) - N1   # (B,)
        agree = agree / n_pairs
    else:
        # Agreement over cross pairs (i in A, j in B): N1 * N2 pairs
        agree = (counts_1 * counts_2).sum(dim=1) / (N1 * N2)
    return agree.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-a', required=True)
    ap.add_argument('--ckpt-b', required=True)
    ap.add_argument('--num-games', type=int, default=500)
    ap.add_argument('--k-min', type=int, default=5)
    ap.add_argument('--k-max', type=int, default=53)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=512)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading A: {args.ckpt_a}")
    me_a, mo_a, N_a, h_a, _ = load_vectorized_from_multi(args.ckpt_a, device)
    print(f"  N_a={N_a}, H={h_a}")
    print(f"Loading B: {args.ckpt_b}")
    me_b, mo_b, N_b, h_b, _ = load_vectorized_from_multi(args.ckpt_b, device)
    print(f"  N_b={N_b}, H={h_b}")

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

    # Compute per-seed argmax for all positions, batched
    all_votes_a = torch.zeros(N_a, n_total, dtype=torch.long, device=device)
    all_votes_b = torch.zeros(N_b, n_total, dtype=torch.long, device=device)

    t0 = time.time()
    for bstart in range(0, n_total, args.batch_size):
        bend = min(bstart + args.batch_size, n_total)
        feats_batch = torch.stack(feats_list[bstart:bend])
        ks_batch = torch.tensor(ks_list[bstart:bend])
        va = per_seed_top1(feats_batch, ks_batch, me_a, mo_a, idx, mask,
                             N_a, device)
        vb = per_seed_top1(feats_batch, ks_batch, me_b, mo_b, idx, mask,
                             N_b, device)
        all_votes_a[:, bstart:bend] = va
        all_votes_b[:, bstart:bend] = vb
        if (bstart // args.batch_size) % 10 == 0:
            print(f"  {bend}/{n_total}  ({int(time.time()-t0)}s)", flush=True)

    # Compute agreements
    intra_a  = agreement_from_votes(all_votes_a, all_votes_a, N_a, N_a,
                                       same_set=True)
    intra_b  = agreement_from_votes(all_votes_b, all_votes_b, N_b, N_b,
                                       same_set=True)
    inter_ab = agreement_from_votes(all_votes_a, all_votes_b, N_a, N_b,
                                       same_set=False)

    print()
    print(f"=== Pairwise top-1 seed agreement ({n_total:,} positions) ===")
    print(f"  Intra-A  (within {N_a} seeds of A):    {intra_a:.4f}")
    print(f"  Intra-B  (within {N_b} seeds of B):    {intra_b:.4f}")
    print(f"  Inter-AB (across ensembles):           {inter_ab:.4f}")
    print()
    print(f"Interpretation:")
    print(f"  Baseline = intra agreement (init-only diversity within one ckpt)")
    print(f"  If inter ~= intra: disjoint training data adds NO diversity")
    print(f"  If inter << intra: data diversity IS producing different seeds")
    delta = intra_a - inter_ab
    print(f"  Δ (intra_A - inter_AB) = {delta:.4f}  "
          f"({100*delta:.2f}pp)")


if __name__ == '__main__':
    main()
