"""Compare two independently-trained multi-seed ensembles on top-K legality
and analyze their agreement/disagreement pattern.

Two H=512 ensembles trained on DISJOINT chunks (chunks 0-9 vs chunks 10-19)
should, in principle, produce different-flavored errors if the training
data drives real diversity.  This measures:

  1. Per-ensemble top-K legality (sum_log_prob_or aggregation).
  2. Overlap matrix at top-1: P(both correct), P(only A), P(only B), P(both wrong).
  3. Combined ensemble (200 seeds total) top-K legality — if the ensembles
     have complementary errors, the combination should beat either alone.

Usage:
    python compare_ensembles_overlap.py \\
        --ckpt-a experiments/.../multi_seed_N100_H512_playedeven_chunks0-9.pt \\
        --ckpt-b experiments/.../multi_seed_N100_H512_playedeven_chunks10-19_seed1.pt \\
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
from eval_multi_seed_ensemble import (
    load_vectorized_from_multi, legal_cells_60,
)


KS = [1, 3, 5, 10]


def build_positions(games, k_min, k_max):
    feats_list, ks_list, legal_list = [], [], []
    for game in games:
        for k in range(k_min, k_max + 1):
            legal = legal_cells_60(game, k)
            if legal is None or not legal:
                continue
            feats_list.append(played_even_features(game[:k]))
            ks_list.append(k)
            legal_list.append(legal)
    return feats_list, ks_list, legal_list


@torch.no_grad()
def ensemble_cell_scores(feats_batch, ks_batch, me, mo, idx, mask, N, device):
    """Return (N, B, 60) prob_or cell scores.  Uses eval convention:
    ks parity 1 -> me model (matches eval_multi_seed_ensemble.py)."""
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
    return -gathered.sum(dim=-1)                                # (N, B, 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-a', required=True,
                    help='First multi-seed checkpoint (label A).')
    ap.add_argument('--ckpt-b', required=True,
                    help='Second multi-seed checkpoint (label B).')
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
    print(f"  N={N_a}, H={h_a}")
    print(f"Loading B: {args.ckpt_b}")
    me_b, mo_b, N_b, h_b, _ = load_vectorized_from_multi(args.ckpt_b, device)
    print(f"  N={N_b}, H={h_b}")

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    print(f"Building test positions from {args.num_games} games...")
    games = load_val_games(args.data_dir, args.num_data_files)[:args.num_games]
    feats_list, ks_list, legal_list = build_positions(
        games, args.k_min, args.k_max)
    n_total = len(feats_list)
    print(f"  {n_total} positions")

    legal_mask = np.zeros((n_total, 60), dtype=bool)
    for i, legal in enumerate(legal_list):
        for c in legal:
            legal_mask[i, c] = True

    # Per-ensemble top-K hits, plus combined
    topk_hits_a = {K: 0 for K in KS}
    topk_hits_b = {K: 0 for K in KS}
    topk_hits_ab = {K: 0 for K in KS}  # sum both ensembles

    # Top-1 overlap counts
    both_correct = both_wrong = only_a = only_b = 0

    t0 = time.time()
    max_K = max(KS)
    for bstart in range(0, n_total, args.batch_size):
        bend = min(bstart + args.batch_size, n_total)
        feats_batch = torch.stack(feats_list[bstart:bend])
        ks_batch = torch.tensor(ks_list[bstart:bend])
        legal_batch = torch.from_numpy(legal_mask[bstart:bend]).to(device)

        cs_a = ensemble_cell_scores(
            feats_batch, ks_batch, me_a, mo_a, idx, mask, N_a, device)  # (N_a, B, 60)
        cs_b = ensemble_cell_scores(
            feats_batch, ks_batch, me_b, mo_b, idx, mask, N_b, device)  # (N_b, B, 60)

        agg_a = cs_a.sum(dim=0)   # (B, 60) - sum_log_prob_or
        agg_b = cs_b.sum(dim=0)
        agg_ab = agg_a + agg_b     # combined 200-seed ensemble

        # Top-1 overlap
        top1_a = agg_a.argmax(dim=1)          # (B,)
        top1_b = agg_b.argmax(dim=1)
        correct_a = legal_batch.gather(1, top1_a.unsqueeze(1)).squeeze(1).bool()
        correct_b = legal_batch.gather(1, top1_b.unsqueeze(1)).squeeze(1).bool()
        both_correct += int((correct_a & correct_b).sum())
        only_a       += int((correct_a & ~correct_b).sum())
        only_b       += int((~correct_a & correct_b).sum())
        both_wrong   += int((~correct_a & ~correct_b).sum())

        # Top-K per aggregator
        for name, agg, hits in [
            ("A",  agg_a,  topk_hits_a),
            ("B",  agg_b,  topk_hits_b),
            ("AB", agg_ab, topk_hits_ab),
        ]:
            topk = agg.topk(max_K, dim=1).indices     # (B, max_K)
            legal_at = legal_batch.gather(1, topk)     # (B, max_K)
            for K in KS:
                hits[K] += int(legal_at[:, :K].sum())

        if (bstart // args.batch_size) % 10 == 0:
            print(f"  {bend}/{n_total}  ({int(time.time()-t0)}s)", flush=True)

    print()
    print(f"=== Per-ensemble top-K legality ({n_total:,} positions) ===")
    print(f"  {'K':<5}  {'A':>8}  {'B':>8}  {'A+B (200 seeds)':>18}")
    for K in KS:
        va = topk_hits_a[K] / (n_total * K)
        vb = topk_hits_b[K] / (n_total * K)
        vab = topk_hits_ab[K] / (n_total * K)
        print(f"  top-{K:<2}  {va:>7.4f}   {vb:>7.4f}   {vab:>16.4f}")

    print()
    print(f"=== Top-1 overlap between A and B ({n_total:,} positions) ===")
    print(f"  both correct:  {both_correct:>7}  ({100*both_correct/n_total:5.2f}%)")
    print(f"  only A right:  {only_a:>7}  ({100*only_a/n_total:5.2f}%)")
    print(f"  only B right:  {only_b:>7}  ({100*only_b/n_total:5.2f}%)")
    print(f"  both wrong:    {both_wrong:>7}  ({100*both_wrong/n_total:5.2f}%)")
    print()
    print(f"Conditional agreements:")
    marginal_a = both_correct + only_a
    marginal_b = both_correct + only_b
    if marginal_a:
        p_b_given_a = both_correct / marginal_a
        print(f"  P(B correct | A correct) = "
              f"{both_correct}/{marginal_a} = {p_b_given_a:.4f}")
    if marginal_b:
        p_a_given_b = both_correct / marginal_b
        print(f"  P(A correct | B correct) = "
              f"{both_correct}/{marginal_b} = {p_a_given_b:.4f}")
    # If ensembles were independent, P(B|A) ~= P(B).  If they agree perfectly,
    # P(B|A) = 1.
    p_b = marginal_b / n_total
    if marginal_a:
        lift = (both_correct / marginal_a) / p_b if p_b else float('nan')
        print(f"  Correlation lift (P(B|A) / P(B)) = {lift:.3f}")
        print(f"    1.0 = independent errors;  > 1 = correlated errors")


if __name__ == '__main__':
    main()
