"""Top-1 legality under top-K vote aggregation.

Sweeps the "vote budget" K each seed contributes: with vote_K=1, each seed
casts one vote (its argmax cell); with vote_K=3, each seed casts three
votes; etc.  The ensemble picks the SINGLE most-voted cell (top-1) and
we measure top-1 legality.

This is different from eval_multi_seed_aggregation.py's topK_vote_count,
which used the same K for voting AND for the metric.  Here we decouple:
K controls how broadly each seed spreads its vote, but the metric is
always top-1.

Usage:
    python eval_vote_budget_top1.py \\
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
from eval_multi_seed_ensemble import (
    load_vectorized_from_multi, legal_cells_60,
)


VOTE_KS = [1, 3, 5, 10]
N_SUBSETS = [1, 5, 10, 20, 50, 100]


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
    me, mo, N, hidden, _ = load_vectorized_from_multi(
        args.multi_ckpt, device)
    print(f"  N={N} seeds, H={hidden}")

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    n_subsets = [n for n in N_SUBSETS if n <= N]
    if N not in n_subsets:
        n_subsets.append(N)
    print(f"N subset sizes: {n_subsets}")
    print(f"Vote budget K values: {VOTE_KS}")

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

    # Running total of top-1 legal hits, keyed by (n_seeds, vote_K)
    hits = {(n, k): 0.0 for n in n_subsets for k in VOTE_KS}

    print("Batched forward + vote aggregation...")
    t0 = time.time()

    with torch.no_grad():
        for bstart in range(0, n_total, args.batch_size):
            bend = min(bstart + args.batch_size, n_total)
            B = bend - bstart
            x = torch.stack(feats_list[bstart:bend]).to(device)
            ks = torch.tensor(ks_list[bstart:bend], device=device)
            use_me = (ks % 2 == 1)
            use_mo = ~use_me

            logits = torch.zeros(N, B, 960, device=device)
            if use_me.any():
                logits[:, use_me] = me(x[use_me])
            if use_mo.any():
                logits[:, use_mo] = mo(x[use_mo])
            log1m = -F.softplus(logits)
            gathered = log1m[:, :, idx]
            gathered = gathered.masked_fill(~mask[None, None], 0.0)
            cell_scores = -gathered.sum(dim=-1)         # (N, B, 60)

            legal_batch = torch.from_numpy(
                legal_mask[bstart:bend]).to(device)      # (B, 60)

            for n_seeds in n_subsets:
                sub_scores = cell_scores[:n_seeds]       # (n, B, 60)

                for vote_K in VOTE_KS:
                    # Each seed casts vote_K votes = its top vote_K cells
                    topK_per_seed = sub_scores.topk(vote_K, dim=-1).indices
                    # (n, B, vote_K)
                    votes = torch.zeros(B, 60, device=device)
                    for k_slot in range(vote_K):
                        votes.scatter_add_(
                            1, topK_per_seed[:, :, k_slot].t(),
                            torch.ones_like(topK_per_seed[:, :, k_slot].t(),
                                             dtype=torch.float32),
                        )
                    # Tie-break with sum_log_prob_or so top-1 is deterministic
                    tie_break = sub_scores.sum(dim=0) * 1e-6
                    top1 = (votes + tie_break).argmax(dim=1)  # (B,)
                    hit = legal_batch.gather(
                        1, top1.unsqueeze(1)).squeeze(1).float()  # (B,)
                    hits[(n_seeds, vote_K)] += hit.sum().item()

            if (bstart // args.batch_size) % 10 == 0:
                print(f"  {bend}/{n_total}  ({int(time.time()-t0)}s)",
                      flush=True)

    print()
    print(f"=== Top-1 legality by (N, vote budget K) — {n_total:,} positions ===")
    print()
    header = f"  {'N':<5}" + "".join(f"  K={k:>2}    " for k in VOTE_KS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for n_seeds in n_subsets:
        row = f"  {n_seeds:<5}"
        for vote_K in VOTE_KS:
            v = hits[(n_seeds, vote_K)] / n_total
            row += f"  {v:>7.4f}"
        print(row)


if __name__ == '__main__':
    main()
