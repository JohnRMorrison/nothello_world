"""Diagnose why aggregation misses legal cells at top-K.

Finds positions where the best output-space aggregator (sum_log_prob_or)
put an illegal cell in its top-K while at least one legal cell fell out.
For each such position, prints:

  - Board metadata (turn, legal cells)
  - Per-seed top-3 picks histogram (which cells got seed argmax votes)
  - The aggregated top-K and where the missed legal cells ranked
  - For each MISSED legal cell: how many seeds ranked it in their own
    top-1 / top-3 / top-5.  If some individual seeds put a missed legal
    cell in top-3 but the aggregate ignored it, we've validated the
    "trust-the-outlier" hypothesis.

Usage:
    python diagnose_aggregation_misses.py \\
        --multi-ckpt experiments/.../multi_seed_N50_H4096_playedeven.pt \\
        --num-games 500  --top-k 5  --max-positions 20
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
from eval_multi_seed_ensemble import load_vectorized_from_multi, legal_cells_60


C60_TO_C64 = {v: k for k, v in C64_TO_C60.items()}


def cell60_label(c60):
    """Return e.g. 'D3' for a 60-index cell."""
    c64 = C60_TO_C64[c60]
    col = c64 % 8
    row = c64 // 8
    return f"{chr(ord('A') + col)}{row + 1}"


def build_test_positions(games, k_min, k_max):
    """Return list of (feats, k, legal_set) for each valid position."""
    out = []
    for g_idx, game in enumerate(games):
        for k in range(k_min, k_max + 1):
            legal = legal_cells_60(game, k)
            if legal is None or not legal:
                continue
            feats = played_even_features(game[:k])
            out.append((feats, k, legal, g_idx))
    return out


def compute_cell_scores(feats_batch, ks_batch, me, mo, N, idx, mask, device):
    """Return cell_scores (N, B, 60) for a batch of positions."""
    x = feats_batch.to(device)
    use_me = (ks_batch % 2 == 1)
    use_mo = ~use_me
    B = x.shape[0]
    logits = torch.zeros(N, B, 960, device=device)
    if use_me.any():
        logits[:, use_me] = me(x[use_me])
    if use_mo.any():
        logits[:, use_mo] = mo(x[use_mo])
    log1m = -F.softplus(logits)
    gathered = log1m[:, :, idx]
    gathered = gathered.masked_fill(~mask[None, None], 0.0)
    cell_scores = -gathered.sum(dim=-1)                        # (N, B, 60)
    return cell_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi-ckpt', required=True)
    ap.add_argument('--num-games', type=int, default=500)
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--max-positions', type=int, default=20)
    ap.add_argument('--k-min', type=int, default=5)
    ap.add_argument('--k-max', type=int, default=53)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0,
                    help='Seed for choosing which miss positions to sample.')
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

    print(f"Building test set from {args.num_games} games...")
    games = load_val_games(args.data_dir, args.num_data_files)[:args.num_games]
    positions = build_test_positions(games, args.k_min, args.k_max)
    n_total = len(positions)
    print(f"  {n_total} positions")

    K = args.top_k

    # Aggregate-and-classify pass: find positions where sum_log_prob_or's
    # top-K missed at least one legal cell that some seed ranked in top-3.
    interesting = []  # list of (pos_idx, agg_topk, missed_legal, cell_scores)

    print(f"Scanning for top-{K} misses...")
    t0 = time.time()
    with torch.no_grad():
        for bstart in range(0, n_total, args.batch_size):
            bend = min(bstart + args.batch_size, n_total)
            feats_batch = torch.stack([positions[i][0] for i in range(bstart, bend)])
            ks_batch = torch.tensor(
                [positions[i][1] for i in range(bstart, bend)],
                device=device,
            )
            cell_scores = compute_cell_scores(
                feats_batch, ks_batch, me, mo, N, idx, mask, device)  # (N, B, 60)
            agg = cell_scores.sum(dim=0)                            # (B, 60)
            topk_idx = agg.topk(K, dim=1).indices                    # (B, K)

            for b in range(bend - bstart):
                pos_idx = bstart + b
                legal = positions[pos_idx][2]           # set of 60-indices
                topk = topk_idx[b].tolist()
                missed = legal - set(topk)
                if not missed:
                    continue
                # For each missed legal cell, check if any single seed ranked
                # it in that seed's top-3.
                cs = cell_scores[:, b, :]                             # (N, 60)
                per_seed_top3 = cs.topk(3, dim=1).indices              # (N, 3)
                trusted_outlier = False
                for c in missed:
                    if (per_seed_top3 == c).any():
                        trusted_outlier = True
                        break
                if not trusted_outlier:
                    # Missed legal cells that NO seed put in its top-3 aren't
                    # useful for the outlier-trust story; skip them (info
                    # genuinely not in the ensemble).
                    continue
                interesting.append({
                    'pos_idx': pos_idx,
                    'topk':    topk,
                    'legal':   legal,
                    'missed':  missed,
                    'cs':      cs.cpu(),   # (N, 60)
                })

            if (bstart // args.batch_size) % 10 == 0:
                print(f"  {bend}/{n_total}  interesting={len(interesting)}  "
                      f"({int(time.time()-t0)}s)", flush=True)

    print(f"\nFound {len(interesting)} positions where sum_log_prob_or's "
          f"top-{K} missed a legal cell AND some seed ranked that missed "
          f"cell in its own top-3.")
    if not interesting:
        print("Nothing to display — all misses were on cells NO seed liked.")
        print("This means the ensemble has genuine consensus-blind spots; "
              "no output-space aggregator can recover them.")
        return

    # Sample max_positions randomly (deterministic via --seed).
    rng = np.random.RandomState(args.seed)
    if len(interesting) > args.max_positions:
        chosen = rng.choice(len(interesting), args.max_positions, replace=False)
    else:
        chosen = np.arange(len(interesting))

    print(f"\nShowing {len(chosen)} sample positions:\n")
    for display_i, i in enumerate(chosen):
        info = interesting[i]
        pos_idx = info['pos_idx']
        feats, k, legal, g_idx = positions[pos_idx]
        cs = info['cs']                              # (N, 60)
        topk = info['topk']
        missed = info['missed']

        print(f"=" * 70)
        print(f"Position #{display_i+1}  (game {g_idx}, turn k={k})")
        legal_labels = sorted(cell60_label(c) for c in legal)
        print(f"  Legal cells ({len(legal)}):  {' '.join(legal_labels)}")

        # Aggregated top-K:
        agg_topk_labels = [cell60_label(c) for c in topk]
        agg_marks = ['*' if c in legal else 'x' for c in topk]
        print(f"  Aggregated top-{K}:  " + "  ".join(
            f"{lab}{mark}" for lab, mark in zip(agg_topk_labels, agg_marks)))

        # For each seed, its top-1 pick.
        seed_top1 = cs.argmax(dim=1).numpy()                            # (N,)
        seed_top3 = cs.topk(3, dim=1).indices.numpy()                    # (N, 3)

        # Vote counts for TOP-1 across seeds.
        vote_count = np.bincount(seed_top1, minlength=60)
        # For each cell, count how many seeds put it in top-3.
        top3_count = np.zeros(60, dtype=int)
        for row in seed_top3:
            for c in row:
                top3_count[c] += 1

        # Show all cells that received >= 1 top-1 vote OR are legal.
        cells_to_show = set(int(c) for c in np.where(vote_count > 0)[0].tolist())
        cells_to_show |= legal

        # Sort by vote count desc, then top-3 count desc.
        cells_sorted = sorted(
            cells_to_show,
            key=lambda c: (-int(vote_count[c]), -int(top3_count[c])),
        )

        print(f"  {'cell':<5}  {'legal?':<6}  {'top-1 votes':<12}  "
              f"{'top-3 votes':<12}  {'in agg top-K?'}")
        for c in cells_sorted:
            is_legal = "YES" if c in legal else ""
            in_agg = ""
            if c in topk:
                rank = topk.index(c) + 1
                in_agg = f"rank {rank}"
            elif c in missed:
                in_agg = "(missed)"
            print(f"  {cell60_label(c):<5}  {is_legal:<6}  "
                  f"{vote_count[c]:<12}  {top3_count[c]:<12}  {in_agg}")

        # Highlight missed legal cells.
        for c in sorted(missed):
            top1_v = int(vote_count[c])
            top3_v = int(top3_count[c])
            print(f"  --> Missed legal {cell60_label(c)}: "
                  f"{top1_v}/{N} seeds' top-1, "
                  f"{top3_v}/{N} seeds' top-3")
        print()

    # Summary statistics.
    print("=" * 70)
    print(f"SUMMARY over {len(interesting)} interesting positions:")
    all_top1_votes_for_missed = []
    all_top3_votes_for_missed = []
    for info in interesting:
        cs = info['cs']
        seed_top1 = cs.argmax(dim=1).numpy()
        seed_top3 = cs.topk(3, dim=1).indices.numpy()
        vote_count = np.bincount(seed_top1, minlength=60)
        top3_count = np.zeros(60, dtype=int)
        for row in seed_top3:
            for c in row:
                top3_count[c] += 1
        for c in info['missed']:
            all_top1_votes_for_missed.append(int(vote_count[c]))
            all_top3_votes_for_missed.append(int(top3_count[c]))

    if all_top1_votes_for_missed:
        arr1 = np.array(all_top1_votes_for_missed)
        arr3 = np.array(all_top3_votes_for_missed)
        print(f"  For MISSED legal cells (that at least 1 seed liked):")
        print(f"    Seeds voting for it as top-1:  "
              f"mean={arr1.mean():.1f}  median={int(np.median(arr1))}  "
              f"max={arr1.max()}")
        print(f"    Seeds ranking it in top-3:     "
              f"mean={arr3.mean():.1f}  median={int(np.median(arr3))}  "
              f"max={arr3.max()}")
        few_seed_missed = (arr1 <= 2).sum()
        print(f"    # missed cells with only 1-2 seeds voting: "
              f"{few_seed_missed}/{len(arr1)} "
              f"({100*few_seed_missed/len(arr1):.1f}%)")


if __name__ == '__main__':
    main()
