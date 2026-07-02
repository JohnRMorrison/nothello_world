"""Dump per-network prob_or cell scores for a sample of positions to CSVs.

For each sampled position, writes one CSV with:

    Row per 60-cell.  Columns:
      cell         - like "D3"
      is_legal     - True/False
      seed_0 ... seed_{N-1}  - prob_or score from each seed
      mean         - average score across seeds
      max          - max score across seeds
      top1_votes   - # seeds where this cell was that seed's argmax
      top3_votes   - # seeds where this cell was in that seed's top-3
      mean_rank    - average rank (1 = each seed's top pick) across seeds

Also writes an index.csv summarizing every sampled position.

Usage:
    python dump_position_scores.py \\
        --multi-ckpt experiments/.../multi_seed_N50_H4096_playedeven.pt \\
        --output-dir position_scores/  --num-positions 30
"""
import argparse
import csv
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
    c64 = C60_TO_C64[c60]
    col = c64 % 8
    row = c64 // 8
    return f"{chr(ord('A') + col)}{row + 1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi-ckpt', required=True)
    ap.add_argument('--output-dir', default='position_scores')
    ap.add_argument('--num-positions', type=int, default=30)
    ap.add_argument('--num-games', type=int, default=500,
                    help='Games to draw positions from.')
    ap.add_argument('--sample-mode', choices=['random', 'misses'],
                    default='misses',
                    help='"misses": positions where sum_log_prob_or top-5 '
                         'missed a legal cell. "random": uniform random.')
    ap.add_argument('--top-k', type=int, default=5,
                    help='K used to decide what counts as a "miss".')
    ap.add_argument('--k-min', type=int, default=5)
    ap.add_argument('--k-max', type=int, default=53)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0,
                    help='RNG seed for sampling.')
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

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Building positions from {args.num_games} games...")
    games = load_val_games(args.data_dir, args.num_data_files)[:args.num_games]
    positions = []
    for g_idx, game in enumerate(games):
        for k in range(args.k_min, args.k_max + 1):
            legal = legal_cells_60(game, k)
            if legal is None or not legal:
                continue
            positions.append((played_even_features(game[:k]), k, legal, g_idx))
    n_total = len(positions)
    print(f"  {n_total} positions")

    # Compute cell_scores for all positions, batched, keep on CPU as float32.
    print("Forward pass through ensemble...")
    all_scores = np.zeros((n_total, N, 60), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for bstart in range(0, n_total, args.batch_size):
            bend = min(bstart + args.batch_size, n_total)
            feats_batch = torch.stack(
                [positions[i][0] for i in range(bstart, bend)]).to(device)
            ks_batch = torch.tensor(
                [positions[i][1] for i in range(bstart, bend)], device=device)
            use_me = (ks_batch % 2 == 1)
            use_mo = ~use_me
            B = bend - bstart
            logits = torch.zeros(N, B, 960, device=device)
            if use_me.any():
                logits[:, use_me] = me(feats_batch[use_me])
            if use_mo.any():
                logits[:, use_mo] = mo(feats_batch[use_mo])
            log1m = -F.softplus(logits)
            gathered = log1m[:, :, idx]
            gathered = gathered.masked_fill(~mask[None, None], 0.0)
            cs = -gathered.sum(dim=-1)                              # (N, B, 60)
            all_scores[bstart:bend] = cs.permute(1, 0, 2).cpu().numpy()
            if (bstart // args.batch_size) % 10 == 0:
                print(f"  {bend}/{n_total}  ({int(time.time()-t0)}s)",
                      flush=True)

    # Choose which positions to dump.
    rng = np.random.RandomState(args.seed)
    if args.sample_mode == 'misses':
        candidates = []
        for pos_idx in range(n_total):
            agg = all_scores[pos_idx].sum(axis=0)                    # (60,)
            topk = np.argsort(-agg)[:args.top_k]
            legal = positions[pos_idx][2]
            missed = legal - set(int(c) for c in topk)
            if missed:
                candidates.append(pos_idx)
        print(f"  {len(candidates)} positions where top-{args.top_k} "
              f"missed a legal cell")
        if not candidates:
            print("Nothing to dump.")
            return
        if len(candidates) > args.num_positions:
            picked = rng.choice(candidates, args.num_positions, replace=False)
        else:
            picked = np.array(candidates)
    else:
        picked = rng.choice(n_total, min(args.num_positions, n_total),
                             replace=False)

    picked = sorted(int(p) for p in picked)
    print(f"Dumping {len(picked)} positions to {args.output_dir}/...")

    # Write index.csv.
    idx_path = os.path.join(args.output_dir, 'index.csv')
    with open(idx_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['position_id', 'game_idx', 'turn_k',
                    'n_legal', 'legal_cells',
                    'agg_top5', 'top5_hits', 'missed_legal',
                    'csv_file'])
        for pos_idx in picked:
            _, k, legal, g_idx = positions[pos_idx]
            agg = all_scores[pos_idx].sum(axis=0)
            top5 = [int(c) for c in np.argsort(-agg)[:5]]
            top5_labels = ' '.join(cell60_label(c) for c in top5)
            top5_hits = sum(1 for c in top5 if c in legal)
            missed = sorted(legal - set(top5))
            missed_labels = ' '.join(cell60_label(c) for c in missed)
            legal_labels = ' '.join(sorted(cell60_label(c) for c in legal))
            fname = f"pos_{pos_idx:06d}_g{g_idx:05d}_k{k:02d}.csv"
            w.writerow([pos_idx, g_idx, k, len(legal), legal_labels,
                        top5_labels, top5_hits, missed_labels, fname])

    # Write one CSV per position.
    for pos_idx in picked:
        _, k, legal, g_idx = positions[pos_idx]
        cs = all_scores[pos_idx]                                    # (N, 60)
        seed_argmax = cs.argmax(axis=1)                             # (N,)
        # Top-3 per seed.
        seed_top3 = np.argsort(-cs, axis=1)[:, :3]                   # (N, 3)
        # Rank per (seed, cell): 1 = seed's top pick.
        seed_ranks = (-cs).argsort(axis=1).argsort(axis=1) + 1       # (N, 60)

        vote_top1 = np.bincount(seed_argmax, minlength=60)
        vote_top3 = np.zeros(60, dtype=int)
        for row in seed_top3:
            for c in row:
                vote_top3[c] += 1
        mean_score = cs.mean(axis=0)                                # (60,)
        max_score = cs.max(axis=0)                                  # (60,)
        mean_rank = seed_ranks.mean(axis=0)                         # (60,)

        # Sort cells by mean_score desc for readability.
        cell_order = np.argsort(-mean_score)

        fname = f"pos_{pos_idx:06d}_g{g_idx:05d}_k{k:02d}.csv"
        path = os.path.join(args.output_dir, fname)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            header = ['cell', 'is_legal',
                       'mean', 'max',
                       'top1_votes', 'top3_votes', 'mean_rank']
            header += [f'seed_{s}' for s in range(N)]
            w.writerow(header)
            for c in cell_order:
                row = [
                    cell60_label(int(c)),
                    'YES' if c in legal else '',
                    f"{mean_score[c]:.6f}",
                    f"{max_score[c]:.6f}",
                    int(vote_top1[c]),
                    int(vote_top3[c]),
                    f"{mean_rank[c]:.2f}",
                ]
                row += [f"{cs[s, c]:.6f}" for s in range(N)]
                w.writerow(row)

    print(f"Wrote index.csv + {len(picked)} per-position CSVs.")
    print(f"Suggested next: open index.csv to browse, then open any of "
          f"the per-position files to see the per-seed scores for that "
          f"position.")


if __name__ == '__main__':
    main()
