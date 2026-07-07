"""Consistent-board analysis across multiple (MLP, probe) pairs — SHARES
the expensive biased-MC board enumeration across all Hs.

Refactor of consistent_board_saved_probe.py.  Given N (mlp_ckpt, probe_ckpt)
pairs, for each game position:

  1. Run enumerate_consistent_boards ONCE (the expensive per-position step).
  2. For EACH (MLP, probe) pair, run a fast MLP+probe forward pass and
     write one row per (game, distinct_board, H).

Output CSV has an H column so results for all pairs land in one file.

Usage:
    python consistent_board_multi_H.py \\
        --mlp-ckpts \\
            experiments/.../pattern_simple_direct_H512_playedeven.pt \\
            experiments/.../pattern_simple_direct_H1024_playedeven.pt \\
            experiments/.../pattern_simple_direct_H2048_playedeven.pt \\
            experiments/.../pattern_simple_direct_H4096_playedeven.pt \\
            experiments/.../pattern_simple_direct_H8192_playedeven.pt \\
        --probe-ckpts \\
            experiments/.../probe_direct_H512_playedeven.pt \\
            experiments/.../probe_direct_H1024_playedeven.pt \\
            experiments/.../probe_direct_H2048_playedeven.pt \\
            experiments/.../probe_direct_H4096_playedeven.pt \\
            experiments/.../probe_direct_H8192_playedeven.pt \\
        --k 25 --num-games 300 --n-samples 1000 \\
        --output-csv consist_board_k25_all_H.csv
"""
import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from train_pattern_simple import DirectMLP, _get_cell_pat_index
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from compare_v4_vs_mlp import (
    played_even_features, C64_TO_C60, load_val_games,
)
from data.othello import OthelloBoardState

from consistent_board_saved_probe import (
    sample_one_valid_ordering,
    enumerate_consistent_boards,
    legal_from_state,
    board_state_target_from_state,
    load_mlp_and_probe,
    forward_hidden_scores_probe,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mlp-ckpts', nargs='+', required=True,
                    help='List of MLP checkpoints — one per H value.')
    ap.add_argument('--probe-ckpts', nargs='+', required=True,
                    help='List of probe checkpoints — same order/length as --mlp-ckpts.')
    ap.add_argument('--k', type=int, default=25)
    ap.add_argument('--num-games', type=int, default=300)
    ap.add_argument('--n-samples', type=int, default=1000)
    ap.add_argument('--game-offset', type=int, default=2000)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    ap.add_argument('--output-csv', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if len(args.mlp_ckpts) != len(args.probe_ckpts):
        raise SystemExit(f"Got {len(args.mlp_ckpts)} MLP ckpts but "
                          f"{len(args.probe_ckpts)} probe ckpts.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load all (MLP, probe, H) triples
    triples = []
    for mlp_p, probe_p in zip(args.mlp_ckpts, args.probe_ckpts):
        print(f"\nLoading pair:")
        print(f"  MLP:   {mlp_p}")
        print(f"  Probe: {probe_p}")
        models, probes, H = load_mlp_and_probe(
            mlp_p, probe_p, input_dim=120, device=device)
        triples.append((H, models, probes))
    print(f"\nLoaded {len(triples)} H values: {[t[0] for t in triples]}")

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    print(f"\nLoading games...")
    games = load_val_games(args.data_dir, args.num_data_files)
    experiment_games = games[args.game_offset:
                              args.game_offset + args.num_games]
    print(f"  experiment: {len(experiment_games)} games at k={args.k}")

    print(f"\nRunning experiment for {len(experiment_games)} games "
          f"× {len(triples)} Hs...")

    with open(args.output_csv, 'w', newline='') as f_out:
        w = csv.writer(f_out)
        w.writerow([
            'game_idx', 'k', 'H', 'board_hash', 'board_count',
            'n_distinct_boards',
            'is_training_board',
            'top1_cell_60', 'top1_is_legal',
            'probe_cell_accuracy',
        ])

        n_written = 0
        n_positions = 0
        n_with_ambiguity = 0
        t0 = time.time()

        for g_idx, game in enumerate(experiment_games):
            prefix = game[:args.k]

            # SHARED across all Hs — the expensive step
            boards = enumerate_consistent_boards(prefix, args.n_samples)
            n_distinct = len(boards)
            n_positions += 1
            if n_distinct < 2:
                if (g_idx + 1) % 50 == 0:
                    print(f"  {g_idx+1}/{len(experiment_games)}  "
                          f"ambiguous: {n_with_ambiguity}  "
                          f"rows: {n_written}  "
                          f"({int(time.time()-t0)}s)", flush=True)
                    f_out.flush()
                continue
            n_with_ambiguity += 1

            # Training-observed board (shared across Hs)
            b_actual = OthelloBoardState()
            valid_actual = True
            try:
                for m in prefix:
                    b_actual.umpire(m)
            except Exception:
                valid_actual = False
            training_hash = (b_actual.state.tobytes()
                              if valid_actual else None)

            feats = played_even_features(prefix)                    # (120,)

            # Per-H: fast forward pass, then write one row per distinct board
            for H, models, probes in triples:
                cell_scores, probe_out = forward_hidden_scores_probe(
                    feats, args.k, models, probes, idx, mask, device)
                top1_60 = int(cell_scores.argmax().item())
                probe_argmax = probe_out.argmax(dim=-1).cpu().numpy()

                for b_hash, (state, next_c, count) in boards.items():
                    legal_set = legal_from_state(state, next_c)
                    top1_legal = int(top1_60 in legal_set)
                    board_target = board_state_target_from_state(state)
                    probe_acc = float((probe_argmax == board_target).mean())
                    w.writerow([
                        g_idx, args.k, H, b_hash.hex()[:16],
                        count, n_distinct,
                        int(b_hash == training_hash),
                        top1_60, top1_legal,
                        f"{probe_acc:.4f}",
                    ])
                    n_written += 1

            if (g_idx + 1) % 50 == 0:
                print(f"  {g_idx+1}/{len(experiment_games)}  "
                      f"ambiguous: {n_with_ambiguity}  "
                      f"rows: {n_written}  "
                      f"({int(time.time()-t0)}s)", flush=True)
                f_out.flush()

    print()
    print(f"Done.  Wrote {n_written} rows over "
          f"{n_with_ambiguity}/{n_positions} ambiguous positions "
          f"× {len(triples)} Hs.")
    print(f"Output: {args.output_csv}")


if __name__ == '__main__':
    main()
