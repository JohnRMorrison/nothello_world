"""Compare two played+even pattern-detector MLPs on legal-move prediction.

For each (game, position k), each model produces a 60-d cell score (via the
prob_or aggregator over its 960 pattern logits).  We measure:

  - Individual top-1 legal accuracy (each model)
  - Mistake overlap: P(both wrong | either wrong)
  - Top-1 agreement: when do they pick the same cell?
  - Disjoint counts: A-wrong-only, B-wrong-only, both-wrong, both-correct
  - Ensemble:  averaged log-prob_or scores  (this is the test of whether
               combining seed=0 + seed=44 beats either alone)

Usage:
    python compare_mlp_seeds.py \\
      --ckpt-a experiments/.../pattern_simple_direct_H512_playedeven_seed0.pt \\
      --ckpt-b experiments/.../pattern_simple_direct_H512_playedeven_seed44.pt \\
      --hidden 512 --num-games 500 --k-min 5 --k-max 53
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from compare_v4_vs_mlp import (
    load_mlp, mlp_cell_scores, load_val_games,
    C60_TO_C64, C64_TO_C60,
)
from data.othello import OthelloBoardState


def legal_cells_60(game, k):
    """Return set of legal cells (in 60-cell index) for the position after
    playing the first k moves of `game`."""
    board = OthelloBoardState()
    for c in game[:k]:
        try:
            board.umpire(c)
        except Exception:
            return set()
    legal_64 = board.get_valid_moves()
    return {C64_TO_C60[c] for c in legal_64 if c in C64_TO_C60}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-a', required=True,
                    help='Path to MLP A checkpoint (e.g., seed=0)')
    ap.add_argument('--ckpt-b', required=True,
                    help='Path to MLP B checkpoint (e.g., seed=44)')
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--num-games', type=int, default=500)
    ap.add_argument('--k-min', type=int, default=5)
    ap.add_argument('--k-max', type=int, default=53)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading A: {args.ckpt_a}")
    mlp_a = load_mlp(args.ckpt_a, args.hidden, device)
    print(f"Loading B: {args.ckpt_b}")
    mlp_b = load_mlp(args.ckpt_b, args.hidden, device)

    print(f"Loading val games from {args.data_dir} ({args.num_data_files} files)")
    games = load_val_games(args.data_dir, args.num_data_files)
    games = games[:args.num_games]
    print(f"Evaluating on {len(games)} games × {args.k_max-args.k_min+1} positions/game"
          f" = {len(games)*(args.k_max-args.k_min+1):,} total positions")

    # Counters
    n_total = 0
    a_correct = 0
    b_correct = 0
    ensemble_correct = 0
    n_a_correct_only = 0
    n_b_correct_only = 0
    n_both_correct = 0
    n_both_wrong = 0
    n_top1_agree = 0
    n_top1_agree_correct = 0

    for gi, game in enumerate(games):
        for k in range(args.k_min, args.k_max + 1):
            legal = legal_cells_60(game, k)
            if not legal:
                continue
            scores_a = mlp_cell_scores(mlp_a, game, k, device)
            scores_b = mlp_cell_scores(mlp_b, game, k, device)
            pred_a = int(np.argmax(scores_a))
            pred_b = int(np.argmax(scores_b))
            # Ensemble = sum of log_prob_or scores (both already in log-prob space)
            scores_ens = scores_a + scores_b
            pred_ens = int(np.argmax(scores_ens))

            ok_a = pred_a in legal
            ok_b = pred_b in legal
            ok_ens = pred_ens in legal

            n_total += 1
            a_correct += int(ok_a)
            b_correct += int(ok_b)
            ensemble_correct += int(ok_ens)
            if ok_a and ok_b:
                n_both_correct += 1
            elif ok_a and not ok_b:
                n_a_correct_only += 1
            elif ok_b and not ok_a:
                n_b_correct_only += 1
            else:
                n_both_wrong += 1
            if pred_a == pred_b:
                n_top1_agree += 1
                if ok_a:
                    n_top1_agree_correct += 1

        if (gi + 1) % 50 == 0:
            print(f"  {gi+1}/{len(games)} games  positions={n_total:,}  "
                  f"A={a_correct/n_total:.4f}  B={b_correct/n_total:.4f}  "
                  f"ensemble={ensemble_correct/n_total:.4f}", flush=True)

    print()
    print("=" * 65)
    print(f"Total positions: {n_total:,}")
    print(f"  Individual top-1 legal:")
    print(f"    A:        {a_correct/n_total:.4f}  ({a_correct:,}/{n_total:,})")
    print(f"    B:        {b_correct/n_total:.4f}  ({b_correct:,}/{n_total:,})")
    print(f"    Ensemble: {ensemble_correct/n_total:.4f}  "
          f"({ensemble_correct:,}/{n_total:,})")
    print()
    print(f"  Disjoint breakdown:")
    print(f"    Both correct:    {n_both_correct/n_total:.4f}")
    print(f"    A correct only:  {n_a_correct_only/n_total:.4f}")
    print(f"    B correct only:  {n_b_correct_only/n_total:.4f}")
    print(f"    Both wrong:      {n_both_wrong/n_total:.4f}")
    print()
    either_wrong = n_a_correct_only + n_b_correct_only + n_both_wrong
    if either_wrong > 0:
        p_both_wrong = n_both_wrong / either_wrong
        print(f"  P(both wrong | either wrong) = "
              f"{n_both_wrong:,}/{either_wrong:,} = {p_both_wrong:.4f}")
        print(f"    interpretation: low=disjoint mistakes (ensemble helps); "
              f"high=correlated mistakes (ensemble doesn't help)")
    print()
    print(f"  Top-1 agreement: {n_top1_agree/n_total:.4f}  "
          f"({n_top1_agree:,}/{n_total:,})")
    if n_top1_agree > 0:
        print(f"    When they agree, accuracy: "
              f"{n_top1_agree_correct/n_top1_agree:.4f}")


if __name__ == '__main__':
    main()
