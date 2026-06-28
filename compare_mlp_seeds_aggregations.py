"""Try different aggregation strategies for the 3-MLP ensemble.

Each model produces a 60-d log_prob_or score (cell_scores[c] = -log(1 - p_or_c)).
We convert to per-cell legality probability p_c = 1 - exp(-cell_scores[c]) and
explore several aggregations:

  Score-space (sum of log scores):
    sum  log_prob_or (OR; current)   --  -sum log(1 - p_i)
    geom mean p_i (AND; consensus)   --  sum log(p_i)

  Probability-space:
    mean p_i (arith)                 --  (p1 + p2 + p3) / 3
    min  p_i (most conservative)     --  needs all to agree
    max  p_i (most permissive)       --  any model rescues

  Vote-based:
    majority vote on argmax          --  count top-1 picks, choose plurality

Output: top-1 legal for each aggregation, plus the disjoint breakdown
relative to ground-truth legality.

Usage:
    python compare_mlp_seeds_aggregations.py \\
      --ckpt-a ...seed0.pt --ckpt-b ...seed43.pt --ckpt-c ...seed44.pt \\
      --hidden 512 --num-games 500
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from compare_v4_vs_mlp import (
    load_mlp, mlp_cell_scores, load_val_games,
    C64_TO_C60,
)
from data.othello import OthelloBoardState


EPS = 1e-10


def legal_cells_60(game, k):
    board = OthelloBoardState()
    for c in game[:k]:
        try:
            board.umpire(c)
        except Exception:
            return set()
    legal_64 = board.get_valid_moves()
    return {C64_TO_C60[c] for c in legal_64 if c in C64_TO_C60}


def score_to_prob(s):
    """cell_scores = -log(1 - p_or)  =>  p_or = 1 - exp(-s).  Clip for safety."""
    p = 1.0 - np.exp(-np.maximum(s, 0))
    return np.clip(p, EPS, 1.0 - EPS)


# Each aggregator takes (sa, sb, sc, pa, pb, pc) and returns a 60-d score
# vector whose ARGMAX is the predicted cell.
AGGREGATORS = {
    # Single-model baselines
    'A only':                 lambda sa, sb, sc, pa, pb, pc: sa,
    'B only':                 lambda sa, sb, sc, pa, pb, pc: sb,
    'C only':                 lambda sa, sb, sc, pa, pb, pc: sc,
    # Score-space OR (current method)
    'sum log_prob_or (OR)':   lambda sa, sb, sc, pa, pb, pc: sa + sb + sc,
    # Score-space AND  (= log of product of probabilities = geom mean ranking)
    'sum log(p) (AND)':       lambda sa, sb, sc, pa, pb, pc: (np.log(pa) + np.log(pb) + np.log(pc)),
    # Probability-space mean
    'mean p (arith)':         lambda sa, sb, sc, pa, pb, pc: (pa + pb + pc) / 3.0,
    # Probability-space min/max
    'min p (consensus)':      lambda sa, sb, sc, pa, pb, pc: np.minimum(np.minimum(pa, pb), pc),
    'max p (any)':            lambda sa, sb, sc, pa, pb, pc: np.maximum(np.maximum(pa, pb), pc),
    # Pairwise sums (already in compare_mlp_seeds_3way, included for reference)
    'A+B':                    lambda sa, sb, sc, pa, pb, pc: sa + sb,
    'A+C':                    lambda sa, sb, sc, pa, pb, pc: sa + sc,
    'B+C':                    lambda sa, sb, sc, pa, pb, pc: sb + sc,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-a', required=True)
    ap.add_argument('--ckpt-b', required=True)
    ap.add_argument('--ckpt-c', required=True)
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
    print(f"Loading C: {args.ckpt_c}")
    mlp_c = load_mlp(args.ckpt_c, args.hidden, device)

    games = load_val_games(args.data_dir, args.num_data_files)[:args.num_games]
    print(f"Evaluating on {len(games)} games × {args.k_max-args.k_min+1} positions/game")

    # Counters
    agg_correct = {name: 0 for name in AGGREGATORS}
    majority_correct = 0
    majority_total = 0     # positions where majority vote produced a non-tie
    n_total = 0

    for gi, game in enumerate(games):
        for k in range(args.k_min, args.k_max + 1):
            legal = legal_cells_60(game, k)
            if not legal:
                continue
            n_total += 1
            sa = mlp_cell_scores(mlp_a, game, k, device)
            sb = mlp_cell_scores(mlp_b, game, k, device)
            sc = mlp_cell_scores(mlp_c, game, k, device)
            pa = score_to_prob(sa)
            pb = score_to_prob(sb)
            pc = score_to_prob(sc)

            for name, fn in AGGREGATORS.items():
                scores = fn(sa, sb, sc, pa, pb, pc)
                pred = int(np.argmax(scores))
                agg_correct[name] += int(pred in legal)

            # Majority vote on top-1 picks (handles ties by score-sum tiebreak)
            picks = [int(np.argmax(sa)), int(np.argmax(sb)), int(np.argmax(sc))]
            from collections import Counter
            ctr = Counter(picks)
            top, top_count = ctr.most_common(1)[0]
            if top_count >= 2:
                # Clear majority (2 or 3 of 3)
                majority_pred = top
                majority_total += 1
                majority_correct += int(majority_pred in legal)
            else:
                # 3-way tie -> fall back to OR ensemble
                majority_pred = int(np.argmax(sa + sb + sc))
                majority_total += 1
                majority_correct += int(majority_pred in legal)

        if (gi + 1) % 50 == 0:
            best = max(agg_correct.values()) / n_total
            print(f"  {gi+1}/{len(games)} games  positions={n_total:,}  "
                  f"best so far={best:.4f}", flush=True)

    print()
    print("=" * 65)
    print(f"Total positions: {n_total:,}")
    print()
    print(f"  {'Aggregation':<28}  Top-1 legal")
    print(f"  {'-'*28}  -----------")
    # Sort by accuracy for easy comparison
    sorted_results = sorted(agg_correct.items(), key=lambda kv: -kv[1])
    for name, c in sorted_results:
        flag = ""
        if name == 'sum log_prob_or (OR)':
            flag = "  <-- current method"
        print(f"  {name:<28}  {c/n_total:.4f}  ({c:,}/{n_total:,}){flag}")
    print()
    print(f"  Majority vote on argmax:    "
          f"{majority_correct/majority_total:.4f}  "
          f"({majority_correct:,}/{majority_total:,})")


if __name__ == '__main__':
    main()
