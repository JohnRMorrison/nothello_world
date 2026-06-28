"""Compare THREE played+even pattern-detector MLPs on legal-move prediction.

Three independently-trained MLPs (e.g., seed=0, seed=43, seed=44).  We measure:

  - Individual top-1 legal accuracy (each model)
  - 3-way ensemble accuracy (sum of log_prob_or scores)
  - Mistake overlap:
       P(all 3 wrong | any wrong)              -- the "intrinsic floor"
       P(>=2 wrong  | any wrong)               -- majority-wrong rate
  - 3-way top-1 agreement
  - Per-position disjoint breakdown (8 cases: each model correct/wrong)
  - Pairwise overlaps (A&B, A&C, B&C) for diagnostic detail

Usage:
    python compare_mlp_seeds_3way.py \\
      --ckpt-a experiments/.../pattern_simple_direct_H512_playedeven_seed0.pt \\
      --ckpt-b experiments/.../pattern_simple_direct_H512_playedeven_seed43.pt \\
      --ckpt-c experiments/.../pattern_simple_direct_H512_playedeven_seed44.pt \\
      --hidden 512 --num-games 500
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

    print(f"Loading val games from {args.data_dir} ({args.num_data_files} files)")
    games = load_val_games(args.data_dir, args.num_data_files)
    games = games[:args.num_games]
    n_positions = len(games) * (args.k_max - args.k_min + 1)
    print(f"Evaluating on {len(games)} games × {args.k_max-args.k_min+1} positions/game"
          f" = {n_positions:,} total positions")

    # Counters
    n_total = 0
    correct = {'A': 0, 'B': 0, 'C': 0, 'EAB': 0, 'EAC': 0, 'EBC': 0, 'EABC': 0}

    # 8 disjoint cases: each of A,B,C is correct (1) or wrong (0)
    # Index = 4*A + 2*B + C  (so 0b111=7 = all correct, 0b000=0 = all wrong)
    disjoint = [0] * 8

    # Top-1 agreement counters
    n_all_agree = 0
    n_all_agree_correct = 0
    n_pair_AB_agree = 0
    n_pair_AC_agree = 0
    n_pair_BC_agree = 0

    for gi, game in enumerate(games):
        for k in range(args.k_min, args.k_max + 1):
            legal = legal_cells_60(game, k)
            if not legal:
                continue
            n_total += 1
            sa = mlp_cell_scores(mlp_a, game, k, device)
            sb = mlp_cell_scores(mlp_b, game, k, device)
            sc = mlp_cell_scores(mlp_c, game, k, device)

            pa = int(np.argmax(sa))
            pb = int(np.argmax(sb))
            pc = int(np.argmax(sc))
            ok_a = pa in legal
            ok_b = pb in legal
            ok_c = pc in legal
            correct['A'] += int(ok_a)
            correct['B'] += int(ok_b)
            correct['C'] += int(ok_c)

            # Pairwise ensembles (sum of log_prob_or scores)
            p_ab = int(np.argmax(sa + sb))
            p_ac = int(np.argmax(sa + sc))
            p_bc = int(np.argmax(sb + sc))
            p_abc = int(np.argmax(sa + sb + sc))
            correct['EAB'] += int(p_ab in legal)
            correct['EAC'] += int(p_ac in legal)
            correct['EBC'] += int(p_bc in legal)
            correct['EABC'] += int(p_abc in legal)

            # 8-case disjoint
            disjoint[(int(ok_a) << 2) | (int(ok_b) << 1) | int(ok_c)] += 1

            # Agreement counts
            if pa == pb:
                n_pair_AB_agree += 1
            if pa == pc:
                n_pair_AC_agree += 1
            if pb == pc:
                n_pair_BC_agree += 1
            if pa == pb == pc:
                n_all_agree += 1
                if ok_a:
                    n_all_agree_correct += 1

        if (gi + 1) % 50 == 0:
            print(f"  {gi+1}/{len(games)} games  positions={n_total:,}  "
                  f"A={correct['A']/n_total:.4f}  B={correct['B']/n_total:.4f}  "
                  f"C={correct['C']/n_total:.4f}  "
                  f"3-ens={correct['EABC']/n_total:.4f}", flush=True)

    n = n_total
    print()
    print("=" * 72)
    print(f"Total positions: {n:,}")
    print()
    print(f"  Individual top-1 legal:")
    print(f"    A:        {correct['A']/n:.4f}  ({correct['A']:,}/{n:,})")
    print(f"    B:        {correct['B']/n:.4f}  ({correct['B']:,}/{n:,})")
    print(f"    C:        {correct['C']/n:.4f}  ({correct['C']:,}/{n:,})")
    print()
    print(f"  Pairwise ensembles (sum of log_prob_or scores):")
    print(f"    A+B:      {correct['EAB']/n:.4f}")
    print(f"    A+C:      {correct['EAC']/n:.4f}")
    print(f"    B+C:      {correct['EBC']/n:.4f}")
    print(f"  3-way ensemble (A+B+C): {correct['EABC']/n:.4f}")
    print()
    print(f"  8-way disjoint breakdown (each digit = A,B,C correct?):")
    labels = ['none', 'C only', 'B only', 'BC', 'A only', 'AC', 'AB', 'ABC (all)']
    for code, lab in enumerate(labels):
        print(f"    {bin(code)[2:].zfill(3)} ({lab:>9}): {disjoint[code]/n:.4f}  "
              f"({disjoint[code]:,})")
    print()
    n_all_wrong = disjoint[0b000]
    n_two_or_more_wrong = sum(disjoint[code] for code in range(8)
                              if bin(code).count('1') <= 1)
    n_any_wrong = sum(disjoint[code] for code in range(8) if code != 0b111)
    if n_any_wrong > 0:
        print(f"  Conditional mistake rates (given any model is wrong):")
        print(f"    P(all 3 wrong | any wrong)       = {n_all_wrong/n_any_wrong:.4f}  "
              f"({n_all_wrong:,}/{n_any_wrong:,})")
        print(f"    P(>=2 wrong   | any wrong)       = {n_two_or_more_wrong/n_any_wrong:.4f}")
        print()
        print(f"    Interpretation: P(all 3 wrong | any wrong) is the FLOOR for")
        print(f"    ensemble error; cannot be fixed by adding more models in")
        print(f"    the family.")
    print()
    print(f"  Top-1 agreement rates:")
    print(f"    All 3 agree:   {n_all_agree/n:.4f}  "
          f"(when so, acc={n_all_agree_correct/max(1,n_all_agree):.4f})")
    print(f"    A,B agree:     {n_pair_AB_agree/n:.4f}")
    print(f"    A,C agree:     {n_pair_AC_agree/n:.4f}")
    print(f"    B,C agree:     {n_pair_BC_agree/n:.4f}")


if __name__ == '__main__':
    main()
