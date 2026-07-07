"""Summarize the consist_board_k25_all_H.csv output from phase 2.

Reports, per H:
  - Fraction of (position, distinct-board) rows where the MLP's top-1
    prediction is legal (unweighted).
  - Weighted-by-MC-count version — approximates a random board sampled
    from the consistent set.
  - Mean probe cell-accuracy.
  - Split by "training board" (the actual observed board) vs. other
    consistent boards, to see whether the MLP/probe are biased toward
    the training-observed board.
  - Per-position consistency of the MLP's top-1: fraction of positions
    where the top-1 is legal for ALL consistent boards vs. NONE vs. MIXED.

Usage:
    python summarize_consist_board_csv.py --csv consist_board_k25_all_H.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows from {args.csv}")
    print(f"  H values:   {sorted(df['H'].unique().tolist())}")
    print(f"  positions:  {df['game_idx'].nunique()}")
    print(f"  k:          {sorted(df['k'].unique().tolist())}")

    # ---- 1. Per-H top-1 legal rate on distinct boards ----
    print()
    print("=" * 78)
    print("Top-1 legality across consistent boards (per H)")
    print("=" * 78)
    print(f"  {'H':>6}  {'unweighted':>12}  {'MC-weighted':>12}  {'probe acc':>12}")
    print("  " + "-" * 50)
    for H, sub in df.groupby('H'):
        unw = sub['top1_is_legal'].mean()
        mw = (sub['top1_is_legal'] * sub['board_count']).sum() / \
             sub['board_count'].sum()
        pa = sub['probe_cell_accuracy'].mean()
        print(f"  {H:>6}  {unw:>12.4f}  {mw:>12.4f}  {pa:>12.4f}")

    # ---- 2. Training-board vs other-board split ----
    print()
    print("=" * 78)
    print("Training board vs. other consistent boards")
    print("=" * 78)
    print(f"  {'H':>6}  {'source':>12}  {'top1_legal':>12}  {'probe_acc':>12}"
          f"  {'n_rows':>10}")
    print("  " + "-" * 60)
    for H, sub in df.groupby('H'):
        for is_train, name in [(1, 'training'), (0, 'other')]:
            s = sub[sub['is_training_board'] == is_train]
            if len(s) == 0:
                continue
            print(f"  {H:>6}  {name:>12}  "
                  f"{s['top1_is_legal'].mean():>12.4f}  "
                  f"{s['probe_cell_accuracy'].mean():>12.4f}  "
                  f"{len(s):>10,}")

    # ---- 3. Per-position consistency of MLP top-1 across boards ----
    print()
    print("=" * 78)
    print("Per-position consistency of top-1 across consistent boards")
    print("(counts positions where top-1 is legal for ALL, NONE, or SOME "
          "consistent boards)")
    print("=" * 78)
    print(f"  {'H':>6}  {'ALL legal':>12}  {'NONE legal':>12}  "
          f"{'MIXED':>12}  {'total':>10}")
    print("  " + "-" * 60)
    for H, sub in df.groupby('H'):
        # Group by position → fraction of that position's boards where top-1 legal
        pos_rates = sub.groupby('game_idx')['top1_is_legal'].mean()
        n_all = int((pos_rates == 1.0).sum())
        n_none = int((pos_rates == 0.0).sum())
        n_mixed = int(((pos_rates > 0) & (pos_rates < 1)).sum())
        n_total = len(pos_rates)
        print(f"  {H:>6}  {n_all:>7} ({n_all/n_total:>4.1%})  "
              f"{n_none:>7} ({n_none/n_total:>4.1%})  "
              f"{n_mixed:>7} ({n_mixed/n_total:>4.1%})  "
              f"{n_total:>10}")

    # ---- 4. Ambiguity structure ----
    print()
    print("=" * 78)
    print("Ambiguity structure (unique per position, H-independent)")
    print("=" * 78)
    positions = df.drop_duplicates('game_idx').set_index('game_idx')
    print(f"  positions with ambiguity (n_distinct_boards >= 2): "
          f"{(positions['n_distinct_boards'] >= 2).sum()}")
    print(f"  distinct-board count distribution:")
    print(f"    min:    {positions['n_distinct_boards'].min()}")
    print(f"    median: {int(positions['n_distinct_boards'].median())}")
    print(f"    mean:   {positions['n_distinct_boards'].mean():.1f}")
    print(f"    max:    {positions['n_distinct_boards'].max()}")


if __name__ == '__main__':
    main()
