"""Plot top-1 illegality (count + rate) by move number for a played+even MLP.

Mirrors the Othello-GPT 'Top-1 prediction is illegal' figure with two stacked
panels:
  top: count of positions where the top-1 prediction is illegal
  bot: rate (count / total positions at that move number)

Usage:
    python plot_mlp_topn_illegal_by_move.py \\
        --ckpt experiments/.../pattern_simple_direct_H512_playedeven_seed0.pt \\
        --hidden 512 --num-games 1000
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from compare_v4_vs_mlp import (
    load_mlp, mlp_cell_scores, load_val_games, C64_TO_C60,
)
from data.othello import OthelloBoardState


def legal_cells_60(game, k):
    board = OthelloBoardState()
    for c in game[:k]:
        try:
            board.umpire(c)
        except Exception:
            return None
    legal_64 = board.get_valid_moves()
    return {C64_TO_C60[c] for c in legal_64 if c in C64_TO_C60}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--num-games', type=int, default=1000)
    ap.add_argument('--k-min', type=int, default=0)
    ap.add_argument('--k-max', type=int, default=59)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    ap.add_argument('--output',
                    default='experiments/plots/mlp_topn_illegal_by_move.png')
    ap.add_argument('--title', default='Othello-MLP')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading MLP: {args.ckpt}")
    mlp = load_mlp(args.ckpt, args.hidden, device)

    print(f"Loading val games from {args.data_dir}")
    games = load_val_games(args.data_dir, args.num_data_files)
    games = games[:args.num_games]
    print(f"Evaluating on {len(games)} games × "
          f"{args.k_max - args.k_min + 1} positions/game")

    n_moves = args.k_max - args.k_min + 1
    illegal_count = np.zeros(n_moves, dtype=int)
    total_count = np.zeros(n_moves, dtype=int)

    for gi, game in enumerate(games):
        for k in range(args.k_min, args.k_max + 1):
            legal = legal_cells_60(game, k)
            if legal is None or len(legal) == 0:
                continue
            scores = mlp_cell_scores(mlp, game, k, device)
            pred = int(np.argmax(scores))
            mi = k - args.k_min
            total_count[mi] += 1
            if pred not in legal:
                illegal_count[mi] += 1
        if (gi + 1) % 200 == 0:
            print(f"  {gi+1}/{len(games)} games", flush=True)

    rate = np.where(total_count > 0,
                    illegal_count / np.maximum(total_count, 1), 0.0)
    moves = np.arange(args.k_min, args.k_max + 1)

    # Match OGPT figure visuals: stacked panels, salmon top, blue bottom
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_top.bar(moves, illegal_count, color='#c0504d', width=0.85)
    ax_top.set_title('(Top-1 prediction is illegal)',
                      fontsize=14, fontweight='bold')
    ax_top.set_ylabel('Count')
    ax_top.grid(False)

    ax_bot.bar(moves, rate, color='#5b9bd5', width=0.85)
    ax_bot.set_ylabel('Rate')
    ax_bot.set_xlabel('Move number', fontsize=14)
    ax_bot.grid(False)

    # Slim formatting (no top/right spines, like the OGPT version)
    for ax in (ax_top, ax_bot):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"Saved {args.output}")

    # Print summary numbers
    n_total = int(total_count.sum())
    n_illegal = int(illegal_count.sum())
    print()
    print(f"Total positions: {n_total:,}")
    print(f"Illegal top-1:   {n_illegal:,}  ({n_illegal/n_total*100:.2f}%)")
    print(f"Range:           moves {args.k_min}..{args.k_max}")


if __name__ == '__main__':
    main()
