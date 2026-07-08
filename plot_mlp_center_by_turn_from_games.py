"""Center-cell (4 cells: d4, e4, d5, e5) decoding accuracy vs game turn for a
saved MLP probe, computed on val games loaded directly from pickle files.

Same data source (load_games) as plot_ogpt_center_by_turn.py so the two curves
are directly comparable.

Usage:
    python plot_mlp_center_by_turn_from_games.py \\
        --pat-ckpt   experiments/.../pattern_simple_direct_H512_playedeven.pt \\
        --probe-ckpt experiments/.../probe_direct_H512_playedeven.pt \\
        --hidden 512 --n-games 500 --pos-start 0 --pos-end 60
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    get_device, OPTIONS,
)
from train_pattern_simple import DirectMLP
from compare_v4_vs_mlp import played_even_features
from data.othello import OthelloBoardState

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import load_games, GAME_LEN


CENTER_CELLS_64 = [27, 28, 35, 36]  # d4, e4, d5, e5 in 0-indexed row-major


def hidden_of(model, x):
    return torch.relu(model.net[0](x))


def state_to_gt(state_8x8):
    """OthelloBoardState.state -> probe GT: 0=empty, 1=white, 2=black."""
    gt = np.zeros(64, dtype=np.int64)
    flat = state_8x8.flatten()
    gt[flat == -1] = 1
    gt[flat == 1]  = 2
    return gt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pat-ckpt", required=True)
    p.add_argument("--probe-ckpt", required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--n-games", type=int, default=500)
    p.add_argument("--max-files", type=int, default=2)
    p.add_argument("--pos-start", type=int, default=0)
    p.add_argument("--pos-end", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--output",
                   default="experiments/plots/mlp_center_by_turn_games.png")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.pat_ckpt, map_location='cpu')
    input_dim = ckpt.get('input_dim', 120)
    n_patterns = ckpt.get('n_patterns', 960)
    model_e = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    model_o = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    model_e.load_state_dict(ckpt['even'])
    model_o.load_state_dict(ckpt['odd'])
    model_e.eval(); model_o.eval()
    print(f"Loaded MLP (input_dim={input_dim}, H={args.hidden})")

    probe_ckpt = torch.load(args.probe_ckpt, map_location='cpu')
    probe_e = nn.Linear(args.hidden, 64 * OPTIONS).to(device)
    probe_o = nn.Linear(args.hidden, 64 * OPTIONS).to(device)
    probe_e.load_state_dict(probe_ckpt['even'])
    probe_o.load_state_dict(probe_ckpt['odd'])
    probe_e.eval(); probe_o.eval()
    print(f"Loaded probe (best_acc reported: {probe_ckpt.get('best_acc', '?')})")

    games = load_games(max_files=args.max_files)
    games = [g for g in games if len(g) == GAME_LEN][:args.n_games]
    print(f"Using {len(games)} games")

    pos_start = args.pos_start
    pos_end = args.pos_end
    T = pos_end - pos_start
    turns = np.arange(pos_start, pos_end)
    per_turn_correct = np.zeros(T, dtype=np.int64)
    per_turn_total   = np.zeros(T, dtype=np.int64)

    for g_start in range(0, len(games), 32):
        for game in games[g_start:g_start + 32]:
            board = OthelloBoardState()
            feats_by_turn = {}
            gt_by_turn = {}
            for t in range(min(pos_end, len(game))):
                try:
                    board.umpire(game[t])
                except Exception:
                    break
                if t < pos_start:
                    continue
                feats_by_turn[t] = played_even_features(game[:t + 1])
                gt_by_turn[t] = state_to_gt(np.asarray(board.state, dtype=np.int8))

            if not feats_by_turn:
                continue

            turns_present = sorted(feats_by_turn.keys())
            xb = torch.stack([feats_by_turn[t] for t in turns_present]).to(device)
            yb = np.stack([gt_by_turn[t] for t in turns_present])
            with torch.no_grad():
                preds = np.zeros_like(yb)
                for i, t in enumerate(turns_present):
                    use_me = ((t + 1) % 2 == 1)
                    m = model_e if use_me else model_o
                    pr = probe_e if use_me else probe_o
                    h = hidden_of(m, xb[i:i + 1])
                    p_ = pr(h).view(64, OPTIONS).argmax(-1).cpu().numpy()
                    preds[i] = p_
            for i, t in enumerate(turns_present):
                # Only average over the 4 center cells
                match = int((preds[i][CENTER_CELLS_64] ==
                              yb[i][CENTER_CELLS_64]).sum())
                per_turn_correct[t - pos_start] += match
                per_turn_total[t - pos_start] += len(CENTER_CELLS_64)

        if (g_start // 32) % 10 == 0:
            print(f"  {g_start + 32}/{len(games)} games", flush=True)

    per_turn_center = np.where(per_turn_total > 0,
                                per_turn_correct / np.maximum(per_turn_total, 1),
                                np.nan)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    valid = ~np.isnan(per_turn_center)
    ax.plot(turns[valid], per_turn_center[valid], 'o-', color='C2',
             linewidth=2, markersize=5)
    ax.set_xlabel("Move number")
    ax.set_ylabel("Center-cell decoding accuracy (4 cells)")
    ax.set_title(f"MLP probe (H={args.hidden} played+even): "
                 f"center-cell accuracy by turn")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved {args.output}")

    print(f"\n{'turn':>4s}  {'n':>8s}  {'center_acc':>10s}")
    for t, a, n in zip(turns, per_turn_center,
                        per_turn_total // len(CENTER_CELLS_64)):
        if np.isnan(a):
            continue
        print(f"  {t:>3d}  {int(n):>8d}  {a:.4%}")


if __name__ == "__main__":
    main()
