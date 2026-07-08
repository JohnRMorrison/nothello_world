"""Overall (all-64-cells) decoding accuracy vs game turn for a saved MLP probe,
computed on val games loaded directly from pickle files.

Mirrors plot_ogpt_overall_by_turn.py's data-loading approach so the two
plots use the same source games and are directly comparable.

Usage:
    python plot_mlp_overall_by_turn_from_games.py \\
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
    p.add_argument("--pos-end", type=int, default=60,
                   help="Exclusive upper bound; effectively capped at "
                        "min(pos_end, len(game)).")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--output",
                   default="experiments/plots/mlp_overall_by_turn_games.png")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load MLP
    ckpt = torch.load(args.pat_ckpt, map_location='cpu')
    input_dim = ckpt.get('input_dim', 120)
    n_patterns = ckpt.get('n_patterns', 960)
    model_e = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    model_o = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    model_e.load_state_dict(ckpt['even'])
    model_o.load_state_dict(ckpt['odd'])
    model_e.eval(); model_o.eval()
    print(f"Loaded MLP (input_dim={input_dim}, H={args.hidden})")

    # Load probe
    probe_ckpt = torch.load(args.probe_ckpt, map_location='cpu')
    probe_e = nn.Linear(args.hidden, 64 * OPTIONS).to(device)
    probe_o = nn.Linear(args.hidden, 64 * OPTIONS).to(device)
    probe_e.load_state_dict(probe_ckpt['even'])
    probe_o.load_state_dict(probe_ckpt['odd'])
    probe_e.eval(); probe_o.eval()
    print(f"Loaded probe (best_acc reported: {probe_ckpt.get('best_acc', '?')})")

    # Load games (same source as plot_ogpt_overall_by_turn.py)
    games = load_games(max_files=args.max_files)
    games = [g for g in games if len(g) == GAME_LEN][:args.n_games]
    print(f"Using {len(games)} games")

    # Build per-turn feature + GT tensors
    # We'll accumulate per-turn correct/total counters as we iterate.
    pos_start = args.pos_start
    pos_end = args.pos_end
    T = pos_end - pos_start
    turns = np.arange(pos_start, pos_end)
    per_turn_correct = np.zeros(T, dtype=np.int64)
    per_turn_total   = np.zeros(T, dtype=np.int64)

    # For each game, precompute per-turn features (batched by turn)
    # and per-turn board states.
    for g_start in range(0, len(games), 32):
        game_batch = games[g_start:g_start + 32]
        for game in game_batch:
            board = OthelloBoardState()
            feats_by_turn = {}
            gt_by_turn = {}
            for t in range(min(pos_end, len(game))):
                # Position AFTER t+1 moves have been played means t moves-so-far
                # for our indexing.  We want board state AFTER move at index t,
                # matching probe training convention.
                try:
                    board.umpire(game[t])
                except Exception:
                    break
                if t < pos_start:
                    continue
                feats_by_turn[t] = played_even_features(game[:t + 1])  # (120,)
                gt_by_turn[t] = state_to_gt(np.asarray(board.state, dtype=np.int8))

            if not feats_by_turn:
                continue

            # Batch across turns for one game
            turns_present = sorted(feats_by_turn.keys())
            xb = torch.stack([feats_by_turn[t] for t in turns_present]).to(device)
            yb = np.stack([gt_by_turn[t] for t in turns_present])   # (T', 64)
            with torch.no_grad():
                preds = np.zeros_like(yb)
                for i, t in enumerate(turns_present):
                    # Parity routing: use_me at (t + 1) % 2 == 1 for val-game
                    # convention (matches eval_single_mlp_val_games.py).
                    # Here t is 0-indexed moves already played (position=t+1).
                    use_me = ((t + 1) % 2 == 1)
                    m = model_e if use_me else model_o
                    pr = probe_e if use_me else probe_o
                    h = hidden_of(m, xb[i:i + 1])
                    p_ = pr(h).view(64, OPTIONS).argmax(-1).cpu().numpy()
                    preds[i] = p_
            # Accumulate per-turn counters
            for i, t in enumerate(turns_present):
                match = int((preds[i] == yb[i]).sum())
                per_turn_correct[t - pos_start] += match
                per_turn_total[t - pos_start] += 64

        if (g_start // 32) % 10 == 0:
            print(f"  {g_start + len(game_batch)}/{len(games)} games", flush=True)

    per_turn_overall = np.where(per_turn_total > 0,
                                per_turn_correct / np.maximum(per_turn_total, 1),
                                np.nan)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    valid = ~np.isnan(per_turn_overall)
    ax.plot(turns[valid], per_turn_overall[valid], 'o-', color='C2',
             linewidth=2, markersize=5)
    ax.set_xlabel("Move number")
    ax.set_ylabel("Decoding accuracy (all 64 cells)")
    ax.set_title(f"MLP probe (H={args.hidden} played+even): "
                 f"overall accuracy by turn")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved {args.output}")

    # Match plot_ogpt_overall_by_turn.py's output format
    print(f"\n{'turn':>4s}  {'n':>8s}  {'overall':>9s}")
    for t, a, n in zip(turns, per_turn_overall,
                        per_turn_total // 64):
        if np.isnan(a):
            continue
        print(f"  {t:>3d}  {int(n):>8d}  {a:.4%}")


if __name__ == "__main__":
    main()
