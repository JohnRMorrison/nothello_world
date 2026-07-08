"""Number of games at each move # where the model assigns more than
`--threshold` probability to some illegal cell.

Two thresholds by default (0.01 and 0.05).  For each of OGPT and the MLP,
produces one single-panel bar chart per threshold — 4 PNGs total.

  For OGPT: max_j softmax(logits)[j]  over j where cell(j) is illegal.
  For MLP:  max_c (1 - exp(-cell_score[c]))  over c where c is illegal.

Note the MLP's prob_or is per-cell independent (not a distribution over
cells like OGPT's softmax), so both metrics are directly comparable only
in the "how confident the model is in an illegal move" sense — not as
normalized probabilities.  Left this way to mirror how the top-1 metric
compared them.

Usage:
    python plot_illegal_prob_threshold_by_move.py \\
        --mlp-ckpt experiments/.../pattern_simple_direct_H512_playedeven.pt \\
        --ogpt-ckpt ckpts/gpt_nanda_synthetic.ckpt \\
        --num-games 1000
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
import torch.nn.functional as F
from mingpt.model import GPT, GPTConfig
from compare_v4_vs_mlp import (
    load_mlp, load_val_games, C64_TO_C60,
    played_even_features,
)
from data.othello import OthelloBoardState


VOCAB_SIZE = 61
GAME_LEN = 60

DEFAULT_THRESHOLDS = [0.01, 0.05]


def tokenize_game(game_64):
    return [C64_TO_C60[c] + 1 for c in game_64 if c in C64_TO_C60]


def load_ogpt(ckpt_path, device):
    config = GPTConfig(vocab_size=VOCAB_SIZE, block_size=59,
                       n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    state = torch.load(ckpt_path, map_location="cpu")
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def mlp_probs_per_position(mlp_bundle, game_64, k_min, k_max, device):
    """Return (K, 60) prob_or probabilities per cell for k=k_min..k_max."""
    me, mo, idx, mask = mlp_bundle
    feats = [played_even_features(game_64[:k])
             for k in range(k_min, k_max + 1)]
    x = torch.stack(feats).to(device)
    ks = torch.arange(k_min, k_max + 1, device=device)
    use_me = (ks % 2 == 1)
    use_mo = ~use_me
    K = x.shape[0]
    logits = torch.zeros(K, 960, device=device)
    with torch.no_grad():
        if use_me.any():
            logits[use_me] = me(x[use_me])
        if use_mo.any():
            logits[use_mo] = mo(x[use_mo])
    log1m = -F.softplus(logits)
    gathered = log1m[:, idx]
    gathered = gathered.masked_fill(~mask, 0.0)
    cell_scores = -gathered.sum(dim=-1)                                # (K, 60)
    return (1.0 - torch.exp(-cell_scores.clamp(min=0))).cpu().numpy()  # (K, 60)


def ogpt_probs_per_position(model, game_64, device):
    """Return list of length ~len(tokens) of 60-cell softmax probabilities."""
    tokens = tokenize_game(game_64)
    if len(tokens) < 2:
        return None
    x = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x)
    logits = logits[0].cpu()                                    # (k, vocab)
    all_probs = []
    for k_idx in range(logits.shape[0]):
        # softmax over the 60 cell tokens (1..60); token 0 is pad
        p = F.softmax(logits[k_idx, 1:61], dim=0).numpy()
        all_probs.append(p)
    return all_probs


def legal_cells_60(game, k):
    board = OthelloBoardState()
    for c in game[:k]:
        try:
            board.umpire(c)
        except Exception:
            return None
    legal_64 = board.get_valid_moves()
    return {C64_TO_C60[c] for c in legal_64 if c in C64_TO_C60}


def plot_count_only(moves, counts, title, output_path, color):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(moves, counts, color=color, width=0.85)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Move number', fontsize=14)
    ax.set_ylabel('Count', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mlp-ckpt', required=True)
    ap.add_argument('--mlp-hidden', type=int, default=512)
    ap.add_argument('--ogpt-ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--num-games', type=int, default=1000)
    ap.add_argument('--k-min', type=int, default=1)
    ap.add_argument('--k-max', type=int, default=58)
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=DEFAULT_THRESHOLDS)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    ap.add_argument('--output-dir', default='experiments/plots')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading MLP: {args.mlp_ckpt}")
    mlp = load_mlp(args.mlp_ckpt, args.mlp_hidden, device)

    print(f"Loading OGPT: {args.ogpt_ckpt}")
    ogpt = load_ogpt(args.ogpt_ckpt, device)

    games = load_val_games(args.data_dir, args.num_data_files)
    games = games[:args.num_games]
    print(f"Evaluating on {len(games)} games × "
          f"{args.k_max - args.k_min + 1} positions/game")

    n_moves = args.k_max - args.k_min + 1
    n_thr = len(args.thresholds)
    # counts_by_threshold[t][mi] = count of games where max illegal prob > thresholds[t] at move mi
    mlp_counts  = np.zeros((n_thr, n_moves), dtype=int)
    ogpt_counts = np.zeros((n_thr, n_moves), dtype=int)
    totals = np.zeros(n_moves, dtype=int)

    t0 = time.time()
    for gi, game in enumerate(games):
        ogpt_probs = ogpt_probs_per_position(ogpt, game, device)
        if ogpt_probs is None:
            continue
        mlp_probs = mlp_probs_per_position(
            mlp, game, args.k_min, args.k_max, device)   # (K, 60)

        for k in range(args.k_min, args.k_max + 1):
            legal = legal_cells_60(game, k)
            if legal is None or len(legal) == 0:
                continue
            mi = k - args.k_min
            illegal_mask = np.ones(60, dtype=bool)
            for c in legal:
                illegal_mask[c] = False

            # MLP max illegal probability
            mlp_max = float((mlp_probs[mi] * illegal_mask).max())
            # OGPT max illegal probability (predictions are for move k, from context k-1)
            if k - 1 < len(ogpt_probs):
                ogpt_max = float((ogpt_probs[k - 1] * illegal_mask).max())
            else:
                continue

            totals[mi] += 1
            for ti, th in enumerate(args.thresholds):
                if mlp_max > th:
                    mlp_counts[ti, mi] += 1
                if ogpt_max > th:
                    ogpt_counts[ti, mi] += 1
        if (gi + 1) % 200 == 0:
            rate = (gi + 1) / (time.time() - t0)
            print(f"  {gi+1}/{len(games)} games  "
                  f"({rate:.1f} games/sec, ETA "
                  f"{(len(games) - gi - 1) / rate / 60:.1f} min)",
                  flush=True)

    moves = np.arange(args.k_min, args.k_max + 1)

    print()
    for ti, th in enumerate(args.thresholds):
        print(f"Threshold {th * 100:.1f}%: "
              f"MLP total {mlp_counts[ti].sum():,}  "
              f"OGPT total {ogpt_counts[ti].sum():,}  "
              f"Positions {totals.sum():,}")

    # Plot per (model, threshold)
    for ti, th in enumerate(args.thresholds):
        th_str = (f"{th * 100:.0f}"
                  if th * 100 == int(th * 100) else f"{th * 100:.1f}").replace('.', 'p')
        plot_count_only(
            moves, mlp_counts[ti],
            f"Othello-MLP: illegal cell prob > {th * 100:.0f}%",
            os.path.join(args.output_dir,
                          f"mlp_illegal_prob_gt_{th_str}pct_by_move.png"),
            color='#c0504d',
        )
        plot_count_only(
            moves, ogpt_counts[ti],
            f"Othello-GPT: illegal cell prob > {th * 100:.0f}%",
            os.path.join(args.output_dir,
                          f"ogpt_illegal_prob_gt_{th_str}pct_by_move.png"),
            color='#c0504d',
        )


if __name__ == '__main__':
    main()
