"""Top-1 illegal-prediction counts per move number, for both OGPT and the MLP,
evaluated on the SAME set of (game, position) pairs.

Produces two separate single-panel PNGs (count only — no rate panel).

Usage:
    python plot_topn_illegal_comparison.py \\
        --mlp-ckpt experiments/.../pattern_simple_direct_H512_playedeven_seed0.pt \\
        --ogpt-ckpt ckpts/gpt_nanda_synthetic.ckpt \\
        --num-games 1000
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
import torch.nn.functional as F
from mingpt.model import GPT, GPTConfig
from compare_v4_vs_mlp import (
    load_mlp, mlp_cell_scores, load_val_games, C64_TO_C60, C60_TO_C64,
    played_even_features,
)
from data.othello import OthelloBoardState


# Nanda-style OGPT tokenization (60-cell vocab + 1 pad)
VOCAB_SIZE = 61
GAME_LEN = 60


def tokenize_game(game_64):
    """Convert a list of 64-cell move indices to vocab indices (1-60)."""
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


def mlp_predict_per_position(mlp_bundle, game_64, k_min, k_max, device):
    """Batched MLP predictions at all positions k=k_min..k_max in one game.
    Returns list of 60-cell argmax predictions.
    """
    me, mo, idx, mask = mlp_bundle
    # Build features for all k in one batch
    feats = []
    for k in range(k_min, k_max + 1):
        feats.append(played_even_features(game_64[:k]))
    x = torch.stack(feats).to(device)        # (K, 120)
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
    log1m = -F.softplus(logits)               # (K, 960)
    gathered = log1m[:, idx]                  # (K, 60, max_K)
    gathered = gathered.masked_fill(~mask, 0.0)
    cell_scores = -gathered.sum(dim=-1)       # (K, 60)
    return cell_scores.argmax(dim=-1).cpu().numpy()


def ogpt_predict_per_position(model, game_64, device):
    """Returns (60,) array of top-1 predicted 60-cell index for each k=1..59."""
    tokens = tokenize_game(game_64)
    if len(tokens) < 2:
        return None
    # Drop last token to match block_size=59
    x = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x)
    # logits[0, k-1] is the prediction conditioned on the first k tokens
    logits = logits[0].cpu()             # (k, vocab)
    # For each k (position to predict), take argmax over the 60 cell tokens
    # (indices 1..60).  Index 0 is pad — exclude.
    preds_60 = []
    for k_idx in range(logits.shape[0]):
        # token logits at index k_idx are the prediction for the (k_idx+1)th move
        token_logits = logits[k_idx, 1:61]   # 60 cell tokens
        pred_60 = int(token_logits.argmax())
        preds_60.append(pred_60)
    return preds_60   # length k+1 (predictions at k=1..k+1)


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
    mlp_illegal = np.zeros(n_moves, dtype=int)
    ogpt_illegal = np.zeros(n_moves, dtype=int)
    totals = np.zeros(n_moves, dtype=int)

    import time
    t0 = time.time()
    for gi, game in enumerate(games):
        ogpt_preds = ogpt_predict_per_position(ogpt, game, device)
        if ogpt_preds is None:
            continue
        # Batched MLP forward across all positions for this game
        mlp_preds = mlp_predict_per_position(
            mlp, game, args.k_min, args.k_max, device)
        for k in range(args.k_min, args.k_max + 1):
            legal = legal_cells_60(game, k)
            if legal is None or len(legal) == 0:
                continue
            mi = k - args.k_min
            mlp_pred = int(mlp_preds[mi])
            if k - 1 < len(ogpt_preds):
                ogpt_pred = ogpt_preds[k - 1]
            else:
                continue
            totals[mi] += 1
            if mlp_pred not in legal:
                mlp_illegal[mi] += 1
            if ogpt_pred not in legal:
                ogpt_illegal[mi] += 1
        if (gi + 1) % 500 == 0:
            rate = (gi + 1) / (time.time() - t0)
            print(f"  {gi+1}/{len(games)} games  "
                  f"({rate:.1f} games/sec, ETA "
                  f"{(len(games) - gi - 1) / rate / 60:.1f} min)",
                  flush=True)

    moves = np.arange(args.k_min, args.k_max + 1)

    print()
    print(f"MLP total illegal:  {mlp_illegal.sum():,}  "
          f"({mlp_illegal.sum() / max(1, totals.sum()) * 100:.2f}%)")
    print(f"OGPT total illegal: {ogpt_illegal.sum():,}  "
          f"({ogpt_illegal.sum() / max(1, totals.sum()) * 100:.2f}%)")
    print(f"Total positions:    {totals.sum():,}")

    plot_count_only(
        moves, mlp_illegal,
        "Othello-MLP: top-1 prediction is illegal",
        os.path.join(args.output_dir, "mlp_topn_illegal_by_move.png"),
        color='#c0504d',
    )
    plot_count_only(
        moves, ogpt_illegal,
        "Othello-GPT: top-1 prediction is illegal",
        os.path.join(args.output_dir, "ogpt_topn_illegal_by_move.png"),
        color='#c0504d',
    )

    # Save the per-move data so downstream plotting (e.g. the presentation
    # notebook) can render without re-running.
    data_path = os.path.join(args.output_dir, "topn_illegal_by_move.npz")
    np.savez(
        data_path,
        moves=moves,
        mlp_illegal=mlp_illegal,
        ogpt_illegal=ogpt_illegal,
        totals=totals,
        num_games=len(games),
        k_min=args.k_min,
        k_max=args.k_max,
    )
    print(f"Saved per-move data to {data_path}")


if __name__ == '__main__':
    main()
