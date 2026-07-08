"""Build the threshold-vs-#games table for a single MLP.

Iterates through N val games.  For each game, at each position in
[k_min, k_max], computes the MLP's prob_or per cell and records the
MAXIMUM probability the MLP assigns to any currently-illegal cell.
Per game, tracks the highest such value over the whole game.

Reports how many games have at least one position where illegal
probability exceeds each of the thresholds — the analog of the OGPT
table.

Usage:
    python adversarial_threshold_table_mlp.py \\
        --mlp-ckpt $BASE/pattern_simple_direct_H512_playedeven.pt \\
        --hidden 512 --num-games 100000 --num-data-files 3
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from train_pattern_simple import DirectMLP, _get_cell_pat_index
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from compare_v4_vs_mlp import played_even_features, C64_TO_C60, load_val_games
from data.othello import OthelloBoardState


THRESHOLDS = [0.00001, 0.001, 0.05, 0.10, 0.50]


def load_mlp(mlp_ckpt_path, hidden, device):
    ckpt = torch.load(mlp_ckpt_path, map_location='cpu')
    input_dim = ckpt.get('input_dim', 120)
    n_patterns = ckpt.get('n_patterns', 960)
    me = DirectMLP(input_dim, hidden, n_patterns)
    mo = DirectMLP(input_dim, hidden, n_patterns)
    me.load_state_dict(ckpt['even'])
    mo.load_state_dict(ckpt['odd'])
    me = me.to(device).eval()
    mo = mo.to(device).eval()
    return me, mo, input_dim, n_patterns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mlp-ckpt', required=True)
    ap.add_argument('--hidden', type=int, required=True)
    ap.add_argument('--num-games', type=int, default=100000)
    ap.add_argument('--k-min', type=int, default=5)
    ap.add_argument('--k-max', type=int, default=53)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=512)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading MLP {args.mlp_ckpt}...")
    me, mo, input_dim, n_patterns = load_mlp(args.mlp_ckpt, args.hidden, device)
    print(f"  H={args.hidden}, input_dim={input_dim}, n_patterns={n_patterns}")

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    print(f"Loading games...")
    games = load_val_games(args.data_dir, args.num_data_files)[:args.num_games]
    print(f"  {len(games)} games")

    # Build per-(game, k) feature list, tracking which game each position belongs to
    print(f"Building test set (turns {args.k_min}..{args.k_max})...")
    feats_list, ks_list, legal_list, game_id_list = [], [], [], []
    from eval_multi_seed_ensemble import legal_cells_60
    for g_id, game in enumerate(games):
        for k in range(args.k_min, args.k_max + 1):
            legal = legal_cells_60(game, k)
            if legal is None or not legal:
                continue
            feats_list.append(played_even_features(game[:k]))
            ks_list.append(k)
            legal_list.append(legal)
            game_id_list.append(g_id)
    n_total = len(feats_list)
    print(f"  {n_total:,} scored positions")

    # Build legal mask per position
    legal_mask = np.zeros((n_total, 60), dtype=bool)
    for i, lg in enumerate(legal_list):
        for c in lg:
            legal_mask[i, c] = True
    illegal_mask_np = ~legal_mask                                # (n_total, 60)
    game_ids = np.array(game_id_list, dtype=np.int64)

    # For each game, track the maximum illegal-cell prob across all its positions
    max_illegal_prob_per_game = np.zeros(len(games), dtype=np.float32)

    print("Running MLP forward + illegal-prob accumulation...")
    t0 = time.time()
    with torch.no_grad():
        for bstart in range(0, n_total, args.batch_size):
            bend = min(bstart + args.batch_size, n_total)
            B = bend - bstart
            x = torch.stack(feats_list[bstart:bend]).to(device)
            ks = torch.tensor(ks_list[bstart:bend], device=device)
            use_me = (ks % 2 == 1)
            use_mo = ~use_me
            logits = torch.zeros(B, n_patterns, device=device)
            if use_me.any():
                logits[use_me] = me(x[use_me])
            if use_mo.any():
                logits[use_mo] = mo(x[use_mo])
            log1m = -F.softplus(logits)                                # (B, 960)
            gathered = log1m[:, idx]                                    # (B, 60, K)
            gathered = gathered.masked_fill(~mask[None], 0.0)
            cell_scores = -gathered.sum(dim=-1)                        # (B, 60)
            cell_probs = 1.0 - torch.exp(-cell_scores.clamp(min=0))    # (B, 60)
            probs_np = cell_probs.cpu().numpy()

            # Zero out probabilities on legal cells; take max over illegal cells
            probs_np = probs_np * illegal_mask_np[bstart:bend]
            max_illegal = probs_np.max(axis=1)                         # (B,)

            # Update per-game max
            for i in range(B):
                gid = int(game_ids[bstart + i])
                if max_illegal[i] > max_illegal_prob_per_game[gid]:
                    max_illegal_prob_per_game[gid] = max_illegal[i]

            if (bstart // args.batch_size) % 20 == 0:
                print(f"  {bend}/{n_total}  ({int(time.time()-t0)}s)",
                      flush=True)

    # Build the table
    print()
    print(f"=== Threshold table (H={args.hidden}, {len(games):,} games) ===")
    print(f"  {'Threshold':>10}  {'# Games':>10}  {'% Games':>8}")
    print("  " + "-" * 32)
    for th in THRESHOLDS:
        n = int((max_illegal_prob_per_game > th).sum())
        pct = n / len(games)
        # Format threshold as percentage for readability
        th_pct = th * 100
        if th_pct < 0.01:
            th_str = f"{th_pct:.5f}%"
        elif th_pct < 1:
            th_str = f"{th_pct:.3f}%"
        else:
            th_str = f"{th_pct:.1f}%"
        print(f"  {th_str:>10}  {n:>10,}  {pct:>8.1%}")

    # Also save the per-game max for later analysis
    np.savez_compressed(
        f"adversarial_threshold_H{args.hidden}.npz",
        max_illegal_prob_per_game=max_illegal_prob_per_game,
        thresholds=np.array(THRESHOLDS),
    )


if __name__ == '__main__':
    main()
