"""Evaluate a single-model DirectMLP on the val-games dataset used by
eval_multi_seed_aggregation.py.  Reports top-1/3/5/10 legality with the
achievability-aware metric (denominator = min(K, n_legal)).

Usage:
    python eval_single_mlp_val_games.py \\
        --ckpt experiments/.../pattern_simple_direct_H512_playedeven.pt \\
        --hidden 512 --num-games 500
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
from compare_v4_vs_mlp import (
    load_val_games, played_even_features, C64_TO_C60,
)
from eval_multi_seed_ensemble import legal_cells_60


KS = [1, 3, 5, 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True,
                    help='pattern_simple_direct_H*_playedeven.pt')
    ap.add_argument('--hidden', type=int, required=True)
    ap.add_argument('--num-games', type=int, default=500)
    ap.add_argument('--k-min', type=int, default=5)
    ap.add_argument('--k-max', type=int, default=53)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=512)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device)
    input_dim = ckpt.get('input_dim', 120)
    n_patterns = ckpt.get('n_patterns', 960)
    me = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    mo = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even'])
    mo.load_state_dict(ckpt['odd'])
    me.eval(); mo.eval()
    print(f"  H={args.hidden}, input_dim={input_dim}")

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    print(f"Building test set from {args.num_games} val games...")
    games = load_val_games(args.data_dir, args.num_data_files)[:args.num_games]
    feats_list, ks_list, legal_list = [], [], []
    for game in games:
        for k in range(args.k_min, args.k_max + 1):
            legal = legal_cells_60(game, k)
            if legal is None or not legal:
                continue
            feats_list.append(played_even_features(game[:k]))
            ks_list.append(k)
            legal_list.append(legal)
    n_total = len(feats_list)
    print(f"  {n_total} positions")

    legal_mask = np.zeros((n_total, 60), dtype=bool)
    for i, lg in enumerate(legal_list):
        for c in lg:
            legal_mask[i, c] = True

    hits = {K: 0.0 for K in KS}
    max_K = max(KS)

    t0 = time.time()
    with torch.no_grad():
        for bstart in range(0, n_total, args.batch_size):
            bend = min(bstart + args.batch_size, n_total)
            x = torch.stack(feats_list[bstart:bend]).to(device)
            ks = torch.tensor(ks_list[bstart:bend], device=device)
            # Route by parity.  MLPs were trained with even parity of
            # positions -> me.  For val games, `k` is "moves played
            # before this position" - so odd k means we're about to play
            # move k+1 which is even parity, hence me.  Same convention
            # as eval_multi_seed_ensemble.py.
            use_me = (ks % 2 == 1); use_mo = ~use_me
            B = bend - bstart
            logits = torch.zeros(B, 960, device=device)
            if use_me.any():
                logits[use_me] = me(x[use_me])
            if use_mo.any():
                logits[use_mo] = mo(x[use_mo])
            log1m = -F.softplus(logits)
            gathered = log1m[:, idx]
            gathered = gathered.masked_fill(~mask[None], 0.0)
            cell_scores = -gathered.sum(dim=-1)                       # (B, 60)

            legal_batch = torch.from_numpy(
                legal_mask[bstart:bend]).to(device)
            n_legal_pos = legal_batch.sum(dim=1).clamp(min=1).float()

            topk_idx = cell_scores.topk(max_K, dim=1).indices
            legal_at_topk = legal_batch.gather(1, topk_idx.to(torch.long))
            for K in KS:
                got = legal_at_topk[:, :K].sum(dim=1).float()
                denom = torch.minimum(torch.full_like(n_legal_pos, K),
                                        n_legal_pos)
                hits[K] += (got / denom).sum().item()

            if (bstart // args.batch_size) % 10 == 0:
                print(f"  {bend}/{n_total}  ({int(time.time()-t0)}s)",
                      flush=True)

    print()
    print(f"=== Single MLP top-K legality on {n_total:,} val positions ===")
    print(f"H={args.hidden}, ckpt={os.path.basename(args.ckpt)}")
    print(f"  {'K':<5}  {'acc (recall of achievable)':<25}")
    print("  " + "-" * 32)
    for K in KS:
        v = hits[K] / n_total
        print(f"  top-{K:<2}  {v:>10.4f}")


if __name__ == '__main__':
    main()
