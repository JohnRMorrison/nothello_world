"""Learned readout for the N-seed multi-seed MLP ensemble.

Trains a small MLP that maps the N seeds' cell scores to per-cell legality
logits, then reports top-K legality (K = 1, 3, 5, 10) on a held-out set.

  Architecture:
      MLP:  [N × 60 cell scores] -> hidden (default 128) -> 60 output
      Loss: BCE with pos_weight from the training legal_rate.

Usage:
    python train_multi_seed_readout.py \\
        --multi-ckpt experiments/.../multi_seed_N100_H512_playedeven.pt \\
        --num-train-games 5000 --num-test-games 500
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from train_multi_seed_mlp import VectorizedMLP
from train_pattern_simple import _get_cell_pat_index
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from compare_v4_vs_mlp import (
    load_val_games, played_even_features,
)
from eval_multi_seed_ensemble import (
    load_vectorized_from_multi, legal_cells_60,
)


TOP_KS = [1, 3, 5, 10]


class ReadoutMLP(nn.Module):
    """Readout: [N × 60] -> hidden -> 60."""
    def __init__(self, n_seeds, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_seeds * 60, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 60),
        )

    def forward(self, scores):
        # scores: (B, N, 60)
        B = scores.shape[0]
        return self.net(scores.reshape(B, -1))


def build_dataset(games, mlp_bundle_me, mlp_bundle_mo, idx, mask, N, device,
                   k_min, k_max, batch_size=1024):
    """Return: scores (n_pos, N, 60), legal (n_pos, 60)."""
    feats_list, ks_list, legal_list = [], [], []
    for game in games:
        for k in range(k_min, k_max + 1):
            legal = legal_cells_60(game, k)
            if legal is None or not legal:
                continue
            feats_list.append(played_even_features(game[:k]))
            ks_list.append(k)
            legal_list.append(legal)
    n_total = len(feats_list)

    legal_mask = np.zeros((n_total, 60), dtype=bool)
    for i, legal in enumerate(legal_list):
        for c in legal:
            legal_mask[i, c] = True

    # MLP forward pass to get per-model cell scores
    scores = np.zeros((n_total, N, 60), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            end = min(i + batch_size, n_total)
            x = torch.stack(feats_list[i:end]).to(device)
            ks_t = torch.tensor(ks_list[i:end], device=device)
            use_me = (ks_t % 2 == 1)
            use_mo = ~use_me
            B = end - i
            logits = torch.zeros(N, B, 960, device=device)
            if use_me.any():
                logits[:, use_me] = mlp_bundle_me(x[use_me])
            if use_mo.any():
                logits[:, use_mo] = mlp_bundle_mo(x[use_mo])
            log1m = -F.softplus(logits)
            gathered = log1m[:, :, idx]
            gathered = gathered.masked_fill(~mask[None, None], 0.0)
            cell_scores = -gathered.sum(dim=-1)                # (N, B, 60)
            # Move to (B, N, 60) layout for downstream
            scores[i:end, :, :] = cell_scores.permute(1, 0, 2).cpu().numpy()
            if (i // batch_size) % 20 == 0:
                print(f"    {end}/{n_total}  ({int(time.time()-t0)}s)",
                      flush=True)

    return scores, legal_mask


def topk_legality_torch(logits, legal_mask_np, k):
    """logits: (n_pos, 60) tensor.  legal_mask_np: (n_pos, 60) bool."""
    topk_idx = logits.topk(k, dim=1).indices.cpu().numpy()  # (n_pos, k)
    n_pos = topk_idx.shape[0]
    hits = legal_mask_np[np.arange(n_pos)[:, None], topk_idx]
    return hits.mean()


def train_and_eval(args, train_scores, train_legal, test_scores, test_legal,
                    device):
    print(f"\n=== Training readout ===")
    N = train_scores.shape[1]
    model = ReadoutMLP(N, hidden=args.readout_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_train = train_scores.shape[0]

    # Keep training tensors PINNED on CPU and move per-batch to GPU.
    # Full N=100, H=512, 5M positions is ~120 GiB — exceeds any single GPU.
    train_scores_cpu = torch.from_numpy(train_scores).pin_memory()
    train_legal_cpu  = torch.from_numpy(train_legal.astype(np.float32)).pin_memory()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}   hidden={args.readout_hidden}")

    legal_rate = train_legal.mean()
    pos_weight = torch.tensor([(1 - legal_rate) / max(legal_rate, 1e-6)],
                               device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)              # CPU perm
        total_loss = 0.0
        for i in range(0, n_train, args.batch_size):
            idxs = perm[i:i + args.batch_size]
            x = train_scores_cpu[idxs].to(device, non_blocking=True)
            y = train_legal_cpu[idxs].to(device, non_blocking=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idxs)
        avg_loss = total_loss / n_train

        # Eval — batch through the test set too (test is only ~10x smaller,
        # but for consistency and to keep memory low we still batch).
        model.eval()
        test_scores_cpu = torch.from_numpy(test_scores)
        n_test = test_scores.shape[0]
        test_logits_chunks = []
        with torch.no_grad():
            for i in range(0, n_test, args.batch_size):
                x = test_scores_cpu[i:i + args.batch_size].to(device,
                                                                non_blocking=True)
                test_logits_chunks.append(model(x).cpu())
        test_logits = torch.cat(test_logits_chunks, dim=0)
        results = {K: topk_legality_torch(test_logits, test_legal, K)
                   for K in TOP_KS}
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}  " +
              "  ".join(f"top-{K}={results[K]:.4f}" for K in TOP_KS),
              flush=True)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi-ckpt', required=True)
    ap.add_argument('--num-train-games', type=int, default=5000)
    ap.add_argument('--num-test-games', type=int, default=500)
    ap.add_argument('--k-min', type=int, default=5)
    ap.add_argument('--k-max', type=int, default=53)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--readout-hidden', type=int, default=128)
    ap.add_argument('--num-seeds-used', type=int, default=None,
                    help='If set, use only the first N seeds from the checkpoint.')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading {args.multi_ckpt}")
    me, mo, N, hidden, input_dim = load_vectorized_from_multi(
        args.multi_ckpt, device)
    print(f"  Loaded N={N} seeds, H={hidden}")
    if args.num_seeds_used is not None and args.num_seeds_used < N:
        k = args.num_seeds_used
        with torch.no_grad():
            for m in (me, mo):
                m.n_models = k
                m.W1 = torch.nn.Parameter(m.W1.data[:k].clone())
                m.b1 = torch.nn.Parameter(m.b1.data[:k].clone())
                m.W2 = torch.nn.Parameter(m.W2.data[:k].clone())
                m.b2 = torch.nn.Parameter(m.b2.data[:k].clone())
        N = k
        print(f"  Sliced to first {N} seeds")

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    print(f"Loading games...")
    games = load_val_games(args.data_dir, args.num_data_files)
    train_games = games[:args.num_train_games]
    test_games  = games[args.num_train_games:
                        args.num_train_games + args.num_test_games]
    print(f"  train={len(train_games)}  test={len(test_games)}")

    print("Building train set (MLP forward)...")
    train_scores, train_legal = build_dataset(
        train_games, me, mo, idx, mask, N, device,
        args.k_min, args.k_max, args.batch_size,
    )
    print(f"  {train_scores.shape[0]:,} training positions")

    print("Building test set...")
    test_scores, test_legal = build_dataset(
        test_games, me, mo, idx, mask, N, device,
        args.k_min, args.k_max, args.batch_size,
    )
    print(f"  {test_scores.shape[0]:,} test positions")

    results = train_and_eval(args, train_scores, train_legal,
                              test_scores, test_legal, device)

    print()
    print(f"=== Learned readout results ({test_scores.shape[0]:,} test positions) ===")
    for K in TOP_KS:
        print(f"  top-{K:<3}  {results[K]:.4f}")


if __name__ == '__main__':
    main()
