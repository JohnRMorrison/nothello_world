"""Learned readouts for the N-seed multi-seed MLP ensemble.

Trains three learned aggregators on the outputs of N MLPs and reports
top-K legality (K = 1, 3, 5, 10) on a held-out set.

  Variant A (readout, no features):
      MLP:  [N × 60 cell scores] -> hidden -> 60 output
      Trained with BCE against 60-d cell legal mask.

  Variant B (readout + features):
      MLP:  [N × 60 cell scores + 120 played+even features] -> hidden -> 60 output

  Variant C (mixture-of-experts gate):
      gate_net: [120 features] -> softmax over N experts
      output:   sum_n (gate_weight[n] * cell_scores[n]) (weighted sum, 60-d)
      No extra MLP; the gate is trained end-to-end with BCE against 60-d
      cell legal mask.  This variant tests whether MLPs specialize by
      position — if gate weights become non-uniform conditional on features,
      that's evidence of specialization.

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
    load_val_games, played_even_features, C64_TO_C60,
)
from data.othello import OthelloBoardState
from eval_multi_seed_ensemble import (
    load_vectorized_from_multi, legal_cells_60,
)


TOP_KS = [1, 3, 5, 10]


class VariantA(nn.Module):
    """Readout: [N × 60] -> hidden -> 60."""
    def __init__(self, n_seeds, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_seeds * 60, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 60),
        )

    def forward(self, scores, features):
        # scores: (B, N, 60)   features: (B, 120)   [features unused]
        B = scores.shape[0]
        x = scores.reshape(B, -1)
        return self.net(x)


class VariantB(nn.Module):
    """Readout + features: [N × 60 + 120] -> hidden -> 60."""
    def __init__(self, n_seeds, hidden=128, feature_dim=120):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_seeds * 60 + feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 60),
        )

    def forward(self, scores, features):
        B = scores.shape[0]
        x = torch.cat([scores.reshape(B, -1), features], dim=1)
        return self.net(x)


class VariantC(nn.Module):
    """Mixture-of-experts gate: features -> softmax over N -> weighted sum
    of the N cell-score vectors."""
    def __init__(self, n_seeds, gate_hidden=128, feature_dim=120):
        super().__init__()
        self.n_seeds = n_seeds
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, n_seeds),
        )

    def forward(self, scores, features):
        # scores: (B, N, 60)   features: (B, 120)
        gate_logits = self.gate(features)                # (B, N)
        gate = F.softmax(gate_logits, dim=1)              # (B, N)
        # Weighted sum: (B, N, 1) * (B, N, 60) -> (B, 60)
        weighted = (gate.unsqueeze(-1) * scores).sum(dim=1)
        return weighted


def build_dataset(games, mlp_bundle_me, mlp_bundle_mo, idx, mask, N, device,
                   k_min, k_max, batch_size=1024):
    """Return: scores (n_pos, N, 60), features (n_pos, 120),
       legal (n_pos, 60), ks (n_pos,)."""
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

    # Build features on host
    features_arr = torch.stack(feats_list).numpy().astype(np.float32)   # (n_total, 120)
    ks_arr = np.array(ks_list, dtype=np.int64)

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

    return scores, features_arr, legal_mask, ks_arr


def topk_legality_torch(logits, legal_mask_np, k):
    """logits: (n_pos, 60) tensor.  legal_mask_np: (n_pos, 60) bool."""
    topk_idx = logits.topk(k, dim=1).indices.cpu().numpy()  # (n_pos, k)
    n_pos = topk_idx.shape[0]
    hits = legal_mask_np[np.arange(n_pos)[:, None], topk_idx]
    return hits.mean()


def train_and_eval(model_cls, name, args, train_scores, train_feats,
                    train_legal, test_scores, test_feats, test_legal, device):
    print(f"\n=== Training {name} ===")
    N = train_scores.shape[1]
    model = model_cls(N).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_train = train_scores.shape[0]
    n_test  = test_scores.shape[0]

    # Move data to device
    train_scores_t = torch.from_numpy(train_scores).to(device)
    train_feats_t  = torch.from_numpy(train_feats).to(device)
    train_legal_t  = torch.from_numpy(train_legal.astype(np.float32)).to(device)
    test_scores_t  = torch.from_numpy(test_scores).to(device)
    test_feats_t   = torch.from_numpy(test_feats).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # BCE with pos_weight from legal_rate
    legal_rate = train_legal.mean()
    pos_weight = torch.tensor([(1 - legal_rate) / max(legal_rate, 1e-6)],
                               device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, args.batch_size):
            idxs = perm[i:i + args.batch_size]
            logits = model(train_scores_t[idxs], train_feats_t[idxs])
            loss = F.binary_cross_entropy_with_logits(
                logits, train_legal_t[idxs], pos_weight=pos_weight,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idxs)
        avg_loss = total_loss / n_train

        model.eval()
        with torch.no_grad():
            test_logits = model(test_scores_t, test_feats_t)
        results = {K: topk_legality_torch(test_logits, test_legal, K)
                   for K in TOP_KS}
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}  " +
              "  ".join(f"top-{K}={results[K]:.4f}" for K in TOP_KS),
              flush=True)

    return {K: topk_legality_torch(test_logits, test_legal, K) for K in TOP_KS}


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
    train_scores, train_feats, train_legal, _ = build_dataset(
        train_games, me, mo, idx, mask, N, device,
        args.k_min, args.k_max, args.batch_size,
    )
    print(f"  {train_scores.shape[0]:,} training positions")

    print("Building test set...")
    test_scores, test_feats, test_legal, _ = build_dataset(
        test_games, me, mo, idx, mask, N, device,
        args.k_min, args.k_max, args.batch_size,
    )
    print(f"  {test_scores.shape[0]:,} test positions")

    all_results = {}
    for cls, name in [
        (VariantA, "Variant A (readout, outputs only)"),
        (VariantB, "Variant B (readout + features)"),
        (VariantC, "Variant C (MoE gate, softmax over experts)"),
    ]:
        r = train_and_eval(cls, name, args, train_scores, train_feats,
                           train_legal, test_scores, test_feats,
                           test_legal, device)
        all_results[name] = r

    print()
    print(f"=== Learned readout results ({test_scores.shape[0]:,} test positions) ===")
    for K in TOP_KS:
        print()
        print(f"Top-{K} legality:")
        for name in all_results:
            print(f"  {name:<45}  {all_results[name][K]:.4f}")


if __name__ == '__main__':
    main()
