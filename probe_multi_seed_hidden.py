"""Board-state probing on the hidden layers of the N-seed multi-seed ensemble.

For each position, extract the (N, 512) hidden-layer activations across all
N MLPs.  Train a linear probe from various features to the 64-d board state
(3 classes per cell: empty / black / white).

Variants:
  1. concat        : concat of N × 512 hidden = (N*512,) input
  2. mean          : mean of N × 512 hidden = (512,) input
  3. concat + board features: (N*512 + 120,) input
  4. concat + per-MLP confidence (max cell score per MLP): (N*512 + N,)
  5. concat + cross-MLP cell agreement (std of cell scores across MLPs, per cell): (N*512 + 60,)
  6. MoE gate: features -> softmax over N experts -> weighted sum of hidden (512,) -> probe

Metric: per-cell 3-class accuracy, averaged over 64 cells.

Usage:
    python probe_multi_seed_hidden.py \\
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
from eval_multi_seed_ensemble import load_vectorized_from_multi


N_CELLS = 64
N_CLASSES = 3   # empty (0), black (+1 → 1), white (-1 → 2)


def board_state_target(game, k):
    """Return 64-d int array of board state after k moves.
    0=empty, 1=black, 2=white. Returns None if the game is malformed."""
    board = OthelloBoardState()
    for c in game[:k]:
        try:
            board.umpire(c)
        except Exception:
            return None
    flat = np.asarray(board.state).flatten()
    out = np.zeros(64, dtype=np.int64)
    out[flat == 1] = 1
    out[flat == -1] = 2
    return out


def extract_hidden_and_scores(games, W1_e, b1_e, W1_o, b1_o, W2_e, b2_e,
                                W2_o, b2_o, idx, mask, device, k_min, k_max,
                                batch_size, N):
    """Return dict of:
      hidden: (n_pos, N, 512) — post-ReLU
      cell_scores: (n_pos, N, 60)
      features: (n_pos, 120)   — played+even
      board:  (n_pos, 64)      — target
    """
    feats_list, ks_list, board_list = [], [], []
    for game in games:
        for k in range(k_min, k_max + 1):
            board_target = board_state_target(game, k)
            if board_target is None:
                continue
            feats_list.append(played_even_features(game[:k]))
            ks_list.append(k)
            board_list.append(board_target)
    n_total = len(feats_list)
    hidden_dim = W1_e.shape[2]

    features = torch.stack(feats_list).numpy().astype(np.float32)  # (n, 120)
    board = np.stack(board_list)                                    # (n, 64)
    ks_arr = np.array(ks_list, dtype=np.int64)

    # Store as float16 to halve host memory (49GB -> 24GB for 245K positions).
    # Cast back to float32 per-batch inside probe training.
    hidden = np.zeros((n_total, N, hidden_dim), dtype=np.float16)
    cell_scores = np.zeros((n_total, N, 60), dtype=np.float16)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            end = min(i + batch_size, n_total)
            x = torch.stack(feats_list[i:end]).to(device)          # (B, 120)
            ks = torch.tensor(ks_list[i:end], device=device)
            use_me = (ks % 2 == 1)
            use_mo = ~use_me
            B = end - i

            # We need per-model hidden. Instead of calling model(x),
            # do the two-layer forward manually so we can capture h.
            h_all = torch.zeros(N, B, hidden_dim, device=device)
            logits_all = torch.zeros(N, B, 960, device=device)

            def forward_slice(W1, b1, W2, b2, x_slice):
                x_nbi = x_slice.unsqueeze(0).expand(N, -1, -1)
                h = torch.bmm(x_nbi, W1) + b1                       # (N, B, H)
                h = F.relu(h)
                y = torch.bmm(h, W2) + b2                            # (N, B, 960)
                return h, y

            if use_me.any():
                x_me = x[use_me]
                h_me, y_me = forward_slice(W1_e, b1_e, W2_e, b2_e, x_me)
                h_all[:, use_me] = h_me
                logits_all[:, use_me] = y_me
            if use_mo.any():
                x_mo = x[use_mo]
                h_mo, y_mo = forward_slice(W1_o, b1_o, W2_o, b2_o, x_mo)
                h_all[:, use_mo] = h_mo
                logits_all[:, use_mo] = y_mo

            # cell scores via prob_or aggregation from logits
            log1m = -F.softplus(logits_all)
            gathered = log1m[:, :, idx]
            gathered = gathered.masked_fill(~mask[None, None], 0.0)
            scores = -gathered.sum(dim=-1)                          # (N, B, 60)

            hidden[i:end]      = h_all.permute(1, 0, 2).cpu().numpy().astype(np.float16)
            cell_scores[i:end] = scores.permute(1, 0, 2).cpu().numpy().astype(np.float16)
            if (i // batch_size) % 20 == 0:
                print(f"    {end}/{n_total}  ({int(time.time()-t0)}s)",
                      flush=True)

    return {
        'hidden':      hidden,        # (n, N, 512)
        'cell_scores': cell_scores,   # (n, N, 60)
        'features':    features,      # (n, 120)
        'board':       board,         # (n, 64)
    }


def get_vectorized_weights(me_module, mo_module):
    """Extract raw stacked weights from two VectorizedMLPs."""
    return (
        me_module.W1.detach(),
        me_module.b1.detach(),
        me_module.W2.detach(),
        me_module.b2.detach(),
        mo_module.W1.detach(),
        mo_module.b1.detach(),
        mo_module.W2.detach(),
        mo_module.b2.detach(),
    )


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, N_CELLS * N_CLASSES)

    def forward(self, x):
        # x: (B, input_dim) -> (B, 64, 3)
        return self.linear(x).view(-1, N_CELLS, N_CLASSES)


class MoEProbe(nn.Module):
    """Features -> softmax over N experts, weighted sum of hidden (512,),
    linear probe."""
    def __init__(self, n_seeds, hidden_dim, feature_dim=120, gate_hidden=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, n_seeds),
        )
        self.probe = nn.Linear(hidden_dim, N_CELLS * N_CLASSES)

    def forward(self, hidden_tensor, features):
        # hidden_tensor: (B, N, hidden_dim); features: (B, 120)
        gate = F.softmax(self.gate(features), dim=1)                # (B, N)
        weighted = (gate.unsqueeze(-1) * hidden_tensor).sum(dim=1)   # (B, hidden)
        return self.probe(weighted).view(-1, N_CELLS, N_CLASSES)


def build_variant_features(dset, variant, N, hidden_dim):
    """Return (train_input, test_input) tensors for a given variant.
    dset: {'train': {...}, 'test': {...}} with 'hidden', 'cell_scores', 'features'."""
    train = dset['train']
    test  = dset['test']

    def compute(h, cs, f):
        # h: (n, N, 512) float16   cs: (n, N, 60) float16   f: (n, 120) float32
        # Cast to float32 for downstream training math
        h32 = h.astype(np.float32)
        cs32 = cs.astype(np.float32)
        if variant == 'concat':
            return h32.reshape(h32.shape[0], -1)                     # (n, N*512)
        if variant == 'mean':
            return h32.mean(axis=1)                                   # (n, 512)
        if variant == 'concat+features':
            return np.concatenate([h32.reshape(h32.shape[0], -1), f], axis=1)
        if variant == 'concat+confidence':
            conf = cs32.max(axis=2)                                   # (n, N)
            return np.concatenate([h32.reshape(h32.shape[0], -1), conf], axis=1)
        if variant == 'concat+agreement':
            agree = cs32.std(axis=1)                                  # (n, 60) — std across MLPs per cell
            return np.concatenate([h32.reshape(h32.shape[0], -1), agree], axis=1)
        raise ValueError(variant)

    x_train = compute(train['hidden'], train['cell_scores'], train['features'])
    x_test  = compute(test['hidden'],  test['cell_scores'],  test['features'])
    return x_train, x_test


def train_probe(model, x_train, y_train_board, x_test, y_test_board,
                 device, epochs, batch_size, lr, extra_train=None,
                 extra_test=None):
    """Train the probe with cross-entropy per cell."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train_board).to(device)
    x_test_t  = torch.from_numpy(x_test).to(device)
    y_test_t  = torch.from_numpy(y_test_board).to(device)

    if extra_train is not None:
        extra_train_t = torch.from_numpy(extra_train).to(device)
    if extra_test is not None:
        extra_test_t = torch.from_numpy(extra_test).to(device)

    n_train = x_train_t.shape[0]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {n_params:,}")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idxs = perm[i:i + batch_size]
            if extra_train is not None:
                logits = model(x_train_t[idxs], extra_train_t[idxs])
            else:
                logits = model(x_train_t[idxs])
            loss = F.cross_entropy(
                logits.view(-1, N_CLASSES),
                y_train_t[idxs].view(-1),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idxs)

    model.eval()
    with torch.no_grad():
        if extra_test is not None:
            test_logits = model(x_test_t, extra_test_t)
        else:
            test_logits = model(x_test_t)
        pred = test_logits.argmax(dim=-1)                            # (n, 64)
        acc = (pred == y_test_t).float().mean().item()
    return acc


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
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading {args.multi_ckpt}")
    me, mo, N, hidden, input_dim = load_vectorized_from_multi(
        args.multi_ckpt, device)
    print(f"  N={N} seeds, H={hidden}")

    W1_e, b1_e, W2_e, b2_e, W1_o, b1_o, W2_o, b2_o = \
        get_vectorized_weights(me, mo)

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    games = load_val_games(args.data_dir, args.num_data_files)
    train_games = games[:args.num_train_games]
    test_games  = games[args.num_train_games:
                        args.num_train_games + args.num_test_games]
    print(f"  train games: {len(train_games)}, test games: {len(test_games)}")

    print("Extracting hidden + cell_scores for train...")
    train_dset = extract_hidden_and_scores(
        train_games, W1_e, b1_e, W1_o, b1_o, W2_e, b2_e, W2_o, b2_o,
        idx, mask, device, args.k_min, args.k_max, args.batch_size, N,
    )
    print(f"  {train_dset['hidden'].shape[0]:,} train positions")

    print("Extracting hidden + cell_scores for test...")
    test_dset = extract_hidden_and_scores(
        test_games, W1_e, b1_e, W1_o, b1_o, W2_e, b2_e, W2_o, b2_o,
        idx, mask, device, args.k_min, args.k_max, args.batch_size, N,
    )
    print(f"  {test_dset['hidden'].shape[0]:,} test positions")

    dsets = {'train': train_dset, 'test': test_dset}
    results = {}

    variants = ['concat', 'mean', 'concat+features',
                'concat+confidence', 'concat+agreement']
    for v in variants:
        print(f"\n=== Probe: {v} ===")
        x_tr, x_te = build_variant_features(dsets, v, N, hidden)
        print(f"    input_dim={x_tr.shape[1]}")
        input_dim = x_tr.shape[1]
        probe = LinearProbe(input_dim)
        acc = train_probe(
            probe, x_tr,
            train_dset['board'],
            x_te,
            test_dset['board'],
            device, args.epochs, args.batch_size, args.lr,
        )
        print(f"    test 3-class per-cell accuracy: {acc:.4f}")
        results[v] = acc

    # MoE probe: use hidden tensor (n, N, 512) directly + features
    print(f"\n=== Probe: MoE gate ===")
    moe = MoEProbe(N, hidden)
    acc = train_probe(
        moe,
        train_dset['hidden'],
        train_dset['board'],
        test_dset['hidden'],
        test_dset['board'],
        device, args.epochs, args.batch_size, args.lr,
        extra_train=train_dset['features'],
        extra_test=test_dset['features'],
    )
    print(f"    test 3-class per-cell accuracy: {acc:.4f}")
    results['moe_gate'] = acc

    print()
    print(f"=== Probing results ({test_dset['hidden'].shape[0]:,} test positions) ===")
    for v, a in results.items():
        print(f"  {v:<25}  {a:.4f}")


if __name__ == '__main__':
    main()
