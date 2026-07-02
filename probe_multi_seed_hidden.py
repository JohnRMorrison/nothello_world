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
  7. shared: single Linear(512, 60×3) applied to each seed's hidden separately,
     then mean per-seed logits.  ~92K params for any N.
  8. shared_attn: shared per-seed probe as in (7), weighted by feature-conditioned
     attention over seeds.  ~100K params for any N.

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


class NonLinearProbe(nn.Module):
    """MLP readout: input -> hidden -> 64×3. Can learn multiplicative
    interactions (gating) between the concat and any auxiliary features."""
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_CELLS * N_CLASSES),
        )

    def forward(self, x):
        return self.net(x).view(-1, N_CELLS, N_CLASSES)


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


class SharedProbe(nn.Module):
    """Weight-shared per-seed probe.  Apply the same Linear(H, 60×3) to every
    seed's hidden vector, then mean the per-seed logits.  Parameter count is
    independent of N."""
    def __init__(self, hidden_dim, feature_dim=120):
        super().__init__()
        self.probe = nn.Linear(hidden_dim, N_CELLS * N_CLASSES)

    def forward(self, hidden_tensor, features):
        # hidden_tensor: (B, N, H); features unused
        B, N, H = hidden_tensor.shape
        per_seed = self.probe(hidden_tensor.reshape(B * N, H))
        per_seed = per_seed.reshape(B, N, N_CELLS, N_CLASSES)
        return per_seed.mean(dim=1)


class AttentionSharedProbe(nn.Module):
    """Shared per-seed probe combined by feature-conditioned attention over
    seeds (not over the hidden dimension).  Also O(1) params in N."""
    def __init__(self, hidden_dim, feature_dim=120, attn_dim=32):
        super().__init__()
        self.probe = nn.Linear(hidden_dim, N_CELLS * N_CLASSES)
        self.attn_key = nn.Linear(hidden_dim, attn_dim)
        self.attn_query = nn.Linear(feature_dim, attn_dim)
        self.attn_dim = attn_dim

    def forward(self, hidden_tensor, features):
        B, N, H = hidden_tensor.shape
        per_seed = self.probe(hidden_tensor.reshape(B * N, H))
        per_seed = per_seed.reshape(B, N, N_CELLS, N_CLASSES)
        keys = self.attn_key(hidden_tensor)                             # (B, N, D)
        query = self.attn_query(features).unsqueeze(1)                  # (B, 1, D)
        scores = (keys * query).sum(-1) / (self.attn_dim ** 0.5)        # (B, N)
        weights = F.softmax(scores, dim=1)                              # (B, N)
        return (weights.unsqueeze(-1).unsqueeze(-1) * per_seed).sum(dim=1)


def compute_variant_batch(h_batch, cs_batch, f_batch, variant):
    """Compute variant features for a batch on GPU.

    h_batch: (B, N, 512) float32   cs_batch: (B, N, 60) float32   f_batch: (B, 120)
    Returns (B, input_dim) float32 tensor.
    """
    if variant == 'concat':
        return h_batch.reshape(h_batch.shape[0], -1)
    if variant == 'mean':
        return h_batch.mean(dim=1)
    if variant == 'concat+features':
        return torch.cat([h_batch.reshape(h_batch.shape[0], -1), f_batch], dim=1)
    if variant == 'concat+confidence':
        conf = cs_batch.max(dim=2).values                            # (B, N)
        return torch.cat([h_batch.reshape(h_batch.shape[0], -1), conf], dim=1)
    if variant == 'concat+agreement':
        agree = cs_batch.std(dim=1)                                   # (B, 60)
        return torch.cat([h_batch.reshape(h_batch.shape[0], -1), agree], dim=1)
    raise ValueError(variant)


def variant_input_dim(variant, N, hidden_dim, feature_dim=120):
    if variant == 'concat':
        return N * hidden_dim
    if variant == 'mean':
        return hidden_dim
    if variant == 'concat+features':
        return N * hidden_dim + feature_dim
    if variant == 'concat+confidence':
        return N * hidden_dim + N
    if variant == 'concat+agreement':
        return N * hidden_dim + 60
    raise ValueError(variant)


def train_probe_lazy(model_cls, input_dim, variant, dsets, N, hidden_dim,
                      device, epochs, batch_size, lr):
    """Train a probe by computing variant features per-batch (memory efficient).

    dsets: dict with 'train' and 'test' each containing float16 numpy arrays
       for 'hidden', 'cell_scores', 'features', and int64 for 'board'.
    """
    model = model_cls(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train = dsets['train']
    test  = dsets['test']
    n_train = train['hidden'].shape[0]
    n_test  = test['hidden'].shape[0]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {n_params:,}")

    def to_gpu_batch(dset, idxs):
        # Cast float16 -> float32 only for the sliced batch
        h = torch.from_numpy(dset['hidden'][idxs].astype(np.float32)).to(device)
        cs = torch.from_numpy(dset['cell_scores'][idxs].astype(np.float32)).to(device)
        f = torch.from_numpy(dset['features'][idxs]).to(device)
        b = torch.from_numpy(dset['board'][idxs]).to(device)
        return h, cs, f, b

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss = 0.0
        t0 = time.time()
        for i in range(0, n_train, batch_size):
            idxs = perm[i:i + batch_size]
            h, cs, f, b = to_gpu_batch(train, idxs)
            x = compute_variant_batch(h, cs, f, variant)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, N_CLASSES), b.view(-1),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idxs)
        print(f"    epoch {epoch}: loss={total_loss/n_train:.4f}  "
              f"({int(time.time()-t0)}s)", flush=True)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            end = min(i + batch_size, n_test)
            idxs = np.arange(i, end)
            h, cs, f, b = to_gpu_batch(test, idxs)
            x = compute_variant_batch(h, cs, f, variant)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == b).sum().item()
            total   += b.numel()
    return correct / total


def train_stream_probe(model, dsets, device, epochs, batch_size, lr):
    """Train a probe that takes (hidden_tensor, features) inputs.  Used for
    MoE / shared / attention-shared probes."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train = dsets['train']
    test  = dsets['test']
    n_train = train['hidden'].shape[0]
    n_test  = test['hidden'].shape[0]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {n_params:,}")

    def to_gpu_batch(dset, idxs):
        h = torch.from_numpy(dset['hidden'][idxs].astype(np.float32)).to(device)
        f = torch.from_numpy(dset['features'][idxs]).to(device)
        b = torch.from_numpy(dset['board'][idxs]).to(device)
        return h, f, b

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss = 0.0
        t0 = time.time()
        for i in range(0, n_train, batch_size):
            idxs = perm[i:i + batch_size]
            h, f, b = to_gpu_batch(train, idxs)
            logits = model(h, f)
            loss = F.cross_entropy(
                logits.view(-1, N_CLASSES), b.view(-1),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idxs)
        print(f"    epoch {epoch}: loss={total_loss/n_train:.4f}  "
              f"({int(time.time()-t0)}s)", flush=True)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            end = min(i + batch_size, n_test)
            idxs = np.arange(i, end)
            h, f, b = to_gpu_batch(test, idxs)
            logits = model(h, f)
            pred = logits.argmax(dim=-1)
            correct += (pred == b).sum().item()
            total   += b.numel()
    return correct / total


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
    ap.add_argument('--num-seeds-used', type=int, default=None,
                    help='If set, use only the first N seeds from the '
                         'checkpoint (default: use all)')
    ap.add_argument('--variants', type=str, default='all',
                    help='Comma-separated list of variants to run '
                         '(concat,concat+features,concat+confidence,'
                         'concat+agreement,mean,nonlin_concat,'
                         'nonlin_concat+features,moe,shared,shared_attn) '
                         'or "all" (default)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading {args.multi_ckpt}")
    me, mo, N_total, hidden, input_dim = load_vectorized_from_multi(
        args.multi_ckpt, device)
    print(f"  N_total={N_total} seeds in ckpt, H={hidden}")

    W1_e_all, b1_e_all, W2_e_all, b2_e_all, \
        W1_o_all, b1_o_all, W2_o_all, b2_o_all = \
        get_vectorized_weights(me, mo)

    # Optionally use only the first N seeds
    if args.num_seeds_used is not None and args.num_seeds_used < N_total:
        N = args.num_seeds_used
        print(f"  Using only first {N} of {N_total} seeds")
        W1_e = W1_e_all[:N]; b1_e = b1_e_all[:N]
        W2_e = W2_e_all[:N]; b2_e = b2_e_all[:N]
        W1_o = W1_o_all[:N]; b1_o = b1_o_all[:N]
        W2_o = W2_o_all[:N]; b2_o = b2_o_all[:N]
    else:
        N = N_total
        W1_e, b1_e, W2_e, b2_e = W1_e_all, b1_e_all, W2_e_all, b2_e_all
        W1_o, b1_o, W2_o, b2_o = W1_o_all, b1_o_all, W2_o_all, b2_o_all
    print(f"  Effective N={N}")

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

    all_variants = ['concat', 'concat+features',
                    'concat+confidence', 'concat+agreement', 'mean',
                    'nonlin_concat', 'nonlin_concat+features',
                    'moe', 'shared', 'shared_attn']
    if args.variants == 'all':
        variants_to_run = all_variants
    else:
        variants_to_run = [v.strip() for v in args.variants.split(',')]

    linear_variants = ['concat', 'concat+features',
                        'concat+confidence', 'concat+agreement', 'mean']
    for v in linear_variants:
        if v not in variants_to_run:
            continue
        print(f"\n=== Probe: {v} (linear) ===")
        input_dim = variant_input_dim(v, N, hidden)
        print(f"    input_dim={input_dim}")
        acc = train_probe_lazy(
            LinearProbe, input_dim, v, dsets, N, hidden,
            device, args.epochs, args.batch_size, args.lr,
        )
        print(f"    test 3-class per-cell accuracy: {acc:.4f}")
        results[v] = acc

    for v_label, base_v in [('nonlin_concat', 'concat'),
                              ('nonlin_concat+features', 'concat+features')]:
        if v_label not in variants_to_run:
            continue
        print(f"\n=== Probe: {v_label} (non-linear MLP) ===")
        input_dim = variant_input_dim(base_v, N, hidden)
        print(f"    input_dim={input_dim}")
        acc = train_probe_lazy(
            NonLinearProbe, input_dim, base_v, dsets, N, hidden,
            device, args.epochs, args.batch_size, args.lr,
        )
        print(f"    test 3-class per-cell accuracy: {acc:.4f}")
        results[v_label] = acc

    if 'moe' in variants_to_run:
        print(f"\n=== Probe: MoE gate ===")
        model = MoEProbe(N, hidden).to(device)
        acc = train_stream_probe(
            model, dsets, device, args.epochs, args.batch_size, args.lr,
        )
        print(f"    test 3-class per-cell accuracy: {acc:.4f}")
        results['moe_gate'] = acc

    if 'shared' in variants_to_run:
        print(f"\n=== Probe: shared (per-seed probe + mean logits) ===")
        model = SharedProbe(hidden).to(device)
        acc = train_stream_probe(
            model, dsets, device, args.epochs, args.batch_size, args.lr,
        )
        print(f"    test 3-class per-cell accuracy: {acc:.4f}")
        results['shared'] = acc

    if 'shared_attn' in variants_to_run:
        print(f"\n=== Probe: shared_attn (per-seed probe + attention combiner) ===")
        model = AttentionSharedProbe(hidden).to(device)
        acc = train_stream_probe(
            model, dsets, device, args.epochs, args.batch_size, args.lr,
        )
        print(f"    test 3-class per-cell accuracy: {acc:.4f}")
        results['shared_attn'] = acc

    print()
    print(f"=== Probing results ({test_dset['hidden'].shape[0]:,} test positions) ===")
    for v, a in results.items():
        print(f"  {v:<25}  {a:.4f}")


if __name__ == '__main__':
    main()
