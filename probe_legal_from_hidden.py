"""Legal-move probing on the hidden layers of the N-seed multi-seed ensemble.

Analogous to probe_multi_seed_hidden.py but targets legal cells instead of
board state.  Tests hypothesis A: "the ensemble's hidden reps contain more
information than any output-space aggregation can recover."

For each position, extract the (N, hidden) hidden-layer activations across all
N MLPs and train a probe to predict the 60-d legal-cell mask.  Compare the
probe's top-K legality against the best output-space aggregator.

Variants:
  1. concat     : concat of N * hidden = (N*hidden,) input, linear
  2. shared     : single Linear(hidden, 60) applied per seed, mean logits
  3. moe        : features -> softmax over N experts -> weighted hidden -> probe

Usage:
    python probe_legal_from_hidden.py \\
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


N_CELLS_60 = 60
TOP_KS = [1, 3, 5, 10]


def extract_hidden_and_legal(games, W1_e, b1_e, W1_o, b1_o, W2_e, b2_e,
                               W2_o, b2_o, device, k_min, k_max,
                               batch_size, N):
    """Return dict of:
      hidden:   (n_pos, N, hidden)  float16
      features: (n_pos, 120)         float32
      legal:    (n_pos, 60)          bool
    Skips positions with no legal moves.
    """
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
    hidden_dim = W1_e.shape[2]

    features = torch.stack(feats_list).numpy().astype(np.float32)
    legal_mask = np.zeros((n_total, N_CELLS_60), dtype=bool)
    for i, legal in enumerate(legal_list):
        for c in legal:
            legal_mask[i, c] = True

    # Store hidden as float16 to halve host memory.
    hidden = np.zeros((n_total, N, hidden_dim), dtype=np.float16)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            end = min(i + batch_size, n_total)
            x = torch.stack(feats_list[i:end]).to(device)
            ks = torch.tensor(ks_list[i:end], device=device)
            use_me = (ks % 2 == 1)
            use_mo = ~use_me
            B = end - i

            h_all = torch.zeros(N, B, hidden_dim, device=device)

            def forward_hidden(W1, b1, x_slice):
                x_nbi = x_slice.unsqueeze(0).expand(N, -1, -1)
                h = torch.bmm(x_nbi, W1) + b1
                return F.relu(h)

            if use_me.any():
                h_all[:, use_me] = forward_hidden(W1_e, b1_e, x[use_me])
            if use_mo.any():
                h_all[:, use_mo] = forward_hidden(W1_o, b1_o, x[use_mo])

            hidden[i:end] = h_all.permute(1, 0, 2).cpu().numpy().astype(np.float16)
            if (i // batch_size) % 20 == 0:
                print(f"    {end}/{n_total}  ({int(time.time()-t0)}s)",
                      flush=True)

    return {
        'hidden':   hidden,        # (n, N, hidden)
        'features': features,      # (n, 120)
        'legal':    legal_mask,    # (n, 60) bool
    }


def get_vectorized_weights(me_module, mo_module):
    """Extract raw stacked weights from two VectorizedMLPs.  W2/b2 unused."""
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


class ConcatProbe(nn.Module):
    """Linear (N * hidden) -> 60 legal-cell logits."""
    def __init__(self, n_seeds, hidden_dim):
        super().__init__()
        self.probe = nn.Linear(n_seeds * hidden_dim, N_CELLS_60)

    def forward(self, hidden, features):
        return self.probe(hidden.flatten(1))


class SharedProbe(nn.Module):
    """One Linear(hidden, 60) applied per seed, mean logits.
    O(1) params in N."""
    def __init__(self, n_seeds, hidden_dim):
        super().__init__()
        self.probe = nn.Linear(hidden_dim, N_CELLS_60)

    def forward(self, hidden, features):
        B, N, H = hidden.shape
        per_seed = self.probe(hidden.reshape(B * N, H)).reshape(B, N, N_CELLS_60)
        return per_seed.mean(dim=1)


class MoEProbe(nn.Module):
    """Features -> softmax over N experts, weighted-sum hidden, linear probe."""
    def __init__(self, n_seeds, hidden_dim, feature_dim=120, gate_hidden=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, n_seeds),
        )
        self.probe = nn.Linear(hidden_dim, N_CELLS_60)

    def forward(self, hidden, features):
        gate = F.softmax(self.gate(features), dim=1)                # (B, N)
        weighted = (gate.unsqueeze(-1) * hidden).sum(dim=1)          # (B, H)
        return self.probe(weighted)


def topk_legality(logits, legal_mask, k):
    """logits: (n, 60) tensor  legal_mask: (n, 60) bool numpy"""
    topk_idx = logits.topk(k, dim=1).indices.cpu().numpy()
    n = topk_idx.shape[0]
    hits = legal_mask[np.arange(n)[:, None], topk_idx]
    return hits.mean()


def train_probe(model, dsets, device, epochs, batch_size, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.75, patience=1)

    train, test = dsets['train'], dsets['test']
    n_train = train['hidden'].shape[0]
    n_test  = test['hidden'].shape[0]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {n_params:,}")

    def to_gpu_batch(dset, idxs):
        h = torch.from_numpy(dset['hidden'][idxs].astype(np.float32)).to(device)
        f = torch.from_numpy(dset['features'][idxs]).to(device)
        y = torch.from_numpy(dset['legal'][idxs].astype(np.float32)).to(device)
        return h, f, y

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss = 0.0
        t0 = time.time()
        for i in range(0, n_train, batch_size):
            idxs = perm[i:i + batch_size]
            h, f, y = to_gpu_batch(train, idxs)
            logits = model(h, f)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idxs)
        epoch_loss = total_loss / n_train
        scheduler.step(epoch_loss)
        cur_lr = opt.param_groups[0]['lr']
        print(f"    epoch {epoch}: loss={epoch_loss:.4f}  "
              f"lr={cur_lr:.2e}  ({int(time.time()-t0)}s)", flush=True)

    # Evaluate top-K legality on test.
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            end = min(i + batch_size, n_test)
            idxs = np.arange(i, end)
            h, f, _ = to_gpu_batch(test, idxs)
            all_logits.append(model(h, f))
    all_logits = torch.cat(all_logits, dim=0)  # (n_test, 60)
    results = {K: topk_legality(all_logits, test['legal'], K) for K in TOP_KS}
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
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num-seeds-used', type=int, default=None,
                    help='If set, use only the first N seeds.')
    ap.add_argument('--variants', type=str, default='concat,shared,moe',
                    help='concat,shared,moe or "all"')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading {args.multi_ckpt}")
    me, mo, N_total, hidden, input_dim = load_vectorized_from_multi(
        args.multi_ckpt, device)
    print(f"  N_total={N_total} seeds in ckpt, H={hidden}")

    (W1_e_all, b1_e_all, W2_e_all, b2_e_all,
     W1_o_all, b1_o_all, W2_o_all, b2_o_all) = get_vectorized_weights(me, mo)

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

    games = load_val_games(args.data_dir, args.num_data_files)
    train_games = games[:args.num_train_games]
    test_games  = games[args.num_train_games:
                        args.num_train_games + args.num_test_games]
    print(f"  train games: {len(train_games)}, test games: {len(test_games)}")

    print("Extracting hidden + legal for train...")
    train_dset = extract_hidden_and_legal(
        train_games, W1_e, b1_e, W1_o, b1_o, W2_e, b2_e, W2_o, b2_o,
        device, args.k_min, args.k_max, args.batch_size, N,
    )
    print(f"  {train_dset['hidden'].shape[0]:,} train positions")

    print("Extracting hidden + legal for test...")
    test_dset = extract_hidden_and_legal(
        test_games, W1_e, b1_e, W1_o, b1_o, W2_e, b2_e, W2_o, b2_o,
        device, args.k_min, args.k_max, args.batch_size, N,
    )
    print(f"  {test_dset['hidden'].shape[0]:,} test positions")

    dsets = {'train': train_dset, 'test': test_dset}
    results = {}

    all_variants = ['concat', 'shared', 'moe']
    if args.variants == 'all':
        variants_to_run = all_variants
    else:
        variants_to_run = [v.strip() for v in args.variants.split(',')]

    if 'concat' in variants_to_run:
        print(f"\n=== Probe: concat (linear) ===")
        model = ConcatProbe(N, hidden).to(device)
        results['concat'] = train_probe(
            model, dsets, device, args.epochs, args.batch_size, args.lr)
        for K in TOP_KS:
            print(f"    top-{K} legality: {results['concat'][K]:.4f}")

    if 'shared' in variants_to_run:
        print(f"\n=== Probe: shared (per-seed probe + mean logits) ===")
        model = SharedProbe(N, hidden).to(device)
        results['shared'] = train_probe(
            model, dsets, device, args.epochs, args.batch_size, args.lr)
        for K in TOP_KS:
            print(f"    top-{K} legality: {results['shared'][K]:.4f}")

    if 'moe' in variants_to_run:
        print(f"\n=== Probe: MoE gate ===")
        model = MoEProbe(N, hidden).to(device)
        results['moe'] = train_probe(
            model, dsets, device, args.epochs, args.batch_size, args.lr)
        for K in TOP_KS:
            print(f"    top-{K} legality: {results['moe'][K]:.4f}")

    print()
    print(f"=== Legal-move probing ({test_dset['hidden'].shape[0]:,} test positions) ===")
    header = f"  {'variant':<12}" + "".join(f"  top-{K:>2}" for K in TOP_KS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for v, r in results.items():
        row = f"  {v:<12}"
        for K in TOP_KS:
            row += f"  {r[K]:>6.4f}"
        print(row)


if __name__ == '__main__':
    main()
