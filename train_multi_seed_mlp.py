"""Train N pattern-detector MLPs simultaneously by sharing data loading.

I/O dominates wall-clock (loading 40 chunks per epoch over NFS), so training
N models on the same data takes roughly the same wall-clock as training 1.

Architecture: maintain N independent (model_even, model_odd) MLP pairs.
For each batch, all N models forward+backward+step on the SAME shared data.
Plain nn.Linear modules (no einsum) so PyTorch's optimized matmul kernels
are used.

Output: single .pt file containing all N model states.  Use
split_multi_seed_ckpt.py to extract individual seed checkpoints compatible
with compare_mlp_seeds.py and load_mlp.

Usage:
    sbatch train_multi_seed_mlp.sh  (defaults to H=512 N=100)
    HIDDEN=4096 NUM_SEEDS=50 sbatch train_multi_seed_mlp.sh
"""
import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import compute_pattern_labels_batch, DirectMLP


N_PATTERNS = 960
_PATTERNS = enumerate_flanking_patterns()
_PAT_TARGETS, _PAT_TERMINALS, _PAT_OPP_CELLS, _PAT_OPP_MASK = (
    precompute_pattern_arrays(_PATTERNS)
)


def derive_pattern_labels(board_labels, positions, batch_size=200_000):
    """Compute 960-d pattern legality from 64-d board state, store as uint8."""
    n = len(board_labels)
    out = np.zeros((n, N_PATTERNS), dtype=np.uint8)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pat = compute_pattern_labels_batch(
            board_labels[start:end].astype(np.int8),
            positions[start:end].astype(np.int64),
            _PAT_TARGETS, _PAT_TERMINALS, _PAT_OPP_CELLS, _PAT_OPP_MASK,
        )
        out[start:end] = (pat > 0).astype(np.uint8)
    return out


def slice_played_even(features_180):
    return np.concatenate(
        [features_180[:, :60], features_180[:, 120:180]], axis=1
    )


class VectorizedMLP(nn.Module):
    """N independent MLPs stacked into one module.  Uses bmm (NOT einsum)
    so the matmuls map to cuBLAS kernels.

    Forward: x (B, in_dim) -> (N, B, out_dim)
    """
    def __init__(self, n_models, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.n_models = n_models
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        bound1 = 1.0 / math.sqrt(input_dim)
        bound2 = 1.0 / math.sqrt(hidden_dim)
        # Storage layout for bmm: (N, in_dim, hidden) so we can do
        # bmm((N, B, in), (N, in, hidden)) = (N, B, hidden).
        self.W1 = nn.Parameter(
            (torch.rand(n_models, input_dim, hidden_dim) * 2 - 1) * bound1
        )
        self.b1 = nn.Parameter(
            (torch.rand(n_models, 1, hidden_dim) * 2 - 1) * bound1
        )
        self.W2 = nn.Parameter(
            (torch.rand(n_models, hidden_dim, output_dim) * 2 - 1) * bound2
        )
        self.b2 = nn.Parameter(
            (torch.rand(n_models, 1, output_dim) * 2 - 1) * bound2
        )

    def forward(self, x):
        # x: (B, input_dim) -> (N, B, input_dim) by expand (no copy)
        N = self.n_models
        x_nbi = x.unsqueeze(0).expand(N, -1, -1)
        h = torch.bmm(x_nbi, self.W1) + self.b1     # (N, B, hidden)
        h = F.relu(h)
        y = torch.bmm(h, self.W2) + self.b2          # (N, B, output)
        return y

    def state_per_seed(self, seed):
        """Return a regular DirectMLP-compatible state_dict for one seed."""
        # DirectMLP layout: net.0.weight (out, in), net.0.bias (out,),
        # net.2.weight (out, hidden), net.2.bias (out,)
        return {
            'net.0.weight': self.W1[seed].t().detach().cpu().clone(),
            'net.0.bias':   self.b1[seed].squeeze(0).detach().cpu().clone(),
            'net.2.weight': self.W2[seed].t().detach().cpu().clone(),
            'net.2.bias':   self.b2[seed].squeeze(0).detach().cpu().clone(),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-seeds', type=int, default=100)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=8192)
    ap.add_argument('--chunk-dir',
                    default='experiments/mathematical_transformation_experiments/'
                            'heuristic_probe_results/feature_chunks')
    ap.add_argument('--chunk-glob', default='chunk_ext_*.npz')
    ap.add_argument('--output-dir',
                    default='experiments/mathematical_transformation_experiments/'
                            'heuristic_probe_results/pattern_detector_checkpoints/'
                            'multi_seed_preliminary')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pos-weight', type=float, default=None)
    ap.add_argument('--max-chunks', type=int, default=None,
                    help='If set, train on at most this many chunks per epoch '
                         '(useful when full pass would exceed SLURM limit)')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"N seeds: {args.num_seeds}", flush=True)
    print(f"Hidden: {args.hidden}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)

    INPUT_DIM = 120
    print(f"Building VectorizedMLP (N={args.num_seeds}) for even and odd parity...",
          flush=True)
    t_build = time.time()
    model_even = VectorizedMLP(args.num_seeds, INPUT_DIM, args.hidden,
                                 N_PATTERNS).to(device)
    model_odd = VectorizedMLP(args.num_seeds, INPUT_DIM, args.hidden,
                                N_PATTERNS).to(device)
    optimizer = torch.optim.Adam(
        list(model_even.parameters()) + list(model_odd.parameters()),
        lr=args.lr,
    )
    n_params = sum(p.numel() for p in model_even.parameters()) * 2
    print(f"Total params (both parities, all {args.num_seeds} seeds): {n_params:,}  "
          f"({int(time.time() - t_build)}s to build)", flush=True)

    chunks = sorted(glob.glob(os.path.join(args.chunk_dir, args.chunk_glob)))
    train_chunks = chunks[:-1]
    eval_chunk = chunks[-1]
    if args.max_chunks is not None:
        train_chunks = train_chunks[:args.max_chunks]
        print(f"Limiting to first {len(train_chunks)} training chunks",
              flush=True)
    print(f"Found {len(chunks)} chunks  Train={len(train_chunks)}  Eval=1",
          flush=True)

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    base_save_path = os.path.join(
        save_dir,
        f"multi_seed_N{args.num_seeds}_H{args.hidden}_playedeven.pt",
    )

    for epoch in range(1, args.epochs + 1):
        model_even.train()
        model_odd.train()
        rng = np.random.RandomState(epoch * 1000 + args.seed)
        chunk_order = rng.permutation(len(train_chunks))

        total_loss = 0.0
        total_n = 0
        t_epoch_start = time.time()

        for ci_idx, ci in enumerate(chunk_order):
            cp = train_chunks[ci]
            t0 = time.time()
            with np.load(cp) as z:
                feats_180 = z['features'].astype(np.float16)
                board_labels = z['labels'].astype(np.int8)
                positions = z['positions'].astype(np.int64)
            feats_120 = slice_played_even(feats_180)
            t_load = time.time() - t0
            print(f"  epoch {epoch} chunk {ci_idx+1}/{len(train_chunks)} "
                  f"({os.path.basename(cp)}): loaded n={len(feats_120):,} "
                  f"in {t_load:.1f}s", flush=True)

            t0 = time.time()
            pattern_legal = derive_pattern_labels(board_labels, positions)
            t_derive = time.time() - t0
            print(f"    derived 960-d labels in {t_derive:.1f}s", flush=True)

            if args.pos_weight is None and epoch == 1 and ci_idx == 0:
                legal_rate = pattern_legal.mean()
                args.pos_weight = (1 - legal_rate) / max(legal_rate, 1e-6)
                print(f"    pos_weight auto-set to {args.pos_weight:.2f}",
                      flush=True)
            pw_tensor = torch.tensor([args.pos_weight], dtype=torch.float32,
                                     device=device)

            n_rows = len(feats_120)
            perm = rng.permutation(n_rows)
            t0 = time.time()
            n_batches = (n_rows + args.batch_size - 1) // args.batch_size
            for bi, i in enumerate(range(0, n_rows, args.batch_size)):
                batch_idx = perm[i:i + args.batch_size]
                x = torch.from_numpy(
                    feats_120[batch_idx].astype(np.float32)
                ).to(device)
                y_pat = torch.from_numpy(
                    pattern_legal[batch_idx].astype(np.float32)
                ).to(device)
                pos = torch.from_numpy(positions[batch_idx]).to(device)
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                loss = torch.tensor(0.0, device=device)
                n_in_batch = 0
                if even_mask.any():
                    x_e = x[even_mask]
                    y_e = y_pat[even_mask]
                    logits_e = model_even(x_e)  # (N, B_e, 960)
                    target_e = y_e.unsqueeze(0).expand(args.num_seeds, -1, -1)
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logits_e, target_e, pos_weight=pw_tensor,
                    )
                    n_in_batch += int(even_mask.sum())
                if odd_mask.any():
                    x_o = x[odd_mask]
                    y_o = y_pat[odd_mask]
                    logits_o = model_odd(x_o)
                    target_o = y_o.unsqueeze(0).expand(args.num_seeds, -1, -1)
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logits_o, target_o, pos_weight=pw_tensor,
                    )
                    n_in_batch += int(odd_mask.sum())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * n_in_batch
                total_n += n_in_batch

                if (bi + 1) % 50 == 0:
                    t_batch = time.time() - t0
                    print(f"    batch {bi+1}/{n_batches}  "
                          f"loss={total_loss/total_n:.4f}  "
                          f"{int(t_batch)}s for {bi+1} batches  "
                          f"(epoch elapsed {int(time.time()-t_epoch_start)}s)",
                          flush=True)

            del feats_180, feats_120, board_labels, positions, pattern_legal

        # End-of-epoch save
        save_path = base_save_path
        all_seeds = []
        for s in range(args.num_seeds):
            all_seeds.append({
                'even': model_even.state_per_seed(s),
                'odd':  model_odd.state_per_seed(s),
            })
        torch.save({
            'all_seeds': all_seeds,
            'hidden_dim': args.hidden,
            'input_dim': INPUT_DIM,
            'n_patterns': N_PATTERNS,
            'num_seeds': args.num_seeds,
            'epoch': epoch,
        }, save_path)
        print(f"Epoch {epoch}: n={total_n:,} loss={total_loss/total_n:.4f}  "
              f"saved {save_path}  "
              f"(epoch elapsed {int(time.time()-t_epoch_start)}s)", flush=True)


if __name__ == '__main__':
    main()
