"""Train N pattern-detector MLPs simultaneously by sharing data loading.

When I/O dominates wall-clock (as it does for these chunks over NFS), training
one model and training N models take roughly the same time, because the
expensive part (loading 40 chunks per epoch) is shared.

Architecture: VectorizedDirectMLP stacks N independent (per-seed) sets of
weights and uses bmm/einsum to run all N models in one CUDA kernel launch
per layer.  Each model trains on the same batches but with different random
initializations -> independent solutions.

Output: single .pt file containing all N model states.  Use
split_multi_seed_ckpt.py to extract individual seed checkpoints compatible
with compare_mlp_seeds.py and load_mlp.

Usage:
    sbatch train_multi_seed_mlp.sh  (defaults to H=512 N=100)
    HIDDEN=4096 NUM_SEEDS=25 sbatch train_multi_seed_mlp.sh
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
from train_pattern_simple import compute_pattern_labels_batch


N_PATTERNS = 960
_PATTERNS = enumerate_flanking_patterns()
_PAT_TARGETS, _PAT_TERMINALS, _PAT_OPP_CELLS, _PAT_OPP_MASK = (
    precompute_pattern_arrays(_PATTERNS)
)


class VectorizedDirectMLP(nn.Module):
    """N independent MLPs (input_dim -> hidden_dim -> output_dim) stacked into
    one nn.Module.  Forward returns (N, B, output_dim).

    Initialization mirrors nn.Linear's default (uniform(-1/sqrt(fan_in), ...)).
    """
    def __init__(self, n_models, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.n_models = n_models
        # Stacked weights with nn.Linear default init bounds
        bound1 = 1.0 / math.sqrt(input_dim)
        bound2 = 1.0 / math.sqrt(hidden_dim)
        self.W1 = nn.Parameter(
            (torch.rand(n_models, hidden_dim, input_dim) * 2 - 1) * bound1
        )
        self.b1 = nn.Parameter(
            (torch.rand(n_models, hidden_dim) * 2 - 1) * bound1
        )
        self.W2 = nn.Parameter(
            (torch.rand(n_models, output_dim, hidden_dim) * 2 - 1) * bound2
        )
        self.b2 = nn.Parameter(
            (torch.rand(n_models, output_dim) * 2 - 1) * bound2
        )

    def forward(self, x):
        """x: (B, input_dim).  Returns (N, B, output_dim)."""
        # h[n, b, h] = sum_i x[b, i] * W1[n, h, i] + b1[n, h]
        h = torch.einsum('bi,nhi->nbh', x, self.W1) + self.b1.unsqueeze(1)
        h = F.relu(h)
        # y[n, b, o] = sum_h h[n, b, h] * W2[n, o, h] + b2[n, o]
        y = torch.einsum('nbh,noh->nbo', h, self.W2) + self.b2.unsqueeze(1)
        return y

    def state_per_seed(self, seed):
        """Return a regular nn.Linear-compatible state_dict for one seed."""
        return {
            'net.0.weight': self.W1[seed].detach().cpu().clone(),
            'net.0.bias':   self.b1[seed].detach().cpu().clone(),
            'net.2.weight': self.W2[seed].detach().cpu().clone(),
            'net.2.bias':   self.b2[seed].detach().cpu().clone(),
        }


def derive_pattern_labels(board_labels, positions, batch_size=200_000):
    """Compute 960-d pattern legality from 64-d board state in row batches."""
    n = len(board_labels)
    out = np.zeros((n, N_PATTERNS), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        out[start:end] = compute_pattern_labels_batch(
            board_labels[start:end].astype(np.int8),
            positions[start:end].astype(np.int64),
            _PAT_TARGETS, _PAT_TERMINALS, _PAT_OPP_CELLS, _PAT_OPP_MASK,
        )
    return out


def slice_played_even(features_180):
    """180-d (played+when+even) -> 120-d (played+even)."""
    return np.concatenate(
        [features_180[:, :60], features_180[:, 120:180]], axis=1
    )


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
                            'heuristic_probe_results/pattern_detector_checkpoints')
    ap.add_argument('--seed', type=int, default=0,
                    help='Master seed for the random init of all N models')
    ap.add_argument('--pos-weight', type=float, default=None,
                    help='BCE pos_weight; auto-computed if not set')
    ap.add_argument('--save-every-epoch', action='store_true',
                    help='Save a separate ckpt per epoch (else overwrite)')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"N seeds: {args.num_seeds}")
    print(f"Hidden: {args.hidden}")
    print(f"Batch size: {args.batch_size}")

    INPUT_DIM = 120  # played+even

    # Build the vectorized model
    model_even = VectorizedDirectMLP(args.num_seeds, INPUT_DIM, args.hidden,
                                      N_PATTERNS).to(device)
    model_odd = VectorizedDirectMLP(args.num_seeds, INPUT_DIM, args.hidden,
                                     N_PATTERNS).to(device)
    n_params = sum(p.numel() for p in model_even.parameters()) * 2
    print(f"Total params (even + odd, all {args.num_seeds} seeds): {n_params:,}")

    optimizer = torch.optim.Adam(
        list(model_even.parameters()) + list(model_odd.parameters()),
        lr=args.lr,
    )

    # Find chunks
    chunks = sorted(glob.glob(os.path.join(args.chunk_dir, args.chunk_glob)))
    train_chunks = chunks[:-1]
    eval_chunk = chunks[-1]
    print(f"Found {len(chunks)} chunks  Train={len(train_chunks)}  Eval=1")

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
        last_log = time.time()
        t0 = time.time()

        for ci_idx, ci in enumerate(chunk_order):
            cp = train_chunks[ci]
            # Load chunk
            with np.load(cp) as z:
                feats_180 = z['features'].astype(np.float16)
                board_labels = z['labels'].astype(np.int8)
                positions = z['positions'].astype(np.int64)
            feats_120 = slice_played_even(feats_180)
            # Derive pattern labels from board state
            pattern_legal = derive_pattern_labels(board_labels, positions)

            # Auto pos_weight from first chunk only (representative)
            if args.pos_weight is None and epoch == 1 and ci_idx == 0:
                legal_rate = pattern_legal.mean()
                args.pos_weight = (1 - legal_rate) / max(legal_rate, 1e-6)
                print(f"  pos_weight auto-set to {args.pos_weight:.2f}")
            pw_tensor = torch.tensor([args.pos_weight], dtype=torch.float32,
                                     device=device)

            n_rows = len(feats_120)
            perm = rng.permutation(n_rows)
            for i in range(0, n_rows, args.batch_size):
                batch_idx = perm[i:i + args.batch_size]
                x = torch.from_numpy(
                    feats_120[batch_idx].astype(np.float32)
                ).to(device)
                y_pat = torch.from_numpy(
                    pattern_legal[batch_idx]
                ).to(device)
                pos = torch.from_numpy(positions[batch_idx]).to(device)
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                loss = torch.tensor(0.0, device=device)
                n_in_batch = 0
                if even_mask.any():
                    x_e = x[even_mask]                 # (B_e, 120)
                    y_e = y_pat[even_mask]             # (B_e, 960)
                    logits_e = model_even(x_e)         # (N, B_e, 960)
                    target_e = y_e.unsqueeze(0).expand(args.num_seeds, -1, -1)
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logits_e, target_e, pos_weight=pw_tensor
                    )
                    n_in_batch += int(even_mask.sum())
                if odd_mask.any():
                    x_o = x[odd_mask]
                    y_o = y_pat[odd_mask]
                    logits_o = model_odd(x_o)
                    target_o = y_o.unsqueeze(0).expand(args.num_seeds, -1, -1)
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logits_o, target_o, pos_weight=pw_tensor
                    )
                    n_in_batch += int(odd_mask.sum())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * n_in_batch
                total_n += n_in_batch

            del feats_180, feats_120, board_labels, positions, pattern_legal
            now = time.time()
            if now - last_log > 60:
                print(f"  epoch {epoch} chunk {ci_idx+1}/{len(train_chunks)}: "
                      f"n={total_n:,} loss={total_loss/max(1,total_n):.4f} "
                      f"elapsed={int(now-t0)}s", flush=True)
                last_log = now

        # End-of-epoch eval (random sample from eval chunk for speed)
        model_even.eval()
        model_odd.eval()
        with np.load(eval_chunk) as z:
            ev_feats_180 = z['features'].astype(np.float16)
            ev_board = z['labels'].astype(np.int8)
            ev_pos = z['positions'].astype(np.int64)
        n_ev = min(len(ev_feats_180), 200_000)
        ev_idx = rng.choice(len(ev_feats_180), size=n_ev, replace=False)
        ev_feats_180 = ev_feats_180[ev_idx]
        ev_board = ev_board[ev_idx]
        ev_pos = ev_pos[ev_idx]
        ev_feats_120 = slice_played_even(ev_feats_180)
        ev_pat = derive_pattern_labels(ev_board, ev_pos)

        ev_n = 0
        ev_correct_per_seed = torch.zeros(args.num_seeds, device=device)
        with torch.no_grad():
            for i in range(0, n_ev, args.batch_size):
                x = torch.from_numpy(
                    ev_feats_120[i:i + args.batch_size].astype(np.float32)
                ).to(device)
                y_pat = torch.from_numpy(
                    ev_pat[i:i + args.batch_size]
                ).to(device)
                pos = torch.from_numpy(ev_pos[i:i + args.batch_size]).to(device)
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask
                # Use both nets to fill out predictions, route by parity
                logits = torch.zeros(args.num_seeds, len(pos), N_PATTERNS,
                                     device=device)
                if even_mask.any():
                    logits[:, even_mask] = model_even(x[even_mask])
                if odd_mask.any():
                    logits[:, odd_mask] = model_odd(x[odd_mask])
                # per-pattern accuracy
                pred = (logits > 0).float()
                target = y_pat.unsqueeze(0).expand(args.num_seeds, -1, -1)
                ev_correct_per_seed += (pred == target).float().sum(dim=(1, 2))
                ev_n += y_pat.numel()
        ev_acc_per_seed = (ev_correct_per_seed / ev_n).cpu().numpy()
        print(f"Epoch {epoch}: train_n={total_n:,} loss={total_loss/total_n:.4f} "
              f"  eval pat_acc: mean={ev_acc_per_seed.mean():.4f}  "
              f"min={ev_acc_per_seed.min():.4f}  "
              f"max={ev_acc_per_seed.max():.4f}", flush=True)

        # Save all seeds (single big file)
        suffix = f"_epoch{epoch}" if args.save_every_epoch else ""
        save_path = base_save_path.replace('.pt', f'{suffix}.pt')
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
            'eval_acc_per_seed': ev_acc_per_seed.tolist(),
        }, save_path)
        print(f"  Saved {save_path}  ({args.num_seeds} seeds)", flush=True)


if __name__ == '__main__':
    main()
