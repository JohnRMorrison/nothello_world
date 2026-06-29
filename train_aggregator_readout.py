"""Variant A stacking readout for the 3-MLP ensemble.

Workflow:
  1. Load 3 MLP checkpoints
  2. Precompute (scores_a, scores_b, scores_c) per position + legal mask on
     a TRAIN chunk and a TEST chunk
  3. Train a small readout MLP on TRAIN  (input: 180-d cell scores, target: 60-d legal mask)
  4. Evaluate top-1 legal on TEST and compare to simple-sum baseline

The readout only sees the 3 models' outputs, not the board features --
"output-only stacking."  This isolates the effect of smarter aggregation
without giving the readout extra capacity to re-learn the legality task.

Usage:
  sbatch train_aggregator_readout.sh
  or:
  python train_aggregator_readout.py \\
    --ckpts seed0.pt seed43.pt seed44.pt \\
    --train-chunk chunk_ext_0037.npz --test-chunk chunk_ext_0039.npz \\
    --train-size 5000000 --test-size 5000000 \\
    --readout-hidden 128 --readout-epochs 10
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
from compare_v4_vs_mlp import load_mlp
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import compute_pattern_labels_batch


_PATTERNS = enumerate_flanking_patterns()
PATTERN_TO_CELL = np.array(
    [MOVE_TO_IDX[p['target']] for p in _PATTERNS], dtype=np.int64,
)
_PAT_TARGETS, _PAT_TERMINALS, _PAT_OPP_CELLS, _PAT_OPP_MASK = (
    precompute_pattern_arrays(_PATTERNS)
)


class ReadoutMLP(nn.Module):
    def __init__(self, hidden, input_dim=180, output_dim=60):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_dim),
            )
        else:
            self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.net(x)


def derive_cell_legal(board_labels, positions, batch_size=500_000):
    """Derive 60-d cell-legal mask from 64-d board state, in row batches."""
    n = len(board_labels)
    cell_legal = np.zeros((n, 60), dtype=np.uint8)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        board = board_labels[start:end].astype(np.int8)
        pos = positions[start:end].astype(np.int64)
        pat_legal = compute_pattern_labels_batch(
            board, pos, _PAT_TARGETS, _PAT_TERMINALS,
            _PAT_OPP_CELLS, _PAT_OPP_MASK,
        )
        for p in range(960):
            c = PATTERN_TO_CELL[p]
            np.maximum(cell_legal[start:end, c],
                       (pat_legal[:, p] > 0).astype(np.uint8),
                       out=cell_legal[start:end, c])
    return cell_legal


def mlp_scores_batch(mlp_bundle, feats_120, positions, device):
    """Batched per-model cell scores (B, 60).

    compare_v4_vs_mlp.mlp_cell_scores uses `use_even = (k % 2 == 1)` where
    k = number of moves played.  In chunk_ext, the stored `position p` is
    OFF BY ONE: features at position p encode (p+1) played cells. So the
    equivalent k = position + 1, and the parity formula must flip:
        use_even = ((position+1) % 2 == 1) = (position % 2 == 0)
    Training in train_pattern_simple uses the same chunk_ext convention
    (even_mask = pos % 2 == 0 trains model_even).
    """
    me, mo, idx, mask = mlp_bundle
    B = feats_120.shape[0]
    cell_scores = torch.zeros(B, 60, device=device)
    use_me_mask = (positions % 2 == 0)
    use_mo_mask = ~use_me_mask

    if use_me_mask.any():
        with torch.no_grad():
            logits = me(feats_120[use_me_mask])
        log1m = -F.softplus(logits)
        gathered = log1m[:, idx]
        gathered = gathered.masked_fill(~mask, 0.0)
        cell_scores[use_me_mask] = -gathered.sum(dim=-1)
    if use_mo_mask.any():
        with torch.no_grad():
            logits = mo(feats_120[use_mo_mask])
        log1m = -F.softplus(logits)
        gathered = log1m[:, idx]
        gathered = gathered.masked_fill(~mask, 0.0)
        cell_scores[use_mo_mask] = -gathered.sum(dim=-1)
    return cell_scores


def precompute_chunk(chunk_path, mlps, max_rows, batch_size, device, seed,
                      pos_min=None, pos_max=None):
    """Returns (scores: (N, 3, 60) float32, cell_legal: (N, 60) uint8).

    If pos_min/pos_max are set, drop rows whose chunk_ext position is outside
    [pos_min, pos_max].  Recall chunk_ext position p = (number of moves
    played) - 1, so for "k in 5..53" (compare_mlp_seeds_3way convention) we
    want chunk_ext positions in 4..52.
    """
    print(f"Loading chunk {os.path.basename(chunk_path)}")
    with np.load(chunk_path) as z:
        N = z['features'].shape[0]
        rng = np.random.RandomState(seed)
        # Sample 1.5x what we need so the post-filter total is close to target
        oversample = int(min(N, max_rows * 1.5)) if pos_min is not None else min(N, max_rows)
        sample_idx = rng.choice(N, size=oversample, replace=False)
        sample_idx.sort()
        feats = z['features'][sample_idx].astype(np.float16)
        labels = z['labels'][sample_idx].astype(np.int8)
        positions = z['positions'][sample_idx].astype(np.int64)
    if pos_min is not None and pos_max is not None:
        keep = (positions >= pos_min) & (positions <= pos_max)
        feats = feats[keep]
        labels = labels[keep]
        positions = positions[keep]
        # Trim to requested size
        if len(feats) > max_rows:
            feats = feats[:max_rows]
            labels = labels[:max_rows]
            positions = positions[:max_rows]
        print(f"  Filtered to positions in [{pos_min}, {pos_max}]: "
              f"{len(feats):,} rows remain")
    n = len(feats)
    print(f"  Sampled {n:,} rows")
    feats_120 = np.concatenate([feats[:, :60], feats[:, 120:180]], axis=1)
    print(f"  Deriving 60-d legal mask (batched compute_pattern_labels)...")
    cell_legal = derive_cell_legal(labels, positions)

    print(f"  Computing 3-model cell scores...")
    scores = np.zeros((n, 3, 60), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        x = torch.from_numpy(feats_120[i:end].astype(np.float32)).to(device)
        pos = torch.from_numpy(positions[i:end]).to(device)
        for mi, mlp in enumerate(mlps):
            scores[i:end, mi] = mlp_scores_batch(mlp, x, pos, device).cpu().numpy()
        if (i // batch_size) % 200 == 0:
            print(f"    {end:,}/{n:,}  ({int(time.time()-t0)}s)", flush=True)
    return scores, cell_legal


def top1_legal(scores_60d, legal):
    preds = scores_60d.argmax(axis=1)
    return legal[np.arange(len(preds)), preds].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs=3, required=True)
    ap.add_argument('--train-chunk', required=True)
    ap.add_argument('--test-chunk', required=True)
    ap.add_argument('--train-size', type=int, default=5_000_000)
    ap.add_argument('--test-size', type=int, default=5_000_000)
    ap.add_argument('--mlp-hidden', type=int, default=512)
    ap.add_argument('--readout-hidden', type=int, default=128,
                    help='0 = linear readout')
    ap.add_argument('--readout-epochs', type=int, default=10)
    ap.add_argument('--readout-lr', type=float, default=1e-3)
    ap.add_argument('--readout-batch-size', type=int, default=8192)
    ap.add_argument('--batch-size', type=int, default=8192,
                    help='Batch size for the MLP forward passes during precompute')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pos-min', type=int, default=None,
                    help='If set, filter rows to chunk_ext position >= pos_min.  '
                         'For compare_mlp_seeds_3way parity (k=5..53), use --pos-min 4')
    ap.add_argument('--pos-max', type=int, default=None,
                    help='If set, filter rows to chunk_ext position <= pos_max.  '
                         'For compare_mlp_seeds_3way parity (k=5..53), use --pos-max 52')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading 3 MLPs:")
    for p in args.ckpts:
        print(f"  {p}")
    mlps = [load_mlp(p, args.mlp_hidden, device) for p in args.ckpts]

    print()
    print("=== Precompute TRAIN scores ===")
    train_scores, train_legal = precompute_chunk(
        args.train_chunk, mlps, args.train_size,
        args.batch_size, device, args.seed,
        pos_min=args.pos_min, pos_max=args.pos_max,
    )
    print()
    print("=== Precompute TEST scores ===")
    test_scores, test_legal = precompute_chunk(
        args.test_chunk, mlps, args.test_size,
        args.batch_size, device, args.seed + 1,
        pos_min=args.pos_min, pos_max=args.pos_max,
    )

    print()
    print("=== Baselines on TEST set ===")
    baselines = {
        'A only':                       test_scores[:, 0],
        'B only':                       test_scores[:, 1],
        'C only':                       test_scores[:, 2],
        'A+B+C  (sum log_prob_or)':     test_scores.sum(axis=1),
        'B+C    (best pairwise)':       test_scores[:, 1] + test_scores[:, 2],
    }
    for name, sel in baselines.items():
        print(f"  {name:<32}  {top1_legal(sel, test_legal):.4f}")
    # Oracle ceiling (any of the 3 models correct, with simple sum picking)
    # — this is the rescuable ceiling for 3-model output-only stacking
    any_legal = test_legal.max(axis=1) > 0   # any cell legal (trivially true)
    # Best each model can pick: take its individual argmax then check legality
    n_test = len(test_scores)
    correct_by_model = np.stack([
        test_legal[np.arange(n_test), test_scores[:, m].argmax(axis=1)]
        for m in range(3)
    ], axis=0)              # (3, N) booleans
    oracle_correct = correct_by_model.max(axis=0).mean()
    print(f"  ORACLE (any of 3 correct)        {oracle_correct:.4f}")

    print()
    print(f"=== Train readout (hidden={args.readout_hidden}) ===")
    readout = ReadoutMLP(args.readout_hidden, 180, 60).to(device)
    n_params = sum(p.numel() for p in readout.parameters())
    print(f"  Architecture: {'linear' if args.readout_hidden==0 else 'MLP 180->%d->60' % args.readout_hidden}")
    print(f"  Readout params: {n_params:,}")

    opt = torch.optim.Adam(readout.parameters(), lr=args.readout_lr)
    n_train = len(train_scores)
    n_test = len(test_scores)
    train_X = torch.from_numpy(train_scores.reshape(n_train, 180).astype(np.float32))
    train_Y = torch.from_numpy(train_legal.astype(np.float32))
    # Keep test on CPU; batch-stream to GPU during eval to avoid OOM
    test_X = torch.from_numpy(test_scores.reshape(n_test, 180).astype(np.float32))
    test_Y_np = test_legal.astype(np.float32)

    # BCE pos_weight scaled by class imbalance
    legal_rate = train_Y.mean().item()
    pos_weight = torch.tensor([(1 - legal_rate) / legal_rate], device=device)
    print(f"  legal_rate={legal_rate:.4f}  pos_weight={pos_weight.item():.2f}")
    print()

    best_test = 0.0
    for epoch in range(1, args.readout_epochs + 1):
        readout.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        t0 = time.time()
        for i in range(0, n_train, args.readout_batch_size):
            idx = perm[i:i + args.readout_batch_size]
            x = train_X[idx].to(device)
            y = train_Y[idx].to(device)
            logits = readout(x)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        avg_loss = total_loss / n_train

        readout.eval()
        # Batched test eval to avoid OOM on small GPUs
        test_preds = np.empty(n_test, dtype=np.int64)
        eval_bs = args.readout_batch_size * 4
        with torch.no_grad():
            for i in range(0, n_test, eval_bs):
                end = min(i + eval_bs, n_test)
                xb = test_X[i:end].to(device)
                test_preds[i:end] = readout(xb).argmax(dim=-1).cpu().numpy()
        test_acc = test_Y_np[np.arange(n_test), test_preds].mean()
        best_test = max(best_test, test_acc)
        print(f"  Epoch {epoch}: train_loss={avg_loss:.4f}  "
              f"test_top1_legal={test_acc:.4f}  ({int(time.time()-t0)}s)",
              flush=True)

    print()
    print(f"=== Summary ===")
    print(f"  Best test top-1 legal (readout):     {best_test:.4f}")
    print(f"  Simple sum baseline (A+B+C):         "
          f"{top1_legal(test_scores.sum(axis=1), test_legal):.4f}")
    print(f"  Best pairwise (B+C):                 "
          f"{top1_legal(test_scores[:, 1] + test_scores[:, 2], test_legal):.4f}")
    print(f"  Oracle (any of 3 correct):           {oracle_correct:.4f}")


if __name__ == '__main__':
    main()
