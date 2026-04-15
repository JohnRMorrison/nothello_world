"""Train MLP on move features using chunk-streaming (constant memory).

Loads one chunk at a time — scales to any number of games.

Usage:
    python train_streaming.py --hidden 1024 --epochs 10 [--features when+even]
    python train_streaming.py --hidden 1024 --epochs 1 --features when --random-proj --seed 0
"""
import sys, os, json, math
sys.path.insert(0, '.')

import argparse
import numpy as np
import torch
import torch.nn as nn
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _train_mlp_streaming, _load_features, get_device, N_MOVES, OPTIONS,
)

# Spatial feature transform (row/col per square)
_VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})
_ROWS = np.array([pos // 8 / 7.0 for pos in _VALID_MOVES], dtype=np.float32)
_COLS = np.array([pos % 8 / 7.0 for pos in _VALID_MOVES], dtype=np.float32)

def _add_spatial(X):
    """(N, 180) -> (N, 300): append played*row and played*col."""
    played = X[:, :N_MOVES]
    rows_feat = played * torch.from_numpy(_ROWS)
    cols_feat = played * torch.from_numpy(_COLS)
    return torch.cat([X, rows_feat, cols_feat], dim=1)


def _train_random_proj_streaming(chunk_dir, device, input_dim, hidden_dim,
                                  feature_cols, seed, lr=1e-3, epochs=1,
                                  batch_size=1024, save_path=None):
    """Train linear output on top of a frozen random projection.

    Architecture per even/odd split:
        x (input_dim) -> [frozen] Linear+ReLU (hidden_dim) -> [trained] Linear (64*3)
    """
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir) if f.endswith(".npz"))
    if not chunk_files:
        raise ValueError(f"No chunk files in {chunk_dir}")

    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    print(f"Random projection: {len(chunk_files)} chunks, H={hidden_dim}, "
          f"input={input_dim}, seed={seed}, {epochs} epochs")
    print(f"  Train chunks: {len(train_paths)}, Eval chunk: {os.path.basename(eval_path)}")

    # Fixed random projection (Kaiming init for ReLU), shared across even/odd
    torch.manual_seed(seed)
    proj_W = torch.randn(input_dim, hidden_dim, device=device) * math.sqrt(2.0 / input_dim)
    proj_b = torch.zeros(hidden_dim, device=device)

    # Trainable output layers only
    out_even = nn.Linear(hidden_dim, 64 * OPTIONS).to(device)
    out_odd = nn.Linear(hidden_dim, 64 * OPTIONS).to(device)
    optimizer = torch.optim.Adam(
        list(out_even.parameters()) + list(out_odd.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    # Load eval data
    ev_X, ev_Y, ev_pos = _load_features(eval_path)
    if feature_cols is not None:
        ev_X = ev_X[:, feature_cols]
    n_eval = min(len(ev_X), 49 * 10000)
    ev_X, ev_Y, ev_pos = ev_X[:n_eval].clone(), ev_Y[:n_eval].clone(), ev_pos[:n_eval].clone()
    print(f"  Eval samples: {len(ev_X)}")

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        out_even.train()
        out_odd.train()

        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))

        epoch_loss = 0.0
        epoch_batches = 0

        for ci in chunk_order:
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            if feature_cols is not None:
                tr_X = tr_X[:, feature_cols]

            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), batch_size):
                idx = perm[i:i + batch_size]
                x = tr_X[idx].to(device)
                y = tr_Y[idx].to(device)
                pos = tr_pos[idx]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                loss = torch.tensor(0.0, device=device)
                if even_mask.any():
                    h = torch.relu(x[even_mask] @ proj_W + proj_b)
                    logits = out_even(h).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[even_mask].reshape(-1))
                if odd_mask.any():
                    h = torch.relu(x[odd_mask] @ proj_W + proj_b)
                    logits = out_odd(h).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[odd_mask].reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1

            del tr_X, tr_Y, tr_pos

        # Eval
        out_even.eval()
        out_odd.eval()
        correct = 0
        total = 0
        losses = []
        with torch.no_grad():
            for i in range(0, len(ev_X), batch_size):
                x = ev_X[i:i + batch_size].to(device)
                y = ev_Y[i:i + batch_size].to(device)
                pos = ev_pos[i:i + batch_size]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                preds = torch.zeros_like(y)
                if even_mask.any():
                    h = torch.relu(x[even_mask] @ proj_W + proj_b)
                    logits = out_even(h).view(-1, 64, OPTIONS)
                    preds[even_mask] = logits.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[even_mask].reshape(-1)).item())
                if odd_mask.any():
                    h = torch.relu(x[odd_mask] @ proj_W + proj_b)
                    logits = out_odd(h).view(-1, 64, OPTIONS)
                    preds[odd_mask] = logits.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[odd_mask].reshape(-1)).item())
                correct += (preds == y).sum().item()
                total += y.numel()

        acc = correct / total
        mean_loss = np.mean(losses)
        if acc > best_acc:
            best_acc = acc
            best_state = {
                'proj_W': proj_W.cpu(), 'proj_b': proj_b.cpu(),
                'even_out': {k: v.cpu().clone() for k, v in out_even.state_dict().items()},
                'odd_out': {k: v.cpu().clone() for k, v in out_odd.state_dict().items()},
            }
        scheduler.step(mean_loss)
        cur_lr = optimizer.param_groups[0]['lr']
        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"  Epoch {epoch}: acc={acc:.4%}  loss={avg_loss:.5f}  lr={cur_lr:.2e}",
              flush=True)

    if best_state is not None and save_path is not None:
        best_state.update({
            'hidden_dim': hidden_dim, 'input_dim': input_dim, 'seed': seed,
            'best_acc': best_acc,
        })
        torch.save(best_state, save_path)
        print(f"  Saved checkpoint to {save_path}")

    return best_acc


parser = argparse.ArgumentParser()
parser.add_argument("--hidden", type=int, default=1024)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--features", type=str, default="all",
                    choices=["all", "when+even", "played+when", "played+even",
                             "when", "played", "even"],
                    help="Which feature subset to use")
parser.add_argument("--spatial", action="store_true",
                    help="Add 120-d spatial features (row/col per square) -> 300-d")
parser.add_argument("--random-proj", action="store_true",
                    help="Freeze first layer (random projection), only train output layer")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for random projection initialization")
parser.add_argument("--output-dir",
                    default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
args = parser.parse_args()

# Feature column selections
FEATURE_SUBSETS = {
    "all":          (list(range(3 * N_MOVES)), 3 * N_MOVES),
    "when+even":    (list(range(N_MOVES, 3 * N_MOVES)), 2 * N_MOVES),
    "played+when":  (list(range(0, 2 * N_MOVES)), 2 * N_MOVES),
    "played+even":  (list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)), 2 * N_MOVES),
    "when":         (list(range(N_MOVES, 2 * N_MOVES)), N_MOVES),
    "played":       (list(range(0, N_MOVES)), N_MOVES),
    "even":         (list(range(2 * N_MOVES, 3 * N_MOVES)), N_MOVES),
}

feature_cols, input_dim = FEATURE_SUBSETS[args.features]
if args.features == "all":
    feature_cols = None  # no slicing needed
    input_dim = 3 * N_MOVES

transform_fn = None
if args.spatial:
    if args.features != "all":
        print("ERROR: --spatial requires --features all (needs played[] columns)")
        sys.exit(1)
    transform_fn = _add_spatial
    input_dim = 300  # 180 + 120

device = get_device()

if args.random_proj:
    feat_desc = f"{input_dim}-d -> randproj(seed={args.seed}) -> {args.hidden}-d"
    print(f"Device: {device}")
    print(f"Features: {args.features} ({feat_desc}), {args.epochs} epochs")

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    save_dir = os.path.join(args.output_dir, "mlp_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,
        f"mlp_{args.features}_randproj_s{args.seed}_H{args.hidden}_streaming.pt")

    acc = _train_random_proj_streaming(
        chunk_dir, device, input_dim, args.hidden,
        feature_cols=feature_cols, seed=args.seed,
        epochs=args.epochs, save_path=save_path,
    )
    print(f"\nFinal: {args.features} randproj(s={args.seed}) H={args.hidden}: {acc:.4%}")

else:
    suffix = "_spatial" if args.spatial else ""
    feat_desc = f"{input_dim}-d" + (" +spatial" if args.spatial else "")
    print(f"Device: {device}")
    print(f"Features: {args.features} ({feat_desc}), H={args.hidden}, {args.epochs} epochs")

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    save_dir = os.path.join(args.output_dir, "mlp_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,
        f"mlp_{args.features}{suffix}_H{args.hidden}_streaming.pt")

    acc = _train_mlp_streaming(
        chunk_dir, device, input_dim, args.hidden,
        feature_cols=feature_cols,
        transform_fn=transform_fn,
        epochs=args.epochs,
        save_path=save_path,
    )
    print(f"\nFinal: {args.features}{suffix} H={args.hidden}: {acc:.4%}")
