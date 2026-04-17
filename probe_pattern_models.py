"""Train linear probe on hidden layer of pattern detector models.

Streams through all chunks (same as train_pattern_simple.py) to use
the full 12M games for probe training.

Tests whether board state is encoded in the hidden layer even when
the model wasn't trained to represent it (e.g., direct, emergent modes).

Usage:
    python probe_pattern_models.py --ckpt pattern_simple_direct_H512.pt --mode direct --hidden 512
"""
import sys, os
sys.path.insert(0, '.')

import argparse
import numpy as np
import torch
import torch.nn as nn
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES, OPTIONS,
)


def get_hidden(model, x, mode):
    """Extract H-d hidden activation (after first Linear + ReLU)."""
    with torch.no_grad():
        if mode == "direct":
            return torch.relu(model.net[0](x))
        else:
            return torch.relu(model.backbone[0](x))


def train_probe(chunk_dir, device, model_even, model_odd, mode,
                feature_cols, hidden_dim, epochs=10, lr=1e-3, batch_size=1024):
    """Train linear probe streaming through all chunks."""

    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    if not chunk_files:
        raise ValueError(f"No chunks in {chunk_dir}")

    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]

    print(f"Probe training: {len(chunk_files)} chunks, H={hidden_dim}, {epochs} epochs")

    # Even/odd probes (separate, matching the model split)
    probe_even = nn.Linear(hidden_dim, 64 * OPTIONS).to(device)
    probe_odd = nn.Linear(hidden_dim, 64 * OPTIONS).to(device)
    optimizer = torch.optim.Adam(
        list(probe_even.parameters()) + list(probe_odd.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    # Load eval data
    ev_X, ev_Y, ev_pos = _load_features(eval_path)
    if feature_cols is not None:
        ev_X = ev_X[:, feature_cols]
    n_eval = min(len(ev_X), 49 * 10000)
    ev_X = ev_X[:n_eval].clone()
    ev_Y = ev_Y[:n_eval].clone()
    ev_pos = ev_pos[:n_eval].clone()
    print(f"  Eval samples: {len(ev_X)}")

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        probe_even.train(); probe_odd.train()
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
                    h = get_hidden(model_even, x[even_mask], mode)
                    logits = probe_even(h).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[even_mask].reshape(-1))
                if odd_mask.any():
                    h = get_hidden(model_odd, x[odd_mask], mode)
                    logits = probe_odd(h).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[odd_mask].reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1

            del tr_X, tr_Y, tr_pos

        # Eval
        probe_even.eval(); probe_odd.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(ev_X), batch_size):
                x = ev_X[i:i + batch_size].to(device)
                y = ev_Y[i:i + batch_size].to(device)
                pos = ev_pos[i:i + batch_size]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask
                preds = torch.zeros_like(y)

                if even_mask.any():
                    h = get_hidden(model_even, x[even_mask], mode)
                    preds[even_mask] = probe_even(h).view(-1, 64, OPTIONS).argmax(-1)
                if odd_mask.any():
                    h = get_hidden(model_odd, x[odd_mask], mode)
                    preds[odd_mask] = probe_odd(h).view(-1, 64, OPTIONS).argmax(-1)

                correct += (preds == y).sum().item()
                total += y.numel()

        acc = correct / total
        best_acc = max(best_acc, acc)
        scheduler.step(epoch_loss / max(epoch_batches, 1))
        cur_lr = optimizer.param_groups[0]['lr']
        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"  Probe epoch {epoch}: acc={acc:.4%}  loss={avg_loss:.5f}  lr={cur_lr:.2e}",
              flush=True)

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to saved checkpoint")
    parser.add_argument("--mode", required=True,
                        choices=["direct", "emergent", "e2e", "two-stage"])
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    feature_cols = list(range(N_MOVES, 2 * N_MOVES))
    input_dim = N_MOVES

    print(f"Device: {device}")
    print(f"Probing {args.mode} H={args.hidden} from {args.ckpt}")

    # Load checkpoint and reconstruct models
    ckpt = torch.load(args.ckpt, map_location=device)
    from train_pattern_simple import DirectMLP, EndToEndMLP, TwoStageMLP
    n_patterns = ckpt.get('n_patterns', 960)

    if args.mode == "direct":
        model_even = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
        model_odd = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    elif args.mode == "two-stage":
        model_even = TwoStageMLP(input_dim, args.hidden, n_patterns).to(device)
        model_odd = TwoStageMLP(input_dim, args.hidden, n_patterns).to(device)
    else:
        model_even = EndToEndMLP(input_dim, args.hidden, n_patterns).to(device)
        model_odd = EndToEndMLP(input_dim, args.hidden, n_patterns).to(device)

    model_even.load_state_dict(ckpt['even'])
    model_odd.load_state_dict(ckpt['odd'])
    model_even.eval()
    model_odd.eval()
    # Freeze all model params (we only train the probe)
    for p in model_even.parameters():
        p.requires_grad = False
    for p in model_odd.parameters():
        p.requires_grad = False
    print(f"  Loaded checkpoint (pat_acc={ckpt.get('best_pat_acc', '?')})")

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")

    best_acc = train_probe(
        chunk_dir, device, model_even, model_odd, args.mode,
        feature_cols, args.hidden, epochs=args.epochs)

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}, H={args.hidden}")
    print(f"Pattern accuracy: {ckpt.get('best_pat_acc', '?')}")
    print(f"Probe board accuracy: {best_acc:.4%}")
    print(f"{'='*60}")
