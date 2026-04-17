"""Train linear probe on hidden layer of pattern detector models.

For each saved checkpoint, extract H-dimensional hidden activations
and train a linear probe to predict board state (64×3).

Tests whether board state is encoded in the hidden layer even when
the model wasn't trained to represent it (e.g., direct, emergent modes).

Usage:
    python probe_pattern_models.py --ckpt pattern_simple_direct_H512.pt --mode direct --hidden 512
    python probe_pattern_models.py --ckpt pattern_simple_emergent_H512.pt --mode emergent --hidden 512
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


def extract_hidden(model_even, model_odd, X, Y, pos, device,
                   mode, batch_size=1024):
    """Extract H-d hidden activations from the first linear layer.

    For all modes, the hidden layer is the ReLU output of the first Linear.
    """
    all_hidden = []
    all_labels = []
    all_pos = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = X[i:i + batch_size].to(device)
            y = Y[i:i + batch_size]
            p = pos[i:i + batch_size]
            even_mask = (p % 2 == 0)
            odd_mask = ~even_mask

            h = torch.zeros(len(x), model_even.net[0].out_features if mode == "direct"
                            else model_even.backbone[0].out_features, device=device)

            for mask, model in [(even_mask, model_even), (odd_mask, model_odd)]:
                if not mask.any():
                    continue
                xm = x[mask]
                if mode == "direct":
                    # net = Sequential(Linear, ReLU, Linear)
                    h[mask] = torch.relu(model.net[0](xm))
                else:
                    # backbone = Sequential(Linear, ReLU, Linear)
                    h[mask] = torch.relu(model.backbone[0](xm))

            all_hidden.append(h.cpu())
            all_labels.append(y)
            all_pos.append(p)

    return torch.cat(all_hidden), torch.cat(all_labels), torch.cat(all_pos)


def train_probe(hidden, labels, device, epochs=10, lr=1e-3, batch_size=2048):
    """Train a linear probe: H-d → 64×3 on board state labels."""
    h_dim = hidden.shape[1]
    n = len(hidden)
    n_eval = min(n // 10, 50000)
    n_train = n - n_eval

    tr_h, tr_y = hidden[:n_train], labels[:n_train]
    ev_h, ev_y = hidden[n_train:], labels[n_train:]

    probe = nn.Linear(h_dim, 64 * OPTIONS).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            h = tr_h[idx].to(device)
            y = tr_y[idx].to(device)

            logits = probe(h).view(-1, 64, OPTIONS)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, OPTIONS), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Eval
        probe.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(ev_h), batch_size):
                h = ev_h[i:i + batch_size].to(device)
                y = ev_y[i:i + batch_size].to(device)
                logits = probe(h).view(-1, 64, OPTIONS)
                preds = logits.argmax(-1)
                correct += (preds == y).sum().item()
                total += y.numel()

        acc = correct / total
        best_acc = max(best_acc, acc)
        print(f"  Probe epoch {epoch}: acc={acc:.4%}  loss={epoch_loss/n_batches:.5f}",
              flush=True)

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to saved checkpoint")
    parser.add_argument("--mode", required=True,
                        choices=["direct", "emergent", "e2e", "two-stage"])
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=500000,
                        help="Max samples to extract (memory)")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    feature_cols = list(range(N_MOVES, 2 * N_MOVES))
    input_dim = N_MOVES

    print(f"Device: {device}")
    print(f"Probing {args.mode} H={args.hidden} from {args.ckpt}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)

    # Reconstruct models
    from train_pattern_simple import DirectMLP, EndToEndMLP, TwoStageMLP
    n_patterns = ckpt.get('n_patterns', 960)

    if args.mode == "direct":
        model_even = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
        model_odd = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    else:
        if args.mode == "two-stage":
            model_even = TwoStageMLP(input_dim, args.hidden, n_patterns).to(device)
            model_odd = TwoStageMLP(input_dim, args.hidden, n_patterns).to(device)
        else:
            model_even = EndToEndMLP(input_dim, args.hidden, n_patterns).to(device)
            model_odd = EndToEndMLP(input_dim, args.hidden, n_patterns).to(device)

    model_even.load_state_dict(ckpt['even'])
    model_odd.load_state_dict(ckpt['odd'])
    model_even.eval()
    model_odd.eval()
    print(f"  Loaded checkpoint (pat_acc={ckpt.get('best_pat_acc', '?')})")

    # Load data chunks for probing
    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)

    # Use last chunk for probing (same as eval in training)
    print(f"Loading data from {os.path.basename(chunk_files[-1])}...")
    X, Y, pos = _load_features(chunk_files[-1])
    if feature_cols is not None:
        X = X[:, feature_cols]
    n = min(len(X), args.max_samples)
    X, Y, pos = X[:n], Y[:n], pos[:n]
    print(f"  {n} samples")

    # Extract hidden activations
    print("Extracting hidden activations...")
    hidden, labels, positions = extract_hidden(
        model_even, model_odd, X, Y, pos, device, args.mode)
    print(f"  Hidden: {hidden.shape}")

    # Free original data
    del X, Y, pos

    # Train probe
    print(f"\nTraining linear probe: {hidden.shape[1]}-d → 64×{OPTIONS}")
    best_acc = train_probe(hidden, labels, device)
    print(f"\nBest probe accuracy: {best_acc:.4%}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}, H={args.hidden}")
    print(f"Pattern accuracy: {ckpt.get('best_pat_acc', '?')}")
    print(f"Probe board accuracy: {best_acc:.4%}")
    print(f"{'='*60}")
