"""Train a Nanda-style board-state probe on top of a pattern detector
whose input includes the precomputed by_black channel (loaded from sidecar
.npy files alongside each chunk).

Mirrors probe_pattern_models.py but uses load_chunk_with_by_black so that
the trained MLP sees the same input distribution it was trained on.

Usage:
    python probe_with_by_black.py \\
        --ckpt experiments/.../pattern_simple_direct_H512_wheneven_byblack.pt \\
        --features when+even+by_black --hidden 512 --epochs 5
"""
import sys, os, argparse, time
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    get_device, OPTIONS,
)
from train_pattern_simple import DirectMLP
from train_with_by_black import load_chunk_with_by_black


def hidden_of(model, x):
    """Hidden = ReLU(linear[0](x)); matches DirectMLP / TwoStageMLP layout."""
    if hasattr(model, "net"):
        return torch.relu(model.net[0](x))
    return torch.relu(model.backbone[0](x))


def train_probe(chunk_dir, device, model_even, model_odd, features,
                hidden_dim, epochs, save_path, lr=1e-3, batch_size=1024):
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.startswith("chunk_") and f.endswith(".npz"))
    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    print(f"Train chunks: {len(train_paths)}, eval: {os.path.basename(eval_path)}")

    probe_even = nn.Linear(hidden_dim, 64 * OPTIONS).to(device)
    probe_odd = nn.Linear(hidden_dim, 64 * OPTIONS).to(device)
    optimizer = torch.optim.Adam(
        list(probe_even.parameters()) + list(probe_odd.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    # Eval chunk: stride-slice for uniform turn coverage
    ev_X, ev_Y, ev_pos = load_chunk_with_by_black(eval_path, features)
    n_eval = min(len(ev_X), 49 * 10000)
    stride = max(1, len(ev_X) // n_eval)
    ev_X = ev_X[::stride][:n_eval].clone()
    ev_Y = ev_Y[::stride][:n_eval].clone()
    ev_pos = ev_pos[::stride][:n_eval].clone()
    print(f"  Eval samples: {len(ev_X)} (stride={stride} across turns 5-53)")

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        probe_even.train(); probe_odd.train()
        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))
        epoch_loss = 0.0; n_batches = 0
        t0 = time.time()

        for ci in chunk_order:
            tr_X, tr_Y, tr_pos = load_chunk_with_by_black(
                train_paths[ci], features)
            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), batch_size):
                idx = perm[i:i + batch_size]
                x = tr_X[idx].to(device)
                y = tr_Y[idx].to(device)
                pos = tr_pos[idx]
                em = (pos % 2 == 0); om = ~em

                loss = torch.tensor(0.0, device=device)
                with torch.no_grad():
                    h_e = hidden_of(model_even, x[em]) if em.any() else None
                    h_o = hidden_of(model_odd, x[om]) if om.any() else None
                if em.any():
                    logits = probe_even(h_e).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[em].reshape(-1))
                if om.any():
                    logits = probe_odd(h_o).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[om].reshape(-1))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += float(loss.item()); n_batches += 1
            del tr_X, tr_Y, tr_pos

        # Eval
        probe_even.eval(); probe_odd.eval()
        correct = 0; total = 0
        per_cell_correct = np.zeros(64, dtype=np.int64)
        with torch.no_grad():
            for i in range(0, len(ev_X), batch_size):
                x = ev_X[i:i + batch_size].to(device)
                y = ev_Y[i:i + batch_size].to(device)
                pos = ev_pos[i:i + batch_size]
                em = (pos % 2 == 0); om = ~em
                preds = torch.zeros_like(y)
                if em.any():
                    h = hidden_of(model_even, x[em])
                    preds[em] = probe_even(h).view(-1, 64, OPTIONS).argmax(-1)
                if om.any():
                    h = hidden_of(model_odd, x[om])
                    preds[om] = probe_odd(h).view(-1, 64, OPTIONS).argmax(-1)
                correct += (preds == y).sum().item()
                total += y.numel()
                per_cell_correct += (preds == y).sum(dim=0).cpu().numpy()

        acc = correct / total
        per_cell_acc = per_cell_correct / max(len(ev_X), 1)
        avg = epoch_loss / max(n_batches, 1)
        scheduler.step(avg)
        dt = time.time() - t0
        print(f"Epoch {epoch}: acc={acc:.4%}  loss={avg:.5f}  time={dt:.0f}s  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)

        # Per-region breakdown
        CENTER = {27, 28, 35, 36}
        def reg(label):
            cells = []
            for c in range(64):
                r, col = c // 8, c % 8
                if label == 'corner' and r in (0, 7) and col in (0, 7): cells.append(c)
                elif label == 'edge' and (r in (0, 7) or col in (0, 7)) and not (r in (0, 7) and col in (0, 7)): cells.append(c)
                elif label == 'center' and c in CENTER: cells.append(c)
                elif label == 'inner' and c not in CENTER and r not in (0, 7) and col not in (0, 7): cells.append(c)
            return np.mean([per_cell_acc[c] for c in cells]) if cells else 0
        print(f"  corner={reg('corner'):.4f}  edge={reg('edge'):.4f}  "
              f"inner={reg('inner'):.4f}  center={reg('center'):.4f}",
              flush=True)

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'even': {k: v.cpu().clone() for k, v in probe_even.state_dict().items()},
                'odd':  {k: v.cpu().clone() for k, v in probe_odd.state_dict().items()},
                'hidden_dim': hidden_dim,
                'best_acc': best_acc,
                'per_cell_acc': per_cell_acc,
                'features': features,
            }, save_path)
            print(f"  Saved {save_path}", flush=True)

    print(f"\n=== Final best: {best_acc:.4%} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True,
                        help="Path to pattern detector checkpoint")
    parser.add_argument("--features", required=True,
                        choices=["when+by_black", "when+even+by_black"])
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Probe from {args.ckpt}  features={args.features}  H={args.hidden}")

    ckpt = torch.load(args.ckpt, map_location=device)
    input_dim = ckpt.get('input_dim',
                          120 if args.features == "when+by_black" else 180)

    model_even = DirectMLP(input_dim, args.hidden, ckpt.get('n_patterns', 960)).to(device)
    model_odd = DirectMLP(input_dim, args.hidden, ckpt.get('n_patterns', 960)).to(device)
    model_even.load_state_dict(ckpt['even'])
    model_odd.load_state_dict(ckpt['odd'])
    model_even.eval(); model_odd.eval()
    for p in list(model_even.parameters()) + list(model_odd.parameters()):
        p.requires_grad = False

    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    feat_tag = {
        "when+even+by_black": "wheneven_byblack",
        "when+by_black":      "when_byblack",
    }[args.features]
    save_path = os.path.join(save_dir,
        f"probe_direct_H{args.hidden}_{feat_tag}_uniform.pt")
    print(f"Save path: {save_path}")

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    train_probe(chunk_dir, device, model_even, model_odd,
                args.features, args.hidden, args.epochs, save_path)
