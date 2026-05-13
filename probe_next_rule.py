"""Nanda-style board-state probe on the hidden state of a next-rule MLP.

The next-rule MLP (train_next_rule.py) is a single model (no parity split)
trained with soft-target CE on 961 classes. We extract its hidden activation
and train a linear probe to decode the 64-cell board state (3 classes per
cell). Stride-slice eval for uniform turn coverage.

Usage:
    python probe_next_rule.py \\
        --ckpt experiments/.../next_rule_H512_wheneven.pt \\
        --features when+even --hidden 512 --epochs 5
"""
import sys, os, argparse, time, glob
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, OPTIONS, N_MOVES,
)
from train_next_rule import NextRuleMLP, feature_slice


def hidden_of(model, x):
    return torch.relu(model.net[0](x))


CENTER = {27, 28, 35, 36}


def cell_region(c):
    if c in CENTER: return 'center'
    r, col = c // 8, c % 8
    if r in (0, 7) and col in (0, 7): return 'corner'
    if r in (0, 7) or col in (0, 7):  return 'edge'
    return 'inner'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--features", default="when+even",
                        choices=["when", "when+even", "all"])
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-dir",
        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Probe next-rule MLP: {args.ckpt}  features={args.features}  H={args.hidden}")

    ckpt = torch.load(args.ckpt, map_location=device)
    input_dim = ckpt.get('input_dim',
        {"when": 60, "when+even": 120, "all": 180}[args.features])
    model = NextRuleMLP(input_dim, args.hidden).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Nanda-style parity-split probes: same hidden state from the single
    # next-rule MLP, but two separate Linear(H, 64*OPTIONS) heads for
    # even-parity and odd-parity positions.
    probe_even = nn.Linear(args.hidden, 64 * OPTIONS).to(device)
    probe_odd  = nn.Linear(args.hidden, 64 * OPTIONS).to(device)
    optimizer = torch.optim.Adam(
        list(probe_even.parameters()) + list(probe_odd.parameters()),
        lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.startswith("chunk_") and f.endswith(".npz"))
    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    print(f"Train chunks: {len(train_paths)}, eval: {os.path.basename(eval_path)}")

    # Eval chunk — stride-slice
    ev_X, ev_Y, ev_pos = _load_features(eval_path)
    n_eval = min(len(ev_X), 49 * 10000)
    stride = max(1, len(ev_X) // n_eval)
    ev_X = ev_X[::stride][:n_eval].clone()
    ev_Y = ev_Y[::stride][:n_eval].clone()
    ev_pos = ev_pos[::stride][:n_eval].clone()
    print(f"  Eval samples: {len(ev_X)} (stride={stride} across turns 5-53)")

    ev_Xf = feature_slice(ev_X, args.features)

    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    feat_tag = args.features.replace("+", "")
    save_path = os.path.join(save_dir,
        f"probe_next_rule_H{args.hidden}_{feat_tag}.pt")
    print(f"Save path: {save_path}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        probe_even.train(); probe_odd.train()
        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))
        epoch_loss = 0.0; n_batches = 0
        t0 = time.time()

        for ci in chunk_order:
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            tr_Xf = feature_slice(tr_X, args.features)

            perm = torch.randperm(len(tr_Xf))
            for i in range(0, len(perm), args.batch_size):
                idx = perm[i:i + args.batch_size]
                x = tr_Xf[idx].to(device)
                y = tr_Y[idx].to(device)
                pos = tr_pos[idx]
                em = (pos % 2 == 0); om = ~em
                with torch.no_grad():
                    h = hidden_of(model, x)
                loss = torch.tensor(0.0, device=device)
                if em.any():
                    logits = probe_even(h[em]).view(-1, 64, OPTIONS)
                    loss = loss + F.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[em].reshape(-1))
                if om.any():
                    logits = probe_odd(h[om]).view(-1, 64, OPTIONS)
                    loss = loss + F.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[om].reshape(-1))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += float(loss.item()); n_batches += 1
            del tr_X, tr_Y, tr_pos, tr_Xf

        # Eval
        probe_even.eval(); probe_odd.eval()
        correct = 0; total = 0
        per_cell_correct = np.zeros(64, dtype=np.int64)
        with torch.no_grad():
            for i in range(0, len(ev_Xf), args.batch_size):
                x = ev_Xf[i:i + args.batch_size].to(device)
                y = ev_Y[i:i + args.batch_size].to(device)
                pos = ev_pos[i:i + args.batch_size]
                em = (pos % 2 == 0); om = ~em
                h = hidden_of(model, x)
                preds = torch.zeros_like(y)
                if em.any():
                    preds[em] = probe_even(h[em]).view(-1, 64, OPTIONS).argmax(-1)
                if om.any():
                    preds[om] = probe_odd(h[om]).view(-1, 64, OPTIONS).argmax(-1)
                correct += (preds == y).sum().item()
                total += y.numel()
                per_cell_correct += (preds == y).sum(dim=0).cpu().numpy()
        acc = correct / total
        per_cell_acc = per_cell_correct / max(len(ev_Xf), 1)
        avg = epoch_loss / max(n_batches, 1)
        scheduler.step(avg)
        dt = time.time() - t0
        print(f"Epoch {epoch}: train_loss={avg:.5f}  acc={acc:.4%}  "
              f"time={dt:.0f}s  lr={optimizer.param_groups[0]['lr']:.2e}",
              flush=True)
        for label in ('corner', 'edge', 'inner', 'center'):
            cells = [c for c in range(64) if cell_region(c) == label]
            m = np.mean([per_cell_acc[c] for c in cells])
            print(f"  {label:>6s}: mean={m:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'even': {k: v.cpu().clone() for k, v in probe_even.state_dict().items()},
                'odd':  {k: v.cpu().clone() for k, v in probe_odd.state_dict().items()},
                'hidden': args.hidden,
                'best_acc': best_acc,
                'per_cell_acc': per_cell_acc,
                'features': args.features,
            }, save_path)
            print(f"  Saved {save_path}", flush=True)

    print(f"\n=== Final best probe acc: {best_acc:.4%} ===")
