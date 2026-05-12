"""Train a 1-layer MLP to predict 64-cell board state (3-class per cell)
from 960-d cumulative rule-firings.

Uses the precomputed rule_firings_NNNN.npz chunks (output of
precompute_rule_firings.py). Same chunk-streaming structure as
train_pattern_simple.py.

This is the headline test: if the 1-layer MLP can hit ~100% on board
decoding from cumulative firings, the bottleneck of the existing
move-history MLP is the input format, not the architectural depth.

Usage:
    python train_board_from_firings.py --hidden 512 --epochs 3
"""
import sys, os, argparse, time, glob
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    get_device,
)


CENTER_64 = {27, 28, 35, 36}


def cell_class(c64):
    if c64 in CENTER_64: return 'center'
    r, c = c64 // 8, c64 % 8
    if r in (0, 7) and c in (0, 7): return 'corner'
    if r in (0, 7) or c in (0, 7):  return 'edge'
    return 'inner'


class BoardFromFiringsMLP(nn.Module):
    def __init__(self, in_dim=960, hidden=512, n_cells=64, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_cells * n_classes),
        )
        self.n_cells = n_cells
        self.n_classes = n_classes

    def forward(self, x):
        return self.net(x).view(-1, self.n_cells, self.n_classes)


def load_chunk(path):
    d = np.load(path)
    return d['features'], d['labels'], d['positions']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--chunk-glob", default="rule_firings_*.npz")
    parser.add_argument("--use-mod2", action="store_true")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"H={args.hidden}, epochs={args.epochs}, mod2={args.use_mod2}")

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, args.chunk_glob)))
    if not chunk_files:
        raise FileNotFoundError(f"No chunks matching {args.chunk_glob}")
    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    print(f"Train chunks: {len(train_paths)}, eval: {os.path.basename(eval_path)}")

    # Load eval chunk once
    ev_X, ev_Y, ev_pos = load_chunk(eval_path)
    print(f"  Eval samples: {len(ev_X)}, features dim: {ev_X.shape[1]}")
    in_dim = ev_X.shape[1]
    if args.use_mod2:
        ev_X = (ev_X % 2).astype(np.uint8)
    ev_Xt = torch.from_numpy(ev_X.astype(np.float32)).to(device)
    ev_Yt = torch.from_numpy(ev_Y.astype(np.int64)).to(device)
    ev_post = torch.from_numpy(ev_pos.astype(np.int64)).to(device)

    model = BoardFromFiringsMLP(in_dim=in_dim, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,
        f"board_from_firings_H{args.hidden}"
        f"{'_mod2' if args.use_mod2 else ''}.pt")
    print(f"Save path: {save_path}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))
        epoch_loss = 0.0; n_batches = 0
        t0 = time.time()
        for ci in chunk_order:
            X, Y, _ = load_chunk(train_paths[ci])
            if args.use_mod2:
                X = (X % 2).astype(np.uint8)
            X = X.astype(np.float32)
            Y = Y.astype(np.int64)
            perm = torch.randperm(len(X))
            for i in range(0, len(X), args.batch_size):
                idx = perm[i:i + args.batch_size]
                x = torch.from_numpy(X[idx]).to(device)
                y = torch.from_numpy(Y[idx]).to(device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, model.n_classes),
                    y.reshape(-1))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += float(loss.item()); n_batches += 1
            del X, Y

        # Eval (batched to fit in memory)
        model.eval()
        correct = 0; total = 0
        per_cell_correct = np.zeros(64, dtype=np.int64)
        per_cell_total = 0
        with torch.no_grad():
            for i in range(0, len(ev_Xt), args.batch_size):
                xb = ev_Xt[i:i + args.batch_size]
                yb = ev_Yt[i:i + args.batch_size]
                preds = model(xb).argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
                per_cell_correct += (preds == yb).sum(dim=0).cpu().numpy()
                per_cell_total += len(yb)
        acc = correct / total
        per_cell_acc = per_cell_correct / max(per_cell_total, 1)
        dt = time.time() - t0
        print(f"Epoch {epoch}: train_loss={epoch_loss/max(n_batches,1):.5f}  "
              f"eval_acc={acc:.4%}  time={dt:.0f}s", flush=True)
        # Per-region
        for label in ('corner', 'edge', 'inner', 'center'):
            cells = [c for c in range(64) if cell_class(c) == label]
            mean = np.mean([per_cell_acc[c] for c in cells])
            print(f"  {label:>8s}: mean={mean:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'hidden': args.hidden,
                'in_dim': in_dim,
                'best_acc': best_acc,
                'per_cell_acc': per_cell_acc,
                'use_mod2': args.use_mod2,
            }, save_path)
            print(f"  Saved {save_path}", flush=True)

    print()
    print(f"=== Final best eval accuracy: {best_acc:.4%} ===")

    # Turn-stratified
    print()
    print("Per-region accuracy by turn (final epoch):")
    print(f"{'turn':>8s} {'n':>8s} {'overall':>9s} "
          f"{'corner':>9s} {'edge':>9s} {'inner':>9s} {'center':>9s}")
    bins = [(5, 10), (10, 15), (15, 20), (20, 25),
            (25, 30), (30, 35), (35, 40), (40, 45), (45, 54)]
    model.eval()
    with torch.no_grad():
        all_preds = torch.zeros_like(ev_Yt)
        for i in range(0, len(ev_Xt), args.batch_size):
            all_preds[i:i + args.batch_size] = model(ev_Xt[i:i + args.batch_size]).argmax(dim=-1)
    pos_np = ev_pos.astype(np.int64)
    match = (all_preds == ev_Yt).cpu().numpy()
    for lo, hi in bins:
        m = (pos_np >= lo) & (pos_np < hi)
        if m.sum() == 0: continue
        per_cell = match[m].mean(axis=0)
        overall = per_cell.mean()
        def reg_mean(label):
            cells = [c for c in range(64) if cell_class(c) == label]
            return per_cell[cells].mean()
        print(f"  {lo:>3d}-{hi-1:<3d} {int(m.sum()):>8d} {overall:>9.4f} "
              f"{reg_mean('corner'):>9.4f} {reg_mean('edge'):>9.4f} "
              f"{reg_mean('inner'):>9.4f} {reg_mean('center'):>9.4f}")
