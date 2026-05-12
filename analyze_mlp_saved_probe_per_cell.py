"""Per-cell accuracy of the saved Nanda-style probe on the MLP hidden.

Uses probe_direct_H512_wheneven.pt (the 99.25%-trained probe) instead of
refitting a conservative Ridge probe inline. Tests whether the "center
cells are weak" pattern from probe_per_cell.py was a probe-quality
artifact (as it turned out to be for OGPT) or a real property of the
MLP's internal representation.

Outputs the 8x8 per-cell accuracy grid + summary by region (corner /
edge / inner / center).

Usage:
    python analyze_mlp_saved_probe_per_cell.py \\
        --ckpt experiments/.../pattern_simple_direct_H512_wheneven.pt \\
        --probe experiments/.../probe_direct_H512_wheneven.pt \\
        --hidden 512
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES,
)
from train_pattern_simple import DirectMLP


_FEAT_COLS = {
    "when":      list(range(N_MOVES, 2 * N_MOVES)),
    "when+even": list(range(N_MOVES, 3 * N_MOVES)),
    "all":       list(range(0, 3 * N_MOVES)),
}
CENTER_64 = {27, 28, 35, 36}


def cell_class(c64):
    if c64 in CENTER_64: return 'center'
    r, col = c64 // 8, c64 % 8
    if r in (0, 7) and col in (0, 7): return 'corner'
    if r in (0, 7) or col in (0, 7):  return 'edge'
    return 'inner'


def cell_alg(c64):
    return f"{'abcdefgh'[c64 % 8]}{c64 // 8 + 1}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--features", default="when+even")
    parser.add_argument("--n-positions", type=int, default=50000)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model
    ck = torch.load(args.ckpt, map_location=device)
    input_dim = ck.get('input_dim', 120)
    n_patterns = ck.get('n_patterns', 960)
    me = DirectMLP(input_dim, args.hidden, n_patterns).to(device); me.load_state_dict(ck['even']); me.eval()
    mo = DirectMLP(input_dim, args.hidden, n_patterns).to(device); mo.load_state_dict(ck['odd']); mo.eval()
    print(f"Loaded MLP from {args.ckpt}")

    # Load probe
    probe_ck = torch.load(args.probe, map_location='cpu')
    print(f"Loaded probe from {args.probe}, best_acc={probe_ck.get('best_acc'):.4f}")
    # nn.Linear(H, 64*3); weight shape (64*3, H)
    probe_even = nn.Linear(args.hidden, 64 * 3).to(device); probe_even.load_state_dict(probe_ck['even']); probe_even.eval()
    probe_odd  = nn.Linear(args.hidden, 64 * 3).to(device); probe_odd .load_state_dict(probe_ck['odd']); probe_odd .eval()

    # Load feature chunk (last as eval)
    out_dir = "experiments/mathematical_transformation_experiments/heuristic_probe_results"
    chunk_dir = os.path.join(out_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    eval_path = chunk_files[-1]
    print(f"Loading {eval_path}")

    X, Y, pos = _load_features(eval_path)
    feat_X = X[:, _FEAT_COLS[args.features]]
    del X
    N = len(Y)
    n = min(args.n_positions, N)
    rng = np.random.RandomState(0)
    si = np.sort(rng.choice(N, n, replace=False))
    feat_X = feat_X[si]; Y_np = Y[si].numpy(); pos_np = pos[si].numpy()
    print(f"Sampled {n} positions")

    # Forward + apply probe, accumulate per-cell hits
    correct = np.zeros(64, dtype=np.int64)
    total   = np.zeros(64, dtype=np.int64)
    batch = 4096
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = feat_X[i:i+batch].to(device)
            yb = Y_np[i:i+batch]
            pb = pos_np[i:i+batch]
            em = (pb % 2 == 0); om = ~em
            preds = np.zeros((len(xb), 64), dtype=np.int64)
            if em.any():
                xs = xb[em]
                h = torch.relu(me.net[0](xs))
                logits = probe_even(h).view(-1, 64, 3)
                preds[em] = logits.argmax(dim=-1).cpu().numpy()
            if om.any():
                xs = xb[om]
                h = torch.relu(mo.net[0](xs))
                logits = probe_odd(h).view(-1, 64, 3)
                preds[om] = logits.argmax(dim=-1).cpu().numpy()
            correct += (preds == yb).sum(axis=0)
            total   += yb.shape[0]

    acc = correct / total
    # Print 8x8 grid
    print()
    print("Per-cell accuracy of saved MLP probe (99.25%-trained Nanda-style):")
    print("       " + "   ".join(c for c in "abcdefgh"))
    for r in range(8):
        row = []
        for c in range(8):
            sq = r * 8 + c
            v = acc[sq]
            tag = "*" if sq in CENTER_64 else " "
            row.append(f"{v:.3f}{tag}")
        print(f"  {r+1}  " + " ".join(row))

    print(f"\nMean: {acc.mean():.4f}  Std: {acc.std():.4f}  "
          f"Min: {acc.min():.4f}  Max: {acc.max():.4f}")

    for label in ('corner', 'edge', 'inner', 'center'):
        cells = [c for c in range(64) if cell_class(c) == label]
        vals = acc[cells]
        print(f"  {label:>6s} n={len(cells):2d}  mean={vals.mean():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")

    order = np.argsort(acc)
    print("\nWorst 10 cells:")
    for c in order[:10]:
        print(f"  {cell_alg(int(c)):>3s} ({cell_class(int(c)):>6s})  acc={acc[c]:.4f}")
    print("Best 10 cells:")
    for c in order[-10:][::-1]:
        print(f"  {cell_alg(int(c)):>3s} ({cell_class(int(c)):>6s})  acc={acc[c]:.4f}")
