"""Re-evaluate the saved MLP probe by pooling random positions across
MANY chunks (rather than only chunk_0039). Tests whether chunk_0039 is
representative or whether systematic per-chunk differences are biasing
the per-cell accuracy measurements.

Usage:
    python analyze_probe_multi_chunk.py \\
        --ckpt experiments/.../pattern_simple_direct_H512_wheneven.pt \\
        --probe experiments/.../probe_direct_H512_wheneven.pt \\
        --hidden 512 --n-chunks 10 --positions-per-chunk 5000
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


CENTER_64 = {27, 28, 35, 36}


def cell_class(c64):
    if c64 in CENTER_64: return 'center'
    r, c = c64 // 8, c64 % 8
    if r in (0, 7) and c in (0, 7): return 'corner'
    if r in (0, 7) or c in (0, 7):  return 'edge'
    return 'inner'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--features", default="when+even")
    parser.add_argument("--n-chunks", type=int, default=10,
                        help="Spread eval across this many chunks.")
    parser.add_argument("--positions-per-chunk", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ck = torch.load(args.ckpt, map_location=device)
    input_dim = ck.get('input_dim', 120)
    me = DirectMLP(input_dim, args.hidden, 960).to(device); me.load_state_dict(ck['even']); me.eval()
    mo = DirectMLP(input_dim, args.hidden, 960).to(device); mo.load_state_dict(ck['odd']); mo.eval()

    probe_ck = torch.load(args.probe, map_location='cpu')
    print(f"Loaded probe, reported best_acc={probe_ck.get('best_acc'):.4f}")
    probe_even = nn.Linear(args.hidden, 64 * 3).to(device); probe_even.load_state_dict(probe_ck['even']); probe_even.eval()
    probe_odd  = nn.Linear(args.hidden, 64 * 3).to(device); probe_odd .load_state_dict(probe_ck['odd']); probe_odd .eval()

    out_dir = "experiments/mathematical_transformation_experiments/heuristic_probe_results"
    chunk_dir = os.path.join(out_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f
                         and "_when60" not in f and "_by_black" not in f)
    # Pick `n_chunks` chunks evenly spaced
    n_avail = len(chunk_files)
    idxs = np.linspace(0, n_avail - 1, args.n_chunks).astype(int)
    chunks_to_use = [chunk_files[i] for i in idxs]
    print(f"Using {len(chunks_to_use)} chunks (spread across all {n_avail}):")
    for c in chunks_to_use:
        print(f"  {os.path.basename(c)}")

    # Per-chunk + total accumulation
    per_chunk_acc = np.zeros((len(chunks_to_use), 64))
    correct_total = np.zeros(64, dtype=np.int64)
    total_total = 0
    rng = np.random.RandomState(args.seed)

    for ci, cf in enumerate(chunks_to_use):
        print(f"\n[{ci+1}/{len(chunks_to_use)}] Loading {os.path.basename(cf)}...")
        X, Y, pos = _load_features(cf)
        feat = X[:, N_MOVES:3 * N_MOVES]
        del X
        n = min(args.positions_per_chunk, len(Y))
        si = np.sort(rng.choice(len(Y), n, replace=False))
        feat = feat[si]; Y_np = Y[si].numpy(); pos_np = pos[si].numpy()

        correct_c = np.zeros(64, dtype=np.int64)
        batch = 4096
        with torch.no_grad():
            for i in range(0, n, batch):
                xb = feat[i:i+batch].to(device)
                yb = Y_np[i:i+batch]
                pb = pos_np[i:i+batch]
                em = (pb % 2 == 0); om = ~em
                preds = np.zeros((len(xb), 64), dtype=np.int64)
                if em.any():
                    h = torch.relu(me.net[0](xb[em]))
                    preds[em] = probe_even(h).view(-1, 64, 3).argmax(dim=-1).cpu().numpy()
                if om.any():
                    h = torch.relu(mo.net[0](xb[om]))
                    preds[om] = probe_odd(h).view(-1, 64, 3).argmax(dim=-1).cpu().numpy()
                correct_c += (preds == yb).sum(axis=0)
        per_chunk_acc[ci] = correct_c / n
        correct_total += correct_c
        total_total += n

    pooled_acc = correct_total / max(total_total, 1)

    print()
    print("=" * 80)
    print("Per-chunk MEAN per-cell accuracy (averaged over 64 cells)")
    print("=" * 80)
    print(f"{'chunk':>20s} {'mean_acc':>10s} {'center':>10s} {'inner':>10s} "
          f"{'edge':>10s} {'corner':>10s}")
    for ci, cf in enumerate(chunks_to_use):
        acc = per_chunk_acc[ci]
        def reg_mean(label):
            cells = [c for c in range(64) if cell_class(c) == label]
            return acc[cells].mean()
        print(f"  {os.path.basename(cf):>18s} {acc.mean():>10.4f} "
              f"{reg_mean('center'):>10.4f} {reg_mean('inner'):>10.4f} "
              f"{reg_mean('edge'):>10.4f} {reg_mean('corner'):>10.4f}")

    print("-" * 80)
    print(f"{'POOLED':>20s} {pooled_acc.mean():>10.4f} ", end="")
    for label in ('center', 'inner', 'edge', 'corner'):
        cells = [c for c in range(64) if cell_class(c) == label]
        print(f"{pooled_acc[cells].mean():>10.4f} ", end="")
    print()

    # Variation summary
    print()
    print("Across-chunk variation (std of per-chunk means):")
    print(f"  overall: {per_chunk_acc.mean(axis=1).std():.4f}")
    for label in ('center', 'inner', 'edge', 'corner'):
        cells = [c for c in range(64) if cell_class(c) == label]
        per_chunk_reg = per_chunk_acc[:, cells].mean(axis=1)
        print(f"  {label:>6s}: {per_chunk_reg.std():.4f}  "
              f"(min {per_chunk_reg.min():.4f}, max {per_chunk_reg.max():.4f})")
