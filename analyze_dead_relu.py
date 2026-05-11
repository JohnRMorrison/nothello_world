"""Quick dead-ReLU diagnostic on the H=1024 wheneven hidden layer.

For each of H hidden units, computes the fraction of input positions where
its activation is exactly 0 (dead). Reports the dead-unit count at several
thresholds; if a large fraction is dead >99% of the time, effective capacity
is much lower than H and a wider model or different activation/init would
unlock real capacity.

Usage:
    python analyze_dead_relu.py --ckpt <path> --hidden 1024
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES,
)
from train_pattern_simple import (
    DirectMLP, EndToEndMLP, TwoStageMLP,
)


_FEAT_COLS = {
    "when":      list(range(N_MOVES, 2 * N_MOVES)),
    "when+even": list(range(N_MOVES, 3 * N_MOVES)),
    "all":       list(range(0, 3 * N_MOVES)),
}


def get_hidden(model, x):
    lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return torch.relu(lins[0](x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--mode", default="direct")
    parser.add_argument("--features", default="when+even")
    parser.add_argument("--n-positions", type=int, default=20000)
    args = parser.parse_args()

    output_dir = "experiments/mathematical_transformation_experiments/heuristic_probe_results"
    chunk_dir = os.path.join(output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f
                         and "_when60" not in f)
    eval_path = chunk_files[-1]
    print(f"Loading {eval_path}")

    X, Y, pos = _load_features(eval_path)
    N = len(Y)
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    input_dim = ckpt.get('input_dim', N_MOVES)
    n_patterns = ckpt.get('n_patterns', 960)

    Cls = {"direct": DirectMLP, "emergent": EndToEndMLP, "e2e": EndToEndMLP,
           "two-stage": TwoStageMLP, "randproj": DirectMLP}[args.mode]
    me = Cls(input_dim, args.hidden, n_patterns).to(device)
    mo = Cls(input_dim, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()
    print(f"Loaded {args.ckpt}")

    feat_X = X[:, _FEAT_COLS[args.features]]
    del X

    rng = np.random.RandomState(0)
    n = min(args.n_positions, N)
    si = np.sort(rng.choice(N, n, replace=False))
    feat_X = feat_X[si]; pos = pos[si]

    H_em, H_om = [], []
    batch = 4096
    em_total = om_total = 0
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = feat_X[i:i + batch].to(device)
            pb = pos[i:i + batch]
            em = (pb % 2 == 0); om = ~em
            if em.any():
                H_em.append(get_hidden(me, xb[em]).cpu().numpy())
                em_total += int(em.sum())
            if om.any():
                H_om.append(get_hidden(mo, xb[om]).cpu().numpy())
                om_total += int(om.sum())

    H_em = np.concatenate(H_em, axis=0) if H_em else np.zeros((0, args.hidden))
    H_om = np.concatenate(H_om, axis=0) if H_om else np.zeros((0, args.hidden))
    print(f"\nCollected hidden activations: ME {H_em.shape}, MO {H_om.shape}")

    def report(H, label):
        # Fraction of positions where each unit is exactly 0
        zero_frac = (H == 0).mean(axis=0)
        mean_act  = H.mean(axis=0)
        max_act   = H.max(axis=0)
        nonzero_mean_per_unit = np.where(H > 0, H, np.nan)
        per_unit_mean_when_active = np.nanmean(nonzero_mean_per_unit, axis=0)

        print(f"\n=== {label} (H={H.shape[1]}, n_positions={H.shape[0]}) ===")
        print(f"Mean fraction-of-time-zero across units: {zero_frac.mean():.4f}")
        print(f"Median fraction-of-time-zero:            {np.median(zero_frac):.4f}")
        print()
        thresholds = [0.50, 0.80, 0.90, 0.95, 0.99, 0.999, 1.0]
        print(f"{'zero_frac >=':>14s} {'# units dead':>14s} {'% of H':>10s}")
        for t in thresholds:
            n_dead = int((zero_frac >= t).sum())
            print(f"{t:>14.3f} {n_dead:>14d} {n_dead/H.shape[1]:>10.2%}")
        # Effective capacity rough estimate: count units active at least 1% of time
        live = int((zero_frac < 0.99).sum())
        print(f"\nEffective live units (active >1% of positions): "
              f"{live} / {H.shape[1]}  ({live/H.shape[1]:.1%})")
        print(f"Max activation per unit -- top 5: "
              f"{np.sort(max_act)[-5:].tolist()}")
        print(f"Max activation per unit -- median: {float(np.median(max_act)):.3f}")
        print(f"Active-mean per unit median: "
              f"{float(np.nanmedian(per_unit_mean_when_active)):.3f}")

    report(H_em, "ME (even-turn model)")
    report(H_om, "MO (odd-turn model)")
