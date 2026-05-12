"""Summarize the 20 per-turn probes (10 specialist + 10 unified) trained by
exp_per_turn_probes.sh. Each probe was saved with per_cell_acc in its
checkpoint dict. We just need to load them and tabulate.

Usage:
    python summarize_per_turn_probes.py
"""
import os, torch
import numpy as np

CKPT_DIR = "experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints"
TURNS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
CENTER_64 = {27, 28, 35, 36}


def cell_class(c64):
    if c64 in CENTER_64: return 'center'
    r, c = c64 // 8, c64 % 8
    if r in (0, 7) and c in (0, 7): return 'corner'
    if r in (0, 7) or c in (0, 7):  return 'edge'
    return 'inner'


def region_means(per_cell):
    """Returns dict region -> mean acc."""
    out = {}
    for label in ('corner', 'edge', 'inner', 'center'):
        cells = [c for c in range(64) if cell_class(c) == label]
        out[label] = float(np.mean([per_cell[c] for c in cells]))
    return out


def load_probe_meta(path):
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location='cpu')
    pca = d.get('per_cell_acc')
    if pca is None:
        return None
    return {
        'best_acc': d.get('best_acc'),
        'target_turn': d.get('target_turn'),
        'per_cell': np.array(pca, dtype=np.float64),
    }


if __name__ == "__main__":
    print(f"{'turn':>5s} {'mean_S':>8s} {'mean_U':>8s} {'Δ':>7s}  "
          f"{'ctr_S':>8s} {'ctr_U':>8s} {'Δ':>7s}  "
          f"{'inn_S':>8s} {'inn_U':>8s} {'Δ':>7s}  "
          f"{'edge_S':>8s} {'edge_U':>8s} {'Δ':>7s}")
    print("-" * 130)
    for T in TURNS:
        spec_path = (f"{CKPT_DIR}/probe_direct_H512_wheneven_turn{T}"
                     f"_turnprobe{T}.pt")
        unif_path = (f"{CKPT_DIR}/probe_direct_H512_wheneven_turnprobe{T}.pt")

        spec = load_probe_meta(spec_path)
        unif = load_probe_meta(unif_path)

        if spec is None or unif is None:
            missing = []
            if spec is None: missing.append(f"spec-T{T}")
            if unif is None: missing.append(f"unif-T{T}")
            print(f"  {T:>3d}  -- missing: {', '.join(missing)} --")
            continue

        s_mean = float(np.mean(spec['per_cell']))
        u_mean = float(np.mean(unif['per_cell']))
        s_reg = region_means(spec['per_cell'])
        u_reg = region_means(unif['per_cell'])

        line = (f"  {T:>3d} "
                f"{s_mean:>8.4f} {u_mean:>8.4f} {s_mean - u_mean:>+7.4f}  "
                f"{s_reg['center']:>8.4f} {u_reg['center']:>8.4f} "
                f"{s_reg['center'] - u_reg['center']:>+7.4f}  "
                f"{s_reg['inner']:>8.4f} {u_reg['inner']:>8.4f} "
                f"{s_reg['inner'] - u_reg['inner']:>+7.4f}  "
                f"{s_reg['edge']:>8.4f} {u_reg['edge']:>8.4f} "
                f"{s_reg['edge'] - u_reg['edge']:>+7.4f}")
        print(line)
