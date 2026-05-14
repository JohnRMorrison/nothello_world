"""Load a saved probe checkpoint and report per-cell + per-region board
decoding accuracy. Reads the per_cell_acc array saved by
probe_pattern_models.py / probe_next_rule.py / probe_with_by_black.py.

Usage:
    python analyze_probe_per_cell.py PROBE_CKPT [PROBE_CKPT ...]
"""
import sys, os
import numpy as np
import torch


CENTER = {27, 28, 35, 36}


def cell_region(c):
    if c in CENTER: return 'center'
    r, col = c // 8, c % 8
    if r in (0, 7) and col in (0, 7): return 'corner'
    if r in (0, 7) or col in (0, 7):  return 'edge'
    return 'inner'


def analyze_one(path):
    ckpt = torch.load(path, map_location='cpu')
    label = os.path.basename(path).replace('.pt', '')
    print(f"\n=== {label} ===")
    if 'per_cell_acc' not in ckpt:
        print("  per_cell_acc not saved in this checkpoint")
        if 'best_acc' in ckpt:
            print(f"  best overall acc: {ckpt['best_acc']:.4%}")
        return

    pc = np.asarray(ckpt['per_cell_acc'], dtype=np.float64)
    if 'best_acc' in ckpt:
        print(f"  overall: {ckpt['best_acc']:.4%}  (saved best_acc)")
    print(f"  per_cell mean: {pc.mean():.4%}")

    # Per-region
    for label in ('corner', 'edge', 'inner', 'center'):
        cells = [c for c in range(64) if cell_region(c) == label]
        vals = np.array([pc[c] for c in cells])
        print(f"  {label:>6s}: n={len(cells):2d}  "
              f"mean={vals.mean():.4%}  min={vals.min():.4%}  max={vals.max():.4%}")

    # 8x8 grid
    print(f"  Per-cell grid:")
    grid = pc.reshape(8, 8)
    print("       " + " ".join(f"{c:6d}" for c in range(8)))
    for r in range(8):
        row = " ".join(f"{grid[r, c]*100:5.1f}%" for c in range(8))
        print(f"    {r}  {row}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_probe_per_cell.py PROBE_CKPT [PROBE_CKPT ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        analyze_one(path)
