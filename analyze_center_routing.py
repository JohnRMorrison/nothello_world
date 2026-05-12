"""For each of 60 target cells, count how many of its ~16 flanking patterns
route through center cells (d4/e4/d5/e5 as opp cell or terminal). Cross-
reference with per-cell recall@K from logs/perpos_H1024_wheneven.npz to
test the hypothesis: cells whose patterns mostly route through center are
exactly the ones the MLP misses on recall@K.

Usage:
    python analyze_center_routing.py
"""
import sys, numpy as np
sys.path.insert(0, '.')

from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX

CENTER_64 = {27, 28, 35, 36}

# Build per-pattern: target cell (60-idx), routes_through_center (bool)
patterns = enumerate_flanking_patterns()
pat_target_60 = np.array([MOVE_TO_IDX[p['target']] for p in patterns])
pat_routes_center = np.array([
    any(c in CENTER_64 for c in p['opponents']) or p['terminal'] in CENTER_64
    for p in patterns
], dtype=bool)

# Per-cell: count patterns, count non-center patterns
n_total = np.zeros(60, dtype=int)
n_non_center = np.zeros(60, dtype=int)
for i in range(len(patterns)):
    n_total[pat_target_60[i]] += 1
    if not pat_routes_center[i]:
        n_non_center[pat_target_60[i]] += 1
frac_non_center = n_non_center / np.maximum(n_total, 1)

# Load per-position data
print("Loading logs/perpos_H1024_wheneven.npz...")
d = np.load('logs/perpos_H1024_wheneven.npz')
legal = d['legal_mask']                       # (N, 60) bool
scores = d['scores_prob_or']                  # (N, 60) float
pos = d['pos']                                # (N,) int
N = len(pos)
print(f"  N positions: {N}")

# For each position, compute top-K predictions (K = num legal)
print("Computing per-cell recall@K contributions...")
K_per_pos = legal.sum(axis=1)
in_topK = np.zeros((N, 60), dtype=bool)
# Vectorized argsort approach
ranks = np.argsort(-scores, axis=1)
for i in range(N):
    k = K_per_pos[i]
    if k > 0:
        in_topK[i, ranks[i, :k]] = True

# Per-cell recall: of positions where C is legal, fraction where C is in top-K
cell_recall = np.zeros(60)
cell_legal_count = legal.sum(axis=0)
for c in range(60):
    mask_c_legal = legal[:, c]
    if mask_c_legal.sum() > 0:
        cell_recall[c] = in_topK[mask_c_legal, c].mean()

# Print per-cell table
print()
print("=" * 78)
print("Per-cell recall vs fraction-of-non-center patterns")
print("=" * 78)
print(f"{'cell':>5s} {'n_total':>8s} {'n_non_ctr':>10s} {'frac_non_ctr':>12s} "
      f"{'recall':>8s} {'n_legal':>10s}")

# Sort by frac_non_center ascending so we see the worst cases first
order = np.argsort(frac_non_center)
movable_64 = [c for c in range(64) if c not in CENTER_64]
m60_to_alg = {}
for i, c64 in enumerate(movable_64):
    m60_to_alg[i] = f"{'abcdefgh'[c64 % 8]}{c64 // 8 + 1}"

for c in order:
    print(f"  {m60_to_alg[c]:>3s} {n_total[c]:>8d} {n_non_center[c]:>10d} "
          f"{frac_non_center[c]:>12.3f} {cell_recall[c]:>8.4f} "
          f"{int(cell_legal_count[c]):>10d}")

# Compute correlation
valid = cell_legal_count > 100
corr = np.corrcoef(frac_non_center[valid], cell_recall[valid])[0, 1]
print(f"\nPearson correlation (frac_non_center, recall): {corr:.4f}")

# Bin recall by quintile of frac_non_center
print(f"\n{'frac_non_ctr quintile':>22s} {'n_cells':>8s} "
      f"{'mean recall':>12s} {'mean n_total':>14s}")
quintiles = np.quantile(frac_non_center[valid], np.linspace(0, 1, 6))
for q in range(5):
    lo, hi = quintiles[q], quintiles[q + 1]
    if q == 4:
        m = valid & (frac_non_center >= lo) & (frac_non_center <= hi)
    else:
        m = valid & (frac_non_center >= lo) & (frac_non_center < hi)
    if m.sum() == 0:
        continue
    print(f"  [{lo:>5.2f}, {hi:>5.2f}]   {int(m.sum()):>8d} "
          f"{cell_recall[m].mean():>12.4f} {n_total[m].mean():>14.1f}")
