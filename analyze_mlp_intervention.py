"""Analyze the saved mlp_intervention.npz.

Reports:
  - Li-style top-K success rate (target appears in top-K of post-intervention
    cell probabilities, K = number of legal moves at that position). This is
    the direct comparison metric to Li et al. 2023.
  - Top-(K+1) variant for context.
  - eps>0 / delta>0 / joint (eps AND delta both positive, equivalent to top-K
    when probe boundary geometry holds).
  - If the npz includes per-position recall stratification, reports
    intervention success in 5 buckets of baseline recall@K so we can see
    whether the intervention success depends on the model already being
    good at the task or works equally on weak positions.

Usage:
    python analyze_mlp_intervention.py logs/mlp_intervention.npz
"""
import sys, numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "logs/mlp_intervention.npz"
d = np.load(path)
print(f"Loaded {path}\n")
print(f"keys: {sorted(d.files)}\n")

scales = sorted(set(float(k.split("_s")[1])
                    for k in d.files if k.startswith("eps_s")))
print(f"Scales found: {scales}\n")

have_pos_recall = any(k.startswith("pos_recall_s") for k in d.files)
have_li = any(k.startswith("li_topk_s") for k in d.files)

print("=" * 78)
print("Per-scale summary")
print("=" * 78)
hdr = f"{'scale':>6s} {'n':>8s} {'eps>0':>7s} {'delta>0':>9s} {'eps&delta':>11s}"
if have_li:
    hdr += f" {'top-K':>8s} {'top-K+1':>9s}"
hdr += f" {'mean eps':>10s} {'mean delta':>12s} {'crosstalk':>11s}"
print(hdr)

for s in scales:
    e = d[f"eps_s{s}"]
    de = d[f"delta_s{s}"]
    c = d[f"cross_s{s}"]
    joint = (e > 0) & (de > 0)
    line = (f"{s:>6.2f} {len(e):>8d} "
            f"{(e > 0).mean():>6.2%} {(de > 0).mean():>8.2%} "
            f"{joint.mean():>10.2%}")
    if have_li:
        tk  = d[f"li_topk_s{s}"]
        tkp = d[f"li_topkp1_s{s}"]
        line += f" {tk.mean():>7.2%} {tkp.mean():>8.2%}"
    line += f" {e.mean():>10.4f} {de.mean():>12.4f} {c.mean():>11.4f}"
    print(line)

if have_li:
    print()
    print(" Note: 'top-K' is Li's strict metric (target in top-K of "
          "post-intervention\n cell ranking, K = # legal moves). "
          "eps&delta is its quick proxy.")

if not have_pos_recall:
    print("\n(No pos_recall_s fields -- rerun mlp_intervention.py to get "
          "per-position-recall stratification.)")
else:
    print()
    print("=" * 78)
    print("Stratification by baseline per-position recall@K")
    print("=" * 78)
    # Bin by quintile of baseline pos_recall.
    pr_all = np.concatenate([d[f"pos_recall_s{s}"] for s in scales])
    edges = np.quantile(pr_all, np.linspace(0, 1, 6))
    # Slight nudges for inclusivity
    edges[0] -= 1e-6; edges[-1] += 1e-6
    print(f"\n(Position recall quintile edges: "
          + ", ".join(f"{x:.3f}" for x in edges) + ")\n")
    for s in scales:
        e  = d[f"eps_s{s}"]
        de = d[f"delta_s{s}"]
        pr = d[f"pos_recall_s{s}"]
        joint = (e > 0) & (de > 0)
        if have_li:
            tk = d[f"li_topk_s{s}"]
        print(f"  scale = {s}")
        print(f"  {'recall bin':>16s} {'n':>8s} {'eps>0':>7s} "
              f"{'delta>0':>9s} {'eps&delta':>11s}" +
              (f" {'top-K':>8s}" if have_li else ""))
        for q in range(5):
            lo, hi = edges[q], edges[q + 1]
            m = (pr >= lo) & (pr < hi)
            n = int(m.sum())
            if n == 0: continue
            row = (f"  [{lo:>5.2f},{hi:>5.2f}) {n:>8d} "
                   f"{(e[m] > 0).mean():>6.2%} {(de[m] > 0).mean():>8.2%} "
                   f"{joint[m].mean():>10.2%}")
            if have_li:
                row += f" {tk[m].mean():>7.2%}"
            print(row)
        print()
