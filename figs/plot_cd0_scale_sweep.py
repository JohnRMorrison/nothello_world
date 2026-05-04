"""Plot the δ > 0 / ε > 0 fractions vs the scale multiplier K
for the cd0 scale-sweep experiment.

Reads experiments/cd0_scale_sweep/k_*/raw_samples.json and produces
two line plots: one for ε (newly-legal cell vs strongest still-illegal),
one for δ (newly-legal cell vs weakest still-legal). The x-axis is the
multiplier K applied to the cd0 calibrated per-cell scales.

Usage:
    python figs/plot_cd0_scale_sweep.py
"""
import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def classify(s):
    orig = set(s["original_legal"])
    cf = set(s["counterfactual_legal"])
    return cf - orig, set(range(64)) - (orig | cf), orig & cf


def load_samples(path):
    data = json.load(open(path))
    out = []
    for cond in data:
        for n_str, lst in data[cond].items():
            for s in lst:
                if all(k in s for k in ("orig_probs", "intv_probs",
                                        "original_legal", "counterfactual_legal")):
                    out.append(s)
    return out


def collect_eps_dlt(samples):
    eps, dlt = [], []
    for s in samples:
        nl, si, sl = classify(s)
        ip = np.asarray(s["intv_probs"])
        si_probs = [ip[c] for c in si if ip[c] >= 0]
        sl_probs = [ip[c] for c in sl if ip[c] >= 0]
        if not si_probs or not sl_probs:
            continue
        max_si, min_sl = max(si_probs), min(sl_probs)
        for c in nl:
            p = ip[c]
            if p < 0:
                continue
            eps.append(p - max_si)
            dlt.append(p - min_sl)
    return np.array(eps), np.array(dlt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="experiments/cd0_scale_sweep")
    args = p.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.root, "k_*"))):
        m = re.search(r"k_(\d+(\.\d+)?)$", d)
        if not m:
            continue
        K = float(m.group(1))
        rs_path = os.path.join(d, "raw_samples.json")
        if not os.path.exists(rs_path):
            print(f"  [skip] no raw_samples.json in {d}")
            continue
        samples = load_samples(rs_path)
        eps, dlt = collect_eps_dlt(samples)
        rows.append({
            "K": K,
            "n": len(eps),
            "eps_pos": float((eps > 0).mean()) if len(eps) else float("nan"),
            "dlt_pos": float((dlt > 0).mean()) if len(dlt) else float("nan"),
            "eps_med": float(np.median(eps)) if len(eps) else float("nan"),
            "dlt_med": float(np.median(dlt)) if len(dlt) else float("nan"),
        })
        print(f"  K={K}: n={len(eps)}, "
              f"ε>0 = {100*rows[-1]['eps_pos']:.1f}%, "
              f"δ>0 = {100*rows[-1]['dlt_pos']:.1f}%, "
              f"med ε={rows[-1]['eps_med']:+.4f}, med δ={rows[-1]['dlt_med']:+.4f}")

    if not rows:
        raise SystemExit(f"No raw_samples.json found under {args.root}")

    rows.sort(key=lambda r: r["K"])
    Ks  = [r["K"] for r in rows]
    eps = [100 * r["eps_pos"] for r in rows]
    dlt = [100 * r["dlt_pos"] for r in rows]

    LINE = "#c84d3e"

    # Figure 1: ε > 0 vs K
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(Ks, eps, color=LINE, linewidth=2.2, marker="o", markersize=6)
    ax.set_xlabel("Scale multiplier K  (relative to cd0 calibrated scale)", fontsize=12)
    ax.set_ylabel("% of newly-legal cells with ε > 0", fontsize=12)
    ax.set_title("ε > 0 (above strongest still-illegal) vs intervention magnitude",
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim(-2, 102)
    out_eps = os.path.join(OUT_DIR, "cd0_scale_sweep_eps.png")
    fig.savefig(out_eps, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {out_eps}")

    # Figure 2: δ > 0 vs K
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(Ks, dlt, color=LINE, linewidth=2.2, marker="o", markersize=6)
    ax.set_xlabel("Scale multiplier K  (relative to cd0 calibrated scale)", fontsize=12)
    ax.set_ylabel("% of newly-legal cells with δ > 0", fontsize=12)
    ax.set_title("δ > 0 (above weakest still-legal) vs intervention magnitude",
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim(-2, 102)
    out_dlt = os.path.join(OUT_DIR, "cd0_scale_sweep_delta.png")
    fig.savefig(out_dlt, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {out_dlt}")


if __name__ == "__main__":
    main()
