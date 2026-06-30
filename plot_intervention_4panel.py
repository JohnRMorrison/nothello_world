"""Combined 2x2 figure: four delta-margin survival curves for intervention.

For each intervention, classify each cell into one of four buckets:
  always-legal   : legal in both original and counterfactual boards
  always-illegal : illegal in both
  newly-legal    : legal only in the counterfactual board
  newly-illegal  : legal only in the original board

Then for each intervention sample, compute four margins per relevant cell:
  Panel (0,0):  P(newly legal) - max P(always illegal)
                  [HIGH = surgical: newly-legal cell ranks above all always-illegal]
  Panel (0,1):  min P(always legal) - P(newly legal)
                  [HIGH = surgical: weakest legal cell still beats newly-legal]
  Panel (1,0):  max P(always legal) - P(newly illegal)
                  [HIGH = surgical: top legal cell still beats newly-illegal]
  Panel (1,1):  P(newly illegal) - max P(always illegal)
                  [LOW (negative) = surgical: newly-illegal joins the illegal range]

Survival curves: y-axis = fraction of (sample, cell) values >= threshold.

All four panels share x-axis range and y-axis range so they're directly
comparable in one paper figure.

Usage:
    python plot_intervention_4panel.py \\
        --results-dir experiments/multi_intervention_probs \\
        --output experiments/plots/intervention_4panel.png
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_intervention_data(results_dir):
    path = os.path.join(results_dir, "raw_samples.json")
    with open(path) as f:
        data = json.load(f)
    samples = []
    for cond in data:
        for n_str in data[cond]:
            for s in data[cond][n_str]:
                if "orig_probs" not in s or "intv_probs" not in s:
                    continue
                if "original_legal" not in s or "counterfactual_legal" not in s:
                    continue
                samples.append({
                    "condition": cond,
                    "n": int(n_str),
                    "orig_probs": np.array(s["orig_probs"]),
                    "intv_probs": np.array(s["intv_probs"]),
                    "original_legal": set(s["original_legal"]),
                    "counterfactual_legal": set(s["counterfactual_legal"]),
                })
    print(f"Loaded {len(samples)} interventions from {path}")
    return samples


def classify_cells(s):
    orig = s["original_legal"]
    cf = s["counterfactual_legal"]
    always_legal = orig & cf
    newly_legal = cf - orig            # legal in counterfactual but not original
    newly_illegal = orig - cf          # legal in original but not counterfactual
    always_illegal = set(range(64)) - (orig | cf)
    return newly_legal, newly_illegal, always_legal, always_illegal


def compute_four_metrics(samples):
    """Returns four 1-D numpy arrays of metric values, one per (sample, cell)."""
    m1, m2, m3, m4 = [], [], [], []
    for s in samples:
        newly_legal, newly_illegal, always_legal, always_illegal = classify_cells(s)
        ip = s["intv_probs"]

        ai_probs = [ip[c] for c in always_illegal if ip[c] >= 0]
        al_probs = [ip[c] for c in always_legal   if ip[c] >= 0]
        if not ai_probs or not al_probs:
            continue
        max_ai = max(ai_probs)
        min_al = min(al_probs)
        max_al = max(al_probs)

        for c in newly_legal:
            if ip[c] < 0:
                continue
            m1.append(ip[c] - max_ai)     # P(NL) - max P(AI)
            m2.append(min_al - ip[c])      # min P(AL) - P(NL)
        for c in newly_illegal:
            if ip[c] < 0:
                continue
            m3.append(max_al - ip[c])      # max P(AL) - P(NI)
            m4.append(ip[c] - max_ai)      # P(NI) - max P(AI)
    return np.array(m1), np.array(m2), np.array(m3), np.array(m4)


def survival(values, thresholds):
    return np.array([np.mean(values >= t) for t in thresholds])


def plot_panel(ax, vals, thresholds, title, color):
    surv = survival(vals, thresholds)
    ax.plot(thresholds, surv, color=color, linewidth=2)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    # Annotate the fraction crossing zero
    frac_zero = np.mean(vals >= 0)
    ax.annotate(f'≥0: {frac_zero*100:.1f}%',
                xy=(0, frac_zero), xytext=(0.02, 0.95),
                xycoords=('data', 'axes fraction'),
                fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True,
                    help='Dir containing raw_samples.json')
    ap.add_argument('--output',
                    default='experiments/plots/intervention_4panel.png')
    ap.add_argument('--x-min', type=float, default=-0.20)
    ap.add_argument('--x-max', type=float, default=0.20)
    ap.add_argument('--label', default='OthelloGPT')
    args = ap.parse_args()

    samples = load_intervention_data(args.results_dir)
    m1, m2, m3, m4 = compute_four_metrics(samples)
    print(f"  N points per metric: m1={len(m1)}, m2={len(m2)}, "
          f"m3={len(m3)}, m4={len(m4)}")

    thresholds = np.linspace(args.x_min, args.x_max, 400)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8),
                              sharex=True, sharey=True)
    (ax00, ax01), (ax10, ax11) = axes

    plot_panel(ax00, m1, thresholds,
                "P(newly legal) − max P(always illegal)",
                color='#1f77b4')
    plot_panel(ax01, m2, thresholds,
                "min P(always legal) − P(newly legal)",
                color='#ff7f0e')
    plot_panel(ax10, m3, thresholds,
                "max P(always legal) − P(newly illegal)",
                color='#2ca02c')
    plot_panel(ax11, m4, thresholds,
                "P(newly illegal) − max P(always illegal)",
                color='#d62728')

    for ax in axes[1, :]:
        ax.set_xlabel('Probability margin (threshold)', fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel('Fraction of interventions ≥ threshold', fontsize=11)

    fig.suptitle(f'{args.label}: Intervention margin survival curves',
                 fontsize=13, y=1.00)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"Saved {args.output}")
    plt.close()

    # Quick text summary
    print()
    print(f"=== Summary ({args.label}) ===")
    for name, vals in [
        ("P(NL) - max P(AI)", m1),
        ("min P(AL) - P(NL)", m2),
        ("max P(AL) - P(NI)", m3),
        ("P(NI) - max P(AI)", m4),
    ]:
        if len(vals) == 0:
            print(f"  {name}: no data")
            continue
        print(f"  {name}:  mean={vals.mean():+.4f}  "
              f"median={np.median(vals):+.4f}  "
              f"frac≥0 = {np.mean(vals >= 0)*100:.1f}%")


if __name__ == '__main__':
    main()
