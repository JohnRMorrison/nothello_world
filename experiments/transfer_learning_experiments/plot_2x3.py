"""
Plot a 2x3 factorial transfer-learning comparison: bag-of-heuristics arms
(B1/B2/B3/C) vs. flanking-antecedent arms (F1/F2), all sharing the same
aligned/random consequent squares per quadruple.

Layout:

    Grid (rows = antecedent kind; cols = consequent kind):

                   Cons: aligned  |  Cons: random
      Ant: heuristic-aligned  B1  |       B2
      Ant: random             B3  |       C
      Ant: flanking           F1  |       F2

    Overlays:
      - headline_B1_vs_F1.png        — the main contrast
      - gap_comparison.png           — (B1 - C) vs (F1 - C) over training
      - grid_<metric>.png            — 2x3 grid per metric

Reuses loading / metric extraction / smoothing from plot_transfer_curves.py
so its behavior stays in sync with the existing 2x2 plots.

Usage:
    python plot_2x3.py \\
        --curves-dir runs/2x2_20260415_160147/results \\
        --out       runs/2x2_20260415_160147/figures
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from plot_transfer_curves import (  # noqa: E402
    METRIC_LABELS,
    extract_metric_multi,
    group_by_condition_mode,
    load_curves_dir,
    smooth,
)


# ---------------------------------------------------------------------------
# Layout constants — mirror plot_transfer_curves but extend with F1/F2
# ---------------------------------------------------------------------------

CONDITIONS = ["B1", "B2", "B3", "C", "F1", "F2"]

GRID_POSITION_2X3 = {
    # (row, col). Rows: 0 = heuristic-aligned ant, 1 = random ant,
    #                  2 = flanking ant.
    # Cols: 0 = aligned cons, 1 = random cons.
    "B1": (0, 0),
    "B2": (0, 1),
    "B3": (1, 0),
    "C":  (1, 1),
    "F1": (2, 0),
    "F2": (2, 1),
}
CONDITION_NAME = {
    "B1": "Heuristic Ant × Aligned Cons",
    "B2": "Heuristic Ant × Random Cons",
    "B3": "Random Ant × Aligned Cons",
    "C":  "Random Ant × Random Cons",
    "F1": "Flanking Ant × Aligned Cons",
    "F2": "Flanking Ant × Random Cons",
}
CONDITION_COLOR = {
    "B1": "#1f77b4",  # blue
    "B2": "#ff7f0e",  # orange
    "B3": "#2ca02c",  # green
    "C":  "#d62728",  # red
    "F1": "#9467bd",  # purple
    "F2": "#8c564b",  # brown
}
MODE_STYLE = {
    "ft":      {"linestyle": "-",  "alpha": 1.0, "linewidth": 2.0},
    "scratch": {"linestyle": "--", "alpha": 0.6, "linewidth": 1.5},
}


# ---------------------------------------------------------------------------
# 2x3 grid
# ---------------------------------------------------------------------------

def plot_2x3_grid(experiments, metric, out_path, smooth_window=1, max_step=None):
    grouped = group_by_condition_mode(experiments)

    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True, sharey=True)
    for cond, (r, c) in GRID_POSITION_2X3.items():
        ax = axes[r, c]
        color = CONDITION_COLOR[cond]
        plotted = False
        for mode in ("ft", "scratch"):
            runs = grouped.get((cond, mode), [])
            if not runs:
                continue
            steps, mean, sem = extract_metric_multi(runs, metric)
            if len(steps) == 0:
                continue
            if max_step is not None:
                mask = steps <= max_step
                steps, mean, sem = steps[mask], mean[mask], sem[mask]
            mean = smooth(mean, smooth_window)
            sty = MODE_STYLE[mode]
            ax.plot(steps, mean, label=f"{mode} (n={len(runs)})",
                    color=color, **sty)
            if sem.max() > 0:
                ax.fill_between(steps, mean - sem, mean + sem,
                                color=color, alpha=0.15)
            plotted = True

        ax.set_title(f"{cond}: {CONDITION_NAME[cond]}", fontsize=10)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=8, loc="lower right")
        else:
            ax.text(0.5, 0.5, "(no runs)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray")

    for ax in axes[-1, :]:
        ax.set_xlabel("Training Steps")
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    fig.suptitle(
        f"2x3 Transfer Curves (Heuristic ↔ Random ↔ Flanking Antecedents) — "
        f"{METRIC_LABELS.get(metric, metric)}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved 2x3 grid -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Headline: B1 vs F1 (same consequent, different antecedent source)
# ---------------------------------------------------------------------------

def plot_headline_b1_vs_f1(experiments, metric, out_path,
                            smooth_window=1, max_step=None):
    """Overlay B1 / F1 / C learning curves on one axis."""
    grouped = group_by_condition_mode(experiments)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cond in ("B1", "F1", "C"):
        for mode in ("ft", "scratch"):
            runs = grouped.get((cond, mode), [])
            if not runs:
                continue
            steps, mean, sem = extract_metric_multi(runs, metric)
            if len(steps) == 0:
                continue
            if max_step is not None:
                mask = steps <= max_step
                steps, mean, sem = steps[mask], mean[mask], sem[mask]
            mean = smooth(mean, smooth_window)
            color = CONDITION_COLOR[cond]
            sty = MODE_STYLE[mode]
            ax.plot(
                steps, mean,
                label=f"{cond}-{mode}: {CONDITION_NAME[cond]} (n={len(runs)})",
                color=color, **sty,
            )
            if sem.max() > 0:
                ax.fill_between(steps, mean - sem, mean + sem,
                                color=color, alpha=0.12)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"Headline: B1 vs F1 (vs C) — {METRIC_LABELS.get(metric, metric)}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved headline B1 vs F1 -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gap-of-gaps: (aligned − null) for heuristic vs flanking antecedents
# ---------------------------------------------------------------------------

def plot_gap_comparison(experiments, metric, out_path,
                         smooth_window=1, max_step=None):
    """Plot (B1 - C) and (F1 - C) as signed gaps over training.

    Both curves live on the same axis. If the flanking hypothesis holds,
    the F1-C gap should match or exceed the B1-C gap (same consequent; only
    the antecedent source differs).
    """
    grouped = group_by_condition_mode(experiments)

    def _curve(cond, mode):
        runs = grouped.get((cond, mode), [])
        return extract_metric_multi(runs, metric) if runs else (
            np.array([]), np.array([]), np.array([]))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mode in ("ft", "scratch"):
        c_steps, c_mean, _ = _curve("C", mode)
        if len(c_steps) == 0:
            continue
        for aligned_cond in ("B1", "F1"):
            a_steps, a_mean, _ = _curve(aligned_cond, mode)
            if len(a_steps) == 0:
                continue
            # Align on common steps via interp
            common = a_steps
            c_aligned = np.interp(common, c_steps, c_mean) \
                if list(c_steps) != list(common) else c_mean
            gap = a_mean - c_aligned
            if max_step is not None:
                mask = common <= max_step
                common = common[mask]
                gap = gap[mask]
            gap = smooth(gap, smooth_window)
            color = CONDITION_COLOR[aligned_cond]
            sty = MODE_STYLE[mode]
            ax.plot(common, gap,
                    label=f"({aligned_cond} − C) [{mode}]",
                    color=color, **sty)

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(f"Gap: {METRIC_LABELS.get(metric, metric)}")
    ax.set_title("Gap-of-gaps: heuristic (B1) vs flanking (F1) alignment effect")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved gap comparison -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(experiments, metric):
    grouped = group_by_condition_mode(experiments)
    print(f"\n{'Condition':<6} {'Mode':<8} {'N':<4} "
          f"{'Step-0':<10} {'Final':<10} {'Delta':<10}")
    print("-" * 52)
    for cond in CONDITIONS:
        for mode in ("ft", "scratch"):
            runs = grouped.get((cond, mode), [])
            if not runs:
                continue
            steps, mean, _ = extract_metric_multi(runs, metric)
            if len(steps) == 0:
                continue
            first, last = mean[0], mean[-1]
            print(f"{cond:<6} {mode:<8} {len(runs):<4} "
                  f"{first:<10.4f} {last:<10.4f} {last - first:<+10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot 2x3 grid + headline overlays for the flanking "
                    "extension on top of an existing 2x2 run.")
    parser.add_argument("--curves-dir", type=str, required=True,
                        help="Directory of result JSONs produced by "
                             "finetune_and_evaluate.py (B1/B2/B3/C and F1/F2).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for figures "
                             "(defaults to <curves-dir>/../figures).")
    parser.add_argument("--metric", type=str, default="top1_legal_when_fires",
                        help="Primary metric for the grid + headline.")
    parser.add_argument("--metrics", nargs="+",
                        default=["top1_legal_when_fires", "top1_legal",
                                 "violation_rate_when_fires", "legal_mass"],
                        help="Secondary metrics (one grid each).")
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--max-step", type=int, default=None)
    args = parser.parse_args()

    experiments = load_curves_dir(args.curves_dir)
    if not experiments:
        print(f"ERROR: no result JSONs found under {args.curves_dir}")
        sys.exit(1)

    out_dir = args.out or os.path.join(args.curves_dir, "..", "figures")
    os.makedirs(out_dir, exist_ok=True)

    # Primary: 2x3 grid + headline + gap
    plot_2x3_grid(
        experiments, args.metric,
        os.path.join(out_dir, f"grid_{args.metric}_2x3.png"),
        smooth_window=args.smooth, max_step=args.max_step,
    )
    plot_headline_b1_vs_f1(
        experiments, args.metric,
        os.path.join(out_dir, "headline_B1_vs_F1.png"),
        smooth_window=args.smooth, max_step=args.max_step,
    )
    plot_gap_comparison(
        experiments, args.metric,
        os.path.join(out_dir, "gap_comparison.png"),
        smooth_window=args.smooth, max_step=args.max_step,
    )

    # Secondary metrics: grid only
    for m in args.metrics:
        if m == args.metric:
            continue
        plot_2x3_grid(
            experiments, m,
            os.path.join(out_dir, f"grid_{m}_2x3.png"),
            smooth_window=args.smooth, max_step=args.max_step,
        )

    print_summary_table(experiments, args.metric)


if __name__ == "__main__":
    main()
