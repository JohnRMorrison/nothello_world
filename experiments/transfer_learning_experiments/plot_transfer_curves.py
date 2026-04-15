"""
Plot transfer learning curves across conditions.

Supports two input modes:

  1. Legacy: --results path1 path2 ...  (one-row plot, one panel per metric)
  2. 2x2:    --curves-dir DIR           (auto-grouped by condition + mode;
                                         produces 2x2 grid + headline overlay)

The 2x2 layout has rows = antecedent {aligned, random} and columns =
consequent {aligned, random}, with ft and scratch runs overlaid per cell.

Usage:
    python plot_transfer_curves.py \\
        --curves-dir results/2x2_run1/ --out figures/2x2_run1/

    python plot_transfer_curves.py \\
        --results results/aligned_ft.json results/random_ft.json \\
        --output transfer_comparison.png
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 2x2 layout
# ---------------------------------------------------------------------------

CONDITIONS = ["B1", "B2", "B3", "C"]
GRID_POSITION = {
    # (row, col) where row ∈ {aligned-ant=0, random-ant=1},
    #           col ∈ {aligned-cons=0, random-cons=1}
    "B1": (0, 0),
    "B2": (0, 1),
    "B3": (1, 0),
    "C":  (1, 1),
}
CONDITION_NAME = {
    "B1": "Aligned Ant × Aligned Cons",
    "B2": "Aligned Ant × Random Cons",
    "B3": "Random Ant × Aligned Cons",
    "C":  "Random Ant × Random Cons",
}
CONDITION_COLOR = {
    "B1": "#1f77b4",  # blue
    "B2": "#ff7f0e",  # orange
    "B3": "#2ca02c",  # green
    "C":  "#d62728",  # red
}
MODE_STYLE = {
    "ft":      {"linestyle": "-",  "alpha": 1.0, "linewidth": 2.0},
    "scratch": {"linestyle": "--", "alpha": 0.6, "linewidth": 1.5},
}

METRIC_LABELS = {
    "top1_legal": "Top-1 Legal Accuracy (all positions)",
    "top1_legal_when_fires": "Top-1 Legal Accuracy (restriction fires)",
    "violation_rate": "Violation Rate (all positions)",
    "violation_rate_when_fires": "Violation Rate (restriction fires)",
    "fire_rate": "Restriction Fire Rate",
    "top1_prob":  "Prob. of Best Legal Move",
    "legal_mass": "Total Legal Probability Mass",
    "train_loss": "Training Loss",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_curves(result_files):
    """Load result JSONs into a list of experiment dicts."""
    experiments = []
    for path in result_files:
        with open(path) as f:
            data = json.load(f)
        experiments.append({
            "label": data.get("label", os.path.basename(path)),
            "condition": data.get("condition"),
            "mode": data.get("mode"),
            "mode_canonical": data.get(
                "mode_canonical",
                "scratch" if data.get("mode") in ("rnd", "scratch") else "ft",
            ),
            "path": path,
            "curves": data["curves"],
        })
    return experiments


def load_curves_dir(curves_dir):
    """Load all result JSONs from a directory (recursively)."""
    paths = sorted(
        glob.glob(os.path.join(curves_dir, "*.json"))
        + glob.glob(os.path.join(curves_dir, "**", "*.json"), recursive=True)
    )
    paths = list(dict.fromkeys(paths))  # dedupe, preserve order
    return load_curves(paths)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_metric_multi(curves_dicts, metric):
    """Aggregate across a list of curves dicts (one per run-file).

    Returns (steps_array, mean, sem) where each run's values are aligned to
    the first run's step schedule via interpolation if needed.
    """
    runs = []
    for curves_dict in curves_dicts:
        for _, curve in sorted(curves_dict.items()):
            steps = [p["step"] for p in curve]
            values = [p.get(metric) for p in curve]
            if any(v is None for v in values):
                continue
            runs.append((steps, values))

    if not runs:
        return np.array([]), np.array([]), np.array([])

    ref_steps = runs[0][0]
    matrix = []
    for steps, values in runs:
        if steps == ref_steps:
            matrix.append(values)
        else:
            matrix.append(np.interp(ref_steps, steps, values).tolist())

    matrix = np.array(matrix)
    mean = matrix.mean(axis=0)
    if matrix.shape[0] > 1:
        sem = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
    else:
        sem = np.zeros_like(mean)
    return np.array(ref_steps), mean, sem


def smooth(values, window):
    """Simple centered moving average; returns unchanged if window <= 1."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    # Use 'same' convolution and trim edges conservatively.
    return np.convolve(values, kernel, mode="same")


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_by_condition_mode(experiments):
    """Return {(condition, mode_canonical): [curves_dict, ...]}."""
    grouped = defaultdict(list)
    for exp in experiments:
        cond = exp.get("condition")
        mode = exp.get("mode_canonical")
        if cond is None or mode is None:
            continue
        grouped[(cond, mode)].append(exp["curves"])
    return grouped


# ---------------------------------------------------------------------------
# 2x2 grid plot
# ---------------------------------------------------------------------------

def plot_2x2_grid(experiments, metric, out_path, smooth_window=1, max_step=None):
    grouped = group_by_condition_mode(experiments)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for cond, (r, c) in GRID_POSITION.items():
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

    # Shared axis labels.
    for ax in axes[-1, :]:
        ax.set_xlabel("Training Steps")
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    fig.suptitle(f"2x2 Factorial Transfer Curves — {METRIC_LABELS.get(metric, metric)}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved 2x2 grid -> {out_path}")
    plt.close(fig)


def plot_headline_overlay(experiments, metric, out_path,
                          smooth_window=1, max_step=None):
    """One-axis overlay of all four conditions (ft only)."""
    grouped = group_by_condition_mode(experiments)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in CONDITIONS:
        runs = grouped.get((cond, "ft"), [])
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
        ax.plot(steps, mean,
                label=f"{cond}: {CONDITION_NAME[cond]} (n={len(runs)})",
                color=color, linewidth=2)
        if sem.max() > 0:
            ax.fill_between(steps, mean - sem, mean + sem,
                            color=color, alpha=0.15)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"Headline: All conditions (ft) — {METRIC_LABELS.get(metric, metric)}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved headline overlay -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(experiments, metric="top1_legal"):
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
# Legacy one-row plot (unchanged API)
# ---------------------------------------------------------------------------

def plot_legacy(experiments, metrics, out_path, max_step=None, figsize=(14, 4)):
    style_map = {}
    colors = plt.cm.tab10.colors
    ft_styles = ["-", "--", "-.", ":"]
    rnd_styles = [(0, (1, 3)), (0, (3, 1, 1, 1))]

    ci = 0
    for exp in experiments:
        key = exp["label"]
        if key not in style_map:
            is_rnd = exp["mode_canonical"] == "scratch"
            style_map[key] = {
                "color": colors[ci % len(colors)],
                "linestyle": rnd_styles[ci % len(rnd_styles)] if is_rnd
                             else ft_styles[ci % len(ft_styles)],
                "alpha": 0.5 if is_rnd else 1.0,
            }
            ci += 1

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
    axes = axes[0]

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for exp in experiments:
            steps, mean, sem = extract_metric_multi([exp["curves"]], metric)
            if len(steps) == 0:
                continue
            if max_step is not None:
                mask = steps <= max_step
                steps, mean, sem = steps[mask], mean[mask], sem[mask]
            sty = style_map[exp["label"]]
            display = f"{exp['label']} ({exp['mode']})"
            ax.plot(steps, mean, label=display,
                    color=sty["color"], linestyle=sty["linestyle"],
                    alpha=sty["alpha"], linewidth=2)
            if sem.max() > 0:
                ax.fill_between(steps, mean - sem, mean + sem,
                                color=sty["color"], alpha=0.15)

        ax.set_xlabel("Training Steps")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.3)
        if mi == len(metrics) - 1:
            ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot transfer learning comparison curves (2x2 or legacy)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--curves-dir", type=str,
                     help="Directory of result JSONs (auto-groups by condition, mode)")
    src.add_argument("--results", nargs="+",
                     help="[legacy] Explicit result JSON paths")

    parser.add_argument("--out", type=str, default=None,
                        help="2x2 mode: output directory. Legacy mode: output file path.")
    parser.add_argument("--output", type=str, default=None,
                        help="[legacy alias for --out when --results is used]")
    parser.add_argument("--metric", type=str, default="top1_legal_when_fires",
                        help="Primary metric for the 2x2 grid / headline plot. "
                             "Default is top1_legal_when_fires (most sensitive to "
                             "the restriction task). Other useful choices: "
                             "top1_legal, violation_rate_when_fires, legal_mass.")
    parser.add_argument("--metrics", nargs="+",
                        default=["top1_legal_when_fires", "top1_legal",
                                 "violation_rate_when_fires", "legal_mass"],
                        help="Secondary metrics to plot as 2x2 grids")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Moving-average window (1 = no smoothing)")
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--figsize", type=float, nargs=2, default=[14, 4])
    args = parser.parse_args()

    if args.curves_dir:
        experiments = load_curves_dir(args.curves_dir)
        if not experiments:
            print(f"ERROR: no result JSONs found under {args.curves_dir}")
            return
        out_dir = args.out or os.path.join(args.curves_dir, "figures")
        os.makedirs(out_dir, exist_ok=True)

        # Primary metric: 2x2 grid + headline overlay.
        plot_2x2_grid(
            experiments, args.metric,
            os.path.join(out_dir, f"grid_{args.metric}.png"),
            smooth_window=args.smooth, max_step=args.max_step,
        )
        plot_headline_overlay(
            experiments, args.metric,
            os.path.join(out_dir, f"headline_{args.metric}.png"),
            smooth_window=args.smooth, max_step=args.max_step,
        )
        # Secondary metrics: grids only (skip headline to keep output small).
        for m in args.metrics:
            if m == args.metric:
                continue
            plot_2x2_grid(
                experiments, m,
                os.path.join(out_dir, f"grid_{m}.png"),
                smooth_window=args.smooth, max_step=args.max_step,
            )

        print_summary_table(experiments, metric=args.metric)
    else:
        experiments = load_curves(args.results)
        out_path = args.out or args.output or "transfer_comparison.png"
        plot_legacy(experiments, args.metrics, out_path,
                    max_step=args.max_step, figsize=tuple(args.figsize))


if __name__ == "__main__":
    main()
