"""Plot training-curve figures for the two transfer-task experiments.

Reads:
  experiments/incoherent_rules/{variant}_n{n_rules}.json   (Task B)
  experiments/new_squares/cond_*.json                      (Task A)

Produces two PNG files, one per task:

  Rule corruption — IL_acc over fine-tune steps for Task B. Coherent vs
    incoherent are drawn as paired curves; default n_rules level is 100
    (use --all-scales to draw every level instead).
    Default output: figs/rule_corruption.png

  New squares — IL_acc over fine-tune steps for Task A. Two curves:
    coherent ninth-row vs incoherent random-neighbor placement.
    Default output: figs/new_squares.png

Usage:
    python plot_fig1ef.py
    # or with custom paths:
    python plot_fig1ef.py --rules-output X.png --squares-output Y.png
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def load_dir(path, pattern="*.json"):
    """Return list of dicts loaded from JSON files in `path`."""
    out = []
    for f in sorted(glob.glob(os.path.join(path, pattern))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def panel_rules(ax, conds, scale_filter=None):
    """Rule-corruption panel. One pair of curves per n_rules level (or just `scale_filter`)."""
    by_scale = defaultdict(dict)  # n_rules -> {coherent: cond, incoherent: cond}
    for c in conds:
        by_scale[c["n_rules"]][c["variant"]] = c
    scales = sorted(by_scale.keys()) if scale_filter is None else [scale_filter]

    for n_rules in scales:
        pair = by_scale.get(n_rules, {})
        for variant, color, ls in [("coherent", "tab:green", "-"),
                                   ("incoherent", "tab:red", "--")]:
            c = pair.get(variant)
            if c is None:
                continue
            label = f"{variant} (n_rules={n_rules})" if scale_filter is None else variant
            ax.plot(c["eval_steps"], c["IL_acc"], color=color, linestyle=ls,
                    marker="o", markersize=3, label=label)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Fine-tune step")
    ax.set_ylabel("IL accuracy")
    ax.set_title("Rule corruption")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)


def panel_squares(ax, conds):
    """New-squares panel. Two curves: coherent ninth row vs incoherent random neighbors."""
    by_name = {c["condition_name"]: c for c in conds}
    for variant, color, ls in [("coherent", "tab:green", "-"),
                               ("incoherent", "tab:red", "--")]:
        c = by_name.get(variant)
        if c is None:
            continue
        ax.plot(c["eval_steps"], c["IL_acc"], color=color, linestyle=ls,
                marker="o", markersize=3, label=variant)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Fine-tune step")
    ax.set_ylabel("IL accuracy")
    ax.set_title("New squares (ninth row)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rules-dir", default="experiments/incoherent_rules")
    p.add_argument("--squares-dir", default="experiments/new_squares")
    p.add_argument("--rules-scale", type=int, default=100,
                   help="n_rules level to plot for rule corruption. Use --all-scales to override.")
    p.add_argument("--all-scales", action="store_true",
                   help="Draw every available n_rules level for rule corruption.")
    p.add_argument("--rules-output", default="figs/rule_corruption.png")
    p.add_argument("--squares-output", default="figs/new_squares.png")
    args = p.parse_args()

    rules_conds = load_dir(args.rules_dir)
    squares_conds = load_dir(args.squares_dir)

    fig_r, ax_r = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    panel_rules(ax_r, rules_conds,
                scale_filter=None if args.all_scales else args.rules_scale)
    _save(fig_r, args.rules_output)

    fig_s, ax_s = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    panel_squares(ax_s, squares_conds)
    _save(fig_s, args.squares_output)


if __name__ == "__main__":
    main()
