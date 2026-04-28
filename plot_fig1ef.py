"""Reproduce paper Fig 1e,f from the JSON output of the two transfer experiments.

Reads:
  experiments/incoherent_rules/{variant}_n{n_rules}.json   (Task B)
  experiments/new_squares/cond_*.json                      (Task A)

Produces a single PNG with two panels:

  Panel e — Coherent vs Incoherent rule corruption
    For each n_rules level we plot IL_acc (probability mass on now-illegal
    cells under the corrupted rule set, at fine-tune step k) over training
    steps. Coherent vs incoherent are plotted as paired curves at each
    n_rules.

  Panel f — Coherent vs Incoherent ninth row (new squares)
    Two curves: coherent ninth-row vs incoherent random-neighbor placement.

Usage:
    python plot_fig1ef.py --output figs/fig1ef.png

Defaults pick n_rules=100 as the representative scale for panel e (matches
the 100-rules condition emphasized in the paper). Pass --rules-scale to
override or --all-scales to draw every n_rules level.
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
    """Panel (e). One pair of curves per n_rules level (or just `scale_filter`)."""
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
    ax.set_title("(e) Rule corruption")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)


def panel_squares(ax, conds):
    """Panel (f). Two curves: coherent ninth row vs incoherent random neighbors."""
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
    ax.set_title("(f) New squares (ninth row)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rules-dir", default="experiments/incoherent_rules")
    p.add_argument("--squares-dir", default="experiments/new_squares")
    p.add_argument("--rules-scale", type=int, default=100,
                   help="n_rules level to plot in panel (e). Use --all-scales to override.")
    p.add_argument("--all-scales", action="store_true",
                   help="Draw every available n_rules level in panel (e).")
    p.add_argument("--output", default="figs/fig1ef.png")
    args = p.parse_args()

    rules_conds = load_dir(args.rules_dir)
    squares_conds = load_dir(args.squares_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    panel_rules(axes[0], rules_conds,
                scale_filter=None if args.all_scales else args.rules_scale)
    panel_squares(axes[1], squares_conds)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
