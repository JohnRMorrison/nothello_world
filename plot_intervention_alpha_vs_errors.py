"""Plot intervention alpha vs board-state reconstruction errors.

Reads results.json produced by sweep_intervention_alpha.py and plots, for each
target square, the mean (with IQR shading) number of squares whose probe
prediction flips relative to the alpha=0 baseline.

Usage:
    python plot_intervention_alpha_vs_errors.py \\
        --results experiments/intervention_alpha_sweep/results.json \\
        --out experiments/intervention_alpha_sweep/alpha_vs_errors.pdf
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def square_label(r, c):
    return f"({r},{c})  {'abcdefgh'[c]}{r+1}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default=None,
                        help="Output PDF path. Default: same dir as --results.")
    parser.add_argument("--logx", action="store_true",
                        help="Use log scale on x-axis (useful if alpha spans "
                             "orders of magnitude). Skips alpha=0.")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    cfg = data["config"]
    squares_data = data["squares"]

    if args.out is None:
        import os
        args.out = args.results.rsplit('.', 1)[0] + ".pdf"

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(squares_data), 2)))
    for (key, d), color in zip(squares_data.items(), colors):
        r, c = (int(x) for x in key.split(','))
        alphas = np.array(d["alpha"])
        mean = np.array(d["flips_mean"])
        q25  = np.array(d["flips_q25"])
        q75  = np.array(d["flips_q75"])

        if args.logx:
            mask = alphas > 0
            alphas, mean, q25, q75 = alphas[mask], mean[mask], q25[mask], q75[mask]

        ax.plot(alphas, mean, marker='o', color=color,
                label=f"{square_label(r, c)}  (n={d['n_positions']})",
                linewidth=2)
        ax.fill_between(alphas, q25, q75, color=color, alpha=0.18)

    if args.logx:
        ax.set_xscale('log')

    ax.set_xlabel(r"Intervention scale $\alpha$", fontsize=12)
    ax.set_ylabel("Squares with flipped prediction (of 64)", fontsize=12)
    ax.set_title(
        f"OGPT layer {cfg['layer']}: intervention crosstalk vs "
        r"$\alpha$" + f"\n(probe mode {cfg['probe_mode']}, "
                      f"n_games={cfg['n_games']}, "
                      f"positions {cfg['pos_start']}-{cfg['pos_end']})",
        fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(args.out, format='pdf', bbox_inches='tight')
    print(f"Saved {args.out}")
