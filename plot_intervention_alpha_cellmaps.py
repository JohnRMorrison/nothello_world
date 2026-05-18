"""Plot per-cell flip frequencies as 8x8 heatmaps, one per (target, alpha).

Reads the raw .npz produced by sweep_intervention_alpha.py (the version that
saves a `<r,c>_percell` array of shape (n_alphas, 8, 8)) and the matching
.json (for n_positions per target). Renders a grid where rows = target squares,
columns = alpha values, each cell is an 8x8 heatmap of the fraction of
positions where the probe's prediction for that cell flipped.

Each heatmap highlights the intervened cell with a red border.

Usage:
    python plot_intervention_alpha_cellmaps.py \\
        --raw experiments/intervention_alpha_sweep/raw_flips_L6_n500_flip.npz \\
        --results experiments/intervention_alpha_sweep/results_L6_n500_flip.json \\
        --out experiments/intervention_alpha_sweep/cellmaps_L6_n500_flip.pdf
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def square_label(r, c):
    return f"({r},{c}) {'abcdefgh'[c]}{r+1}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True,
                        help="raw_flips_*.npz from sweep_intervention_alpha.py.")
    parser.add_argument("--results", required=True,
                        help="results_*.json (used for n_positions per target).")
    parser.add_argument("--out", default=None,
                        help="Output PDF path. Default: alongside --raw.")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Color-scale ceiling (default: per-row auto).")
    args = parser.parse_args()

    npz = np.load(args.raw)
    with open(args.results) as f:
        results = json.load(f)
    cfg = results["config"]
    squares_data = results["squares"]

    alphas = list(npz["alphas"])
    n_alpha = len(alphas)
    squares = [tuple(s) for s in cfg["squares"]]
    n_sq = len(squares)

    if args.out is None:
        args.out = args.raw.rsplit(".", 1)[0].replace("raw_flips", "cellmaps") + ".pdf"

    fig, axes = plt.subplots(n_sq, n_alpha,
                             figsize=(1.8 * n_alpha, 1.9 * n_sq),
                             squeeze=False)

    for row, (r, c) in enumerate(squares):
        key = f"{r},{c}"
        if key not in squares_data:
            for col in range(n_alpha):
                axes[row, col].set_visible(False)
            continue
        n_pos = squares_data[key]["n_positions"]
        per_cell_counts = npz[f"{key}_percell"]            # (n_alpha, 8, 8)
        freq = per_cell_counts / max(n_pos, 1)             # (n_alpha, 8, 8)

        row_vmax = args.vmax if args.vmax is not None else float(freq.max())
        row_vmax = max(row_vmax, 1e-6)

        for col, alpha in enumerate(alphas):
            ax = axes[row, col]
            im = ax.imshow(freq[col], vmin=0, vmax=row_vmax,
                           cmap="magma", origin="upper")
            # Highlight the intervened cell with a red border.
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   fill=False, edgecolor="red", lw=1.5))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(rf"$\alpha={alpha:g}$", fontsize=10)
            if col == 0:
                ax.set_ylabel(square_label(r, c) + f"\nn={n_pos}", fontsize=9)
        # Per-row colorbar on the right-most heatmap.
        cbar = fig.colorbar(im, ax=axes[row, :].tolist(),
                            fraction=0.025, pad=0.01)
        cbar.set_label("flip frequency", fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    mode = cfg.get("mode", "empty")
    fig.suptitle(f"OGPT layer {cfg['layer']} — per-cell flip frequency  "
                 f"(mode={mode}, n_games={cfg['n_games']}, "
                 f"probe mode {cfg['probe_mode']})",
                 fontsize=11, y=0.995)
    plt.savefig(args.out, format="pdf", bbox_inches="tight")
    print(f"Saved {args.out}")
