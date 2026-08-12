"""Per-square board-decode accuracy heatmaps, in the presentation_boards board
style (A-H columns, 1-8 rows rank-1 on top, monospace labels).

Input: the .npz written by `eval_board_decode.py --save-npz`, with keys
  <stem>__{linear,nonlinear}__cell   (64,)  per-square accuracy
  <stem>__{linear,nonlinear}__ply    (60,)  per-ply accuracy (unused here)

Usage:
  python plot_board_decode_heatmap.py --npz board_decode_acc.npz \
      --probe linear --out board_decode_linear.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MONO = {"family": "monospace"}
COLS = "ABCDEFGH"


def _draw(ax, acc64, title, cmap, vmin, vmax):
    """acc64 in board-cell order (c = row*8 + col); rank 1 (row 0) on top."""
    grid = acc64.reshape(8, 8)
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    for r in range(8):
        for c in range(8):
            v = grid[r, c]
            # white text on dark cells, black on light
            frac = (v - vmin) / max(vmax - vmin, 1e-9)
            ax.text(c, r, f"{100*v:.0f}", ha="center", va="center",
                    fontsize=8, fontfamily="monospace",
                    color="white" if frac < 0.5 else "black")
    ax.set_xticks(range(8)); ax.set_xticklabels(list(COLS), fontfamily="monospace", fontsize=9)
    ax.set_yticks(range(8)); ax.set_yticklabels(range(1, 9), fontfamily="monospace", fontsize=9)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontdict=MONO, fontsize=10, pad=6)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--probe", choices=["linear", "nonlinear", "both"], default="linear")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--vmin", type=float, default=None, help="default: data min")
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--out", default="board_decode_heatmap.png")
    args = ap.parse_args()

    d = np.load(args.npz)
    stems = sorted({k.split("__")[0] for k in d.files})
    probes = ["linear", "nonlinear"] if args.probe == "both" else [args.probe]

    # shared color scale across all panels for comparability
    allv = np.concatenate([d[f"{s}__{p}__cell"] for s in stems for p in probes
                           if f"{s}__{p}__cell" in d.files])
    vmin = args.vmin if args.vmin is not None else float(np.floor(allv.min() * 20) / 20)

    ncol = len(stems)
    nrow = len(probes)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.1 * nrow),
                             squeeze=False)
    fig.patch.set_facecolor("white")
    im = None
    for j, stem in enumerate(stems):
        for i, probe in enumerate(probes):
            key = f"{stem}__{probe}__cell"
            ax = axes[i][j]
            if key not in d.files:
                ax.axis("off"); continue
            title = stem if nrow == 1 else f"{stem}\n{probe}"
            im = _draw(ax, d[key], title, args.cmap, vmin, args.vmax)

    fig.suptitle(f"Board-decode accuracy per square ({args.probe}, mine/yours frame)",
                 fontfamily="monospace", fontsize=12)
    if im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("accuracy", fontfamily="monospace", fontsize=9)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved {args.out}  (stems: {', '.join(stems)}; vmin={vmin:.2f})")


if __name__ == "__main__":
    main()
