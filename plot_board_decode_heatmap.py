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
import matplotlib as mpl
import matplotlib.patches as mpatches

MONO = {"family": "monospace"}
COLS = "ABCDEFGH"

# presentation_boards.ipynb board aesthetic
CELL_EDGE = "#aaaaaa"
BOARD_EDGE = "#666666"
LABEL_FS = 11


def _pretty(stem):
    """'H4096_move_grid' -> 'H=4096 move_grid'; 'H512_playedeven' -> 'H=512 move_set'."""
    h, _, rep = stem.partition("_")
    if h.startswith("H") and h[1:].isdigit():
        h = f"H={h[1:]}"
    rep = rep.replace("playedeven", "move_set")
    return f"{h} {rep}"


def _draw_nb(ax, acc64, title, cmap, vmin, vmax):
    """Board-style panel matching presentation_boards.ipynb: fixed-scale cells
    with gray outlines, bold A-H / 1-8 labels, board outline, white background,
    value in each square. Rank 1 (row 0) on top; A-H along the top."""
    grid = acc64.reshape(8, 8)
    norm = mpl.colors.Normalize(vmin, vmax)
    cm = plt.get_cmap(cmap)
    for r in range(8):
        for c in range(8):
            v = grid[r, c]
            ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0,
                         facecolor=cm(norm(v)), edgecolor=CELL_EDGE, linewidth=0.8, zorder=1))
            frac = (v - vmin) / max(vmax - vmin, 1e-9)
            ax.text(c, r, f"{100*v:.0f}", ha="center", va="center", fontsize=7,
                    fontweight="bold", color="white" if frac < 0.5 else "black", zorder=2)
    ax.add_patch(mpatches.Rectangle((-0.5, -0.5), 8, 8, facecolor="none",
                 edgecolor=BOARD_EDGE, linewidth=1.5, zorder=7))
    for c in range(8):                                   # A-H along the top
        ax.text(c, -1.05, COLS[c], ha="center", va="center", fontsize=LABEL_FS, fontweight="bold")
    for r in range(8):                                   # ranks 1-8 down the left
        ax.text(-1.05, r, str(r + 1), ha="center", va="center", fontsize=LABEL_FS, fontweight="bold")
    ax.text(3.5, -2.35, title, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(-1.7, 7.7)
    ax.set_ylim(7.7, -3.1)                               # y inverted (rank 1 top), title gutter
    ax.set_aspect("equal")
    ax.axis("off")


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
    ap.add_argument("--cmap", default="Greys_r",
                    help="default Greys_r = B&W (light=high acc), matching the "
                         "deck; use viridis etc. for color.")
    ap.add_argument("--vmin", type=float, default=None, help="default: data min")
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--style", choices=["plain", "notebook"], default="notebook",
                    help="notebook = presentation_boards board aesthetic "
                         "(drawn cells, gray outlines, bold A-H/1-8 labels).")
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
    nb = (args.style == "notebook")
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=((2.7 if nb else 3.0) * ncol, (2.9 if nb else 3.1) * nrow),
                             squeeze=False)
    fig.patch.set_facecolor("white")
    im = None
    for j, stem in enumerate(stems):
        for i, probe in enumerate(probes):
            key = f"{stem}__{probe}__cell"
            ax = axes[i][j]
            if key not in d.files:
                ax.axis("off"); continue
            title = _pretty(stem) if nrow == 1 else f"{_pretty(stem)}\n{probe}"
            if nb:
                _draw_nb(ax, d[key], title, args.cmap, vmin, args.vmax)
            else:
                im = _draw(ax, d[key], title, args.cmap, vmin, args.vmax)

    if im is not None:                                   # colorbar only for the plain (imshow) style
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("accuracy", fontfamily="monospace", fontsize=9)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved {args.out}  (stems: {', '.join(stems)}; style={args.style}; vmin={vmin:.2f})")


if __name__ == "__main__":
    main()
