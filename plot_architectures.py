"""Generate matching architecture diagrams to experiments/plots/.

  - ogpt_architecture.png : Othello-GPT residual-stream view with 8 layers
  - mlp_architecture.png  : Two parity-specific MLPs side-by-side (matches OGPT style)
"""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PLOTS_DIR = "experiments/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# Shared palette (matched to user's OGPT slide)
COLOR_HISTORY  = "#f5ebd6"   # tan / cream
COLOR_EMBED    = "#c8e6c9"   # light green
COLOR_ATTN     = "#bcd5ef"   # light blue
COLOR_MLP      = "#dccbe4"   # light purple
COLOR_OUTPUT   = "#f5d8b8"   # tan / peach
COLOR_DASH     = "#999999"
EDGE_COLOR     = "#1f3a93"


def add_box(ax, cx, cy, w, h, label, color, edge=EDGE_COLOR,
            fontsize=11, linestyle="-"):
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.3, edgecolor=edge, facecolor=color,
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color="black")


def add_arrow(ax, x0, y0, x1, y1, color="black", lw=1.2):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", linewidth=lw,
                        color=color, mutation_scale=11),
    )


def plot_ogpt():
    fig, ax = plt.subplots(figsize=(6.8, 11.0))
    ax.set_xlim(0, 6.8)
    ax.set_ylim(0, 11.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Centerline x for embedding and output boxes
    cx = 4.0

    # --- top: Move history --> Move embedding ---
    add_box(ax, cx, 10.55, 2.0, 0.55, "Move history", COLOR_HISTORY)
    add_box(ax, cx, 9.80,  2.0, 0.55, "Move embedding", COLOR_EMBED)
    add_arrow(ax, cx, 10.27, cx, 10.08)

    # --- residual stream (left vertical line) ---
    rs_x = 1.45
    rs_top = 9.30
    rs_bot = 0.85

    # path from Move embedding to the top of the residual stream:
    # down 0.2, then left, then down to top of residual line
    ax.plot([cx, cx], [9.52, 9.30], color="black", linewidth=1.3)
    ax.plot([cx, rs_x], [9.30, 9.30], color="black", linewidth=1.3)

    # residual stream vertical line (a single straight line)
    ax.plot([rs_x, rs_x], [rs_top, rs_bot], color="black", linewidth=1.3)

    # --- 8 layers ---
    layer_top = 8.85
    layer_h = 0.78
    layer_gap = 0.21
    attn_cx = 3.40
    mlp_cx  = 5.50
    box_w   = 1.85
    box_h   = 0.50

    layer_centers = []
    for i in range(8):
        cy = layer_top - i * (layer_h + layer_gap) - layer_h / 2
        layer_centers.append(cy)
        # Dashed group box around the layer
        rect = mpatches.Rectangle(
            (2.50, cy - layer_h / 2), 4.05, layer_h,
            linewidth=0.9, edgecolor=COLOR_DASH, facecolor="white",
            linestyle="--",
        )
        ax.add_patch(rect)
        # Layer i label (small italic, top-left of group box)
        ax.text(2.55, cy + layer_h / 2 - 0.05, f"Layer {i}",
                ha="left", va="top", fontsize=8.5,
                style="italic", color="black")
        # attention heads + MLP
        add_box(ax, attn_cx, cy - 0.05, box_w, box_h,
                "8 attention heads", COLOR_ATTN, fontsize=9.5)
        add_box(ax, mlp_cx,  cy - 0.05, box_w, box_h,
                "MLP", COLOR_MLP, fontsize=9.5)
        # bidirectional arrows between residual stream and this layer
        # leftward (input) arrow: rs_x -> 2.50 (group box left edge)
        add_arrow(ax, rs_x, cy + 0.08, 2.48, cy + 0.08)
        # rightward (output) arrow: 2.50 -> rs_x  (return contribution)
        add_arrow(ax, 2.48, cy - 0.18, rs_x, cy - 0.18)

    # --- bottom: residual stream -> output ---
    out_cy = 0.45
    add_box(ax, cx, out_cy, 3.5, 0.75,
            "OUTPUT:  P(next move)\nsoftmax over 60 board cells",
            COLOR_OUTPUT, fontsize=10)
    # path from bottom of residual stream to the output box
    ax.plot([rs_x, rs_x], [rs_bot, out_cy + 0.05], color="black", linewidth=1.3)
    ax.plot([rs_x, cx], [out_cy + 0.05, out_cy + 0.05], color="black", linewidth=1.3)
    add_arrow(ax, cx, out_cy + 0.45, cx, out_cy + 0.4)

    # --- "Residual stream" label (rotated, on the very left) ---
    ax.text(0.55, (rs_top + rs_bot) / 2, "Residual stream",
            rotation=90, ha="center", va="center",
            fontsize=13, color="black")

    out_path = os.path.join(PLOTS_DIR, "ogpt_architecture.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


def plot_mlp():
    """Two parity-specific MLPs side-by-side, matching OGPT palette."""
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.set_xlim(0, 6.0)
    ax.set_ylim(0, 5.6)
    ax.set_aspect("equal")
    ax.axis("off")

    BOX_W = 2.4
    BOX_H = 0.95
    GAP = 1.25

    cols = [1.65, 4.35]
    headers = ["Even Moves", "Odd Moves"]
    for col, header in zip(cols, headers):
        top_y = 4.3
        ys = [top_y, top_y - GAP, top_y - 2 * GAP]
        ax.text(col, top_y + BOX_H / 2 + 0.25, header,
                ha="center", va="center", fontsize=12,
                fontweight="bold", color="black")
        add_box(ax, col, ys[0], BOX_W, BOX_H,
                "Input\n(move history)", COLOR_HISTORY, fontsize=11)
        add_box(ax, col, ys[1], BOX_W, BOX_H,
                "MLP\nH = 512", COLOR_MLP, fontsize=11)
        add_box(ax, col, ys[2], BOX_W, BOX_H,
                "Output\n(move patterns)", COLOR_OUTPUT, fontsize=11)
        add_arrow(ax, col, ys[0] - BOX_H / 2, col, ys[1] + BOX_H / 2)
        add_arrow(ax, col, ys[1] - BOX_H / 2, col, ys[2] + BOX_H / 2)

    out_path = os.path.join(PLOTS_DIR, "mlp_architecture.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_ogpt()
    plot_mlp()
