"""Generate MLP architecture diagram to experiments/plots/.

The Othello-GPT figure is hand-authored elsewhere — this script only
produces the parallel MLP figure (two parity-specific MLPs side-by-side).
"""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PLOTS_DIR = "experiments/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------- helpers ----------

def add_box(ax, cx, cy, w, h, label, fill=True, color="#dfe7ff",
            edge="#1f3a93", fontsize=10):
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor=edge,
        facecolor=color if fill else "white",
    )
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color="black")


def add_arrow(ax, x0, y0, x1, y1, color="black"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", linewidth=1.4,
                        color=color, mutation_scale=14),
    )


# ---------- Othello-GPT ----------

def plot_ogpt():
    fig, ax = plt.subplots(figsize=(4.8, 7.6))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 11.5)
    ax.set_aspect("equal")
    ax.axis("off")

    box_w = 2.6
    box_h = 0.55
    cx = 2.0
    spacing = 0.85

    # Top → bottom layout (input at top, output at bottom)
    ys = [10.7, 9.85, 9.0, 8.15, 6.5, 4.85, 3.5, 2.5]
    labels = [
        "Move sequence (input)",
        "Token embedding",
        "Position embedding",
        "Embedding sum",
        "8 × Transformer block",
        "Final LayerNorm",
        "Output head",
        "Cell logits (output)",
    ]

    # Cluster transformer block centered around y=6.5
    ys = [10.7, 9.7, 8.85, 7.95, 6.5, 4.6, 3.6, 2.6]

    colors = ["#dfe7ff", "#dfe7ff", "#dfe7ff", "#dfe7ff",
              "#fff2cc", "#dfe7ff", "#dfe7ff", "#dfe7ff"]

    # Transformer block needs to be taller
    heights = [box_h] * len(ys)
    heights[4] = 1.4  # tall box for "8 × Transformer block"

    for y, lab, c, hh in zip(ys, labels, colors, heights):
        add_box(ax, cx, y, box_w, hh, lab, color=c, fontsize=11)

    # Arrows
    for i in range(len(ys) - 1):
        y_top = ys[i] - heights[i] / 2
        y_bot = ys[i + 1] + heights[i + 1] / 2
        add_arrow(ax, cx, y_top, cx, y_bot)

    out_path = os.path.join(PLOTS_DIR, "ogpt_architecture.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


# ---------- MLP (two columns side-by-side) ----------

def plot_mlp():
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    box_w = 2.4
    box_h = 0.7

    # Two columns
    cols = [1.6, 4.9]
    # Rows: input, MLP, output
    ys = [7.2, 4.7, 2.2]
    labels = ["Move history\n(input)", "MLP\nH = 512", "Cell\n(output)"]
    colors = ["#dfe7ff", "#fff2cc", "#dfe7ff"]
    heights = [box_h * 1.4, box_h * 1.4, box_h * 1.4]

    for col in cols:
        for y, lab, c, hh in zip(ys, labels, colors, heights):
            add_box(ax, col, y, box_w, hh, lab, color=c, fontsize=11)
        # Arrows between boxes in this column
        for i in range(len(ys) - 1):
            y_top = ys[i] - heights[i] / 2
            y_bot = ys[i + 1] + heights[i + 1] / 2
            add_arrow(ax, col, y_top, col, y_bot)

    out_path = os.path.join(PLOTS_DIR, "mlp_architecture.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_mlp()
