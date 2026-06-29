"""Generate matching architecture diagrams to experiments/plots/.

Both figures use the same box style, colors, and arrow style so they look
like a matched pair when placed side-by-side in a paper.
  - ogpt_architecture.png : Move history -> Othello-GPT -> Output
  - mlp_architecture.png  : Two parity-specific MLPs side-by-side
"""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PLOTS_DIR = "experiments/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# Shared style
INPUT_COLOR = "#dfe7ff"
MODEL_COLOR = "#fff2cc"
OUTPUT_COLOR = "#dfe7ff"
EDGE_COLOR = "#1f3a93"


def add_box(ax, cx, cy, w, h, label, color, edge=EDGE_COLOR, fontsize=12):
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color="black")


def add_arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", linewidth=1.4,
                        color="black", mutation_scale=14),
    )


# Standard box dimensions used by both figures so they match visually.
BOX_W = 2.4
BOX_H = 0.95
GAP = 1.25  # gap between box centers vertically


def draw_three_column(ax, cx, top_y, middle_label,
                       input_label="Move history (input)",
                       output_label="Output",
                       header=None, model_color=MODEL_COLOR):
    """Draw input -> model -> output column centered at cx.

    If header is set, draws a small title above the column.
    """
    ys = [top_y, top_y - GAP, top_y - 2 * GAP]
    if header is not None:
        ax.text(cx, top_y + BOX_H / 2 + 0.25, header,
                ha="center", va="center", fontsize=12,
                fontweight="bold", color="black")
    add_box(ax, cx, ys[0], BOX_W, BOX_H, input_label, color=INPUT_COLOR)
    add_box(ax, cx, ys[1], BOX_W, BOX_H, middle_label, color=model_color)
    add_box(ax, cx, ys[2], BOX_W, BOX_H, output_label, color=OUTPUT_COLOR)
    add_arrow(ax, cx, ys[0] - BOX_H / 2, cx, ys[1] + BOX_H / 2)
    add_arrow(ax, cx, ys[1] - BOX_H / 2, cx, ys[2] + BOX_H / 2)


def plot_ogpt():
    fig, ax = plt.subplots(figsize=(3.6, 4.0))
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_three_column(ax, cx=1.75, top_y=4.2, middle_label="Othello-GPT")
    out_path = os.path.join(PLOTS_DIR, "ogpt_architecture.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


def plot_mlp():
    # Same visual scale as OGPT, but two columns + header labels
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.set_xlim(0, 5.6)
    ax.set_ylim(0, 5.4)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_three_column(
        ax, cx=1.55, top_y=4.2, middle_label="MLP\nH = 512",
        input_label="Input\n(move history)",
        output_label="Output\n(move patterns)",
        header="Even Moves",
    )
    draw_three_column(
        ax, cx=4.05, top_y=4.2, middle_label="MLP\nH = 512",
        input_label="Input\n(move history)",
        output_label="Output\n(move patterns)",
        header="Odd Moves",
    )
    out_path = os.path.join(PLOTS_DIR, "mlp_architecture.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_ogpt()
    plot_mlp()
