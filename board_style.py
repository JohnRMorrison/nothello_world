import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Circle

COLS = list('ABCDEFGH')
ROWS = [str(i) for i in range(1, 9)]

named_colors = {
    'green':  "#72c772",
    'red':    "#d55e5e",
    'orange': '#ff9800',
    'blue':   '#2196f3',
    'white':  '#ffffff',
    'black':  '#000000',
    'gray':   '#9e9e9e',
}


def resolve_color(color):
    return named_colors.get(color, color)


def text_color_for_bg(fill):
    # decide whether to put black or white text based on background brightness
    if isinstance(fill, (list, tuple)) and len(fill) >= 3:
        r, g, b = fill[0], fill[1], fill[2]
    else:
        c = str(fill).lstrip('#')
        if len(c) != 6:
            return 'black'
        r, g, b = int(c[0:2], 16)/255, int(c[2:4], 16)/255, int(c[4:6], 16)/255
    lum = 0.299*r + 0.587*g + 0.114*b
    return 'black' if lum > 0.45 else 'white'


def draw_board(ax, board, highlights=None, heatmap=None, title=None,
               show_coords=True, piece_scale=0.38, fontsize=9, colorbar=False):
    if highlights is None:
        highlights = {}

    ax.set_facecolor('white')

    for row in range(8):
        for col in range(8):
            spec = highlights.get((row, col), {})
            val = board[row, col]

            if heatmap and (row, col) in heatmap and val == 0:
                v = float(np.clip(heatmap[(row, col)], 0, 1))
                fill = (1 - v, 1 - v, 1 - v)
            elif 'fill' in spec:
                fill = resolve_color(spec['fill'])
            else:
                fill = '#e8e8e8'

            ax.add_patch(Rectangle(
                (col - 0.5, row - 0.5), 1.0, 1.0,
                facecolor=fill, edgecolor='#aaaaaa',
                linewidth=0.8, zorder=1))

            if val == 1:
                ax.add_patch(Circle((col, row), piece_scale,
                                    facecolor='black', edgecolor='#333333',
                                    linewidth=1.0, zorder=3))
            elif val == -1:
                ax.add_patch(Circle((col, row), piece_scale,
                                    facecolor='white', edgecolor='black',
                                    linewidth=1.5, zorder=3))
            elif 'label' in spec:
                ax.text(col, row, spec['label'],
                        ha='center', va='center',
                        fontsize=fontsize, fontweight='bold',
                        color=text_color_for_bg(fill), zorder=4)

    # borders on top so they don't get covered by adjacent cells
    for row in range(8):
        for col in range(8):
            spec = highlights.get((row, col), {})
            if 'border' in spec:
                ax.add_patch(Rectangle(
                    (col - 0.5, row - 0.5), 1.0, 1.0,
                    facecolor='none',
                    edgecolor=resolve_color(spec['border']),
                    linewidth=spec.get('border_lw', 3.0),
                    zorder=6))

    ax.add_patch(Rectangle((-0.5, -0.5), 8, 8,
                            facecolor='none', edgecolor='#666666',
                            linewidth=1.5, zorder=7))

    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(7.5, -0.5)
    ax.set_aspect('equal')

    if show_coords:
        ax.set_xticks(range(8))
        ax.set_xticklabels(COLS, fontsize=11, fontweight='bold')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_yticks(range(8))
        ax.set_yticklabels(ROWS, fontsize=11, fontweight='bold')
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

    if colorbar and heatmap:
        sm = plt.cm.ScalarMappable(cmap=matplotlib.colormaps['gray_r'],
                                   norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02).ax.tick_params(labelsize=8)


if __name__ == '__main__':
    matplotlib.use('Agg')

    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = -1; board[4, 4] = -1
    board[3, 4] =  1; board[4, 3] =  1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')

    draw_board(axes[0], board)

    draw_board(axes[1], board,
               highlights={
                   (2, 3): {'fill': 'green', 'label': '23%'},
                   (2, 4): {'fill': 'green', 'label': '18%', 'border': 'orange'},
                   (3, 2): {'fill': 'green', 'label': '15%'},
                   (4, 5): {'fill': 'red',   'label': '5%'},
               })

    np.random.seed(0)
    hmap = {(r, c): np.random.uniform(0, 1)
            for r in range(8) for c in range(8) if board[r, c] == 0}
    best = max(hmap, key=hmap.get)
    draw_board(axes[2], board,
               heatmap=hmap,
               highlights={best: {'label': f'{hmap[best]:.2f}'}},
               colorbar=True)

    plt.tight_layout()
    plt.savefig('board_style_example.png', dpi=120, bbox_inches='tight', facecolor='white')
    print('Saved board_style_example.png')
