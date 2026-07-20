import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_board(board, ax=None, highlight=None, probs=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    board_color = "#277714"
    line_color = "#1a5c0d"

    ax.set_facecolor(board_color)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")

    for i in range(9):
        ax.axhline(i, color=line_color, linewidth=0.8)
        ax.axvline(i, color=line_color, linewidth=0.8)

    for row in range(8):
        for col in range(8):
            # y is flipped because row 0 should appear at the top
            x = col + 0.5
            y = 7.5 - row

            if board[row, col] == 1:
                ax.add_patch(patches.Circle((x, y), 0.42, color="black", zorder=3))
            elif board[row, col] == -1:
                ax.add_patch(patches.Circle((x, y), 0.42, color="white", zorder=3))

            if probs is not None and board[row, col] == 0:
                if probs[row, col] > 0.005:
                    ax.text(x, y, f"{probs[row, col]:.2f}", ha="center", va="center",
                            fontsize=6.5, color="white", zorder=4)

    if highlight:
        for (row, col) in highlight:
            ax.add_patch(patches.Circle((col + 0.5, 7.5 - row), 0.15, color="#ffdd00", zorder=4))

    ax.set_xticks([col + 0.5 for col in range(8)])
    ax.set_xticklabels(list("abcdefgh"), fontsize=9)
    ax.set_yticks([row + 0.5 for row in range(8)])
    ax.set_yticklabels([str(i) for i in range(8, 0, -1)], fontsize=9)
    ax.tick_params(length=0)

    if title:
        ax.set_title(title, fontsize=11, pad=6)

    return ax


def board_from_history(move_history):
    from data.othello import OthelloBoardState
    state = OthelloBoardState()
    state.update(move_history)
    board = np.array(state.board, dtype=int).reshape(8, 8)
    next_color = 1 if len(move_history) % 2 == 0 else -1
    return board, next_color


if __name__ == "__main__":
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = 1
    board[3, 4] = -1
    board[4, 3] = -1
    board[4, 4] = 1

    fig, ax = plt.subplots(figsize=(4, 4))
    draw_board(board, ax=ax, highlight=[(2, 3), (3, 2)], title="Starting position")
    plt.tight_layout()
    plt.savefig("board_example.png", dpi=150)
    plt.show()
