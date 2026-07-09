"""Slide-ready image: actual board vs. probe-decoded board at a given turn
of one adversarial game.  The cell C where OGPT wants to move next is
highlighted in red on the actual board and green on the probe board,
with the model's probability on C written inside the cell.

Standard codebase board style: green board with dark grid, black/white
disks, algebraic labels (a-h columns, 1-8 rows).

Usage:
    python plot_actual_vs_probe.py --game-index 0 --turn 32 \
        --out plots/game1_T32.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

sys.path.insert(0, '.')
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, VOCAB_SIZE, extract_activations,
)


CENTER_CELLS = {27, 28, 35, 36}
VALID_MOVES = [i for i in range(64) if i not in CENTER_CELLS]

BOARD_GREEN = '#0e7a2e'
GRID_LINE = '#062c12'
HIGHLIGHT_ACTUAL = '#d1341a'   # red
HIGHLIGHT_PROBE  = '#c7f26f'   # green


def alg(cell):
    return f"{'abcdefgh'[cell % 8]}{cell // 8 + 1}"


def build_pos_to_token(block_size):
    dummy_game = list(VALID_MOVES)
    toks = tokenize_games([dummy_game], seq_len=block_size)[0].tolist()
    pos_to_token = np.full(64, -1, dtype=np.int64)
    for i, m in enumerate(dummy_game):
        if i < len(toks):
            pos_to_token[m] = toks[i]
    return pos_to_token


def probe_argmax_state(hidden_512, turn, probe):
    """8x8 state via probe argmax at mode 0 (odd) or 1 (even).  Nanda encoding:
    1 = black, -1 = white, 0 = empty."""
    mode = 0 if turn % 2 == 1 else 1
    W = probe[mode]
    h = torch.from_numpy(hidden_512).float()
    logits = torch.einsum('d,drco->rco', h, W)
    cls = logits.argmax(dim=-1).detach().cpu().numpy()
    st = np.zeros((8, 8), dtype=np.int8)
    st[cls == 1] = -1
    st[cls == 2] = 1
    return st


def _draw_piece(ax, cx, cy, color, radius=0.4):
    if color == 'black':
        ax.add_patch(Circle((cx, cy), radius,
                             facecolor='black', edgecolor='black', linewidth=0.5))
    elif color == 'white':
        ax.add_patch(Circle((cx, cy), radius,
                             facecolor='white', edgecolor='black', linewidth=1.4))


def draw_board(ax, state_8x8, C_cell, p_C, highlight_color, title):
    """state_8x8: 1 = black, -1 = white, 0 = empty (nanda convention).
    Draws the standard green-grid board with pieces, highlights C, writes P(C).
    Convention: row 0 is at TOP of the image.  Columns a-h left-to-right.
    Row labels 1-8 top-to-bottom (matches alg() which uses cell//8 + 1 for row).
    """
    ax.add_patch(Rectangle((0, 0), 8, 8, facecolor=BOARD_GREEN,
                           edgecolor=GRID_LINE, linewidth=1.5))
    for k in range(1, 8):
        ax.plot([0, 8], [k, k], color=GRID_LINE, linewidth=0.5)
        ax.plot([k, k], [0, 8], color=GRID_LINE, linewidth=0.5)

    # Highlight C's square with a filled overlay first (so pieces sit on top)
    cr, cc = C_cell // 8, C_cell % 8
    # y-coordinate: row 0 at top -> y = 7.5, row 7 at bottom -> y = 0.5
    y_of_row = lambda r: 7 - r
    ax.add_patch(Rectangle((cc, y_of_row(cr)), 1, 1,
                            facecolor=highlight_color, edgecolor=GRID_LINE,
                            linewidth=1.5, alpha=0.85))

    # Draw pieces
    for cell in range(64):
        r, c = cell // 8, cell % 8
        v = state_8x8[r, c]
        if v == 1:
            _draw_piece(ax, c + 0.5, y_of_row(r) + 0.5, 'black')
        elif v == -1:
            _draw_piece(ax, c + 0.5, y_of_row(r) + 0.5, 'white')

    # Probability label on C
    pct = p_C * 100
    txt = f"{pct:.1f}%"
    # Choose contrasting text color depending on highlight bg
    text_color = 'white' if highlight_color == HIGHLIGHT_ACTUAL else 'black'
    ax.text(cc + 0.5, y_of_row(cr) + 0.5, txt,
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=text_color)

    # Column labels (a-h) on top, row labels (1-8) on left
    for c in range(8):
        ax.text(c + 0.5, 8.15, 'abcdefgh'[c],
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    for r in range(8):
        ax.text(-0.2, y_of_row(r) + 0.5, str(r + 1),
                ha='right', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(-0.6, 8)
    ax.set_ylim(0, 8.6)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=13, pad=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_by_depth')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe',
                    default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--game-index', type=int, default=0,
                    help="Index into the shuffled adversarial records "
                         "(same shuffle seed 0 as prob_evolution).")
    ap.add_argument('--turn', type=int, default=-1,
                    help='Turn to visualize; -1 = the error turn.')
    ap.add_argument('--out', default='plots/actual_vs_probe.png')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    probe = torch.load(args.probe, map_location='cpu')
    assert probe.shape == (3, 512, 8, 8, 3)

    pos_to_token = build_pos_to_token(block_size)

    records_path = os.path.join(args.adversarial_dir, 'adversarial_records.npz')
    d = np.load(records_path, allow_pickle=True)
    games = d['games']; turns = d['turns']; cells = d['illegal_cells']
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(games))
    i = int(idx[args.game_index])
    game = list(games[i])
    T = int(turns[i])
    C = int(cells[i])
    turn = args.turn if args.turn >= 0 else T

    # Feed to model, get logits + layer activations
    L = min(T + 1, block_size)
    tokens = tokenize_games([game[:L]], seq_len=block_size).to(device)
    with torch.no_grad():
        logits, _ = model(tokens)
        acts = extract_activations(model, tokens, args.layer)
    acts_np = acts.cpu().numpy()[0]

    # Play to `turn` for the actual board
    board = OthelloBoardState()
    for t in range(0, turn + 1):
        board.umpire(game[t])
    actual_state = np.asarray(board.state, dtype=np.int8)

    # Probe-decoded state at `turn`
    probe_state = probe_argmax_state(acts_np[turn], turn, probe)

    # P(C) at `turn`
    probs = F.softmax(logits[0, turn, :], dim=-1).detach().cpu().numpy()
    probs_60 = np.zeros(60, dtype=np.float32)
    for k, m in enumerate(VALID_MOVES):
        tok = int(pos_to_token[m])
        if tok >= 0:
            probs_60[k] = probs[tok]
    probs_60 = probs_60 / max(probs_60.sum(), 1e-9)
    p_C = float(probs_60[VALID_MOVES.index(C)])

    # Legality of C on actual board
    legal_set = set(board.get_valid_moves())
    C_legal_actual = C in legal_set
    status = "LEGAL" if C_legal_actual else "ILLEGAL"

    # Draw
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    draw_board(axes[0], actual_state, C, p_C, HIGHLIGHT_ACTUAL,
                f"Actual board — turn {turn}\n{alg(C)} is {status}")
    draw_board(axes[1], probe_state, C, p_C, HIGHLIGHT_PROBE,
                f"Probe-decoded board — turn {turn}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f"Wrote {args.out}")
    print(f"Game {args.game_index}: error turn = {T}, C = {alg(C)}, "
          f"turn plotted = {turn}, P({alg(C)}) = {p_C:.4f}, "
          f"C {'legal' if C_legal_actual else 'illegal'} on actual")


if __name__ == '__main__':
    main()
