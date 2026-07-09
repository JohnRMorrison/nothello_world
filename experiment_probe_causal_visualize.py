"""Print N adversarial examples: actual board vs. probe-decoded board,
with the illegal move indicated and the critical/flank-creating errors
highlighted.

Reads the same adversarial_records.npz that experiment_probe_causal_analysis.py
uses, so you can point it at the same directory.

Board rendering (each cell is 3 chars wide):
   .   = empty
   X   = black
   O   = white
   #   = the illegal top-1 cell (C)
   !   = a critical mismatch (fix this and the flank breaks)
   *   = a non-critical probe error

Two boards printed side by side per example.

Usage:
    python experiment_probe_causal_visualize.py \\
        --adversarial-dir experiment1_by_depth \\
        --n-examples 50 \\
        --output-txt causal_examples.txt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import VOCAB_SIZE

from experiment_probe_on_adversarial import (
    state_to_gt, _build_token_to_board_pos,
    get_hidden_and_state, probe_predict,
)
from experiment_probe_causal_analysis import (
    probe_to_nanda_state, next_hand_color_at_turn,
    flank_providing_directions, critical_errors_for_direction, DIRS,
)


def alg(cell):
    """0..63 -> algebraic notation like d3."""
    return f"{'abcdefgh'[cell % 8]}{cell // 8 + 1}"


def render_board(state_8x8, marks=None):
    """Return list of 8 strings (rows).  marks: {cell_idx: suffix_char} —
    the cell shows its state (X/O/.) plus the suffix.  If a mark starts with
    '#' the whole cell is displayed as the mark (used for the illegal cell)."""
    marks = marks or {}
    rows = []
    header = "     a   b   c   d   e   f   g   h"
    rows.append(header)
    rows.append("    " + "-" * 34)
    for r in range(8):
        cells = []
        for c in range(8):
            cell_i = r * 8 + c
            v = state_8x8[r, c]
            base = "." if v == 0 else ("X" if v == 1 else "O")
            suffix = marks.get(cell_i, " ")
            if suffix == "#":
                base = "#"
                suffix = " "
            cells.append(f" {base}{suffix} ")
        rows.append(f"  {r + 1} |" + "".join(cells))
    return rows


def side_by_side(rowsA, rowsB, gap=6):
    """Combine two rendered boards side by side."""
    width = max(len(r) for r in rowsA)
    out = []
    for a, b in zip(rowsA, rowsB):
        out.append(f"{a:<{width}}" + " " * gap + b)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_by_depth')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe',
                    default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--n-examples', type=int, default=50)
    ap.add_argument('--output-txt', default='causal_examples.txt')
    ap.add_argument('--only-c-legal-under-probe', action='store_true',
                    default=True,
                    help='Only show examples where C is legal under probe board.')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    probe = torch.load(args.probe, map_location='cpu')
    assert probe.shape == (3, 512, 8, 8, 3)

    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    print(f"Loaded OGPT, probing layer {args.layer}")

    records_path = os.path.join(args.adversarial_dir,
                                  'adversarial_records.npz')
    d = np.load(records_path, allow_pickle=True)
    games = d['games']; turns = d['turns']; cells = d['illegal_cells']
    print(f"Loaded {len(games)} adversarial records")

    # Shuffle for a diverse sample (same seed as causal analysis)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(games))

    lines = []
    n_written = 0
    for i in idx:
        if n_written >= args.n_examples:
            break
        game = tuple(games[i]); turn = int(turns[i]); C = int(cells[i])

        hidden, gt_state = get_hidden_and_state(
            model, device, game, turn, args.layer, block_size)
        pred = probe_predict(hidden, turn, probe)                # (8, 8)
        pred_flat = pred.flatten()
        probe_state = probe_to_nanda_state(pred_flat)
        color_next = next_hand_color_at_turn(turn)

        flank_dirs = flank_providing_directions(probe_state, C, color_next)
        if args.only_c_legal_under_probe and not flank_dirs:
            continue

        # Critical errors
        crit_cells = set()
        for dr in flank_dirs:
            crit_cells.update(critical_errors_for_direction(
                probe_state, gt_state, C, dr, color_next))

        # Any probe error (excluding C itself, which per (a) is essentially 0)
        gt_flat = state_to_gt(gt_state).flatten()  # 0/1/2 encoding for compare
        # We compare directly on nanda state
        probe_wrong = {c for c in range(64)
                        if probe_state[c // 8, c % 8]
                            != gt_state[c // 8, c % 8]}

        # Build marks
        # Actual board: mark C with '#'
        actual_marks = {C: '#'}
        # Probe board: mark C with '#', critical errors with '!',
        # other probe errors with '*'
        probe_marks = {C: '#'}
        for c in probe_wrong:
            if c == C: continue
            if c in crit_cells:
                probe_marks[c] = '!'
            else:
                probe_marks[c] = '*'

        # Determine mover
        mover = "BLACK" if color_next == 1 else "WHITE"

        header = (
            f"===== Example {n_written + 1} / {args.n_examples} =====\n"
            f"Turn (moves played so far): {turn + 1} | "
            f"To move: {mover} | "
            f"Illegal top-1 cell: {alg(C)} (index {C})\n"
            f"Flank-providing directions on probe board: {len(flank_dirs)}\n"
            f"Critical mismatches: {len(crit_cells)}   "
            f"Total probe errors: {len(probe_wrong)}\n"
            f"Legend:  # = illegal top-1 cell   ! = critical error   "
            f"* = other probe error\n"
        )
        titles = f"{'ACTUAL BOARD':<32}{'':<6}{'PROBE-DECODED BOARD':<32}"
        rowsA = render_board(gt_state, actual_marks)
        rowsB = render_board(probe_state, probe_marks)
        body = side_by_side(rowsA, rowsB)
        lines.append(header + "\n" + titles + "\n" + body + "\n")
        n_written += 1

    with open(args.output_txt, 'w') as f:
        f.write("\n".join(lines))
    print(f"Wrote {n_written} examples to {args.output_txt}")


if __name__ == '__main__':
    main()
