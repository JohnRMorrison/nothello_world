"""
Structural rule change conditions for comparison with heuristic-aligned
restrictions.

A₁ ("no diagonal captures"):
    Moves are legal only if they capture along horizontal or vertical lines.
    Diagonal-only captures are disallowed. Tests whether the model's low
    violation rate on B₁ is due to general directional knowledge.

A₂ ("quadrant dominance"):
    Moves in a quadrant are forbidden when the opponent has more pieces there
    than you. Tests whether the model's low violation rate on B₁ is due to
    general spatial knowledge.

Both are implemented as filter functions that take a board state and
standard legal moves, returning a restricted legal set. The evaluate_structural()
function runs the same metrics as the restriction-based evaluate().
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from data.othello import eights, OthelloBoardState

from restriction_utils import get_flipped_squares


# ---------------------------------------------------------------------------
# A₁: No-diagonal captures
# ---------------------------------------------------------------------------

CARDINAL = [eights[i] for i in [0, 2, 4, 6]]  # N, E, S, W only


def get_no_diagonal_legal_moves(board, standard_legal):
    """Return legal moves using only cardinal (non-diagonal) capture lines.

    A move is legal if placing the current player's piece captures at least
    one opponent piece along a horizontal or vertical line. Diagonal captures
    do not count for legality.

    Args:
        board: OthelloBoardState after the latest move.
        standard_legal: list of standard-legal moves (unused but kept for
            consistent interface; we recompute from board state).

    Returns:
        List of board positions legal under cardinal-only rules. Falls back
        to standard_legal if every cardinal-legal move is empty (avoids
        stuck games).
    """
    color = board.next_hand_color
    legal = []
    for move in range(64):
        r, c = move // 8, move % 8
        if board.state[r, c] != 0:
            continue
        has_capture = False
        for direction in CARDINAL:
            buffer_len = 0
            cur_r, cur_c = r, c
            while True:
                cur_r += direction[0]
                cur_c += direction[1]
                if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if board.state[cur_r, cur_c] == 0:
                    break
                elif board.state[cur_r, cur_c] == color:
                    if buffer_len > 0:
                        has_capture = True
                    break
                else:
                    buffer_len += 1
            if has_capture:
                break
        if has_capture:
            legal.append(move)
    return legal if legal else standard_legal


# ---------------------------------------------------------------------------
# A₂: Quadrant dominance
# ---------------------------------------------------------------------------

# Board quadrants (all 64 squares, including center — we'll check board state
# for any square, playable or not)
QUADRANTS = [
    [(r, c) for r in range(4) for c in range(4)],      # Q0 top-left
    [(r, c) for r in range(4) for c in range(4, 8)],    # Q1 top-right
    [(r, c) for r in range(4, 8) for c in range(4)],    # Q2 bottom-left
    [(r, c) for r in range(4, 8) for c in range(4, 8)], # Q3 bottom-right
]

# Pre-compute position sets for fast forbidden lookup
QUADRANT_POSITIONS = [
    {r * 8 + c for r, c in quad}
    for quad in QUADRANTS
]


def get_quadrant_dominance_legal_moves(board, standard_legal, n_quadrants=2):
    """Return legal moves under quadrant-dominance restriction.

    For the first `n_quadrants` quadrants (Q0, Q1, ...), if the opponent
    has strictly more pieces than the current player in that quadrant,
    all moves in that quadrant are forbidden.

    Falls back to standard_legal if everything would be forbidden.

    Args:
        board: OthelloBoardState after the latest move.
        standard_legal: list of standard-legal board positions.
        n_quadrants: how many quadrants to apply (1–4). Default 2.
    """
    mine_color = board.next_hand_color
    forbidden = set()
    for qi in range(n_quadrants):
        mine_count = 0
        opp_count = 0
        for r, c in QUADRANTS[qi]:
            val = board.state[r, c]
            if val == mine_color:
                mine_count += 1
            elif val == -mine_color:
                opp_count += 1
        if opp_count > mine_count:
            forbidden |= QUADRANT_POSITIONS[qi]
    restricted = [m for m in standard_legal if m not in forbidden]
    return restricted if restricted else standard_legal


# ---------------------------------------------------------------------------
# Structural evaluation (parallels finetune_and_evaluate.evaluate)
# ---------------------------------------------------------------------------

def evaluate_structural(model, eval_games, legal_filter_fn, stoi, itos,
                        device, max_games=200, max_seq_len=59):
    """Evaluate model predictions against a structurally modified legal set.

    Same metrics as the restriction-based evaluate(), but the restricted
    legal set is computed by `legal_filter_fn(board, standard_legal)`
    instead of the restriction framework.

    "fires" = the structural rule made any change (restricted ≠ standard).

    Returns dict with top1_legal, top1_legal_when_fires,
    violation_rate, violation_rate_when_fires, fire_rate,
    top1_prob, legal_mass, n_positions, n_fires.
    """
    model.eval()

    total_top1_legal = 0
    total_top1_prob = 0.0
    total_legal_mass = 0.0
    total_positions = 0
    total_fires = 0
    total_top1_legal_fires = 0
    total_violations = 0
    total_violations_fires = 0

    with torch.no_grad():
        for game in eval_games[:max_games]:
            if len(game) < 2:
                continue

            encoded = [stoi[m] for m in game]
            if len(encoded) > max_seq_len + 1:
                encoded = encoded[:max_seq_len + 1]
            x = torch.tensor(encoded[:-1], dtype=torch.long)[None].to(device)
            logits, _ = model(x)
            probs = F.softmax(logits[0], dim=-1)

            board = OthelloBoardState()

            for pos in range(min(len(game) - 1, max_seq_len)):
                move = game[pos]
                board.umpire(move)

                standard_legal = board.get_valid_moves()
                if not standard_legal:
                    continue

                # Apply structural filter
                restricted_legal = legal_filter_fn(board, standard_legal)
                forbidden = set(standard_legal) - set(restricted_legal)

                pos_probs = probs[pos]
                pred_token = pos_probs.argmax().item()
                pred_move = itos[pred_token]

                in_restricted = pred_move in restricted_legal
                in_forbidden = pred_move in forbidden
                fires_here = len(forbidden) > 0

                if in_restricted:
                    total_top1_legal += 1
                if in_forbidden:
                    total_violations += 1

                if fires_here:
                    total_fires += 1
                    if in_restricted:
                        total_top1_legal_fires += 1
                    if in_forbidden:
                        total_violations_fires += 1

                legal_token_indices = [stoi[bp] for bp in restricted_legal
                                       if bp in stoi]
                if legal_token_indices:
                    legal_probs = pos_probs[legal_token_indices]
                    total_top1_prob += legal_probs.max().item()
                    total_legal_mass += legal_probs.sum().item()

                total_positions += 1

    n = max(total_positions, 1)
    nf = max(total_fires, 1)
    return {
        "top1_legal": total_top1_legal / n,
        "top1_prob": total_top1_prob / n,
        "legal_mass": total_legal_mass / n,
        "violation_rate": total_violations / n,
        "fire_rate": total_fires / n,
        "top1_legal_when_fires": (total_top1_legal_fires / nf) if total_fires else None,
        "violation_rate_when_fires": (total_violations_fires / nf) if total_fires else None,
        "n_positions": total_positions,
        "n_fires": total_fires,
    }
