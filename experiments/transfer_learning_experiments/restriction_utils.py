"""Shared utilities for heuristic-based restriction experiments.

Provides constants, feature evaluation, and restriction application logic
used by both game generation and fine-tuning evaluation.
"""

import re

# ---------------------------------------------------------------------------
# Board / token constants
# ---------------------------------------------------------------------------

CENTER_SQUARES = {27, 28, 35, 36}
VALID_POSITIONS = sorted(set(range(64)) - CENTER_SQUARES)

# Token mapping: matches CharDataset's stoi/itos for standard Othello games
ALL_TOKENS = sorted([-100] + VALID_POSITIONS)
STOI = {ch: i for i, ch in enumerate(ALL_TOKENS)}
ITOS = {i: ch for i, ch in enumerate(ALL_TOKENS)}


def board_pos_to_square(pos):
    """Board position (0-63) -> feature square name, e.g. 28 -> 'D4'."""
    row, col = pos // 8, pos % 8
    return chr(ord('A') + row) + str(col)


def square_to_board_pos(square):
    """Feature square name -> board position, e.g. 'D4' -> 28."""
    row = ord(square[0]) - ord('A')
    col = int(square[1])
    return row * 8 + col


# ---------------------------------------------------------------------------
# Restriction condition parsing
# ---------------------------------------------------------------------------

def parse_rule_conditions(rule_str):
    """Parse a rule string into structured conditions.

    Input:  '(D4_theirs) AND (NOT D4_flipped)'
    Output: [{"square": "D4", "feature_type": "theirs", "polarity": True},
             {"square": "D4", "feature_type": "flipped", "polarity": False}]
    Returns None if unparseable.
    """
    if not rule_str or not rule_str.strip():
        return None
    conditions = []
    parts = rule_str.split(" AND ")
    for part in parts:
        part = part.strip()
        negated = part.startswith("(NOT ")
        m = re.match(r"^\((?:NOT\s+)?(.+?)\)$", part)
        if not m:
            return None
        feature_name = m.group(1).strip()
        m2 = re.match(r"^([A-H]\d)_(.+)$", feature_name)
        if not m2:
            return None
        conditions.append({
            "square": m2.group(1),
            "feature_type": m2.group(2),
            "polarity": not negated,
        })
    return conditions


# ---------------------------------------------------------------------------
# Restriction evaluation at game time
# ---------------------------------------------------------------------------

def evaluate_restriction(restriction, board, move_just_played, flipped_squares):
    """Check whether a restriction's conditions are all satisfied.

    Args:
        restriction: dict with "conditions" list and "forbidden_position" int.
        board: OthelloBoardState *after* the latest move was executed.
        move_just_played: int board position of the move just played.
        flipped_squares: set[int] of board positions flipped by that move.

    Returns True if every condition is met (restriction fires).
    """
    # "mine" = player who just moved, "theirs" = player about to move
    mine_color = -board.next_hand_color

    for cond in restriction["conditions"]:
        r = ord(cond["square"][0]) - ord('A')
        c = int(cond["square"][1])
        pos = r * 8 + c
        ft = cond["feature_type"]
        want = cond["polarity"]

        if ft == "mine":
            got = (board.state[r, c] == mine_color)
        elif ft == "theirs":
            got = (board.state[r, c] == -mine_color)
        elif ft == "empty":
            got = (board.state[r, c] == 0)
        elif ft == "flipped":
            got = (pos in flipped_squares)
        elif ft == "just_played":
            got = (pos == move_just_played)
        else:
            return False  # unknown feature type — treat as not firing

        if got != want:
            return False

    return True


def apply_restrictions(standard_legal, restrictions, board,
                       move_just_played, flipped_squares):
    """Remove forbidden positions from the legal-move set.

    If every legal move would be removed, falls back to the full standard set
    (avoids creating stuck games).
    """
    forbidden = set()
    for r in restrictions:
        if evaluate_restriction(r, board, move_just_played, flipped_squares):
            forbidden.add(r["forbidden_position"])

    restricted = [m for m in standard_legal if m not in forbidden]
    return restricted if restricted else standard_legal


def get_flipped_squares(board_state_before, board_state_after, move):
    """Compute which squares were flipped (not the placed piece).

    Args:
        board_state_before: np.ndarray (8, 8) before umpire()
        board_state_after:  np.ndarray (8, 8) after umpire()
        move: int board position that was played

    Returns set[int] of flipped board positions.
    """
    flipped = set()
    for pos in range(64):
        r, c = pos // 8, pos % 8
        if pos == move:
            continue  # placed piece, not a flip
        if board_state_before[r, c] != 0 and board_state_after[r, c] != board_state_before[r, c]:
            flipped.add(pos)
    return flipped
