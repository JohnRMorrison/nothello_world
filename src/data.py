"""Data utilities: loading games, replaying to a position, computing legal
moves on arbitrary board states, and sampling positions that satisfy a
specific (category, sub-condition) intervention condition.

Board encoding: +1 = black, -1 = white, 0 = empty (matches OthelloBoardState).

Categories of intervention:
  remove    — occupied cell (mine or yours) -> empty
  add_mine  — empty cell -> current-player's color
  add_yours — empty cell -> opponent's color
  flip      — occupied cell -> opposite color

Sub-conditions (over the change in the legal-move set):
  newly_legal   — at least one move becomes legal, no moves become illegal
  newly_illegal — at least one move becomes illegal, no moves become legal
  both          — at least one of each
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# Allow `from data.othello import OthelloBoardState` from the repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from data.othello import OthelloBoardState  # noqa: E402

from . import config


# ---------------------------------------------------------------------------
# Loading games
# ---------------------------------------------------------------------------
def load_games(int_path: str = None, string_path: str = None):
    """Load the two parallel game arrays:
      board_seqs_int    — tokenized for the model (N, 60), vocab 0..60
      board_seqs_string — flat cell indices 0..63 for OthelloBoardState replay

    Returns (board_seqs_int, board_seqs_string).
    """
    int_path = int_path or config.INT_DATA_PATH
    string_path = string_path or config.STRING_DATA_PATH
    return np.load(int_path), np.load(string_path)


# ---------------------------------------------------------------------------
# Board state replay / legality
# ---------------------------------------------------------------------------
def replay_to_position(game_string, pos: int):
    """Replay a game's first `pos+1` moves and return:
      board_state — np.ndarray (8, 8) int with values in {-1, 0, +1}
      next_color  — +1 if black plays next, -1 if white plays next

    Raises if the replay fails (e.g. corrupt game).
    """
    b = OthelloBoardState()
    for i in range(pos + 1):
        b.umpire(int(game_string[i]))
    return b.state.copy().astype(np.int8), int(b.next_hand_color)


def legal_moves_on(board_2d: np.ndarray, next_color: int) -> set[int]:
    """Compute the set of legal flat-cell-indices given an arbitrary board
    state and side-to-move. Used for both original and counterfactual boards."""
    b = OthelloBoardState()
    b.state = board_2d.copy().astype(np.float64)
    b.next_hand_color = int(next_color)
    return set(b.get_valid_moves())


# ---------------------------------------------------------------------------
# Condition logic — given a (square, category), what target_val?
# ---------------------------------------------------------------------------
def classify_change(legal_orig: set[int], legal_cf: set[int]) -> str | None:
    """Return one of {newly_legal, newly_illegal, both} or None if no change."""
    added = legal_cf - legal_orig
    removed = legal_orig - legal_cf
    if not added and not removed:
        return None
    if added and not removed:
        return "newly_legal"
    if removed and not added:
        return "newly_illegal"
    return "both"


def target_val_for_category(category: str, orig_val: int, next_color: int) -> int | None:
    """Return the post-intervention board value for a (category, orig_val) pair,
    or None if the category is not applicable to this cell state.

    Mine convention: a piece "mine" iff its color == next_color.
    """
    if category == "remove":
        return 0 if orig_val != 0 else None
    if category == "add_mine":
        return next_color if orig_val == 0 else None
    if category == "add_yours":
        return -next_color if orig_val == 0 else None
    if category == "flip":
        return -orig_val if orig_val != 0 else None
    raise ValueError(f"unknown category: {category}")


def category_applicable(category: str, orig_val: int) -> bool:
    """Cheaper precheck used during sampling."""
    if category == "remove":
        return orig_val != 0
    if category in ("add_mine", "add_yours"):
        return orig_val == 0
    if category == "flip":
        return orig_val != 0
    raise ValueError(f"unknown category: {category}")


# ---------------------------------------------------------------------------
# Position record
# ---------------------------------------------------------------------------
@dataclass
class PositionRecord:
    """One position with all the context needed to apply an intervention.

    Move history is the canonical reproducibility key; everything else is
    derivable from it (but we cache for speed).
    """
    move_history: list[int]            # flat cell indices 0..63
    position: int                       # = len(move_history) - 1
    square: tuple[int, int]             # (r, c) of the intervention target
    category: str
    sub_condition: str                  # newly_legal / newly_illegal / both
    orig_val: int                       # board value at square before edit
    target_val: int                     # board value at square after edit
    next_color: int                     # +1 = black to move, -1 = white
    board_state: np.ndarray             # (8,8) int8
    legal_orig: list[int]               # sorted flat-cell indices
    legal_cf: list[int]                 # sorted flat-cell indices (counterfactual)
    # `extras` lets downstream code attach cached forward-pass results
    # (clean logits, residuals, intervention direction, etc.) without
    # changing the schema. Stored as a dict; not serialized by default.
    extras: dict = field(default_factory=dict)


def position_from_replay(
    game_string: np.ndarray,
    pos: int,
    square: tuple[int, int],
    category: str,
) -> PositionRecord | None:
    """Replay `game_string` to `pos`; if the resulting board state admits an
    intervention of `category` at `square` that changes legality, return a
    full PositionRecord, else None."""
    r, c = square
    # NOTE: board-center starting cells (3,3)/(3,4)/(4,3)/(4,4) are valid
    # intervention targets — they're never empty, so `add_*` won't apply
    # to them but `remove` and `flip` will. The crosstalk count excludes
    # them from being counted as "another flipped cell."

    try:
        board, next_color = replay_to_position(game_string, pos)
    except Exception:
        return None

    orig_val = int(board[r, c])
    if not category_applicable(category, orig_val):
        return None
    target_val = target_val_for_category(category, orig_val, next_color)
    if target_val is None:
        return None

    legal_orig = legal_moves_on(board, next_color)
    if not legal_orig:
        return None   # forfeit position, skip

    modified = board.copy()
    modified[r, c] = target_val
    legal_cf = legal_moves_on(modified, next_color)
    if not legal_cf:
        return None

    sub = classify_change(legal_orig, legal_cf)
    if sub is None:
        return None

    return PositionRecord(
        move_history=[int(x) for x in game_string[: pos + 1]],
        position=pos,
        square=square,
        category=category,
        sub_condition=sub,
        orig_val=orig_val,
        target_val=target_val,
        next_color=next_color,
        board_state=board,
        legal_orig=sorted(legal_orig),
        legal_cf=sorted(legal_cf),
    )


# ---------------------------------------------------------------------------
# Bulk sampling: build the database
# ---------------------------------------------------------------------------
def sample_database(
    board_seqs_string: np.ndarray,
    squares: Iterable[tuple[int, int]],
    *,
    n_positions_per_condition: int = None,
    max_games: int = None,
    pos_lo: int = None,
    pos_hi: int = None,
    seed: int = None,
    categories: Iterable[str] = None,
    sub_conditions: Iterable[str] = None,
    verbose: bool = True,
) -> dict:
    """Sample up to `n_positions_per_condition` positions per
    (square, category, sub_condition), drawing at most `max_games` games.

    Returns nested dict: db[square_label][category][sub_condition] -> list[PositionRecord]
    where square_label is the string "(r,c)".
    """
    n_positions_per_condition = (
        n_positions_per_condition or config.DEFAULT_N_POSITIONS_PER_CONDITION
    )
    max_games = max_games or config.DEFAULT_MAX_GAMES
    pos_lo = pos_lo if pos_lo is not None else config.DEFAULT_POS_LO
    pos_hi = pos_hi if pos_hi is not None else config.DEFAULT_POS_HI
    seed = seed if seed is not None else config.DEFAULT_SEED
    categories = list(categories or config.CATEGORIES)
    sub_conditions = list(sub_conditions or config.SUB_CONDITIONS)
    squares = list(squares)

    rng = random.Random(seed)
    n_total_games = len(board_seqs_string)
    game_pool = rng.sample(range(n_total_games), min(max_games, n_total_games))

    # Initialize storage.
    db = {}
    for (r, c) in squares:
        sq_label = f"({r},{c})"
        db[sq_label] = {cat: {sub: [] for sub in sub_conditions}
                        for cat in categories}

    # We iterate games and try to fill every (square, category, sub) bucket
    # simultaneously. Stop when every bucket is full or game pool exhausted.
    def bucket_full(sq_label, cat, sub):
        return len(db[sq_label][cat][sub]) >= n_positions_per_condition

    def all_full():
        for sq in db:
            for cat in db[sq]:
                for sub in db[sq][cat]:
                    if not bucket_full(sq, cat, sub):
                        return False
        return True

    games_scanned = 0
    positions_kept = 0
    for gi in game_pool:
        games_scanned += 1
        if all_full():
            break
        game_string = board_seqs_string[gi]
        # Within each game, try one random position from each square × category;
        # for "newly_legal" / "newly_illegal" / "both" we get whatever the
        # change classifies as. We sample multiple positions per game to
        # increase yield per replay.
        positions_to_try = list(range(pos_lo, pos_hi))
        rng.shuffle(positions_to_try)
        # cap how many positions we try per game to avoid quadratic blowups
        for pos in positions_to_try[:8]:
            for (r, c) in squares:
                sq_label = f"({r},{c})"
                for cat in categories:
                    rec = position_from_replay(game_string, pos, (r, c), cat)
                    if rec is None:
                        continue
                    sub = rec.sub_condition
                    if sub not in db[sq_label][cat]:
                        continue
                    if bucket_full(sq_label, cat, sub):
                        continue
                    # store game index for reference (not strictly needed since
                    # move_history captures the position)
                    rec.extras["game_idx"] = int(gi)
                    db[sq_label][cat][sub].append(rec)
                    positions_kept += 1
        if verbose and games_scanned % 500 == 0:
            counts = _summarize_counts(db)
            print(f"  scanned {games_scanned} games, kept {positions_kept} "
                  f"positions; min bucket = {min(counts.values())}, "
                  f"max bucket = {max(counts.values())}")

    if verbose:
        print(f"Done sampling. Scanned {games_scanned} games. "
              f"Kept {positions_kept} positions.")
    return db


def _summarize_counts(db: dict) -> dict:
    """Flat dict keyed by 'square|category|sub' -> count."""
    out = {}
    for sq in db:
        for cat in db[sq]:
            for sub in db[sq][cat]:
                out[f"{sq}|{cat}|{sub}"] = len(db[sq][cat][sub])
    return out


def database_counts_table(db: dict) -> list[list]:
    """Tabular summary: rows = (square, category), cols = sub_conditions.

    Returns a list of rows (first row is the header) suitable for tabulate.
    """
    if not db:
        return [["square", "category"] + list(config.SUB_CONDITIONS)]
    any_sq = next(iter(db.values()))
    any_cat = next(iter(any_sq.values()))
    subs = list(any_cat.keys())

    rows = [["square", "category"] + subs]
    for sq in db:
        for cat in db[sq]:
            row = [sq, cat]
            for sub in subs:
                row.append(len(db[sq][cat][sub]))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Flat iteration helpers
# ---------------------------------------------------------------------------
def iter_records(db: dict):
    """Yield (square_label, category, sub_condition, PositionRecord) tuples."""
    for sq in db:
        for cat in db[sq]:
            for sub in db[sq][cat]:
                for rec in db[sq][cat][sub]:
                    yield sq, cat, sub, rec


def count_total(db: dict) -> int:
    return sum(1 for _ in iter_records(db))
