"""
Adapter: flanking patterns -> restriction-rule antecedents.

Translates a flanking pattern from `hand_crafted_flanking.py` into the
`conditions` / `rule_str` format that `restriction_utils.parse_rule_conditions`
(and `evaluate_restriction`) consume, so a flanking pattern can be dropped
into the existing 2x2 framework as a third antecedent type.

Feature-type mapping (IMPORTANT — the two files use opposite conventions for
what "mine" means):

    hand_crafted_flanking.py:       restriction_utils.py:
      "is_mine"     (player about     "theirs"   (player about to move)
                     to move)
      "is_opponent" (player who       "mine"     (player who just moved)
                     just moved)
      "empty"       (neither)         "empty"    (cell is 0)

A flanking pattern fires when its target is empty, every opponent cell
between target and terminal holds the opponent-of-the-player-about-to-move
(i.e. the player who just moved), and the terminal cell holds the
player-about-to-move. Under the restriction_utils convention the target's
"empty" stays empty, the opponent cells become "mine", and the terminal
cell becomes "theirs".

The adapter is pure (no model weights, no data) and can be unit-tested
round-trip against the pattern predicate.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hand_crafted_flanking import (  # noqa: E402  (sys.path tweak first)
    enumerate_flanking_patterns,
    CENTER_CELLS,
    DIRECTIONS,
)
from restriction_utils import board_pos_to_square  # noqa: E402


# ---------------------------------------------------------------------------
# Pattern -> conditions / rule string
# ---------------------------------------------------------------------------

def pattern_to_conditions(pattern):
    """Convert a flanking pattern dict to a list of condition dicts.

    Each returned condition matches the schema consumed by
    `restriction_utils.evaluate_restriction`:
        {"square": "D4", "feature_type": "empty"|"mine"|"theirs", "polarity": True}

    All polarities are positive (no NOT clauses) — a flanking pattern is a
    pure conjunction of positive feature checks. The order is:
        1. target empty
        2. opponents[0] mine, opponents[1] mine, ..., opponents[k-1] mine
        3. terminal theirs
    """
    conditions = [
        {
            "square": board_pos_to_square(pattern["target"]),
            "feature_type": "empty",
            "polarity": True,
        }
    ]
    for opp_pos in pattern["opponents"]:
        conditions.append({
            "square": board_pos_to_square(opp_pos),
            "feature_type": "mine",
            "polarity": True,
        })
    conditions.append({
        "square": board_pos_to_square(pattern["terminal"]),
        "feature_type": "theirs",
        "polarity": True,
    })
    return conditions


def conditions_to_rule_str(conditions):
    """Inverse of restriction_utils.parse_rule_conditions.

    Emits the canonical '(SQ_feat) AND (NOT SQ_feat) ...' form.
    """
    parts = []
    for c in conditions:
        inner = f"{c['square']}_{c['feature_type']}"
        if c["polarity"]:
            parts.append(f"({inner})")
        else:
            parts.append(f"(NOT {inner})")
    return " AND ".join(parts)


def pattern_to_rule_str(pattern):
    return conditions_to_rule_str(pattern_to_conditions(pattern))


def direction_to_name(direction):
    """Human-readable name for one of the 8 flanking directions."""
    dr, dc = direction
    names = {
        (-1,  0): "N",  ( 1,  0): "S",
        ( 0, -1): "W",  ( 0,  1): "E",
        (-1, -1): "NW", (-1,  1): "NE",
        ( 1, -1): "SW", ( 1,  1): "SE",
    }
    return names.get((dr, dc), f"({dr},{dc})")


def pattern_id(pattern):
    """Stable short id, e.g. 'F_D4_E_L1' = flanking at D4 eastward length 1."""
    sq = board_pos_to_square(pattern["target"])
    d = direction_to_name(pattern["direction"])
    return f"F_{sq}_{d}_L{pattern['length']}"


# ---------------------------------------------------------------------------
# Direct predicate (ground truth for round-trip testing)
# ---------------------------------------------------------------------------

def pattern_fires_on_snapshot(pattern, snap, move_just_played, flipped_squares):
    """Evaluate a flanking pattern directly on a board snapshot.

    Mirrors the semantics in `HandCraftedFlanking.forward` but without the
    network — used as ground truth in the adapter round-trip test.

    A flanking pattern fires for the player whose turn is next
    (`snap.next_hand_color`). Under that convention:
      - target cell must be empty
      - each opponent cell must hold `-snap.next_hand_color`
        (the player who just moved)
      - terminal cell must hold `snap.next_hand_color`
        (the player about to move)

    `move_just_played` and `flipped_squares` are accepted for signature
    parity with evaluate_restriction but unused — flanking depends only on
    the board state.
    """
    del move_just_played, flipped_squares  # unused
    next_color = snap.next_hand_color
    state = snap.state

    tr, tc = pattern["target"] // 8, pattern["target"] % 8
    if state[tr, tc] != 0:
        return False
    for opp in pattern["opponents"]:
        r, c = opp // 8, opp % 8
        if state[r, c] != -next_color:
            return False
    terminal = pattern["terminal"]
    r, c = terminal // 8, terminal % 8
    if state[r, c] != next_color:
        return False
    return True


# ---------------------------------------------------------------------------
# Firing-rate estimation
# ---------------------------------------------------------------------------

def estimate_pattern_fire_rates(patterns, snapshots):
    """Empirical firing rate of each pattern over a snapshot list.

    Args:
        patterns: list of pattern dicts from `enumerate_flanking_patterns`
        snapshots: list of (snap, move_just_played, flipped_squares) tuples;
                   same shape as `_precompute_game_positions` produces in
                   `build_restriction_configs.py`.

    Returns:
        list[float] of length len(patterns), firing rate of each pattern.
    """
    if not snapshots:
        return [0.0] * len(patterns)
    N = len(snapshots)
    rates = []
    for pat in patterns:
        hits = 0
        for snap, mvp, flipped in snapshots:
            if pattern_fires_on_snapshot(pat, snap, mvp, flipped):
                hits += 1
        rates.append(hits / N)
    return rates


# ---------------------------------------------------------------------------
# Round-trip self-test (invoke as `python flanking_rule_adapter.py`)
# ---------------------------------------------------------------------------

def _round_trip_test(n_games=50):
    """For every flanking pattern, verify:

      adapter-built conditions, when fed to evaluate_restriction,
      fire on exactly the same snapshots as the direct pattern predicate.
    """
    from types import SimpleNamespace

    import numpy as np

    from data.othello import OthelloBoardState, get_ood_game
    from restriction_utils import (
        evaluate_restriction, get_flipped_squares, parse_rule_conditions,
    )

    # --- build snapshots identical to build_restriction_configs ---
    snapshots = []
    for gi in range(n_games):
        game = get_ood_game(gi)
        board = OthelloBoardState()
        for mv in game[:-1]:
            state_before = board.state.copy()
            board.umpire(mv)
            flipped = get_flipped_squares(state_before, board.state, mv)
            snap = SimpleNamespace(
                state=board.state.copy(),
                next_hand_color=board.next_hand_color,
            )
            snapshots.append((snap, mv, flipped))
    print(f"Generated {len(snapshots)} snapshots from {n_games} games")

    patterns = enumerate_flanking_patterns()
    print(f"Checking {len(patterns)} flanking patterns for round-trip fidelity...")

    n_checked = 0
    n_match = 0
    fire_totals = 0
    for pat in patterns:
        conds = pattern_to_conditions(pat)
        rule_str = conditions_to_rule_str(conds)
        parsed = parse_rule_conditions(rule_str)
        assert parsed is not None, f"unparseable: {rule_str}"
        assert parsed == conds, (
            f"parse round-trip mismatch for {rule_str}:\n"
            f"  original: {conds}\n"
            f"  parsed:   {parsed}"
        )
        dummy = {"conditions": parsed}
        for snap, mvp, flipped in snapshots:
            direct = pattern_fires_on_snapshot(pat, snap, mvp, flipped)
            via_rule = evaluate_restriction(dummy, snap, mvp, flipped)
            if direct != via_rule:
                raise AssertionError(
                    f"Mismatch on pattern {pattern_id(pat)} "
                    f"(rule={rule_str}): direct={direct} via_rule={via_rule}"
                )
            if direct:
                fire_totals += 1
            n_checked += 1
        n_match += 1
        if n_match % 100 == 0:
            print(f"  {n_match}/{len(patterns)} patterns OK "
                  f"({n_checked} snapshot checks, {fire_totals} fires)")

    print(f"ALL {len(patterns)} patterns round-trip cleanly "
          f"({n_checked} snapshot checks, {fire_totals} total fires).")


if __name__ == "__main__":
    _round_trip_test(n_games=50)
