"""
Augment an existing 2x2 run with flanking-pattern antecedent arms (F1, F2).

Given a completed or partially-built run directory (the output of
`build_restriction_configs.py`), this script:

    1. Loads `configs/manifest.json` + `configs/B1.json` + `configs/B2.json`
       to recover each quadruple's aligned and random consequent squares
       and its aligned antecedent firing rate.

    2. Enumerates the 960 flanking patterns from `hand_crafted_flanking.py`
       and estimates each pattern's empirical firing rate on standard
       Othello snapshots (shared across all rows for efficiency).

    3. For each existing quadruple, selects ONE flanking pattern subject to:
         - Length-stratified coverage across the K-pattern set (patterns of
           lengths 1..6 are distributed evenly across quadruples).
         - Firing-rate closest to that quadruple's `fire_rate_aligned`
           (within `--max-firing-rate-diff`, default 0.05).
         - Pattern's target / terminal / opponent squares disjoint from the
           quadruple's aligned + random consequent squares (preserves the
           "consequent never in antecedent" invariant the existing
           pipeline enforces).

    4. Emits two new config JSONs with schema identical to B1/B2:
         F1.json  =  flanking antecedent  +  cons_aligned (from B1)
         F2.json  =  flanking antecedent  +  cons_random (from B2)
       Each restriction in F1/F2 carries a `flanking_pattern` block with
       the pattern id, target/terminal/length/direction, and empirical
       firing rate.

    5. Writes `flanking_manifest.json` alongside the existing
       `manifest.json` with per-quadruple pattern selections and firing-rate
       diagnostics. `manifest.json` itself is NOT modified (so downstream
       scripts that read it keep working).

Usage:
    python augment_configs_with_flanking.py \\
        --base-run runs/2x2_20260415_160147 \\
        --snapshot-games 200

    # Reuse a run's own firing-rate cache after a first invocation:
    python augment_configs_with_flanking.py \\
        --base-run runs/2x2_20260415_160147  (cache at configs/flanking_fire_rates.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.othello import OthelloBoardState, get_ood_game  # noqa: E402
from restriction_utils import (  # noqa: E402
    board_pos_to_square, square_to_board_pos,
    get_flipped_squares,
)

from flanking_rule_adapter import (  # noqa: E402
    conditions_to_rule_str,
    direction_to_name,
    enumerate_flanking_patterns,
    estimate_pattern_fire_rates,
    pattern_id,
    pattern_to_conditions,
)


# ---------------------------------------------------------------------------
# Snapshot precomputation (matches build_restriction_configs._precompute_game_positions)
# ---------------------------------------------------------------------------

def precompute_snapshots(n_games):
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
    return snapshots


# ---------------------------------------------------------------------------
# Firing-rate cache (patterns never change, so we can cache per snapshot set)
# ---------------------------------------------------------------------------

def load_or_compute_fire_rates(patterns, snapshots, cache_path, n_games):
    """Return list of firing rates aligned with `patterns`.

    Cache format: {"n_games": int, "rates": [float, ...]}  — invalidated if
    `n_games` doesn't match (snapshots would differ).
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            if cache.get("n_games") == n_games and len(cache.get("rates", [])) == len(patterns):
                print(f"  Loaded cached firing rates from {cache_path}")
                return cache["rates"]
        except (json.JSONDecodeError, KeyError):
            pass

    print(f"  Estimating firing rates for {len(patterns)} patterns "
          f"on {len(snapshots)} snapshots...")
    rates = estimate_pattern_fire_rates(patterns, snapshots)

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump({"n_games": n_games, "rates": rates}, f)
        print(f"  Cached firing rates -> {cache_path}")
    return rates


# ---------------------------------------------------------------------------
# Pattern selection
# ---------------------------------------------------------------------------

def squares_occupied_by_pattern(pat):
    """Every board square the pattern mentions (target, opponents, terminal)."""
    occ = {pat["target"], pat["terminal"]}
    occ.update(pat["opponents"])
    return {board_pos_to_square(p) for p in occ}


def _assign_length_strata(K, lengths=(1, 2, 3, 4, 5, 6)):
    """Return a list of length K assigning a target flank-length to each row,
    distributing across `lengths` as evenly as possible.

    Example: K=20 with 6 lengths -> [1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2]
    (4 rows each for lengths 1-2, 3 rows each for lengths 3-6).
    """
    assignment = []
    for i in range(K):
        assignment.append(lengths[i % len(lengths)])
    return assignment


def select_flanking_patterns(quadruples, patterns, pattern_rates,
                              max_fire_rate_diff=0.05):
    """Pick one flanking pattern per quadruple.

    Length stratification is honored as a hard constraint (each row gets a
    pattern of its assigned target length), and firing-rate matching is
    best-effort within that length bucket. We flag a quadruple as an
    "offender" when its resulting |fire_rate_F - fire_rate_aligned| exceeds
    `max_fire_rate_diff`, but we still emit the arm — the mismatch is then
    a known confound to disclose, not a silent reassignment.

    If a target-length bucket has no pattern whose squares are disjoint
    from the quadruple's consequents, we fall back to the nearest length
    (target ±1, ±2, ...) and record the fallback in `selected_length`.
    This is rare in practice; the 960-pattern pool is dense.

    Returns:
        list[dict], one per quadruple, with keys:
            pattern_index       : int, index into `patterns`
            pattern             : the pattern dict
            fire_rate           : float, empirical firing rate on snapshots
            fire_rate_diff      : |fire_rate - fire_rate_aligned|
            target_length       : int, the length bucket requested
            selected_length     : int, the length bucket actually used
            unavailable         : None if fire_rate within threshold;
                                  explanation string otherwise
    """
    K = len(quadruples)
    target_lengths = _assign_length_strata(K)

    # Pre-index patterns by length
    by_length = defaultdict(list)  # length -> [(index, pattern, rate), ...]
    for idx, (pat, rate) in enumerate(zip(patterns, pattern_rates)):
        by_length[pat["length"]].append((idx, pat, rate))

    selections = []
    used_pattern_indices = set()

    for qi, q in enumerate(quadruples):
        target_len = target_lengths[qi]
        target_rate = q["fire_rate_aligned"]
        forbidden_sq = set(q["cons_aligned_squares"]) | set(q["cons_random_squares"])
        # Stratification: try target length first, fall back to nearest only
        # if NO disjoint pattern exists at the target length.
        length_order = [target_len] + sorted(
            [L for L in range(1, 7) if L != target_len],
            key=lambda L: abs(L - target_len),
        )

        chosen = None
        for L in length_order:
            candidates = by_length.get(L, [])
            # Within the length bucket, best firing-rate match wins.
            ranked = sorted(
                candidates,
                key=lambda t: (abs(t[2] - target_rate), t[2]),
            )
            for idx, pat, rate in ranked:
                if idx in used_pattern_indices:
                    continue
                pat_sq = squares_occupied_by_pattern(pat)
                if pat_sq & forbidden_sq:
                    continue
                diff = abs(rate - target_rate)
                chosen = {
                    "pattern_index": idx, "pattern": pat,
                    "fire_rate": rate, "fire_rate_diff": diff,
                    "target_length": target_len,
                    "selected_length": L,
                    "unavailable": None if diff <= max_fire_rate_diff else (
                        f"fire_rate_diff={diff:.4f} exceeds "
                        f"threshold {max_fire_rate_diff}"
                    ),
                }
                used_pattern_indices.add(idx)
                break
            if chosen is not None:
                break  # don't advance to other lengths once target is satisfied

        if chosen is None:
            chosen = {
                "pattern_index": None, "pattern": None,
                "fire_rate": None, "fire_rate_diff": None,
                "target_length": target_len, "selected_length": None,
                "unavailable": "no pattern with disjoint squares available",
            }

        selections.append(chosen)

    return selections


# ---------------------------------------------------------------------------
# Emit F1 / F2 restriction dicts (shape matches _restriction_for_arm in
# build_restriction_configs.py, with arm-specific fields stripped / swapped)
# ---------------------------------------------------------------------------

def _flanking_meta_block(pat, fire_rate):
    return {
        "pattern_id": pattern_id(pat),
        "target_board_pos": pat["target"],
        "target_square": board_pos_to_square(pat["target"]),
        "terminal_board_pos": pat["terminal"],
        "terminal_square": board_pos_to_square(pat["terminal"]),
        "opponent_board_positions": list(pat["opponents"]),
        "opponent_squares": [board_pos_to_square(p) for p in pat["opponents"]],
        "direction": list(pat["direction"]),
        "direction_name": direction_to_name(pat["direction"]),
        "length": pat["length"],
        "fire_rate_empirical": round(float(fire_rate), 4),
    }


def build_flanking_restriction(b_restriction, pat, fire_rate, arm, cons_kind):
    """Produce an F1 / F2 restriction dict using the B-arm's consequent.

    Handles both schemas:
      - New: `forbidden_positions` (list) + `forbidden_squares` (list) +
             legacy scalar fields.
      - Old: only scalar `forbidden_position` / `forbidden_square`
             (N_FORBIDDEN=1 legacy runs).
    In the legacy case we emit length-1 lists so downstream consumers (and
    restriction_utils._get_forbidden_positions) always find the expected
    plural fields.
    """
    conditions = pattern_to_conditions(pat)
    rule_str = conditions_to_rule_str(conditions)

    if "forbidden_positions" in b_restriction and \
            b_restriction["forbidden_positions"] is not None:
        forbidden_positions = list(b_restriction["forbidden_positions"])
        forbidden_squares = list(b_restriction["forbidden_squares"])
    else:
        forbidden_positions = [b_restriction["forbidden_position"]]
        forbidden_squares = [b_restriction["forbidden_square"]]

    return {
        # Identity — keep quadruple_id so matched pairs can be plotted together
        "id": b_restriction["id"],
        "quadruple_id": b_restriction["quadruple_id"],
        "arm": arm,
        # Source: a flanking pattern instead of a neuron
        "source_neuron": None,
        "source_flanking_pattern": pattern_id(pat),
        "antecedent_kind": "flanking",
        "consequent_kind": cons_kind,
        # Antecedent: the flanking conjunction
        "conditions": conditions,
        "rule_str": rule_str,
        # Consequent: inherited verbatim from the B arm
        "forbidden_positions": forbidden_positions,
        "forbidden_squares": forbidden_squares,
        "forbidden_position": forbidden_positions[0],
        "forbidden_square": forbidden_squares[0],
        "targets": b_restriction.get("targets", []),
        # Firing-rate bookkeeping (mirrors existing `fire_rate` field shape)
        "fire_rate": round(float(fire_rate), 4),
        # Full pattern details (for reproducibility / inspection)
        "flanking_pattern": _flanking_meta_block(pat, fire_rate),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Augment an existing 2x2 run with F1/F2 flanking-antecedent "
                    "arms that reuse the run's aligned/random consequents.")
    parser.add_argument("--base-run", type=str, required=True,
                        help="Path to an existing run directory (must contain "
                             "configs/manifest.json and configs/B1.json+B2.json).")
    parser.add_argument("--snapshot-games", type=int, default=200,
                        help="Games used to estimate flanking firing rates. "
                             "Default mirrors build_restriction_configs's 200.")
    parser.add_argument("--max-firing-rate-diff", type=float, default=0.05,
                        help="Soft threshold; patterns with |diff| > this are "
                             "still selected if nothing else fits, but reported.")
    parser.add_argument("--strict", action="store_true",
                        help="Hard-fail if any quadruple exceeds threshold.")
    parser.add_argument("--fire-rate-cache", type=str, default=None,
                        help="Optional path for the pattern firing-rate cache. "
                             "Defaults to <base-run>/configs/flanking_fire_rates.json.")
    args = parser.parse_args()

    configs_dir = os.path.join(args.base_run, "configs")
    manifest_path = os.path.join(configs_dir, "manifest.json")
    b1_path = os.path.join(configs_dir, "B1.json")
    b2_path = os.path.join(configs_dir, "B2.json")
    for p in (manifest_path, b1_path, b2_path):
        if not os.path.exists(p):
            print(f"ERROR: required file missing: {p}", file=sys.stderr)
            sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(b1_path) as f:
        b1_cfg = json.load(f)
    with open(b2_path) as f:
        b2_cfg = json.load(f)

    quadruples = manifest["per_quadruple"]
    # Normalize legacy single-square consequent schema (older runs with
    # N_FORBIDDEN=1 wrote `cons_aligned_square` / `cons_random_square`
    # as scalars; newer runs use `cons_aligned_squares` / `cons_random_squares`
    # lists). We fold scalars into length-1 lists so downstream code is uniform.
    for q in quadruples:
        if "cons_aligned_squares" not in q and "cons_aligned_square" in q:
            q["cons_aligned_squares"] = [q["cons_aligned_square"]]
        if "cons_random_squares" not in q and "cons_random_square" in q:
            q["cons_random_squares"] = [q["cons_random_square"]]
    K = len(quadruples)
    print(f"Loaded {K} existing quadruples from {manifest_path}")

    # --- Snapshots + firing-rate cache ---
    print(f"\nPrecomputing {args.snapshot_games} sample games for firing-rate "
          f"estimation...")
    snapshots = precompute_snapshots(args.snapshot_games)
    print(f"  {len(snapshots)} positions")

    cache_path = args.fire_rate_cache or os.path.join(
        configs_dir, "flanking_fire_rates.json")
    patterns = enumerate_flanking_patterns()
    pattern_rates = load_or_compute_fire_rates(
        patterns, snapshots, cache_path, args.snapshot_games)
    print(f"  {len(patterns)} flanking patterns; firing-rate summary: "
          f"mean={np.mean(pattern_rates):.4f} "
          f"median={np.median(pattern_rates):.4f} "
          f"max={np.max(pattern_rates):.4f}")

    # --- Pattern selection ---
    print(f"\nSelecting {K} flanking patterns (stratified × firing-rate matched)...")
    selections = select_flanking_patterns(
        quadruples, patterns, pattern_rates,
        max_fire_rate_diff=args.max_firing_rate_diff,
    )

    # --- Report ---
    offenders = []
    print(f"\n{'#':<3} {'quadruple':<18} {'pat_id':<18} "
          f"{'targL':<6} {'selL':<6} {'fire_F':<8} {'fire_B':<8} "
          f"{'|diff|':<8} {'status'}")
    print("-" * 100)
    for i, (q, sel) in enumerate(zip(quadruples, selections)):
        if sel["pattern"] is None:
            offenders.append((i, q["quadruple_id"], "NO PATTERN"))
            print(f"{i:<3} {q['quadruple_id']:<18} "
                  f"{'-':<18} {sel['target_length']:<6} {'-':<6} "
                  f"{'-':<8} {q['fire_rate_aligned']:<8.4f} {'-':<8} DROP")
            continue
        status = "OK" if sel["unavailable"] is None else "OVER"
        print(f"{i:<3} {q['quadruple_id']:<18} "
              f"{pattern_id(sel['pattern']):<18} "
              f"{sel['target_length']:<6} {sel['selected_length']:<6} "
              f"{sel['fire_rate']:<8.4f} {q['fire_rate_aligned']:<8.4f} "
              f"{sel['fire_rate_diff']:<8.4f} {status}")
        if sel["unavailable"] is not None:
            offenders.append((i, q["quadruple_id"], sel["unavailable"]))

    if offenders:
        msg = f"{len(offenders)}/{K} quadruples over threshold or without pattern:"
        print(f"\n{msg}")
        for i, qid, reason in offenders:
            print(f"  row {i} ({qid}): {reason}")
        if args.strict:
            sys.exit(1)

    # Drop rows with no pattern (cannot emit F1/F2 for them)
    paired_rows = [(q, sel, b1_cfg["restrictions"][i], b2_cfg["restrictions"][i])
                   for i, (q, sel) in enumerate(zip(quadruples, selections))
                   if sel["pattern"] is not None]
    if not paired_rows:
        print("\nERROR: no rows with a valid flanking pattern; cannot emit F1/F2.")
        sys.exit(1)

    # --- Build F1 / F2 restrictions ---
    f1_restrictions = []
    f2_restrictions = []
    for q, sel, b1_r, b2_r in paired_rows:
        pat = sel["pattern"]
        fr = sel["fire_rate"]
        f1_restrictions.append(
            build_flanking_restriction(b1_r, pat, fr, arm="F1", cons_kind="aligned"))
        f2_restrictions.append(
            build_flanking_restriction(b2_r, pat, fr, arm="F2", cons_kind="random"))

    # --- Augment meta with flanking-specific provenance ---
    flanking_meta = dict(b1_cfg.get("meta", {}))
    flanking_meta.update({
        "flanking_snapshot_games": args.snapshot_games,
        "flanking_max_firing_rate_diff": args.max_firing_rate_diff,
        "flanking_source": "hand_crafted_flanking.enumerate_flanking_patterns (960 patterns)",
    })

    arm_descriptions = {
        "F1": "Flanking antecedent + aligned consequent "
              "(inherits cons_aligned from B1)",
        "F2": "Flanking antecedent + random consequent "
              "(inherits cons_random from B2)",
    }

    for arm, restrictions in (("F1", f1_restrictions), ("F2", f2_restrictions)):
        config = {
            "label": arm,
            "description": arm_descriptions[arm],
            "num_restrictions": len(restrictions),
            "meta": flanking_meta,
            "restrictions": restrictions,
        }
        out_path = os.path.join(configs_dir, f"{arm}.json")
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nSaved {arm} ({len(restrictions)} restrictions) -> {out_path}")

    # --- Flanking-only manifest (augments, does not replace, manifest.json) ---
    flanking_manifest = {
        "meta": flanking_meta,
        "base_manifest": os.path.relpath(manifest_path, configs_dir),
        "arms_emitted": ["F1", "F2"],
        "per_quadruple": [
            {
                "quadruple_id": q["quadruple_id"],
                "fire_rate_aligned_heuristic": q["fire_rate_aligned"],
                "pattern_id": pattern_id(sel["pattern"]) if sel["pattern"] else None,
                "pattern_target_square": (
                    board_pos_to_square(sel["pattern"]["target"])
                    if sel["pattern"] else None
                ),
                "pattern_length": sel["selected_length"],
                "pattern_target_length_bucket": sel["target_length"],
                "fire_rate_flanking": sel["fire_rate"],
                "fire_rate_diff": sel["fire_rate_diff"],
                "status": "OK" if sel["unavailable"] is None else sel["unavailable"],
            }
            for q, sel in zip(quadruples, selections)
        ],
        "offenders": [
            {"row": i, "quadruple_id": qid, "reason": reason}
            for i, qid, reason in offenders
        ],
    }
    fm_path = os.path.join(configs_dir, "flanking_manifest.json")
    with open(fm_path, "w") as f:
        json.dump(flanking_manifest, f, indent=2)
    print(f"Saved flanking manifest -> {fm_path}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"Flanking augmentation summary")
    print(f"{'=' * 60}")
    diffs = [s["fire_rate_diff"] for s in selections if s["fire_rate_diff"] is not None]
    if diffs:
        print(f"  Fire-rate |diff|: "
              f"mean={np.mean(diffs):.4f} median={np.median(diffs):.4f} "
              f"max={np.max(diffs):.4f}")
    length_counts = defaultdict(int)
    for s in selections:
        if s["selected_length"] is not None:
            length_counts[s["selected_length"]] += 1
    print(f"  Length coverage: "
          f"{dict(sorted(length_counts.items()))}")
    print(f"  Offenders: {len(offenders)}/{K}")
    print(f"  Emitted arms: F1, F2")


if __name__ == "__main__":
    main()
