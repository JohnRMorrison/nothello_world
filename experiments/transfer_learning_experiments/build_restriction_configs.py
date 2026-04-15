"""
Build restriction configs for a 2x2 factorial transfer learning experiment.

The 2x2 crosses {antecedent, consequent} x {aligned, random}, producing four
conditions per quadruple:

    B1 = (Ant_aligned, Cons_aligned)   — extracted rule + DLA-argmax target
    B2 = (Ant_aligned, Cons_random)    — extracted rule + random target
    B3 = (Ant_random,  Cons_aligned)   — random antecedent + DLA-argmax target
    C  = (Ant_random,  Cons_random)    — random both (full control)

For each selected neuron, we co-construct all four arms so they share the
same `quadruple_id` and differ only on the intended axes. Construction order
(avoids circular self-reference constraints):

    1. Ant_aligned from the neuron's extracted rule;       squares = S_A
    2. Cons_aligned = tautology-aware DLA argmax, excluding S_A
    3. Ant_random   = frequency-matched, excluding S_A u {Cons_aligned}; squares = S_R
    4. Cons_random  = uniform random, excluding S_A u S_R u {Cons_aligned}

Firing-rate matching: B1 and B2 share antecedent (identical fire rates by
construction); B3 is matched to B1 via best-of-N random sampling; C inherits
B3's antecedent. Post-build we hard-fail if any |B1 - B3| exceeds
--max-firing-rate-diff.

Usage:
    python build_restriction_configs.py \\
        --rules ../reverse_engineering_experiments/rules_085_200_2-6.json \\
        --output-dir configs/2x2_run1

    # Only emit a subset:
    python build_restriction_configs.py --rules ... --emit B1,C
"""

import argparse
import json
import os
import random
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "../.."))

from restriction_utils import (
    VALID_POSITIONS, ITOS,
    board_pos_to_square, square_to_board_pos,
    parse_rule_conditions, evaluate_restriction, apply_restrictions,
    get_flipped_squares,
)


ARMS = ["B1", "B2", "B3", "C"]


# ---------------------------------------------------------------------------
# Neuron selection
# ---------------------------------------------------------------------------

def select_neurons(rules_data, layers, K, min_conditions=2):
    """Pick top-K neurons by influence from the specified layers."""
    candidates = []
    for layer_str, neurons in rules_data.items():
        layer = int(layer_str)
        if layer not in layers:
            continue
        for neuron_str, info in neurons.items():
            neuron = int(neuron_str)
            influence = info.get("influence_score")
            if influence is None:
                continue
            rules = info.get("rules", [])
            if not rules:
                continue
            best = rules[0]
            conds = parse_rule_conditions(best["rule"])
            if conds is None or len(conds) < min_conditions:
                continue
            candidates.append({
                "layer": layer,
                "neuron": neuron,
                "influence_score": influence,
                "f1": info.get("test_F1", 0),
                "conditions": conds,
                "rule_str": best["rule"],
                "precision": best.get("precision", 0),
                "samples_frac": best["samples"] / info["total_training_samples"],
            })

    candidates.sort(key=lambda x: -x["influence_score"])
    # Over-select: some neurons will fail exclusion / tautology checks.
    overselect = min(len(candidates), K * 3)
    selected = candidates[:overselect]
    print(f"Pre-selected {len(selected)} neurons from {len(candidates)} candidates "
          f"(layers {sorted(layers)}, will trim to K={K} after filtering)")
    return selected


# ---------------------------------------------------------------------------
# Game snapshot precomputation (used for tautology + firing-rate estimation)
# ---------------------------------------------------------------------------

def _precompute_game_positions(n_games):
    """Generate sample games; return per-position snapshots + legality matrix."""
    from data.othello import OthelloBoardState, get_ood_game

    snapshots = []
    legal_rows = []

    for gi in range(n_games):
        game = get_ood_game(gi)
        board = OthelloBoardState()
        for move in game[:-1]:
            state_before = board.state.copy()
            board.umpire(move)
            flipped = get_flipped_squares(state_before, board.state, move)
            snap = SimpleNamespace(
                state=board.state.copy(),
                next_hand_color=board.next_hand_color,
            )
            snapshots.append((snap, move, flipped))

            row = np.zeros(64, dtype=bool)
            for m in board.get_valid_moves():
                row[m] = True
            legal_rows.append(row)

    return snapshots, np.array(legal_rows)


def _cond_fires_on_snapshots(conditions, snapshots):
    """Return a bool array [N] of whether `conditions` (ANDed) fire on each snapshot."""
    dummy = {"conditions": conditions}
    return np.array([
        evaluate_restriction(dummy, snap, move, flipped)
        for snap, move, flipped in snapshots
    ], dtype=bool)


# ---------------------------------------------------------------------------
# Consequent helpers
# ---------------------------------------------------------------------------

def choose_aligned_consequent(dla, cond_fires, legal_matrix, forbid_positions,
                              tautology_threshold, min_legal=10):
    """Walk DLA targets in descending strength; pick the first non-tautological
    target whose board position is not in `forbid_positions`.

    Returns a dict {target_board_pos, target_square, dla_value, tautology_score,
    dla_rank} or None if no target satisfies the constraints.
    """
    sorted_indices = np.argsort(-dla)
    for rank, idx in enumerate(sorted_indices):
        target_pos = VALID_POSITIONS[idx]
        if target_pos in forbid_positions:
            continue
        target_legal = legal_matrix[:, target_pos]
        n_legal = int(target_legal.sum())
        if n_legal < min_legal:
            continue
        n_fires_when_legal = int((cond_fires & target_legal).sum())
        tautology_score = n_fires_when_legal / n_legal
        if tautology_score <= tautology_threshold:
            return {
                "target_board_pos": target_pos,
                "target_square": board_pos_to_square(target_pos),
                "dla_value": round(float(dla[idx]), 4),
                "tautology_score": round(tautology_score, 4),
                "dla_rank": int(rank),
            }
    return None


def choose_random_consequent(forbid_positions, rng):
    """Uniform sample from VALID_POSITIONS minus `forbid_positions`.

    Returns a consequent dict in the same shape as the aligned version
    (dla_value / tautology_score / dla_rank are None for random).
    """
    available = [p for p in VALID_POSITIONS if p not in forbid_positions]
    if not available:
        return None
    pos = rng.choice(available)
    return {
        "target_board_pos": pos,
        "target_square": board_pos_to_square(pos),
        "dla_value": None,
        "tautology_score": None,
        "dla_rank": None,
    }


# ---------------------------------------------------------------------------
# Antecedent randomization
# ---------------------------------------------------------------------------

def _generate_one_random(aligned_conditions, forbidden_squares, rng):
    """Replace each distinct square in `aligned_conditions` with a random
    different square, avoiding anything in `forbidden_squares`.

    Preserves number of conditions, feature types, polarities, and which
    conditions share the same square. Returns None if not enough squares
    available.

    Args:
        aligned_conditions: list of condition dicts.
        forbidden_squares: set[str] of squares to avoid as replacements.
        rng: random.Random instance.
    """
    all_squares = [board_pos_to_square(i) for i in range(64)]
    original_squares = list({c["square"] for c in aligned_conditions})

    exclude = set(original_squares) | set(forbidden_squares or ())
    available = [s for s in all_squares if s not in exclude]
    if len(available) < len(original_squares):
        return None
    rng.shuffle(available)
    replacement = {sq: available.pop() for sq in original_squares}

    return [
        {
            "square": replacement[c["square"]],
            "feature_type": c["feature_type"],
            "polarity": c["polarity"],
        }
        for c in aligned_conditions
    ]


def generate_frequency_matched_random(aligned_conditions, forbidden_squares,
                                      target_fire_rate, snapshots, rng,
                                      n_attempts=50):
    """Try many random square reassignments; return the one with fire rate
    closest to `target_fire_rate`.

    Returns (conditions_list, actual_fire_rate) or (None, None) if no valid
    assignment could be produced.
    """
    best_conds = None
    best_diff = float("inf")
    best_rate = 0.0
    N = len(snapshots)

    for _ in range(n_attempts):
        cands = _generate_one_random(aligned_conditions, forbidden_squares, rng)
        if cands is None:
            continue
        fires = int(_cond_fires_on_snapshots(cands, snapshots).sum())
        rate = fires / N
        diff = abs(rate - target_fire_rate)
        if diff < best_diff:
            best_diff = diff
            best_conds = cands
            best_rate = rate

    if best_conds is None:
        return None, None
    return best_conds, best_rate


# ---------------------------------------------------------------------------
# Per-neuron quadruple construction
# ---------------------------------------------------------------------------

def build_quadruple(neuron_info, state_dict, snapshots, legal_matrix,
                    rng, tautology_threshold, n_random_attempts=50):
    """Construct all four arms for one neuron.

    Construction order (see module docstring) is strict: ant_aligned,
    cons_aligned, ant_random (with ant_aligned U cons_aligned forbidden),
    cons_random (with ant_aligned U ant_random U cons_aligned forbidden).

    Returns a dict with keys:
        quadruple_id (str), source_neuron (str), layer (int), neuron (int),
        ant_aligned, ant_random,
        cons_aligned, cons_random,
        fire_rate_aligned (float), fire_rate_random (float),
        rule_str (str), influence_score, f1
    or None if any arm cannot be satisfied.
    """
    layer, neuron = neuron_info["layer"], neuron_info["neuron"]
    src = f"L{layer}N{neuron}"
    quadruple_id = f"{src}_r0"

    # --- 1. Ant_aligned ---
    ant_aligned = neuron_info["conditions"]
    S_A_squares = {c["square"] for c in ant_aligned}
    S_A_positions = {square_to_board_pos(s) for s in S_A_squares}

    cond_fires_A = _cond_fires_on_snapshots(ant_aligned, snapshots)
    fire_rate_aligned = float(cond_fires_A.sum() / len(snapshots))

    # --- 2. Cons_aligned ---
    W_U = state_dict["head.weight"]
    W_out_col = state_dict[f"blocks.{layer}.mlp.2.weight"][:, neuron]
    dla = (W_U @ W_out_col)[1:].detach().numpy()  # [60]

    cons_aligned = choose_aligned_consequent(
        dla, cond_fires_A, legal_matrix,
        forbid_positions=S_A_positions,
        tautology_threshold=tautology_threshold,
    )
    if cons_aligned is None:
        return None  # no non-tautological DLA target exists

    # --- 3. Ant_random (freq-matched; excludes S_A and cons_aligned square) ---
    forbidden_ant_squares = set(S_A_squares) | {cons_aligned["target_square"]}
    ant_random, fire_rate_random = generate_frequency_matched_random(
        ant_aligned, forbidden_ant_squares, fire_rate_aligned,
        snapshots, rng, n_attempts=n_random_attempts,
    )
    if ant_random is None:
        return None

    S_R_squares = {c["square"] for c in ant_random}

    # --- 4. Cons_random (uniform; excludes S_A u S_R u cons_aligned) ---
    forbidden_cons_positions = (
        S_A_positions
        | {square_to_board_pos(s) for s in S_R_squares}
        | {cons_aligned["target_board_pos"]}
    )
    cons_random = choose_random_consequent(forbidden_cons_positions, rng)
    if cons_random is None:
        return None

    return {
        "quadruple_id": quadruple_id,
        "source_neuron": src,
        "layer": layer,
        "neuron": neuron,
        "ant_aligned": ant_aligned,
        "ant_random": ant_random,
        "cons_aligned": cons_aligned,
        "cons_random": cons_random,
        "fire_rate_aligned": fire_rate_aligned,
        "fire_rate_random": fire_rate_random,
        "rule_str": neuron_info["rule_str"],
        "influence_score": round(neuron_info["influence_score"], 6),
        "f1": round(neuron_info["f1"], 4),
    }


# ---------------------------------------------------------------------------
# Assemble the four per-arm restriction dicts from a quadruple
# ---------------------------------------------------------------------------

def _restriction_for_arm(q, arm):
    """Build the JSON-serializable restriction dict for one arm of a quadruple."""
    if arm == "B1":
        ant, cons, ant_kind, cons_kind = (
            q["ant_aligned"], q["cons_aligned"], "aligned", "aligned")
        fire_rate = q["fire_rate_aligned"]
    elif arm == "B2":
        ant, cons, ant_kind, cons_kind = (
            q["ant_aligned"], q["cons_random"], "aligned", "random")
        fire_rate = q["fire_rate_aligned"]
    elif arm == "B3":
        ant, cons, ant_kind, cons_kind = (
            q["ant_random"], q["cons_aligned"], "random", "aligned")
        fire_rate = q["fire_rate_random"]
    elif arm == "C":
        ant, cons, ant_kind, cons_kind = (
            q["ant_random"], q["cons_random"], "random", "random")
        fire_rate = q["fire_rate_random"]
    else:
        raise ValueError(f"unknown arm: {arm}")

    return {
        "id": q["quadruple_id"],
        "quadruple_id": q["quadruple_id"],
        "arm": arm,
        "source_neuron": q["source_neuron"],
        "antecedent_kind": ant_kind,
        "consequent_kind": cons_kind,
        "conditions": ant,
        # Legacy keys (kept for compatibility with generate_restricted_games.py
        # and finetune_and_evaluate.py which read these directly):
        "forbidden_position": cons["target_board_pos"],
        "forbidden_square": cons["target_square"],
        # New explicit keys:
        "target_board_pos": cons["target_board_pos"],
        "target_square": cons["target_square"],
        "dla_value": cons["dla_value"],
        "tautology_score": cons["tautology_score"],
        "dla_rank": cons["dla_rank"],
        "fire_rate": round(fire_rate, 4),
        "rule_str": q["rule_str"],
        "influence_score": q["influence_score"],
        "neuron_f1": q["f1"],
        "rule_source": {
            "layer": q["layer"],
            "neuron": q["neuron"],
            "rule_idx": 0,
        },
    }


# ---------------------------------------------------------------------------
# Quadruple invariant checks
# ---------------------------------------------------------------------------

def verify_quadruple(q):
    """Assert all self-reference exclusions and structural invariants."""
    S_A = {c["square"] for c in q["ant_aligned"]}
    S_R = {c["square"] for c in q["ant_random"]}
    c_aligned_sq = q["cons_aligned"]["target_square"]
    c_random_sq = q["cons_random"]["target_square"]

    assert c_aligned_sq not in S_A, (
        f"{q['quadruple_id']}: cons_aligned {c_aligned_sq} in S_A {S_A}")
    assert c_aligned_sq not in S_R, (
        f"{q['quadruple_id']}: cons_aligned {c_aligned_sq} in S_R {S_R}")
    assert c_random_sq not in S_A, (
        f"{q['quadruple_id']}: cons_random {c_random_sq} in S_A {S_A}")
    assert c_random_sq not in S_R, (
        f"{q['quadruple_id']}: cons_random {c_random_sq} in S_R {S_R}")
    assert c_aligned_sq != c_random_sq, (
        f"{q['quadruple_id']}: cons_aligned == cons_random ({c_aligned_sq})")
    assert len(q["ant_aligned"]) == len(q["ant_random"]), (
        f"{q['quadruple_id']}: antecedent length mismatch")


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def parse_layers(spec):
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.update(range(int(lo), int(hi) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def parse_emit(spec):
    if spec == "all":
        return list(ARMS)
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if p not in ARMS:
            raise argparse.ArgumentTypeError(
                f"--emit value {p!r} not in {ARMS + ['all']}")
    return parts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build 2x2 factorial restriction configs "
                    "(B1 / B2 / B3 / C) for transfer learning experiments")
    parser.add_argument("--rules", type=str, required=True,
                        help="Path to rules JSON from extract_rules.py")
    parser.add_argument("--ckpt", type=str, default="../../ckpts/gpt_synthetic.ckpt",
                        help="Path to pre-trained minGPT checkpoint")
    parser.add_argument("--layers", type=str, default="2-5",
                        help="Layers to draw neurons from (e.g. '2-5', '2,3,4,5')")
    parser.add_argument("--K", type=int, default=20,
                        help="Target number of surviving quadruples")
    parser.add_argument("--min-conditions", type=int, default=2,
                        help="Minimum conditions per rule (default: 2 to avoid "
                             "single-feature tautologies)")
    parser.add_argument("--tautology-threshold", type=float, default=0.85,
                        help="Max P(cond fires | target legal). Targets above "
                             "this are skipped. Default: 0.85")
    parser.add_argument("--tautology-games", type=int, default=200,
                        help="Sample games for snapshot precomputation")
    parser.add_argument("--n-random-attempts", type=int, default=50,
                        help="Best-of-N for frequency-matched random antecedent")
    parser.add_argument("--max-firing-rate-diff", type=float, default=0.05,
                        help="Hard-fail if any quadruple's |rate_aligned - "
                             "rate_random| exceeds this threshold")
    parser.add_argument("--strict-K", action="store_true",
                        help="Hard-fail if fewer than K quadruples survive "
                             "(default: warn and continue)")
    parser.add_argument("--emit", type=parse_emit, default="all",
                        help="Which arms to write: comma-separated subset of "
                             "B1,B2,B3,C, or 'all' (default)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output JSON files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)
    layers = parse_layers(args.layers)
    emit_arms = args.emit if isinstance(args.emit, list) else parse_emit(args.emit)

    # --- Load rules ---
    with open(args.rules) as f:
        rules_data = json.load(f)

    pre_selected = select_neurons(rules_data, layers, args.K,
                                  min_conditions=args.min_conditions)
    if not pre_selected:
        print("ERROR: no neurons selected — check --rules, --layers, --K")
        sys.exit(1)

    # --- Load model weights for DLA ---
    ckpt_path = args.ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)
    print(f"\nLoading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # --- Precompute board snapshots (shared across all neurons) ---
    print(f"\nPrecomputing {args.tautology_games} sample games for tautology / "
          f"firing-rate estimation...")
    snapshots, legal_matrix = _precompute_game_positions(args.tautology_games)
    print(f"  {len(snapshots)} total positions")

    # --- Build quadruples ---
    print(f"\nBuilding quadruples (target K={args.K}, overselect pool="
          f"{len(pre_selected)})...")
    quadruples = []
    drops = {"cons_aligned_fail": 0, "ant_random_fail": 0, "cons_random_fail": 0}

    for info in pre_selected:
        if len(quadruples) >= args.K:
            break
        q = build_quadruple(
            info, state_dict, snapshots, legal_matrix,
            rng=random.Random(args.seed + info["layer"] * 10_000 + info["neuron"]),
            tautology_threshold=args.tautology_threshold,
            n_random_attempts=args.n_random_attempts,
        )
        if q is None:
            # Recompute causes to give useful diagnostics.
            # (Cheap: we already did most of the work inside build_quadruple.)
            S_A = {c["square"] for c in info["conditions"]}
            S_A_pos = {square_to_board_pos(s) for s in S_A}
            cond_fires_A = _cond_fires_on_snapshots(info["conditions"], snapshots)
            W_U = state_dict["head.weight"]
            W_out_col = state_dict[f"blocks.{info['layer']}.mlp.2.weight"][:, info["neuron"]]
            dla = (W_U @ W_out_col)[1:].detach().numpy()
            cons = choose_aligned_consequent(
                dla, cond_fires_A, legal_matrix, S_A_pos,
                args.tautology_threshold,
            )
            if cons is None:
                drops["cons_aligned_fail"] += 1
                reason = "no non-tautological DLA target"
            else:
                drops["ant_random_fail"] += 1
                reason = "no valid frequency-matched antecedent"
            print(f"  L{info['layer']}N{info['neuron']}: DROPPED ({reason})")
            continue

        try:
            verify_quadruple(q)
        except AssertionError as e:
            print(f"  INTERNAL ERROR: {e}")
            sys.exit(2)

        diff = abs(q["fire_rate_aligned"] - q["fire_rate_random"])
        print(f"  {q['source_neuron']:<10} target={q['cons_aligned']['target_square']} "
              f"(rank {q['cons_aligned']['dla_rank']})  "
              f"cons_rand={q['cons_random']['target_square']}  "
              f"fire_A={q['fire_rate_aligned']:.3f} fire_R={q['fire_rate_random']:.3f} "
              f"diff={diff:.3f}  rule: {q['rule_str']}")
        quadruples.append(q)

    # --- Check K ---
    if len(quadruples) < args.K:
        msg = (f"Only {len(quadruples)}/{args.K} quadruples survived "
               f"(drops: {drops}).")
        if args.strict_K:
            print(f"\nERROR: {msg}")
            sys.exit(1)
        print(f"\nWARNING: {msg}")

    if not quadruples:
        print("ERROR: zero quadruples; nothing to emit.")
        sys.exit(1)

    # --- Firing-rate hard-fail ---
    print(f"\nFiring-rate diff check (threshold={args.max_firing_rate_diff}):")
    offenders = []
    for q in quadruples:
        diff = abs(q["fire_rate_aligned"] - q["fire_rate_random"])
        if diff > args.max_firing_rate_diff:
            offenders.append((q["source_neuron"], diff))
    if offenders:
        print("ERROR: firing-rate diff exceeds threshold for:")
        for src, diff in offenders:
            print(f"  {src}: diff={diff:.4f}")
        print(f"Raise --max-firing-rate-diff (currently {args.max_firing_rate_diff}) "
              f"or --n-random-attempts (currently {args.n_random_attempts}), "
              f"or loosen selection criteria.")
        sys.exit(1)
    print(f"  All {len(quadruples)} quadruples within threshold.")

    # --- Emit per-arm JSONs ---
    os.makedirs(args.output_dir, exist_ok=True)
    meta = {
        "K_target": args.K,
        "K_actual": len(quadruples),
        "layers": layers,
        "min_conditions": args.min_conditions,
        "tautology_threshold": args.tautology_threshold,
        "tautology_games": args.tautology_games,
        "max_firing_rate_diff": args.max_firing_rate_diff,
        "n_random_attempts": args.n_random_attempts,
        "seed": args.seed,
        "source_rules_file": args.rules,
        "ckpt": args.ckpt,
        "drops": drops,
    }

    arm_descriptions = {
        "B1": "Aligned antecedent + aligned consequent (full heuristic alignment)",
        "B2": "Aligned antecedent + random consequent (antecedent-only alignment)",
        "B3": "Random antecedent + aligned consequent (consequent-only alignment)",
        "C":  "Random antecedent + random consequent (no alignment; full control)",
    }

    for arm in emit_arms:
        restrictions = [_restriction_for_arm(q, arm) for q in quadruples]
        config = {
            "label": arm,
            "description": arm_descriptions[arm],
            "num_restrictions": len(restrictions),
            "meta": meta,
            "restrictions": restrictions,
        }
        out_path = os.path.join(args.output_dir, f"{arm}.json")
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved {arm} ({len(restrictions)} restrictions) -> {out_path}")

    # --- Manifest ---
    manifest = {
        "meta": meta,
        "arms_emitted": emit_arms,
        "quadruple_ids": [q["quadruple_id"] for q in quadruples],
        "per_quadruple": [
            {
                "quadruple_id": q["quadruple_id"],
                "source_neuron": q["source_neuron"],
                "fire_rate_aligned": round(q["fire_rate_aligned"], 4),
                "fire_rate_random": round(q["fire_rate_random"], 4),
                "fire_rate_diff": round(
                    abs(q["fire_rate_aligned"] - q["fire_rate_random"]), 4),
                "cons_aligned_square": q["cons_aligned"]["target_square"],
                "cons_random_square": q["cons_random"]["target_square"],
                "dla_rank": q["cons_aligned"]["dla_rank"],
                "tautology_score": q["cons_aligned"]["tautology_score"],
                "rule_str": q["rule_str"],
            }
            for q in quadruples
        ],
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest -> {manifest_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"2x2 factorial config summary ({len(quadruples)} quadruples)")
    print(f"{'='*60}")
    diffs = [abs(q["fire_rate_aligned"] - q["fire_rate_random"]) for q in quadruples]
    print(f"  Firing-rate diff: mean={np.mean(diffs):.4f} "
          f"median={np.median(diffs):.4f} max={np.max(diffs):.4f}")
    print(f"  Drops: {drops}")
    print(f"  Emitted arms: {', '.join(emit_arms)}")


if __name__ == "__main__":
    main()
