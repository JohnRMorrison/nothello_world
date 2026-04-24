"""
Zero-shot evaluation of pretrained OthelloGPT against restriction configs.

Measures how well the pretrained model (with NO fine-tuning) already satisfies
each condition's restrictions. The step-0 gap between aligned and random
conditions is direct evidence of causal heuristic engagement.

No game generation needed — only the restriction configs and checkpoint.

Usage:
    python zero_shot_eval.py \\
        --configs-dir runs/2x2_run1/configs/ \\
        --eval-games 1000 --seeds 10

    # Quick check:
    python zero_shot_eval.py \\
        --configs-dir runs/2x2_run1/configs/ \\
        --eval-games 50 --seeds 2
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from mingpt.model import GPT, GPTConfig
from mingpt.dataset import CharDataset
from mingpt.utils import set_seed
from data.othello import get_ood_game, OthelloBoardState

from finetune_and_evaluate import evaluate
from restriction_utils import get_flipped_squares


ARMS = ["B1", "B2", "B3", "C"]
METRICS = [
    "top1_legal", "top1_legal_when_fires",
    "violation_rate", "violation_rate_when_fires",
    "fire_rate", "legal_mass", "top1_prob",
    "n_positions", "n_fires",
]


def load_model(ckpt_path, device):
    """Load pretrained OthelloGPT."""
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Standard OthelloGPT config: 8 layers, 8 heads, 512 embed, vocab 61
    mconf = GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def build_token_maps():
    """Build stoi/itos from a dummy CharDataset."""
    # Generate a few games to get the vocabulary
    dummy_games = [get_ood_game(i) for i in range(10)]
    ds = CharDataset(dummy_games)
    return ds.stoi, ds.itos


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of pretrained OthelloGPT "
                    "against 2x2 restriction configs")
    parser.add_argument("--configs-dir", type=str, required=True,
                        help="Directory containing B1.json, B2.json, B3.json, C.json")
    parser.add_argument("--ckpt", type=str, default="../../ckpts/gpt_synthetic.ckpt",
                        help="Pretrained checkpoint path")
    parser.add_argument("--eval-games", type=int, default=1000,
                        help="Number of standard Othello games per eval seed")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of eval seeds (different game sets)")
    parser.add_argument("--structural", action="store_true",
                        help="Also evaluate structural controls A₁ (no diagonal "
                             "captures) and A₂ (quadrant dominance)")
    parser.add_argument("--n-quadrants", type=int, default=2,
                        help="Number of quadrants for A₂ (default: 2)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: stdout only)")
    args = parser.parse_args()

    # --- Device ---
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}", flush=True)

    # --- Load model ---
    ckpt_path = args.ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    model = load_model(ckpt_path, device)

    # --- Load configs ---
    configs = {}
    for arm in ARMS:
        path = os.path.join(args.configs_dir, f"{arm}.json")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found — skipping {arm}")
            continue
        with open(path) as f:
            configs[arm] = json.load(f)
        print(f"Loaded {arm}: {len(configs[arm]['restrictions'])} restrictions",
              flush=True)

    if not configs:
        print("ERROR: no configs loaded")
        sys.exit(1)

    # --- Build token maps ---
    stoi, itos = build_token_maps()

    # --- Set up structural conditions (if requested) ---
    structural_filters = {}
    if args.structural:
        from structural_conditions import (
            get_no_diagonal_legal_moves,
            get_quadrant_dominance_legal_moves,
            evaluate_structural,
        )
        from functools import partial
        structural_filters["A1_no_diagonal"] = get_no_diagonal_legal_moves
        structural_filters["A2_quadrant"] = partial(
            get_quadrant_dominance_legal_moves, n_quadrants=args.n_quadrants)
        print(f"Structural controls enabled: {list(structural_filters.keys())}",
              flush=True)

    all_arms = list(configs.keys()) + list(structural_filters.keys())

    # --- Run evaluation across seeds ---
    per_seed = {}
    for seed_idx in range(args.seeds):
        seed = seed_idx * 1000  # spread seeds
        set_seed(seed)
        eval_games = [get_ood_game(i) for i in range(args.eval_games)]

        seed_results = {}

        # Config-based arms (B1, B2, B3, C)
        for arm, config in configs.items():
            restrictions = config["restrictions"]
            metrics = evaluate(
                model, eval_games, restrictions, stoi, itos, device,
                max_games=args.eval_games,
            )
            seed_results[arm] = {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in metrics.items()
            }

        # Structural arms (A1, A2)
        for arm, filter_fn in structural_filters.items():
            metrics = evaluate_structural(
                model, eval_games, filter_fn, stoi, itos, device,
                max_games=args.eval_games,
            )
            seed_results[arm] = {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in metrics.items()
            }

        per_seed[f"seed_{seed_idx}"] = seed_results
        # Progress
        viol_str = "  ".join(
            f"{arm}={seed_results[arm].get('violation_rate_when_fires', 'n/a'):.4f}"
            if isinstance(seed_results[arm].get('violation_rate_when_fires'), float)
            else f"{arm}=n/a"
            for arm in all_arms if arm in seed_results
        )
        print(f"  Seed {seed_idx}/{args.seeds}: viol_fires: {viol_str}",
              flush=True)

    # --- Aggregate: mean ± SEM ---
    summary = {}
    for arm in all_arms:
        summary[arm] = {}
        for metric in METRICS:
            values = []
            for seed_key in per_seed:
                v = per_seed[seed_key].get(arm, {}).get(metric)
                if v is not None and isinstance(v, (int, float)):
                    values.append(v)
            if values:
                arr = np.array(values)
                summary[arm][metric] = {
                    "mean": round(float(arr.mean()), 6),
                    "std": round(float(arr.std(ddof=1)), 6) if len(arr) > 1 else 0.0,
                    "sem": round(float(arr.std(ddof=1) / np.sqrt(len(arr))), 6)
                           if len(arr) > 1 else 0.0,
                    "n": len(arr),
                }

    # --- Print summary table ---
    print(f"\n{'='*80}", flush=True)
    print(f"Zero-shot evaluation summary ({args.eval_games} games × "
          f"{args.seeds} seeds)", flush=True)
    print(f"{'='*80}", flush=True)

    key_metrics = ["top1_legal", "top1_legal_when_fires",
                   "violation_rate_when_fires", "fire_rate", "legal_mass"]
    header = f"{'Arm':<18}"
    for m in key_metrics:
        short = m.replace("_when_fires", "_wf").replace("violation_rate", "viol")
        header += f"  {short:<18}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for arm in all_arms:
        if arm not in summary:
            continue
        row = f"{arm:<18}"
        for m in key_metrics:
            s = summary[arm].get(m, {})
            mean = s.get("mean", 0)
            sem = s.get("sem", 0)
            row += f"  {mean:.4f} ± {sem:.4f}  "
        print(row, flush=True)

    # --- 2x2 contrast table ---
    print(f"\n2x2 violation_rate_when_fires:", flush=True)
    print(f"{'':>20} Cons:aligned    Cons:random", flush=True)
    for ant_label, arms in [("Ant:aligned", ("B1", "B2")),
                             ("Ant:random ", ("B3", "C"))]:
        vals = []
        for a in arms:
            s = summary.get(a, {}).get("violation_rate_when_fires", {})
            mean = s.get("mean", 0)
            sem = s.get("sem", 0)
            vals.append(f"{mean:.4f}±{sem:.4f}")
        print(f"  {ant_label:<20} {vals[0]:<16} {vals[1]}", flush=True)

    # --- Structural comparison (if present) ---
    if structural_filters:
        print(f"\nStructural controls comparison (violation_rate_when_fires):",
              flush=True)
        for arm in ["B1", "C"] + list(structural_filters.keys()):
            s = summary.get(arm, {}).get("violation_rate_when_fires", {})
            mean = s.get("mean", 0)
            sem = s.get("sem", 0)
            fr = summary.get(arm, {}).get("fire_rate", {}).get("mean", 0)
            print(f"  {arm:<18} viol={mean:.4f}±{sem:.4f}  fire_rate={fr:.4f}",
                  flush=True)

    # --- Base-rate diagnostic (config arms only) ---
    # For each arm, compute how often the model's argmax falls on the
    # union of all forbidden squares at ALL positions — ignoring whether
    # any antecedent fires. If DLA-aligned arms have a lower base rate
    # than random, the violation gap is explained by target selection
    # bias (DLA squares are simply unpopular predictions), not by
    # causal heuristic engagement.
    print(f"\nBase-rate diagnostic: P(argmax ∈ forbidden_set) over ALL "
          f"positions (ignoring antecedents):", flush=True)
    base_rates = {}
    for arm in configs:
        restrictions = configs[arm]["restrictions"]
        # Collect union of all forbidden positions for this arm
        from restriction_utils import _get_forbidden_positions
        all_forbidden = set()
        for r in restrictions:
            all_forbidden.update(_get_forbidden_positions(r))
        n_forbidden = len(all_forbidden)

        # Check argmax against this static set on the last seed's games
        hits = 0
        total = 0
        with torch.no_grad():
            for game in eval_games[:args.eval_games]:
                if len(game) < 2:
                    continue
                encoded = [stoi[m] for m in game]
                if len(encoded) > 60:
                    encoded = encoded[:60]
                x = torch.tensor(encoded[:-1], dtype=torch.long)[None].to(device)
                logits, _ = model(x)
                probs = F.softmax(logits[0], dim=-1)
                for pos in range(len(encoded) - 1):
                    pred_token = probs[pos].argmax().item()
                    pred_move = itos[pred_token]
                    if pred_move in all_forbidden:
                        hits += 1
                    total += 1
        base_rate = hits / max(total, 1)
        base_rates[arm] = {
            "base_rate": round(base_rate, 6),
            "n_forbidden_squares": n_forbidden,
            "n_positions": total,
        }
        print(f"  {arm:<4}  P(argmax ∈ forbidden) = {base_rate:.4f}  "
              f"({n_forbidden} forbidden squares)", flush=True)

    # Interpretation
    aligned_cons = [base_rates[a]["base_rate"] for a in ["B1", "B3"]
                    if a in base_rates]
    random_cons = [base_rates[a]["base_rate"] for a in ["B2", "C"]
                   if a in base_rates]
    if aligned_cons and random_cons:
        avg_a = np.mean(aligned_cons)
        avg_r = np.mean(random_cons)
        print(f"\n  Aligned-cons avg base rate: {avg_a:.4f}", flush=True)
        print(f"  Random-cons avg base rate:  {avg_r:.4f}", flush=True)
        if avg_a < avg_r * 0.85:
            print(f"  >> DLA targets are unpopular predictions (base rate "
                  f"{avg_a:.4f} vs {avg_r:.4f}). The violation gap is likely "
                  f"explained by target selection bias.", flush=True)
        else:
            print(f"  >> Base rates are similar — the violation gap is NOT "
                  f"explained by target selection bias alone.", flush=True)

    # --- Per-restriction conditional vs counterfactual ---
    # For each restriction individually, compare P(argmax ∈ S_15 | fires)
    # vs P(argmax ∈ S_15 | ¬fires). If the conditional is lower for aligned
    # consequents, the model specifically avoids those squares when it detects
    # the heuristic condition — clean causal evidence.
    print(f"\nPer-restriction conditional vs counterfactual diagnostic:",
          flush=True)
    from restriction_utils import evaluate_restriction
    per_restriction_results = {}
    with torch.no_grad():
        # Pre-compute model predictions on the last seed's eval games
        all_predictions = []  # list of (pred_move, board_state, move_played, flipped)
        for game in eval_games[:args.eval_games]:
            if len(game) < 2:
                continue
            encoded = [stoi[m] for m in game]
            if len(encoded) > 60:
                encoded = encoded[:60]
            x = torch.tensor(encoded[:-1], dtype=torch.long)[None].to(device)
            logits, _ = model(x)
            probs = F.softmax(logits[0], dim=-1)

            board = OthelloBoardState()
            for pos in range(len(encoded) - 1):
                move = game[pos]
                state_before = board.state.copy()
                board.umpire(move)
                flipped = get_flipped_squares(state_before, board.state, move)
                pred_token = probs[pos].argmax().item()
                pred_move = itos[pred_token]
                all_predictions.append((pred_move, board, move, flipped))

    for arm in configs:
        restrictions = configs[arm]["restrictions"]
        arm_conditionals = []
        arm_counterfactuals = []
        for ri, r in enumerate(restrictions):
            forbidden = set(_get_forbidden_positions(r))
            hits_fires, n_fires = 0, 0
            hits_no_fires, n_no_fires = 0, 0
            for pred_move, board, move_played, flipped in all_predictions:
                fires = evaluate_restriction(r, board, move_played, flipped)
                in_forbidden = pred_move in forbidden
                if fires:
                    n_fires += 1
                    if in_forbidden:
                        hits_fires += 1
                else:
                    n_no_fires += 1
                    if in_forbidden:
                        hits_no_fires += 1
            cond = hits_fires / max(n_fires, 1)
            counterfact = hits_no_fires / max(n_no_fires, 1)
            arm_conditionals.append(cond)
            arm_counterfactuals.append(counterfact)

        avg_cond = np.mean(arm_conditionals)
        avg_cf = np.mean(arm_counterfactuals)
        ratio = avg_cond / max(avg_cf, 1e-9)
        per_restriction_results[arm] = {
            "avg_conditional": round(avg_cond, 6),
            "avg_counterfactual": round(avg_cf, 6),
            "ratio": round(ratio, 4),
            "per_restriction": [
                {"conditional": round(c, 6), "counterfactual": round(cf, 6)}
                for c, cf in zip(arm_conditionals, arm_counterfactuals)
            ],
        }
        print(f"  {arm:<4}  P(argmax∈S|fires)={avg_cond:.4f}  "
              f"P(argmax∈S|¬fires)={avg_cf:.4f}  "
              f"ratio={ratio:.4f}", flush=True)

    # Interpretation
    aligned_ratios = [per_restriction_results[a]["ratio"]
                      for a in ["B1", "B3"] if a in per_restriction_results]
    random_ratios = [per_restriction_results[a]["ratio"]
                     for a in ["B2", "C"] if a in per_restriction_results]
    if aligned_ratios and random_ratios:
        print(f"\n  Aligned-cons avg ratio: {np.mean(aligned_ratios):.4f}  "
              f"(< 1.0 = model avoids these squares when antecedent fires)",
              flush=True)
        print(f"  Random-cons avg ratio:  {np.mean(random_ratios):.4f}  "
              f"(≈ 1.0 = no conditional avoidance)", flush=True)
        if np.mean(aligned_ratios) < 0.9 and np.mean(random_ratios) > 0.9:
            print(f"  >> CAUSAL EVIDENCE: model specifically avoids aligned "
                  f"forbidden squares when antecedent fires.", flush=True)
        elif np.mean(aligned_ratios) < np.mean(random_ratios) * 0.9:
            print(f"  >> SUGGESTIVE: aligned ratio is lower than random, but "
                  f"both deviate from 1.0.", flush=True)
        else:
            print(f"  >> NO CAUSAL SIGNAL: both ratios are similar.", flush=True)

    # --- Save ---
    result = {
        "meta": {
            "ckpt": args.ckpt,
            "configs_dir": args.configs_dir,
            "eval_games": args.eval_games,
            "seeds": args.seeds,
            "structural": args.structural,
            "n_quadrants": args.n_quadrants if args.structural else None,
            "device": device,
        },
        "per_seed": per_seed,
        "summary": summary,
        "base_rate_diagnostic": base_rates,
        "per_restriction_diagnostic": per_restriction_results,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results to {args.output}", flush=True)
    else:
        print("\n(Use --output to save full results to JSON)", flush=True)


if __name__ == "__main__":
    main()
