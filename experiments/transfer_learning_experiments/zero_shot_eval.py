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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from mingpt.model import GPT, GPTConfig
from mingpt.dataset import CharDataset
from mingpt.utils import set_seed
from data.othello import get_ood_game

from finetune_and_evaluate import evaluate


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

    # --- Run evaluation across seeds ---
    per_seed = {}
    for seed_idx in range(args.seeds):
        seed = seed_idx * 1000  # spread seeds
        set_seed(seed)
        eval_games = [get_ood_game(i) for i in range(args.eval_games)]

        seed_results = {}
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

        per_seed[f"seed_{seed_idx}"] = seed_results
        # Progress
        viol_str = "  ".join(
            f"{arm}={seed_results[arm].get('violation_rate_when_fires', 'n/a'):.4f}"
            if isinstance(seed_results[arm].get('violation_rate_when_fires'), float)
            else f"{arm}=n/a"
            for arm in configs
        )
        print(f"  Seed {seed_idx}/{args.seeds}: viol_fires: {viol_str}",
              flush=True)

    # --- Aggregate: mean ± SEM ---
    summary = {}
    for arm in configs:
        summary[arm] = {}
        for metric in METRICS:
            values = []
            for seed_key in per_seed:
                v = per_seed[seed_key][arm].get(metric)
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
    header = f"{'Arm':<4}"
    for m in key_metrics:
        short = m.replace("_when_fires", "_wf").replace("violation_rate", "viol")
        header += f"  {short:<18}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for arm in configs:
        row = f"{arm:<4}"
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

    # --- Save ---
    result = {
        "meta": {
            "ckpt": args.ckpt,
            "configs_dir": args.configs_dir,
            "eval_games": args.eval_games,
            "seeds": args.seeds,
            "device": device,
        },
        "per_seed": per_seed,
        "summary": summary,
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
