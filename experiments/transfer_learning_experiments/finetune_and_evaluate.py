"""
Fine-tune OthelloGPT on restricted games with periodic legal-accuracy evaluation.

Measures three metrics (matching John's protocol):
  1. top1_legal  — fraction of positions where the model's top prediction is legal
  2. top1_prob   — average probability assigned to the most probable legal move
  3. legal_mass  — average total probability on all legal moves

Supports multiple runs with different seeds for confidence intervals.

Usage:
    # Fine-tune pre-trained model on aligned-restriction games
    python finetune_and_evaluate.py \\
        --games-dir ../../data/restricted_aligned \\
        --config restrictions_aligned.json \\
        --label aligned_ft --mode ft

    # Train from random init (baseline)
    python finetune_and_evaluate.py \\
        --games-dir ../../data/restricted_aligned \\
        --config restrictions_aligned.json \\
        --label aligned_rnd --mode rnd

    # Multiple runs for confidence intervals
    python finetune_and_evaluate.py \\
        --games-dir ../../data/restricted_aligned \\
        --config restrictions_aligned.json \\
        --label aligned_ft --mode ft --runs 3
"""

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from mingpt.model import GPT, GPTConfig
from mingpt.dataset import CharDataset
from mingpt.utils import set_seed
from data.othello import OthelloBoardState, get_ood_game

from restriction_utils import (
    VALID_POSITIONS, CENTER_SQUARES,
    evaluate_restriction, get_flipped_squares, _get_forbidden_positions,
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, eval_games, restrictions, stoi, itos, device,
             max_games=200, max_seq_len=59):
    """Evaluate legal-move prediction under restriction rules.

    Replays standard Othello games, applies restrictions at each position,
    and checks the model's predictions against the restricted legal set.

    Returns dict with:
        top1_legal            : argmax is in restricted legal set (all positions)
        top1_legal_when_fires : same, restricted to positions where ≥1 restriction fires
        violation_rate        : argmax is in the forbidden set (all positions)
        violation_rate_when_fires : same, among fired positions
        fire_rate             : fraction of positions where ≥1 restriction fires
        top1_prob, legal_mass : average probability metrics (all positions)
        n_positions, n_fires  : raw counts for downstream diagnostics
    """
    model.eval()

    total_top1_legal = 0
    total_top1_prob = 0.0
    total_legal_mass = 0.0
    total_positions = 0
    total_fires = 0                   # positions where ≥1 restriction fires
    total_top1_legal_fires = 0        # top1_legal restricted to fired positions
    total_violations = 0              # argmax ∈ forbidden (all positions)
    total_violations_fires = 0        # same, restricted to fired positions

    with torch.no_grad():
        for game in eval_games[:max_games]:
            if len(game) < 2:
                continue

            # Encode full game for model (all positions at once)
            encoded = [stoi[m] for m in game]
            if len(encoded) > max_seq_len + 1:
                encoded = encoded[: max_seq_len + 1]
            x = torch.tensor(encoded[:-1], dtype=torch.long)[None].to(device)
            logits, _ = model(x)          # [1, L-1, vocab]
            probs = F.softmax(logits[0], dim=-1)  # [L-1, vocab]

            # Replay game to get board states and restrictions
            board = OthelloBoardState()
            last_flipped = set()
            last_move = -1

            for pos in range(min(len(game) - 1, max_seq_len)):
                move = game[pos]

                # Execute move and track flips
                state_before = board.state.copy()
                board.umpire(move)
                flipped = get_flipped_squares(state_before, board.state, move)

                # Standard legal moves for the next player
                standard_legal = board.get_valid_moves()
                if not standard_legal:
                    last_move = move
                    last_flipped = flipped
                    continue

                # Apply restrictions (condition on the move just played)
                forbidden = set()
                for r in restrictions:
                    if evaluate_restriction(r, board, move, flipped):
                        forbidden.update(_get_forbidden_positions(r))
                restricted_legal = [m for m in standard_legal if m not in forbidden]
                if not restricted_legal:
                    restricted_legal = standard_legal

                # Model predictions at this position
                pos_probs = probs[pos]  # [vocab]

                # 1) top-1 legal accuracy
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

                # 2) probability of best legal move & 3) total legal mass
                legal_token_indices = []
                for bp in restricted_legal:
                    if bp in stoi:
                        legal_token_indices.append(stoi[bp])
                if legal_token_indices:
                    legal_probs = pos_probs[legal_token_indices]
                    total_top1_prob += legal_probs.max().item()
                    total_legal_mass += legal_probs.sum().item()

                total_positions += 1
                last_move = move
                last_flipped = flipped

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_games_from_dir(games_dir, max_files=None):
    """Load games from all .pickle files in a directory."""
    games = []
    files = sorted(f for f in os.listdir(games_dir) if f.endswith(".pickle"))
    if max_files:
        files = files[:max_files]
    for f in files:
        path = os.path.join(games_dir, f)
        with open(path, "rb") as fh:
            batch = pickle.load(fh)
        games.extend(batch)
        print(f"  Loaded {len(batch)} games from {f}")
    return games


# ---------------------------------------------------------------------------
# Training loop for a single run
# ---------------------------------------------------------------------------

def run_single(
    run_idx, args, train_games, eval_games, restrictions, pretrained_sd, device
):
    """Execute one full training run. Returns the learning curve."""
    seed = args.seed + run_idx
    set_seed(seed)
    print(f"\n{'='*60}", flush=True)
    print(f"  Run {run_idx + 1}/{args.runs}  (seed={seed})", flush=True)
    print(f"{'='*60}", flush=True)

    # --- Dataset ---
    dataset = CharDataset(train_games)
    stoi = dataset.stoi
    itos = dataset.itos

    # --- Model ---
    mconf = GPTConfig(
        dataset.vocab_size, dataset.block_size,
        n_layer=8, n_head=8, n_embd=512,
    )
    model = GPT(mconf)
    if args.mode == "ft":
        model.load_state_dict(pretrained_sd)
    model = model.to(device)

    # --- Optimizer ---
    # Scratch/rnd mode may use a separate LR.
    is_scratch = args.mode in ("scratch", "rnd")
    lr = args.lr_scratch if (is_scratch and args.lr_scratch is not None) else args.lr
    optimizer = model.configure_optimizers(
        argparse.Namespace(
            learning_rate=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
    )

    # --- DataLoader ---
    # pin_memory only helps (and is only supported) on CUDA. MPS uses unified
    # memory so pinning is meaningless; CPU doesn't benefit either.
    pin_memory = str(device).startswith("cuda")
    loader = DataLoader(
        dataset, shuffle=True, pin_memory=pin_memory,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # --- Initial evaluation ---
    metrics = evaluate(
        model, eval_games, restrictions, stoi, itos, device,
        max_games=args.eval_games,
    )
    curve = [{
        "step": 0, "train_loss": None,
        **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
    }]
    vf0 = metrics.get('violation_rate_when_fires')
    vf0_str = f"  viol_fires={vf0:.4f}" if vf0 is not None else ""
    tf0 = metrics.get('top1_legal_when_fires')
    tf0_str = f"  top1_fires={tf0:.4f}" if tf0 is not None else ""
    fr0 = metrics.get('fire_rate')
    fr0_str = f"  fire_rate={fr0:.4f}" if fr0 is not None else ""
    print(f"  Step 0: top1_legal={metrics['top1_legal']:.4f}  "
          f"legal_mass={metrics['legal_mass']:.4f}"
          f"{tf0_str}{vf0_str}{fr0_str}", flush=True)

    # --- Training ---
    global_step = 0
    recent_losses = []
    t0 = time.time()

    pbar = tqdm(total=args.max_steps, desc=f"  Run {run_idx+1}/{args.runs}",
                unit="step", dynamic_ncols=True)
    for epoch in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss.mean()

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            recent_losses.append(loss.item())
            pbar.update(1)

            # --- Periodic eval ---
            early_phase = (args.eval_every_early is not None
                           and global_step <= args.eval_early_until)
            eval_interval = args.eval_every_early if early_phase else args.eval_every
            if global_step % eval_interval == 0:
                avg_loss = np.mean(recent_losses[-eval_interval:])
                metrics = evaluate(
                    model, eval_games, restrictions, stoi, itos, device,
                    max_games=args.eval_games,
                )
                entry = {
                    "step": global_step,
                    "train_loss": round(float(avg_loss), 6),
                    **{k: round(v, 6) if isinstance(v, float) else v
                       for k, v in metrics.items()},
                }
                curve.append(entry)

                vf = metrics.get('violation_rate_when_fires')
                vf_str = f"{vf:.3f}" if vf is not None else "n/a"
                tf = metrics.get('top1_legal_when_fires')
                tf_str = f"{tf:.3f}" if tf is not None else "n/a"
                pbar.set_postfix_str(
                    f"loss={avg_loss:.3f} top1={metrics['top1_legal']:.3f} "
                    f"top1_fires={tf_str} viol_fires={vf_str}")

            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break
    pbar.close()

    # --- Final eval ---
    if global_step % args.eval_every != 0:
        avg_loss = np.mean(recent_losses[-args.eval_every:])
        metrics = evaluate(
            model, eval_games, restrictions, stoi, itos, device,
            max_games=args.eval_games,
        )
        curve.append({
            "step": global_step,
            "train_loss": round(float(avg_loss), 6),
            **{k: round(v, 6) if isinstance(v, float) else v
               for k, v in metrics.items()},
        })

    # --- Optionally save checkpoint ---
    if args.save_ckpts:
        ckpt_dir = os.path.join(args.output_dir, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{args.label}_run{run_idx}.ckpt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint to {ckpt_path}")

    return curve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune OthelloGPT on restricted games with evaluation")
    parser.add_argument("--games-dir", type=str, required=True,
                        help="Directory with restricted game .pickle files")
    parser.add_argument("--config", type=str, required=True,
                        help="Restriction config JSON (for evaluation)")
    parser.add_argument("--label", type=str, required=True,
                        help="Label for this experiment (used in output filenames)")
    parser.add_argument("--condition", type=str, default=None,
                        choices=["B1", "B2", "B3", "C"],
                        help="2x2 arm identifier; embedded in output JSON and "
                             "used for structured filenames (optional)")
    parser.add_argument("--mode", choices=["ft", "rnd", "scratch"], default="ft",
                        help="ft = fine-tune pre-trained; scratch/rnd = train "
                             "from random init")
    parser.add_argument("--lr-scratch", type=float, default=None,
                        help="Override learning rate for scratch/rnd mode "
                             "(default: same as --lr)")
    parser.add_argument("--ckpt", type=str, default="../../ckpts/gpt_synthetic.ckpt",
                        help="Pre-trained checkpoint path (used if --mode ft)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for output JSON results")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of independent runs (different seeds)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Stop training after this many gradient steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size (default: 16, matching protocol)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Evaluate every N training steps")
    parser.add_argument("--eval-every-early", type=int, default=None,
                        help="Eval every N steps during early training "
                             "(before --eval-early-until). Default: disabled")
    parser.add_argument("--eval-early-until", type=int, default=200,
                        help="Step at which to switch from --eval-every-early "
                             "back to --eval-every. Default: 200")
    parser.add_argument("--eval-games", type=int, default=200,
                        help="Number of standard-Othello games for evaluation")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-ckpts", action="store_true",
                        help="Save model checkpoint after each run")
    args = parser.parse_args()

    # --- Device ---
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load restriction config ---
    with open(args.config) as f:
        config = json.load(f)
    restrictions = config["restrictions"]
    print(f"Loaded {len(restrictions)} restrictions from {args.config}", flush=True)

    # --- Load training games ---
    print(f"\nLoading training games from {args.games_dir}...", flush=True)
    train_games = load_games_from_dir(args.games_dir)
    train_games = [g for g in train_games if len(g) >= 5]
    print(f"Training games: {len(train_games)}", flush=True)

    # --- Generate eval games (standard Othello, fixed across runs) ---
    print(f"\nGenerating {args.eval_games} standard Othello games for evaluation...",
          flush=True)
    set_seed(0)  # fixed seed for eval games
    eval_games = [get_ood_game(i) for i in range(args.eval_games)]
    print(f"Eval games: {len(eval_games)}, "
          f"avg length: {np.mean([len(g) for g in eval_games]):.1f}", flush=True)

    # --- Load pre-trained weights (only for ft; scratch/rnd skip) ---
    pretrained_sd = None
    if args.mode == "ft":
        ckpt_path = args.ckpt
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)
        print(f"\nLoading pre-trained checkpoint: {ckpt_path}", flush=True)
        pretrained_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # --- Run experiments ---
    all_curves = {}
    for run_idx in range(args.runs):
        curve = run_single(
            run_idx, args, train_games, eval_games,
            restrictions, pretrained_sd, device,
        )
        all_curves[f"run_{run_idx}"] = curve

    # --- Save results ---
    # Normalize mode for downstream aggregation: both 'rnd' and 'scratch'
    # represent training from random init.
    mode_canonical = "scratch" if args.mode in ("scratch", "rnd") else "ft"
    result = {
        "label": args.label,
        "condition": args.condition,
        "mode": args.mode,
        "mode_canonical": mode_canonical,
        "config_file": args.config,
        "games_dir": args.games_dir,
        "num_restrictions": len(restrictions),
        "num_train_games": len(train_games),
        "num_eval_games": args.eval_games,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_scratch": args.lr_scratch,
        "max_steps": args.max_steps,
        "eval_every": args.eval_every,
        "runs": args.runs,
        "seed": args.seed,
        "curves": all_curves,
    }
    if args.condition is not None:
        fname = f"curves_{args.condition}_{mode_canonical}_{args.label}.json"
    else:
        fname = f"{args.label}.json"
    out_path = os.path.join(args.output_dir, fname)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_path}", flush=True)

    # --- Print summary ---
    print(f"\n{'='*60}", flush=True)
    print(f"Summary: {args.label} ({args.mode})", flush=True)
    print(f"{'='*60}", flush=True)
    for run_key, curve in all_curves.items():
        first = curve[0]
        last = curve[-1]
        print(f"  {run_key}: step 0 top1_legal={first['top1_legal']:.4f} "
              f"-> step {last['step']} top1_legal={last['top1_legal']:.4f}  "
              f"legal_mass: {first['legal_mass']:.4f} -> {last['legal_mass']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
