"""
Nonlinear probe vs logit direction comparison (B3b).

Two analyses that go beyond the linear gradient comparison in B3:

A) Actual vs predicted logit changes: Apply the probe intervention and measure
   the ACTUAL logit change for each affected move. Compare to the linear
   prediction (gradient dot perturbation). If actual >> predicted, the probe
   acts through nonlinear pathways the gradient doesn't capture.

B) Optimized directions: For each affected move, find the direction in 512-d
   that MAXIMALLY changes that logit (via gradient ascent on direction, with
   fixed perturbation norm). Compare this optimal direction to the probe
   direction. This accounts for nonlinearities that the raw gradient misses.

Usage:
  python probe_vs_logit_nonlinear.py \
      --probe-dir ../experiments/.../probe_checkpoints \
      --n-games 200 --output-dir ../experiments/probe_vs_logit_nonlinear
"""

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mingpt.model import GPT, GPTConfig
from multi_intervention import (
    board_val_to_probe_class,
    compute_prefix_activations,
    select_flip_noninteracting,
    select_add_noninteracting,
    replay_to_position,
    compute_legal_moves,
    compute_counterfactual_legal,
    apply_intervention,
    stoi_indices,
    CENTER_CELLS,
    STOI_SET,
)
from layer_propagation import (
    load_probes,
    build_direction_probe,
    compute_flip_dirs_from_direction_probe,
    compute_per_cell_scales_downstream,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAYER_INTERVENE = 5
LAYER_PROBE = 6
POS_RANGE = (10, 50)

# Map board cell index (0-63) to model logit index (1-60)
_cell_to_logit_idx = {}
for i, cell in enumerate(stoi_indices):
    _cell_to_logit_idx[cell] = i + 1


def cosine_sim(a, b):
    na, nb = a.norm(), b.norm()
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return (a @ b / (na * nb)).item()


def forward_from_layer(model, x, layer_start):
    """Forward pass from layer_start to logits. x: (1, seq, 512)."""
    for block in model.blocks[layer_start:]:
        x = block(x)
    x = model.ln_f(x)
    return model.head(x)


def compute_actual_logit_changes(model, prefix_acts, pos, layer_intervene,
                                  flip_dirs, scale, target_moves):
    """Apply probe intervention and measure actual logit changes.

    Returns dict mapping move -> actual logit change (float).
    Also returns the perturbation vector applied to h.
    """
    # Clean forward pass
    with torch.no_grad():
        x_clean = prefix_acts.detach().clone()
        logits_clean = forward_from_layer(model, x_clean, layer_intervene)

    # Intervened forward pass
    with torch.no_grad():
        x_intv = prefix_acts.detach().clone()
        flip_dir = flip_dirs[0]
        d_hat = flip_dir / flip_dir.norm()
        coeff = x_intv[0, pos] @ d_hat
        perturbation = -scale * coeff * d_hat
        x_intv[0, pos] = x_intv[0, pos] + perturbation
        logits_intv = forward_from_layer(model, x_intv, layer_intervene)

    changes = {}
    for move in target_moves:
        if move not in _cell_to_logit_idx:
            continue
        lidx = _cell_to_logit_idx[move]
        changes[move] = (logits_intv[0, pos, lidx] - logits_clean[0, pos, lidx]).item()

    return changes, perturbation.detach()


def compute_gradient_predicted_changes(model, prefix_acts, pos, layer_intervene,
                                        perturbation, target_moves):
    """Compute linear prediction: grad(logit_m) · perturbation for each move.

    Returns dict mapping move -> predicted logit change (float).
    Also returns the gradient vectors.
    """
    valid = [(m, _cell_to_logit_idx[m]) for m in target_moves if m in _cell_to_logit_idx]
    if not valid:
        return {}, {}

    x = prefix_acts.detach().clone()
    h = x[0, pos].detach().clone().requires_grad_(True)
    seq = x[0]
    x_mod = torch.cat([seq[:pos], h.unsqueeze(0), seq[pos+1:]], dim=0).unsqueeze(0)

    for block in model.blocks[layer_intervene:]:
        x_mod = block(x_mod)
    x_mod = model.ln_f(x_mod)
    logits = model.head(x_mod)

    predictions = {}
    grads = {}
    for i, (move, logit_idx) in enumerate(valid):
        retain = (i < len(valid) - 1)
        target_logit = logits[0, pos, logit_idx]
        target_logit.backward(retain_graph=retain)
        grad = h.grad.detach().clone()
        grads[move] = grad
        predictions[move] = (grad @ perturbation).item()
        h.grad.zero_()

    return predictions, grads


def optimize_direction(model, prefix_acts, pos, layer_intervene,
                       logit_idx, perturb_norm, n_steps=100, lr=0.05,
                       maximize=True):
    """Find the unit direction that maximally changes logit[logit_idx].

    Optimizes delta (unit norm) such that:
      h_perturbed = h + perturb_norm * delta
      logit_change = logits(h_perturbed)[logit_idx] - logits(h)[logit_idx]
    is maximized (or minimized if maximize=False).

    Uses projected gradient ascent (project back to unit sphere each step).
    Returns the optimal direction (512,) and the achieved logit change.
    """
    device = prefix_acts.device
    x_base = prefix_acts.detach().clone()
    h_clean = x_base[0, pos].detach().clone()

    # Clean logit
    with torch.no_grad():
        logits_clean = forward_from_layer(model, x_base, layer_intervene)
        logit_clean = logits_clean[0, pos, logit_idx].item()

    # Initialize delta as unit random vector
    delta = torch.randn(512, device=device)
    delta = delta / delta.norm()
    delta = delta.requires_grad_(True)

    optimizer = torch.optim.Adam([delta], lr=lr)

    best_change = 0.0
    best_delta = delta.detach().clone()

    for step in range(n_steps):
        optimizer.zero_grad()

        # Build perturbed activation
        h_perturbed = h_clean + perturb_norm * delta
        seq = x_base[0].detach().clone()
        x_mod = torch.cat([seq[:pos], h_perturbed.unsqueeze(0), seq[pos+1:]], dim=0).unsqueeze(0)

        logits = forward_from_layer(model, x_mod, layer_intervene)
        logit_new = logits[0, pos, logit_idx]

        if maximize:
            loss = -(logit_new - logit_clean)
        else:
            loss = (logit_new - logit_clean)

        loss.backward()
        optimizer.step()

        # Project back to unit sphere
        with torch.no_grad():
            delta.div_(delta.norm().clamp(min=1e-8))

        change = (logit_new.item() - logit_clean)
        if maximize and change > best_change:
            best_change = change
            best_delta = delta.detach().clone()
        elif not maximize and change < best_change:
            best_change = change
            best_delta = delta.detach().clone()

    return best_delta, best_change


def run_experiment(model, probes, direction_probe, board_seqs_int,
                   board_seqs_string, n_games=200, seed=42, device="cuda",
                   opt_steps=100):
    rng = random.Random(seed)
    samples = []

    for gi in tqdm(range(n_games), desc="Games"):
        game_str = board_seqs_string[gi]
        game_int = board_seqs_int[gi]
        pos = rng.randint(*POS_RANGE)

        board_state, color = replay_to_position(game_str, pos)
        is_even = (pos % 2 == 0)
        parity = "even" if is_even else "odd"
        mode = 0 if is_even else 1

        input_tokens = game_int[:pos + 1].unsqueeze(0).to(device)

        with torch.no_grad():
            prefix_acts = compute_prefix_activations(
                model, input_tokens, LAYER_INTERVENE)

        original_legal = compute_legal_moves(board_state, color)

        for intervention_type in ["flip", "add"]:
            if intervention_type == "flip":
                mods = select_flip_noninteracting(board_state, 1, rng)
            else:
                mods = select_add_noninteracting(board_state, 1, rng)

            if mods is None:
                continue

            cf_mods = [(r, c, tgt) for (r, c, _, tgt) in mods]
            cf_legal = compute_counterfactual_legal(board_state, cf_mods, color)

            newly_legal = cf_legal - original_legal
            newly_illegal = original_legal - cf_legal
            if len(newly_legal) == 0 and len(newly_illegal) == 0:
                continue

            # Compute probe direction and scale
            flip_dirs = compute_flip_dirs_from_direction_probe(
                direction_probe, mods, mode)
            probe_dir = flip_dirs[0]
            probe_dir_hat = probe_dir / probe_dir.norm()

            target_probe = probes[(LAYER_PROBE, parity)]
            scales = compute_per_cell_scales_downstream(
                model, prefix_acts, direction_probe, target_probe,
                mods, pos, LAYER_INTERVENE, LAYER_PROBE, mode)
            scale = scales[0]

            all_affected = list(newly_legal | newly_illegal)

            # === Analysis A: Actual vs predicted logit changes ===
            actual_changes, perturbation = compute_actual_logit_changes(
                model, prefix_acts, pos, LAYER_INTERVENE,
                flip_dirs, scale, all_affected)

            predicted_changes, grads = compute_gradient_predicted_changes(
                model, prefix_acts, pos, LAYER_INTERVENE,
                perturbation, all_affected)

            perturb_norm = perturbation.norm().item()

            move_results = []
            for move in all_affected:
                if move not in actual_changes or move not in predicted_changes:
                    continue

                legality = "newly_legal" if move in newly_legal else "newly_illegal"
                actual = actual_changes[move]
                predicted = predicted_changes[move]
                grad = grads[move]
                cos_probe_grad = cosine_sim(probe_dir, grad)

                # === Analysis B: Optimized direction ===
                lidx = _cell_to_logit_idx[move]
                # Optimize to maximize logit change (same sign as actual)
                do_maximize = (actual >= 0)
                opt_dir, opt_change = optimize_direction(
                    model, prefix_acts, pos, LAYER_INTERVENE,
                    lidx, perturb_norm, n_steps=opt_steps,
                    maximize=do_maximize)

                cos_probe_opt = cosine_sim(probe_dir, opt_dir)
                cos_grad_opt = cosine_sim(grad, opt_dir)

                move_results.append({
                    "move": int(move),
                    "legality": legality,
                    "actual_logit_change": actual,
                    "predicted_logit_change": predicted,
                    "nonlinearity_ratio": actual / predicted if abs(predicted) > 1e-6 else float("nan"),
                    "cos_probe_grad": cos_probe_grad,
                    "cos_probe_optimized": cos_probe_opt,
                    "cos_grad_optimized": cos_grad_opt,
                    "opt_logit_change": opt_change,
                    "grad_norm": grad.norm().item(),
                })

            r, c, orig_val, target_val = mods[0]
            sample = {
                "game_idx": gi,
                "pos": pos,
                "color": int(color),
                "intervention_type": intervention_type,
                "cell": [int(r), int(c)],
                "orig_val": int(orig_val),
                "target_val": int(target_val),
                "scale": float(scale),
                "perturb_norm": perturb_norm,
                "n_newly_legal": len(newly_legal),
                "n_newly_illegal": len(newly_illegal),
                "moves": move_results,
            }
            samples.append(sample)

    return samples


def plot_results(samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Collect per-move data
    actual_all, predicted_all = [], []
    cos_probe_grad_legal, cos_probe_opt_legal = [], []
    cos_probe_grad_illegal, cos_probe_opt_illegal = [], []
    opt_changes, actual_from_probe = [], []
    nonlin_ratios = []

    for s in samples:
        for m in s["moves"]:
            actual_all.append(m["actual_logit_change"])
            predicted_all.append(m["predicted_logit_change"])
            if not np.isnan(m["nonlinearity_ratio"]):
                nonlin_ratios.append(m["nonlinearity_ratio"])
            if m["legality"] == "newly_legal":
                cos_probe_grad_legal.append(m["cos_probe_grad"])
                cos_probe_opt_legal.append(m["cos_probe_optimized"])
            else:
                cos_probe_grad_illegal.append(m["cos_probe_grad"])
                cos_probe_opt_illegal.append(m["cos_probe_optimized"])
            opt_changes.append(m["opt_logit_change"])
            actual_from_probe.append(m["actual_logit_change"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Actual vs predicted logit change (scatter)
    ax = axes[0, 0]
    ax.scatter(predicted_all, actual_all, alpha=0.4, s=15)
    lims = [min(min(predicted_all), min(actual_all)),
            max(max(predicted_all), max(actual_all))]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x (linear)")
    ax.set_xlabel("Predicted logit change (gradient · perturbation)")
    ax.set_ylabel("Actual logit change")
    ax.set_title("A: Actual vs predicted logit change")
    ax.legend(fontsize=9)

    # Panel 2: Nonlinearity ratio histogram
    ax = axes[0, 1]
    clipped = [max(min(r, 10), -10) for r in nonlin_ratios]
    ax.hist(clipped, bins=50, alpha=0.7, color="tab:purple", edgecolor="black")
    ax.axvline(1.0, color="red", linestyle="--", label="Linear (ratio=1)")
    ax.set_xlabel("Nonlinearity ratio (actual / predicted)")
    ax.set_ylabel("Count")
    ax.set_title(f"A: Nonlinearity ratio (median={np.median(nonlin_ratios):.2f})")
    ax.legend(fontsize=9)

    # Panel 3: Cosine sim comparison — gradient vs optimized (newly legal)
    ax = axes[0, 2]
    random_baseline = np.sqrt(2 / np.pi) / np.sqrt(512)  # E[|cos|] for random
    if cos_probe_grad_legal and cos_probe_opt_legal:
        ax.scatter(cos_probe_grad_legal, cos_probe_opt_legal,
                   alpha=0.4, s=15, color="tab:green", label="Newly legal")
    if cos_probe_grad_illegal and cos_probe_opt_illegal:
        ax.scatter(cos_probe_grad_illegal, cos_probe_opt_illegal,
                   alpha=0.4, s=15, color="tab:red", label="Newly illegal")
    ax.axhline(random_baseline, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(random_baseline, color="gray", linestyle=":", alpha=0.5)
    ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3)
    ax.set_xlabel("cos(probe, gradient)")
    ax.set_ylabel("cos(probe, optimized)")
    ax.set_title("B: Probe alignment — gradient vs optimized")
    ax.legend(fontsize=9)

    # Panel 4: Optimized logit change vs probe logit change
    ax = axes[1, 0]
    ax.scatter(actual_from_probe, opt_changes, alpha=0.4, s=15, color="tab:orange")
    ax.plot([min(actual_from_probe + opt_changes),
             max(actual_from_probe + opt_changes)],
            [min(actual_from_probe + opt_changes),
             max(actual_from_probe + opt_changes)],
            "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("Logit change from probe direction")
    ax.set_ylabel("Logit change from optimized direction")
    ax.set_title("B: Optimized vs probe logit change (same norm)")
    ax.legend(fontsize=9)

    # Panel 5: cos(probe, optimized) histogram by legality
    ax = axes[1, 1]
    bins = np.linspace(-1, 1, 50)
    if cos_probe_opt_legal:
        ax.hist(cos_probe_opt_legal, bins=bins, alpha=0.6,
                label=f"Newly legal (n={len(cos_probe_opt_legal)})", color="tab:green")
    if cos_probe_opt_illegal:
        ax.hist(cos_probe_opt_illegal, bins=bins, alpha=0.6,
                label=f"Newly illegal (n={len(cos_probe_opt_illegal)})", color="tab:red")
    ax.axvline(random_baseline, color="gray", linestyle=":", alpha=0.5, label=f"Random ±{random_baseline:.3f}")
    ax.axvline(-random_baseline, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("cos(probe direction, optimized direction)")
    ax.set_ylabel("Count")
    ax.set_title("B: Probe vs optimized direction alignment")
    ax.legend(fontsize=9)

    # Panel 6: Summary stats text
    ax = axes[1, 2]
    ax.axis("off")
    stats = []
    stats.append(f"N samples: {len(samples)}")
    n_moves = sum(len(s['moves']) for s in samples)
    stats.append(f"N move comparisons: {n_moves}")
    stats.append("")
    stats.append("=== A: Nonlinearity ===")
    stats.append(f"Median ratio (actual/predicted): {np.median(nonlin_ratios):.3f}")
    stats.append(f"Mean ratio: {np.mean(nonlin_ratios):.3f}")
    stats.append(f"Frac |ratio| > 2: {np.mean([abs(r) > 2 for r in nonlin_ratios]):.2%}")
    stats.append("")
    stats.append("=== B: Optimized alignment ===")
    if cos_probe_opt_legal:
        stats.append(f"cos(probe, opt) newly legal:")
        stats.append(f"  mean |cos| = {np.mean(np.abs(cos_probe_opt_legal)):.4f}")
        stats.append(f"  (vs gradient: {np.mean(np.abs(cos_probe_grad_legal)):.4f})")
    if cos_probe_opt_illegal:
        stats.append(f"cos(probe, opt) newly illegal:")
        stats.append(f"  mean |cos| = {np.mean(np.abs(cos_probe_opt_illegal)):.4f}")
        stats.append(f"  (vs gradient: {np.mean(np.abs(cos_probe_grad_illegal)):.4f})")
    stats.append(f"\nRandom baseline: {random_baseline:.4f}")
    if opt_changes and actual_from_probe:
        stats.append(f"\nOpt/probe logit change ratio:")
        ratios = [o / a for o, a in zip(opt_changes, actual_from_probe) if abs(a) > 0.01]
        if ratios:
            stats.append(f"  median: {np.median(ratios):.2f}x")
            stats.append(f"  mean: {np.mean(ratios):.2f}x")

    ax.text(0.05, 0.95, "\n".join(stats), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nonlinear_analysis.png"), dpi=150)
    plt.close()
    print(f"Saved plot to {output_dir}/nonlinear_analysis.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opt-steps", type=int, default=100,
                        help="Optimization steps per move for direction search")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(script_dir, "..", "ckpts", "gpt_synthetic.ckpt")
    ckpt = torch.load(ckpt_path, map_location=device)
    conf = GPTConfig(
        vocab_size=61, block_size=59,
        n_layer=8, n_head=8, n_embd=512
    )
    model = GPT(conf)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("Model loaded")

    # Load data
    data_path = os.path.join(script_dir, "board_seqs_int.pth")
    board_seqs_int = torch.load(data_path, map_location="cpu")
    str_path = os.path.join(script_dir, "board_seqs_string.pth")
    board_seqs_string = torch.load(str_path, map_location="cpu")
    print(f"Data loaded: {board_seqs_int.shape[0]} games")

    # Load probes
    probes = load_probes(args.probe_dir, device)
    print(f"Loaded probes for layers: {sorted(set(l for l, _ in probes.keys()))}")

    direction_probe = build_direction_probe(probes, layer=LAYER_PROBE, device=device)
    print(f"Direction probe shape: {direction_probe.shape}")

    # Run experiment
    samples = run_experiment(
        model, probes, direction_probe,
        board_seqs_int, board_seqs_string,
        n_games=args.n_games, seed=args.seed, device=device,
        opt_steps=args.opt_steps
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "raw_samples.json"), "w") as f:
        json.dump(samples, f, indent=2)

    # Aggregate stats
    all_moves = [m for s in samples for m in s["moves"]]
    legal_moves = [m for m in all_moves if m["legality"] == "newly_legal"]
    illegal_moves = [m for m in all_moves if m["legality"] == "newly_illegal"]
    nonlin = [m["nonlinearity_ratio"] for m in all_moves if not np.isnan(m["nonlinearity_ratio"])]

    results = {
        "n_samples": len(samples),
        "n_moves": len(all_moves),
        "n_newly_legal": len(legal_moves),
        "n_newly_illegal": len(illegal_moves),
        "nonlinearity_ratio_median": float(np.median(nonlin)) if nonlin else None,
        "nonlinearity_ratio_mean": float(np.mean(nonlin)) if nonlin else None,
        "cos_probe_grad_legal": float(np.mean(np.abs([m["cos_probe_grad"] for m in legal_moves]))) if legal_moves else None,
        "cos_probe_opt_legal": float(np.mean(np.abs([m["cos_probe_optimized"] for m in legal_moves]))) if legal_moves else None,
        "cos_probe_grad_illegal": float(np.mean(np.abs([m["cos_probe_grad"] for m in illegal_moves]))) if illegal_moves else None,
        "cos_probe_opt_illegal": float(np.mean(np.abs([m["cos_probe_optimized"] for m in illegal_moves]))) if illegal_moves else None,
        "cos_grad_opt_legal": float(np.mean(np.abs([m["cos_grad_optimized"] for m in legal_moves]))) if legal_moves else None,
        "cos_grad_opt_illegal": float(np.mean(np.abs([m["cos_grad_optimized"] for m in illegal_moves]))) if illegal_moves else None,
        "opt_steps": args.opt_steps,
    }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_results(samples, args.output_dir)
    print(f"\nResults saved to {args.output_dir}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Samples: {len(samples)}, Moves: {len(all_moves)}")
    if nonlin:
        print(f"Nonlinearity ratio: median={np.median(nonlin):.3f}, mean={np.mean(nonlin):.3f}")
    if legal_moves:
        print(f"Newly legal — cos(probe,grad): {results['cos_probe_grad_legal']:.4f}, "
              f"cos(probe,opt): {results['cos_probe_opt_legal']:.4f}")
    if illegal_moves:
        print(f"Newly illegal — cos(probe,grad): {results['cos_probe_grad_illegal']:.4f}, "
              f"cos(probe,opt): {results['cos_probe_opt_illegal']:.4f}")


if __name__ == "__main__":
    main()
