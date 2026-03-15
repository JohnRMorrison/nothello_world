"""
Layer propagation experiment: how do causal interventions propagate across layers?

Intervenes at layer 5 (resid_post block 4) using layer-6 probe directions.
For each calibration target (layer 6, layer 7), finds the minimal per-cell
scale that achieves 100% probe accuracy at that target layer, then measures
probe accuracy and crosstalk at ALL measurement layers (5, 6, 7).

Key questions:
  - Does calibrating to flip layer 6 also flip layer 7?
  - Does calibrating to flip layer 7 require more intervention?
  - How much crosstalk increases with larger interventions?
  - Are representations of different cells independent or coupled?

Usage:
  python layer_propagation.py \
      --probe-dir ../experiments/.../probe_checkpoints \
      --n-games 200 --output-dir ../experiments/layer_propagation
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

# Import shared utilities from multi_intervention
from multi_intervention import (
    board_val_to_probe_class,
    compute_flip_dirs,
    apply_intervention,
    compute_prefix_activations,
    select_flip_noninteracting,
    replay_to_position,
    compute_legal_moves,
    compute_counterfactual_legal,
    measure_logit_metrics,
    stoi_indices,
    CENTER_CELLS,
    STOI_SET,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAYER_INTERVENE = 5       # Intervene on resid_post of block 4
MEASUREMENT_LAYERS = [5, 6, 7]  # Measure at resid_post of blocks 5, 6, 7
POS_RANGE = (10, 50)
OPTIONS = 3  # empty, white, black


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------
def load_probes(probe_dir, device="cuda"):
    """Load per-layer heuristic probes as nn.Linear objects.

    Returns dict mapping (layer, parity) -> nn.Linear on device,
    where parity is 'even' or 'odd'.
    """
    probes = {}
    for layer in range(9):  # layers 0-8
        path = os.path.join(probe_dir, f"othello_layer{layer}.pt")
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location="cpu")
        for parity in ("even", "odd"):
            probe = nn.Linear(512, 64 * OPTIONS)
            probe.load_state_dict(ckpt[parity])
            probe.to(device)
            probe.eval()
            probes[(layer, parity)] = probe
    return probes


def convert_probe_to_direction_tensor(probe_linear, device="cuda"):
    """Convert nn.Linear(512, 192) to tensor (512, 8, 8, 3) for direction computation.

    Ignores bias — directions are defined by weight vectors only.
    """
    W = probe_linear.weight.data  # (192, 512)
    # Output is organized as 64 cells × 3 classes
    W_reshaped = W.view(64, OPTIONS, 512)  # (64, 3, 512)
    W_board = W_reshaped.permute(2, 0, 1).view(512, 8, 8, OPTIONS)  # (512, 8, 8, 3)
    return W_board.to(device)


def build_direction_probe(probes, layer=6, device="cuda"):
    """Build direction tensor (2, 512, 8, 8, 3) from even/odd probes at a layer.

    Index 0 = even, index 1 = odd.
    """
    tensors = []
    for parity in ("even", "odd"):
        t = convert_probe_to_direction_tensor(probes[(layer, parity)], device)
        tensors.append(t)
    return torch.stack(tensors, dim=0)  # (2, 512, 8, 8, 3)


# ---------------------------------------------------------------------------
# Probe measurement (with bias)
# ---------------------------------------------------------------------------
def measure_probe_acc(h, probe_linear, modifications):
    """Fraction of modified cells where probe argmax matches target.

    h: (512,) activation vector
    probe_linear: nn.Linear(512, 192)
    """
    with torch.no_grad():
        logits = probe_linear(h)  # (192,)
        logits = logits.view(64, OPTIONS)  # (64, 3)

    correct = 0
    for (r, c, _, target_val) in modifications:
        cell = r * 8 + c
        target_class = board_val_to_probe_class(target_val)
        if logits[cell].argmax().item() == target_class:
            correct += 1
    return correct / len(modifications)


def measure_crosstalk(h_clean, h_intv, probe_linear, modifications):
    """Mean absolute probe logit change for non-modified cells.

    h_clean, h_intv: (512,) activation vectors
    """
    modified_cells = {r * 8 + c for (r, c, _, _) in modifications}

    with torch.no_grad():
        logits_clean = probe_linear(h_clean).view(64, OPTIONS)
        logits_intv = probe_linear(h_intv).view(64, OPTIONS)

    diffs = []
    for cell in range(64):
        if cell in modified_cells or cell in CENTER_CELLS:
            continue
        diff = (logits_intv[cell] - logits_clean[cell]).abs().mean().item()
        diffs.append(diff)
    return np.mean(diffs) if diffs else 0.0


# ---------------------------------------------------------------------------
# Downstream per-cell scale calibration
# ---------------------------------------------------------------------------
def compute_flip_dirs_from_direction_probe(direction_probe, modifications, mode):
    """Compute flip directions from direction probe tensor.

    direction_probe: (2, 512, 8, 8, 3) — [even, odd]
    mode: 0 for even positions, 1 for odd
    """
    flip_dirs = []
    for (r, c, orig_val, target_val) in modifications:
        current_class = board_val_to_probe_class(orig_val)
        target_class = board_val_to_probe_class(target_val)
        flip_dir = (direction_probe[mode, :, r, c, target_class]
                    - direction_probe[mode, :, r, c, current_class])
        flip_dirs.append(flip_dir)
    return flip_dirs


def compute_per_cell_scales_downstream(
    model, prefix_acts, direction_probe, target_probe,
    modifications, pos, layer_intervene, target_layer, mode
):
    """Binary search for minimal per-cell scale that flips target probe downstream.

    For each cell:
      1. Modify activation at layer_intervene
      2. Run forward through blocks layer_intervene..target_layer
      3. Check target_probe argmax == target_class

    direction_probe: (2, 512, 8, 8, 3) tensor for computing flip directions
    target_probe: nn.Linear(512, 192) — probe at target layer
    mode: 0 (even) or 1 (odd)
    """
    scales = []

    for (r, c, orig_val, target_val) in modifications:
        target_class = board_val_to_probe_class(target_val)
        current_class = board_val_to_probe_class(orig_val)

        # Flip direction from direction probe
        flip_dir = (direction_probe[mode, :, r, c, target_class]
                    - direction_probe[mode, :, r, c, current_class])
        d_hat = flip_dir / flip_dir.norm()
        h = prefix_acts[0, pos].detach()
        coeff = (h @ d_hat).item()

        def probe_pred_at_scale(s):
            h_mod = h - s * coeff * d_hat
            x = prefix_acts.clone()
            x[0, pos] = h_mod
            # Run through blocks layer_intervene..target_layer (inclusive)
            # Block i produces resid_post for layer i
            # target_layer probe expects resid_post of block target_layer
            for block in model.blocks[layer_intervene:target_layer + 1]:
                x = block(x)
            act = x[0, pos]
            with torch.no_grad():
                logits = target_probe(act).view(64, OPTIONS)
            cell = r * 8 + c
            return logits[cell].argmax().item()

        # Binary search in [0, 10]
        lo, hi = 0.0, 10.0
        if probe_pred_at_scale(hi) != target_class:
            scales.append(hi)  # can't flip — use max
            continue
        if probe_pred_at_scale(0.0) == target_class:
            scales.append(0.5)  # already correct
            continue

        for _ in range(20):
            mid = (lo + hi) / 2
            if probe_pred_at_scale(mid) == target_class:
                hi = mid
            else:
                lo = mid

        scales.append(min(hi * 1.1, 10.0))

    return scales


# ---------------------------------------------------------------------------
# Forward pass with multi-layer capture
# ---------------------------------------------------------------------------
def forward_multi_capture(model, prefix_acts, flip_dirs, per_cell_scales,
                          pos, layer_intervene, capture_layers):
    """Apply intervention and capture residuals at multiple downstream layers.

    Returns (logits_at_pos, resids_dict) where resids_dict maps layer -> (512,).
    """
    x = prefix_acts.clone()
    x = apply_intervention(x, flip_dirs, pos, scale=None,
                           per_cell_scales=per_cell_scales)

    resids = {}
    for i, block in enumerate(model.blocks[layer_intervene:], start=layer_intervene):
        x = block(x)
        if i in capture_layers:
            resids[i] = x[0, pos].detach().clone()

    x = model.ln_f(x)
    logits = model.head(x)
    return logits[0, pos], resids


def clean_forward_multi_capture(model, input_tokens, pos,
                                layer_intervene, capture_layers):
    """Clean forward pass (no intervention), capture at multiple layers.

    Returns (logits_at_pos, resids_dict, prefix_acts).
    """
    prefix_acts = compute_prefix_activations(model, input_tokens, layer_intervene)
    x = prefix_acts.clone()

    resids = {}
    for i, block in enumerate(model.blocks[layer_intervene:], start=layer_intervene):
        x = block(x)
        if i in capture_layers:
            resids[i] = x[0, pos].detach().clone()

    x = model.ln_f(x)
    logits = model.head(x)
    return logits[0, pos], resids, prefix_acts


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(model, probes, direction_probe, board_seqs_int,
                   board_seqs_string, cal_targets, n_games=200,
                   seed=42, device="cuda"):
    """Run layer propagation experiment.

    For each game:
      1. Sample position, select 1 cell to flip
      2. Clean forward, capture resids at measurement layers
      3. For each calibration target + no-intervention baseline:
         a. Find per-cell scale (binary search at target layer)
         b. Apply intervention, capture resids at all layers
         c. Measure probe accuracy and crosstalk at each layer
         d. Compute logit metrics

    Returns list of per-sample dicts.
    """
    rng = random.Random(seed)
    samples = []

    for gi in tqdm(range(n_games), desc="Games"):
        game_str = board_seqs_string[gi]
        game_int = board_seqs_int[gi]
        pos = rng.randint(*POS_RANGE)

        board_state, color = replay_to_position(game_str, pos)

        mods = select_flip_noninteracting(board_state, 1, rng)
        if mods is None:
            continue

        is_even = (pos % 2 == 0)
        parity = "even" if is_even else "odd"
        mode = 0 if is_even else 1

        input_tokens = game_int[:pos + 1].unsqueeze(0).to(device)

        with torch.no_grad():
            # Clean forward pass
            clean_logits, clean_resids, prefix_acts = \
                clean_forward_multi_capture(
                    model, input_tokens, pos,
                    LAYER_INTERVENE, MEASUREMENT_LAYERS
                )

            # Legal moves for logit metrics
            original_legal = compute_legal_moves(board_state, color)
            cf_mods = [(r, c, tgt) for (r, c, _, tgt) in mods]
            cf_legal = compute_counterfactual_legal(board_state, cf_mods, color)

            # Board state as flat list for reconstruction
            board_flat = board_state.flatten().tolist()

            # --- No-intervention baseline ---
            sample_base = {
                "game_idx": gi, "pos": pos, "color": int(color),
                "board_state": board_flat,
                "modifications": [(r, c, int(o), int(t)) for r, c, o, t in mods],
                "cal_target": "none", "scale": 0.0,
                "n_original_legal": len(original_legal),
                "n_cf_legal": len(cf_legal),
            }
            for layer in MEASUREMENT_LAYERS:
                probe = probes[(layer, parity)]
                sample_base[f"probe_acc_L{layer}"] = measure_probe_acc(
                    clean_resids[layer], probe, mods)
                sample_base[f"crosstalk_L{layer}"] = 0.0  # no intervention
            # Logit metrics (no change)
            logit_m = measure_logit_metrics(clean_logits, clean_logits,
                                            original_legal, cf_legal)
            sample_base.update(logit_m)
            samples.append(sample_base)

            # --- Each calibration target ---
            for cal_target in cal_targets:
                target_probe = probes[(cal_target, parity)]

                # Binary search for per-cell scale at target layer
                per_cell_scales = compute_per_cell_scales_downstream(
                    model, prefix_acts, direction_probe, target_probe,
                    mods, pos, LAYER_INTERVENE, cal_target, mode
                )

                # Apply intervention, capture at all layers
                flip_dirs = compute_flip_dirs_from_direction_probe(
                    direction_probe, mods, mode)
                intv_logits, intv_resids = forward_multi_capture(
                    model, prefix_acts, flip_dirs, per_cell_scales,
                    pos, LAYER_INTERVENE, MEASUREMENT_LAYERS
                )

                sample = {
                    "game_idx": gi, "pos": pos, "color": int(color),
                    "board_state": board_flat,
                    "modifications": [(r, c, int(o), int(t))
                                      for r, c, o, t in mods],
                    "cal_target": cal_target,
                    "scale": per_cell_scales[0],  # N=1
                    "n_original_legal": len(original_legal),
                    "n_cf_legal": len(cf_legal),
                }

                for layer in MEASUREMENT_LAYERS:
                    probe = probes[(layer, parity)]
                    sample[f"probe_acc_L{layer}"] = measure_probe_acc(
                        intv_resids[layer], probe, mods)
                    sample[f"crosstalk_L{layer}"] = measure_crosstalk(
                        clean_resids[layer], intv_resids[layer], probe, mods)

                logit_m = measure_logit_metrics(clean_logits, intv_logits,
                                                original_legal, cf_legal)
                sample.update(logit_m)
                samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_results(samples, cal_targets):
    """Aggregate per-sample results by calibration target."""
    agg = {}
    for ct in ["none"] + cal_targets:
        ct_key = str(ct)
        subset = [s for s in samples if str(s["cal_target"]) == ct_key]
        if not subset:
            continue
        entry = {"n_samples": len(subset)}
        # Numeric keys to aggregate
        keys = [f"probe_acc_L{l}" for l in MEASUREMENT_LAYERS]
        keys += [f"crosstalk_L{l}" for l in MEASUREMENT_LAYERS]
        keys += ["scale", "boundary_margin", "boundary_margin_frac_positive",
                 "legal_prob_mass"]
        for k in keys:
            vals = [s[k] for s in subset if s.get(k) is not None]
            if vals:
                entry[k] = float(np.mean(vals))
                entry[f"{k}_std"] = float(np.std(vals))
        agg[ct_key] = entry
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(agg, cal_targets, output_dir):
    """Generate plots for the layer propagation experiment."""
    os.makedirs(output_dir, exist_ok=True)

    conditions = ["none"] + [str(ct) for ct in cal_targets]
    cond_labels = ["No intervention"] + [f"Cal → L{ct}" for ct in cal_targets]
    colors = ["gray", "#1f77b4", "#d62728", "#2ca02c"]

    # 1. Probe accuracy across layers
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (ct, label) in enumerate(zip(conditions, cond_labels)):
        if ct not in agg:
            continue
        vals = [agg[ct].get(f"probe_acc_L{l}", 0) for l in MEASUREMENT_LAYERS]
        stds = [agg[ct].get(f"probe_acc_L{l}_std", 0) for l in MEASUREMENT_LAYERS]
        ax.errorbar(MEASUREMENT_LAYERS, vals, yerr=stds, fmt="o-",
                    color=colors[i], label=label, capsize=4, linewidth=2)
    ax.set_xlabel("Measurement Layer")
    ax.set_ylabel("Probe Accuracy (modified cell)")
    ax.set_title("Probe Accuracy Across Layers by Calibration Target")
    ax.set_xticks(MEASUREMENT_LAYERS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probe_accuracy.png"), dpi=150)
    print(f"  Saved probe_accuracy.png")
    plt.close()

    # 2. Crosstalk across layers
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (ct, label) in enumerate(zip(conditions, cond_labels)):
        if ct not in agg or ct == "none":
            continue
        vals = [agg[ct].get(f"crosstalk_L{l}", 0) for l in MEASUREMENT_LAYERS]
        stds = [agg[ct].get(f"crosstalk_L{l}_std", 0) for l in MEASUREMENT_LAYERS]
        ax.errorbar(MEASUREMENT_LAYERS, vals, yerr=stds, fmt="o-",
                    color=colors[conditions.index(ct)], label=label,
                    capsize=4, linewidth=2)
    ax.set_xlabel("Measurement Layer")
    ax.set_ylabel("Crosstalk (mean abs logit change, non-modified cells)")
    ax.set_title("Crosstalk Across Layers by Calibration Target")
    ax.set_xticks(MEASUREMENT_LAYERS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "crosstalk.png"), dpi=150)
    print(f"  Saved crosstalk.png")
    plt.close()

    # 3. Scale distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ct in enumerate([str(c) for c in cal_targets]):
        if ct not in agg:
            continue
        ax.bar(i, agg[ct].get("scale", 0),
               yerr=agg[ct].get("scale_std", 0),
               color=colors[i + 1], label=f"Cal → L{ct}", capsize=4)
    ax.set_xlabel("Calibration Target")
    ax.set_ylabel("Per-Cell Scale")
    ax.set_title("Intervention Scale by Calibration Target")
    ax.set_xticks(range(len(cal_targets)))
    ax.set_xticklabels([f"L{ct}" for ct in cal_targets])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scale_distribution.png"), dpi=150)
    print(f"  Saved scale_distribution.png")
    plt.close()

    # 4. Boundary margin per condition
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (ct, label) in enumerate(zip(conditions[1:], cond_labels[1:])):
        if ct not in agg:
            continue
        val = agg[ct].get("boundary_margin", 0)
        std = agg[ct].get("boundary_margin_std", 0)
        ax.bar(i, val, yerr=std, color=colors[i + 1], label=label, capsize=4)
    ax.set_xlabel("Calibration Target")
    ax.set_ylabel("Boundary Margin")
    ax.set_title("Boundary Margin by Calibration Target")
    ax.set_xticks(range(len(cal_targets)))
    ax.set_xticklabels([f"L{ct}" for ct in cal_targets])
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boundary_margin.png"), dpi=150)
    print(f"  Saved boundary_margin.png")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Layer propagation experiment for OthelloGPT")
    parser.add_argument("--ckpt", default="ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--probe-dir", required=True,
                        help="Directory with othello_layer{0-8}.pt files")
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="layer_propagation_results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cal-targets", default="6,7",
                        help="Comma-separated calibration target layers")
    args = parser.parse_args()

    cal_targets = [int(x) for x in args.cal_targets.split(",")]
    device = args.device

    # Load model
    print("Loading model...")
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.join(script_dir, "..")
    mconf = GPTConfig(vocab_size=61, block_size=59,
                      n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    ckpt_full = os.path.join(parent_dir, args.ckpt)
    model.load_state_dict(torch.load(ckpt_full, map_location=device))
    model.to(device)
    model.eval()

    # Load game data
    print("Loading game data...")
    board_seqs_int = torch.load(
        os.path.join(script_dir, "board_seqs_int.pth"), map_location="cpu")
    board_seqs_string = torch.load(
        os.path.join(script_dir, "board_seqs_string.pth"), map_location="cpu")

    # Load per-layer probes
    print(f"Loading probes from {args.probe_dir}...")
    probes = load_probes(args.probe_dir, device)
    print(f"  Loaded probes for layers: "
          f"{sorted(set(l for l, p in probes.keys()))}")

    # Build direction probe from layer 6
    direction_probe = build_direction_probe(probes, layer=6, device=device)
    print(f"  Direction probe shape: {direction_probe.shape}")

    # Run experiment
    print(f"\nRunning experiment: {args.n_games} games, "
          f"cal targets = {cal_targets}")
    samples = run_experiment(
        model, probes, direction_probe,
        board_seqs_int, board_seqs_string,
        cal_targets=cal_targets,
        n_games=args.n_games, seed=args.seed, device=device,
    )

    # Save raw samples
    output_dir = os.path.join(parent_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving {len(samples)} samples to {output_dir}/")
    with open(os.path.join(output_dir, "raw_samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved raw_samples.json")

    # Aggregate and save
    agg = aggregate_results(samples, cal_targets)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(f"  Saved results.json")

    # Print summary
    print("\n=== Summary ===")
    for ct in ["none"] + cal_targets:
        ct_key = str(ct)
        if ct_key not in agg:
            continue
        entry = agg[ct_key]
        label = "No intervention" if ct == "none" else f"Cal → L{ct}"
        accs = "  ".join(f"L{l}={entry.get(f'probe_acc_L{l}', 0):.3f}"
                         for l in MEASUREMENT_LAYERS)
        xtalks = "  ".join(f"L{l}={entry.get(f'crosstalk_L{l}', 0):.3f}"
                           for l in MEASUREMENT_LAYERS)
        scale = entry.get("scale", 0)
        bm = entry.get("boundary_margin", 0)
        print(f"  {label} (n={entry['n_samples']}):")
        print(f"    Scale: {scale:.3f}")
        print(f"    Probe acc: {accs}")
        print(f"    Crosstalk: {xtalks}")
        print(f"    Boundary margin: {bm:.3f}")

    # Plot
    print("\nGenerating plots...")
    plot_results(agg, cal_targets, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
