"""
Layer propagation experiment: how do causal interventions propagate across layers?

Intervenes at a specified layer using that layer's native probe directions.
Calibrates locally (at the intervention point), then measures probe accuracy
and crosstalk at all downstream layers.

Key question: if we modify the residual stream so the native probe reads the
target board state, does the model's own computation preserve this through
subsequent layers? And do output logits follow?

Usage:
  python layer_propagation.py --layer 4 \
      --probe-dir ../experiments/.../probe_checkpoints \
      --n-games 200 --output-dir ../experiments/layer_propagation/L4
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
# Flip directions from direction probe
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


# ---------------------------------------------------------------------------
# Local per-cell scale calibration (at intervention point, no forward pass)
# ---------------------------------------------------------------------------
def compute_per_cell_scales_local(h, probe_linear, direction_probe,
                                  modifications, mode):
    """Binary search for minimal per-cell scale that flips native probe locally.

    h: (512,) activation at the intervention point
    probe_linear: nn.Linear(512, 192) — native probe at intervention layer
    direction_probe: (2, 512, 8, 8, 3) tensor
    mode: 0 (even) or 1 (odd)

    Returns list of per-cell scales.
    """
    scales = []

    for (r, c, orig_val, target_val) in modifications:
        target_class = board_val_to_probe_class(target_val)
        current_class = board_val_to_probe_class(orig_val)
        cell = r * 8 + c

        # Flip direction from direction probe
        flip_dir = (direction_probe[mode, :, r, c, target_class]
                    - direction_probe[mode, :, r, c, current_class])
        d_hat = flip_dir / flip_dir.norm()
        coeff = (h @ d_hat).item()

        def probe_pred_at_scale(s):
            h_mod = h - s * coeff * d_hat
            with torch.no_grad():
                logits = probe_linear(h_mod).view(64, OPTIONS)
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
# Downstream per-cell scale calibration (kept for backward compat / B1)
# ---------------------------------------------------------------------------
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
            # "Layer N probe" = trained on resid_post of block N
            # So we need to run through block target_layer
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
                          pos, layer_intervene, capture_layers,
                          native_layer=None):
    """Apply intervention and capture residuals at multiple downstream layers.

    Returns (logits_at_pos, resids_dict) where resids_dict maps layer -> (512,).
    If native_layer is specified, also captures the modified activation at the
    intervention point (before any blocks run).
    """
    x = prefix_acts.clone()
    x = apply_intervention(x, flip_dirs, pos, scale=None,
                           per_cell_scales=per_cell_scales)

    resids = {}
    # Capture at the intervention point (native layer)
    if native_layer is not None:
        resids[native_layer] = x[0, pos].detach().clone()

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
                   board_seqs_string, layer, cal_depths=None, n_games=200,
                   seed=42, device="cuda"):
    """Run layer propagation experiment.

    Intervenes at resid_post of block `layer` using layer `layer` probe.
    For each cal_depth, calibrates at layer + cal_depth (0 = local).
    Measures at all downstream layers.

    layer: the block/probe layer (0-indexed). Intervention modifies resid_post
           of this block. compute_prefix_activations is called with layer+1.
    cal_depths: list of calibration depths (0 = local, 1 = +1 layer, etc.)

    Returns list of per-sample dicts.
    """
    if cal_depths is None:
        cal_depths = [0]
    layer_intervene = layer + 1  # compute_prefix_activations convention
    # Downstream layers captured by forward pass through remaining blocks
    downstream_layers = list(range(layer + 1, 8))
    # All measurement layers: native + downstream
    measurement_layers = [layer] + downstream_layers
    rng = random.Random(seed)
    samples = []

    # Validate cal_depths: can't calibrate beyond layer 7
    valid_cal_depths = [d for d in cal_depths if layer + d <= 7]
    if len(valid_cal_depths) < len(cal_depths):
        skipped = [d for d in cal_depths if layer + d > 7]
        print(f"  Skipping cal_depth {skipped} (would exceed layer 7)")
    cal_depths = valid_cal_depths

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
            # Clean forward pass — capture at downstream layers
            clean_logits, clean_resids, prefix_acts = \
                clean_forward_multi_capture(
                    model, input_tokens, pos,
                    layer_intervene, downstream_layers
                )
            # Native layer resid = prefix_acts (resid_post of block `layer`)
            clean_resids[layer] = prefix_acts[0, pos].detach().clone()

            # Legal moves for logit metrics
            original_legal = compute_legal_moves(board_state, color)
            cf_mods = [(r, c, tgt) for (r, c, _, tgt) in mods]
            cf_legal = compute_counterfactual_legal(board_state, cf_mods, color)

            board_flat = board_state.flatten().tolist()

            # --- No-intervention baseline ---
            sample_base = {
                "game_idx": gi, "pos": pos, "color": int(color),
                "board_state": board_flat,
                "modifications": [(r, c, int(o), int(t)) for r, c, o, t in mods],
                "condition": "none", "cal_depth": -1, "scale": 0.0,
                "n_original_legal": len(original_legal),
                "n_cf_legal": len(cf_legal),
            }
            for ml in measurement_layers:
                probe = probes[(ml, parity)]
                sample_base[f"probe_acc_L{ml}"] = measure_probe_acc(
                    clean_resids[ml], probe, mods)
                sample_base[f"crosstalk_L{ml}"] = 0.0
            logit_m = measure_logit_metrics(clean_logits, clean_logits,
                                            original_legal, cf_legal)
            sample_base.update(logit_m)
            samples.append(sample_base)

            # --- Intervention at each cal_depth ---
            flip_dirs = compute_flip_dirs_from_direction_probe(
                direction_probe, mods, mode)

            for cal_depth in cal_depths:
                cal_layer = layer + cal_depth

                if cal_depth == 0:
                    # Local calibration
                    native_probe = probes[(layer, parity)]
                    h = prefix_acts[0, pos].detach()
                    per_cell_scales = compute_per_cell_scales_local(
                        h, native_probe, direction_probe, mods, mode
                    )
                else:
                    # Downstream calibration
                    target_probe = probes[(cal_layer, parity)]
                    per_cell_scales = compute_per_cell_scales_downstream(
                        model, prefix_acts, direction_probe, target_probe,
                        mods, pos, layer_intervene, cal_layer, mode
                    )

                intv_logits, intv_resids = forward_multi_capture(
                    model, prefix_acts, flip_dirs, per_cell_scales,
                    pos, layer_intervene, downstream_layers,
                    native_layer=layer
                )

                cond_name = f"cal_depth_{cal_depth}"
                sample = {
                    "game_idx": gi, "pos": pos, "color": int(color),
                    "board_state": board_flat,
                    "modifications": [(r, c, int(o), int(t))
                                      for r, c, o, t in mods],
                    "condition": cond_name,
                    "cal_depth": cal_depth,
                    "scale": per_cell_scales[0],  # N=1
                    "n_original_legal": len(original_legal),
                    "n_cf_legal": len(cf_legal),
                }

                for ml in measurement_layers:
                    probe = probes[(ml, parity)]
                    sample[f"probe_acc_L{ml}"] = measure_probe_acc(
                        intv_resids[ml], probe, mods)
                    sample[f"crosstalk_L{ml}"] = measure_crosstalk(
                        clean_resids[ml], intv_resids[ml], probe, mods)

                logit_m = measure_logit_metrics(clean_logits, intv_logits,
                                                original_legal, cf_legal)
                sample.update(logit_m)
                samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_results(samples, measurement_layers):
    """Aggregate per-sample results by condition."""
    conditions = sorted(set(s["condition"] for s in samples))
    agg = {}
    for cond in conditions:
        subset = [s for s in samples if s["condition"] == cond]
        if not subset:
            continue
        entry = {"n_samples": len(subset)}
        keys = [f"probe_acc_L{l}" for l in measurement_layers]
        keys += [f"crosstalk_L{l}" for l in measurement_layers]
        keys += ["scale", "boundary_margin", "boundary_margin_frac_positive",
                 "legal_prob_mass", "li_topn_accuracy", "li_topn_errors"]
        for k in keys:
            vals = [s[k] for s in subset if s.get(k) is not None]
            if vals:
                entry[k] = float(np.mean(vals))
                entry[f"{k}_std"] = float(np.std(vals))
        agg[cond] = entry
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(agg, layer, measurement_layers, cal_depths, output_dir):
    """Generate plots for the layer propagation experiment."""
    os.makedirs(output_dir, exist_ok=True)

    # Build condition list: none + each cal_depth
    plot_colors = ["gray", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    conditions = ["none"]
    cond_labels = ["No intervention"]
    for d in sorted(cal_depths):
        conditions.append(f"cal_depth_{d}")
        cal_layer = layer + d
        cond_labels.append(f"Cal depth {d} (→L{cal_layer})")

    # 1. Probe accuracy across layers
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (ct, label) in enumerate(zip(conditions, cond_labels)):
        if ct not in agg:
            continue
        vals = [agg[ct].get(f"probe_acc_L{l}", 0) for l in measurement_layers]
        stds = [agg[ct].get(f"probe_acc_L{l}_std", 0) for l in measurement_layers]
        ax.errorbar(measurement_layers, vals, yerr=stds, fmt="o-",
                    color=plot_colors[i % len(plot_colors)],
                    label=label, capsize=4, linewidth=2)
    ax.set_xlabel("Measurement Layer")
    ax.set_ylabel("Probe Accuracy (modified cell)")
    ax.set_title(f"Probe Accuracy: Intervene at Layer {layer}")
    ax.set_xticks(measurement_layers)
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
        if ct == "none" or ct not in agg:
            continue
        vals = [agg[ct].get(f"crosstalk_L{l}", 0) for l in measurement_layers]
        stds = [agg[ct].get(f"crosstalk_L{l}_std", 0) for l in measurement_layers]
        ax.errorbar(measurement_layers, vals, yerr=stds, fmt="o-",
                    color=plot_colors[i % len(plot_colors)],
                    label=label, capsize=4, linewidth=2)
    ax.set_xlabel("Measurement Layer")
    ax.set_ylabel("Crosstalk (mean abs logit change, non-modified cells)")
    ax.set_title(f"Crosstalk: Intervene at Layer {layer}")
    ax.set_xticks(measurement_layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "crosstalk.png"), dpi=150)
    print(f"  Saved crosstalk.png")
    plt.close()

    # 3. Scale comparison across cal_depths
    intv_conds = [c for c in conditions if c != "none" and c in agg]
    if intv_conds:
        fig, ax = plt.subplots(figsize=(6, 4))
        x_pos = range(len(intv_conds))
        means = [agg[c].get("scale", 0) for c in intv_conds]
        stds = [agg[c].get("scale_std", 0) for c in intv_conds]
        labels = [f"depth {c.split('_')[-1]}" for c in intv_conds]
        ax.bar(x_pos, means, yerr=stds, color=plot_colors[1:len(intv_conds)+1],
               capsize=4, width=0.5)
        ax.set_ylabel("Per-Cell Scale")
        ax.set_title(f"Intervention Scale by Cal Depth (Layer {layer})")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "scale_distribution.png"), dpi=150)
        print(f"  Saved scale_distribution.png")
        plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Layer propagation experiment for OthelloGPT")
    parser.add_argument("--layer", type=int, required=True,
                        help="Intervention layer (block/probe number, 0-indexed). "
                             "Modifies resid_post of this block using this layer's probe.")
    parser.add_argument("--ckpt", default="ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--probe-dir", required=True,
                        help="Directory with othello_layer{0-8}.pt files")
    parser.add_argument("--cal-depths", type=str, default="0",
                        help="Comma-separated calibration depths "
                             "(0=local, 1=+1 layer downstream, etc.)")
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="layer_propagation_results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    layer = args.layer
    cal_depths = [int(x) for x in args.cal_depths.split(",")]
    measurement_layers = list(range(layer, 8))
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

    # Build direction probe from native layer
    direction_probe = build_direction_probe(probes, layer=layer, device=device)
    print(f"  Direction probe from layer {layer}, shape: {direction_probe.shape}")
    print(f"  Measurement layers: {measurement_layers}")
    print(f"  Calibration depths: {cal_depths}")

    # Run experiment
    print(f"\nRunning experiment: {args.n_games} games, "
          f"intervene at layer {layer}")
    samples = run_experiment(
        model, probes, direction_probe,
        board_seqs_int, board_seqs_string,
        layer=layer, cal_depths=cal_depths,
        n_games=args.n_games, seed=args.seed, device=device,
    )

    # Save raw samples
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving {len(samples)} samples to {output_dir}/")
    with open(os.path.join(output_dir, "raw_samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved raw_samples.json")

    # Aggregate and save
    agg = aggregate_results(samples, measurement_layers)
    config = {
        "layer": layer,
        "cal_depths": cal_depths,
        "measurement_layers": measurement_layers,
        "n_games": args.n_games,
        "seed": args.seed,
    }
    output = {"config": config, "results": agg}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved results.json")

    # Print summary
    print("\n=== Summary ===")
    for cond in sorted(agg.keys()):
        entry = agg[cond]
        if cond == "none":
            label = "No intervention"
        else:
            d = cond.split("_")[-1]
            cal_layer = layer + int(d)
            label = f"Cal depth {d} (→L{cal_layer})"
        accs = "  ".join(f"L{l}={entry.get(f'probe_acc_L{l}', 0):.3f}"
                         for l in measurement_layers)
        xtalks = "  ".join(f"L{l}={entry.get(f'crosstalk_L{l}', 0):.3f}"
                           for l in measurement_layers)
        scale = entry.get("scale", 0)
        bm = entry.get("boundary_margin")
        bm_s = f"{bm:.3f}" if bm is not None else "N/A"
        bmf = entry.get("boundary_margin_frac_positive")
        bmf_s = f"{bmf:.3f}" if bmf is not None else "N/A"
        li = entry.get("li_topn_accuracy")
        li_s = f"{li:.3f}" if li is not None else "N/A"
        print(f"  {label} (n={entry['n_samples']}):")
        print(f"    Scale: {scale:.3f}")
        print(f"    Probe acc: {accs}")
        print(f"    Crosstalk: {xtalks}")
        print(f"    Boundary margin: {bm_s} (frac_pos={bmf_s})")
        print(f"    Li et al. top-N acc: {li_s}")

    # Plot
    print("\nGenerating plots...")
    plot_results(agg, layer, measurement_layers, cal_depths, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
