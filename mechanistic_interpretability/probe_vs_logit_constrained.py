"""
Constrained optimization: what pushes logit-optimal directions toward probe directions?

Tests multiple constraints added to the logit optimization objective:
  1. Lambda sweep: selective penalty with λ = [1, 5, 10, 50]
  2. Multi-move coherence: optimize for ALL affected moves simultaneously
  3. Layer consistency: direction must also change logits at layer 7
  4. Probe crosstalk: penalize changes to probe readings of non-modified cells

For each constraint, measure cos(probe, optimized_direction).

Usage:
  python probe_vs_logit_constrained.py \
      --probe-dir ../experiments/.../probe_checkpoints \
      --n-games 50 --output-dir ../experiments/probe_vs_logit_constrained
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
from probe_vs_logit_nonlinear import (
    forward_from_layer,
    cosine_sim,
    _cell_to_logit_idx,
    compute_actual_logit_changes,
)

LAYER_INTERVENE = 5
LAYER_PROBE = 6
POS_RANGE = (10, 50)


def optimize_selective_lambda(model, prefix_acts, pos, layer_intervene,
                               logit_idx, perturb_norm, lam,
                               n_steps=100, lr=0.05, maximize=True):
    """Selective optimization with configurable lambda."""
    device = prefix_acts.device
    x_base = prefix_acts.detach().clone()
    h_clean = x_base[0, pos].detach().clone()

    with torch.no_grad():
        logits_clean = forward_from_layer(model, x_base, layer_intervene)
        logits_clean_pos = logits_clean[0, pos].detach().clone()

    delta = torch.randn(512, device=device)
    delta = delta / delta.norm()
    delta = delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    other_mask = torch.ones(61, dtype=torch.bool, device=device)
    other_mask[logit_idx] = False
    other_mask[0] = False

    best_obj = float("-inf")
    best_delta = delta.detach().clone()

    for step in range(n_steps):
        optimizer.zero_grad()
        h_perturbed = h_clean + perturb_norm * delta
        seq = x_base[0].detach().clone()
        x_mod = torch.cat([seq[:pos], h_perturbed.unsqueeze(0), seq[pos+1:]], dim=0).unsqueeze(0)
        logits = forward_from_layer(model, x_mod, layer_intervene)
        logits_pos = logits[0, pos]

        target_change = logits_pos[logit_idx] - logits_clean_pos[logit_idx]
        other_changes = (logits_pos[other_mask] - logits_clean_pos[other_mask]).abs().mean()

        if maximize:
            obj = target_change - lam * other_changes
        else:
            obj = -target_change - lam * other_changes

        (-obj).backward()
        optimizer.step()
        with torch.no_grad():
            delta.div_(delta.norm().clamp(min=1e-8))

        if obj.item() > best_obj:
            best_obj = obj.item()
            best_delta = delta.detach().clone()

    return best_delta


def optimize_multi_move(model, prefix_acts, pos, layer_intervene,
                         newly_legal_idxs, newly_illegal_idxs, perturb_norm,
                         lam=1.0, n_steps=100, lr=0.05):
    """Optimize direction for ALL affected moves simultaneously.

    Objective: Σ logit_change(newly_legal) - Σ logit_change(newly_illegal)
               - lam * mean(|other logit changes|)
    """
    device = prefix_acts.device
    x_base = prefix_acts.detach().clone()
    h_clean = x_base[0, pos].detach().clone()

    with torch.no_grad():
        logits_clean = forward_from_layer(model, x_base, layer_intervene)
        logits_clean_pos = logits_clean[0, pos].detach().clone()

    delta = torch.randn(512, device=device)
    delta = delta / delta.norm()
    delta = delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    # "Other" logits = everything except affected moves and pass token
    affected_set = set(newly_legal_idxs + newly_illegal_idxs)
    other_mask = torch.ones(61, dtype=torch.bool, device=device)
    other_mask[0] = False
    for idx in affected_set:
        other_mask[idx] = False

    best_obj = float("-inf")
    best_delta = delta.detach().clone()

    for step in range(n_steps):
        optimizer.zero_grad()
        h_perturbed = h_clean + perturb_norm * delta
        seq = x_base[0].detach().clone()
        x_mod = torch.cat([seq[:pos], h_perturbed.unsqueeze(0), seq[pos+1:]], dim=0).unsqueeze(0)
        logits = forward_from_layer(model, x_mod, layer_intervene)
        logits_pos = logits[0, pos]

        changes = logits_pos - logits_clean_pos

        # Want newly legal logits to increase, newly illegal to decrease
        obj = torch.tensor(0.0, device=device)
        for idx in newly_legal_idxs:
            obj = obj + changes[idx]
        for idx in newly_illegal_idxs:
            obj = obj - changes[idx]

        if other_mask.any():
            other_changes = changes[other_mask].abs().mean()
            obj = obj - lam * other_changes

        (-obj).backward()
        optimizer.step()
        with torch.no_grad():
            delta.div_(delta.norm().clamp(min=1e-8))

        if obj.item() > best_obj:
            best_obj = obj.item()
            best_delta = delta.detach().clone()

    return best_delta


def optimize_layer_consistent(model, prefix_acts, pos, layer_intervene,
                               logit_idx, perturb_norm, lam_other=1.0,
                               n_steps=100, lr=0.05, maximize=True):
    """Optimize direction that changes logit consistently across layers 5-7.

    The direction is applied at layer_intervene. We measure logit change
    from layer_intervene forward AND from layer_intervene+1 forward (after
    one block). The direction must work at both.

    Objective: logit_change_from_L5 + logit_change_from_L6
               - lam_other * mean(|other logit changes at L5|)
    """
    device = prefix_acts.device
    x_base = prefix_acts.detach().clone()
    h_clean = x_base[0, pos].detach().clone()

    # Clean logits from L5
    with torch.no_grad():
        logits_clean_L5 = forward_from_layer(model, x_base, layer_intervene)
        logits_clean_L5_pos = logits_clean_L5[0, pos].detach().clone()

    # Clean activations at L6 (after one block from L5)
    with torch.no_grad():
        x_one_block = model.blocks[layer_intervene](x_base.clone())
        h_clean_L6 = x_one_block[0, pos].detach().clone()
        logits_clean_L6 = forward_from_layer(model, x_one_block, layer_intervene + 1)
        logits_clean_L6_pos = logits_clean_L6[0, pos].detach().clone()

    delta = torch.randn(512, device=device)
    delta = delta / delta.norm()
    delta = delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    other_mask = torch.ones(61, dtype=torch.bool, device=device)
    other_mask[logit_idx] = False
    other_mask[0] = False

    best_obj = float("-inf")
    best_delta = delta.detach().clone()

    for step in range(n_steps):
        optimizer.zero_grad()

        # Perturb at L5
        h_perturbed = h_clean + perturb_norm * delta
        seq = x_base[0].detach().clone()
        x_mod = torch.cat([seq[:pos], h_perturbed.unsqueeze(0), seq[pos+1:]], dim=0).unsqueeze(0)

        # Logit change from L5
        logits_L5 = forward_from_layer(model, x_mod, layer_intervene)
        change_L5 = logits_L5[0, pos, logit_idx] - logits_clean_L5_pos[logit_idx]

        # Also perturb at L6 with same delta (direction should work at both layers)
        h_perturbed_L6 = h_clean_L6 + perturb_norm * delta
        seq_L6 = x_one_block[0].detach().clone()
        x_mod_L6 = torch.cat([seq_L6[:pos], h_perturbed_L6.unsqueeze(0), seq_L6[pos+1:]], dim=0).unsqueeze(0)
        logits_L6 = forward_from_layer(model, x_mod_L6, layer_intervene + 1)
        change_L6 = logits_L6[0, pos, logit_idx] - logits_clean_L6_pos[logit_idx]

        other_changes = (logits_L5[0, pos, other_mask] - logits_clean_L5_pos[other_mask]).abs().mean()

        if maximize:
            obj = change_L5 + change_L6 - lam_other * other_changes
        else:
            obj = -change_L5 - change_L6 - lam_other * other_changes

        (-obj).backward()
        optimizer.step()
        with torch.no_grad():
            delta.div_(delta.norm().clamp(min=1e-8))

        if obj.item() > best_obj:
            best_obj = obj.item()
            best_delta = delta.detach().clone()

    return best_delta


def optimize_probe_crosstalk(model, prefix_acts, pos, layer_intervene,
                              logit_idx, perturb_norm, probe_linear,
                              mod_cell_idx, lam_probe=1.0,
                              n_steps=100, lr=0.05, maximize=True):
    """Optimize direction that changes target logit without changing probe
    readings for non-modified cells.

    Objective: logit_change(target) - lam_probe * mean(|probe_change(other cells)|)
    """
    device = prefix_acts.device
    x_base = prefix_acts.detach().clone()
    h_clean = x_base[0, pos].detach().clone()

    with torch.no_grad():
        logits_clean = forward_from_layer(model, x_base, layer_intervene)
        logits_clean_pos = logits_clean[0, pos].detach().clone()

    # Clean probe output at the probe layer
    # We need to run from L5 to L6 to get the activation, then apply probe
    with torch.no_grad():
        x_to_probe = x_base.clone()
        for block in model.blocks[layer_intervene:LAYER_PROBE]:
            x_to_probe = block(x_to_probe)
        probe_clean = probe_linear(x_to_probe[0, pos])  # (192,) = 64*3
        probe_clean = probe_clean.view(64, 3)  # (64, 3)

    # Mask for non-modified cells
    cell_mask = torch.ones(64, dtype=torch.bool, device=device)
    cell_mask[mod_cell_idx] = False

    delta = torch.randn(512, device=device)
    delta = delta / delta.norm()
    delta = delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    best_obj = float("-inf")
    best_delta = delta.detach().clone()

    for step in range(n_steps):
        optimizer.zero_grad()

        h_perturbed = h_clean + perturb_norm * delta
        seq = x_base[0].detach().clone()
        x_mod = torch.cat([seq[:pos], h_perturbed.unsqueeze(0), seq[pos+1:]], dim=0).unsqueeze(0)

        # Logit change
        logits = forward_from_layer(model, x_mod, layer_intervene)
        target_change = logits[0, pos, logit_idx] - logits_clean_pos[logit_idx]

        # Probe change at non-modified cells
        x_to_probe_mod = x_mod.clone()
        for block in model.blocks[layer_intervene:LAYER_PROBE]:
            x_to_probe_mod = block(x_to_probe_mod)
        probe_mod = probe_linear(x_to_probe_mod[0, pos]).view(64, 3)
        probe_diff = (probe_mod[cell_mask] - probe_clean[cell_mask]).abs().mean()

        if maximize:
            obj = target_change - lam_probe * probe_diff
        else:
            obj = -target_change - lam_probe * probe_diff

        (-obj).backward()
        optimizer.step()
        with torch.no_grad():
            delta.div_(delta.norm().clamp(min=1e-8))

        if obj.item() > best_obj:
            best_obj = obj.item()
            best_delta = delta.detach().clone()

    return best_delta


def run_experiment(model, probes, direction_probe, board_seqs_int,
                   board_seqs_string, n_games=50, seed=42, device="cpu",
                   opt_steps=100):
    rng = random.Random(seed)
    samples = []
    lambdas = [1, 5, 10, 50]

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

            # Probe direction and scale
            flip_dirs = compute_flip_dirs_from_direction_probe(
                direction_probe, mods, mode)
            probe_dir = flip_dirs[0]
            probe_dir_hat = probe_dir / probe_dir.norm()

            target_probe = probes[(LAYER_PROBE, parity)]
            scales = compute_per_cell_scales_downstream(
                model, prefix_acts, direction_probe, target_probe,
                mods, pos, LAYER_INTERVENE, LAYER_PROBE, mode)
            scale = scales[0]

            # Get perturbation norm from probe intervention
            with torch.no_grad():
                d_hat = probe_dir / probe_dir.norm()
                coeff = prefix_acts[0, pos] @ d_hat
                perturbation = -scale * coeff * d_hat
                perturb_norm = perturbation.norm().item()

            if perturb_norm < 1e-6:
                continue

            # Get affected move logit indices
            newly_legal_idxs = [_cell_to_logit_idx[m] for m in newly_legal if m in _cell_to_logit_idx]
            newly_illegal_idxs = [_cell_to_logit_idx[m] for m in newly_illegal if m in _cell_to_logit_idx]
            all_affected = list(newly_legal | newly_illegal)
            all_affected_idxs = [_cell_to_logit_idx[m] for m in all_affected if m in _cell_to_logit_idx]

            if not all_affected_idxs:
                continue

            # Modified cell index (0-63) for probe crosstalk
            r, c, orig_val, target_val = mods[0]
            mod_cell_idx = r * 8 + c

            # Pick a representative target move (first newly legal, or first newly illegal)
            if newly_legal_idxs:
                rep_lidx = newly_legal_idxs[0]
                rep_maximize = True
            else:
                rep_lidx = newly_illegal_idxs[0]
                rep_maximize = False

            sample = {
                "game_idx": gi,
                "pos": pos,
                "intervention_type": intervention_type,
                "n_newly_legal": len(newly_legal),
                "n_newly_illegal": len(newly_illegal),
                "perturb_norm": perturb_norm,
            }

            # === 1. Lambda sweep (per representative move) ===
            for lam in lambdas:
                sel_dir = optimize_selective_lambda(
                    model, prefix_acts, pos, LAYER_INTERVENE,
                    rep_lidx, perturb_norm, lam=lam,
                    n_steps=opt_steps, maximize=rep_maximize)
                sample[f"cos_probe_sel_lam{lam}"] = cosine_sim(probe_dir, sel_dir)

            # === 2. Multi-move coherence ===
            multi_dir = optimize_multi_move(
                model, prefix_acts, pos, LAYER_INTERVENE,
                newly_legal_idxs, newly_illegal_idxs, perturb_norm,
                lam=1.0, n_steps=opt_steps)
            sample["cos_probe_multi"] = cosine_sim(probe_dir, multi_dir)

            multi_dir_lam10 = optimize_multi_move(
                model, prefix_acts, pos, LAYER_INTERVENE,
                newly_legal_idxs, newly_illegal_idxs, perturb_norm,
                lam=10.0, n_steps=opt_steps)
            sample["cos_probe_multi_lam10"] = cosine_sim(probe_dir, multi_dir_lam10)

            # === 3. Layer consistency ===
            layer_dir = optimize_layer_consistent(
                model, prefix_acts, pos, LAYER_INTERVENE,
                rep_lidx, perturb_norm, lam_other=1.0,
                n_steps=opt_steps, maximize=rep_maximize)
            sample["cos_probe_layer"] = cosine_sim(probe_dir, layer_dir)

            layer_dir_lam10 = optimize_layer_consistent(
                model, prefix_acts, pos, LAYER_INTERVENE,
                rep_lidx, perturb_norm, lam_other=10.0,
                n_steps=opt_steps, maximize=rep_maximize)
            sample["cos_probe_layer_lam10"] = cosine_sim(probe_dir, layer_dir_lam10)

            # === 4. Probe crosstalk penalty ===
            probe_linear = target_probe
            xtalk_dir = optimize_probe_crosstalk(
                model, prefix_acts, pos, LAYER_INTERVENE,
                rep_lidx, perturb_norm, probe_linear,
                mod_cell_idx, lam_probe=1.0,
                n_steps=opt_steps, maximize=rep_maximize)
            sample["cos_probe_xtalk"] = cosine_sim(probe_dir, xtalk_dir)

            xtalk_dir_lam10 = optimize_probe_crosstalk(
                model, prefix_acts, pos, LAYER_INTERVENE,
                rep_lidx, perturb_norm, probe_linear,
                mod_cell_idx, lam_probe=10.0,
                n_steps=opt_steps, maximize=rep_maximize)
            sample["cos_probe_xtalk_lam10"] = cosine_sim(probe_dir, xtalk_dir_lam10)

            samples.append(sample)

    return samples


def plot_results(samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # B3b baseline from previous experiment
    baseline_cos = 0.18  # approximate cos(probe, gradient) from B3b

    # Collect data for each condition
    conditions = [
        ("sel_lam1", "cos_probe_sel_lam1", "Selective λ=1"),
        ("sel_lam5", "cos_probe_sel_lam5", "Selective λ=5"),
        ("sel_lam10", "cos_probe_sel_lam10", "Selective λ=10"),
        ("sel_lam50", "cos_probe_sel_lam50", "Selective λ=50"),
        ("multi", "cos_probe_multi", "Multi-move λ=1"),
        ("multi_lam10", "cos_probe_multi_lam10", "Multi-move λ=10"),
        ("layer", "cos_probe_layer", "Layer-consistent λ=1"),
        ("layer_lam10", "cos_probe_layer_lam10", "Layer-consistent λ=10"),
        ("xtalk", "cos_probe_xtalk", "Probe-crosstalk λ=1"),
        ("xtalk_lam10", "cos_probe_xtalk_lam10", "Probe-crosstalk λ=10"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    random_baseline = np.sqrt(2 / np.pi) / np.sqrt(512)

    # Panel 1: Bar chart of mean |cos(probe, optimized)| per condition
    ax = axes[0, 0]
    names = []
    means = []
    stds = []
    for _, key, label in conditions:
        vals = [abs(s[key]) for s in samples if key in s]
        if vals:
            names.append(label)
            means.append(np.mean(vals))
            stds.append(np.std(vals) / np.sqrt(len(vals)))

    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=3, alpha=0.7,
                  color=['tab:blue']*4 + ['tab:green']*2 + ['tab:orange']*2 + ['tab:red']*2)
    ax.axhline(baseline_cos, color="black", linestyle="--", alpha=0.7,
               label=f"B3b gradient baseline ({baseline_cos:.2f})")
    ax.axhline(random_baseline, color="gray", linestyle=":", alpha=0.5,
               label=f"Random baseline ({random_baseline:.3f})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |cos(probe, optimized)|")
    ax.set_title("Probe alignment by constraint type")
    ax.legend(fontsize=8)

    # Panel 2: Lambda sweep detail
    ax = axes[0, 1]
    lam_vals = [1, 5, 10, 50]
    lam_means = []
    lam_stds = []
    for lam in lam_vals:
        key = f"cos_probe_sel_lam{lam}"
        vals = [abs(s[key]) for s in samples if key in s]
        lam_means.append(np.mean(vals) if vals else 0)
        lam_stds.append(np.std(vals) / np.sqrt(len(vals)) if vals else 0)

    ax.errorbar(lam_vals, lam_means, yerr=lam_stds, marker="o", linewidth=2,
                capsize=4, color="tab:blue", label="Selective opt")
    ax.axhline(baseline_cos, color="black", linestyle="--", alpha=0.7,
               label=f"Gradient baseline")
    ax.axhline(random_baseline, color="gray", linestyle=":", alpha=0.5,
               label="Random")
    ax.set_xlabel("λ (selectivity penalty)")
    ax.set_ylabel("Mean |cos(probe, optimized)|")
    ax.set_title("Lambda sweep: selectivity penalty")
    ax.set_xscale("log")
    ax.legend(fontsize=9)

    # Panel 3: Histograms for top conditions
    ax = axes[1, 0]
    bins = np.linspace(-1, 1, 40)
    top_conditions = [
        ("cos_probe_sel_lam50", "Selective λ=50", "tab:blue"),
        ("cos_probe_multi_lam10", "Multi-move λ=10", "tab:green"),
        ("cos_probe_xtalk_lam10", "Probe-xtalk λ=10", "tab:red"),
    ]
    for key, label, color in top_conditions:
        vals = [s[key] for s in samples if key in s]
        if vals:
            ax.hist(vals, bins=bins, alpha=0.4, label=label, color=color)
    ax.axvline(random_baseline, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(-random_baseline, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("cos(probe, optimized)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of probe alignment (strongest constraints)")
    ax.legend(fontsize=9)

    # Panel 4: Summary stats
    ax = axes[1, 1]
    ax.axis("off")
    stats = [f"N samples: {len(samples)}"]
    stats.append(f"Random baseline: {random_baseline:.4f}")
    stats.append(f"Gradient baseline (B3b): ~{baseline_cos:.3f}")
    stats.append("")
    for _, key, label in conditions:
        vals = [abs(s[key]) for s in samples if key in s]
        if vals:
            stats.append(f"{label}: {np.mean(vals):.4f} ± {np.std(vals)/np.sqrt(len(vals)):.4f}")
    ax.text(0.05, 0.95, "\n".join(stats), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "constrained_analysis.png"), dpi=150)
    plt.close()
    print(f"Saved plot to {output_dir}/constrained_analysis.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--n-games", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opt-steps", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    # Try .pth first (cluster), fall back to .npy (local)
    pth_path = os.path.join(script_dir, "board_seqs_int.pth")
    npy_path = os.path.join(script_dir, "board_seqs_int_small.npy")
    if os.path.exists(pth_path):
        board_seqs_int = torch.load(pth_path, map_location="cpu")
        board_seqs_string = torch.load(
            os.path.join(script_dir, "board_seqs_string.pth"), map_location="cpu")
    else:
        board_seqs_int = torch.from_numpy(
            np.load(npy_path)).long()
        board_seqs_string = torch.from_numpy(
            np.load(os.path.join(script_dir, "board_seqs_string_small.npy"))).long()
    print(f"Data loaded: {board_seqs_int.shape[0]} games")

    probes = load_probes(args.probe_dir, device)
    direction_probe = build_direction_probe(probes, layer=LAYER_PROBE, device=device)
    print(f"Direction probe shape: {direction_probe.shape}")

    samples = run_experiment(
        model, probes, direction_probe,
        board_seqs_int, board_seqs_string,
        n_games=args.n_games, seed=args.seed, device=device,
        opt_steps=args.opt_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "raw_samples.json"), "w") as f:
        json.dump(samples, f, indent=2)

    # Aggregate
    conditions = [
        "cos_probe_sel_lam1", "cos_probe_sel_lam5",
        "cos_probe_sel_lam10", "cos_probe_sel_lam50",
        "cos_probe_multi", "cos_probe_multi_lam10",
        "cos_probe_layer", "cos_probe_layer_lam10",
        "cos_probe_xtalk", "cos_probe_xtalk_lam10",
    ]
    results = {"n_samples": len(samples)}
    for key in conditions:
        vals = [abs(s[key]) for s in samples if key in s]
        if vals:
            results[key] = float(np.mean(vals))
            results[key + "_std"] = float(np.std(vals))

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_results(samples, args.output_dir)

    print(f"\nResults saved to {args.output_dir}")
    print(f"\n=== Summary ===")
    print(f"N samples: {len(samples)}")
    for key in conditions:
        vals = [abs(s[key]) for s in samples if key in s]
        if vals:
            print(f"  {key}: {np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
