"""
Probe vs logit direction comparison experiment.

Tests whether probe-based interventions (change a board cell) are geometrically
similar to logit-gradient interventions (change a move's logit). If cosine
similarity is high, it's ambiguous whether we're intervening on board state
or on the space of possible moves.

For each game/position:
  1. Apply probe-based intervention at layer 5 to flip/add a cell
  2. Identify which moves change legality (newly legal / newly illegal)
  3. For each affected move m, compute gradient of logit(m) w.r.t. layer 5
     activation (clean, no intervention)
  4. Compare probe direction and logit gradient via cosine similarity

Usage:
  python probe_vs_logit_directions.py \
      --probe-dir ../experiments/.../probe_checkpoints \
      --n-games 200 --output-dir ../experiments/probe_vs_logit
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAYER_INTERVENE = 5
LAYER_PROBE = 6  # probe used for direction & calibration
POS_RANGE = (10, 50)
OPTIONS = 3

# Map board cell index (0-63) to model logit index (1-60)
# Model output: index 0 = pass, indices 1-60 = stoi_indices
_cell_to_logit_idx = {}
for i, cell in enumerate(stoi_indices):
    _cell_to_logit_idx[cell] = i + 1  # +1 because index 0 = pass


def compute_logit_gradients_batch(model, prefix_acts, pos, layer_intervene, target_moves):
    """Compute gradients of logit(m) w.r.t. layer 5 activation for multiple moves.

    Single forward pass, one backward per move (reusing the graph).
    Returns dict mapping move -> (512,) gradient vector.

    target_moves: list of board cell indices (0-63)
    """
    # Filter to valid moves (skip center cells)
    valid = [(m, _cell_to_logit_idx[m]) for m in target_moves if m in _cell_to_logit_idx]
    if not valid:
        return {}

    # Build activation tensor with gradient tracking at pos
    x = prefix_acts.detach().clone()
    h = x[0, pos].detach().clone().requires_grad_(True)

    # Reconstruct sequence: replace position pos with grad-tracked h
    seq = x[0]  # (seq_len, 512)
    x_mod = torch.cat([seq[:pos], h.unsqueeze(0), seq[pos+1:]], dim=0).unsqueeze(0)

    # Single forward pass through remaining blocks
    for block in model.blocks[layer_intervene:]:
        x_mod = block(x_mod)
    x_mod = model.ln_f(x_mod)
    logits = model.head(x_mod)

    # One backward per move, reusing the computation graph
    grads = {}
    for i, (move, logit_idx) in enumerate(valid):
        retain = (i < len(valid) - 1)  # retain graph for all but last
        target_logit = logits[0, pos, logit_idx]
        target_logit.backward(retain_graph=retain)
        grads[move] = h.grad.detach().clone()
        h.grad.zero_()

    return grads


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    norm_a = a.norm()
    norm_b = b.norm()
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return (a @ b / (norm_a * norm_b)).item()


def subspace_similarity(probe_dir, grad_dirs):
    """How much of probe_dir lies in the subspace spanned by grad_dirs.

    Returns the fraction of probe_dir's variance explained by the subspace
    (i.e., ||proj||^2 / ||probe_dir||^2).
    """
    if len(grad_dirs) == 0:
        return 0.0
    # Stack gradients and compute SVD
    G = torch.stack(grad_dirs, dim=0)  # (n_moves, 512)
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    # Project probe_dir onto column space of G
    # Vh rows are the right singular vectors (basis of row space)
    p = probe_dir / probe_dir.norm()
    coeffs = Vh @ p  # project onto each basis vector
    proj_norm_sq = (coeffs ** 2).sum().item()
    return proj_norm_sq  # already normalized since p is unit


def run_experiment(model, probes, direction_probe, board_seqs_int,
                   board_seqs_string, n_games=200, seed=42, device="cuda"):
    """Run probe vs logit direction comparison.

    For each game, do both a flip and an add (if possible), then for each
    affected move, compute logit gradient and compare to probe direction.
    """
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

        # Compute prefix activations (shared across all interventions)
        with torch.no_grad():
            prefix_acts = compute_prefix_activations(
                model, input_tokens, LAYER_INTERVENE)

        # Legal moves before intervention
        original_legal = compute_legal_moves(board_state, color)

        # Try both flip and add conditions
        for intervention_type in ["flip", "add"]:
            if intervention_type == "flip":
                mods = select_flip_noninteracting(board_state, 1, rng)
            else:
                mods = select_add_noninteracting(board_state, 1, rng)

            if mods is None:
                continue

            # Counterfactual legal moves
            cf_mods = [(r, c, tgt) for (r, c, _, tgt) in mods]
            cf_legal = compute_counterfactual_legal(board_state, cf_mods, color)

            newly_legal = cf_legal - original_legal
            newly_illegal = original_legal - cf_legal

            # Skip if no moves changed legality
            if len(newly_legal) == 0 and len(newly_illegal) == 0:
                continue

            # Compute probe flip direction
            flip_dirs = compute_flip_dirs_from_direction_probe(
                direction_probe, mods, mode)
            probe_dir = flip_dirs[0]  # N=1, single direction
            probe_dir_hat = probe_dir / probe_dir.norm()

            # Calibrate scale so probe flips at layer 6
            target_probe = probes[(LAYER_PROBE, parity)]
            scales = compute_per_cell_scales_downstream(
                model, prefix_acts, direction_probe, target_probe,
                mods, pos, LAYER_INTERVENE, LAYER_PROBE, mode)
            scale = scales[0]

            # Compute logit gradients for all affected moves (single forward pass)
            all_affected = list(newly_legal | newly_illegal)
            grads = compute_logit_gradients_batch(
                model, prefix_acts, pos, LAYER_INTERVENE, all_affected)

            cosine_sims_legal = []
            cosine_sims_illegal = []
            all_grads = []

            for move in newly_legal:
                if move not in grads:
                    continue
                grad = grads[move]
                cs = cosine_sim(probe_dir, grad)
                cosine_sims_legal.append({
                    "move": int(move),
                    "cosine_sim": cs,
                    "grad_norm": grad.norm().item(),
                })
                all_grads.append(grad)

            for move in newly_illegal:
                if move not in grads:
                    continue
                grad = grads[move]
                cs = cosine_sim(probe_dir, grad)
                cosine_sims_illegal.append({
                    "move": int(move),
                    "cosine_sim": cs,
                    "grad_norm": grad.norm().item(),
                })
                all_grads.append(grad)

            # Subspace similarity: how much of probe_dir is in the span
            # of all affected-move gradients?
            sub_sim = subspace_similarity(probe_dir, all_grads) if all_grads else 0.0

            # Mean gradient direction (if we combine all affected moves)
            if all_grads:
                mean_grad = torch.stack(all_grads).mean(dim=0)
                cos_with_mean = cosine_sim(probe_dir, mean_grad)
            else:
                cos_with_mean = 0.0

            r, c, orig_val, target_val = mods[0]
            sample = {
                "game_idx": gi,
                "pos": pos,
                "color": int(color),
                "intervention_type": intervention_type,
                "cell": [int(r), int(c)],
                "orig_val": int(orig_val),
                "target_val": int(target_val),
                "board_state": board_state.flatten().tolist(),
                "scale": float(scale),
                "n_newly_legal": len(newly_legal),
                "n_newly_illegal": len(newly_illegal),
                "cosine_sims_newly_legal": cosine_sims_legal,
                "cosine_sims_newly_illegal": cosine_sims_illegal,
                "cosine_with_mean_grad": cos_with_mean,
                "subspace_similarity": sub_sim,
                "probe_dir_norm": probe_dir.norm().item(),
            }
            samples.append(sample)

    return samples


def plot_results(samples, output_dir):
    """Generate analysis plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all cosine similarities
    all_cs_legal = []
    all_cs_illegal = []
    all_cs_mean = []
    all_sub_sim = []

    for s in samples:
        for entry in s["cosine_sims_newly_legal"]:
            all_cs_legal.append(entry["cosine_sim"])
        for entry in s["cosine_sims_newly_illegal"]:
            all_cs_illegal.append(entry["cosine_sim"])
        all_cs_mean.append(s["cosine_with_mean_grad"])
        all_sub_sim.append(s["subspace_similarity"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Histogram of cosine similarities (per move)
    ax = axes[0, 0]
    bins = np.linspace(-1, 1, 50)
    if all_cs_legal:
        ax.hist(all_cs_legal, bins=bins, alpha=0.6, label=f"Newly legal (n={len(all_cs_legal)})", color="tab:green")
    if all_cs_illegal:
        ax.hist(all_cs_illegal, bins=bins, alpha=0.6, label=f"Newly illegal (n={len(all_cs_illegal)})", color="tab:red")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Cosine similarity (probe dir vs logit gradient)")
    ax.set_ylabel("Count")
    ax.set_title("Per-move cosine similarity")
    ax.legend(fontsize=9)

    # Panel 2: Cosine sim with mean gradient (per sample)
    ax = axes[0, 1]
    flip_cs = [s["cosine_with_mean_grad"] for s in samples if s["intervention_type"] == "flip"]
    add_cs = [s["cosine_with_mean_grad"] for s in samples if s["intervention_type"] == "add"]
    if flip_cs:
        ax.hist(flip_cs, bins=bins, alpha=0.6, label=f"Flip (n={len(flip_cs)})", color="tab:blue")
    if add_cs:
        ax.hist(add_cs, bins=bins, alpha=0.6, label=f"Add (n={len(add_cs)})", color="tab:orange")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Cosine similarity (probe dir vs mean gradient)")
    ax.set_ylabel("Count")
    ax.set_title("Cosine sim with mean logit gradient")
    ax.legend(fontsize=9)

    # Panel 3: Subspace similarity (per sample)
    ax = axes[1, 0]
    flip_ss = [s["subspace_similarity"] for s in samples if s["intervention_type"] == "flip"]
    add_ss = [s["subspace_similarity"] for s in samples if s["intervention_type"] == "add"]
    bins_ss = np.linspace(0, 1, 30)
    if flip_ss:
        ax.hist(flip_ss, bins=bins_ss, alpha=0.6, label=f"Flip (n={len(flip_ss)})", color="tab:blue")
    if add_ss:
        ax.hist(add_ss, bins=bins_ss, alpha=0.6, label=f"Add (n={len(add_ss)})", color="tab:orange")
    ax.set_xlabel("Subspace similarity (frac of probe dir in gradient span)")
    ax.set_ylabel("Count")
    ax.set_title("Subspace similarity")
    ax.legend(fontsize=9)

    # Panel 4: Number of affected moves per sample
    ax = axes[1, 1]
    n_affected = [s["n_newly_legal"] + s["n_newly_illegal"] for s in samples]
    ax.hist(n_affected, bins=range(0, max(n_affected) + 2), alpha=0.7, color="tab:purple", edgecolor="black")
    ax.set_xlabel("Number of moves changing legality")
    ax.set_ylabel("Count")
    ax.set_title("Affected moves per intervention")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cosine_similarity.png"), dpi=150)
    plt.close()

    # Summary stats
    print("\n=== Summary ===")
    if all_cs_legal:
        print(f"Newly legal moves: mean cos={np.mean(all_cs_legal):.4f}, "
              f"median={np.median(all_cs_legal):.4f}, std={np.std(all_cs_legal):.4f}")
    if all_cs_illegal:
        print(f"Newly illegal moves: mean cos={np.mean(all_cs_illegal):.4f}, "
              f"median={np.median(all_cs_illegal):.4f}, std={np.std(all_cs_illegal):.4f}")
    print(f"Mean gradient cos: mean={np.mean(all_cs_mean):.4f}, "
          f"median={np.median(all_cs_mean):.4f}")
    print(f"Subspace similarity: mean={np.mean(all_sub_sim):.4f}, "
          f"median={np.median(all_sub_sim):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", required=True,
                        help="Path to probe checkpoint directory")
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
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

    # Build direction probe from layer 6
    direction_probe = build_direction_probe(probes, layer=LAYER_PROBE, device=device)
    print(f"Direction probe shape: {direction_probe.shape}")

    # Run experiment
    samples = run_experiment(
        model, probes, direction_probe,
        board_seqs_int, board_seqs_string,
        n_games=args.n_games, seed=args.seed, device=device
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "raw_samples.json"), "w") as f:
        json.dump(samples, f, indent=2)

    # Aggregate stats
    results = {}
    for itype in ["flip", "add"]:
        ss = [s for s in samples if s["intervention_type"] == itype]
        if not ss:
            continue
        all_cs = []
        for s in ss:
            for entry in s["cosine_sims_newly_legal"] + s["cosine_sims_newly_illegal"]:
                all_cs.append(entry["cosine_sim"])
        results[itype] = {
            "n_samples": len(ss),
            "n_move_comparisons": len(all_cs),
            "mean_cosine_per_move": float(np.mean(all_cs)) if all_cs else None,
            "median_cosine_per_move": float(np.median(all_cs)) if all_cs else None,
            "std_cosine_per_move": float(np.std(all_cs)) if all_cs else None,
            "mean_cosine_with_mean_grad": float(np.mean([s["cosine_with_mean_grad"] for s in ss])),
            "mean_subspace_sim": float(np.mean([s["subspace_similarity"] for s in ss])),
        }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_results(samples, args.output_dir)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
