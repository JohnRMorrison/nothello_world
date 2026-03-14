"""
Multi-intervention causal experiment on OthelloGPT.

Tests whether OthelloGPT correctly adjusts its predictions when multiple
board cells are simultaneously modified via probe-direction subtraction
in the residual stream.

5 conditions x N interventions (N=1,2,3,5,8):
  1. flip_noninteracting  — flip occupied cells, no shared lines
  2. flip_interacting     — flip occupied cells on shared lines
  3. add_noninteracting   — add pieces to empty cells, no shared lines
  4. add_interacting      — add pieces to empty cells on shared lines
  5. mixed                — random flip/add, log composition

Metrics:
  - Direction accuracy, top-1/5 legal accuracy, legal probability mass
  - Mean logit shift for newly legal/illegal moves
  - Probe cross-talk on non-modified cells
  - Probe accuracy on modified cells

Usage:
  # With scale calibration:
  python multi_intervention.py --calibrate --n-games 200 --output-dir results/

  # With fixed scale:
  python multi_intervention.py --layer-intervene 4 --scale 2.0 --n-games 200
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
from collections import defaultdict
from tqdm import tqdm

# Add parent directory so we can import from top-level modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from data.othello import OthelloBoardState
from generate_variant_games import _flips_vec, DIR_MASK_ALL

# stoi_indices and to_board_label defined inline to avoid importing
# mech_interp_othello_utils (which requires neel_plotly)
stoi_indices = list(range(27)) + list(range(29, 35)) + list(range(37, 64))
alpha = "ABCDEFGH"
def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"

from mingpt.model import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_VALUES = [1, 2, 3, 5, 8]
POS_RANGE = (10, 50)
PROBE_MODE = 2  # mode 2 = all positions (no parity concern)
CENTER_CELLS = {27, 28, 35, 36}  # not in model vocabulary
STOI_SET = set(stoi_indices)  # 60 valid board positions


# ---------------------------------------------------------------------------
# Probe class mapping
# ---------------------------------------------------------------------------
def board_val_to_probe_class(val):
    """Map board value (-1, 0, 1) to probe class index (0, 1, 2)."""
    if val == 0:
        return 0   # empty
    if val == -1:
        return 1   # white
    if val == 1:
        return 2   # black
    raise ValueError(f"Invalid board value: {val}")


# ---------------------------------------------------------------------------
# Model and data loading
# ---------------------------------------------------------------------------
def load_model_and_data(probe_path, ckpt_path, device="cuda"):
    """Load mingpt OthelloGPT model, linear probe, and game data."""
    mconf = GPTConfig(
        vocab_size=61, block_size=59,
        n_layer=8, n_head=8, n_embd=512,
    )
    model = GPT(mconf)

    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.join(script_dir, "..")

    ckpt_full = os.path.join(parent_dir, ckpt_path)
    model.load_state_dict(torch.load(ckpt_full, map_location=device))
    model.to(device)
    model.eval()

    probe_full_path = os.path.join(script_dir, probe_path)
    linear_probe = torch.load(probe_full_path, map_location=device)
    # Shape: (3, 512, 8, 8, 3) — modes x d_model x rows x cols x options

    board_seqs_int = torch.load(
        os.path.join(script_dir, "board_seqs_int.pth"), map_location="cpu"
    )
    board_seqs_string = torch.load(
        os.path.join(script_dir, "board_seqs_string.pth"), map_location="cpu"
    )

    return model, linear_probe, board_seqs_int, board_seqs_string


# ---------------------------------------------------------------------------
# Board replay
# ---------------------------------------------------------------------------
def replay_to_position(game_str, pos):
    """Replay game to position pos, return (board_state_2d, next_color)."""
    board = OthelloBoardState()
    for i in range(pos + 1):
        board.umpire(game_str[i].item() if hasattr(game_str[i], 'item') else int(game_str[i]))
    return np.copy(board.state), board.next_hand_color


# ---------------------------------------------------------------------------
# Counterfactual legal moves
# ---------------------------------------------------------------------------
def compute_legal_moves(board_2d, color):
    """Compute standard Othello legal moves from 8x8 board state."""
    flat = board_2d.flatten().astype(np.int8)
    n_flips, _ = _flips_vec(flat, color, DIR_MASK_ALL)
    empty = (flat == 0)
    regular = np.where(empty & (n_flips > 0))[0].tolist()
    if regular:
        return set(regular)
    # Forfeit
    opp_flips, _ = _flips_vec(flat, -color, DIR_MASK_ALL)
    return set(np.where(empty & (opp_flips > 0))[0].tolist())


def compute_counterfactual_legal(board_2d, modifications, color):
    """Apply modifications to board, compute legal moves.

    modifications: list of (r, c, new_val)
    """
    modified = board_2d.copy()
    for (r, c, new_val) in modifications:
        modified[r, c] = new_val
    return compute_legal_moves(modified, color)


# ---------------------------------------------------------------------------
# Cell selection helpers
# ---------------------------------------------------------------------------
def shares_line(r1, c1, r2, c2):
    """Check if two cells share a row, column, or diagonal."""
    return r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2)


def _no_line_conflict(r, c, selected):
    """Check that (r, c) shares no line with any selected cell."""
    for (sr, sc, _, _) in selected:
        if shares_line(r, c, sr, sc):
            return False
    return True


def _has_line_with_selected(r, c, selected):
    """Check that (r, c) shares a line with at least one selected cell."""
    for (sr, sc, _, _) in selected:
        if shares_line(r, c, sr, sc):
            return True
    return False


# -- Condition 1: flip, non-interacting --
def select_flip_noninteracting(board_state, n, rng):
    """Select n occupied cells with no shared row/col/diagonal.

    Returns list of (r, c, orig_val, target_val) or None if not enough cells.
    """
    occupied = [(r, c) for r in range(8) for c in range(8)
                if board_state[r, c] != 0 and (r * 8 + c) not in CENTER_CELLS]
    rng.shuffle(occupied)
    selected = []
    for (r, c) in occupied:
        if _no_line_conflict(r, c, selected):
            orig = int(board_state[r, c])
            selected.append((r, c, orig, -orig))
            if len(selected) == n:
                return selected
    return None


# -- Condition 2: flip, interacting --
def select_flip_interacting(board_state, n, rng):
    """Select n occupied cells where each successive cell shares a line."""
    occupied = [(r, c) for r in range(8) for c in range(8)
                if board_state[r, c] != 0 and (r * 8 + c) not in CENTER_CELLS]
    if not occupied:
        return None
    seed = occupied[rng.randint(0, len(occupied) - 1)]
    orig = int(board_state[seed[0], seed[1]])
    selected = [(seed[0], seed[1], orig, -orig)]

    for _ in range(n - 1):
        candidates = [(r, c) for (r, c) in occupied
                       if (r, c) not in {(s[0], s[1]) for s in selected}
                       and _has_line_with_selected(r, c, selected)]
        if not candidates:
            return None
        pick = candidates[rng.randint(0, len(candidates) - 1)]
        orig = int(board_state[pick[0], pick[1]])
        selected.append((pick[0], pick[1], orig, -orig))
    return selected


# -- Condition 3: add, non-interacting --
def select_add_noninteracting(board_state, n, rng):
    """Select n empty cells with no shared row/col/diagonal."""
    empty = [(r, c) for r in range(8) for c in range(8)
             if board_state[r, c] == 0 and (r * 8 + c) not in CENTER_CELLS]
    rng.shuffle(empty)
    selected = []
    for (r, c) in empty:
        if _no_line_conflict(r, c, selected):
            color = rng.choice([1, -1])
            selected.append((r, c, 0, color))
            if len(selected) == n:
                return selected
    return None


# -- Condition 4: add, interacting --
def select_add_interacting(board_state, n, rng):
    """Select n empty cells where each successive cell shares a line."""
    empty = [(r, c) for r in range(8) for c in range(8)
             if board_state[r, c] == 0 and (r * 8 + c) not in CENTER_CELLS]
    if not empty:
        return None

    # Seed: any empty cell
    seed = empty[rng.randint(0, len(empty) - 1)]
    color = rng.choice([1, -1])
    selected = [(seed[0], seed[1], 0, color)]

    for _ in range(n - 1):
        candidates = [(r, c) for (r, c) in empty
                       if (r, c) not in {(s[0], s[1]) for s in selected}
                       and _has_line_with_selected(r, c, selected)]
        if not candidates:
            return None
        pick = candidates[rng.randint(0, len(candidates) - 1)]
        color = rng.choice([1, -1])
        selected.append((pick[0], pick[1], 0, color))
    return selected


# -- Condition 5: mixed --
def select_mixed(board_state, n, rng):
    """Randomly choose flip or add for each slot. No interaction constraint.

    Returns (modifications, (n_flips, n_adds)) or (None, (0, 0)).
    """
    occupied = [(r, c) for r in range(8) for c in range(8)
                if board_state[r, c] != 0 and (r * 8 + c) not in CENTER_CELLS]
    empty = [(r, c) for r in range(8) for c in range(8)
             if board_state[r, c] == 0 and (r * 8 + c) not in CENTER_CELLS]
    rng.shuffle(occupied)
    rng.shuffle(empty)

    selected = []
    used = set()
    occ_idx, emp_idx = 0, 0
    n_flips, n_adds = 0, 0

    for _ in range(n):
        action = rng.choice(["flip", "add"])
        if action == "flip" and occ_idx < len(occupied):
            r, c = occupied[occ_idx]; occ_idx += 1
            selected.append((r, c, int(board_state[r, c]), -int(board_state[r, c])))
            n_flips += 1
        elif emp_idx < len(empty):
            r, c = empty[emp_idx]; emp_idx += 1
            selected.append((r, c, 0, rng.choice([1, -1])))
            n_adds += 1
        elif occ_idx < len(occupied):
            r, c = occupied[occ_idx]; occ_idx += 1
            selected.append((r, c, int(board_state[r, c]), -int(board_state[r, c])))
            n_flips += 1
        else:
            return None, (n_flips, n_adds)

    return selected, (n_flips, n_adds)


# ---------------------------------------------------------------------------
# Intervention hooks
# ---------------------------------------------------------------------------
def compute_flip_dirs(linear_probe, modifications, mode=PROBE_MODE):
    """Compute probe-direction vectors for each modification."""
    flip_dirs = []
    for (r, c, orig_val, target_val) in modifications:
        current_class = board_val_to_probe_class(orig_val)
        target_class = board_val_to_probe_class(target_val)
        flip_dir = linear_probe[mode, :, r, c, target_class] - \
                   linear_probe[mode, :, r, c, current_class]
        flip_dirs.append(flip_dir)
    return flip_dirs


def apply_intervention(x, flip_dirs, pos, scale):
    """Apply probe-direction subtraction to activation tensor x in-place.

    x: (batch, seq, d_model)
    """
    for flip_dir in flip_dirs:
        coeff = x[0, pos] @ flip_dir / flip_dir.norm()
        x[0, pos] -= scale * coeff * flip_dir / flip_dir.norm()
    return x


def run_with_intervention(model, input_tokens, modifications, linear_probe,
                          pos, scale, layer_intervene, layer_probe,
                          mode=PROBE_MODE, device="cuda"):
    """Run model with intervention, return (logits_at_pos, resid_at_probe_layer).

    Uses mingpt: manually split forward pass at layer_intervene,
    apply intervention, continue forward, capture activations at layer_probe.
    """
    flip_dirs = compute_flip_dirs(linear_probe, modifications, mode)

    # Forward through embedding + layers 0..layer_intervene-1
    b, t = input_tokens.size()
    token_emb = model.tok_emb(input_tokens)
    pos_emb = model.pos_emb[:, :t, :]
    x = model.drop(token_emb + pos_emb)

    for block in model.blocks[:layer_intervene]:
        x = block(x)

    # Apply intervention at layer_intervene output
    x = apply_intervention(x, flip_dirs, pos, scale)

    # Continue through remaining layers, capture at layer_probe
    resid_at_probe = None
    for i, block in enumerate(model.blocks[layer_intervene:], start=layer_intervene):
        x = block(x)
        if i == layer_probe - 1:
            # Capture after this block (= resid_post for layer_probe)
            # layer_probe is 0-indexed: block i produces resid_post layer i
            resid_at_probe = x[0, pos].detach().clone()

    # If layer_probe is the last layer or beyond what we captured
    if resid_at_probe is None:
        resid_at_probe = x[0, pos].detach().clone()

    # Final layernorm + head
    x = model.ln_f(x)
    logits = model.head(x)  # (B, T, vocab)

    return logits[0, pos], resid_at_probe


def _forward_with_resid(model, input_tokens, pos, layer_probe):
    """Forward pass through mingpt, returning logits and residual at layer_probe.

    Returns (logits_at_pos, resid_at_probe_layer).
    """
    b, t = input_tokens.size()
    token_emb = model.tok_emb(input_tokens)
    pos_emb = model.pos_emb[:, :t, :]
    x = model.drop(token_emb + pos_emb)

    resid_at_probe = None
    for i, block in enumerate(model.blocks):
        x = block(x)
        if i == layer_probe - 1:
            resid_at_probe = x[0, pos].detach().clone()

    if resid_at_probe is None:
        resid_at_probe = x[0, pos].detach().clone()

    x = model.ln_f(x)
    logits = model.head(x)
    return logits[0, pos], resid_at_probe


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------
def measure_logit_metrics(original_logits, intervened_logits,
                          original_legal, counterfactual_legal):
    """Compute all logit-based metrics.

    Returns dict with: direction_acc, top1_legal, top5_legal_frac,
    legal_prob_mass, mean_shift_newly_illegal, mean_shift_newly_legal.
    """
    newly_legal = counterfactual_legal - original_legal
    newly_illegal = original_legal - counterfactual_legal

    # Map model logits to per-cell values (60 valid cells)
    # Model output: index 0 = pass, indices 1-60 = stoi_indices
    orig_cell = torch.full((64,), float('-inf'), device=original_logits.device)
    orig_cell[stoi_indices] = original_logits[1:61]

    intv_cell = torch.full((64,), float('-inf'), device=intervened_logits.device)
    intv_cell[stoi_indices] = intervened_logits[1:61]

    result = {}

    # 1. Direction accuracy
    n_changed = len(newly_legal) + len(newly_illegal)
    if n_changed > 0:
        correct = 0
        for cell in newly_legal:
            if cell in STOI_SET and intv_cell[cell] > orig_cell[cell]:
                correct += 1
            elif cell not in STOI_SET:
                n_changed -= 1  # exclude center cells
        for cell in newly_illegal:
            if cell in STOI_SET and intv_cell[cell] < orig_cell[cell]:
                correct += 1
            elif cell not in STOI_SET:
                n_changed -= 1
        result["direction_acc"] = correct / n_changed if n_changed > 0 else None
    else:
        result["direction_acc"] = None
    result["n_changed"] = n_changed

    # 2-3. Top-1 and top-5 legal accuracy
    # Use only valid board positions (exclude pass token)
    valid_logits = intervened_logits[1:61]  # 60 values for stoi_indices
    sorted_indices = valid_logits.argsort(descending=True)
    top_cells = [stoi_indices[idx] for idx in sorted_indices[:5].tolist()]

    cf_legal_stoi = counterfactual_legal & STOI_SET
    result["top1_legal"] = 1.0 if top_cells[0] in cf_legal_stoi else 0.0
    top5_legal = sum(1 for c in top_cells if c in cf_legal_stoi)
    result["top5_legal_frac"] = top5_legal / 5.0

    # 4. Legal probability mass
    probs = torch.softmax(intervened_logits[1:61], dim=0)  # over 60 valid cells
    legal_mask = torch.zeros(60, device=probs.device)
    for i, cell in enumerate(stoi_indices):
        if cell in counterfactual_legal:
            legal_mask[i] = 1.0
    result["legal_prob_mass"] = (probs * legal_mask).sum().item()

    # 5-6. Mean logit shifts
    shifts_illegal = []
    shifts_legal = []
    for cell in newly_illegal:
        if cell in STOI_SET:
            shifts_illegal.append((intv_cell[cell] - orig_cell[cell]).item())
    for cell in newly_legal:
        if cell in STOI_SET:
            shifts_legal.append((intv_cell[cell] - orig_cell[cell]).item())

    result["mean_shift_newly_illegal"] = np.mean(shifts_illegal) if shifts_illegal else None
    result["mean_shift_newly_legal"] = np.mean(shifts_legal) if shifts_legal else None

    return result


def measure_probe_crosstalk(original_resid, intervened_resid, linear_probe,
                            modifications, mode=PROBE_MODE):
    """Mean absolute change in probe logits for non-modified cells."""
    modified_set = {(m[0], m[1]) for m in modifications}

    orig_probe = torch.einsum("d, d r c o -> r c o", original_resid,
                              linear_probe[mode])
    intv_probe = torch.einsum("d, d r c o -> r c o", intervened_resid,
                              linear_probe[mode])

    diff = (intv_probe - orig_probe).abs()  # (8, 8, 3)

    mask = torch.ones(8, 8, dtype=torch.bool, device=diff.device)
    for (r, c, _, _) in modifications:
        mask[r, c] = False

    if mask.sum() == 0:
        return 0.0
    return diff[mask].mean().item()


def measure_probe_accuracy(intervened_resid, linear_probe, modifications,
                           mode=PROBE_MODE):
    """Fraction of modified cells whose probe argmax matches target state."""
    probe_out = torch.einsum("d, d r c o -> r c o", intervened_resid,
                             linear_probe[mode])

    correct = 0
    for (r, c, orig_val, target_val) in modifications:
        target_class = board_val_to_probe_class(target_val)
        predicted_class = probe_out[r, c].argmax().item()
        if predicted_class == target_class:
            correct += 1

    return correct / len(modifications)


# ---------------------------------------------------------------------------
# Scale calibration
# ---------------------------------------------------------------------------
def calibrate_scale(model, linear_probe, board_seqs_int, board_seqs_string,
                    layers=(3, 4, 5, 6), scales=(0.5, 1, 2, 3, 4, 6, 8),
                    n_games=50, device="cuda"):
    """Find optimal (layer_intervene, scale) via N=1 flip interventions.

    Returns (best_layer, best_scale, results_dict).
    """
    print("=== Scale Calibration ===")
    rng = random.Random(42)
    results = {}

    for layer in layers:
        for scale in scales:
            direction_accs = []
            for gi in tqdm(range(n_games), desc=f"L{layer} s{scale}", leave=False):
                game_str = board_seqs_string[gi]
                game_int = board_seqs_int[gi]

                pos = rng.randint(POS_RANGE[0], POS_RANGE[1])
                board_state, color = replay_to_position(game_str, pos)
                original_legal = compute_legal_moves(board_state, color)

                # Select one cell to flip
                mods = select_flip_noninteracting(board_state, 1, rng)
                if mods is None:
                    continue

                # Counterfactual
                cf_mods = [(r, c, tgt) for (r, c, _, tgt) in mods]
                cf_legal = compute_counterfactual_legal(board_state, cf_mods, color)

                # Run intervention
                input_tokens = game_int[:pos + 1].unsqueeze(0).to(device)
                with torch.no_grad():
                    orig_logits = model(input_tokens)[0][0, pos]
                    intv_logits, _ = run_with_intervention(
                        model, input_tokens, mods, linear_probe,
                        pos, scale, layer, 6, device=device
                    )

                metrics = measure_logit_metrics(orig_logits, intv_logits,
                                                original_legal, cf_legal)
                if metrics["direction_acc"] is not None:
                    direction_accs.append(metrics["direction_acc"])

            mean_acc = np.mean(direction_accs) if direction_accs else 0.0
            results[(layer, scale)] = mean_acc
            print(f"  Layer {layer}, scale {scale}: direction_acc = {mean_acc:.3f} "
                  f"(n={len(direction_accs)})")

    best = max(results, key=results.get)
    print(f"\nBest: layer={best[0]}, scale={best[1]}, "
          f"direction_acc={results[best]:.3f}")
    return best[0], best[1], results


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
CONDITIONS = [
    ("flip_noninteracting", select_flip_noninteracting),
    ("flip_interacting", select_flip_interacting),
    ("add_noninteracting", select_add_noninteracting),
    ("add_interacting", select_add_interacting),
]


def run_experiment(model, linear_probe, board_seqs_int, board_seqs_string,
                   layer_intervene, layer_probe, scale,
                   n_games=200, n_values=None, seed=42, device="cuda"):
    """Run the full multi-intervention experiment."""
    if n_values is None:
        n_values = N_VALUES

    rng = random.Random(seed)

    # Initialize results storage
    results = {cond: {str(n): [] for n in n_values}
               for cond, _ in CONDITIONS}
    results["mixed"] = {str(n): [] for n in n_values}

    print(f"\n=== Multi-Intervention Experiment ===")
    print(f"Layer intervene: {layer_intervene}, Layer probe: {layer_probe}, "
          f"Scale: {scale}")
    print(f"Games: {n_games}, N values: {n_values}")

    for gi in tqdm(range(n_games), desc="Games"):
        game_str = board_seqs_string[gi]
        game_int = board_seqs_int[gi]

        # Sample a mid-game position
        pos = rng.randint(POS_RANGE[0], min(POS_RANGE[1], 58))
        board_state, color = replay_to_position(game_str, pos)
        original_legal = compute_legal_moves(board_state, color)

        # Get original model output + residual at probe layer
        input_tokens = game_int[:pos + 1].unsqueeze(0).to(device)
        with torch.no_grad():
            orig_logits_at_pos, orig_resid = _forward_with_resid(
                model, input_tokens, pos, layer_probe
            )

        for n in n_values:
            # Standard conditions
            for cond_name, select_fn in CONDITIONS:
                mods = select_fn(board_state, n, rng)
                if mods is None:
                    continue

                cf_mods = [(r, c, tgt) for (r, c, _, tgt) in mods]
                cf_legal = compute_counterfactual_legal(
                    board_state, cf_mods, color
                )

                with torch.no_grad():
                    intv_logits, intv_resid = run_with_intervention(
                        model, input_tokens, mods, linear_probe,
                        pos, scale, layer_intervene, layer_probe,
                        device=device
                    )

                logit_metrics = measure_logit_metrics(
                    orig_logits_at_pos, intv_logits,
                    original_legal, cf_legal
                )
                crosstalk = measure_probe_crosstalk(
                    orig_resid, intv_resid, linear_probe, mods
                )
                probe_acc = measure_probe_accuracy(
                    intv_resid, linear_probe, mods
                )

                sample = {
                    **logit_metrics,
                    "crosstalk": crosstalk,
                    "probe_acc": probe_acc,
                }
                results[cond_name][str(n)].append(sample)

            # Mixed condition
            mods, (n_flips, n_adds) = select_mixed(board_state, n, rng)
            if mods is not None:
                cf_mods = [(r, c, tgt) for (r, c, _, tgt) in mods]
                cf_legal = compute_counterfactual_legal(
                    board_state, cf_mods, color
                )

                with torch.no_grad():
                    intv_logits, intv_resid = run_with_intervention(
                        model, input_tokens, mods, linear_probe,
                        pos, scale, layer_intervene, layer_probe,
                        device=device
                    )

                logit_metrics = measure_logit_metrics(
                    orig_logits_at_pos, intv_logits,
                    original_legal, cf_legal
                )
                crosstalk = measure_probe_crosstalk(
                    orig_resid, intv_resid, linear_probe, mods
                )
                probe_acc = measure_probe_accuracy(
                    intv_resid, linear_probe, mods
                )

                sample = {
                    **logit_metrics,
                    "crosstalk": crosstalk,
                    "probe_acc": probe_acc,
                    "n_flips": n_flips,
                    "n_adds": n_adds,
                }
                results["mixed"][str(n)].append(sample)

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_results(results):
    """Aggregate per-sample results into means and stds."""
    aggregated = {}
    for cond in results:
        aggregated[cond] = {}
        for n_str, samples in results[cond].items():
            if not samples:
                aggregated[cond][n_str] = {"n_samples": 0}
                continue

            agg = {"n_samples": len(samples)}

            # Scalar metrics
            for key in ["direction_acc", "top1_legal", "top5_legal_frac",
                         "legal_prob_mass", "mean_shift_newly_illegal",
                         "mean_shift_newly_legal", "crosstalk", "probe_acc"]:
                vals = [s[key] for s in samples if s.get(key) is not None]
                if vals:
                    agg[key] = float(np.mean(vals))
                    agg[key + "_std"] = float(np.std(vals))
                else:
                    agg[key] = None

            agg["n_changed_mean"] = float(np.mean([s["n_changed"] for s in samples]))

            # Mixed: composition breakdown
            if cond == "mixed" and "n_flips" in samples[0]:
                compositions = defaultdict(list)
                for s in samples:
                    comp_key = f"{s['n_flips']}f_{s['n_adds']}a"
                    compositions[comp_key].append(s)
                agg["compositions"] = {}
                for comp_key, comp_samples in compositions.items():
                    comp_agg = {"n_samples": len(comp_samples)}
                    for key in ["direction_acc", "top1_legal", "legal_prob_mass",
                                 "crosstalk", "probe_acc"]:
                        vals = [s[key] for s in comp_samples if s.get(key) is not None]
                        if vals:
                            comp_agg[key] = float(np.mean(vals))
                    agg["compositions"][comp_key] = comp_agg

            aggregated[cond][n_str] = agg

    return aggregated


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
COND_COLORS = {
    "flip_noninteracting": "#1f77b4",
    "flip_interacting": "#ff7f0e",
    "add_noninteracting": "#2ca02c",
    "add_interacting": "#d62728",
    "mixed": "#9467bd",
}
COND_LABELS = {
    "flip_noninteracting": "Flip (non-int.)",
    "flip_interacting": "Flip (interact.)",
    "add_noninteracting": "Add (non-int.)",
    "add_interacting": "Add (interact.)",
    "mixed": "Mixed",
}


def plot_results(aggregated, n_values, output_dir):
    """Generate 6 summary plots."""
    metrics_to_plot = [
        ("direction_acc", "Direction Accuracy", "Fraction correct"),
        ("legal_prob_mass", "Legal Probability Mass", "Probability"),
        ("top1_legal", "Top-1 Legal Accuracy", "Fraction legal"),
        ("crosstalk", "Probe Cross-talk", "Mean |change| (non-modified cells)"),
        ("probe_acc", "Probe Accuracy (Modified Cells)", "Fraction correct"),
    ]

    for metric_key, title, ylabel in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cond in COND_COLORS:
            xs, ys, errs = [], [], []
            for n in n_values:
                data = aggregated.get(cond, {}).get(str(n), {})
                val = data.get(metric_key)
                if val is not None:
                    xs.append(n)
                    ys.append(val)
                    errs.append(data.get(metric_key + "_std", 0))
            if xs:
                ax.errorbar(xs, ys, yerr=errs, marker='o', label=COND_LABELS[cond],
                            color=COND_COLORS[cond], capsize=3)
        ax.set_xlabel("Number of interventions (N)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xticks(n_values)
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{metric_key}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname}")

    # Special plot: mean logit shifts (two lines per condition)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in COND_COLORS:
        xs_il, ys_il = [], []
        xs_le, ys_le = [], []
        for n in n_values:
            data = aggregated.get(cond, {}).get(str(n), {})
            val_il = data.get("mean_shift_newly_illegal")
            val_le = data.get("mean_shift_newly_legal")
            if val_il is not None:
                xs_il.append(n)
                ys_il.append(val_il)
            if val_le is not None:
                xs_le.append(n)
                ys_le.append(val_le)
        if xs_il:
            ax.plot(xs_il, ys_il, marker='v', linestyle='--',
                    color=COND_COLORS[cond], alpha=0.7,
                    label=f"{COND_LABELS[cond]} (illegal)")
        if xs_le:
            ax.plot(xs_le, ys_le, marker='^', linestyle='-',
                    color=COND_COLORS[cond], alpha=0.7,
                    label=f"{COND_LABELS[cond]} (legal)")
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Number of interventions (N)")
    ax.set_ylabel("Mean logit shift")
    ax.set_title("Mean Logit Shift (Newly Legal vs Newly Illegal)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xticks(n_values)
    plt.tight_layout()
    fname = os.path.join(output_dir, "logit_shifts.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multi-intervention causal experiment on OthelloGPT"
    )
    parser.add_argument("--probe-path", default="main_linear_probe.pth",
                        help="Path to linear probe (relative to script dir)")
    parser.add_argument("--ckpt", default="ckpts/gpt_synthetic.ckpt",
                        help="Path to model checkpoint (relative to repo root)")
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--layer-intervene", type=int, default=None,
                        help="Intervention layer (default: calibrate)")
    parser.add_argument("--layer-probe", type=int, default=6,
                        help="Probe layer (default: 6)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Intervention scale (default: calibrate)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run scale calibration before experiment")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Only run calibration, skip main experiment")
    parser.add_argument("--n-values", type=str, default="1,2,3,5,8",
                        help="Comma-separated N values")
    parser.add_argument("--output-dir", default="multi_intervention_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model and data
    print("Loading model, probe, and game data...")
    model, linear_probe, board_seqs_int, board_seqs_string = \
        load_model_and_data(args.probe_path, args.ckpt, device=args.device)

    # Scale calibration
    layer_intervene = args.layer_intervene
    scale = args.scale
    calibration_results = None

    if args.calibrate or args.calibrate_only or layer_intervene is None or scale is None:
        layer_intervene, scale, calibration_results = calibrate_scale(
            model, linear_probe, board_seqs_int, board_seqs_string,
            n_games=min(args.n_games, 50), device=args.device
        )
        # Save calibration
        cal_path = os.path.join(args.output_dir, "calibration.json")
        cal_data = {
            "best_layer": layer_intervene,
            "best_scale": scale,
            "results": {f"L{k[0]}_s{k[1]}": v
                        for k, v in calibration_results.items()},
        }
        with open(cal_path, "w") as f:
            json.dump(cal_data, f, indent=2)
        print(f"Calibration saved to {cal_path}")

        if args.calibrate_only:
            return

    # Use provided values if specified (override calibration)
    if args.layer_intervene is not None:
        layer_intervene = args.layer_intervene
    if args.scale is not None:
        scale = args.scale

    # Run experiment
    raw_results = run_experiment(
        model, linear_probe, board_seqs_int, board_seqs_string,
        layer_intervene, args.layer_probe, scale,
        n_games=args.n_games, n_values=n_values,
        seed=args.seed, device=args.device
    )

    # Aggregate
    aggregated = aggregate_results(raw_results)

    # Save
    output = {
        "config": {
            "n_games": args.n_games,
            "layer_intervene": layer_intervene,
            "layer_probe": args.layer_probe,
            "scale": scale,
            "n_values": n_values,
            "seed": args.seed,
        },
        "results": aggregated,
    }
    if calibration_results is not None:
        output["calibration"] = {
            "best_layer": layer_intervene,
            "best_scale": scale,
        }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print("\n=== Summary ===")
    for cond in list(COND_LABELS.keys()):
        print(f"\n{COND_LABELS[cond]}:")
        for n in n_values:
            data = aggregated.get(cond, {}).get(str(n), {})
            if data.get("n_samples", 0) == 0:
                print(f"  N={n}: no samples")
                continue
            da = data.get("direction_acc")
            t1 = data.get("top1_legal")
            lpm = data.get("legal_prob_mass")
            ct = data.get("crosstalk")
            pa = data.get("probe_acc")
            ns = data.get("n_samples", 0)
            da_s = f"{da:.3f}" if da is not None else "N/A"
            t1_s = f"{t1:.3f}" if t1 is not None else "N/A"
            lpm_s = f"{lpm:.3f}" if lpm is not None else "N/A"
            ct_s = f"{ct:.4f}" if ct is not None else "N/A"
            pa_s = f"{pa:.3f}" if pa is not None else "N/A"
            print(f"  N={n}: dir_acc={da_s}  top1={t1_s}  "
                  f"legal_mass={lpm_s}  xtalk={ct_s}  "
                  f"probe={pa_s}  (n={ns})")

    # Plot
    print("\nGenerating plots...")
    plot_results(aggregated, n_values, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
