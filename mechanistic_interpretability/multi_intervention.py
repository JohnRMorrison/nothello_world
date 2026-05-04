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
import torch.nn as nn

# NOTE: layer_propagation imports are done lazily to avoid circular import
# (layer_propagation.py imports from this module)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_VALUES = [1, 2, 3]
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


def apply_intervention(x, flip_dirs, pos, scale, per_cell_scales=None):
    """Apply probe-direction subtraction to activation tensor x in-place.

    x: (batch, seq, d_model)
    per_cell_scales: if provided, list of per-cell scales (one per flip_dir).
                     Each per-cell scale is MULTIPLIED by `scale` (default 1.0
                     when only per-cell scales are used). This lets you sweep
                     a scale multiplier on top of the cd0 calibrated direction.
    """
    for i, flip_dir in enumerate(flip_dirs):
        if per_cell_scales is not None:
            s = per_cell_scales[i] * scale
        else:
            s = scale
        coeff = x[0, pos] @ flip_dir / flip_dir.norm()
        x[0, pos] -= s * coeff * flip_dir / flip_dir.norm()
    return x


def compute_per_cell_scales(h, linear_probe, modifications, mode=PROBE_MODE):
    """Compute the minimal scale per cell that flips the probe's argmax.

    Uses binary search: for each modification, find the smallest s in [0, 10]
    such that after h' = h - s * (h . d_hat) * d_hat, the probe predicts
    target_class for that cell.

    Returns list of scales (one per modification).
    """
    scales = []
    for (r, c, orig_val, target_val) in modifications:
        current_class = board_val_to_probe_class(orig_val)
        target_class = board_val_to_probe_class(target_val)

        W = linear_probe[mode, :, r, c, :]  # (d_model, 3)
        flip_dir = W[:, target_class] - W[:, current_class]
        d_hat = flip_dir / flip_dir.norm()
        coeff = h @ d_hat

        def probe_pred_at_scale(s):
            h_mod = h - s * coeff * d_hat
            logits = W.T @ h_mod  # (3,)
            return logits.argmax().item()

        # Binary search for minimal s that flips to target_class
        lo, hi = 0.0, 10.0
        # First check if any scale works
        if probe_pred_at_scale(hi) != target_class:
            scales.append(hi)  # can't flip — use max
            continue
        if probe_pred_at_scale(0.0) == target_class:
            scales.append(0.5)  # already correct — use small nudge
            continue

        for _ in range(30):  # ~30 iterations gives precision < 1e-8
            mid = (lo + hi) / 2
            if probe_pred_at_scale(mid) == target_class:
                hi = mid
            else:
                lo = mid

        # Use the found scale + 10% margin to ensure clean flip
        scales.append(min(hi * 1.1, 10.0))

    return scales


def compute_prefix_activations(model, input_tokens, layer_intervene):
    """Forward through embedding + layers 0..layer_intervene-1.

    Returns the activation tensor at the intervention point. This can be
    cached and reused across multiple interventions on the same input.
    """
    b, t = input_tokens.size()
    token_emb = model.tok_emb(input_tokens)
    pos_emb = model.pos_emb[:, :t, :]
    x = model.drop(token_emb + pos_emb)

    for block in model.blocks[:layer_intervene]:
        x = block(x)

    return x


def run_with_cascade_intervention(model, input_tokens, modifications,
                                  pos, start_layer, downstream_probes,
                                  device="cuda", prefix_acts=None):
    """Run model with cascading interventions at every layer from start_layer.

    At each layer L (from start_layer to 7):
      1. Build direction tensor from layer L's probe
      2. Calibrate per-cell scales locally (layer L's probe)
      3. Apply intervention to residual stream
      4. Pass through block L

    This follows Li et al.'s approach of intervening at every layer.

    Returns (logits_at_pos, resid_at_start_layer).
    """
    from layer_propagation import (convert_probe_to_direction_tensor,
                                   compute_per_cell_scales_local,
                                   compute_flip_dirs_from_direction_probe)

    pm = 0 if pos % 2 == 0 else 1
    parity_str = "even" if pm == 0 else "odd"

    # Use cached prefix or compute from scratch
    if prefix_acts is not None:
        x = prefix_acts.clone()
    else:
        x = compute_prefix_activations(model, input_tokens, start_layer)

    resid_at_start = None

    # Cascade: intervene at each layer from start_layer to 7
    for layer_idx in range(start_layer, len(model.blocks)):
        native_layer = layer_idx  # resid_post of previous block = input to this block
        # layer_idx is the block index; native_layer for probe = layer_idx
        # But the probe at "layer L" reads resid_post of block L-1,
        # which is the current x before passing through block layer_idx.
        # The probe naming: layer 4 probe reads resid_post of block 3.
        # So when we're about to run block layer_idx, x is resid_post of
        # block (layer_idx - 1), which is what layer layer_idx probe reads.

        probe_key = (layer_idx, parity_str)
        if probe_key not in downstream_probes:
            # No probe for this layer, just run the block
            x = model.blocks[layer_idx](x)
            continue

        probe = downstream_probes[probe_key]
        direction_tensor = convert_probe_to_direction_tensor(probe, device)
        # Stack into (2, 512, 8, 8, 3) with same tensor for both parities
        # (we only use the correct parity via pm)
        dir_probe = torch.zeros(2, 512, 8, 8, 3, device=device)
        dir_probe[pm] = direction_tensor

        # Local calibration at this layer
        h = x[0, pos].detach()
        per_cell_scales = compute_per_cell_scales_local(
            h, probe, dir_probe, modifications, pm
        )

        # Compute flip directions from this layer's probe
        flip_dirs = compute_flip_dirs_from_direction_probe(
            dir_probe, modifications, pm
        )

        # Apply intervention
        x = apply_intervention(x, flip_dirs, pos, scale=1.0,
                               per_cell_scales=per_cell_scales)

        if resid_at_start is None:
            resid_at_start = x[0, pos].detach().clone()

        # Pass through this block
        x = model.blocks[layer_idx](x)

    # Final layernorm + head
    x = model.ln_f(x)
    logits = model.head(x)

    return logits[0, pos], resid_at_start


def run_with_intervention(model, input_tokens, modifications, linear_probe,
                          pos, scale, layer_intervene, layer_probe,
                          mode=PROBE_MODE, device="cuda",
                          per_cell_calibrate=False,
                          prefix_acts=None,
                          direction_probe=None, downstream_probes=None,
                          cal_depth=0):
    """Run model with intervention, return (logits_at_pos, resid_at_probe_layer).

    Uses mingpt: manually split forward pass at layer_intervene,
    apply intervention, continue forward, capture activations at layer_probe.

    When per_cell_calibrate=True with direction_probe and downstream_probes:
      - cal_depth=0: LOCAL calibration at the intervention layer
      - cal_depth>0: calibrate at intervention_layer + cal_depth downstream
      - Direction probe should be built from the native layer's probe
      - downstream_probes provides nn.Linear probes for calibration

    direction_probe: (2, 512, 8, 8, 3) tensor for even/odd flip directions
    downstream_probes: dict mapping (layer, parity_str) -> nn.Linear
    cal_depth: 0 = local, 1 = +1 layer downstream, 2 = +2 layers, etc.
    """
    # Use cached prefix or compute from scratch
    if prefix_acts is not None:
        x = prefix_acts.clone()
    else:
        x = compute_prefix_activations(model, input_tokens, layer_intervene)

    # Compute per-cell scales if requested
    per_cell_scales = None
    if per_cell_calibrate:
        if direction_probe is not None and downstream_probes is not None:
            pm = 0 if pos % 2 == 0 else 1
            parity_str = "even" if pm == 0 else "odd"
            native_layer = layer_intervene - 1  # resid_post of this block
            from layer_propagation import (compute_per_cell_scales_local,
                                           compute_per_cell_scales_downstream,
                                           compute_flip_dirs_from_direction_probe)

            if cal_depth == 0:
                # Local calibration at the intervention layer using native probe
                native_probe = downstream_probes[(native_layer, parity_str)]
                h = x[0, pos].detach()
                per_cell_scales = compute_per_cell_scales_local(
                    h, native_probe, direction_probe,
                    modifications, pm
                )
            else:
                # Downstream calibration
                cal_layer = native_layer + cal_depth
                cal_layer = min(cal_layer, 7)  # clamp to max layer
                target_probe = downstream_probes[(cal_layer, parity_str)]
                per_cell_scales = compute_per_cell_scales_downstream(
                    model, x, direction_probe, target_probe,
                    modifications, pos, layer_intervene, cal_layer, pm
                )

            # Use direction probe for flip dirs (parity-aware)
            flip_dirs = compute_flip_dirs_from_direction_probe(
                direction_probe, modifications, pm
            )
        else:
            # Legacy: local calibration at intervention layer using Nanda probe
            h = x[0, pos].detach().clone()
            per_cell_scales = compute_per_cell_scales(
                h, linear_probe, modifications, mode
            )
            flip_dirs = compute_flip_dirs(linear_probe, modifications, mode)
    else:
        flip_dirs = compute_flip_dirs(linear_probe, modifications, mode)

    # Apply intervention at layer_intervene output
    x = apply_intervention(x, flip_dirs, pos, scale,
                           per_cell_scales=per_cell_scales)

    # Capture resid at the intervention point (before any blocks run)
    # This is where the native probe should read ~100%
    resid_at_native = x[0, pos].detach().clone()

    # Continue through remaining layers to get logits
    for block in model.blocks[layer_intervene:]:
        x = block(x)

    # Final layernorm + head
    x = model.ln_f(x)
    logits = model.head(x)  # (B, T, vocab)

    return logits[0, pos], resid_at_native


def _forward_with_resid(model, input_tokens, pos, native_layer):
    """Forward pass through mingpt, returning logits and residual.

    Captures resid_post of block native_layer (= the activation at the
    intervention point). The "layer N probe" is trained on resid_post of
    block N, so native_layer = layer_intervene - 1.

    Returns (logits_at_pos, resid_at_native_layer).
    """
    b, t = input_tokens.size()
    token_emb = model.tok_emb(input_tokens)
    pos_emb = model.pos_emb[:, :t, :]
    x = model.drop(token_emb + pos_emb)

    resid = None
    for i, block in enumerate(model.blocks):
        x = block(x)
        if i == native_layer:
            # resid_post of block native_layer
            resid = x[0, pos].detach().clone()

    x = model.ln_f(x)
    logits = model.head(x)
    return logits[0, pos], resid


# ---------------------------------------------------------------------------
# MLP baseline
# ---------------------------------------------------------------------------
_VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})
_MOVE_TO_IDX = {m: i for i, m in enumerate(_VALID_MOVES)}
N_MOVES = 60


def _build_mlp(input_dim, hidden_dim, output_dim, num_hidden_layers=1):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_hidden_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def load_mlp(mlp_ckpt_path, device="cuda"):
    """Load MLP checkpoint (even/odd models).

    Returns (mlp_even, mlp_odd, mlp_probe) where mlp_probe has shape
    (1024, 8, 8, 3) — analogous to linear_probe[mode, :, r, c, class].
    """
    ckpt = torch.load(mlp_ckpt_path, map_location=device)
    hidden_dim = ckpt["hidden_dim"]
    input_dim = ckpt["input_dim"]
    num_hidden = ckpt.get("num_hidden_layers", 1)

    mlp_even = _build_mlp(input_dim, hidden_dim, 64 * 3, num_hidden)
    mlp_even.load_state_dict(ckpt["even"])
    mlp_even.to(device).eval()

    mlp_odd = _build_mlp(input_dim, hidden_dim, 64 * 3, num_hidden)
    mlp_odd.load_state_dict(ckpt["odd"])
    mlp_odd.to(device).eval()

    # Extract output layer weights as "probe": (192, H) -> (H, 8, 8, 3)
    def _extract_probe(mlp):
        W2 = mlp[-1].weight.detach()  # (192, H)
        p = W2.reshape(64, 3, hidden_dim).reshape(8, 8, 3, hidden_dim)
        return p.permute(3, 0, 1, 2)  # (H, 8, 8, 3)

    mlp_probe_even = _extract_probe(mlp_even)
    mlp_probe_odd = _extract_probe(mlp_odd)

    return mlp_even, mlp_odd, mlp_probe_even, mlp_probe_odd, hidden_dim


def build_180d_features(game_str, pos):
    """Build 180-d move-history features at position pos.

    game_str: sequence of flat board indices (0-63).
    Returns numpy array of shape (180,).
    """
    played = np.zeros(N_MOVES, dtype=np.float32)
    when = np.zeros(N_MOVES, dtype=np.float32)
    even = np.zeros(N_MOVES, dtype=np.float32)

    for s in range(pos + 1):
        move = game_str[s].item() if hasattr(game_str[s], 'item') else int(game_str[s])
        if move not in _MOVE_TO_IDX:
            continue
        idx = _MOVE_TO_IDX[move]
        played[idx] = 1.0
        when[idx] = (s + 1) / 60.0
        even[idx] = 1.0 if (s % 2 == 0) else 0.0

    return np.concatenate([played, when, even])


def mlp_per_cell_scales(h, mlp_probe, mlp_output_layer, modifications):
    """Binary search for per-cell scale that flips MLP output argmax.

    h: (H,) hidden activation
    mlp_probe: (H, 8, 8, 3) for computing flip directions
    mlp_output_layer: nn.Linear — last layer of MLP
    """
    scales = []
    for (r, c, orig_val, target_val) in modifications:
        current_class = board_val_to_probe_class(orig_val)
        target_class = board_val_to_probe_class(target_val)
        flip_dir = mlp_probe[:, r, c, target_class] - mlp_probe[:, r, c, current_class]
        d_hat = flip_dir / flip_dir.norm()
        coeff = (h @ d_hat).item()
        cell = r * 8 + c

        def pred_at_scale(s):
            h_mod = h - s * coeff * d_hat
            out = mlp_output_layer(h_mod.unsqueeze(0)).view(64, 3)
            return out[cell].argmax().item()

        lo, hi = 0.0, 10.0
        if pred_at_scale(hi) != target_class:
            scales.append(hi)
            continue
        if pred_at_scale(0.0) == target_class:
            scales.append(0.5)
            continue

        for _ in range(30):
            mid = (lo + hi) / 2
            if pred_at_scale(mid) == target_class:
                hi = mid
            else:
                lo = mid
        scales.append(min(hi * 1.1, 10.0))
    return scales


def mlp_forward_with_intervention(mlp, features, modifications, mlp_probe,
                                   scale, device="cuda", per_cell_calibrate=False):
    """Forward MLP with intervention on hidden activations.

    mlp: nn.Sequential (Linear, ReLU, Linear)
    features: (180,) numpy array
    modifications: list of (r, c, orig_val, target_val)
    mlp_probe: (H, 8, 8, 3) tensor — output layer weights reshaped
    scale: intervention scale (used as fallback if not per_cell_calibrate)
    per_cell_calibrate: if True, binary search for per-cell scale

    Returns (output_logits_64x3, hidden_original, hidden_intervened).
    """
    x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

    # Forward through hidden layer(s) — everything except last Linear
    h = x
    for layer in mlp[:-1]:
        h = layer(h)
    h_orig = h[0].detach().clone()

    # Per-cell calibration if requested
    if per_cell_calibrate:
        scales = mlp_per_cell_scales(h[0], mlp_probe, mlp[-1], modifications)
    else:
        scales = [scale] * len(modifications)

    # Compute intervention directions from mlp_probe
    h_mod = h.clone()
    for i, (r, c, orig_val, target_val) in enumerate(modifications):
        current_class = board_val_to_probe_class(orig_val)
        target_class = board_val_to_probe_class(target_val)
        flip_dir = mlp_probe[:, r, c, target_class] - mlp_probe[:, r, c, current_class]
        coeff = h_mod[0] @ flip_dir / flip_dir.norm()
        h_mod[0] = h_mod[0] - scales[i] * coeff * flip_dir / flip_dir.norm()

    h_intv = h_mod[0].detach().clone()

    # Forward through output layer
    output = mlp[-1](h_mod)  # (1, 192)
    output = output.view(64, 3)  # (64, 3) — per-cell class logits

    return output, h_orig, h_intv


def mlp_forward_clean(mlp, features, device="cuda"):
    """Clean forward pass through MLP. Returns (output_64x3, hidden)."""
    x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    h = x
    for layer in mlp[:-1]:
        h = layer(h)
    h_out = h[0].detach().clone()
    output = mlp[-1](h).view(64, 3)
    return output, h_out


def measure_mlp_probe_metrics(orig_output, intv_output, modifications):
    """Measure probe-based metrics for MLP baseline.

    orig_output, intv_output: (64, 3) tensors — per-cell class logits.

    Returns dict with: probe_acc, crosstalk, direction_acc_probe.
    """
    result = {}
    modified_set = {(m[0], m[1]) for m in modifications}

    # Probe accuracy on modified cells
    correct = 0
    dir_correct = 0
    for (r, c, orig_val, target_val) in modifications:
        cell = r * 8 + c
        target_class = board_val_to_probe_class(target_val)
        current_class = board_val_to_probe_class(orig_val)
        pred = intv_output[cell].argmax().item()
        if pred == target_class:
            correct += 1
        # Direction: did target class logit increase relative to current?
        orig_diff = orig_output[cell, target_class] - orig_output[cell, current_class]
        intv_diff = intv_output[cell, target_class] - intv_output[cell, current_class]
        if intv_diff > orig_diff:
            dir_correct += 1

    result["probe_acc"] = correct / len(modifications)
    result["direction_acc_probe"] = dir_correct / len(modifications)

    # Cross-talk: mean absolute change in logits for non-modified cells
    diff = (intv_output - orig_output).abs()  # (64, 3)
    mask = torch.ones(64, dtype=torch.bool, device=diff.device)
    for (r, c, _, _) in modifications:
        mask[r * 8 + c] = False
    if mask.sum() > 0:
        result["crosstalk"] = diff[mask].mean().item()
    else:
        result["crosstalk"] = 0.0

    return result


def calibrate_mlp_scale(mlp_even, mlp_odd, mlp_probe_even, mlp_probe_odd,
                         board_seqs_string, scales=(0.5, 1, 2, 3, 4, 6, 8),
                         n_games=50, device="cuda"):
    """Find optimal scale for MLP baseline via N=1 flip interventions."""
    print("=== MLP Scale Calibration ===")
    rng = random.Random(42)
    results = {}

    for scale in scales:
        probe_accs = []
        for gi in tqdm(range(n_games), desc=f"s{scale}", leave=False):
            game_str = board_seqs_string[gi]
            pos = rng.randint(POS_RANGE[0], POS_RANGE[1])
            board_state, color = replay_to_position(game_str, pos)

            mods = select_flip_noninteracting(board_state, 1, rng)
            if mods is None:
                continue

            features = build_180d_features(game_str, pos)
            is_even = (pos % 2 == 0)
            mlp = mlp_even if is_even else mlp_odd
            mlp_probe = mlp_probe_even if is_even else mlp_probe_odd

            with torch.no_grad():
                orig_out, _ = mlp_forward_clean(mlp, features, device)
                intv_out, _, _ = mlp_forward_with_intervention(
                    mlp, features, mods, mlp_probe, scale, device
                )
                metrics = measure_mlp_probe_metrics(orig_out, intv_out, mods)
                probe_accs.append(metrics["probe_acc"])

        mean_acc = np.mean(probe_accs) if probe_accs else 0.0
        results[scale] = mean_acc
        print(f"  scale {scale}: probe_acc = {mean_acc:.3f} (n={len(probe_accs)})")

    best_scale = max(results, key=results.get)
    print(f"\nBest MLP scale: {best_scale}, probe_acc={results[best_scale]:.3f}")
    return best_scale, results


def run_mlp_experiment(mlp_even, mlp_odd, mlp_probe_even, mlp_probe_odd,
                        board_seqs_string, scale,
                        n_games=200, n_values=None, seed=42, device="cuda",
                        per_cell_calibrate=False):
    """Run multi-intervention experiment on MLP baseline."""
    if n_values is None:
        n_values = N_VALUES

    rng = random.Random(seed)

    results = {cond: {str(n): [] for n in n_values}
               for cond, _ in CONDITIONS}
    results["mixed"] = {str(n): [] for n in n_values}

    print(f"\n=== MLP Baseline Experiment ===")
    print(f"Scale: {scale}")
    print(f"Games: {n_games}, N values: {n_values}")

    for gi in tqdm(range(n_games), desc="MLP Games"):
        game_str = board_seqs_string[gi]
        pos = rng.randint(POS_RANGE[0], min(POS_RANGE[1], 58))
        board_state, color = replay_to_position(game_str, pos)

        features = build_180d_features(game_str, pos)
        is_even = (pos % 2 == 0)
        mlp = mlp_even if is_even else mlp_odd
        mlp_probe = mlp_probe_even if is_even else mlp_probe_odd

        with torch.no_grad():
            orig_out, orig_h = mlp_forward_clean(mlp, features, device)

        for n in n_values:
            for cond_name, select_fn in CONDITIONS:
                mods = select_fn(board_state, n, rng)
                if mods is None:
                    continue

                with torch.no_grad():
                    intv_out, _, intv_h = mlp_forward_with_intervention(
                        mlp, features, mods, mlp_probe, scale, device,
                        per_cell_calibrate=per_cell_calibrate
                    )

                metrics = measure_mlp_probe_metrics(orig_out, intv_out, mods)

                # Hidden-layer cross-talk using mlp_probe
                orig_probe = torch.einsum("d, d r c o -> r c o", orig_h, mlp_probe)
                intv_probe = torch.einsum("d, d r c o -> r c o", intv_h, mlp_probe)
                diff = (intv_probe - orig_probe).abs()
                mask = torch.ones(8, 8, dtype=torch.bool, device=diff.device)
                for (r, c, _, _) in mods:
                    mask[r, c] = False
                if mask.sum() > 0:
                    metrics["hidden_crosstalk"] = diff[mask].mean().item()
                else:
                    metrics["hidden_crosstalk"] = 0.0

                results[cond_name][str(n)].append(metrics)

            # Mixed condition
            mods, (n_flips, n_adds) = select_mixed(board_state, n, rng)
            if mods is not None:
                with torch.no_grad():
                    intv_out, _, intv_h = mlp_forward_with_intervention(
                        mlp, features, mods, mlp_probe, scale, device,
                        per_cell_calibrate=per_cell_calibrate
                    )
                metrics = measure_mlp_probe_metrics(orig_out, intv_out, mods)

                orig_probe = torch.einsum("d, d r c o -> r c o", orig_h, mlp_probe)
                intv_probe = torch.einsum("d, d r c o -> r c o", intv_h, mlp_probe)
                diff = (intv_probe - orig_probe).abs()
                mask = torch.ones(8, 8, dtype=torch.bool, device=diff.device)
                for (r, c, _, _) in mods:
                    mask[r, c] = False
                if mask.sum() > 0:
                    metrics["hidden_crosstalk"] = diff[mask].mean().item()
                else:
                    metrics["hidden_crosstalk"] = 0.0

                metrics["n_flips"] = n_flips
                metrics["n_adds"] = n_adds
                results["mixed"][str(n)].append(metrics)

    return results


def aggregate_mlp_results(results):
    """Aggregate MLP per-sample results into means and stds."""
    aggregated = {}
    for cond in results:
        aggregated[cond] = {}
        for n_str, samples in results[cond].items():
            if not samples:
                aggregated[cond][n_str] = {"n_samples": 0}
                continue

            agg = {"n_samples": len(samples)}
            for key in ["probe_acc", "direction_acc_probe", "crosstalk",
                         "hidden_crosstalk"]:
                vals = [s[key] for s in samples if s.get(key) is not None]
                if vals:
                    agg[key] = float(np.mean(vals))
                    agg[key + "_std"] = float(np.std(vals))
                else:
                    agg[key] = None

            if cond == "mixed" and "n_flips" in samples[0]:
                compositions = defaultdict(list)
                for s in samples:
                    comp_key = f"{s['n_flips']}f_{s['n_adds']}a"
                    compositions[comp_key].append(s)
                agg["compositions"] = {}
                for comp_key, comp_samples in compositions.items():
                    comp_agg = {"n_samples": len(comp_samples)}
                    for key in ["probe_acc", "direction_acc_probe", "crosstalk"]:
                        vals = [s[key] for s in comp_samples if s.get(key) is not None]
                        if vals:
                            comp_agg[key] = float(np.mean(vals))
                    agg["compositions"][comp_key] = comp_agg

            aggregated[cond][n_str] = agg

    return aggregated


def plot_mlp_results(aggregated, n_values, output_dir):
    """Generate plots for MLP baseline results."""
    metrics_to_plot = [
        ("probe_acc", "MLP Probe Accuracy (Modified Cells)", "Fraction correct"),
        ("direction_acc_probe", "MLP Direction Accuracy (Modified Cells)",
         "Fraction correct direction"),
        ("crosstalk", "MLP Output Cross-talk (Non-modified Cells)",
         "Mean |change| in output logits"),
        ("hidden_crosstalk", "MLP Hidden Cross-talk via Probe",
         "Mean |change| in probe logits"),
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
                ax.errorbar(xs, ys, yerr=errs, marker='o',
                            label=COND_LABELS[cond],
                            color=COND_COLORS[cond], capsize=3)
        ax.set_xlabel("Number of interventions (N)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xticks(n_values)
        plt.tight_layout()
        fname = os.path.join(output_dir, f"mlp_{metric_key}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------
def measure_logit_metrics(original_logits, intervened_logits,
                          original_legal, counterfactual_legal,
                          save_probs=False):
    """Compute logit-based metrics for an intervention.

    Returns dict with: boundary_margin, boundary_margin_frac_positive,
    top1_legal_strict, legal_prob_mass, mean_logprob_shift_illegal,
    mean_logprob_shift_legal, mean_rank_shift_illegal, mean_rank_shift_legal.

    If save_probs=True, also includes orig_probs and intv_probs (64-element
    lists, indexed by board cell, -1 for non-valid cells), plus
    original_legal and counterfactual_legal as sorted lists.
    """
    newly_legal = counterfactual_legal - original_legal
    newly_illegal = original_legal - counterfactual_legal
    cf_legal_stoi = counterfactual_legal & STOI_SET

    # Map model logits to per-cell values (60 valid cells)
    # Model output: index 0 = pass, indices 1-60 = stoi_indices
    intv_cell = torch.full((64,), float('-inf'), device=intervened_logits.device)
    intv_cell[stoi_indices] = intervened_logits[1:61]

    result = {}
    result["n_changed"] = len(newly_legal) + len(newly_illegal)

    # 1. Top-1 legal (strict): only when top move changed
    valid_logits = intervened_logits[1:61]
    orig_valid_logits = original_logits[1:61]
    orig_top_cell = stoi_indices[orig_valid_logits.argmax().item()]
    intv_top_cell = stoi_indices[valid_logits.argmax().item()]
    if orig_top_cell != intv_top_cell:
        result["top1_legal_strict"] = 1.0 if intv_top_cell in cf_legal_stoi else 0.0
    else:
        result["top1_legal_strict"] = None

    # 2. Legal probability mass
    probs = torch.softmax(valid_logits, dim=0)
    legal_mask = torch.zeros(60, device=probs.device)
    for i, cell in enumerate(stoi_indices):
        if cell in counterfactual_legal:
            legal_mask[i] = 1.0
    result["legal_prob_mass"] = (probs * legal_mask).sum().item()

    # 3. Log-probability shift (natural unit of cross-entropy)
    orig_logprobs = torch.log_softmax(orig_valid_logits, dim=0)
    intv_logprobs = torch.log_softmax(valid_logits, dim=0)
    orig_lp = torch.full((64,), float('-inf'), device=original_logits.device)
    intv_lp = torch.full((64,), float('-inf'), device=intervened_logits.device)
    orig_lp[stoi_indices] = orig_logprobs
    intv_lp[stoi_indices] = intv_logprobs

    lp_shifts_illegal = []
    lp_shifts_legal = []
    for cell in newly_illegal:
        if cell in STOI_SET:
            lp_shifts_illegal.append((intv_lp[cell] - orig_lp[cell]).item())
    for cell in newly_legal:
        if cell in STOI_SET:
            lp_shifts_legal.append((intv_lp[cell] - orig_lp[cell]).item())

    result["mean_logprob_shift_illegal"] = np.mean(lp_shifts_illegal) if lp_shifts_illegal else None
    result["mean_logprob_shift_legal"] = np.mean(lp_shifts_legal) if lp_shifts_legal else None

    # 4. Rank shift
    orig_ranks = torch.zeros(64, dtype=torch.long, device=original_logits.device)
    intv_ranks = torch.zeros(64, dtype=torch.long, device=intervened_logits.device)
    for rank, idx in enumerate(orig_valid_logits.argsort(descending=True).tolist()):
        orig_ranks[stoi_indices[idx]] = rank
    for rank, idx in enumerate(valid_logits.argsort(descending=True).tolist()):
        intv_ranks[stoi_indices[idx]] = rank

    rank_shifts_illegal = []
    rank_shifts_legal = []
    for cell in newly_illegal:
        if cell in STOI_SET:
            rank_shifts_illegal.append((intv_ranks[cell] - orig_ranks[cell]).item())
    for cell in newly_legal:
        if cell in STOI_SET:
            rank_shifts_legal.append((intv_ranks[cell] - orig_ranks[cell]).item())

    result["mean_rank_shift_illegal"] = np.mean(rank_shifts_illegal) if rank_shifts_illegal else None
    result["mean_rank_shift_legal"] = np.mean(rank_shifts_legal) if rank_shifts_legal else None

    # 5. Boundary margin: distance from the legal/illegal boundary
    # Boundary = logit of the worst counterfactual-legal move after intervention
    # Newly illegal: margin = boundary - cell_logit (positive = correct side)
    # Newly legal:   margin = cell_logit - boundary (positive = correct side)
    if cf_legal_stoi:
        boundary = min(intv_cell[cell].item() for cell in cf_legal_stoi)
    else:
        boundary = None

    margins = []
    if boundary is not None:
        for cell in newly_illegal:
            if cell in STOI_SET:
                margins.append(boundary - intv_cell[cell].item())
        for cell in newly_legal:
            if cell in STOI_SET:
                margins.append(intv_cell[cell].item() - boundary)

    result["boundary_margin"] = np.mean(margins) if margins else None
    result["boundary_margin_frac_positive"] = (
        np.mean([m > 0 for m in margins]) if margins else None
    )

    # 6. Li et al. metric: top-N accuracy
    # N = number of legal moves in counterfactual board
    # Compare model's top-N predictions against ground truth legal moves
    # Error = |false positives| + |false negatives|
    n_cf_legal = len(cf_legal_stoi)
    if n_cf_legal > 0:
        topn_indices = valid_logits.argsort(descending=True)[:n_cf_legal]
        topn_cells = set(stoi_indices[i] for i in topn_indices.tolist())
        false_positives = topn_cells - cf_legal_stoi
        false_negatives = cf_legal_stoi - topn_cells
        result["li_topn_errors"] = len(false_positives) + len(false_negatives)
        result["li_topn_accuracy"] = 1.0 - result["li_topn_errors"] / (2 * n_cf_legal)
    else:
        result["li_topn_errors"] = None
        result["li_topn_accuracy"] = None

    if save_probs:
        # Full probability vectors indexed by board cell (0-63)
        orig_probs_64 = [-1.0] * 64
        intv_probs_64 = [-1.0] * 64
        orig_probs_valid = torch.softmax(orig_valid_logits, dim=0)
        intv_probs_valid = torch.softmax(valid_logits, dim=0)
        for i, cell in enumerate(stoi_indices):
            orig_probs_64[cell] = orig_probs_valid[i].item()
            intv_probs_64[cell] = intv_probs_valid[i].item()
        result["orig_probs"] = orig_probs_64
        result["intv_probs"] = intv_probs_64
        result["original_legal"] = sorted(original_legal)
        result["counterfactual_legal"] = sorted(counterfactual_legal)

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
                           mode=PROBE_MODE, downstream_probe=None):
    """Fraction of modified cells whose probe argmax matches target state.

    If downstream_probe (nn.Linear) is provided, use it instead of the
    Nanda probe (linear_probe). This ensures we measure with the same
    probe used for calibration.
    """
    if downstream_probe is not None:
        with torch.no_grad():
            logits = downstream_probe(intervened_resid).view(64, 3)
        correct = 0
        for (r, c, orig_val, target_val) in modifications:
            cell = r * 8 + c
            target_class = board_val_to_probe_class(target_val)
            predicted_class = logits[cell].argmax().item()
            if predicted_class == target_class:
                correct += 1
        return correct / len(modifications)

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
                if metrics["boundary_margin"] is not None:
                    direction_accs.append(metrics["boundary_margin"])

            mean_val = np.mean(direction_accs) if direction_accs else 0.0
            results[(layer, scale)] = mean_val
            print(f"  Layer {layer}, scale {scale}: boundary_margin = {mean_val:.3f} "
                  f"(n={len(direction_accs)})")

    best = max(results, key=results.get)
    print(f"\nBest: layer={best[0]}, scale={best[1]}, "
          f"boundary_margin={results[best]:.3f}")
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
                   n_games=200, n_values=None, seed=42, device="cuda",
                   per_cell_calibrate=False,
                   direction_probe=None, downstream_probes=None,
                   cal_depth=0, save_probs=False, cascade=False):
    """Run the full multi-intervention experiment."""
    if n_values is None:
        n_values = N_VALUES

    rng = random.Random(seed)

    # Initialize results storage
    results = {cond: {str(n): [] for n in n_values}
               for cond, _ in CONDITIONS}
    results["mixed"] = {str(n): [] for n in n_values}

    print(f"\n=== Multi-Intervention Experiment ===")
    if cascade:
        print(f"CASCADE mode: intervene at every layer from {layer_intervene} to 7")
    else:
        print(f"Layer intervene: {layer_intervene}, Native layer: {layer_probe}, "
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
            # Cache prefix activations (layers 0..layer_intervene-1)
            prefix_acts = compute_prefix_activations(
                model, input_tokens, layer_intervene
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
                    if cascade:
                        intv_logits, intv_resid = run_with_cascade_intervention(
                            model, input_tokens, mods,
                            pos, layer_intervene, downstream_probes,
                            device=device, prefix_acts=prefix_acts,
                        )
                    else:
                        intv_logits, intv_resid = run_with_intervention(
                            model, input_tokens, mods, linear_probe,
                            pos, scale, layer_intervene, layer_probe,
                            device=device,
                            per_cell_calibrate=per_cell_calibrate,
                            prefix_acts=prefix_acts,
                            direction_probe=direction_probe,
                            downstream_probes=downstream_probes,
                            cal_depth=cal_depth,
                        )

                logit_metrics = measure_logit_metrics(
                    orig_logits_at_pos, intv_logits,
                    original_legal, cf_legal,
                    save_probs=save_probs,
                )
                # Use per-layer probe if available for crosstalk + accuracy
                if downstream_probes is not None:
                    parity_str = "even" if pos % 2 == 0 else "odd"
                    ds_probe = downstream_probes.get((layer_probe, parity_str))
                    from layer_propagation import measure_crosstalk as measure_crosstalk_nn
                    crosstalk = measure_crosstalk_nn(
                        orig_resid, intv_resid, ds_probe, mods)
                else:
                    ds_probe = None
                    crosstalk = measure_probe_crosstalk(
                        orig_resid, intv_resid, linear_probe, mods)
                probe_acc = measure_probe_accuracy(
                    intv_resid, linear_probe, mods,
                    downstream_probe=ds_probe
                )

                sample = {
                    "game_idx": gi,
                    "pos": pos,
                    "color": color,
                    "modifications": [(r, c, int(o), int(t)) for r, c, o, t in mods],
                    "n_original_legal": len(original_legal),
                    "n_cf_legal": len(cf_legal),
                    **logit_metrics,
                    "crosstalk": crosstalk,
                    "probe_acc": probe_acc,
                }
                if save_probs:
                    sample["board_state"] = board_state.flatten().tolist()
                results[cond_name][str(n)].append(sample)

            # Mixed condition
            mods, (n_flips, n_adds) = select_mixed(board_state, n, rng)
            if mods is not None:
                cf_mods = [(r, c, tgt) for (r, c, _, tgt) in mods]
                cf_legal = compute_counterfactual_legal(
                    board_state, cf_mods, color
                )

                with torch.no_grad():
                    if cascade:
                        intv_logits, intv_resid = run_with_cascade_intervention(
                            model, input_tokens, mods,
                            pos, layer_intervene, downstream_probes,
                            device=device, prefix_acts=prefix_acts,
                        )
                    else:
                        intv_logits, intv_resid = run_with_intervention(
                            model, input_tokens, mods, linear_probe,
                            pos, scale, layer_intervene, layer_probe,
                            device=device,
                            per_cell_calibrate=per_cell_calibrate,
                            prefix_acts=prefix_acts,
                            direction_probe=direction_probe,
                            downstream_probes=downstream_probes,
                            cal_depth=cal_depth,
                        )

                logit_metrics = measure_logit_metrics(
                    orig_logits_at_pos, intv_logits,
                    original_legal, cf_legal,
                    save_probs=save_probs,
                )
                if downstream_probes is not None:
                    parity_str = "even" if pos % 2 == 0 else "odd"
                    ds_probe = downstream_probes.get((layer_probe, parity_str))
                    from layer_propagation import measure_crosstalk as measure_crosstalk_nn
                    crosstalk = measure_crosstalk_nn(
                        orig_resid, intv_resid, ds_probe, mods)
                else:
                    ds_probe = None
                    crosstalk = measure_probe_crosstalk(
                        orig_resid, intv_resid, linear_probe, mods)
                probe_acc = measure_probe_accuracy(
                    intv_resid, linear_probe, mods,
                    downstream_probe=ds_probe
                )

                sample = {
                    "game_idx": gi,
                    "pos": pos,
                    "color": color,
                    "modifications": [(r, c, int(o), int(t)) for r, c, o, t in mods],
                    "n_original_legal": len(original_legal),
                    "n_cf_legal": len(cf_legal),
                    **logit_metrics,
                    "crosstalk": crosstalk,
                    "probe_acc": probe_acc,
                    "n_flips": n_flips,
                    "n_adds": n_adds,
                }
                if save_probs:
                    sample["board_state"] = board_state.flatten().tolist()
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
            for key in ["boundary_margin", "boundary_margin_frac_positive",
                         "top1_legal_strict", "legal_prob_mass",
                         "mean_logprob_shift_illegal", "mean_logprob_shift_legal",
                         "mean_rank_shift_illegal", "mean_rank_shift_legal",
                         "crosstalk", "probe_acc",
                         "li_topn_errors", "li_topn_accuracy"]:
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
                    for key in ["boundary_margin", "boundary_margin_frac_positive",
                                 "top1_legal_strict", "legal_prob_mass",
                                 "mean_logprob_shift_illegal", "mean_logprob_shift_legal",
                                 "mean_rank_shift_illegal", "mean_rank_shift_legal",
                                 "crosstalk", "probe_acc",
                                 "li_topn_errors", "li_topn_accuracy"]:
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
        ("boundary_margin", "Boundary Margin (mean)", "Logit margin (+ = correct side)"),
        ("boundary_margin_frac_positive", "Boundary Margin (fraction correct)", "Fraction on correct side"),
        ("top1_legal_strict", "Top-1 Legal (Strict: top move changed)", "Fraction legal"),
        ("legal_prob_mass", "Legal Probability Mass", "Probability"),
        ("crosstalk", "Probe Cross-talk", "Mean |change| (non-modified cells)"),
        ("probe_acc", "Probe Accuracy (Modified Cells)", "Fraction correct"),
        ("li_topn_accuracy", "Li et al. Top-N Accuracy", "Accuracy (1 - errors/2N)"),
        ("li_topn_errors", "Li et al. Top-N Errors (FP+FN)", "Mean errors"),
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

    # Special plot: log-probability shifts (two lines per condition)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in COND_COLORS:
        xs_il, ys_il = [], []
        xs_le, ys_le = [], []
        for n in n_values:
            data = aggregated.get(cond, {}).get(str(n), {})
            val_il = data.get("mean_logprob_shift_illegal")
            val_le = data.get("mean_logprob_shift_legal")
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
    ax.set_ylabel("Mean log-probability shift (nats)")
    ax.set_title("Log-Probability Shift (Newly Legal vs Newly Illegal)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xticks(n_values)
    plt.tight_layout()
    fname = os.path.join(output_dir, "logprob_shifts.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")

    # Special plot: rank shifts (two lines per condition)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in COND_COLORS:
        xs_il, ys_il = [], []
        xs_le, ys_le = [], []
        for n in n_values:
            data = aggregated.get(cond, {}).get(str(n), {})
            val_il = data.get("mean_rank_shift_illegal")
            val_le = data.get("mean_rank_shift_legal")
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
    ax.set_ylabel("Mean rank shift (+ = worse rank)")
    ax.set_title("Rank Shift (Newly Legal vs Newly Illegal)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xticks(n_values)
    plt.tight_layout()
    fname = os.path.join(output_dir, "rank_shifts.png")
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
    parser.add_argument("--layer-probe", type=int, default=None,
                        help="Probe layer (default: layer-intervene - 1, i.e. native). "
                             "Deprecated: now auto-derived from --layer-intervene.")
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
    # MLP baseline
    parser.add_argument("--per-cell-scale", action="store_true",
                        help="Use per-cell calibrated scales via local "
                             "binary search (ensures native probe flips at intervention point)")
    parser.add_argument("--cal-depth", type=int, default=0,
                        help="Calibration depth: 0=local (native probe), "
                             "1=+1 layer downstream, 2=+2 layers, etc.")
    parser.add_argument("--probe-dir", type=str, default=None,
                        help="Directory with per-layer probe checkpoints "
                             "(required for --per-cell-scale)")
    parser.add_argument("--save-probs", action="store_true",
                        help="Save full 64-cell probability vectors per sample")
    parser.add_argument("--cascade", action="store_true",
                        help="Cascade intervention: apply at every layer from "
                             "layer-intervene to 7, each using its native probe")
    # MLP baseline
    parser.add_argument("--mlp-baseline", action="store_true",
                        help="Run MLP baseline instead of OthelloGPT")
    parser.add_argument("--mlp-ckpt", type=str, default=None,
                        help="Path to MLP checkpoint (e.g. mlp_180_H1024.pt)")
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.mlp_baseline:
        # ---- MLP baseline mode ----
        if args.mlp_ckpt is None:
            parser.error("--mlp-ckpt is required when using --mlp-baseline")

        print("Loading MLP and game data...")
        mlp_even, mlp_odd, mlp_probe_even, mlp_probe_odd, hidden_dim = load_mlp(
            args.mlp_ckpt, device=args.device
        )
        # Load game data (still needed for board replay)
        script_dir = os.path.dirname(__file__)
        board_seqs_string = torch.load(
            os.path.join(script_dir, "board_seqs_string.pth"), map_location="cpu"
        )
        print(f"MLP loaded: hidden_dim={hidden_dim}")

        # MLP scale calibration
        scale = args.scale
        if scale is None or args.calibrate:
            scale, cal_results = calibrate_mlp_scale(
                mlp_even, mlp_odd, mlp_probe_even, mlp_probe_odd,
                board_seqs_string,
                n_games=min(args.n_games, 50), device=args.device
            )
            cal_path = os.path.join(args.output_dir, "mlp_calibration.json")
            with open(cal_path, "w") as f:
                json.dump({"best_scale": scale,
                           "results": {str(k): v for k, v in cal_results.items()}},
                          f, indent=2)
            print(f"MLP calibration saved to {cal_path}")

            if args.calibrate_only:
                return

        if args.scale is not None:
            scale = args.scale

        # Run MLP experiment
        raw_results = run_mlp_experiment(
            mlp_even, mlp_odd, mlp_probe_even, mlp_probe_odd,
            board_seqs_string, scale,
            n_games=args.n_games, n_values=n_values,
            seed=args.seed, device=args.device,
            per_cell_calibrate=args.per_cell_scale
        )

        aggregated = aggregate_mlp_results(raw_results)

        output = {
            "config": {
                "mode": "mlp_baseline",
                "mlp_ckpt": args.mlp_ckpt,
                "n_games": args.n_games,
                "scale": scale,
                "n_values": n_values,
                "seed": args.seed,
            },
            "results": aggregated,
        }

        results_path = os.path.join(args.output_dir, "mlp_results.json")
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nMLP results saved to {results_path}")

        # Print summary
        print("\n=== MLP Baseline Summary ===")
        for cond in list(COND_LABELS.keys()):
            print(f"\n{COND_LABELS[cond]}:")
            for n in n_values:
                data = aggregated.get(cond, {}).get(str(n), {})
                if data.get("n_samples", 0) == 0:
                    print(f"  N={n}: no samples")
                    continue
                pa = data.get("probe_acc")
                da = data.get("direction_acc_probe")
                ct = data.get("crosstalk")
                hc = data.get("hidden_crosstalk")
                ns = data.get("n_samples", 0)
                pa_s = f"{pa:.3f}" if pa is not None else "N/A"
                da_s = f"{da:.3f}" if da is not None else "N/A"
                ct_s = f"{ct:.4f}" if ct is not None else "N/A"
                hc_s = f"{hc:.4f}" if hc is not None else "N/A"
                print(f"  N={n}: probe={pa_s}  dir={da_s}  "
                      f"xtalk={ct_s}  h_xtalk={hc_s}  (n={ns})")

        print("\nGenerating MLP plots...")
        plot_mlp_results(aggregated, n_values, args.output_dir)
        print("Done.")
        return

    # ---- OthelloGPT mode (original) ----
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

    # Load per-layer probes for local calibration using native probe
    direction_probe_tensor = None
    downstream_probes = None
    native_layer = layer_intervene - 1  # resid_post of this block
    if args.per_cell_scale or args.cascade:
        if args.probe_dir is None:
            parser.error("--probe-dir is required when using --per-cell-scale or --cascade")
        from layer_propagation import load_probes, build_direction_probe
        downstream_probes = load_probes(args.probe_dir, args.device)
        direction_probe_tensor = build_direction_probe(
            downstream_probes, layer=native_layer, device=args.device)
        print(f"Loaded per-layer probes for "
              f"{'cascade' if args.cascade else 'local calibration'} "
              f"(native layer: L{native_layer})")

    # Run experiment
    cal_depth = args.cal_depth
    raw_results = run_experiment(
        model, linear_probe, board_seqs_int, board_seqs_string,
        layer_intervene, native_layer, scale,
        n_games=args.n_games, n_values=n_values,
        seed=args.seed, device=args.device,
        per_cell_calibrate=args.per_cell_scale,
        direction_probe=direction_probe_tensor,
        downstream_probes=downstream_probes,
        cal_depth=cal_depth,
        save_probs=args.save_probs,
        cascade=args.cascade,
    )

    # Aggregate
    aggregated = aggregate_results(raw_results)

    # Save
    output = {
        "config": {
            "n_games": args.n_games,
            "layer_intervene": layer_intervene,
            "native_layer": native_layer,
            "scale": scale,
            "cal_depth": cal_depth,
            "cascade": args.cascade,
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

    # Save raw per-sample data for downstream analysis
    raw_path = os.path.join(args.output_dir, "raw_samples.json")
    with open(raw_path, "w") as f:
        json.dump(raw_results, f, indent=1)
    print(f"Raw per-sample data saved to {raw_path}")

    # Print summary table
    print("\n=== Summary ===")
    for cond in list(COND_LABELS.keys()):
        print(f"\n{COND_LABELS[cond]}:")
        for n in n_values:
            data = aggregated.get(cond, {}).get(str(n), {})
            if data.get("n_samples", 0) == 0:
                print(f"  N={n}: no samples")
                continue
            bm = data.get("boundary_margin")
            bmf = data.get("boundary_margin_frac_positive")
            t1s = data.get("top1_legal_strict")
            lpm = data.get("legal_prob_mass")
            lp_il = data.get("mean_logprob_shift_illegal")
            lp_le = data.get("mean_logprob_shift_legal")
            rk_il = data.get("mean_rank_shift_illegal")
            rk_le = data.get("mean_rank_shift_legal")
            ct = data.get("crosstalk")
            pa = data.get("probe_acc")
            ns = data.get("n_samples", 0)
            bm_s = f"{bm:+.2f}" if bm is not None else "N/A"
            bmf_s = f"{bmf:.3f}" if bmf is not None else "N/A"
            t1s_s = f"{t1s:.3f}" if t1s is not None else "N/A"
            lpm_s = f"{lpm:.3f}" if lpm is not None else "N/A"
            lp_il_s = f"{lp_il:+.2f}" if lp_il is not None else "N/A"
            lp_le_s = f"{lp_le:+.2f}" if lp_le is not None else "N/A"
            rk_il_s = f"{rk_il:+.1f}" if rk_il is not None else "N/A"
            rk_le_s = f"{rk_le:+.1f}" if rk_le is not None else "N/A"
            ct_s = f"{ct:.4f}" if ct is not None else "N/A"
            pa_s = f"{pa:.3f}" if pa is not None else "N/A"
            li_acc = data.get("li_topn_accuracy")
            li_err = data.get("li_topn_errors")
            li_acc_s = f"{li_acc:.3f}" if li_acc is not None else "N/A"
            li_err_s = f"{li_err:.2f}" if li_err is not None else "N/A"
            print(f"  N={n}: margin={bm_s}  margin_frac={bmf_s}  "
                  f"top1s={t1s_s}  mass={lpm_s}  "
                  f"lp_il/le={lp_il_s}/{lp_le_s}  "
                  f"rank_il/le={rk_il_s}/{rk_le_s}  "
                  f"xtalk={ct_s}  probe={pa_s}  "
                  f"li_acc={li_acc_s}  li_err={li_err_s}  (n={ns})")

    # Plot
    print("\nGenerating plots...")
    plot_results(aggregated, n_values, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
