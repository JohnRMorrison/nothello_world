"""Probe loading + direction extraction (empty direction, flip direction,
between-class direction).

The Nanda probe `main_linear_probe.pth` has shape (3 modes, d_model=512,
rows=8, cols=8, classes=3). Modes 0 and 1 are parity-aware mine/yours/empty
labels; mode 0 has (empty=0, yours=1, mine=2) and mode 1 the reverse. Mode 2
is degraded (~79% acc) and should not be used.

"Mine" = the color of the player about to move. This is determined by the
position; this module's helpers take a `next_color` argument.
"""

from __future__ import annotations

import torch

from . import config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_probe(probe_path: str = None, device: str = "cpu") -> torch.Tensor:
    """Load the Nanda probe. Returns a (3, 512, 8, 8, 3) tensor on `device`."""
    probe_path = probe_path or config.PROBE_PATH
    probe = torch.load(probe_path, map_location=device, weights_only=False)
    return probe.detach()


def probe_slice(probe: torch.Tensor, mode: int) -> torch.Tensor:
    """Return the (d_model, 8, 8, 3) slice for a single probe mode."""
    return probe[mode].detach()


# ---------------------------------------------------------------------------
# Class indexing under mode 0 mine/yours convention
# ---------------------------------------------------------------------------
def board_val_to_probe_class(
    val: int, next_color: int, probe_mode: int = 0
) -> int:
    """Map a board value (+1/-1/0) to a probe class index.

    Mode 0: empty=0, yours=1, mine=2 (mine = next_color).
    Mode 1: empty=0, mine=1, yours=2.
    Mode 2: empty=0, white(-1)=1, black(+1)=2 (absolute; degraded probe).
    """
    if probe_mode == 0:
        if val == 0:
            return 0
        return 2 if val == next_color else 1
    if probe_mode == 1:
        if val == 0:
            return 0
        return 1 if val == next_color else 2
    if probe_mode == 2:
        if val == 0:
            return 0
        return 1 if val == -1 else 2
    raise ValueError(f"unknown probe_mode: {probe_mode}")


# ---------------------------------------------------------------------------
# Direction vectors
# ---------------------------------------------------------------------------
def direction_between_classes(
    probe: torch.Tensor,
    r: int, c: int,
    current_class: int, target_class: int,
    probe_mode: int = 0,
) -> torch.Tensor:
    """`d = w_target - w_current` for the cell (r,c) under the chosen mode.

    Returns an unnormalized (d_model,) tensor. Callers usually want d/||d||.
    """
    W = probe[probe_mode, :, r, c, :]   # (d_model, 3)
    return W[:, target_class] - W[:, current_class]


def empty_direction(probe: torch.Tensor, r: int, c: int,
                    probe_mode: int = 0) -> torch.Tensor:
    """Symmetric "push toward empty" direction:
        d = w_empty - 0.5 * (w_yours + w_mine)
    Doesn't depend on parity (class 0 = empty in all modes)."""
    W = probe[probe_mode, :, r, c, :]
    return W[:, 0] - 0.5 * (W[:, 1] + W[:, 2])


def flip_direction(probe: torch.Tensor, r: int, c: int,
                   probe_mode: int = 0) -> torch.Tensor:
    """Color-flip direction:
        mode 0: w_mine - w_yours = w[:, 2] - w[:, 1]
        mode 1: w_mine - w_yours = w[:, 1] - w[:, 2]
        mode 2: w_black - w_white = w[:, 2] - w[:, 1]
    """
    W = probe[probe_mode, :, r, c, :]
    if probe_mode in (0, 2):
        return W[:, 2] - W[:, 1]
    return W[:, 1] - W[:, 2]


def normalize(d: torch.Tensor) -> torch.Tensor:
    n = d.norm()
    if n == 0:
        return d
    return d / n


# ---------------------------------------------------------------------------
# Decoding (probe argmax at every cell, given a residual)
# ---------------------------------------------------------------------------
def decode_board(resid: torch.Tensor, probe_full_mode: torch.Tensor) -> torch.Tensor:
    """resid: (d_model,) tensor. probe_full_mode: (d_model, 8, 8, 3).
    Returns (8, 8) int tensor of argmax class predictions."""
    logits = torch.einsum("d,drco->rco", resid, probe_full_mode)
    return logits.argmax(-1)
