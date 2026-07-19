"""Load and compute activations for the 960 hand-crafted flanking patterns.

Each pattern encodes an Othello legality rule:
  target C is legal iff
    - C is empty
    - each cell in opponents is opponent-color
    - terminal cell is mover-color

Under the moveset+parity approximation (ignoring captures), we treat
"placed color" as "current color".  For non-center cells:
  is_opp(c)   = played[c] AND (even[c] == mover_parity)
  is_mover(c) = played[c] AND (even[c] != mover_parity)

For the 4 pre-placed center cells (D4=W, E4=B, D5=B, E5=W) we use their
initial colors.  Since initial positions never appear "empty", patterns
whose target is a center cell will never fire — that's fine, they are
Othello-illegal anyway.
"""
import numpy as np
import torch

CENTER_64 = {27, 28, 35, 36}
NON_CENTER_64 = sorted(set(range(64)) - CENTER_64)
C64_TO_C60 = {c: i for i, c in enumerate(NON_CENTER_64)}

# Initial center colors: 0 = BLACK, 1 = WHITE.
# D4=27 W, E4=28 B, D5=35 B, E5=36 W.
_CENTER_INITIAL_COLOR = {27: 1, 28: 0, 35: 0, 36: 1}


def load_patterns(path):
    """Load the .pt file and return the list of 960 pattern dicts."""
    d = torch.load(path, map_location='cpu')
    return d['patterns']


def _cell_indexers(played_c60, even_c60):
    """Return (played_c64_lookup, even_c64_lookup): functions from cell in
    0..63 to (N,) numpy arrays representing played bit and even bit.  For
    center cells: played is always 1, even follows the initial-color rule
    (initial-color BLACK → even bit 1, initial-color WHITE → even bit 0)."""
    N = played_c60.shape[0]
    played_c60_bool = played_c60.astype(bool)
    even_c60_u8 = even_c60.astype(np.uint8)
    def played(cell):
        if cell in CENTER_64:
            return np.ones(N, dtype=bool)
        return played_c60_bool[:, C64_TO_C60[cell]]
    def even(cell):
        if cell in CENTER_64:
            # BLACK initial → placed at even turn → even bit = 1.
            # WHITE initial → placed at odd turn → even bit = 0.
            val = 1 if _CENTER_INITIAL_COLOR[cell] == 0 else 0
            return np.full(N, val, dtype=np.uint8)
        return even_c60_u8[:, C64_TO_C60[cell]]
    return played, even


def compute_pattern_activations(patterns, played_c60, even_c60,
                                    mover_parity):
    """Return (N, K) uint8 matrix of pattern activations.

    played_c60:   (N, 60) uint8 or bool.
    even_c60:     (N, 60) uint8 or bool.
    mover_parity: (N,) uint8 — 0 (BLACK to move) or 1 (WHITE to move).
    """
    N = played_c60.shape[0]
    K = len(patterns)
    out = np.zeros((N, K), dtype=np.uint8)
    played, even = _cell_indexers(played_c60, even_c60)
    mp = mover_parity.astype(np.uint8)

    def is_empty(cell):
        return ~played(cell)

    def is_opp(cell):
        return played(cell) & (even(cell) == mp)

    def is_mover(cell):
        return played(cell) & (even(cell) != mp)

    for j, pat in enumerate(patterns):
        active = is_empty(pat['target'])
        for o in pat['opponents']:
            active = active & is_opp(o)
            if not active.any():
                break
        else:
            active = active & is_mover(pat['terminal'])
        out[:, j] = active.astype(np.uint8)
    return out


def pattern_target(pattern):
    return pattern['target']


def patterns_by_target(patterns):
    """Group patterns by target cell.  Returns dict {cell: [pattern_idx, ...]}."""
    from collections import defaultdict
    d = defaultdict(list)
    for j, p in enumerate(patterns):
        d[p['target']].append(j)
    return dict(d)


def legal_from_state_probs_via_patterns(patterns, state_probs):
    """Compute per-cell legal-move probability by evaluating each of the 960
    flanking patterns on the SOFT state predictions, then combining via
    prob-OR per target cell.

    Under state class convention:
      class 0 = empty
      class 1 = mine (mover)
      class 2 = opp (opponent of mover)

    For pattern P targeting cell C:
      P(pattern P fires) = P(state[target]  = empty)
                          · Π P(state[opp]    = opp)   over opponents
                          · P(state[terminal] = mover)

    Per-cell combination:
      P(cell C legal) = 1 - Π (1 - P(pattern P fires))
                        over patterns targeting C

    Args:
      patterns: list of pattern dicts (as returned by load_patterns).
      state_probs: (N, 64, 3) numpy array or tensor of soft state
                    predictions.

    Returns:
      (N, 64) numpy array of per-cell legal-move probabilities.
    """
    if hasattr(state_probs, 'cpu'):
        state_np = state_probs.cpu().numpy()
    else:
        state_np = state_probs
    N = state_np.shape[0]
    K = len(patterns)
    pattern_p = np.zeros((N, K), dtype=np.float32)
    for j, pat in enumerate(patterns):
        p_empty = state_np[:, pat['target'], 0]
        p_term = state_np[:, pat['terminal'], 1]
        p_opps = np.ones(N, dtype=np.float32)
        for o in pat['opponents']:
            p_opps = p_opps * state_np[:, o, 2]
        pattern_p[:, j] = p_empty * p_opps * p_term

    by_tgt = patterns_by_target(patterns)
    per_cell = np.zeros((N, 64), dtype=np.float32)
    for cell, pattern_ids in by_tgt.items():
        p_per_pat = pattern_p[:, pattern_ids]
        # Numerically stable: log(1 - x) summation → 1 - exp(sum).
        # But 1 - Π (1 - p) directly is fine for p in [0, 1].
        per_cell[:, cell] = 1.0 - np.prod(1.0 - p_per_pat, axis=1)
    return per_cell
