"""Threshold-on-count hidden nodes for the interpretable-tree MLP pipeline.

Each count node is a strict {0/±1}-weighted conjunction over played_even
features: "at least K of a specified cell set are played at parity P".
Encoded as a single hidden unit with input weights in {0, +1} and bias
-(K - 0.5).  Fires (0/1) iff the count reaches the threshold.

Provides:
  - Structured pool (rows, columns, boxes, rings, per-cell 8-neighborhoods,
    board totals) — ~1500 nodes total.
  - Random pool (uniform random cell subsets with random thresholds).
  - Activation computation from the raw played_even features (efficient,
    vectorized).
  - Human-readable naming for each node.
"""
import numpy as np

CENTER_64 = {27, 28, 35, 36}
NON_CENTER_64 = sorted(set(range(64)) - CENTER_64)     # length 60
C64_TO_C60 = {c: i for i, c in enumerate(NON_CENTER_64)}


def _cells_to_c60_mask(cell_set):
    """Convert a set of c64 cell indices (skipping center cells) to a length-60
    boolean mask over c60 indices."""
    mask = np.zeros(60, dtype=bool)
    for c in cell_set:
        if c in C64_TO_C60:
            mask[C64_TO_C60[c]] = True
    return mask


def _algebraic(c64):
    """Convert a c64 index to 'A1' style name."""
    return 'ABCDEFGH'[c64 % 8] + str(c64 // 8 + 1)


# ------------------------------------------------------------------------------
# Structured pool
# ------------------------------------------------------------------------------

def build_structured_count_nodes():
    """Return a list of (name, c60_mask (bool[60]), parity (0|1|None), threshold)
    tuples.  parity=None means either parity (played at all)."""
    nodes = []

    # Full board totals (parity 0, 1, and any).
    all_cells = set(NON_CENTER_64)
    for p_label, parity in [('black', 0), ('white', 1), ('any', None)]:
        for k in range(1, 61):
            nodes.append(
                (f'board_{p_label}_k{k}',
                 _cells_to_c60_mask(all_cells), parity, k))

    # Rows (8 rows × 3 parity variants × ~8 thresholds).
    for r in range(8):
        row_cells = set(r * 8 + c for c in range(8)) - CENTER_64
        n_row = len(row_cells)
        if n_row == 0:
            continue
        for p_label, parity in [('black', 0), ('white', 1), ('any', None)]:
            for k in range(1, n_row + 1):
                nodes.append(
                    (f'row{r + 1}_{p_label}_k{k}',
                     _cells_to_c60_mask(row_cells), parity, k))

    # Columns (8 cols × 3 parity variants × ~8 thresholds).
    for c in range(8):
        col_cells = set(r * 8 + c for r in range(8)) - CENTER_64
        n_col = len(col_cells)
        if n_col == 0:
            continue
        for p_label, parity in [('black', 0), ('white', 1), ('any', None)]:
            for k in range(1, n_col + 1):
                nodes.append(
                    (f'col{"ABCDEFGH"[c]}_{p_label}_k{k}',
                     _cells_to_c60_mask(col_cells), parity, k))

    # 3×3 boxes centered on each non-center cell.
    for c64 in NON_CENTER_64:
        r0, c0 = c64 // 8, c64 % 8
        box_cells = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    box_cells.add(nr * 8 + nc)
        n_box = len(box_cells - CENTER_64)
        if n_box == 0:
            continue
        alg = _algebraic(c64)
        for p_label, parity in [('black', 0), ('white', 1), ('any', None)]:
            for k in range(1, n_box + 1):
                nodes.append(
                    (f'box{alg}_{p_label}_k{k}',
                     _cells_to_c60_mask(box_cells), parity, k))

    return nodes


# ------------------------------------------------------------------------------
# Random pool
# ------------------------------------------------------------------------------

def build_random_count_nodes(n_nodes, seed=42, min_size=3, max_size=15):
    """Generate n_nodes random count-node specs with random cell subsets
    (size in [min_size, max_size]) and random thresholds."""
    rng = np.random.RandomState(seed)
    nodes = []
    for i in range(n_nodes):
        size = rng.randint(min_size, max_size + 1)
        cell_ids = rng.choice(NON_CENTER_64, size=size, replace=False)
        cells = set(int(x) for x in cell_ids)
        parity = int(rng.randint(0, 3))       # 0, 1, or 2 (2 = "any")
        parity_val = None if parity == 2 else parity
        threshold = int(rng.randint(1, size + 1))
        p_label = ('black', 'white', 'any')[parity]
        nodes.append(
            (f'rand{i:05d}_{p_label}_s{size}_k{threshold}',
             _cells_to_c60_mask(cells), parity_val, threshold))
    return nodes


# ------------------------------------------------------------------------------
# Activation computation
# ------------------------------------------------------------------------------

def compute_count_activations(count_nodes, played_np, even_np, chunk=8192):
    """For each count node, compute a (N,) bool activation vector.

    Args:
      count_nodes: list of (name, c60_mask, parity, threshold) tuples.
      played_np:   (N, 60) float32 or bool — played bits per non-center cell.
      even_np:     (N, 60) float32 or bool — even bits per non-center cell.
      chunk:       row chunk size for memory efficiency.

    Returns:
      (N, len(count_nodes)) bool activation matrix.
    """
    N = played_np.shape[0]
    H = len(count_nodes)
    out = np.zeros((N, H), dtype=np.bool_)
    played_np = played_np.astype(np.uint8)
    even_np = even_np.astype(np.uint8)

    for i in range(0, N, chunk):
        pl = played_np[i:i + chunk]      # (b, 60)
        ev = even_np[i:i + chunk]

        # Precompute indicators per parity variant.
        # ind_p0[i, c] = 1 iff cell c was played at parity 0 (played AND even)
        # ind_p1[i, c] = 1 iff cell c was played at parity 1 (played AND !even)
        # ind_any[i, c] = played[i, c]
        ind_p0 = pl * ev
        ind_p1 = pl * (1 - ev)

        for j, (name, mask, parity, k) in enumerate(count_nodes):
            if parity == 0:
                counts = ind_p0[:, mask].sum(axis=1)
            elif parity == 1:
                counts = ind_p1[:, mask].sum(axis=1)
            else:
                counts = pl[:, mask].sum(axis=1)
            out[i:i + chunk, j] = counts >= k

    return out


def count_node_names(count_nodes):
    """Return list of names for each count node."""
    return [n[0] for n in count_nodes]
