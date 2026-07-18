"""Order-aware hidden nodes for the interpretable-tree MLP pipeline.

Each builder computes a (N, H) activation matrix and per-unit metadata
directly from the (N, 60, 60) movegrid tensor — no {0/±1}-weight mask
abstraction (some features can't be expressed that way).

Feature families:
  - turn_bucket:    "cell c played in turns [k*S, (k+1)*S) at parity P"
                    60 cells × #buckets × 3 parities.
  - recency:        "cell c played within last K turns of current turn T"
                    60 cells × #Ks.
  - ordinal:        "cell c was the K-th move"  (identical to raw movegrid).
                    60 cells × 60 turns = 3600.
  - pairwise_order: "cell A played before cell B"
                    restricted to spatially-close pairs (Chebyshev ≤ D).
  - streak:         "cells A, B played within N turns of each other"
                    restricted to spatially-close pairs.

Activations are uint8 (0/1) by default, float32 excess-above-threshold under
use_relu=True for the count-based families (turn_bucket, recency).  Binary
families (ordinal, pairwise_order, streak) are always 0/1 — under use_relu
they are simply cast to float32 to interoperate with the ReLU-activated
hidden layer.
"""
import numpy as np

CENTER_64 = {27, 28, 35, 36}
NON_CENTER_64 = sorted(set(range(64)) - CENTER_64)  # length 60
C64_TO_C60 = {c: i for i, c in enumerate(NON_CENTER_64)}
C60_TO_C64 = {i: c for c, i in C64_TO_C60.items()}


def _algebraic_c60(c60):
    c64 = C60_TO_C64[c60]
    return 'ABCDEFGH'[c64 % 8] + str(c64 // 8 + 1)


def _act_dtype(use_relu):
    return np.float32 if use_relu else np.uint8


def _threshold_k1(counts, use_relu):
    """Turn (N,) integer counts into K=1 activation.  Under step: >=1.
    Under ReLU: max(0, count - 0.5)."""
    if use_relu:
        return np.maximum(0.0, counts.astype(np.float32) - 0.5)
    return (counts >= 1).astype(np.uint8)


def movegrid_from_flat(Xnp, offset=121):
    """Reshape the 3600-column movegrid slice of Xnp to (N, 60, 60).

    Layout matches playedeven_features: feat[121 + t * 60 + i] = 1 iff move
    at turn t was cell c60=i.  So reshape (N, 60, 60) with (turn, cell60).
    """
    N = Xnp.shape[0]
    return Xnp[:, offset:offset + 3600].reshape(N, 60, 60)


def _turn_of(movegrid):
    """For each (position, cell60), return the turn c was played (or -1)."""
    N = movegrid.shape[0]
    played = movegrid.any(axis=1)                     # (N, 60)
    # argmax along turn axis gives first True index if any played.
    first_turn = movegrid.argmax(axis=1).astype(np.int16)  # (N, 60)
    first_turn[~played] = -1
    return first_turn


# ------------------------------------------------------------------------------
# Turn buckets — "cell c played in turn window [t_lo, t_hi) at parity P"
# ------------------------------------------------------------------------------

def build_turn_bucket(movegrid, bucket_size=10, use_relu=False):
    N = movegrid.shape[0]
    n_buckets = (60 + bucket_size - 1) // bucket_size
    parities = [(None, 'any'), (0, 'black'), (1, 'white')]
    H = 60 * n_buckets * len(parities)
    out = np.zeros((N, H), dtype=_act_dtype(use_relu))
    meta = []
    turns = np.arange(60)
    idx = 0
    for cell60 in range(60):
        slab_c = movegrid[:, :, cell60]              # (N, 60)
        for k in range(n_buckets):
            t_lo = k * bucket_size
            t_hi = min((k + 1) * bucket_size, 60)
            window = (turns >= t_lo) & (turns < t_hi)
            for parity, p_label in parities:
                if parity is None:
                    mask = window
                else:
                    mask = window & (turns % 2 == parity)
                if not mask.any():
                    idx += 1
                    continue
                counts = slab_c[:, mask].sum(axis=1)
                out[:, idx] = _threshold_k1(counts, use_relu)
                meta.append({
                    'kind': 'turn_bucket',
                    'name': f'tb_{_algebraic_c60(cell60)}_b{k}_{p_label}',
                    'cell60': cell60,
                    'bucket': (t_lo, t_hi),
                    'parity': parity,
                })
                idx += 1
    return out, meta


# ------------------------------------------------------------------------------
# Recency — "cell c played within last K turns of current turn T"
# ------------------------------------------------------------------------------

def build_recency(movegrid, current_turns, Ks=(1, 2, 5, 10, 20),
                    use_relu=False):
    N = movegrid.shape[0]
    turns = np.arange(60)[None, :]                    # (1, 60)
    T = current_turns[:, None].astype(np.int32)       # (N, 1)
    H = 60 * len(Ks)
    out = np.zeros((N, H), dtype=_act_dtype(use_relu))
    meta = []
    idx = 0
    for K in Ks:
        # (N, 60) mask over turns: t ∈ [T-K, T)
        m = ((turns >= (T - K)) & (turns < T)).astype(np.int32)
        # (N, 60) — for each cell, count of movegrid[n, t, c] where mask[n, t].
        counts = np.einsum('nt,ntc->nc', m, movegrid.astype(np.int32))
        for cell60 in range(60):
            out[:, idx] = _threshold_k1(counts[:, cell60], use_relu)
            meta.append({
                'kind': 'recency',
                'name': f'rec_{_algebraic_c60(cell60)}_K{K}',
                'cell60': cell60,
                'K': K,
            })
            idx += 1
    return out, meta


# ------------------------------------------------------------------------------
# Ordinal — "cell c was the K-th move" (== raw movegrid)
# ------------------------------------------------------------------------------

def build_ordinal(movegrid, use_relu=False):
    N = movegrid.shape[0]
    out = movegrid.reshape(N, 60 * 60).astype(_act_dtype(use_relu))
    meta = []
    for k in range(60):
        for cell60 in range(60):
            meta.append({
                'kind': 'ordinal',
                'name': f'ord_{_algebraic_c60(cell60)}_t{k}',
                'cell60': cell60,
                'turn': k,
            })
    return out, meta


# ------------------------------------------------------------------------------
# Pair restriction — spatial neighbours only
# ------------------------------------------------------------------------------

def _spatially_close_pairs(max_chebyshev=2, ordered=True):
    """Non-center cell pairs (a, b) within Chebyshev distance <= D.  When
    ordered, includes both (a, b) and (b, a) — needed for pairwise-order.
    Excludes a == b."""
    pairs = []
    for a60 in range(60):
        a64 = C60_TO_C64[a60]
        ra, ca = a64 // 8, a64 % 8
        for b60 in range(60):
            if b60 == a60:
                continue
            b64 = C60_TO_C64[b60]
            rb, cb = b64 // 8, b64 % 8
            if abs(ra - rb) <= max_chebyshev and abs(ca - cb) <= max_chebyshev:
                if ordered or a60 < b60:
                    pairs.append((a60, b60))
    return pairs


# ------------------------------------------------------------------------------
# Pairwise order — "A played before B" (both played, turn(A) < turn(B))
# ------------------------------------------------------------------------------

def build_pairwise_order(movegrid, max_chebyshev=2, use_relu=False):
    pairs = _spatially_close_pairs(max_chebyshev, ordered=True)
    N = movegrid.shape[0]
    H = len(pairs)
    out = np.zeros((N, H), dtype=_act_dtype(use_relu))
    turn_of = _turn_of(movegrid)                     # (N, 60), -1 if unplayed
    meta = []
    for idx, (a, b) in enumerate(pairs):
        ta, tb = turn_of[:, a], turn_of[:, b]
        fires = (ta >= 0) & (tb >= 0) & (ta < tb)
        if use_relu:
            out[:, idx] = fires.astype(np.float32)
        else:
            out[:, idx] = fires
        meta.append({
            'kind': 'pairwise_order',
            'name': f'{_algebraic_c60(a)}_before_{_algebraic_c60(b)}',
            'a60': a, 'b60': b,
            'max_chebyshev': max_chebyshev,
        })
    return out, meta


# ------------------------------------------------------------------------------
# Streak / co-timing — "A and B played within N turns of each other"
# ------------------------------------------------------------------------------

def build_streak(movegrid, max_chebyshev=2, N_gap=3, use_relu=False):
    pairs = _spatially_close_pairs(max_chebyshev, ordered=False)
    N = movegrid.shape[0]
    H = len(pairs)
    out = np.zeros((N, H), dtype=_act_dtype(use_relu))
    turn_of = _turn_of(movegrid)
    meta = []
    for idx, (a, b) in enumerate(pairs):
        ta, tb = turn_of[:, a], turn_of[:, b]
        both = (ta >= 0) & (tb >= 0)
        fires = both & (np.abs(ta.astype(np.int32)
                                 - tb.astype(np.int32)) <= N_gap)
        if use_relu:
            out[:, idx] = fires.astype(np.float32)
        else:
            out[:, idx] = fires
        meta.append({
            'kind': 'streak',
            'name': f'streak_{_algebraic_c60(a)}_{_algebraic_c60(b)}_N{N_gap}',
            'a60': a, 'b60': b, 'N_gap': N_gap,
            'max_chebyshev': max_chebyshev,
        })
    return out, meta
