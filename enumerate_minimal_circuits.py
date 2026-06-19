"""Enumerate all minimal sufficient input collections AND minimal hidden subsets
for one (pattern, parity) instance of a flanking-pattern detector MLP.

For pattern p:
1. Find all (cell, state) candidates with W_eff[p, i] = W2[p] @ W1[:, i] > 0
   (sound pre-filter to inputs that COULD contribute via the linear path).
2. Enumerate K-tuples for K = 1..max_K:
   - Skip tuples that violate the cell constraint (two states for same cell)
   - Skip supersets of already-found minimal sets (anti-subset pruning)
   - For each surviving candidate set, run a real forward pass
   - If pred_p > 0, record as minimal (sufficiency + smallest-first means minimal)
3. For each minimal input set S:
   a. Compute h(S) = ReLU(W1 @ x_S + b1)
   b. Identify positively-contributing firing hidden nodes
   c. Enumerate minimal subsets of those nodes whose contributions sum > -b2[p]
      (at K_min_hidden; can extend with --max-k-hidden)

Output is a gzipped pickle with the raw data needed for downstream per-hidden-node
aggregation (which patterns each j appears in, which input collections, etc).

Usage:
    python enumerate_minimal_circuits.py \\
        --ckpt PATH \\
        --features {playedeven|movegrid} \\
        --pattern-idx N \\
        --parity {even|odd} \\
        --max-k 6 \\
        --output OUT.pkl.gz
"""

import argparse
import gzip
import pickle
import sys
import time
from itertools import combinations
import numpy as np
import torch


def get_candidates(W1, W2, b2, p_idx, features_type):
    """Compute linear effective contribution for each possible (cell, state).
    Returns sorted list of (cell, state, w_eff) with w_eff > 0.

    For playedeven: state 0 = black (played=1, even=0); state 1 = white (played=1, even=1)
    For movegrid:   state t = "cell played on turn t" (t in 0..59)
    """
    candidates = []
    if features_type == 'playedeven':
        for cell in range(60):
            # State 0: black (only played feature active)
            w_eff = float(W2[p_idx] @ W1[:, cell])
            if w_eff > 0:
                candidates.append((cell, 0, w_eff))
            # State 1: white (both played and even features active)
            w_eff = float(W2[p_idx] @ (W1[:, cell] + W1[:, 60 + cell]))
            if w_eff > 0:
                candidates.append((cell, 1, w_eff))
    elif features_type == 'movegrid':
        for cell in range(60):
            for turn in range(60):
                feat_idx = cell * 60 + turn
                w_eff = float(W2[p_idx] @ W1[:, feat_idx])
                if w_eff > 0:
                    candidates.append((cell, turn, w_eff))
    candidates.sort(key=lambda c: -c[2])
    return candidates


def apply_state(x, cell, state, features_type):
    if features_type == 'playedeven':
        x[cell] = 1.0
        if state == 1:  # white
            x[60 + cell] = 1.0
    elif features_type == 'movegrid':
        x[cell * 60 + state] = 1.0


def build_input_vector(candidates, S, input_dim, features_type):
    x = np.zeros(input_dim, dtype=np.float32)
    for i in S:
        cell, state, _ = candidates[i]
        apply_state(x, cell, state, features_type)
    return x


def find_minimal_input_sets(W1, b1, W2, b2, p_idx, candidates, input_dim,
                             features_type, max_K, verbose=False):
    """Enumerate all minimal sufficient input sets up to size max_K.
    Returns list of frozensets of candidate indices, plus per-size counts and
    a per-size elapsed-time dict."""
    n_cands = len(candidates)
    cand_cells = np.array([c[0] for c in candidates], dtype=np.int64)

    minimal_sets = []  # list of frozensets
    by_size = {}
    elapsed_per_K = {}

    for K in range(1, max_K + 1):
        t0 = time.time()
        new_minimal = []
        n_checked = 0
        n_skipped_cell = 0
        n_skipped_subset = 0
        n_forwards = 0

        for combo in combinations(range(n_cands), K):
            n_checked += 1
            # Cell constraint
            cells = cand_cells[list(combo)]
            if len(np.unique(cells)) < K:
                n_skipped_cell += 1
                continue
            s = frozenset(combo)
            # Anti-subset pruning
            subsumed = False
            for ms in minimal_sets:
                if ms <= s:
                    subsumed = True
                    break
            if subsumed:
                n_skipped_subset += 1
                continue
            # Forward pass
            n_forwards += 1
            x = build_input_vector(candidates, combo, input_dim, features_type)
            z = W1 @ x + b1
            h = np.maximum(z, 0)
            pred = W2[p_idx] @ h + b2[p_idx]
            if pred > 0:
                new_minimal.append(s)

        minimal_sets.extend(new_minimal)
        by_size[K] = len(new_minimal)
        elapsed_per_K[K] = time.time() - t0

        if verbose:
            print(f"  K={K}: checked {n_checked}, "
                  f"cell-collision skip {n_skipped_cell}, "
                  f"subset-skip {n_skipped_subset}, "
                  f"forwards {n_forwards}, "
                  f"new minimal {len(new_minimal)}, "
                  f"time {elapsed_per_K[K]:.1f}s", flush=True)

    return minimal_sets, by_size, elapsed_per_K


def classify_inhibitors(W2, b2, p_idx, h_S, h_default):
    """Classify hidden nodes by their inhibitory role for pattern p under input S.

    W2[p, j] < 0 means j inhibits p when firing.

    Returns dict:
      inhibitors_firing_S         : j with W2[p,j]<0 AND h_j(S)>0
                                    (currently pushing p down)
      inhibitors_silent_S         : j with W2[p,j]<0 AND h_j(S)==0
                                    (currently silent)
      inhibitors_firing_default   : j with W2[p,j]<0 AND h_j(default)>0
                                    (would inhibit on empty input)
      silenced_by_S               : j with W2[p,j]<0 AND h_j(default)>0 AND h_j(S)==0
                                    (S specifically silenced these inhibitors)
      newly_activated_by_S        : j with W2[p,j]<0 AND h_j(default)==0 AND h_j(S)>0
                                    (S brought these inhibitors online)
      inhibition_overcome         : sum_{j in inhibitors_firing_S} |W2[p,j] * h_j(S)|
                                    (magnitude of current downward pressure)
    """
    W2_p = W2[p_idx]
    inh_mask = W2_p < 0
    firing_S_mask = h_S > 0
    firing_default_mask = h_default > 0

    inh_firing_S = np.where(inh_mask & firing_S_mask)[0]
    inh_silent_S = np.where(inh_mask & ~firing_S_mask)[0]
    inh_firing_default = np.where(inh_mask & firing_default_mask)[0]
    silenced_by_S = np.where(inh_mask & firing_default_mask & ~firing_S_mask)[0]
    newly_activated_by_S = np.where(inh_mask & ~firing_default_mask & firing_S_mask)[0]

    inhibition_overcome = float(np.abs(W2_p[inh_firing_S] * h_S[inh_firing_S]).sum())

    return {
        'inhibitors_firing_S': frozenset(int(j) for j in inh_firing_S),
        'inhibitors_silent_S': frozenset(int(j) for j in inh_silent_S),
        'inhibitors_firing_default': frozenset(int(j) for j in inh_firing_default),
        'silenced_by_S': frozenset(int(j) for j in silenced_by_S),
        'newly_activated_by_S': frozenset(int(j) for j in newly_activated_by_S),
        'inhibition_overcome': inhibition_overcome,
    }


def find_minimal_hidden_subsets(W2, b2, p_idx, h, max_K_hidden=None):
    """For a fixed hidden activation vector h, find minimal subsets of
    positive-contributing firing nodes whose contributions sum > -b2[p].

    If max_K_hidden is None, enumerate only at K_min_hidden (smallest size).
    Otherwise enumerate at K_min..max_K_hidden.

    Returns:
        K_min_hidden (or None if no sufficient subset exists; or 0 if bias > 0)
        list of frozensets of hidden node indices
    """
    threshold = -float(b2[p_idx])
    if threshold <= 0:
        return 0, []  # fires by default (bias positive)

    contribs = W2[p_idx] * h
    pos_mask = contribs > 0
    pos_idx_global = np.where(pos_mask)[0]
    if len(pos_idx_global) == 0:
        return None, []
    pos_c = contribs[pos_idx_global]
    order = np.argsort(pos_c)[::-1]
    cs_sorted = pos_c[order]
    idx_sorted = pos_idx_global[order]

    cumsum = np.cumsum(cs_sorted)
    where = np.where(cumsum > threshold)[0]
    if len(where) == 0:
        return None, []
    K_min = int(where[0]) + 1

    if max_K_hidden is None:
        max_K_hidden = K_min

    # Enumerate minimal subsets
    subsets = []
    for K in range(K_min, max_K_hidden + 1):
        for combo in combinations(range(len(cs_sorted)), K):
            sum_c = sum(cs_sorted[i] for i in combo)
            if sum_c <= threshold:
                continue
            s = frozenset(int(idx_sorted[i]) for i in combo)
            # Anti-subset pruning across sizes
            subsumed = False
            for ms in subsets:
                if ms <= s:
                    subsumed = True
                    break
            if subsumed:
                continue
            subsets.append(s)

    return K_min, subsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--features', choices=['playedeven', 'movegrid'], required=True)
    ap.add_argument('--pattern-idx', type=int, required=True)
    ap.add_argument('--parity', choices=['even', 'odd'], required=True)
    ap.add_argument('--max-k', type=int, default=6,
                    help='Maximum K for input minimal-set enumeration.')
    ap.add_argument('--max-k-hidden', type=int, default=None,
                    help='Max K for hidden minimal-set enumeration (default: only at K_min_hidden).')
    ap.add_argument('--output', required=True)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--save-h', action='store_true',
                    help='Save full h vector per input collection (large; default off).')
    args = ap.parse_args()

    print(f"Loading {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    sd = ckpt[args.parity]
    W1 = sd['net.0.weight'].numpy()
    b1 = sd['net.0.bias'].numpy()
    W2 = sd['net.2.weight'].numpy()
    b2 = sd['net.2.bias'].numpy()
    H, input_dim = W1.shape
    n_patterns = W2.shape[0]
    assert 0 <= args.pattern_idx < n_patterns

    print(f"Pattern {args.pattern_idx}/{n_patterns}, parity={args.parity}, "
          f"features={args.features}, H={H}, input_dim={input_dim}", flush=True)
    print(f"b2[p]={float(b2[args.pattern_idx]):.4f}; "
          f"{'fires by default' if b2[args.pattern_idx] > 0 else 'needs input'}", flush=True)

    candidates = get_candidates(W1, W2, b2, args.pattern_idx, args.features)
    print(f"Positive-contribution candidates: {len(candidates)}", flush=True)

    # Compute default hidden activation (empty input) once -- used for inhibitor classification
    h_default = np.maximum(b1.copy(), 0)

    if b2[args.pattern_idx] > 0:
        # Fires by default; "minimal sufficient input" is the empty set
        result = {
            'pattern_idx': args.pattern_idx, 'parity': args.parity,
            'features_type': args.features, 'max_k': args.max_k,
            'status': 'fires_by_default', 'bias': float(b2[args.pattern_idx]),
            'candidates': candidates,
            'minimal_input_sets': [frozenset()],
            'minimal_input_by_size': {0: 1},
            'per_S_data': {},
        }
        # Still process the empty input to get the default hidden circuit
        x = np.zeros(input_dim, dtype=np.float32)
        h = np.maximum(W1 @ x + b1, 0)
        K_min_h, hidden_subsets = find_minimal_hidden_subsets(
            W2, b2, args.pattern_idx, h, args.max_k_hidden)
        inhibitor_info = classify_inhibitors(W2, b2, args.pattern_idx, h, h_default)
        result['per_S_data'][frozenset()] = {
            'h_S': h.astype(np.float32) if args.save_h else None,
            'firing_count': int((h > 0).sum()),
            'positive_contrib_nodes': int(((W2[args.pattern_idx] * h) > 0).sum()),
            'K_min_hidden': K_min_h,
            'hidden_minimal_subsets': hidden_subsets,
            'inhibitor_info': inhibitor_info,
        }
        with gzip.open(args.output, 'wb') as f:
            pickle.dump(result, f)
        print(f"Done (fires by default). Saved to {args.output}", flush=True)
        return

    if len(candidates) == 0:
        result = {
            'pattern_idx': args.pattern_idx, 'parity': args.parity,
            'features_type': args.features, 'max_k': args.max_k,
            'status': 'no_positive_candidates',
            'candidates': [], 'minimal_input_sets': [],
            'minimal_input_by_size': {}, 'per_S_data': {},
        }
        with gzip.open(args.output, 'wb') as f:
            pickle.dump(result, f)
        print(f"Done (no positive candidates). Saved to {args.output}", flush=True)
        return

    # Enumerate minimal input sets
    t0 = time.time()
    minimal_input_sets, by_size, elapsed_per_K = find_minimal_input_sets(
        W1, b1, W2, b2, args.pattern_idx, candidates, input_dim,
        args.features, args.max_k, verbose=args.verbose)
    t_input = time.time() - t0
    print(f"\nFound {len(minimal_input_sets)} minimal input sets in {t_input:.1f}s", flush=True)
    for K, c in by_size.items():
        print(f"  K={K}: {c} minimal", flush=True)

    if len(minimal_input_sets) == 0:
        result = {
            'pattern_idx': args.pattern_idx, 'parity': args.parity,
            'features_type': args.features, 'max_k': args.max_k,
            'status': 'no_minimal_within_max_k',
            'candidates': candidates,
            'minimal_input_sets': [], 'minimal_input_by_size': by_size,
            'per_S_data': {}, 'elapsed_input_per_K': elapsed_per_K,
            'time_input_search': t_input,
        }
        with gzip.open(args.output, 'wb') as f:
            pickle.dump(result, f)
        print(f"Done (no minimal within K={args.max_k}). Saved to {args.output}", flush=True)
        return

    # For each minimal input set: forward to hidden, find minimal hidden subsets
    print(f"\nProcessing {len(minimal_input_sets)} input sets through hidden...", flush=True)
    t0 = time.time()
    per_S_data = {}
    for s_idx, s in enumerate(minimal_input_sets):
        x = build_input_vector(candidates, s, input_dim, args.features)
        z = W1 @ x + b1
        h = np.maximum(z, 0)
        K_min_h, hidden_subsets = find_minimal_hidden_subsets(
            W2, b2, args.pattern_idx, h, args.max_k_hidden)
        inhibitor_info = classify_inhibitors(W2, b2, args.pattern_idx, h, h_default)
        per_S_data[s] = {
            'h_S': h.astype(np.float32) if args.save_h else None,
            'firing_count': int((h > 0).sum()),
            'positive_contrib_nodes': int(((W2[args.pattern_idx] * h) > 0).sum()),
            'K_min_hidden': K_min_h,
            'hidden_minimal_subsets': hidden_subsets,
            'inhibitor_info': inhibitor_info,
        }
        if args.verbose and (s_idx + 1) % 100 == 0:
            print(f"  {s_idx+1}/{len(minimal_input_sets)}", flush=True)
    t_hidden = time.time() - t0
    print(f"Done in {t_hidden:.1f}s", flush=True)

    result = {
        'pattern_idx': args.pattern_idx, 'parity': args.parity,
        'features_type': args.features, 'max_k': args.max_k,
        'max_k_hidden': args.max_k_hidden,
        'status': 'complete', 'bias': float(b2[args.pattern_idx]),
        'candidates': candidates,
        'minimal_input_sets': minimal_input_sets,
        'minimal_input_by_size': by_size,
        'per_S_data': per_S_data,
        'time_input_search': t_input,
        'time_hidden_search': t_hidden,
        'elapsed_input_per_K': elapsed_per_K,
    }
    with gzip.open(args.output, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved to {args.output}", flush=True)


if __name__ == '__main__':
    main()
