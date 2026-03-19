"""Generate spatially impossible Othello flanking patterns via cross-wiring.

Cross-wiring swaps ray bodies (opponents + terminal) between individual
patterns of the same direction and length but from different board halves.
This breaks spatial properties that a real grid guarantees:

  - Embeddability: no consistent 8x8 grid assignment satisfies all constraints
  - Reflexivity: A→C flanks B, but C→A may not flank B
  - Transitivity: "right from A" → B, "right from B" → C, but C ≠ A+2
  - Collinearity: length-2 and length-3 patterns from same cell disagree on ray
  - Commutativity: right-then-down ≠ down-then-right
  - Distance symmetry: A references B at distance 2, B may never reference A
  - Neighborhood consistency: two directions from A may point to same cell
  - Betweenness: intermediate cells in long patterns may not appear in shorter ones

Spatial impossibility is measured by Grid Embedding Residual (GER):
  Extract all consecutive-pair constraints from patterns, solve least-squares
  for (x,y) coordinates. GER = mean squared residual. Standard Othello → 0.

Usage:
  python generate_impossible_patterns.py                    # sweep beta, print table
  python generate_impossible_patterns.py --target-ger 1.0 2.0 4.0  # search for targets
  python generate_impossible_patterns.py --target-ger 1.0 --save-dir experiments/impossible/patterns
"""

import argparse
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hand_crafted_flanking import enumerate_flanking_patterns, DIRECTIONS


def patterns_to_bytes(patterns):
    """Serialize patterns as a flat byte array."""
    buf = bytearray()
    for p in patterns:
        opps = p['opponents']
        buf.append(len(opps))
        buf.append(p['target'])
        for o in opps:
            buf.append(o)
        buf.append(p['terminal'])
    return bytes(buf)


def compressed_size(data_bytes):
    """Return gzip compressed size in bytes."""
    return len(gzip.compress(data_bytes, compresslevel=9))


def cross_wire_patterns(patterns, beta, rng):
    """Cross-wire fraction beta of patterns between board halves.

    For each individual pattern, with probability beta, swap its ray body
    (opponents + terminal) with another pattern of the same direction and
    length whose target is in the opposite board half (rows 0-3 vs 4-7).

    Args:
        patterns: list of pattern dicts with 'target', 'opponents', 'terminal',
                  'direction', 'length' keys
        beta: fraction of patterns to cross-wire (0.0 to 1.0)
        rng: numpy RandomState

    Returns:
        New list of pattern dicts (originals not modified)
    """
    patterns = [dict(p, opponents=list(p['opponents'])) for p in patterns]

    # Group by (direction, length)
    from collections import defaultdict
    groups = defaultdict(lambda: ([], []))  # (half_A, half_B)
    for i, p in enumerate(patterns):
        key = (p['direction'], p['length'])
        if p['target'] // 8 < 4:
            groups[key][0].append(i)
        else:
            groups[key][1].append(i)

    for (direction, length), (half_A, half_B) in groups.items():
        n_swap = int(beta * min(len(half_A), len(half_B)))
        if n_swap == 0:
            continue

        swap_A = rng.choice(half_A, size=n_swap, replace=False)
        swap_B = rng.choice(half_B, size=n_swap, replace=False)

        for a_idx, b_idx in zip(swap_A, swap_B):
            pa, pb = patterns[a_idx], patterns[b_idx]
            # Swap opponents and terminal (keep target)
            pa_opps, pa_term = pa['opponents'], pa['terminal']
            pb_opps, pb_term = pb['opponents'], pb['terminal']
            pa['opponents'], pa['terminal'] = list(pb_opps), pb_term
            pb['opponents'], pb['terminal'] = list(pa_opps), pa_term

    return patterns


def extract_constraints(patterns):
    """Extract positional constraints from patterns.

    Each pattern with direction (dr, dc) and cells [target, opp1, ..., oppk, terminal]
    yields k+1 constraints: pos(cell_{i+1}) - pos(cell_i) = (dr, dc).

    Returns:
        List of (cell_from, cell_to, dr, dc) tuples
    """
    constraints = []
    for p in patterns:
        dr, dc = p['direction']
        cells = [p['target']] + list(p['opponents']) + [p['terminal']]
        for i in range(len(cells) - 1):
            constraints.append((cells[i], cells[i + 1], dr, dc))
    return constraints


def compute_ger(patterns):
    """Compute Grid Embedding Residual.

    Builds least-squares system from pattern constraints, solves for optimal
    (x, y) coordinates, returns mean squared residual across all constraints.

    Standard Othello → GER = 0.0.
    """
    constraints = extract_constraints(patterns)
    if not constraints:
        return 0.0

    n_cells = 64
    n_con = len(constraints)

    # Separate x and y systems
    # For x: x(to) - x(from) = dr  →  A_x @ x = b_x
    # For y: y(to) - y(from) = dc  →  A_y @ y = b_y
    # Pin cell 0: x(0) = 0, y(0) = 0 (add as constraint with high weight)

    A = np.zeros((n_con + 1, n_cells), dtype=np.float64)
    b_x = np.zeros(n_con + 1, dtype=np.float64)
    b_y = np.zeros(n_con + 1, dtype=np.float64)

    for i, (c_from, c_to, dr, dc) in enumerate(constraints):
        A[i, c_to] = 1.0
        A[i, c_from] = -1.0
        b_x[i] = dr
        b_y[i] = dc

    # Pin cell 0 with high weight
    pin_weight = 100.0
    A[n_con, 0] = pin_weight
    b_x[n_con] = 0.0
    b_y[n_con] = 0.0

    # Solve least-squares
    x_coords, res_x, _, _ = np.linalg.lstsq(A, b_x, rcond=None)
    y_coords, res_y, _, _ = np.linalg.lstsq(A, b_y, rcond=None)

    # Compute residuals manually (lstsq may not return res for underdetermined)
    residual_x = A[:n_con] @ x_coords - b_x[:n_con]
    residual_y = A[:n_con] @ y_coords - b_y[:n_con]

    mse = (np.sum(residual_x ** 2) + np.sum(residual_y ** 2)) / (2 * n_con)
    return float(mse)


def search_target_ger(base_patterns, target_ger, seed=42, max_iter=5000,
                      tolerance=0.01):
    """Hill-climbing search for a cross-wired pattern set with GER ≈ target.

    Starts from standard patterns, randomly proposes cross-wire swaps,
    accepts if GER moves closer to target.

    Args:
        base_patterns: standard flanking patterns
        target_ger: desired GER value
        seed: random seed
        max_iter: maximum iterations
        tolerance: acceptable distance from target

    Returns:
        (patterns, achieved_ger, n_swaps_made)
    """
    if target_ger == 0.0:
        return list(base_patterns), 0.0, 0

    rng = np.random.RandomState(seed)
    patterns = [dict(p, opponents=list(p['opponents'])) for p in base_patterns]

    # Group by (direction, length) for swap candidates
    from collections import defaultdict
    groups = defaultdict(lambda: ([], []))
    for i, p in enumerate(patterns):
        key = (p['direction'], p['length'])
        if p['target'] // 8 < 4:
            groups[key][0].append(i)
        else:
            groups[key][1].append(i)

    # Filter to groups with candidates in both halves
    swap_groups = [(k, hA, hB) for k, (hA, hB) in groups.items()
                   if len(hA) > 0 and len(hB) > 0]

    current_ger = compute_ger(patterns)
    n_swaps = 0

    for iteration in range(max_iter):
        if abs(current_ger - target_ger) < tolerance:
            break

        # Pick a random group and propose a swap
        key, half_A, half_B = swap_groups[rng.randint(len(swap_groups))]
        a_idx = half_A[rng.randint(len(half_A))]
        b_idx = half_B[rng.randint(len(half_B))]

        pa, pb = patterns[a_idx], patterns[b_idx]

        # Save old values
        old_a_opps, old_a_term = list(pa['opponents']), pa['terminal']
        old_b_opps, old_b_term = list(pb['opponents']), pb['terminal']

        # Swap
        pa['opponents'], pa['terminal'] = list(old_b_opps), old_b_term
        pb['opponents'], pb['terminal'] = list(old_a_opps), old_a_term

        new_ger = compute_ger(patterns)

        if abs(new_ger - target_ger) < abs(current_ger - target_ger):
            # Accept
            current_ger = new_ger
            n_swaps += 1
        else:
            # Reject — undo swap
            pa['opponents'], pa['terminal'] = old_a_opps, old_a_term
            pb['opponents'], pb['terminal'] = old_b_opps, old_b_term

    return patterns, current_ger, n_swaps


def count_dead_patterns(patterns):
    """Count patterns that can never fire (target appears in opponents or terminal)."""
    dead = 0
    for p in patterns:
        t = p['target']
        if t in p['opponents'] or t == p['terminal']:
            dead += 1
    return dead


def main():
    parser = argparse.ArgumentParser(description="Generate spatially impossible Othello patterns")
    parser.add_argument("--target-ger", type=float, nargs='+',
                        help="Target GER values to search for")
    parser.add_argument("--sweep-beta", action='store_true', default=True,
                        help="Sweep beta from 0 to 1 (default)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save pattern sets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of seeds for beta sweep")
    args = parser.parse_args()

    base_patterns = enumerate_flanking_patterns()
    print(f"Standard patterns: {len(base_patterns)}")

    std_bytes = patterns_to_bytes(base_patterns)
    std_gz = compressed_size(std_bytes)
    print(f"Standard gzip size: {std_gz} bytes")
    print(f"Standard GER: {compute_ger(base_patterns):.6f}")
    print()

    if args.target_ger:
        # Target-based search
        print(f"{'Target GER':>12} {'Achieved GER':>14} {'Gzip (B)':>10} {'Dead':>6} {'Swaps':>7}")
        print("-" * 55)

        results = {}
        for target in args.target_ger:
            patterns, achieved, n_swaps = search_target_ger(
                base_patterns, target, seed=args.seed)
            gz = compressed_size(patterns_to_bytes(patterns))
            dead = count_dead_patterns(patterns)
            print(f"{target:>12.2f} {achieved:>14.4f} {gz:>10} {dead:>6} {n_swaps:>7}")

            key = f"ger_{target:.2f}"
            results[key] = {
                'target_ger': target,
                'achieved_ger': achieved,
                'gzip_size': gz,
                'n_patterns': len(patterns),
                'n_dead': dead,
                'n_swaps': n_swaps,
            }

            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                pat_path = os.path.join(args.save_dir, f"patterns_ger{target:.2f}.json")
                serializable = []
                for p in patterns:
                    serializable.append({
                        'target': p['target'],
                        'opponents': p['opponents'],
                        'terminal': p['terminal'],
                        'direction': list(p['direction']),
                        'length': p['length'],
                    })
                with open(pat_path, 'w') as f:
                    json.dump(serializable, f)
                print(f"  Saved patterns to {pat_path}")

        if args.save_dir:
            meta_path = os.path.join(args.save_dir, "results.json")
            results['standard'] = {
                'gzip_size': std_gz,
                'n_patterns': len(base_patterns),
                'ger': 0.0,
            }
            with open(meta_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved results to {meta_path}")

    else:
        # Beta sweep
        betas = [round(b * 0.1, 1) for b in range(11)]
        print(f"Beta sweep ({args.n_seeds} seeds per beta):")
        print(f"{'Beta':>6} {'GER (mean±std)':>20} {'Gzip (mean±std)':>20} {'Dead (mean)':>12}")
        print("-" * 65)

        results = {'standard': {'gzip_size': std_gz, 'ger': 0.0}}

        for beta in betas:
            gers, gzs, deads = [], [], []
            for s in range(args.n_seeds):
                rng = np.random.RandomState(args.seed + s * 1000 + int(beta * 100))
                wired = cross_wire_patterns(base_patterns, beta, rng)
                gers.append(compute_ger(wired))
                gzs.append(compressed_size(patterns_to_bytes(wired)))
                deads.append(count_dead_patterns(wired))

            ger_m, ger_s = np.mean(gers), np.std(gers)
            gz_m, gz_s = np.mean(gzs), np.std(gzs)
            dead_m = np.mean(deads)
            print(f"{beta:>6.1f} {ger_m:>10.4f} ± {ger_s:<7.4f} {gz_m:>10.0f} ± {gz_s:<7.0f} {dead_m:>12.1f}")

            results[f'beta_{beta:.1f}'] = {
                'beta': beta,
                'ger_mean': float(ger_m), 'ger_std': float(ger_s),
                'gz_mean': float(gz_m), 'gz_std': float(gz_s),
                'dead_mean': float(dead_m),
                'ger_all': gers, 'gz_all': gzs,
            }

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            with open(os.path.join(args.save_dir, "beta_sweep.json"), 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
