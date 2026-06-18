"""Activation-based per-node heuristic attribution.

For each hidden node j in a parity-split pattern-detector MLP, decompose its
firing activations on real positions into per-input contributions:
    contrib(j, i, n) = |W1[j, i] * X[n, i]|, masked to positions where the
    node fires (ReLU is open).
Average across firing positions to get per-feature contribution magnitude.
Aggregate to cells. Apply methods A (>= 50% of max) and B (top-K explaining
80% of total).

Cells "explain" activations using *data-weighted* contributions, so features
that are usually zero contribute nothing — typically yields much sparser
heuristics than the weights-only analysis.

Usage:
    python compare_node_heuristics_activation.py \
        --ckpt experiments/.../pattern_simple_direct_H512_wheneven.pt \
        --chunk-dir experiments/.../feature_chunks \
        --chunk-prefix chunk_ext_ \
        --out node_heuristics_activation.csv
"""
import argparse
import csv
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hand_crafted_flanking import enumerate_flanking_patterns
from compare_node_heuristics import (
    method_a, method_b, jaccard, cell_name,
    build_pattern_to_cell, aggregate_output_to_cells,
    N_MOVES, IDX_TO_VALID,
)


def load_eval(chunk_dir, chunk_prefix):
    """Load the last chunk (eval) and return X (when+even sliced), positions."""
    files = sorted(f for f in os.listdir(chunk_dir)
                   if f.startswith(chunk_prefix) and f.endswith('.npz')
                   and '_patterns' not in f and '_when60' not in f)
    if not files:
        raise FileNotFoundError(f"No {chunk_prefix}*.npz in {chunk_dir}")
    eval_path = os.path.join(chunk_dir, files[-1])
    print(f"Loading eval chunk: {eval_path}")
    data = np.load(eval_path)
    X_full = data['features'].astype(np.float32)  # (N, 180)
    pos = data['positions'].astype(np.int64)      # (N,)

    # Slice to when+even features (indices 60..179)
    X = X_full[:, 60:180]
    print(f"  raw eval shape: {X.shape}, positions in [{pos.min()}, {pos.max()}]")

    # Match training eval sampling: at most 49*10000 random positions
    n_eval = min(len(X), 49 * 10000)
    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(len(X), n_eval, replace=False))
    X, pos = X[idx], pos[idx]
    print(f"  sampled eval shape: {X.shape}")
    return X, pos


def compute_contributions(W1, b1, X_parity):
    """For each hidden node j, compute mean |W1[j, i] * X[n, i]| over firing
    positions. Returns (H, 120) array.

    W1: (H, 120), b1: (H,), X_parity: (N, 120).
    """
    H, I = W1.shape
    # Pre-activation: (N, H)
    pre = X_parity @ W1.T + b1[None, :]
    firing = pre > 0  # (N, H)
    n_active = firing.sum(axis=0)  # (H,)

    contrib = np.zeros((H, I), dtype=np.float32)
    for j in range(H):
        mask = firing[:, j]
        if mask.sum() == 0:
            continue
        # |W1[j, i] * X[n, i]| averaged over n where firing
        # = |W1[j, i]| * mean over firing(n) of |X[n, i]|  (because W1 is fixed)
        X_active = X_parity[mask]  # (n_a, I)
        contrib[j] = np.abs(W1[j]) * np.abs(X_active).mean(axis=0)
    return contrib, n_active


def aggregate_to_cells(contrib_120):
    """(120,) -> (60,) via max over the 2 features per cell (when, even)."""
    return np.abs(contrib_120).reshape(2, N_MOVES).max(axis=0)


def analyze_parity(state_dict, parity_name, X_all, pos_all, pat_to_cell, writer):
    """Filter positions for this parity, run attribution, write rows."""
    W1 = state_dict['net.0.weight'].detach().cpu().numpy()
    b1 = state_dict['net.0.bias'].detach().cpu().numpy()
    W2 = state_dict['net.2.weight'].detach().cpu().numpy()

    parity_mask = (pos_all % 2 == 0) if parity_name == 'even' else (pos_all % 2 == 1)
    X = X_all[parity_mask]
    print(f"  {parity_name}: {len(X)} positions")

    contrib, n_active = compute_contributions(W1, b1, X)
    H = W1.shape[0]

    for j in range(H):
        v_in = aggregate_to_cells(contrib[j])
        v_out = aggregate_output_to_cells(W2[:, j], pat_to_cell)  # weights-only

        a_in = method_a(v_in)
        b_in = method_b(v_in)
        a_out = method_a(v_out)
        b_out = method_b(v_out)

        a_in_cells = [cell_name(IDX_TO_VALID[i]) for i in a_in]
        b_in_cells = [cell_name(IDX_TO_VALID[i]) for i in b_in]
        a_out_cells = [cell_name(IDX_TO_VALID[i]) for i in a_out]
        b_out_cells = [cell_name(IDX_TO_VALID[i]) for i in b_out]

        writer.writerow({
            'parity': parity_name,
            'hidden_idx': j,
            'fire_rate': f"{n_active[j] / max(len(X), 1):.4f}",
            # input side -- ACTIVATION based
            'in_A_count': len(a_in_cells),
            'in_A_cells': '|'.join(a_in_cells),
            'in_B_count': len(b_in_cells),
            'in_B_cells': '|'.join(b_in_cells),
            'in_AB_jaccard': f"{jaccard(a_in, b_in):.3f}",
            'in_AB_agree': set(a_in) == set(b_in),
            # output side -- still weights based (no decomposition needed)
            'out_A_count': len(a_out_cells),
            'out_A_cells': '|'.join(a_out_cells),
            'out_B_count': len(b_out_cells),
            'out_B_cells': '|'.join(b_out_cells),
            'out_AB_jaccard': f"{jaccard(a_out, b_out):.3f}",
            'out_AB_agree': set(a_out) == set(b_out),
            'AinAout_jaccard': f"{jaccard(a_in, a_out):.3f}",
            'BinBout_jaccard': f"{jaccard(b_in, b_out):.3f}",
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='experiments/mathematical_transformation_experiments/'
                                       'heuristic_probe_results/pattern_detector_checkpoints/'
                                       'pattern_simple_direct_H512_wheneven.pt')
    ap.add_argument('--chunk-dir',
                    default='experiments/mathematical_transformation_experiments/'
                            'heuristic_probe_results/feature_chunks')
    ap.add_argument('--chunk-prefix', default='chunk_ext_')
    ap.add_argument('--out', default='node_heuristics_activation.csv')
    args = ap.parse_args()

    print(f"Loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    patterns = enumerate_flanking_patterns()
    pat_to_cell = build_pattern_to_cell(patterns)

    X_all, pos_all = load_eval(args.chunk_dir, args.chunk_prefix)

    cols = ['parity', 'hidden_idx', 'fire_rate',
            'in_A_count', 'in_A_cells', 'in_B_count', 'in_B_cells',
            'in_AB_jaccard', 'in_AB_agree',
            'out_A_count', 'out_A_cells', 'out_B_count', 'out_B_cells',
            'out_AB_jaccard', 'out_AB_agree',
            'AinAout_jaccard', 'BinBout_jaccard']

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        analyze_parity(ckpt['even'], 'even', X_all, pos_all, pat_to_cell, w)
        analyze_parity(ckpt['odd'], 'odd', X_all, pos_all, pat_to_cell, w)

    print(f"\nWrote {args.out}")

    import pandas as pd
    df = pd.read_csv(args.out)
    print(f"\nRows: {len(df)}")
    print(f"\n--- INPUT side (activation-based) ---")
    print(f"  A count: median={df['in_A_count'].median():.1f}, mean={df['in_A_count'].mean():.2f}")
    print(f"  B count: median={df['in_B_count'].median():.1f}, mean={df['in_B_count'].mean():.2f}")
    print(f"  Mean Jaccard(A, B): {df['in_AB_jaccard'].astype(float).mean():.3f}")
    print(f"  A==B (set eq): {df['in_AB_agree'].sum()}/{len(df)} "
          f"({100*df['in_AB_agree'].mean():.1f}%)")

    print(f"\n--- OUTPUT side (weights-based, unchanged) ---")
    print(f"  A count: median={df['out_A_count'].median():.1f}, mean={df['out_A_count'].mean():.2f}")
    print(f"  B count: median={df['out_B_count'].median():.1f}, mean={df['out_B_count'].mean():.2f}")

    print(f"\nFire rate quartiles: {df['fire_rate'].quantile([0.25, 0.5, 0.75]).tolist()}")


if __name__ == '__main__':
    main()
