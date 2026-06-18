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
import zipfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hand_crafted_flanking import enumerate_flanking_patterns
from compare_node_heuristics import (
    method_a, method_b, jaccard, cell_name,
    build_pattern_to_cell, aggregate_output_to_cells,
    N_MOVES, IDX_TO_VALID,
)


def _stream_array_sample(npz_path, member, sample_idx, n_features):
    """Stream-read npz member, keeping only rows at sample_idx (sorted asc).
    Avoids loading the whole array into memory.
    """
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    with zipfile.ZipFile(npz_path) as z:
        with z.open(member) as fp:
            version = np.lib.format.read_magic(fp)
            shape, _, dtype = np.lib.format._read_array_header(fp, version)
            assert len(shape) in (1, 2)
            if len(shape) == 2:
                assert shape[1] == n_features, \
                    f"Expected {n_features} features, got {shape[1]}"
            row_bytes = n_features * dtype.itemsize if len(shape) == 2 else dtype.itemsize
            out = np.zeros((len(sample_idx), n_features) if len(shape) == 2
                           else (len(sample_idx),), dtype=dtype)
            current_row = 0
            for out_idx, sidx in enumerate(sample_idx):
                skip = int(sidx) - current_row
                if skip > 0:
                    # Read+discard skip rows (zip streams can't seek)
                    remaining = skip * row_bytes
                    while remaining > 0:
                        chunk = fp.read(min(remaining, 64 * 1024 * 1024))
                        if not chunk:
                            raise EOFError(f"Unexpected EOF skipping rows in {member}")
                        remaining -= len(chunk)
                buf = fp.read(row_bytes)
                if len(buf) != row_bytes:
                    raise EOFError(f"Unexpected EOF reading row {sidx} in {member}")
                if len(shape) == 2:
                    out[out_idx] = np.frombuffer(buf, dtype=dtype)
                else:
                    out[out_idx] = np.frombuffer(buf, dtype=dtype)[0]
                current_row = int(sidx) + 1
    return out


def _read_npz_shape(npz_path, member):
    with zipfile.ZipFile(npz_path) as z:
        with z.open(member) as fp:
            version = np.lib.format.read_magic(fp)
            shape, _, _ = np.lib.format._read_array_header(fp, version)
    return shape


def load_eval(chunk_dir, chunk_prefix, n_sample=49 * 10000):
    """Stream-load a random sample from the last chunk (eval). Returns
    X (when+even sliced) and positions, both as np arrays of size n_sample.
    Never loads the full chunk into memory.
    """
    files = sorted(f for f in os.listdir(chunk_dir)
                   if f.startswith(chunk_prefix) and f.endswith('.npz')
                   and '_patterns' not in f and '_when60' not in f)
    if not files:
        raise FileNotFoundError(f"No {chunk_prefix}*.npz in {chunk_dir}")
    eval_path = os.path.join(chunk_dir, files[-1])
    print(f"Streaming eval chunk: {eval_path}")

    # Read shape from header without loading data
    feat_shape = _read_npz_shape(eval_path, 'features.npy')
    N_total, F = feat_shape
    n_sample = min(n_sample, N_total)
    print(f"  full chunk shape: {feat_shape}, sampling {n_sample}")

    # Choose sample indices, sorted so we can stream-skip
    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(N_total, n_sample, replace=False))

    # Stream-read just those rows
    print(f"  reading features...")
    X_full = _stream_array_sample(eval_path, 'features.npy', idx, F).astype(np.float32)
    print(f"  reading positions...")
    pos = _stream_array_sample(eval_path, 'positions.npy', idx, 1).astype(np.int64)

    # Slice features to when+even
    X = X_full[:, 60:180]
    print(f"  sampled X shape: {X.shape}")
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
