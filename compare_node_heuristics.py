"""Compare A (>=50% of max) vs B (top-K explaining 80%) heuristic attribution
for each of the 512 hidden nodes in a pattern-detector MLP, on BOTH the input
and output sides.

For each hidden node j:
  INPUT  side: W1[j, :] is the (120,) input weights into node j.
               Aggregate to 60 cells via max(|w|) over the 2 features per cell.
  OUTPUT side: W2[:, j] is the (960,) output weights from node j to each pattern.
               Aggregate to 60 cells via max(|w|) over all patterns whose target
               cell is c.

Then on each 60-d cell vector v, apply:
  A: cells with v_c >= 0.5 * v.max()
  B: smallest K of sorted-descending v whose cumsum >= 0.8 * v.sum()

CSV: one row per (parity, hidden_node). Columns include A/B cell sets, counts,
and Jaccard overlap on both input and output sides.

Usage:
    python compare_node_heuristics.py [--ckpt PATH] [--out CSV]
"""
import argparse
import csv
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hand_crafted_flanking import enumerate_flanking_patterns

N_MOVES = 60
VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})
IDX_TO_VALID = {i: VALID_MOVES[i] for i in range(N_MOVES)}


def cell_name(pos_64):
    row, col = pos_64 // 8, pos_64 % 8
    return f"{chr(65+row)}{col+1}"


def method_a(v, threshold=0.5):
    """Cells with v_c >= threshold * v.max(). Returns indices sorted by value desc."""
    if v.max() == 0:
        return []
    cutoff = threshold * v.max()
    idx = np.where(v >= cutoff)[0]
    return sorted(idx.tolist(), key=lambda i: -v[i])


def method_b(v, frac=0.8):
    """Top-K sorted indices whose cumsum reaches frac * v.sum()."""
    total = v.sum()
    if total == 0:
        return []
    order = np.argsort(v)[::-1]
    target = frac * total
    cum = 0.0
    out = []
    for i in order:
        out.append(int(i))
        cum += v[i]
        if cum >= target:
            break
    return out


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def build_pattern_to_cell(patterns):
    """For each of the 960 patterns, return the index 0..59 of its target cell."""
    p2c = np.zeros(len(patterns), dtype=np.int64)
    valid_to_idx = {v: i for i, v in enumerate(VALID_MOVES)}
    for p_idx, pat in enumerate(patterns):
        p2c[p_idx] = valid_to_idx[pat['target']]
    return p2c


def aggregate_input_to_cells(w120):
    """Aggregate (120,) input weight to (60,) cell vector via max(|w|)."""
    return np.abs(w120).reshape(2, N_MOVES).max(axis=0)


def aggregate_output_to_cells(w960, pat_to_cell):
    """Aggregate (960,) output weight to (60,) cell vector via max(|w|) over
    all patterns whose target cell equals c."""
    abs_w = np.abs(w960)
    out = np.zeros(N_MOVES, dtype=abs_w.dtype)
    for p_idx, c_idx in enumerate(pat_to_cell):
        if abs_w[p_idx] > out[c_idx]:
            out[c_idx] = abs_w[p_idx]
    return out


def analyze_mlp(state_dict, parity_name, pat_to_cell, writer):
    W1 = state_dict['net.0.weight'].detach().cpu().numpy()  # (H, 120)
    W2 = state_dict['net.2.weight'].detach().cpu().numpy()  # (960, H)
    H = W1.shape[0]
    print(f"  {parity_name}: W1={W1.shape}, W2={W2.shape}, H={H}")

    for j in range(H):
        v_in = aggregate_input_to_cells(W1[j])
        v_out = aggregate_output_to_cells(W2[:, j], pat_to_cell)

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
            # input side
            'in_A_count': len(a_in_cells),
            'in_A_cells': '|'.join(a_in_cells),
            'in_B_count': len(b_in_cells),
            'in_B_cells': '|'.join(b_in_cells),
            'in_AB_jaccard': f"{jaccard(a_in, b_in):.3f}",
            'in_AB_agree': set(a_in) == set(b_in),
            # output side
            'out_A_count': len(a_out_cells),
            'out_A_cells': '|'.join(a_out_cells),
            'out_B_count': len(b_out_cells),
            'out_B_cells': '|'.join(b_out_cells),
            'out_AB_jaccard': f"{jaccard(a_out, b_out):.3f}",
            'out_AB_agree': set(a_out) == set(b_out),
            # cross-side overlap (does the unit fire on cells it pushes?)
            'AinAout_jaccard': f"{jaccard(a_in, a_out):.3f}",
            'BinBout_jaccard': f"{jaccard(b_in, b_out):.3f}",
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='experiments/mathematical_transformation_experiments/'
                                       'heuristic_probe_results/pattern_detector_checkpoints/'
                                       'pattern_simple_direct_H512_wheneven.pt')
    ap.add_argument('--out', default='node_heuristics_A_vs_B.csv')
    args = ap.parse_args()

    print(f"Loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    print(f"  hidden_dim={ckpt['hidden_dim']}, input_dim={ckpt['input_dim']}, "
          f"n_patterns={ckpt['n_patterns']}")

    patterns = enumerate_flanking_patterns()
    assert len(patterns) == ckpt['n_patterns'], \
        f"pattern count mismatch: {len(patterns)} vs {ckpt['n_patterns']}"
    pat_to_cell = build_pattern_to_cell(patterns)

    cols = ['parity', 'hidden_idx',
            'in_A_count', 'in_A_cells', 'in_B_count', 'in_B_cells',
            'in_AB_jaccard', 'in_AB_agree',
            'out_A_count', 'out_A_cells', 'out_B_count', 'out_B_cells',
            'out_AB_jaccard', 'out_AB_agree',
            'AinAout_jaccard', 'BinBout_jaccard']

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        analyze_mlp(ckpt['even'], 'even', pat_to_cell, w)
        analyze_mlp(ckpt['odd'], 'odd', pat_to_cell, w)

    print(f"\nWrote {args.out}")

    # Summary
    import pandas as pd
    df = pd.read_csv(args.out)
    print(f"\nRows: {len(df)}  (512 hidden nodes x 2 parities)")
    for side in ('in', 'out'):
        print(f"\n--- {side.upper()}PUT side ---")
        print(f"  A count: median={df[f'{side}_A_count'].median():.1f}, "
              f"mean={df[f'{side}_A_count'].mean():.2f}")
        print(f"  B count: median={df[f'{side}_B_count'].median():.1f}, "
              f"mean={df[f'{side}_B_count'].mean():.2f}")
        print(f"  Mean Jaccard(A, B): "
              f"{df[f'{side}_AB_jaccard'].astype(float).mean():.3f}")
        agree = df[f'{side}_AB_agree'].sum()
        print(f"  Rows where A == B (set equality): {agree}/{len(df)} "
              f"({100*agree/len(df):.1f}%)")

    print(f"\nMean Jaccard(A_in, A_out): "
          f"{df['AinAout_jaccard'].astype(float).mean():.3f}")
    print(f"Mean Jaccard(B_in, B_out): "
          f"{df['BinBout_jaccard'].astype(float).mean():.3f}")


if __name__ == '__main__':
    main()
