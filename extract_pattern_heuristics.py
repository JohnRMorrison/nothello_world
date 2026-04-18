"""Extract per-unit heuristics from pattern detector MLPs.

For each hidden unit, reports:
- Top-1, Top-3, Top-5 input features (+ cumulative variance explained)
- 70%, 80%, 90% variance-explained feature sets
- Top output patterns (by |weight|)

Usage:
    python extract_pattern_heuristics.py \
        --ckpt pattern_simple_direct_H512.pt --hidden 512
"""
import sys, os, json, argparse
sys.path.insert(0, '.')

import numpy as np
import torch


# 60 valid moves (excluding 4 center squares)
_VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})
_CELL_NAMES = "ABCDEFGH"


def cell_name(cell_idx):
    """0-63 → 'A1' etc."""
    return f"{_CELL_NAMES[cell_idx // 8]}{cell_idx % 8 + 1}"


def feature_name(feat_idx):
    """feature index 0-59 → cell name (since 60-d is only 'when')."""
    return cell_name(_VALID_MOVES[feat_idx])


def analyze_unit(W_in, W_out, unit_idx, patterns, pattern_cells=None):
    """Analyze a single hidden unit.

    W_in:  (H, 60)   first-layer weights
    W_out: (960, H)  output weights to patterns
    unit_idx: which unit
    """
    w_in = W_in[unit_idx]  # (60,)
    w_out = W_out[:, unit_idx]  # (960,)

    abs_in = w_in.abs().numpy()
    sort_idx = np.argsort(-abs_in)  # descending by |w|
    sorted_abs = abs_in[sort_idx]
    sorted_w = w_in.numpy()[sort_idx]

    # Variance explained at each rank
    sq = sorted_abs ** 2
    cum_var = sq.cumsum() / sq.sum() if sq.sum() > 0 else np.zeros_like(sq)

    def top_k(k):
        features = []
        for r in range(k):
            fi = sort_idx[r]
            features.append({
                'cell': feature_name(fi),
                'weight': float(sorted_w[r]),
            })
        return {'features': features, 'var_explained': float(cum_var[k - 1]) if k > 0 else 0.0}

    def features_for_variance(threshold):
        # How many features needed to reach threshold
        if sq.sum() == 0:
            return {'features': [], 'n': 0}
        n = int((cum_var >= threshold).argmax() + 1)
        return top_k(n) | {'n': n}

    # Output side: top patterns by |weight|
    abs_out = w_out.abs().numpy()
    top_pat_idx = np.argsort(-abs_out)[:5]
    top_patterns = []
    for pi in top_pat_idx:
        pat = patterns[pi]
        top_patterns.append({
            'pattern_idx': int(pi),
            'target': cell_name(pat['target']),
            'length': pat['length'],
            'weight': float(w_out[pi].item()),
        })

    # Output L2 norm (importance score)
    l2 = float(w_out.norm().item())

    return {
        'unit_idx': unit_idx,
        'output_l2': l2,
        'top_1': top_k(1),
        'top_3': top_k(3),
        'top_5': top_k(5),
        'variance_70': features_for_variance(0.70),
        'variance_80': features_for_variance(0.80),
        'variance_90': features_for_variance(0.90),
        'top_patterns': top_patterns,
    }


def format_unit_report(unit_data, parity):
    """Format one unit as a readable text block."""
    u = unit_data
    lines = [f"[{parity}] Unit {u['unit_idx']}  (output L2: {u['output_l2']:.3f})"]

    def fmt_features(d, label):
        feats = d['features']
        parts = []
        for f in feats:
            sign = '+' if f['weight'] > 0 else '-'
            parts.append(f"{sign}{f['cell']}({f['weight']:+.2f})")
        var = d.get('var_explained', None)
        var_str = f" [var={var:.1%}]" if var is not None else ""
        n_str = f" ({d.get('n', len(feats))} features)" if 'n' in d and d.get('n') != len(feats) else ""
        return f"  {label}{var_str}{n_str}: {', '.join(parts) if parts else '(none)'}"

    lines.append(fmt_features(u['top_1'], "top-1 "))
    lines.append(fmt_features(u['top_3'], "top-3 "))
    lines.append(fmt_features(u['top_5'], "top-5 "))
    lines.append(fmt_features(u['variance_70'], "var70%"))
    lines.append(fmt_features(u['variance_80'], "var80%"))
    lines.append(fmt_features(u['variance_90'], "var90%"))

    # Top patterns
    pat_strs = []
    for p in u['top_patterns']:
        pat_strs.append(f"p{p['pattern_idx']}→{p['target']}(L{p['length']}, w={p['weight']:+.2f})")
    lines.append(f"  top patterns: {', '.join(pat_strs)}")

    return '\n'.join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--output-dir", default="experiments/pattern_heuristics")
    parser.add_argument("--top-n-units", type=int, default=50,
                        help="How many top units (by output L2) to print in text report")
    args = parser.parse_args()

    from hand_crafted_flanking import enumerate_flanking_patterns
    patterns = enumerate_flanking_patterns()

    ckpt = torch.load(args.ckpt, map_location='cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.ckpt))[0]

    all_results = {}

    for parity in ['even', 'odd']:
        state = ckpt[parity]
        W_in = state['net.0.weight']   # (H, 60)
        W_out = state['net.2.weight']  # (960, H)

        print(f"\n=== {parity} (H={args.hidden}) ===")
        print(f"W_in shape: {tuple(W_in.shape)}, W_out shape: {tuple(W_out.shape)}")

        # Analyze all units
        units = [analyze_unit(W_in, W_out, j, patterns) for j in range(args.hidden)]
        all_results[parity] = units

        # Sort by output L2 norm
        units_sorted = sorted(units, key=lambda u: -u['output_l2'])

        # Summary statistics
        n_for_80 = np.array([u['variance_80']['n'] for u in units])
        n_for_90 = np.array([u['variance_90']['n'] for u in units])
        top1_var = np.array([u['top_1']['var_explained'] for u in units])

        print(f"  Top-1 variance explained: mean={top1_var.mean():.3f}, "
              f"median={np.median(top1_var):.3f}")
        print(f"  # features for 80% var: mean={n_for_80.mean():.1f}, "
              f"median={int(np.median(n_for_80))}, max={n_for_80.max()}")
        print(f"  # features for 90% var: mean={n_for_90.mean():.1f}, "
              f"median={int(np.median(n_for_90))}, max={n_for_90.max()}")

    # Write text report (top units by L2)
    txt_path = os.path.join(args.output_dir, f"{base}_heuristics.txt")
    with open(txt_path, 'w') as f:
        for parity in ['even', 'odd']:
            units_sorted = sorted(all_results[parity], key=lambda u: -u['output_l2'])
            f.write(f"\n{'='*70}\n{parity.upper()} (top {args.top_n_units} by output L2)\n{'='*70}\n\n")
            for u in units_sorted[:args.top_n_units]:
                f.write(format_unit_report(u, parity))
                f.write('\n\n')
    print(f"\nSaved readable report: {txt_path}")

    # Write full JSON
    json_path = os.path.join(args.output_dir, f"{base}_heuristics.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved full JSON: {json_path}")
