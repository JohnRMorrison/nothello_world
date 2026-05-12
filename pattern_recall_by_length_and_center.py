"""Per-pattern recall cross-tabulated by (length, routes_through_center).

Tests whether pattern length alone explains the per-pattern recall variance,
or whether routing through center cells adds independent difficulty when
length is controlled.

Usage:
    python pattern_recall_by_length_and_center.py \\
        --ckpt experiments/.../pattern_simple_direct_H{H}_wheneven.pt \\
        --hidden {H}
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES,
)
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import DirectMLP, compute_pattern_labels_batch


CENTER_64 = {27, 28, 35, 36}


_FEAT_COLS = {
    "when":      list(range(N_MOVES, 2 * N_MOVES)),
    "when+even": list(range(N_MOVES, 3 * N_MOVES)),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--features", default="when+even")
    parser.add_argument("--n-positions", type=int, default=490000)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = \
        precompute_pattern_arrays(patterns)
    pat_length = np.array([len(p['opponents']) for p in patterns])
    pat_routes_center = np.array([
        any(c in CENTER_64 for c in p['opponents']) or p['terminal'] in CENTER_64
        for p in patterns
    ], dtype=bool)
    print(f"Total: {len(patterns)} patterns; "
          f"{int(pat_routes_center.sum())} route through center")

    ckpt = torch.load(args.ckpt, map_location=device)
    input_dim = ckpt.get('input_dim', 120)
    n_patterns = ckpt.get('n_patterns', 960)
    me = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    mo = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()
    print(f"Loaded {args.ckpt}")

    out_dir = "experiments/mathematical_transformation_experiments/heuristic_probe_results"
    chunk_dir = os.path.join(out_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f
                         and "_when60" not in f)
    X, Y, pos = _load_features(chunk_files[-1])
    feat_X = X[:, _FEAT_COLS[args.features]]
    del X
    n = min(args.n_positions, len(Y))
    rng = np.random.RandomState(0)
    si = np.sort(rng.choice(len(Y), n, replace=False))
    feat_X = feat_X[si]; Y_np = Y[si].numpy(); pos_np = pos[si].numpy()
    print(f"Sampled {n} positions")

    tp = np.zeros(len(patterns), dtype=np.int64)
    fn = np.zeros(len(patterns), dtype=np.int64)
    fp = np.zeros(len(patterns), dtype=np.int64)
    batch = 8192
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = feat_X[i:i + batch].to(device)
            yb = Y_np[i:i + batch]
            pb = pos_np[i:i + batch]
            em = (pb % 2 == 0); om = ~em
            preds = torch.zeros(len(xb), len(patterns), device=device)
            if em.any(): preds[em] = me(xb[em])
            if om.any(): preds[om] = mo(xb[om])
            pred_pos = (preds > args.threshold).cpu().numpy().astype(np.int64)
            labels = compute_pattern_labels_batch(
                yb, pb, pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask
            ).astype(np.int64)
            tp += (pred_pos & labels).sum(axis=0)
            fn += ((1 - pred_pos) & labels).sum(axis=0)
            fp += (pred_pos & (1 - labels)).sum(axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)

    print()
    print("=" * 78)
    print("Per-pattern recall by (length, routes_through_center)")
    print("=" * 78)
    print(f"{'length':>8s} {'route through ctr':>20s} {'no ctr routing':>18s} "
          f"{'all':>10s}")
    for L in sorted(set(pat_length.tolist())):
        mask_L = (pat_length == L)
        if mask_L.sum() == 0: continue
        m_ctr = mask_L & pat_routes_center
        m_no  = mask_L & ~pat_routes_center
        line = f"{L:>8d}"
        if m_ctr.sum() > 0:
            line += f" {np.nanmean(recall[m_ctr]):>14.4f} "
            line += f"(n={int(m_ctr.sum()):>3d})"
        else:
            line += f" {'-':>20s}"
        if m_no.sum() > 0:
            line += f" {np.nanmean(recall[m_no]):>13.4f} "
            line += f"(n={int(m_no.sum()):>3d})"
        else:
            line += f" {'-':>18s}"
        line += f" {np.nanmean(recall[mask_L]):>10.4f}"
        print(line)

    print()
    print("Marginal (no length control):")
    print(f"  routes through center:    n={int(pat_routes_center.sum())}  "
          f"mean recall={np.nanmean(recall[pat_routes_center]):.4f}")
    print(f"  no center routing:        n={int((~pat_routes_center).sum())}  "
          f"mean recall={np.nanmean(recall[~pat_routes_center]):.4f}")

    print()
    print("Within-length difference (center-routing recall - no-center recall):")
    for L in sorted(set(pat_length.tolist())):
        mask_L = (pat_length == L)
        m_ctr = mask_L & pat_routes_center
        m_no  = mask_L & ~pat_routes_center
        if m_ctr.sum() == 0 or m_no.sum() == 0: continue
        diff = np.nanmean(recall[m_ctr]) - np.nanmean(recall[m_no])
        print(f"  L={L}: ctr={np.nanmean(recall[m_ctr]):.4f}  "
              f"no_ctr={np.nanmean(recall[m_no]):.4f}  diff={diff:+.4f}")
