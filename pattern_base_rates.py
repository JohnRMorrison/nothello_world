"""Step 0 for the class-imbalance hypothesis: measure per-pattern firing rates
AND per-pattern model accuracy/recall/precision, bucketed by target-cell class
(corner / edge / inner).

If corner-target patterns fire 30-100x less often AND model recall on those
patterns is correspondingly worse, class imbalance is the cause of the per-cell
recall gap. If recall is uniform across base-rate buckets, look elsewhere.

Usage:
    python pattern_base_rates.py \\
        --ckpt experiments/.../pattern_simple_direct_H1024_wheneven.pt \\
        --hidden 1024
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
from train_pattern_simple import (
    DirectMLP, EndToEndMLP, TwoStageMLP, compute_pattern_labels_batch,
    to_signed_parity_input, to_mine_signed_input, to_board_state_input,
    to_color_split_input, to_played_halfmask_input, to_played_bit_input,
    to_move_grid_input, to_move_grid_onehot_input,
)


CENTER_CELLS_64 = {27, 28, 35, 36}   # d4, e4, d5, e5 -- excluded from MOVE_TO_IDX


def idx60_to_square(i):
    j = -1
    for s in range(64):
        if s in CENTER_CELLS_64:
            continue
        j += 1
        if j == i:
            return s
    raise ValueError(i)


def classify_cell(idx60):
    sq = idx60_to_square(idx60)
    row, col = sq // 8, sq % 8
    if row in (0, 7) and col in (0, 7):
        return 'corner'
    if row in (0, 7) or col in (0, 7):
        return 'edge'
    return 'inner'


def idx60_to_alg(i):
    sq = idx60_to_square(i)
    return f"{'abcdefgh'[sq % 8]}{sq // 8 + 1}"


_FEAT_COLS = {
    "when":        list(range(N_MOVES, 2 * N_MOVES)),
    "played":      list(range(0, N_MOVES)),
    "played+when": list(range(0, 2 * N_MOVES)),
    "when+even":   list(range(N_MOVES, 3 * N_MOVES)),
    "played+even": list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)),
    "all":         list(range(0, 3 * N_MOVES)),
}


def infer_features(name, input_dim):
    if "wheneven" in name: return "when+even"
    if "playedeven" in name: return "played+even"
    if input_dim == 120: return "when+even"
    if input_dim == 180: return "all"
    if input_dim == 60: return "when"
    raise ValueError(f"can't infer features for {name} (input_dim={input_dim})")


def select_features(X, Y, pos, features):
    if features in _FEAT_COLS:
        return X[:, _FEAT_COLS[features]]
    if features == "signed_parity": return to_signed_parity_input(X)
    if features == "mine_signed": return to_mine_signed_input(Y, pos)
    if features == "board_state": return to_board_state_input(Y, pos)
    if features == "color_split": return to_color_split_input(X)
    if features == "played+halfmask": return to_played_halfmask_input(X)
    if features == "played+bit": return to_played_bit_input(X)
    if features == "move_grid": return to_move_grid_input(X)
    if features == "move_grid_onehot": return to_move_grid_onehot_input(X)
    raise ValueError(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help="If set, also compute per-pattern accuracy/precision/recall "
                             "using this checkpoint.")
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--mode", default="direct",
                        choices=["direct", "emergent", "e2e", "two-stage", "randproj"])
    parser.add_argument("--features", default=None)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Logit threshold for positive prediction (default 0 = sigmoid > 0.5).")
    args = parser.parse_args()

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = \
        precompute_pattern_arrays(patterns)

    pattern_cell = np.array([MOVE_TO_IDX[p['target']] for p in patterns])
    pattern_class = np.array([classify_cell(c) for c in pattern_cell])
    pattern_length = np.array([len(p['opponents']) for p in patterns])
    print(f"Total patterns: {len(patterns)}")
    print(f"  corner-target patterns: {(pattern_class == 'corner').sum()}")
    print(f"  edge-target patterns:   {(pattern_class == 'edge').sum()}")
    print(f"  inner-target patterns:  {(pattern_class == 'inner').sum()}")

    output_dir = "experiments/mathematical_transformation_experiments/heuristic_probe_results"
    chunk_dir = os.path.join(output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz")
                         and "_patterns" not in f
                         and "_when60" not in f)
    eval_path = chunk_files[-1]
    print(f"\nLoading {eval_path}")

    X, Y, pos = _load_features(eval_path)
    N = len(Y)
    print(f"Positions in chunk: {N}")

    use_model = args.ckpt is not None
    device = get_device() if use_model else None
    me = mo = None
    feat_X = None
    if use_model:
        ckpt = torch.load(args.ckpt, map_location=device)
        n_patterns = ckpt.get('n_patterns', 960)
        input_dim = ckpt.get('input_dim', N_MOVES)
        if args.features is None:
            args.features = infer_features(os.path.basename(args.ckpt), input_dim)
        print(f"Features: {args.features} (input_dim={input_dim})")
        Cls = {"direct": DirectMLP, "randproj": DirectMLP,
               "two-stage": TwoStageMLP,
               "emergent": EndToEndMLP, "e2e": EndToEndMLP}[args.mode]
        if args.hidden is None:
            raise ValueError("--hidden required when --ckpt set")
        me = Cls(input_dim, args.hidden, n_patterns).to(device)
        mo = Cls(input_dim, args.hidden, n_patterns).to(device)
        me.load_state_dict(ckpt['even']); me.eval()
        mo.load_state_dict(ckpt['odd']); mo.eval()
        print(f"Loaded {args.ckpt} (pat_acc={ckpt.get('best_pat_acc', '?')})")
        feat_X = select_features(X, Y, pos, args.features)
    else:
        del X

    n = min(N, 49 * 10000)
    rng = np.random.RandomState(0)
    si = np.sort(rng.choice(N, n, replace=False))
    Y, pos = Y[si], pos[si]
    if use_model:
        feat_X = feat_X[si]
    print(f"Sampled to: {n}")

    counts = np.zeros(len(patterns), dtype=np.int64)
    tp = np.zeros(len(patterns), dtype=np.int64)
    fp = np.zeros(len(patterns), dtype=np.int64)
    fn = np.zeros(len(patterns), dtype=np.int64)

    batch = 8192 if use_model else 50_000
    with torch.no_grad():
        for i in range(0, n, batch):
            yb = Y[i:i + batch].numpy()
            pb = pos[i:i + batch].numpy()
            labels = compute_pattern_labels_batch(
                yb, pb, pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
            counts += labels.sum(axis=0)

            if use_model:
                xb = feat_X[i:i + batch].to(device)
                p = pos[i:i + batch]
                em = (p % 2 == 0); om = ~em
                preds = torch.zeros(len(xb), len(patterns), device=device)
                if args.mode in ("direct", "randproj"):
                    if em.any(): preds[em] = me(xb[em])
                    if om.any(): preds[om] = mo(xb[om])
                else:
                    for msk, m in [(em, me), (om, mo)]:
                        if not msk.any(): continue
                        logits, _ = m(xb[msk], p[msk])
                        preds[msk] = logits
                pos_pred = (preds > args.threshold).cpu().numpy().astype(np.int64)
                lbl = labels.astype(np.int64)
                tp += (pos_pred & lbl).sum(axis=0)
                fp += (pos_pred & (1 - lbl)).sum(axis=0)
                fn += ((1 - pos_pred) & lbl).sum(axis=0)

    print()
    print("=" * 78)
    print("Per-class aggregate firing rates")
    print("=" * 78)
    print(f"{'class':>8s} {'n_pat':>6s} {'total_fires':>12s} "
          f"{'avg/pat':>10s} {'base_rate':>12s}")
    rates = {}
    for cls in ('corner', 'edge', 'inner'):
        m = pattern_class == cls
        npat = int(m.sum())
        total = int(counts[m].sum())
        per_pat = total / max(npat, 1)
        rate = per_pat / n
        rates[cls] = per_pat
        print(f"{cls:>8s} {npat:>6d} {total:>12d} "
              f"{per_pat:>10.1f} {rate:>12.5f}")

    print()
    print("Imbalance ratios (mean fires per pattern):")
    print(f"  inner / edge:   {rates['inner'] / max(rates['edge'],   1e-9):>7.2f}x")
    print(f"  inner / corner: {rates['inner'] / max(rates['corner'], 1e-9):>7.2f}x")
    print(f"  edge  / corner: {rates['edge']  / max(rates['corner'], 1e-9):>7.2f}x")

    print()
    print("=" * 78)
    print("Per-pattern firing-count distribution within each class")
    print("=" * 78)
    print(f"{'class':>8s} {'min':>8s} {'p25':>8s} {'median':>8s} "
          f"{'p75':>8s} {'max':>8s}")
    for cls in ('corner', 'edge', 'inner'):
        sub = counts[pattern_class == cls]
        q = np.percentile(sub, [0, 25, 50, 75, 100])
        print(f"{cls:>8s} {int(q[0]):>8d} {int(q[1]):>8d} "
              f"{int(q[2]):>8d} {int(q[3]):>8d} {int(q[4]):>8d}")

    if use_model:
        with np.errstate(invalid='ignore', divide='ignore'):
            recall    = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
            precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
            f1 = np.where(precision + recall > 0,
                          2 * precision * recall / (precision + recall),
                          np.nan)

        print()
        print("=" * 78)
        print(f"Per-class model metrics (threshold logit > {args.threshold})")
        print("=" * 78)
        print(f"{'class':>8s} {'precision':>10s} {'recall':>10s} {'f1':>10s}")
        for cls in ('corner', 'edge', 'inner'):
            m = pattern_class == cls
            print(f"{cls:>8s} {np.nanmean(precision[m]):>10.4f} "
                  f"{np.nanmean(recall[m]):>10.4f} "
                  f"{np.nanmean(f1[m]):>10.4f}")

        print()
        print("=" * 78)
        print("Recall by pattern base-rate quintile (does rare = bad?)")
        print("=" * 78)
        order = np.argsort(counts)
        bucket_size = len(patterns) // 5
        print(f"{'quintile':>10s} {'fire min':>10s} {'fire max':>10s} "
              f"{'mean recall':>12s} {'mean prec':>10s} {'n_pat':>6s}")
        for q in range(5):
            lo, hi = q * bucket_size, (q + 1) * bucket_size if q < 4 else len(patterns)
            idxs = order[lo:hi]
            print(f"{q + 1:>10d} {int(counts[idxs].min()):>10d} "
                  f"{int(counts[idxs].max()):>10d} "
                  f"{np.nanmean(recall[idxs]):>12.4f} "
                  f"{np.nanmean(precision[idxs]):>10.4f} "
                  f"{len(idxs):>6d}")

        print()
        print("=" * 78)
        print("Recall by pattern length")
        print("=" * 78)
        print(f"{'length':>8s} {'n_pat':>6s} {'mean recall':>12s} {'mean prec':>10s}")
        for L in sorted(set(pattern_length.tolist())):
            m = pattern_length == L
            print(f"{L:>8d} {int(m.sum()):>6d} "
                  f"{np.nanmean(recall[m]):>12.4f} "
                  f"{np.nanmean(precision[m]):>10.4f}")

        print()
        print("Worst 15 patterns by recall (excluding patterns that never fire):")
        valid = (tp + fn) > 50    # ignore very-rare patterns where recall is noise
        order_r = np.argsort(np.where(valid, recall, 1.0))
        for k in range(15):
            i = order_r[k]
            print(f"  {idx60_to_alg(int(pattern_cell[i])):>3s}  "
                  f"length={pattern_length[i]}  "
                  f"fires={counts[i]:>8d}  "
                  f"recall={recall[i]:.4f}  prec={precision[i]:.4f}  "
                  f"class={pattern_class[i]:>6s}")

        print("\nBest 15 patterns by recall:")
        order_r = np.argsort(-np.where(valid, recall, -1.0))
        for k in range(15):
            i = order_r[k]
            print(f"  {idx60_to_alg(int(pattern_cell[i])):>3s}  "
                  f"length={pattern_length[i]}  "
                  f"fires={counts[i]:>8d}  "
                  f"recall={recall[i]:.4f}  prec={precision[i]:.4f}  "
                  f"class={pattern_class[i]:>6s}")

    print()
    print("Bottom 10 patterns by total fires (rarest):")
    order = np.argsort(counts)
    for k in range(10):
        i = order[k]
        print(f"  {idx60_to_alg(int(pattern_cell[i])):>3s}  "
              f"length={pattern_length[i]}  fires={counts[i]:>8d}  "
              f"class={pattern_class[i]:>6s}")
