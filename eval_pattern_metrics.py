"""Pattern accuracy / recall / precision on a random sample across positions.

Fixes the cluster-eval bias where the training script's head slice landed
on positions 5-6 only, inflating metrics. Also reports per-position-bucket
accuracy so you can see where the model struggles.

Usage:
    python eval_pattern_metrics.py --ckpt pattern_simple_direct_H512.pt \
        --mode direct --hidden 512
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES,
)
from hand_crafted_flanking import enumerate_flanking_patterns
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import (
    DirectMLP, EndToEndMLP, TwoStageMLP, compute_pattern_labels_batch,
)


def evaluate(model_even, model_odd, mode, patterns,
             pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
             chunk_path, feature_cols, device, batch_size=1024, n_sample=49 * 10000):
    X, Y, pos = _load_features(chunk_path)
    if feature_cols is not None:
        X = X[:, feature_cols]

    n = min(len(X), n_sample)
    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(len(X), n, replace=False))
    X, Y, pos = X[idx], Y[idx], pos[idx]

    tp = fp = fn = tn = 0
    # Per-bucket accuracy
    buckets = [(5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
    bucket_stats = {b: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for b in buckets}

    model_even.eval(); model_odd.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = X[i:i + batch_size].to(device)
            y_board = Y[i:i + batch_size]
            p = pos[i:i + batch_size]
            em = (p % 2 == 0); om = ~em

            if mode in ("direct", "randproj"):
                pl = torch.zeros(len(x), len(patterns), device=device)
                if em.any(): pl[em] = model_even(x[em])
                if om.any(): pl[om] = model_odd(x[om])
            else:
                pl = torch.zeros(len(x), len(patterns), device=device)
                for mask, model in [(em, model_even), (om, model_odd)]:
                    if not mask.any(): continue
                    logits, _ = model(x[mask], p[mask])
                    pl[mask] = logits

            pred = (pl > 0).cpu().numpy()
            gt = compute_pattern_labels_batch(
                y_board.numpy(), p.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask).astype(bool)

            tp += ((pred & gt)).sum()
            fp += ((pred & ~gt)).sum()
            fn += ((~pred & gt)).sum()
            tn += ((~pred & ~gt)).sum()

            p_np = p.numpy()
            for lo, hi in buckets:
                m = (p_np >= lo) & (p_np < hi)
                if not m.any(): continue
                s = bucket_stats[(lo, hi)]
                s['tp'] += (pred[m] & gt[m]).sum()
                s['fp'] += (pred[m] & ~gt[m]).sum()
                s['fn'] += (~pred[m] & gt[m]).sum()
                s['tn'] += (~pred[m] & ~gt[m]).sum()

    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}, bucket_stats


def _report(s, label):
    tp, fp, fn, tn = s['tp'], s['fp'], s['fn'], s['tn']
    total = tp + fp + fn + tn
    acc = (tp + tn) / max(total, 1)
    recall = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    print(f"  {label:16s} acc={acc:7.4%}  recall={recall:7.4%}  prec={prec:7.4%}  (n={total//960})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["direct", "emergent", "e2e", "two-stage", "randproj"])
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    feature_cols = list(range(N_MOVES, 2 * N_MOVES))

    ckpt = torch.load(args.ckpt, map_location=device)
    n_patterns = ckpt.get('n_patterns', 960)

    if args.mode in ("direct", "randproj"):
        me = DirectMLP(N_MOVES, args.hidden, n_patterns).to(device)
        mo = DirectMLP(N_MOVES, args.hidden, n_patterns).to(device)
    elif args.mode == "two-stage":
        me = TwoStageMLP(N_MOVES, args.hidden, n_patterns).to(device)
        mo = TwoStageMLP(N_MOVES, args.hidden, n_patterns).to(device)
    else:
        me = EndToEndMLP(N_MOVES, args.hidden, n_patterns).to(device)
        mo = EndToEndMLP(N_MOVES, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even'])
    mo.load_state_dict(ckpt['odd'])

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    eval_path = chunk_files[-1]

    print(f"Mode: {args.mode}, H={args.hidden}")
    print(f"Ckpt: {os.path.basename(args.ckpt)} (pat_acc reported={ckpt.get('best_pat_acc', '?')})")
    print(f"Eval: {os.path.basename(eval_path)} (random sample)")

    overall, buckets = evaluate(
        me, mo, args.mode, patterns,
        pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
        eval_path, feature_cols, device)

    print(f"\n  {'metric':16s} {'acc':>10}  {'recall':>10}  {'prec':>10}")
    _report(overall, 'OVERALL')
    print()
    for (lo, hi), s in buckets.items():
        _report(s, f'pos {lo:>2}-{hi:<2}')
