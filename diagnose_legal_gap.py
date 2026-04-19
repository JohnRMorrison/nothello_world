"""Diagnose where top-1 legal predictions fail.

For each position:
  1. Compute pattern logits, aggregate with prob_or to 60-d cell scores.
  2. If top-1 is a legal cell: "correct".
  3. Otherwise: record the *rank of the best legal cell* (smallest index
     among legal cells in the score ranking).

Output:
  - Per-position-bucket top-1 accuracy (opening vs endgame).
  - For MISSES, histogram of "rank of best legal cell":
      rank 1  = (never, this is the correct case)
      rank 2  = the right answer was second-best
      rank 10 = we were far off
  - Distribution of (score margin) between top-1 and the best legal cell.

Usage:
    python diagnose_legal_gap.py --ckpt pattern_simple_direct_H512.pt \
        --mode direct --hidden 512
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES,
)
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import (
    DirectMLP, EndToEndMLP, TwoStageMLP, compute_pattern_labels_batch,
    pat_labels_to_cell_labels, _get_cell_pat_index,
)


def prob_or_scores(pat_logits, idx, mask):
    log1m = -nn.functional.softplus(pat_logits)
    gathered = log1m[:, idx]
    gathered = gathered.masked_fill(~mask, 0.0)
    return -gathered.sum(dim=-1)


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
    Cls = {"direct": DirectMLP, "randproj": DirectMLP,
           "two-stage": TwoStageMLP,
           "emergent": EndToEndMLP, "e2e": EndToEndMLP}[args.mode]
    me = Cls(N_MOVES, args.hidden, n_patterns).to(device)
    mo = Cls(N_MOVES, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()
    print(f"Loaded {args.ckpt}")

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device)
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    eval_path = chunk_files[-1]

    X, Y, pos = _load_features(eval_path)
    X = X[:, feature_cols]
    n = min(len(X), 49 * 10000)
    rng = np.random.RandomState(0)
    si = np.sort(rng.choice(len(X), n, replace=False))
    X, Y, pos = X[si], Y[si], pos[si]

    # Accumulators
    buckets = [(5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
    bucket_totals = {b: {'correct': 0, 'total': 0} for b in buckets}
    best_legal_rank = []    # rank of best legal cell on MISSES (1..60)
    margin_to_best_legal = []  # score(top1) - score(best_legal) on MISSES
    # Legal-cell-count distribution among misses
    n_legal_in_misses = []

    with torch.no_grad():
        for i in range(0, n, 1024):
            x = X[i:i+1024].to(device); yb = Y[i:i+1024]; p = pos[i:i+1024]
            em = (p % 2 == 0); om = ~em
            pl = torch.zeros(len(x), 960, device=device)
            if em.any(): pl[em] = me(x[em]) if args.mode in ("direct","randproj") else me(x[em], p[em])[0]
            if om.any(): pl[om] = mo(x[om]) if args.mode in ("direct","randproj") else mo(x[om], p[om])[0]
            gp = torch.from_numpy(compute_pattern_labels_batch(
                yb.numpy(), p.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
            ).to(device)
            legal = pat_labels_to_cell_labels(gp, pattern_to_cell)
            cs = prob_or_scores(pl, idx, mask)

            cs_np = cs.cpu().numpy()
            gl_np = (legal > 0.5).cpu().numpy()
            p_np = p.numpy()
            for b in range(cs_np.shape[0]):
                legal_cells = np.where(gl_np[b])[0]
                K = len(legal_cells)
                if K == 0: continue
                # Find bucket
                for lo, hi in buckets:
                    if lo <= p_np[b] < hi:
                        break
                else:
                    continue
                ranked = np.argsort(-cs_np[b])  # cells from best to worst
                top1 = ranked[0]
                total = bucket_totals[(lo, hi)]
                total['total'] += 1
                if top1 in legal_cells:
                    total['correct'] += 1
                    continue
                # Miss — find rank of best legal cell
                # ranked[i] is the cell at rank i+1; find smallest i where ranked[i] is legal
                rank = 61
                for r, c in enumerate(ranked):
                    if c in legal_cells:
                        rank = r + 1
                        break
                best_legal_rank.append(rank)
                best_legal_cell = legal_cells[np.argmax(cs_np[b, legal_cells])]
                margin_to_best_legal.append(cs_np[b, top1] - cs_np[b, best_legal_cell])
                n_legal_in_misses.append(K)

    # Report
    print(f"\n{'Bucket':>10s}  {'correct/total':>20s}  {'accuracy':>10s}")
    print("-" * 50)
    for b in buckets:
        d = bucket_totals[b]
        acc = d['correct'] / max(d['total'], 1)
        print(f"  pos {b[0]:2d}-{b[1]:<2d}  {d['correct']:>10d} / {d['total']:>6d}  {acc:>9.4%}")

    n_miss = len(best_legal_rank)
    n_total = sum(d['total'] for d in bucket_totals.values())
    n_correct = sum(d['correct'] for d in bucket_totals.values())
    print(f"\nOVERALL: {n_correct}/{n_total} = {n_correct/n_total:.4%}")
    print(f"Misses: {n_miss}")

    if n_miss > 0:
        ranks = np.array(best_legal_rank)
        print("\nRank of best legal cell on MISSES:")
        for r in (2, 3, 4, 5, 6, 10, 15, 30):
            frac = (ranks <= r).mean()
            print(f"  <= {r:>3d}: {frac:>7.2%}  ({(ranks <= r).sum()} of {n_miss})")
        print(f"  mean rank: {ranks.mean():.2f}, median: {int(np.median(ranks))}")

        margins = np.array(margin_to_best_legal)
        print(f"\nMargin (score of WRONG top-1 minus score of BEST LEGAL on misses):")
        print(f"  mean: {margins.mean():.4f}, median: {np.median(margins):.4f}")
        print(f"  small margins (<0.1): {(margins < 0.1).mean():.2%}")
        print(f"  huge margins (>1.0):  {(margins > 1.0).mean():.2%}")

        print(f"\nLegal-move count on misses: mean={np.mean(n_legal_in_misses):.2f}, median={int(np.median(n_legal_in_misses))}")
