"""Compare different pattern->cell aggregators on a frozen checkpoint.

Pure post-hoc: loads model once, computes pattern logits, then scores
top-N legal under several aggregation rules. No training needed.

Aggregators:
  max          — current default (argmax over patterns per cell)
  mean         — average over the ~16 patterns targeting a cell
  median       — median over patterns
  logsumexp    — smooth max (temperature 1.0)
  topk_mean-K  — average of top-K patterns (K=2,3,5)
  prob_or      — 1 - prod(1 - sigmoid(logit)) over patterns

Usage:
    python compare_aggregators.py --ckpt pattern_simple_direct_H512.pt \
        --mode direct --hidden 512
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
    pat_labels_to_cell_labels, _get_cell_pat_index,
)


def _gather(pat_logits, idx, mask, fill=float('-inf')):
    gathered = pat_logits[:, idx]
    return gathered.masked_fill(~mask, fill)


def agg_max(pat_logits, idx, mask):
    return _gather(pat_logits, idx, mask, float('-inf')).max(dim=-1).values


def agg_mean(pat_logits, idx, mask):
    gathered = _gather(pat_logits, idx, mask, 0.0)
    counts = mask.sum(dim=-1).clamp(min=1).to(gathered.dtype)
    return gathered.sum(dim=-1) / counts


def agg_median(pat_logits, idx, mask):
    # Set invalid to -inf so valid values dominate the median for cells with
    # an odd mix. For even cells this is slightly biased low on invalid-heavy
    # cells — use sort+pick instead.
    gathered = _gather(pat_logits, idx, mask, float('-inf'))
    # For each cell, take median of the valid entries only
    out = torch.zeros(gathered.shape[:2], dtype=gathered.dtype, device=gathered.device)
    counts = mask.sum(dim=-1)
    sorted_, _ = gathered.sort(dim=-1, descending=True)
    # median index per cell
    med_idx = (counts - 1) // 2
    for c in range(gathered.shape[1]):
        out[:, c] = sorted_[:, c, med_idx[c]]
    return out


def agg_logsumexp(pat_logits, idx, mask):
    gathered = _gather(pat_logits, idx, mask, float('-inf'))
    return torch.logsumexp(gathered, dim=-1)


def agg_topk_mean(k):
    def _agg(pat_logits, idx, mask):
        gathered = _gather(pat_logits, idx, mask, float('-inf'))
        # top-K across patterns
        topk_vals, _ = gathered.topk(min(k, gathered.shape[-1]), dim=-1)
        # Average of valid top-K (ignore -inf from padding)
        valid = torch.isfinite(topk_vals)
        safe = torch.where(valid, topk_vals, torch.zeros_like(topk_vals))
        counts = valid.sum(dim=-1).clamp(min=1).to(safe.dtype)
        return safe.sum(dim=-1) / counts
    return _agg


def agg_prob_or(pat_logits, idx, mask):
    # P(cell fires) = 1 - prod_j (1 - sigmoid(logit_j))
    # Log-space: sum_j log(1 - sigmoid(x)) = sum_j -softplus(x) = log(1 - P)
    # Score = -log(1 - P) is monotonic in P, so rank by that.
    log1m = -torch.nn.functional.softplus(pat_logits)
    gathered = log1m[:, idx]
    gathered = gathered.masked_fill(~mask, 0.0)   # log(1 - 0) = 0 for padding
    return -gathered.sum(dim=-1)                   # -log(1-P): higher = more likely legal


def score(agg, pat_logits, idx, mask, legal, top_ns=(1, 3, 5, 10)):
    cs = agg(pat_logits, idx, mask).cpu().numpy()
    gl = (legal > 0.5).cpu().numpy()
    results = {n: {'c': 0, 't': 0} for n in top_ns}
    for b in range(cs.shape[0]):
        ls = set(np.where(gl[b])[0].tolist()); K = len(ls)
        if K == 0: continue
        r = np.argsort(-cs[b])
        for n in top_ns:
            k = min(n, K)
            results[n]['c'] += len(set(r[:k].tolist()) & ls)
            results[n]['t'] += k
    return results


def accumulate(total, new):
    for n in total:
        total[n]['c'] += new[n]['c']
        total[n]['t'] += new[n]['t']


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
    print(f"Loaded {args.ckpt} (pat_acc={ckpt.get('best_pat_acc', '?')})")

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
    print(f"Eval: {os.path.basename(eval_path)} (random sample)")

    X, Y, pos = _load_features(eval_path)
    X = X[:, feature_cols]
    n = min(len(X), 49 * 10000)
    rng = np.random.RandomState(0)
    si = np.sort(rng.choice(len(X), n, replace=False))
    X, Y, pos = X[si], Y[si], pos[si]

    aggregators = {
        'max':         agg_max,
        'mean':        agg_mean,
        'median':      agg_median,
        'logsumexp':   agg_logsumexp,
        'topk_mean-2': agg_topk_mean(2),
        'topk_mean-3': agg_topk_mean(3),
        'topk_mean-5': agg_topk_mean(5),
        'prob_or':     agg_prob_or,
    }
    totals = {name: {n: {'c': 0, 't': 0} for n in (1, 3, 5, 10)}
              for name in aggregators}

    batch = 1024
    with torch.no_grad():
        for i in range(0, n, batch):
            x = X[i:i + batch].to(device)
            yb = Y[i:i + batch]
            p = pos[i:i + batch]
            em = (p % 2 == 0); om = ~em
            if args.mode in ("direct", "randproj"):
                pl = torch.zeros(len(x), 960, device=device)
                if em.any(): pl[em] = me(x[em])
                if om.any(): pl[om] = mo(x[om])
            else:
                pl = torch.zeros(len(x), 960, device=device)
                for msk, m in [(em, me), (om, mo)]:
                    if not msk.any(): continue
                    logits, _ = m(x[msk], p[msk])
                    pl[msk] = logits
            gp = torch.from_numpy(compute_pattern_labels_batch(
                yb.numpy(), p.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
            ).to(device)
            legal = pat_labels_to_cell_labels(gp, pattern_to_cell)
            for name, agg in aggregators.items():
                accumulate(totals[name], score(agg, pl, idx, mask, legal))

    print(f"\n{'aggregator':16s} {'top-1':>10} {'top-3':>10} {'top-5':>10} {'top-10':>10}")
    print("-" * 60)
    for name in aggregators:
        row = [name]
        for nn_ in (1, 3, 5, 10):
            d = totals[name][nn_]
            row.append(f"{d['c']/max(d['t'],1):.4%}")
        print(f"{row[0]:16s} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}")
