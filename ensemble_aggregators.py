"""Ensemble scores from multiple frozen checkpoints, pick argmax.

For each position: compute prob_or cell scores from each model, average,
argmax. No training — just averaging. Since 91% of H=512 misses have the
right answer at rank <=3, even a small score nudge from another model
can flip many argmax decisions.

Usage:
    python ensemble_aggregators.py \
        --ckpts H512=pattern_simple_direct_H512.pt,H1024=pattern_simple_direct_H1024.pt \
        --modes direct,direct --hiddens 512,1024
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


def load_model(ckpt_path, mode, hidden, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    n_patterns = ckpt.get('n_patterns', 960)
    Cls = {"direct": DirectMLP, "randproj": DirectMLP,
           "two-stage": TwoStageMLP,
           "emergent": EndToEndMLP, "e2e": EndToEndMLP}[mode]
    me = Cls(N_MOVES, hidden, n_patterns).to(device)
    mo = Cls(N_MOVES, hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()
    return me, mo


def prob_or_scores(pat_logits, idx, mask):
    log1m = -nn.functional.softplus(pat_logits)
    gathered = log1m[:, idx]
    gathered = gathered.masked_fill(~mask, 0.0)
    return -gathered.sum(dim=-1)


def run_model(me, mo, mode, x, pos):
    em = (pos % 2 == 0); om = ~em
    pl = torch.zeros(len(x), 960, device=x.device)
    if mode in ("direct", "randproj"):
        if em.any(): pl[em] = me(x[em])
        if om.any(): pl[om] = mo(x[om])
    else:
        for mask, m in [(em, me), (om, mo)]:
            if not mask.any(): continue
            logits, _ = m(x[mask], pos[mask])
            pl[mask] = logits
    return pl


def score_one(models_list, x, pos, idx, mask):
    """Return per-model cell scores and the average."""
    scores = []
    for me, mo, mode in models_list:
        pl = run_model(me, mo, mode, x, pos)
        scores.append(prob_or_scores(pl, idx, mask))
    avg = sum(scores) / len(scores)
    return scores, avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", required=True,
                        help="Comma-separated LABEL=PATH pairs")
    parser.add_argument("--modes", required=True,
                        help="Comma-separated modes matching --ckpts")
    parser.add_argument("--hiddens", required=True,
                        help="Comma-separated hidden dims matching --ckpts")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    feature_cols = list(range(N_MOVES, 2 * N_MOVES))

    pairs = args.ckpts.split(",")
    modes = args.modes.split(",")
    hiddens = [int(h) for h in args.hiddens.split(",")]
    labels, paths = zip(*[p.split("=") for p in pairs])
    assert len(paths) == len(modes) == len(hiddens), "mismatched lists"

    models_list = []
    for lbl, path, mode, H in zip(labels, paths, modes, hiddens):
        me, mo = load_model(path, mode, H, device)
        models_list.append((me, mo, mode))
        print(f"Loaded [{lbl}] {path}")

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

    # Accumulate top-1/3/5/10 for each model individually AND for average
    names = list(labels) + ["AVG"]
    totals = {name: {k: {'c': 0, 't': 0} for k in (1, 3, 5, 10)} for name in names}

    with torch.no_grad():
        for i in range(0, n, 1024):
            x = X[i:i+1024].to(device); yb = Y[i:i+1024]; p = pos[i:i+1024]
            scores_list, avg = score_one(models_list, x, p, idx, mask)
            gp = torch.from_numpy(compute_pattern_labels_batch(
                yb.numpy(), p.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
            ).to(device)
            legal = pat_labels_to_cell_labels(gp, pattern_to_cell)
            gl = (legal > 0.5).cpu().numpy()
            all_scores = [s.cpu().numpy() for s in scores_list] + [avg.cpu().numpy()]
            for name, scores_np in zip(names, all_scores):
                for b in range(scores_np.shape[0]):
                    ls = set(np.where(gl[b])[0].tolist()); K = len(ls)
                    if K == 0: continue
                    r = np.argsort(-scores_np[b])
                    for k in (1, 3, 5, 10):
                        kk = min(k, K)
                        totals[name][k]['c'] += len(set(r[:kk].tolist()) & ls)
                        totals[name][k]['t'] += kk

    print(f"\n{'model':>12s}  {'top-1':>10} {'top-3':>10} {'top-5':>10} {'top-10':>10}")
    print("-" * 60)
    for name in names:
        row = [f"{name:>12s}"]
        for k in (1, 3, 5, 10):
            d = totals[name][k]
            row.append(f"{d['c']/max(d['t'],1):>9.4%}")
        print("  " + "  ".join(row))
