"""Evaluate each turn specialist (and the unified H=512 wheneven baseline)
on test positions at the specialist's target turn. Reports top-1, top-3,
top-5, top-10 (recall@K) using prob_or aggregator.

Output table:
    turn   specialist_top1   unified_top1   spec_top10   unif_top10  ...

Usage:
    python eval_turn_specialists.py
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES,
)
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import (
    DirectMLP, compute_pattern_labels_batch, pat_labels_to_cell_labels,
    _get_cell_pat_index,
)


CKPT_DIR = "experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints"
TURNS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def prob_or_score(pat_logits, idx, mask):
    log1m = -F.softplus(pat_logits)
    g = log1m[:, idx].masked_fill(~mask, 0.0)
    return 1.0 - torch.exp(g.sum(dim=-1))   # (B, 60)


def evaluate(model_even, model_odd, feat_X, Y, pos, device,
             pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
             pattern_to_cell, idx_t, mask_t, top_ns=(1, 3, 5, 10)):
    """Returns dict top_n -> recall."""
    results = {n: {'c': 0, 't': 0} for n in top_ns}
    batch = 1024
    with torch.no_grad():
        for i in range(0, len(feat_X), batch):
            xb = feat_X[i:i + batch].to(device)
            yb = Y[i:i + batch]
            pb = pos[i:i + batch]
            em = (pb % 2 == 0); om = ~em
            preds = torch.zeros(len(xb), 960, device=device)
            if em.any(): preds[em] = model_even(xb[em])
            if om.any(): preds[om] = model_odd(xb[om])
            scores = prob_or_score(preds, idx_t, mask_t).cpu().numpy()
            # Compute legal labels
            gp = torch.from_numpy(compute_pattern_labels_batch(
                yb.numpy(), pb.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask
            )).to(device)
            legal = (pat_labels_to_cell_labels(gp, pattern_to_cell) > 0.5).cpu().numpy()
            for b in range(len(xb)):
                ls = set(np.where(legal[b])[0].tolist())
                K = len(ls)
                if K == 0: continue
                r = np.argsort(-scores[b])
                for n in top_ns:
                    k = min(n, K)
                    results[n]['c'] += len(set(r[:k].tolist()) & ls)
                    results[n]['t'] += k
    return {n: results[n]['c'] / max(results[n]['t'], 1) for n in top_ns}


def load_model(ckpt_path, hidden, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    input_dim = ckpt.get('input_dim', 120)
    me = DirectMLP(input_dim, hidden, 960).to(device)
    mo = DirectMLP(input_dim, hidden, 960).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()
    return me, mo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-ckpt",
                        default=f"{CKPT_DIR}/pattern_simple_direct_H512_wheneven.pt")
    parser.add_argument("--specialist-tmpl",
                        default=f"{CKPT_DIR}/pattern_simple_direct_H512_wheneven_turn{{T}}.pt")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--features", default="when+even")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Patterns
    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = \
        precompute_pattern_arrays(patterns)
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device)
    idx_t, mask_t = _get_cell_pat_index(pattern_to_cell, 60)

    # Load eval chunk
    out_dir = "experiments/mathematical_transformation_experiments/heuristic_probe_results"
    chunk_dir = os.path.join(out_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f
                         and "_when60" not in f and "_by_black" not in f)
    eval_path = chunk_files[-1]
    print(f"Loading {eval_path}")
    X, Y, pos = _load_features(eval_path)
    feat_X = X[:, N_MOVES:3 * N_MOVES]
    del X
    pos_np = pos.numpy()

    # Load unified
    print(f"\nLoading unified: {args.unified_ckpt}")
    me_u, mo_u = load_model(args.unified_ckpt, args.hidden, device)

    # Per-turn evaluation
    print()
    print("=" * 78)
    print("Turn-stratified top-K legal recall comparison")
    print("=" * 78)
    header = (f"{'turn':>5s} {'n':>8s} "
              f"{'spec_top1':>10s} {'unif_top1':>10s} {'Δ':>6s} "
              f"{'spec_top10':>11s} {'unif_top10':>11s} {'Δ':>6s}")
    print(header)
    print("-" * len(header))
    for T in TURNS:
        # Filter to positions at this turn
        m = (pos_np == T)
        if m.sum() == 0:
            print(f"  {T:>3d}     0   -- no positions at this turn --")
            continue
        idx = np.where(m)[0]
        feat_T = feat_X[idx]; Y_T = Y[idx]; pos_T = pos[idx]
        n_T = len(idx)

        # Load specialist
        spec_path = args.specialist_tmpl.format(T=T)
        if not os.path.exists(spec_path):
            print(f"  {T:>3d}  {n_T:>8d}  -- specialist missing: {spec_path}")
            continue
        me_s, mo_s = load_model(spec_path, args.hidden, device)

        # Evaluate both
        spec_res = evaluate(me_s, mo_s, feat_T, Y_T, pos_T, device,
                            pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
                            pattern_to_cell, idx_t, mask_t)
        unif_res = evaluate(me_u, mo_u, feat_T, Y_T, pos_T, device,
                            pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
                            pattern_to_cell, idx_t, mask_t)

        d1  = spec_res[1] - unif_res[1]
        d10 = spec_res[10] - unif_res[10]
        print(f"  {T:>3d} {n_T:>8d} "
              f"{spec_res[1]:>10.4f} {unif_res[1]:>10.4f} {d1:>+6.4f} "
              f"{spec_res[10]:>11.4f} {unif_res[10]:>11.4f} {d10:>+6.4f}")
