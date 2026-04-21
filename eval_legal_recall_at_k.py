"""For each position with K legal moves, compute how many of those K
moves are in the model's top K predicted cells. Report the average.

This is "recall at K" where K varies per position.
  - score = 1 means the model perfectly ranks all legal moves above
    every illegal move.
  - score = K_correct / K per position, averaged.

Usage:
    python eval_legal_recall_at_k.py --ckpt <path> --mode direct --hidden 512
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
    to_signed_parity_input, to_mine_signed_input, to_board_state_input,
    to_color_split_input, to_played_halfmask_input, to_played_bit_input,
)


def prob_or_scores(pat_logits, idx, mask):
    log1m = -nn.functional.softplus(pat_logits)
    gathered = log1m[:, idx]
    gathered = gathered.masked_fill(~mask, 0.0)
    return -gathered.sum(dim=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    n_patterns = ckpt.get('n_patterns', 960)
    input_dim = ckpt.get('input_dim', N_MOVES)

    # Infer feature preprocessing
    name = os.path.basename(args.ckpt)
    _feat_cols_map = {
        "when":        list(range(N_MOVES, 2 * N_MOVES)),
        "played":      list(range(0, N_MOVES)),
        "played+when": list(range(0, 2 * N_MOVES)),
        "when+even":   list(range(N_MOVES, 3 * N_MOVES)),
        "played+even": list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)),
        "all":         list(range(0, 3 * N_MOVES)),
    }
    feature_cols, feature_fn = None, None
    if "wheneven" in name: feature_cols = _feat_cols_map["when+even"]
    elif "playedeven" in name: feature_cols = _feat_cols_map["played+even"]
    elif "signed_parity" in name:
        feature_fn = lambda X, Y, pos: to_signed_parity_input(X)
    elif "color_split" in name:
        feature_fn = lambda X, Y, pos: to_color_split_input(X)
    elif input_dim == 120: feature_cols = _feat_cols_map["when+even"]
    elif input_dim == 180: feature_cols = _feat_cols_map["all"]
    else: feature_cols = _feat_cols_map["when"]

    Cls = {"direct": DirectMLP, "randproj": DirectMLP,
           "two-stage": TwoStageMLP,
           "emergent": EndToEndMLP, "e2e": EndToEndMLP}[args.mode]
    me = Cls(input_dim, args.hidden, n_patterns).to(device)
    mo = Cls(input_dim, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    pattern_to_cell = torch.tensor([MOVE_TO_IDX[p['target']] for p in patterns],
                                   dtype=torch.long, device=device)
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    X, Y, pos = _load_features(chunk_files[-1])
    if feature_cols is not None:
        X = X[:, feature_cols]
    elif feature_fn is not None:
        X = feature_fn(X, Y, pos)
    n = min(len(X), 49 * 10000)
    rng = np.random.RandomState(0)
    si = np.sort(rng.choice(len(X), n, replace=False))
    X, Y, pos = X[si], Y[si], pos[si]

    # Running totals for "recall at K per position"
    per_k = {}   # K -> (correct_sum, total_positions)
    overall_correct_frac = 0.0
    overall_n = 0

    with torch.no_grad():
        for i in range(0, n, 1024):
            x = X[i:i+1024].to(device); yb = Y[i:i+1024]; p = pos[i:i+1024]
            em = (p % 2 == 0); om = ~em
            pl = torch.zeros(len(x), 960, device=device)
            if em.any(): pl[em] = me(x[em])
            if om.any(): pl[om] = mo(x[om])
            gp = torch.from_numpy(compute_pattern_labels_batch(
                yb.numpy(), p.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)).to(device)
            legal = pat_labels_to_cell_labels(gp, pattern_to_cell)
            scores = prob_or_scores(pl, idx, mask).cpu().numpy()
            gl = (legal > 0.5).cpu().numpy()

            for b in range(scores.shape[0]):
                legal_set = set(np.where(gl[b])[0].tolist())
                K = len(legal_set)
                if K == 0: continue
                top_k = set(np.argsort(-scores[b])[:K].tolist())
                hits = len(top_k & legal_set)
                frac = hits / K
                overall_correct_frac += frac
                overall_n += 1
                if K not in per_k:
                    per_k[K] = [0.0, 0]
                per_k[K][0] += frac; per_k[K][1] += 1

    mean_frac = overall_correct_frac / overall_n
    print(f"Ckpt: {name}")
    print(f"Positions evaluated: {overall_n}")
    print(f"OVERALL recall@K (K = num legal moves): {mean_frac:.4%}")
    print()
    print(f"{'K':>4}  {'positions':>10}  {'recall@K':>10}")
    for K in sorted(per_k.keys()):
        s, ct = per_k[K]
        print(f"{K:>4}  {ct:>10d}  {s/ct:>10.4%}")
