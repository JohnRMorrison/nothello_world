"""Evaluate top-N legal move accuracy.

For each position with K actual legal moves:
  Look at the model's top min(N, K) predicted moves.
  Accuracy = fraction of those that are actually legal.
  (When N >= K, this is the same as the Li top-N metric.)

Reports for N=1, 3, 5, 10.

Usage:
    python eval_topk_legal.py --ckpt pattern_simple_direct_H512.pt \
        --mode direct --hidden 512
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES, OPTIONS,
)
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import (
    DirectMLP, EndToEndMLP, TwoStageMLP, compute_pattern_labels_batch,
)


def patterns_to_move_scores(pat_logits, patterns):
    """Aggregate 960 pattern logits → 60-d move scores (max per cell)."""
    n = len(pat_logits)
    scores = np.full((n, 60), -1e9, dtype=np.float32)
    for j, pat in enumerate(patterns):
        move_idx = MOVE_TO_IDX[pat['target']]
        scores[:, move_idx] = np.maximum(scores[:, move_idx], pat_logits[:, j])
    return scores


def evaluate(model_even, model_odd, mode, patterns,
             pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
             chunk_path, feature_cols, device, batch_size=1024,
             top_ns=(1, 3, 5, 10)):
    X, Y, pos = _load_features(chunk_path)
    if feature_cols is not None:
        X = X[:, feature_cols]
    n = min(len(X), 49 * 10000)
    rng = np.random.RandomState(0)
    sample_idx = np.sort(rng.choice(len(X), n, replace=False))
    X, Y, pos = X[sample_idx], Y[sample_idx], pos[sample_idx]

    # For each top_n, accumulate (correct, total)
    results = {n_: {'correct': 0, 'total': 0} for n_ in top_ns}
    n_legal_dist = []

    model_even.eval(); model_odd.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = X[i:i + batch_size].to(device)
            y_board = Y[i:i + batch_size]
            p = pos[i:i + batch_size]
            even_mask = (p % 2 == 0)
            odd_mask = ~even_mask

            if mode in ("direct", "randproj"):
                pat_logits = torch.zeros(len(x), len(patterns), device=device)
                if even_mask.any():
                    pat_logits[even_mask] = model_even(x[even_mask])
                if odd_mask.any():
                    pat_logits[odd_mask] = model_odd(x[odd_mask])
            else:
                pat_logits = torch.zeros(len(x), len(patterns), device=device)
                for mask, model in [(even_mask, model_even), (odd_mask, model_odd)]:
                    if not mask.any():
                        continue
                    pl, _ = model(x[mask], p[mask])
                    pat_logits[mask] = pl

            pred_pat = pat_logits.cpu().numpy()
            gt_pat = compute_pattern_labels_batch(
                y_board.numpy(), p.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)

            pred_scores = patterns_to_move_scores(pred_pat, patterns)
            gt_scores = patterns_to_move_scores(gt_pat, patterns)
            gt_legal = (gt_scores > 0.5)  # (batch, 60) bool

            for b in range(len(x)):
                legal_set = set(np.where(gt_legal[b])[0].tolist())
                K = len(legal_set)
                if K == 0:
                    continue
                n_legal_dist.append(K)
                # Sort cells by predicted score, descending
                ranked = np.argsort(-pred_scores[b])
                for top_n in top_ns:
                    n_to_check = min(top_n, K)
                    top_cells = set(ranked[:n_to_check].tolist())
                    correct = len(top_cells & legal_set)
                    results[top_n]['correct'] += correct
                    results[top_n]['total'] += n_to_check

    return results, n_legal_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["direct", "emergent", "e2e", "two-stage", "randproj"])
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    parser.add_argument("--chunk-prefix", default="chunk_",
                        help="Filename prefix filter for chunks (e.g. chunk_ext_ for "
                             "the extended-range chunks used by the played+even MLPs).")
    args = parser.parse_args()

    device = get_device()

    ckpt = torch.load(args.ckpt, map_location=device)
    n_patterns = ckpt.get('n_patterns', 960)

    # Infer feature preprocessing from the checkpoint filename (matches
    # probe_pattern_models.py's convention).  The old default was hardcoded
    # to "when" features (line was `feature_cols = list(range(N_MOVES, 2*N_MOVES))`),
    # which silently gave random accuracy for MLPs trained on other feature sets.
    name = os.path.basename(args.ckpt)
    _feat_cols_map = {
        "when":        list(range(N_MOVES, 2 * N_MOVES)),
        "played":      list(range(0, N_MOVES)),
        "played+when": list(range(0, 2 * N_MOVES)),
        "when+even":   list(range(N_MOVES, 3 * N_MOVES)),
        "played+even": list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)),
        "all":         list(range(0, 3 * N_MOVES)),
    }
    if "playedeven" in name:
        feature_cols = _feat_cols_map["played+even"]
    elif "wheneven" in name:
        feature_cols = _feat_cols_map["when+even"]
    elif "when" in name and "even" not in name:
        feature_cols = _feat_cols_map["when"]
    elif "played" in name and "when" not in name and "even" not in name:
        feature_cols = _feat_cols_map["played"]
    else:
        # Fall back to legacy "when" default so old runs stay reproducible.
        feature_cols = _feat_cols_map["when"]
    input_dim = len(feature_cols)
    print(f"Detected features from ckpt name: input_dim={input_dim}")

    if args.mode in ("direct", "randproj"):
        model_even = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
        model_odd = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    elif args.mode == "two-stage":
        model_even = TwoStageMLP(input_dim, args.hidden, n_patterns).to(device)
        model_odd = TwoStageMLP(input_dim, args.hidden, n_patterns).to(device)
    else:
        model_even = EndToEndMLP(input_dim, args.hidden, n_patterns).to(device)
        model_odd = EndToEndMLP(input_dim, args.hidden, n_patterns).to(device)
    model_even.load_state_dict(ckpt['even'])
    model_odd.load_state_dict(ckpt['odd'])

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.startswith(args.chunk_prefix)
                         and f.endswith(".npz")
                         and "_patterns" not in f and "_when60" not in f)
    if not chunk_files:
        raise SystemExit(
            f"No files matching {args.chunk_prefix}*.npz in {chunk_dir}. "
            f"Pass --chunk-prefix to match the training-time chunk family.")
    eval_path = chunk_files[-1]

    print(f"Mode: {args.mode}, H={args.hidden}")
    print(f"Eval: {os.path.basename(eval_path)}")

    results, n_legal_dist = evaluate(
        model_even, model_odd, args.mode, patterns,
        pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
        eval_path, feature_cols, device)

    print(f"\nLegal moves per position: mean={np.mean(n_legal_dist):.2f}, median={int(np.median(n_legal_dist))}, max={max(n_legal_dist)}")
    print(f"N positions evaluated: {len(n_legal_dist)}")
    print(f"\n{'top-N':>6}  {'accuracy':>10}  {'correct/total':>15}")
    print("-" * 40)
    for top_n, d in results.items():
        acc = d['correct'] / d['total'] if d['total'] > 0 else 0
        print(f"{top_n:>6}  {acc:>10.4%}  {d['correct']:>7}/{d['total']:<7}")
