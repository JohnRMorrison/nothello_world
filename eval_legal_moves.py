"""Evaluate whether the model's top-1 predicted move is legal.

For each position: aggregate 960 pattern logits → 60 move scores,
pick the argmax, check if it's a legal move.

Usage:
    python eval_legal_moves.py --ckpt pattern_simple_direct_H512.pt --mode direct --hidden 512
"""
import sys, os
sys.path.insert(0, '.')

import argparse
import numpy as np
import torch
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES, OPTIONS,
)
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX, VALID_MOVES
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
             chunk_path, feature_cols, device, batch_size=1024):
    """Evaluate top-1 legal move accuracy on one chunk."""
    X, Y, pos = _load_features(chunk_path)
    if feature_cols is not None:
        X = X[:, feature_cols]

    n = min(len(X), 49 * 10000)
    X, Y, pos = X[:n], Y[:n], pos[:n]

    top1_correct = 0
    top1_total = 0
    top5_correct = 0
    all_legal_correct = 0  # all 60 squares classified correctly

    model_even.eval(); model_odd.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = X[i:i + batch_size].to(device)
            y_board = Y[i:i + batch_size]
            p = pos[i:i + batch_size]
            even_mask = (p % 2 == 0)
            odd_mask = ~even_mask

            # Model predictions
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

            # Ground truth patterns → ground truth legal moves
            gt_pat = compute_pattern_labels_batch(
                y_board.numpy(), p.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)

            # Aggregate to 60-d move scores
            pred_scores = patterns_to_move_scores(pred_pat, patterns)
            gt_scores = patterns_to_move_scores(gt_pat, patterns)

            gt_legal = (gt_scores > 0.5)  # (batch, 60) bool

            for b in range(len(x)):
                legal_set = set(np.where(gt_legal[b])[0])
                if not legal_set:
                    continue

                # Top-1: is argmax legal?
                top1 = np.argmax(pred_scores[b])
                if top1 in legal_set:
                    top1_correct += 1
                top1_total += 1

                # Top-5: is any of top-5 legal?
                top5 = set(np.argsort(pred_scores[b])[-5:])
                if top5 & legal_set:
                    top5_correct += 1

                # All-correct: every square classified right
                pred_legal = set(np.where(pred_scores[b] > 0)[0])
                if pred_legal == legal_set:
                    all_legal_correct += 1

    return {
        "top1_legal": top1_correct / top1_total if top1_total > 0 else 0,
        "top5_legal": top5_correct / top1_total if top1_total > 0 else 0,
        "perfect_position": all_legal_correct / top1_total if top1_total > 0 else 0,
        "n_samples": top1_total,
    }


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
    input_dim = N_MOVES

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device)
    n_patterns = ckpt.get('n_patterns', 960)

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

    # Use last chunk as eval
    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    eval_path = chunk_files[-1]

    print(f"Mode: {args.mode}, H={args.hidden}")
    print(f"Eval: {os.path.basename(eval_path)}")

    results = evaluate(
        model_even, model_odd, args.mode, patterns,
        pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
        eval_path, feature_cols, device)

    print(f"\nTop-1 legal:         {results['top1_legal']:.4%}")
    print(f"Top-5 legal:         {results['top5_legal']:.4%}")
    print(f"Perfect position:    {results['perfect_position']:.4%}")
    print(f"N samples:           {results['n_samples']}")
