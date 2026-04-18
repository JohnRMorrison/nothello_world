"""Evaluate legal move prediction accuracy from pattern detector models.

Loads a checkpoint, runs pattern predictions on eval data, aggregates
960 patterns → 60 legal moves, reports precision/recall/F1/perfect-position.

Usage:
    python eval_legal_moves.py --ckpt pattern_simple_direct_H512.pt --mode direct --hidden 512
    python eval_legal_moves.py --ckpt pattern_simple_randproj_s0_H1024.pt --mode direct --hidden 1024
"""
import sys, os
sys.path.insert(0, '.')

import argparse
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


def patterns_to_legal(pat_probs, patterns, threshold=0.5):
    """Aggregate 960 pattern probs → 60-d legal move predictions."""
    n = len(pat_probs)
    legal = np.zeros((n, 60), dtype=np.float32)
    for j, pat in enumerate(patterns):
        move_idx = MOVE_TO_IDX[pat['target']]
        legal[:, move_idx] = np.maximum(legal[:, move_idx], pat_probs[:, j])
    return legal


def evaluate(model_even, model_odd, mode, patterns,
             pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
             chunk_path, feature_cols, device, batch_size=1024):
    """Evaluate legal move accuracy on one chunk."""
    X, Y, pos = _load_features(chunk_path)
    if feature_cols is not None:
        X = X[:, feature_cols]

    n = min(len(X), 49 * 10000)
    X, Y, pos = X[:n], Y[:n], pos[:n]

    all_pred_pat = []
    all_gt_pat = []

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

            all_pred_pat.append(torch.sigmoid(pat_logits).cpu().numpy())

            # Ground truth patterns
            gt = compute_pattern_labels_batch(
                y_board.numpy(), p.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
            all_gt_pat.append(gt)

    pred_pat = np.concatenate(all_pred_pat)
    gt_pat = np.concatenate(all_gt_pat)

    # Pattern-level accuracy
    pat_pred_binary = (pred_pat > 0.5).astype(np.float32)
    pat_acc = (pat_pred_binary == gt_pat).mean()

    # Aggregate to legal moves
    pred_legal = patterns_to_legal(pred_pat, patterns)
    gt_legal = patterns_to_legal(gt_pat, patterns)

    pred_binary = (pred_legal > 0.5).astype(np.float32)
    gt_binary = (gt_legal > 0.5).astype(np.float32)

    tp = ((pred_binary == 1) & (gt_binary == 1)).sum()
    fp = ((pred_binary == 1) & (gt_binary == 0)).sum()
    fn = ((pred_binary == 0) & (gt_binary == 1)).sum()
    tn = ((pred_binary == 0) & (gt_binary == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    perfect = (pred_binary == gt_binary).all(axis=1).mean()

    return {
        "pat_acc": pat_acc,
        "legal_acc": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "perfect_position": perfect,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_samples": n,
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

    print(f"\nPattern accuracy:     {results['pat_acc']:.4%}")
    print(f"Legal move accuracy:  {results['legal_acc']:.4%}")
    print(f"Precision:            {results['precision']:.4f}")
    print(f"Recall:               {results['recall']:.4f}")
    print(f"F1:                   {results['f1']:.4f}")
    print(f"Perfect position:     {results['perfect_position']:.4%}")
    print(f"TP={results['tp']}  FP={results['fp']}  FN={results['fn']}  TN={results['tn']}")
