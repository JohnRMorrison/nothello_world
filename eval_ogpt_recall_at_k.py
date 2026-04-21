"""Compute recall@K (K = num legal moves) for OthelloGPT.

For each chunk position, reconstruct the token sequence up to that point,
run OthelloGPT's forward pass, take the logits at the last position over
the 60 move tokens, rank them, and compare to the ground-truth legal set.

Usage:
    python eval_ogpt_recall_at_k.py --ogpt-ckpt ckpts/gpt_nanda_synthetic.ckpt
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
    compute_pattern_labels_batch, pat_labels_to_cell_labels, _get_cell_pat_index,
)
from probe_from_ogpt_random import features_to_tokens

MAX_MOVES = 59


def load_ogpt_model(ckpt_path, device):
    """Load the mingpt-format OthelloGPT checkpoint."""
    from mingpt.model import GPT, GPTConfig
    sd = torch.load(ckpt_path, map_location=device)
    config = GPTConfig(
        vocab_size=sd['tok_emb.weight'].shape[0],
        block_size=sd['pos_emb'].shape[1],
        n_layer=sum(1 for k in sd if k.startswith('blocks.') and k.endswith('.ln1.weight')),
        n_head=8, n_embd=sd['tok_emb.weight'].shape[1],
    )
    model = GPT(config).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ogpt-ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    model = load_ogpt_model(args.ogpt_ckpt, device)
    print(f"Loaded OGPT from {args.ogpt_ckpt}")

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
    n = min(len(X), 49 * 10000)
    rng = np.random.RandomState(0)
    si = np.sort(rng.choice(len(X), n, replace=False))
    X, Y, pos = X[si], Y[si], pos[si]

    played_cols = list(range(0, N_MOVES))
    when_cols = list(range(N_MOVES, 2 * N_MOVES))

    per_k = {}
    total_frac = 0.0
    total_n = 0
    top1_correct = 0

    with torch.no_grad():
        for i in range(0, n, 512):
            yb = Y[i:i + 512]; p = pos[i:i + 512]
            played = X[i:i + 512, played_cols].numpy().astype(np.float32)
            when = X[i:i + 512, when_cols].numpy().astype(np.float32)
            tokens_np, n_moves = features_to_tokens(played, when, max_n=MAX_MOVES)

            # Run OGPT on the reconstructed sequences
            tokens = torch.from_numpy(tokens_np).to(device)
            # We need to forward up to each position's n_moves, but mingpt
            # expects a fixed-length input. Feed the full padded sequence and
            # pick the logits at index (n_moves-1).
            logits, _ = model(tokens)  # (B, max_moves, vocab)
            # Gather logits at position (n_moves-1) per sample
            last_idx = torch.from_numpy(n_moves - 1).clamp(min=0).to(device)
            gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
            last_logits = logits.gather(1, gather_idx).squeeze(1)  # (B, vocab)
            # Skip token 0 (padding) → 60 move scores
            move_scores = last_logits[:, 1:1 + 60]  # (B, 60)

            # Ground-truth legal
            gp = torch.from_numpy(compute_pattern_labels_batch(
                yb.numpy(), p.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)).to(device)
            legal = pat_labels_to_cell_labels(gp, pattern_to_cell)
            gl = (legal > 0.5).cpu().numpy()
            scores = move_scores.cpu().numpy()

            for b in range(scores.shape[0]):
                legal_set = set(np.where(gl[b])[0].tolist())
                K = len(legal_set)
                if K == 0: continue
                ranked = np.argsort(-scores[b])
                top1_correct += int(ranked[0] in legal_set)
                top_k = set(ranked[:K].tolist())
                frac = len(top_k & legal_set) / K
                total_frac += frac; total_n += 1
                if K not in per_k: per_k[K] = [0.0, 0]
                per_k[K][0] += frac; per_k[K][1] += 1

    print(f"OGPT (synthetic) on chunk_0039 random sample ({total_n} positions)")
    print(f"  top-1 legal: {top1_correct/total_n:.4%}")
    print(f"  recall@K (K = num legal):   {total_frac/total_n:.4%}")
    print()
    print(f"{'K':>4}  {'positions':>10}  {'recall@K':>10}")
    for K in sorted(per_k.keys()):
        s, ct = per_k[K]
        print(f"{K:>4}  {ct:>10d}  {s/ct:>10.4%}")
