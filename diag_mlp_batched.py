"""Minimal diagnostic: compare batched mlp scores vs single-position mlp_cell_scores.

Loads N positions from a chunk, runs each through both pipelines, prints
the top-cell mismatch (if any).

Imports only what's needed -- avoids the heavy precompute_pattern_arrays /
compute_pattern_labels_batch path that may OOM on small login nodes.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
# Only the lightweight imports:
from compare_v4_vs_mlp import load_mlp, mlp_cell_scores, C64_TO_C60, C60_TO_C64


def mlp_scores_batch(mlp_bundle, feats_120, positions, device):
    """Same batched scoring as train_aggregator_readout, inlined here.
    Parity routing flipped for chunk_ext convention: k = position + 1."""
    me, mo, idx, mask = mlp_bundle
    B = feats_120.shape[0]
    cell_scores = torch.zeros(B, 60, device=device)
    use_me_mask = (positions % 2 == 0)
    use_mo_mask = ~use_me_mask
    if use_me_mask.any():
        with torch.no_grad():
            logits = me(feats_120[use_me_mask])
        log1m = -F.softplus(logits)
        gathered = log1m[:, idx]
        gathered = gathered.masked_fill(~mask, 0.0)
        cell_scores[use_me_mask] = -gathered.sum(dim=-1)
    if use_mo_mask.any():
        with torch.no_grad():
            logits = mo(feats_120[use_mo_mask])
        log1m = -F.softplus(logits)
        gathered = log1m[:, idx]
        gathered = gathered.masked_fill(~mask, 0.0)
        cell_scores[use_mo_mask] = -gathered.sum(dim=-1)
    return cell_scores


def slice_played_even(features_180):
    return np.concatenate(
        [features_180[:, :60], features_180[:, 120:180]], axis=1
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--chunk', required=True)
    ap.add_argument('--num-positions', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=512)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    mlp = load_mlp(args.ckpt, args.hidden, device)

    with np.load(args.chunk) as z:
        N = z['positions'].shape[0]
        sample = np.random.RandomState(0).choice(
            N, size=args.num_positions, replace=False
        )
        sample.sort()
        feats_180 = np.asarray(z['features'][sample]).astype(np.float32)
        positions = np.asarray(z['positions'][sample]).astype(np.int64)
    print(f"Loaded {len(feats_180)} sample rows from chunk")

    feats_120 = slice_played_even(feats_180)

    x_dev = torch.from_numpy(feats_120).to(device)
    pos_dev = torch.from_numpy(positions).to(device)
    batched_scores = mlp_scores_batch(mlp, x_dev, pos_dev, device).cpu().numpy()
    print(f"Batched scores shape: {batched_scores.shape}")

    print(f"\n{'idx':>4}  {'pos':>4}  {'k':>3}  "
          f"{'batched argmax':>16}  {'single argmax':>15}  match")
    print("-" * 68)
    n_match = 0
    for i in range(args.num_positions):
        played = feats_180[i, :60]
        when = feats_180[i, 60:120]
        played_idx = np.where(played > 0.5)[0]
        steps = (when[played_idx] * 60 - 1).round().astype(int)
        order = np.argsort(steps)
        game_60 = played_idx[order]
        game_64 = [C60_TO_C64.get(int(c)) for c in game_60 if int(c) in C60_TO_C64]
        k = len(game_64)
        # mlp_cell_scores uses game[:k], so the game just needs to have >= k items
        single_scores = mlp_cell_scores(mlp, game_64 + [0]*5, k, device)
        argmax_batched = int(batched_scores[i].argmax())
        argmax_single = int(single_scores.argmax())
        match = argmax_batched == argmax_single
        if match:
            n_match += 1
        symbol = "OK" if match else "MISMATCH"
        print(f"  {i:>2}  {int(positions[i]):>4}  {k:>3}  "
              f"{argmax_batched:>16}  {argmax_single:>15}  {symbol}")

    print(f"\nMatch rate: {n_match}/{args.num_positions}")
    if n_match < args.num_positions:
        # Show detailed scores for first mismatch
        for i in range(args.num_positions):
            played = feats_180[i, :60]
            when = feats_180[i, 60:120]
            played_idx = np.where(played > 0.5)[0]
            steps = (when[played_idx] * 60 - 1).round().astype(int)
            order = np.argsort(steps)
            game_60 = played_idx[order]
            game_64 = [C60_TO_C64.get(int(c)) for c in game_60
                       if int(c) in C60_TO_C64]
            k = len(game_64)
            single_scores = mlp_cell_scores(mlp, game_64 + [0]*5, k, device)
            if int(batched_scores[i].argmax()) != int(single_scores.argmax()):
                print(f"\nDETAILS for mismatch at idx {i} (position={positions[i]}, k={k}):")
                print(f"  batched top-5: {np.argsort(-batched_scores[i])[:5].tolist()}")
                print(f"  single  top-5: {np.argsort(-single_scores)[:5].tolist()}")
                print(f"  batched score[0..5]: {batched_scores[i][:5].tolist()}")
                print(f"  single  score[0..5]: {single_scores[:5].tolist()}")
                print(f"  max abs diff (over all 60 cells): "
                      f"{np.abs(batched_scores[i] - single_scores).max():.4f}")
                break


if __name__ == '__main__':
    main()
