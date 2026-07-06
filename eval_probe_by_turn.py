"""Evaluate a saved MLP+probe pair on chunk_ext_0039 and break accuracy
down by turn number.

Compares turns 5-53 (old eval range) vs. turns 54-58 (extended range) to
see how much the extended-range chunks depress probe accuracy.

Usage:
    python eval_probe_by_turn.py \\
        --mlp-ckpt experiments/.../pattern_simple_direct_H1024_playedeven.pt \\
        --probe-ckpt experiments/.../probe_direct_H1024_playedeven.pt \\
        --hidden 1024
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from train_pattern_simple import DirectMLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mlp-ckpt', required=True)
    ap.add_argument('--probe-ckpt', required=True)
    ap.add_argument('--hidden', type=int, required=True)
    ap.add_argument('--chunk',
        default='experiments/mathematical_transformation_experiments/'
                'heuristic_probe_results/feature_chunks/chunk_ext_0039.npz')
    ap.add_argument('--n-eval', type=int, default=100_000,
                    help='Positions to eval (stride-sampled from the chunk).')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load MLP
    mlp_ckpt = torch.load(args.mlp_ckpt, map_location=device)
    input_dim = mlp_ckpt.get('input_dim', 120)
    me = DirectMLP(input_dim, args.hidden).to(device)
    mo = DirectMLP(input_dim, args.hidden).to(device)
    me.load_state_dict(mlp_ckpt['even'])
    mo.load_state_dict(mlp_ckpt['odd'])
    me.eval(); mo.eval()
    print(f"MLP: H={args.hidden}, input_dim={input_dim}")

    # Load probe
    probe_ckpt = torch.load(args.probe_ckpt, map_location=device)
    pe = nn.Linear(args.hidden, 192).to(device)
    po = nn.Linear(args.hidden, 192).to(device)
    pe.load_state_dict(probe_ckpt['even'])
    po.load_state_dict(probe_ckpt['odd'])
    pe.eval(); po.eval()
    print(f"Probe: best_acc={probe_ckpt.get('best_acc'):.4f}")

    # Load chunk
    print(f"Loading {args.chunk}...")
    with np.load(args.chunk) as z:
        feats_all = z['features'].astype(np.float32)   # (n, 180) or (n, 120)
        labels_all = z['labels'].astype(np.int8)       # (n, 64) board state
        positions_all = z['positions'].astype(np.int64)  # (n,) turn number
    n_total = len(feats_all)
    print(f"  {n_total:,} rows, turn range [{positions_all.min()}, {positions_all.max()}]")

    # Slice features to played+even (120-d compact: use as-is; 180-d: slice)
    if feats_all.shape[1] == 180:
        played_even_cols = list(range(0, 60)) + list(range(120, 180))
        feats_all = feats_all[:, played_even_cols]
    print(f"  feature dim after slicing: {feats_all.shape[1]}")

    # Stride sample
    stride = max(1, n_total // args.n_eval)
    idx = np.arange(0, n_total, stride)[:args.n_eval]
    print(f"  sampling {len(idx):,} positions (stride={stride})")

    # Convert labels: chunk stores 0=empty, 1=black, 2=white (Li encoding)
    # But probe was trained with same encoding. Just use directly.
    correct_per_turn = {}
    total_per_turn = {}
    correct_all = 0
    total_all = 0

    batch_size = 4096
    with torch.no_grad():
        for bstart in range(0, len(idx), batch_size):
            bend = min(bstart + batch_size, len(idx))
            batch_idx = idx[bstart:bend]
            x = torch.from_numpy(feats_all[batch_idx]).to(device)
            pos = positions_all[batch_idx]
            y = labels_all[batch_idx].astype(np.int64)  # (B, 64)

            # Parity routing (matches training convention)
            even_mask = (pos % 2 == 0)
            odd_mask = ~even_mask

            preds = np.zeros_like(y)
            if even_mask.any():
                h = F.relu(me.net[0](x[even_mask]))
                logits = pe(h).view(-1, 64, 3)
                preds[even_mask] = logits.argmax(-1).cpu().numpy()
            if odd_mask.any():
                h = F.relu(mo.net[0](x[odd_mask]))
                logits = po(h).view(-1, 64, 3)
                preds[odd_mask] = logits.argmax(-1).cpu().numpy()

            correct_matrix = (preds == y)                        # (B, 64)
            for i, turn in enumerate(pos):
                turn = int(turn)
                c = int(correct_matrix[i].sum())
                correct_per_turn[turn] = correct_per_turn.get(turn, 0) + c
                total_per_turn[turn] = total_per_turn.get(turn, 0) + 64
            correct_all += correct_matrix.sum()
            total_all += correct_matrix.size

    print()
    print(f"Overall acc (all turns): {correct_all/total_all:.4f}")
    print()
    print(f"{'turn':>4}  {'acc':>7}  {'count':>7}")
    print("-" * 25)
    for turn in sorted(correct_per_turn.keys()):
        acc = correct_per_turn[turn] / total_per_turn[turn]
        print(f"{turn:>4}  {acc:>7.4f}  {total_per_turn[turn]//64:>6}")

    # Summary buckets
    def bucket_acc(lo, hi):
        c = sum(correct_per_turn.get(t, 0) for t in range(lo, hi + 1))
        t = sum(total_per_turn.get(t, 0) for t in range(lo, hi + 1))
        return c / t if t > 0 else 0.0

    print()
    print(f"Turns 5-53 (old range):  acc = {bucket_acc(5, 53):.4f}")
    print(f"Turns 54-58 (extended):  acc = {bucket_acc(54, 58):.4f}")
    print(f"Turns 5-58 (all):        acc = {bucket_acc(5, 58):.4f}")


if __name__ == '__main__':
    main()
