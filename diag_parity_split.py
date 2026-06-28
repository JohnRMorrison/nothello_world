"""Diagnostic: evaluate each parity network of next_cell_mlp_*_parity.pt
separately on its own parity's rows.  Helps detect whether one of the two
networks failed to train.

Usage:
  python diag_parity_split.py \\
    --ckpt experiments/.../next_cell_mlp_H512_move_grid_parity.pt \\
    --chunk experiments/.../feature_chunks/fired_patterns_0039.npz
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from train_next_cell_mlp_chunks import (
    NextCellMLP, to_move_grid_input, derive_next_cells_batch,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--chunk', required=True)
    ap.add_argument('--max-rows', type=int, default=200000)
    ap.add_argument('--batch-size', type=int, default=4096)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device)
    hidden = ckpt.get('hidden', 512)
    input_dim = ckpt.get('input_dim', 3600)
    print(f"Ckpt: H={hidden}, input={input_dim}")
    print(f"     epoch={ckpt.get('epoch')}, "
          f"best_train_acc={ckpt.get('best_train_acc'):.4f}, "
          f"best_eval_acc={ckpt.get('best_eval_acc'):.4f}")

    model_even = NextCellMLP(input_dim, hidden, 60).to(device)
    model_odd  = NextCellMLP(input_dim, hidden, 60).to(device)
    model_even.load_state_dict(ckpt['even']); model_even.eval()
    model_odd.load_state_dict(ckpt['odd']);  model_odd.eval()

    print(f"Loading chunk {os.path.basename(args.chunk)} (memory-efficient)")
    # First load only the SMALL fields to compute the valid mask without
    # materializing features or fired arrays.
    with np.load(args.chunk) as z:
        is_forfeit = (z['is_forfeit'].astype(bool)
                      if 'is_forfeit' in z.files else None)
        positions = z['positions'].astype(np.int64)
        N = len(positions)
        # Sample max_rows random indices first, derive labels only for those.
        rng = np.random.RandomState(0)
        sample_idx = rng.choice(N, size=min(N, args.max_rows * 5), replace=False)
        sample_idx.sort()  # for slice-friendly access into npz
        # Lazy access by index list — numpy reads only the requested rows.
        sample_fired = z['fired'][sample_idx]
        sample_feats = z['features'][sample_idx].astype(np.float16)
    sample_cell = derive_next_cells_batch(sample_fired)
    sample_pos = positions[sample_idx]
    sample_forfeit = (is_forfeit[sample_idx] if is_forfeit is not None
                      else np.zeros(len(sample_idx), dtype=bool))

    # Filter sampled rows
    keep = (sample_cell >= 0) & (~sample_forfeit)
    feats180 = sample_feats[keep]
    cell_labels = sample_cell[keep]
    positions_filtered = sample_pos[keep]
    valid_idx = np.arange(len(feats180))
    # Trim to requested size
    if len(valid_idx) > args.max_rows:
        valid_idx = valid_idx[:args.max_rows]
        feats180 = feats180[valid_idx]
        cell_labels = cell_labels[valid_idx]
        positions_filtered = positions_filtered[valid_idx]
        valid_idx = np.arange(len(feats180))
    positions = positions_filtered
    print(f"Eval rows (random sample): {len(valid_idx):,}")

    # Split by parity
    pos_sub = positions[valid_idx]
    even_idx = valid_idx[pos_sub % 2 == 0]
    odd_idx  = valid_idx[pos_sub % 2 == 1]
    print(f"  even rows: {len(even_idx):,}  odd rows: {len(odd_idx):,}")

    def eval_on(idx, model, name):
        n = 0; correct = 0; loss_sum = 0.0
        with torch.no_grad():
            for i in range(0, len(idx), args.batch_size):
                b = idx[i:i + args.batch_size]
                x180 = torch.from_numpy(feats180[b].astype(np.float32)).to(device)
                y = torch.from_numpy(cell_labels[b]).to(device)
                x = to_move_grid_input(x180)
                logits = model(x)
                loss = F.cross_entropy(logits, y, reduction='sum')
                loss_sum += loss.item()
                correct += (logits.argmax(dim=-1) == y).sum().item()
                n += len(b)
        acc = correct / max(1, n)
        avg_loss = loss_sum / max(1, n)
        print(f"  [{name}] n={n:,}  acc={acc:.4f}  loss={avg_loss:.4f}")
        return acc

    print()
    print("=== Each network on its OWN parity ===")
    eval_on(even_idx, model_even, "model_even on EVEN rows")
    eval_on(odd_idx, model_odd,  "model_odd  on ODD  rows")

    print()
    print("=== Each network on the OTHER parity (sanity check) ===")
    eval_on(odd_idx, model_even, "model_even on ODD  rows")
    eval_on(even_idx, model_odd, "model_odd  on EVEN rows")

    # Weight norms — if a network failed, weights might be tiny or huge
    print()
    print("=== Weight norms ===")
    for net_name, net in [("model_even", model_even), ("model_odd", model_odd)]:
        for pname, p in net.named_parameters():
            print(f"  {net_name}.{pname:>20}: mean_abs={p.abs().mean().item():.4f}  "
                  f"std={p.std().item():.4f}  shape={tuple(p.shape)}")


if __name__ == '__main__':
    main()
