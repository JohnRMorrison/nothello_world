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

    print(f"Loading chunk {os.path.basename(args.chunk)}")
    with np.load(args.chunk) as z:
        feats180 = z['features'].astype(np.float16)
        cell_labels = derive_next_cells_batch(z['fired'])
        is_forfeit = z['is_forfeit'].astype(bool) if 'is_forfeit' in z.files else None
        positions = z['positions'].astype(np.int64)

    keep = cell_labels >= 0
    if is_forfeit is not None:
        keep &= ~is_forfeit
    valid_idx = np.where(keep)[0]
    np.random.RandomState(0).shuffle(valid_idx)
    valid_idx = valid_idx[:args.max_rows]
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
