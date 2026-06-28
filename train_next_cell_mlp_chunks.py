"""Train next-cell-prediction MLP using precomputed fired_patterns_*.npz
chunks.  Much faster than streaming raw game pickles.

For each row in the chunk:
  - features: 180-d move history at state BEFORE turn t
  - fired: 960-d binary mask; patterns that target the actual played move
  - The next-move CELL is the target cell shared by all fired patterns
    (we derive it by indexing pattern_to_cell[fired_mask].mode())

Input transformation: 180-d → 3600-d move_grid via the standard convention.
Output: 60-way softmax over playable cells.

Usage:
    python train_next_cell_mlp_chunks.py \\
        --hidden 512 --epochs 3 [--max-chunks N] [--features move_grid]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX


# Pattern → target cell mapping (60-cell space)
_PATTERNS = enumerate_flanking_patterns()
PATTERN_TO_CELL60 = np.array(
    [MOVE_TO_IDX[p['target']] for p in _PATTERNS],
    dtype=np.int64,
)


class NextCellMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_cells=60):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_cells),
        )

    def forward(self, x):
        return self.net(x)


def to_move_grid_input(X_tensor):
    """3600-d (60 cells × 60 move-numbers).

    grid[cell, move_num] = -1 if cell played at even-indexed step (0,2,4...)
                            +1 if cell played at odd-indexed step (1,3,5...)
                             0  if not played.
    """
    B = X_tensor.shape[0]
    played = X_tensor[:, :60]
    when = X_tensor[:, 60:120]
    even = X_tensor[:, 120:180]
    move_num = torch.clamp((when * 60.0 - 1.0).round().long(), 0, 59)
    color_pm = (1.0 - 2.0 * even) * played
    grid = torch.zeros(B, 60, 60, device=X_tensor.device, dtype=X_tensor.dtype)
    grid.scatter_(2, move_num.unsqueeze(-1), color_pm.unsqueeze(-1))
    return grid.reshape(B, 60 * 60)


def load_chunk(path):
    """Returns (features_float16, cell_labels, is_forfeit, positions).

    Memory-efficient: features stored as float16 (cast to float32 per batch),
    fired-mask released immediately after deriving cell labels.
    """
    with np.load(path) as z:
        # float16 halves the feature memory (180-d × float16 × 18M ≈ 6.6 GB
        # vs 13.3 GB at float32). Conversion to float32 happens per-batch.
        feats = z['features'].astype(np.float16)
        # Derive cell labels here, while fired is still in scope.
        cell_labels = derive_next_cells_batch(z['fired'])
        # Drop fired -- 17.8 GB recovered.
        is_forfeit = z['is_forfeit'].astype(bool) if 'is_forfeit' in z.files else None
        positions = z['positions'].astype(np.int8) if 'positions' in z.files else None
    return feats, cell_labels, is_forfeit, positions


def derive_next_cell(fired_row):
    """Given a 960-d binary fired mask, return the cell index (0..59) shared
    by all fired patterns.  Returns -1 if no patterns fired."""
    where = np.where(fired_row)[0]
    if len(where) == 0:
        return -1
    cells = PATTERN_TO_CELL60[where]
    # All patterns fired at the same turn share the same target cell
    return int(cells[0])


def derive_next_cells_batch(fired):
    """Vectorized batch version: (N, 960) → (N,) int64 cell labels."""
    # For each row, find FIRST fired pattern's target
    counts = fired.sum(axis=1)
    has_fire = counts > 0
    out = np.full(len(fired), -1, dtype=np.int64)
    if has_fire.any():
        idx = fired[has_fire].argmax(axis=1)  # first nonzero column
        out[has_fire] = PATTERN_TO_CELL60[idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--features', default='move_grid',
                    choices=['move_grid', 'played+even', 'when+even', 'raw180'])
    ap.add_argument('--max-chunks', type=int, default=None,
                    help='Limit how many fired_patterns chunks to load')
    ap.add_argument('--eval-chunks', type=int, default=1,
                    help='Hold out this many chunks for eval')
    ap.add_argument('--chunk-dir',
                    default='experiments/mathematical_transformation_experiments/'
                            'heuristic_probe_results/feature_chunks')
    ap.add_argument('--chunk-glob', default='fired_patterns_*.npz')
    ap.add_argument('--output-dir',
                    default='experiments/mathematical_transformation_experiments/'
                            'heuristic_probe_results/pattern_detector_checkpoints')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chunk_files = sorted(glob.glob(os.path.join(args.chunk_dir, args.chunk_glob)))
    print(f"Found {len(chunk_files)} chunks matching {args.chunk_glob}")
    if args.max_chunks is not None:
        chunk_files = chunk_files[:args.max_chunks]
        print(f"Limiting to first {len(chunk_files)} chunks")
    eval_paths = chunk_files[-args.eval_chunks:] if args.eval_chunks > 0 else []
    train_paths = chunk_files[:len(chunk_files) - len(eval_paths)]
    print(f"  Train: {len(train_paths)} chunks  Eval: {len(eval_paths)} chunks")

    # Determine input_dim for the chosen feature
    feature_dim = {
        'move_grid': 3600,
        'played+even': 120,
        'when+even': 120,
        'raw180': 180,
    }[args.features]
    print(f"Feature: {args.features} (input_dim={feature_dim})")

    def slice_features(X_180):
        """Apply the selected feature transformation on a (B, 180) tensor."""
        if args.features == 'raw180':
            return X_180
        if args.features == 'played+even':
            return torch.cat([X_180[:, :60], X_180[:, 120:180]], dim=1)
        if args.features == 'when+even':
            return X_180[:, 60:180]
        if args.features == 'move_grid':
            return to_move_grid_input(X_180)
        raise ValueError(args.features)

    # Two parity-specific networks (matches train_pattern_simple convention):
    # model_even is trained on rows where position % 2 == 0.
    model_even = NextCellMLP(feature_dim, args.hidden, n_cells=60).to(device)
    model_odd = NextCellMLP(feature_dim, args.hidden, n_cells=60).to(device)
    opt_even = torch.optim.Adam(model_even.parameters(), lr=args.lr)
    opt_odd = torch.optim.Adam(model_odd.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model_even.parameters()) * 2
    print(f"Model params (even + odd combined): {n_params:,}")

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    seed_tag = "" if args.seed == 0 else f"_seed{args.seed}"
    feat_tag = args.features.replace('+', '')
    save_path = os.path.join(save_dir,
        f"next_cell_mlp_H{args.hidden}_{feat_tag}_parity{seed_tag}.pt")

    for epoch in range(1, args.epochs + 1):
        model_even.train()
        model_odd.train()
        t0 = time.time()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        last_log = t0
        # Shuffle chunk order across epochs
        rng = np.random.RandomState(epoch * 1000 + args.seed)
        chunk_order = rng.permutation(len(train_paths))

        for ci_idx, ci in enumerate(chunk_order):
            feats180, cell_labels, is_forfeit, positions = load_chunk(train_paths[ci])
            keep = cell_labels >= 0
            if is_forfeit is not None:
                keep &= ~is_forfeit
            if positions is None:
                raise RuntimeError("Chunk missing 'positions' field — required for parity split")
            valid_idx = np.where(keep)[0]
            if len(valid_idx) == 0:
                continue
            rng.shuffle(valid_idx)

            for i in range(0, len(valid_idx), args.batch_size):
                batch_idx = valid_idx[i:i + args.batch_size]
                x180 = torch.from_numpy(feats180[batch_idx].astype(np.float32)).to(device)
                y = torch.from_numpy(cell_labels[batch_idx]).to(device)
                pos_batch = positions[batch_idx]
                x = slice_features(x180)
                # Split by parity and train each model on its half
                even_mask = torch.from_numpy(pos_batch % 2 == 0).to(device)
                odd_mask = ~even_mask
                if even_mask.any():
                    logits_e = model_even(x[even_mask])
                    loss_e = F.cross_entropy(logits_e, y[even_mask])
                    opt_even.zero_grad()
                    loss_e.backward()
                    opt_even.step()
                    total_loss += loss_e.item() * int(even_mask.sum())
                    total_correct += (logits_e.argmax(dim=-1) == y[even_mask]).sum().item()
                    total_n += int(even_mask.sum())
                if odd_mask.any():
                    logits_o = model_odd(x[odd_mask])
                    loss_o = F.cross_entropy(logits_o, y[odd_mask])
                    opt_odd.zero_grad()
                    loss_o.backward()
                    opt_odd.step()
                    total_loss += loss_o.item() * int(odd_mask.sum())
                    total_correct += (logits_o.argmax(dim=-1) == y[odd_mask]).sum().item()
                    total_n += int(odd_mask.sum())
            del feats180, cell_labels, is_forfeit, positions, valid_idx
            now = time.time()
            if now - last_log > 60:
                print(f"  epoch {epoch} chunk {ci_idx+1}/{len(train_paths)}: "
                      f"n={total_n:,} loss={total_loss/total_n:.4f} "
                      f"acc={total_correct/total_n:.4f} "
                      f"elapsed={int(now-t0)}s", flush=True)
                last_log = now

        # Eval at end of epoch (split by parity)
        model_even.eval()
        model_odd.eval()
        ev_loss = 0.0
        ev_correct = 0
        ev_n = 0
        with torch.no_grad():
            for ep in eval_paths:
                feats180, cell_labels, is_forfeit, positions = load_chunk(ep)
                keep = cell_labels >= 0
                if is_forfeit is not None:
                    keep &= ~is_forfeit
                valid_idx = np.where(keep)[0]
                for i in range(0, len(valid_idx), args.batch_size):
                    batch_idx = valid_idx[i:i + args.batch_size]
                    x180 = torch.from_numpy(feats180[batch_idx].astype(np.float32)).to(device)
                    y = torch.from_numpy(cell_labels[batch_idx]).to(device)
                    pos_batch = positions[batch_idx]
                    x = slice_features(x180)
                    even_mask = torch.from_numpy(pos_batch % 2 == 0).to(device)
                    odd_mask = ~even_mask
                    if even_mask.any():
                        logits_e = model_even(x[even_mask])
                        loss_e = F.cross_entropy(logits_e, y[even_mask], reduction='sum')
                        ev_loss += loss_e.item()
                        ev_correct += (logits_e.argmax(dim=-1) == y[even_mask]).sum().item()
                        ev_n += int(even_mask.sum())
                    if odd_mask.any():
                        logits_o = model_odd(x[odd_mask])
                        loss_o = F.cross_entropy(logits_o, y[odd_mask], reduction='sum')
                        ev_loss += loss_o.item()
                        ev_correct += (logits_o.argmax(dim=-1) == y[odd_mask]).sum().item()
                        ev_n += int(odd_mask.sum())
                del feats180, cell_labels, is_forfeit, positions, valid_idx

        train_acc = total_correct / max(1, total_n)
        ev_acc = ev_correct / max(1, ev_n)
        print(f"Epoch {epoch}: train n={total_n:,} loss={total_loss/total_n:.4f} "
              f"acc={train_acc:.4f}  |  eval n={ev_n:,} "
              f"loss={ev_loss/max(1,ev_n):.4f} acc={ev_acc:.4f}  "
              f"({int(time.time()-t0)}s)", flush=True)

        torch.save({
            'even': {k: v.cpu().clone() for k, v in model_even.state_dict().items()},
            'odd':  {k: v.cpu().clone() for k, v in model_odd.state_dict().items()},
            'epoch': epoch,
            'hidden': args.hidden,
            'input_dim': feature_dim,
            'n_cells': 60,
            'features': args.features,
            'best_train_acc': train_acc,
            'best_eval_acc': ev_acc,
        }, save_path)
        print(f"  Saved {save_path}", flush=True)


if __name__ == '__main__':
    main()
