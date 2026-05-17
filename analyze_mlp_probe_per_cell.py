"""Per-cell accuracy of an MLP probe on an arbitrary turn range.

Mirrors analyze_nanda_probe_per_cell.py but for an MLP pattern detector +
parity-split probe. Reports per-cell 8x8 grid + per-region + worst cells
over the requested position range.

Usage:
    python analyze_mlp_probe_per_cell.py \\
        --pat-ckpt   <pattern_detector>.pt \\
        --probe-ckpt <probe>.pt \\
        --hidden 4096 --features move_grid \\
        --pos-start 5 --pos-end 54     # overall
    python analyze_mlp_probe_per_cell.py ... --pos-start 53 --pos-end 54  # final-move only
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn

from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, OPTIONS, N_MOVES,
)
from train_pattern_simple import (
    DirectMLP, to_move_grid_input, to_move_grid_onehot_input,
)


CENTER_RC = {(3, 3), (3, 4), (4, 3), (4, 4)}
CORNER_RC = {(0, 0), (0, 7), (7, 0), (7, 7)}


def feature_slice(X, features):
    if features == "when":         return X[:, N_MOVES:2 * N_MOVES]
    if features == "when+even":    return X[:, N_MOVES:3 * N_MOVES]
    if features == "all":          return X
    if features == "move_grid":    return to_move_grid_input(X)
    if features == "move_grid_onehot": return to_move_grid_onehot_input(X)
    raise ValueError(f"Unknown features: {features}")


def hidden_of(model, x):
    return torch.relu(model.net[0](x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pat-ckpt", required=True)
    p.add_argument("--probe-ckpt", required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--features", default="move_grid",
                   choices=["when", "when+even", "all", "move_grid",
                            "move_grid_onehot"])
    p.add_argument("--chunk-path",
        default="experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks/chunk_0039.npz")
    p.add_argument("--extra-chunks", nargs="*", default=[],
                   help="Additional chunk paths (e.g. late_turns_eval.npz for turns 54-58).")
    p.add_argument("--pos-start", type=int, default=5)
    p.add_argument("--pos-end", type=int, default=54)
    p.add_argument("--batch-size", type=int, default=1024)
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.pat_ckpt, map_location=device)
    input_dim = ckpt.get('input_dim',
        {"when": 60, "when+even": 120, "all": 180,
         "move_grid": 3600, "move_grid_onehot": 10800}[args.features])
    model_e = DirectMLP(input_dim, args.hidden,
                        ckpt.get('n_patterns', 960)).to(device)
    model_o = DirectMLP(input_dim, args.hidden,
                        ckpt.get('n_patterns', 960)).to(device)
    model_e.load_state_dict(ckpt['even'])
    model_o.load_state_dict(ckpt['odd'])
    model_e.eval(); model_o.eval()
    for q in list(model_e.parameters()) + list(model_o.parameters()):
        q.requires_grad = False

    probe_ckpt = torch.load(args.probe_ckpt, map_location=device)
    probe_e = nn.Linear(args.hidden, 64 * OPTIONS).to(device)
    probe_o = nn.Linear(args.hidden, 64 * OPTIONS).to(device)
    probe_e.load_state_dict(probe_ckpt['even'])
    probe_o.load_state_dict(probe_ckpt['odd'])
    probe_e.eval(); probe_o.eval()
    print(f"  Loaded pat (pat_acc={ckpt.get('best_pat_acc', '?')}) and probe")

    chunks = [args.chunk_path] + list(args.extra_chunks)
    Xs, Ys, Ps = [], [], []
    for cp in chunks:
        x, y, pos = _load_features(cp)
        Xs.append(x); Ys.append(y); Ps.append(pos)
    import torch as _torch
    ev_X  = _torch.cat(Xs, dim=0)
    ev_Y  = _torch.cat(Ys, dim=0)
    ev_pos = _torch.cat(Ps, dim=0)
    mask = (ev_pos >= args.pos_start) & (ev_pos < args.pos_end)
    ev_X = ev_X[mask]; ev_Y = ev_Y[mask]; ev_pos = ev_pos[mask]
    print(f"Positions in range [{args.pos_start}, {args.pos_end}): {len(ev_X)}")
    if len(ev_X) == 0:
        print("No positions in this range — try a different chunk or range.")
        return

    Xf = feature_slice(ev_X, args.features)

    correct_per_cell = np.zeros(64, dtype=np.int64)
    total = 0
    with torch.no_grad():
        for i in range(0, len(Xf), args.batch_size):
            x = Xf[i:i + args.batch_size].to(device)
            y = ev_Y[i:i + args.batch_size].to(device)
            pos = ev_pos[i:i + args.batch_size]
            em = (pos % 2 == 0); om = ~em
            preds = torch.zeros_like(y)
            if em.any():
                h = hidden_of(model_e, x[em])
                preds[em] = probe_e(h).view(-1, 64, OPTIONS).argmax(-1)
            if om.any():
                h = hidden_of(model_o, x[om])
                preds[om] = probe_o(h).view(-1, 64, OPTIONS).argmax(-1)
            correct_per_cell += (preds == y).sum(dim=0).cpu().numpy()
            total += len(y)
    acc = correct_per_cell / max(total, 1)

    # 8x8 grid
    print(f"\nPer-cell accuracy (turns [{args.pos_start}, {args.pos_end})):")
    print("       " + "   ".join(f"{c}" for c in "abcdefgh"))
    for r in range(8):
        row = []
        for c in range(8):
            v = acc[r * 8 + c]
            tag = "*" if (r, c) in CENTER_RC else " "
            row.append(f"{v:.3f}{tag}")
        print(f"  {r+1}  " + " ".join(row))

    grid = acc.reshape(8, 8)
    print(f"\nMean: {grid.mean():.4f}  Std: {grid.std():.4f}  "
          f"Min: {grid.min():.4f}  Max: {grid.max():.4f}")

    # Per-region
    edge_rc = [(r, c) for r in range(8) for c in range(8)
               if (r in (0, 7) or c in (0, 7)) and (r, c) not in CORNER_RC]
    inner_rc = [(r, c) for r in range(8) for c in range(8)
                if r not in (0, 7) and c not in (0, 7)
                and (r, c) not in CENTER_RC]
    print(f"\nBy region:")
    for label, rc_list in [("corner", list(CORNER_RC)),
                           ("edge",   edge_rc),
                           ("inner",  inner_rc),
                           ("center", list(CENTER_RC))]:
        vals = [grid[r, c] for r, c in rc_list]
        print(f"  {label:>6s} n={len(vals):>2d}  "
              f"mean={np.mean(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    # Worst 8
    flat = np.argsort(acc)
    print(f"\nWorst 8 cells:")
    for idx in flat[:8]:
        r, c = divmod(int(idx), 8)
        name = f"{'abcdefgh'[c]}{r+1}"
        if (r, c) in CENTER_RC: cls = "center"
        elif (r, c) in CORNER_RC: cls = "corner"
        elif r in (0, 7) or c in (0, 7): cls = "edge"
        else: cls = "inner"
        print(f"  {name} ({cls:>6s})  acc={acc[r * 8 + c]:.4f}")


if __name__ == "__main__":
    main()
