"""Center-cell decoding accuracy vs game turn, for a saved MLP probe.

Loads:
  - A pattern detector (e.g. pattern_simple_direct_H4096_move_grid.pt)
  - Its companion probe (e.g. probe_direct_H4096_move_grid.pt) with
    parity-split 'even' / 'odd' Linear(H, 64*OPTIONS) heads.

For each turn t in [5, 53], forwards positions at that turn through the
frozen pattern detector to get hidden state, applies the parity-correct
probe, and measures accuracy on the 4 center cells (27, 28, 35, 36).
Saves a PNG.

Usage:
    python plot_mlp_center_by_turn.py \\
        --pat-ckpt experiments/.../pattern_simple_direct_H4096_move_grid.pt \\
        --probe-ckpt experiments/.../probe_direct_H4096_move_grid.pt \\
        --hidden 4096 --features move_grid
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, OPTIONS, N_MOVES,
)
from train_pattern_simple import (
    DirectMLP, to_move_grid_input, to_move_grid_onehot_input,
)


CENTER_64 = [27, 28, 35, 36]


def feature_slice(X, features):
    """Slice/expand the 180-d chunk features to the requested view."""
    if features == "when":         return X[:, N_MOVES:2 * N_MOVES]
    if features == "when+even":    return X[:, N_MOVES:3 * N_MOVES]
    if features == "all":          return X
    if features == "move_grid":    return to_move_grid_input(X)
    if features == "move_grid_onehot":
                                   return to_move_grid_onehot_input(X)
    raise ValueError(f"Unknown features: {features}")


def hidden_of(model, x):
    return torch.relu(model.net[0](x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pat-ckpt", required=True)
    parser.add_argument("--probe-ckpt", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--features", default="move_grid",
                        choices=["when", "when+even", "all", "move_grid",
                                 "move_grid_onehot"])
    parser.add_argument("--chunk-path",
        default="experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks/chunk_0039.npz")
    parser.add_argument("--extra-chunks", nargs="*", default=[],
                        help="Additional chunk paths to load (e.g. late_turns_eval.npz "
                             "for turns 54-58).")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output",
                        default="experiments/plots/mlp_center_by_turn.png")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Pattern detector
    ckpt = torch.load(args.pat_ckpt, map_location=device)
    input_dim = ckpt.get('input_dim',
        {"when": 60, "when+even": 120, "all": 180,
         "move_grid": 3600, "move_grid_onehot": 10800}[args.features])
    model_even = DirectMLP(input_dim, args.hidden, ckpt.get('n_patterns', 960)).to(device)
    model_odd  = DirectMLP(input_dim, args.hidden, ckpt.get('n_patterns', 960)).to(device)
    model_even.load_state_dict(ckpt['even'])
    model_odd.load_state_dict(ckpt['odd'])
    model_even.eval(); model_odd.eval()
    for p in list(model_even.parameters()) + list(model_odd.parameters()):
        p.requires_grad = False
    print(f"Loaded pattern detector (input_dim={input_dim}, H={args.hidden})")

    # Probe (parity-split)
    probe_ckpt = torch.load(args.probe_ckpt, map_location=device)
    probe_even = nn.Linear(args.hidden, 64 * OPTIONS).to(device)
    probe_odd  = nn.Linear(args.hidden, 64 * OPTIONS).to(device)
    probe_even.load_state_dict(probe_ckpt['even'])
    probe_odd.load_state_dict(probe_ckpt['odd'])
    probe_even.eval(); probe_odd.eval()
    print(f"Loaded probe (best_acc reported: {probe_ckpt.get('best_acc', '?')})")

    # Load eval chunks (primary + optional extras for late-game turns)
    chunks = [args.chunk_path] + list(args.extra_chunks)
    Xs, Ys, Ps = [], [], []
    for cp in chunks:
        x, y, p = _load_features(cp)
        Xs.append(x); Ys.append(y); Ps.append(p)
        print(f"Loaded {os.path.basename(cp)}: {len(x)} positions, "
              f"turns {sorted(np.unique(p.numpy()))[:3]}...{sorted(np.unique(p.numpy()))[-3:]}")
    ev_X = torch.cat(Xs, dim=0)
    ev_Y = torch.cat(Ys, dim=0)
    ev_pos = torch.cat(Ps, dim=0)
    print(f"Total: {len(ev_X)} positions, dim={ev_X.shape[1]}")

    # For each turn, evaluate
    turns = sorted(np.unique(ev_pos.numpy()).tolist())
    print(f"Turns in chunk: {turns[:3]}...{turns[-3:]}  (n={len(turns)})")

    per_turn_center = np.zeros(len(turns), dtype=np.float64)
    for ti, t in enumerate(turns):
        mask = (ev_pos == t)
        if not mask.any():
            continue
        X = ev_X[mask]
        Y = ev_Y[mask]
        Xf = feature_slice(X, args.features)
        # Parity
        pm = (t % 2 == 0)   # if true, use 'even' head; else 'odd'
        model = model_even if pm else model_odd
        probe = probe_even if pm else probe_odd
        # Forward in batches
        preds_all = []
        with torch.no_grad():
            for i in range(0, len(Xf), args.batch_size):
                xb = Xf[i:i + args.batch_size].to(device)
                h = hidden_of(model, xb)
                logits = probe(h).view(-1, 64, OPTIONS)
                preds_all.append(logits.argmax(-1).cpu())
        preds = torch.cat(preds_all, dim=0).numpy()    # (N_t, 64)
        center_acc = np.mean([
            (preds[:, c] == Y[:, c].numpy()).mean()
            for c in CENTER_64
        ])
        per_turn_center[ti] = center_acc

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(turns, per_turn_center, 'o-', color='C1', linewidth=2, markersize=5)
    ax.set_xlabel("Move number")
    ax.set_ylabel("Center-cell decoding accuracy")
    ax.set_title(f"MLP probe (H={args.hidden} {args.features}): "
                 f"center-cell accuracy by turn")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved {args.output}")

    print(f"\nturn  center_acc")
    for t, a in zip(turns, per_turn_center):
        print(f"  {t:>3d}    {a:.4%}")


if __name__ == "__main__":
    main()
