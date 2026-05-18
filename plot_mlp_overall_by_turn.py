"""Overall (all-64-cells) decoding accuracy vs game turn for a saved MLP probe.

Mirrors plot_mlp_center_by_turn.py but averages over all 64 cells instead of
just the 4 center cells. Useful for filling out the per-turn analog of the
overall H=... probe accuracy column.

Usage:
    python plot_mlp_overall_by_turn.py \\
        --pat-ckpt   experiments/.../pattern_simple_direct_H512_wheneven.pt \\
        --probe-ckpt experiments/.../probe_direct_H512_wheneven.pt \\
        --hidden 512 --features when+even
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
    p.add_argument("--features", default="when+even",
                   choices=["when", "when+even", "all", "move_grid",
                            "move_grid_onehot"])
    p.add_argument("--chunk-path",
        default="experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks/chunk_0039.npz")
    p.add_argument("--extra-chunks", nargs="*", default=[],
                   help="Extra chunk paths (e.g. late_turns_eval.npz for turns 54-58).")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--output",
                   default="experiments/plots/mlp_overall_by_turn.png")
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
    print(f"Loaded pattern detector (input_dim={input_dim}, H={args.hidden})")

    probe_ckpt = torch.load(args.probe_ckpt, map_location=device)
    probe_e = nn.Linear(args.hidden, 64 * OPTIONS).to(device)
    probe_o = nn.Linear(args.hidden, 64 * OPTIONS).to(device)
    probe_e.load_state_dict(probe_ckpt['even'])
    probe_o.load_state_dict(probe_ckpt['odd'])
    probe_e.eval(); probe_o.eval()
    print(f"Loaded probe (best_acc reported: {probe_ckpt.get('best_acc', '?')})")

    chunks = [args.chunk_path] + list(args.extra_chunks)
    Xs, Ys, Ps = [], [], []
    for cp in chunks:
        x, y, pos = _load_features(cp)
        Xs.append(x); Ys.append(y); Ps.append(pos)
        uniq = sorted(np.unique(pos.numpy()))
        print(f"Loaded {os.path.basename(cp)}: {len(x)} positions, "
              f"turns {uniq[:3]}...{uniq[-3:]}")
    ev_X = torch.cat(Xs, dim=0)
    ev_Y = torch.cat(Ys, dim=0)
    ev_pos = torch.cat(Ps, dim=0)
    print(f"Total: {len(ev_X)} positions")

    turns = sorted(np.unique(ev_pos.numpy()).tolist())
    per_turn_overall = np.zeros(len(turns), dtype=np.float64)
    per_turn_n = np.zeros(len(turns), dtype=np.int64)

    for ti, t in enumerate(turns):
        mask = (ev_pos == t)
        X = ev_X[mask]
        Y = ev_Y[mask]
        Xf = feature_slice(X, args.features)
        even = (t % 2 == 0)
        model = model_e if even else model_o
        probe = probe_e if even else probe_o
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(Xf), args.batch_size):
                xb = Xf[i:i + args.batch_size].to(device)
                yb = Y[i:i + args.batch_size].to(device)
                h = hidden_of(model, xb)
                preds = probe(h).view(-1, 64, OPTIONS).argmax(-1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
        per_turn_overall[ti] = correct / max(total, 1)
        per_turn_n[ti] = mask.sum().item()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(turns, per_turn_overall, 'o-', color='C2', linewidth=2, markersize=5)
    ax.set_xlabel("Move number")
    ax.set_ylabel("Decoding accuracy (all 64 cells)")
    ax.set_title(f"MLP probe (H={args.hidden} {args.features}): "
                 f"overall accuracy by turn")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved {args.output}")

    print(f"\n{'turn':>4s}  {'n':>8s}  {'overall':>9s}")
    for t, a, n in zip(turns, per_turn_overall, per_turn_n):
        print(f"  {t:>3d}  {n:>8d}  {a:.4%}")


if __name__ == "__main__":
    main()
