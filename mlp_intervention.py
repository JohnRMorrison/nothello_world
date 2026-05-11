"""Probe-directed interventions on the H=1024 wheneven MLP hidden layer.

Mirrors the OGPT intervention experiment (multi_intervention.py) but
operates on the 1-layer MLP. For each target cell, we:
  1. Retrain a Ridge probe on hidden -> board-state-at-target-cell.
  2. Compute "empty direction" = coef_[empty] - mean(coef_[other]).
  3. For positions where the target cell is currently illegal in the MLP's
     output, add k * normalized direction to hidden activation.
  4. Re-forward through the final Linear(H, 960) layer, aggregate to per-cell
     scores via prob_or, and measure:
       - epsilon = P(target_legal) - max P(other_illegal)
       - delta   = P(target_legal) - min P(other_legal)
       - crosstalk = mean |Delta P| on non-target cells

If our 1-layer MLP shows similar epsilon > 0 / delta -> 0 success rates as
OGPT under the same probe-directed intervention protocol, then those OGPT
results don't prove "world model" -- they only prove linear decodability,
which our MLP has by construction without any computation.

Usage:
    python mlp_intervention.py \\
        --ckpt experiments/.../pattern_simple_direct_H1024_wheneven.pt \\
        --hidden 1024 --scales 1,2,3,5,8
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES,
)
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import (
    DirectMLP, EndToEndMLP, TwoStageMLP, compute_pattern_labels_batch,
    pat_labels_to_cell_labels, _get_cell_pat_index,
)


_FEAT_COLS = {
    "when":      list(range(N_MOVES, 2 * N_MOVES)),
    "when+even": list(range(N_MOVES, 3 * N_MOVES)),
    "all":       list(range(0, 3 * N_MOVES)),
}


def get_hidden(model, x):
    lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return torch.relu(lins[0](x))


def get_output_layer(model):
    lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return lins[1]   # second Linear maps hidden -> 960


def cell_alg(c):
    return f"{'abcdefgh'[c % 8]}{c // 8 + 1}"


def prob_or_scores(pat_logits, idx, mask):
    log1m = -F.softplus(pat_logits)
    g = log1m[:, idx]
    g = g.masked_fill(~mask, 0.0)
    return torch.sigmoid(-g.sum(dim=-1) * (-1.0)) if False else (
        1.0 - torch.exp(g.sum(dim=-1))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--mode", default="direct")
    parser.add_argument("--features", default="when+even")
    parser.add_argument("--scales", default="1,2,3,5,8",
                        help="Comma-separated intervention scale multipliers.")
    parser.add_argument("--n-train-probe", type=int, default=20000,
                        help="Positions used to train the Ridge probe per cell.")
    parser.add_argument("--n-test", type=int, default=5000,
                        help="Held-out positions on which to test interventions.")
    parser.add_argument("--max-cells-per-pos", type=int, default=5,
                        help="At most this many illegal-target cells per position.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    scales = [float(s) for s in args.scales.split(",")]

    output_dir = "experiments/mathematical_transformation_experiments/heuristic_probe_results"
    chunk_dir = os.path.join(output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f
                         and "_when60" not in f)
    eval_path = chunk_files[-1]
    print(f"Loading {eval_path}")

    X, Y, pos = _load_features(eval_path)
    N = len(Y)

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    input_dim = ckpt.get('input_dim', N_MOVES)
    n_patterns = ckpt.get('n_patterns', 960)

    Cls = {"direct": DirectMLP, "emergent": EndToEndMLP, "e2e": EndToEndMLP,
           "two-stage": TwoStageMLP, "randproj": DirectMLP}[args.mode]
    me = Cls(input_dim, args.hidden, n_patterns).to(device)
    mo = Cls(input_dim, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()
    print(f"Loaded {args.ckpt}")

    feat_X = X[:, _FEAT_COLS[args.features]]
    del X

    n_total = args.n_train_probe + args.n_test
    rng = np.random.RandomState(0)
    si = np.sort(rng.choice(N, n_total, replace=False))
    feat_X = feat_X[si]
    Y_np = Y[si].numpy()
    pos_np = pos[si].numpy()
    print(f"Subsampled to {n_total} positions")

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = \
        precompute_pattern_arrays(patterns)
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device)
    idx_t, mask_t = _get_cell_pat_index(pattern_to_cell, 60)

    # Collect hidden activations for ALL n_total positions, split by me/mo.
    print("\nCollecting hidden activations...")
    H_all = np.zeros((n_total, args.hidden), dtype=np.float32)
    batch = 4096
    with torch.no_grad():
        for i in range(0, n_total, batch):
            xb = feat_X[i:i + batch].to(device)
            pb = pos_np[i:i + batch]
            em = (pb % 2 == 0); om = ~em
            if em.any():
                H_all[np.where(em)[0] + i] = get_hidden(me, xb[em]).cpu().numpy()
            if om.any():
                H_all[np.where(om)[0] + i] = get_hidden(mo, xb[om]).cpu().numpy()

    # Train probes on the first n_train_probe; test interventions on the rest.
    train_slice = slice(0, args.n_train_probe)
    test_slice  = slice(args.n_train_probe, n_total)

    print("\nFitting per-cell Ridge probes on training half...")
    scaler = StandardScaler().fit(H_all[train_slice])
    H_train_s = scaler.transform(H_all[train_slice])
    # Note: we'll need raw-space directions, so we'll undo the scaler later.

    # Per cell c in [0, 60), train probe predicting Y_np[:, c64] where c64 is
    # the 64-cell board index (skipping centers).
    def m60_to_b64(m):
        skip = [27, 28, 35, 36]
        j = 0
        for b in range(64):
            if b in skip: continue
            if j == m: return b
            j += 1
        raise ValueError(m)

    # Direction per cell: coef_[empty] - mean(coef_[other])
    # We map empty class to index 0 (Y_np uses 0=empty in compare_aggregators
    # encoding -- check).
    directions = np.zeros((60, args.hidden), dtype=np.float32)
    probe_accs = np.zeros(60, dtype=np.float32)
    for m in range(60):
        b = m60_to_b64(m)
        y_tr = Y_np[train_slice, b]
        if len(np.unique(y_tr)) < 2:
            continue
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(H_train_s, y_tr)
        probe_accs[m] = clf.score(scaler.transform(H_all[test_slice]),
                                  Y_np[test_slice, b])
        classes = clf.classes_
        coef = clf.coef_   # (k, H) where k <= 3 for the classes present
        # Map class 0 (empty) to its row if present; else skip
        if 0 in classes:
            empty_row = coef[list(classes).index(0)]
            other = np.mean(np.delete(coef, list(classes).index(0), axis=0),
                            axis=0) if coef.shape[0] > 1 else np.zeros_like(empty_row)
            d = empty_row - other
            # Undo scaler to get direction in raw hidden space:
            # x_s = (x - mu) / sigma   ->   coef_s @ x_s = (coef_s / sigma) @ x + const
            # so the direction in raw space is coef / sigma (per-feature scaling).
            d_raw = d / np.maximum(scaler.scale_, 1e-8)
            # Normalize to unit L2.
            n = np.linalg.norm(d_raw)
            if n > 0:
                directions[m] = d_raw / n
    print(f"  Mean probe accuracy across 60 cells: {probe_accs.mean():.4f}")

    # Now run interventions on the test slice.
    out_layer_me = get_output_layer(me)
    out_layer_mo = get_output_layer(mo)

    # Forward the test positions through the final layer to get baseline scores.
    print(f"\nRunning interventions on {n_total - args.n_train_probe} "
          f"test positions, scales = {scales}...")
    test_H = torch.from_numpy(H_all[test_slice]).to(device)
    test_pos = pos_np[test_slice]
    test_Y = Y_np[test_slice]

    def aggregate_cell_probs(pat_logits):
        """prob_or aggregation per cell."""
        log1m = -F.softplus(pat_logits)
        g = log1m[:, idx_t]
        g = g.masked_fill(~mask_t, 0.0)
        return 1.0 - torch.exp(g.sum(dim=-1))  # (B, 60)

    eps_by_scale = {s: [] for s in scales}
    delta_by_scale = {s: [] for s in scales}
    cross_by_scale = {s: [] for s in scales}

    batch = 256
    with torch.no_grad():
        for i in range(0, test_H.shape[0], batch):
            hb = test_H[i:i + batch]
            pb = test_pos[i:i + batch]
            yb = test_Y[i:i + batch]
            em = torch.from_numpy(pb % 2 == 0).to(device)
            # Baseline forward
            out_b = torch.empty(hb.shape[0], n_patterns, device=device)
            if em.any():
                out_b[em] = out_layer_me(hb[em])
            if (~em).any():
                out_b[~em] = out_layer_mo(hb[~em])
            base_cell_p = aggregate_cell_probs(out_b)   # (B, 60)

            # Compute ground-truth legal mask
            yb_t = torch.from_numpy(yb).to(device)
            pb_t = torch.from_numpy(pb).to(device)
            gp = torch.from_numpy(compute_pattern_labels_batch(
                yb, pb,
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask
            )).to(device)
            legal = (pat_labels_to_cell_labels(gp, pattern_to_cell) > 0.5)

            # Pick target cells: top illegal-prob cells in baseline (those the
            # MLP is closest to thinking legal but isn't). Limit per position.
            illegal = ~legal
            for b in range(hb.shape[0]):
                ill_cells = torch.where(illegal[b])[0]
                if len(ill_cells) == 0: continue
                # Sort by current baseline prob descending; take top-K
                ill_scores = base_cell_p[b, ill_cells]
                order = torch.argsort(-ill_scores)
                target_cells = ill_cells[order[:args.max_cells_per_pos]].cpu().numpy()

                legal_cells = torch.where(legal[b])[0].cpu().numpy()
                if len(legal_cells) == 0: continue

                for c in target_cells:
                    d = torch.from_numpy(directions[int(c)]).to(device)
                    if d.norm() == 0: continue
                    for s in scales:
                        h_new = hb[b:b+1] + s * d
                        if em[b]:
                            out_i = out_layer_me(h_new)
                        else:
                            out_i = out_layer_mo(h_new)
                        cell_p = aggregate_cell_probs(out_i)[0]  # (60,)
                        # Metrics
                        p_target = float(cell_p[c])
                        other_illegal = ill_cells[ill_cells != c]
                        if len(other_illegal) > 0:
                            max_other_illegal = float(cell_p[other_illegal].max())
                        else:
                            max_other_illegal = 0.0
                        min_legal = float(cell_p[legal_cells].min())
                        eps = p_target - max_other_illegal
                        delta = p_target - min_legal
                        # Crosstalk: mean |Delta P| on non-target cells
                        non_target_mask = torch.ones(60, dtype=torch.bool, device=device)
                        non_target_mask[c] = False
                        cross = float(
                            (cell_p[non_target_mask] - base_cell_p[b][non_target_mask])
                            .abs().mean())
                        eps_by_scale[s].append(eps)
                        delta_by_scale[s].append(delta)
                        cross_by_scale[s].append(cross)

    print()
    print("=" * 70)
    print("Intervention results (per (position, target_cell, scale) triple)")
    print("=" * 70)
    print(f"{'scale':>6s} {'n':>8s} {'eps>0%':>8s} {'delta>0%':>10s} "
          f"{'eps>0.1':>9s} {'mean eps':>10s} {'mean delta':>12s} "
          f"{'mean crosstalk':>16s}")
    for s in scales:
        e = np.array(eps_by_scale[s])
        d = np.array(delta_by_scale[s])
        c = np.array(cross_by_scale[s])
        if len(e) == 0:
            print(f"{s:>6.2f}  -- no samples --")
            continue
        print(f"{s:>6.2f} {len(e):>8d} "
              f"{(e > 0).mean():>7.2%} {(d > 0).mean():>9.2%} "
              f"{(e > 0.1).mean():>8.2%} "
              f"{e.mean():>10.4f} {d.mean():>12.4f} {c.mean():>16.4f}")

    if args.output:
        np.savez(args.output,
                 probe_accs=probe_accs,
                 **{f"eps_s{s}": np.array(eps_by_scale[s]) for s in scales},
                 **{f"delta_s{s}": np.array(delta_by_scale[s]) for s in scales},
                 **{f"cross_s{s}": np.array(cross_by_scale[s]) for s in scales})
        print(f"\nSaved raw samples to {args.output}")
