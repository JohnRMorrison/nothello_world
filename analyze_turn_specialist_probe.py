"""Per-cell board decoding from each turn specialist's hidden, vs unified's
hidden, at the specialist's target turn.

For each (model, target_turn):
  - Forward held-out positions at that turn through the model
  - Train Ridge classifier per cell on first half, eval per cell on second half
  - Report mean per-cell accuracy + region breakdown (corner/edge/inner/center)

Tests whether specialists develop a better board encoding at their target
turn than the unified model does. If so, "specialization helps the hidden
encode board state" -- a structural advantage of turn-stratification.
If not, the hidden encoding is similar; depth is the limit.

Usage:
    python analyze_turn_specialist_probe.py
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES,
)
from train_pattern_simple import DirectMLP


CKPT_DIR = "experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints"
TURNS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
CENTER_64 = {27, 28, 35, 36}


def cell_class(c64):
    if c64 in CENTER_64: return 'center'
    r, c = c64 // 8, c64 % 8
    if r in (0, 7) and c in (0, 7): return 'corner'
    if r in (0, 7) or c in (0, 7):  return 'edge'
    return 'inner'


def get_hidden(model_even, model_odd, feat_X, pos, device):
    """Returns (N, H) numpy hidden activations via the parity-specific net."""
    hs = []
    batch = 4096
    with torch.no_grad():
        for i in range(0, len(feat_X), batch):
            x = feat_X[i:i + batch].to(device)
            pb = pos[i:i + batch]
            em = (pb % 2 == 0); om = ~em
            h_b = torch.zeros(len(x), model_even.net[0].out_features, device=device)
            if em.any(): h_b[em] = torch.relu(model_even.net[0](x[em]))
            if om.any(): h_b[om] = torch.relu(model_odd.net[0](x[om]))
            hs.append(h_b.cpu().numpy())
    return np.concatenate(hs, axis=0)


def probe_per_cell(H_train, Y_train, H_test, Y_test):
    """Returns (64,) per-cell accuracy and (4,) region means (corner/edge/inner/center)."""
    scaler = StandardScaler()
    H_train = scaler.fit_transform(H_train)
    H_test  = scaler.transform(H_test)
    acc = np.zeros(64)
    for c in range(64):
        ytr = Y_train[:, c]; yte = Y_test[:, c]
        if len(np.unique(ytr)) < 2:
            acc[c] = (yte == ytr[0]).mean() if len(yte) > 0 else np.nan
            continue
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(H_train, ytr)
        acc[c] = clf.score(H_test, yte)
    regions = {}
    for label in ('corner', 'edge', 'inner', 'center'):
        cells = [c for c in range(64) if cell_class(c) == label]
        regions[label] = acc[cells].mean()
    return acc, regions


def load_model(ckpt_path, hidden, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    input_dim = ckpt.get('input_dim', 120)
    me = DirectMLP(input_dim, hidden, 960).to(device)
    mo = DirectMLP(input_dim, hidden, 960).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()
    return me, mo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-ckpt",
                        default=f"{CKPT_DIR}/pattern_simple_direct_H512_wheneven.pt")
    parser.add_argument("--specialist-tmpl",
                        default=f"{CKPT_DIR}/pattern_simple_direct_H512_wheneven_turn{{T}}.pt")
    parser.add_argument("--hidden", type=int, default=512)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load eval chunk
    out_dir = "experiments/mathematical_transformation_experiments/heuristic_probe_results"
    chunk_dir = os.path.join(out_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f
                         and "_when60" not in f and "_by_black" not in f)
    eval_path = chunk_files[-1]
    print(f"Loading {eval_path}")
    X, Y, pos = _load_features(eval_path)
    feat_X = X[:, N_MOVES:3 * N_MOVES]   # when+even
    del X
    pos_np = pos.numpy()
    Y_np = Y.numpy()

    me_u, mo_u = load_model(args.unified_ckpt, args.hidden, device)

    print()
    print("=" * 96)
    print("Per-cell probe accuracy by turn: SPECIALIST vs UNIFIED at the same turn")
    print("=" * 96)
    header = (f"{'turn':>5s} {'n':>8s} "
              f"{'spec_mean':>10s} {'unif_mean':>10s} {'Δ':>7s} "
              f"{'spec_ctr':>10s} {'unif_ctr':>10s} {'Δ':>7s}")
    print(header)
    print("-" * len(header))

    for T in TURNS:
        m = (pos_np == T)
        if m.sum() < 1000:
            print(f"  {T:>3d}  too few positions ({int(m.sum())})")
            continue
        idx = np.where(m)[0]
        # Cap at 20k positions to keep Ridge fast (10k train / 10k test)
        rng = np.random.RandomState(0)
        if len(idx) > 20000:
            idx = rng.choice(idx, 20000, replace=False)
        feat_T = feat_X[idx]
        Y_T = Y_np[idx]
        pos_T = pos[idx]
        n_T = len(idx)
        split = n_T // 2

        spec_path = args.specialist_tmpl.format(T=T)
        if not os.path.exists(spec_path):
            print(f"  {T:>3d}  spec missing: {spec_path}")
            continue
        me_s, mo_s = load_model(spec_path, args.hidden, device)

        H_spec = get_hidden(me_s, mo_s, feat_T, pos_T, device)
        H_unif = get_hidden(me_u, mo_u, feat_T, pos_T, device)

        _, spec_reg = probe_per_cell(H_spec[:split], Y_T[:split],
                                     H_spec[split:], Y_T[split:])
        _, unif_reg = probe_per_cell(H_unif[:split], Y_T[:split],
                                     H_unif[split:], Y_T[split:])

        spec_mean = np.mean(list(spec_reg.values()))
        unif_mean = np.mean(list(unif_reg.values()))
        print(f"  {T:>3d} {n_T:>8d} "
              f"{spec_mean:>10.4f} {unif_mean:>10.4f} "
              f"{spec_mean - unif_mean:>+7.4f} "
              f"{spec_reg['center']:>10.4f} {unif_reg['center']:>10.4f} "
              f"{spec_reg['center'] - unif_reg['center']:>+7.4f}")
