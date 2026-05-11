"""Per-cell linear probe on the H=1024 wheneven hidden layer.

For each of 64 board cells, train a 3-class linear probe (Linear(1024, 3) via
Ridge regression on one-hot targets) from the model's hidden activation to the
cell's ground-truth state (empty / black / white). Then report the per-cell
accuracy distribution.

What the answer means:
  - Tight distribution (all cells 95-98%) -> bottleneck is uniform correlated
    errors. Need architectural change (more capacity, better features, or
    explicit board aux supervision with stronger weight).
  - Wide distribution (some cells 80%, some 99%) -> specific weak cells.
    Targeted training (cell-weighted loss, or features that disambiguate
    those cells) is the right intervention.

The model has separate even-turn (me) and odd-turn (mo) sub-models, so we
probe each on its corresponding positions and report both maps.

Usage:
    python probe_per_cell.py \\
        --ckpt experiments/.../pattern_simple_direct_H1024_wheneven.pt \\
        --hidden 1024
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES,
)
from train_pattern_simple import (
    DirectMLP, EndToEndMLP, TwoStageMLP,
    to_signed_parity_input, to_mine_signed_input, to_board_state_input,
    to_color_split_input, to_played_halfmask_input, to_played_bit_input,
    to_move_grid_input, to_move_grid_onehot_input,
)


CENTER_64 = {27, 28, 35, 36}   # d4, e4, d5, e5


_FEAT_COLS = {
    "when":        list(range(N_MOVES, 2 * N_MOVES)),
    "played":      list(range(0, N_MOVES)),
    "played+when": list(range(0, 2 * N_MOVES)),
    "when+even":   list(range(N_MOVES, 3 * N_MOVES)),
    "played+even": list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)),
    "all":         list(range(0, 3 * N_MOVES)),
}


def select_features(X, Y, pos, features):
    if features in _FEAT_COLS:
        return X[:, _FEAT_COLS[features]]
    if features == "signed_parity":   return to_signed_parity_input(X)
    if features == "mine_signed":     return to_mine_signed_input(Y, pos)
    if features == "board_state":     return to_board_state_input(Y, pos)
    if features == "color_split":     return to_color_split_input(X)
    if features == "played+halfmask": return to_played_halfmask_input(X)
    if features == "played+bit":      return to_played_bit_input(X)
    if features == "move_grid":       return to_move_grid_input(X)
    if features == "move_grid_onehot": return to_move_grid_onehot_input(X)
    raise ValueError(features)


def get_hidden(model, x):
    """Forward pass through the first Linear + ReLU only. Returns (B, H)."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("No Linear layer found")
    return torch.relu(linears[0](x))


def cell_class(c64):
    if c64 in CENTER_64:
        return 'center'
    row, col = c64 // 8, c64 % 8
    if row in (0, 7) and col in (0, 7):
        return 'corner'
    if row in (0, 7) or col in (0, 7):
        return 'edge'
    return 'inner'


def cell_alg(c64):
    return f"{'abcdefgh'[c64 % 8]}{c64 // 8 + 1}"


def collect_hidden(model, feat_X, mask, device, batch=4096):
    """Forward feat_X[mask] through model and return hidden (n, H) cpu numpy."""
    if not mask.any():
        return np.zeros((0, 0), dtype=np.float32)
    idx = np.where(mask)[0]
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b_idx = idx[s:s + batch]
            xb = feat_X[b_idx].to(device)
            h = get_hidden(model, xb)
            out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def train_and_eval_probes(H_train, Y_train, H_test, Y_test, label):
    """Train per-cell Ridge probes; report 64-cell accuracy array."""
    print(f"\n  Standardizing {label} hidden ({H_train.shape})...")
    scaler = StandardScaler()
    H_train = scaler.fit_transform(H_train)
    H_test = scaler.transform(H_test)

    print(f"  Training 64 per-cell probes...")
    cell_acc = np.zeros(64)
    for c in range(64):
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(H_train, Y_train[:, c])
        cell_acc[c] = clf.score(H_test, Y_test[:, c])
    return cell_acc


def print_grid(cell_acc, label):
    print(f"\nPer-cell board-state probe accuracy ({label}):")
    print("     " + " ".join(f"{c:>6s}" for c in "abcdefgh"))
    for r in range(8):
        row = []
        for c in range(8):
            sq = r * 8 + c
            v = cell_acc[sq]
            tag = "*" if sq in CENTER_64 else " "
            row.append(f"{v:>5.3f}{tag}")
        print(f"  {r+1}  " + " ".join(row))


def print_summary(cell_acc, label):
    print(f"\nSummary ({label}):")
    print(f"  mean: {cell_acc.mean():.4f}")
    print(f"  std:  {cell_acc.std():.4f}")
    print(f"  min:  {cell_acc.min():.4f}  ({cell_alg(int(cell_acc.argmin()))})")
    print(f"  max:  {cell_acc.max():.4f}  ({cell_alg(int(cell_acc.argmax()))})")
    print(f"  range:{cell_acc.max() - cell_acc.min():.4f}")
    print(f"\n  Accuracy by cell class ({label}):")
    for cls in ('corner', 'edge', 'inner', 'center'):
        mask = np.array([cell_class(c) == cls for c in range(64)])
        sub = cell_acc[mask]
        print(f"    {cls:>8s}: n={int(mask.sum()):2d}  "
              f"mean={sub.mean():.4f}  min={sub.min():.4f}  max={sub.max():.4f}")
    order = np.argsort(cell_acc)
    print(f"\n  Worst 10 cells ({label}):")
    for c in order[:10]:
        print(f"    {cell_alg(int(c)):>3s} ({cell_class(int(c)):>6s})  "
              f"acc={cell_acc[int(c)]:.4f}")
    print(f"  Best 10 cells ({label}):")
    for c in order[-10:][::-1]:
        print(f"    {cell_alg(int(c)):>3s} ({cell_class(int(c)):>6s})  "
              f"acc={cell_acc[int(c)]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--mode", default="direct",
                        choices=["direct", "emergent", "e2e", "two-stage", "randproj"])
    parser.add_argument("--features", default=None)
    parser.add_argument("--n-train", type=int, default=40000)
    parser.add_argument("--n-test",  type=int, default=20000)
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()

    output_dir = "experiments/mathematical_transformation_experiments/heuristic_probe_results"
    chunk_dir = os.path.join(output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz")
                         and "_patterns" not in f
                         and "_when60" not in f)
    eval_path = chunk_files[-1]
    print(f"Loading {eval_path}")

    X, Y, pos = _load_features(eval_path)
    N = len(Y)
    print(f"Positions in chunk: {N}")

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    n_patterns = ckpt.get('n_patterns', 960)
    input_dim = ckpt.get('input_dim', N_MOVES)
    if args.features is None:
        name = os.path.basename(args.ckpt)
        if "wheneven" in name or input_dim == 120: args.features = "when+even"
        elif input_dim == 60: args.features = "when"
        elif input_dim == 180: args.features = "all"
        else: raise ValueError(f"Can't infer features for {name}")
    print(f"Features: {args.features} (input_dim={input_dim}, hidden={args.hidden})")

    Cls = {"direct": DirectMLP, "randproj": DirectMLP,
           "two-stage": TwoStageMLP,
           "emergent": EndToEndMLP, "e2e": EndToEndMLP}[args.mode]
    me = Cls(input_dim, args.hidden, n_patterns).to(device)
    mo = Cls(input_dim, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()
    print(f"Loaded {args.ckpt}")

    feat_X = select_features(X, Y, pos, args.features)
    del X

    # Subsample
    n_total = args.n_train + args.n_test
    if n_total > N:
        n_total = N
        args.n_test = n_total - args.n_train
    rng = np.random.RandomState(0)
    si = np.sort(rng.choice(N, n_total, replace=False))
    feat_X = feat_X[si]
    Y_np = Y[si].numpy()
    pos_np = pos[si].numpy()
    print(f"Subsampled to {n_total} positions "
          f"({args.n_train} train + {args.n_test} test)")

    train_mask = np.zeros(n_total, dtype=bool); train_mask[:args.n_train] = True
    test_mask  = ~train_mask
    em_mask = (pos_np % 2 == 0)
    om_mask = ~em_mask
    print(f"  even-turn positions: {em_mask.sum()}  odd-turn: {om_mask.sum()}")

    print("\nCollecting hidden activations for me (even-turn model)...")
    H_em = collect_hidden(me, feat_X, em_mask, device)
    print(f"  H_em shape: {H_em.shape}")
    print("Collecting hidden activations for mo (odd-turn model)...")
    H_om = collect_hidden(mo, feat_X, om_mask, device)
    print(f"  H_om shape: {H_om.shape}")

    # Slice train/test
    em_train = train_mask[em_mask]; em_test = test_mask[em_mask]
    om_train = train_mask[om_mask]; om_test = test_mask[om_mask]

    print("\n=== Probing ME (even-turn model) hidden ===")
    acc_me = train_and_eval_probes(
        H_em[em_train], Y_np[em_mask][em_train],
        H_em[em_test],  Y_np[em_mask][em_test], "me")

    print("\n=== Probing MO (odd-turn model) hidden ===")
    acc_mo = train_and_eval_probes(
        H_om[om_train], Y_np[om_mask][om_train],
        H_om[om_test],  Y_np[om_mask][om_test], "mo")

    print_grid(acc_me, "me")
    print_summary(acc_me, "me")
    print_grid(acc_mo, "mo")
    print_summary(acc_mo, "mo")

    print("\n=== Average across me and mo ===")
    acc_mean = (acc_me + acc_mo) / 2.0
    print_grid(acc_mean, "mean")
    print_summary(acc_mean, "mean")

    if args.output:
        np.savez(args.output, acc_me=acc_me, acc_mo=acc_mo, acc_mean=acc_mean)
        print(f"\nSaved per-cell accuracies to {args.output}")
