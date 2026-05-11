"""Cross-Condition Generalization Performance (CCGP) for board state decoding.

Two flavors:
  A (--mode phase):   split positions by turn number into N bins.
                      For each (cell, class), leave-one-bin-out probing.
  B (--mode context): for each cell C, condition on the state of a paired
                      "context" cell D (default: diagonal opposite). Train on
                      positions where D is empty, test where D is occupied
                      (and vice versa).

For each (cell, class):
  CCGP    = mean held-out-condition probe accuracy
  Within  = mean within-condition probe accuracy (50/50 train-test on the
            same bin) — context-blind ceiling
  Gap     = Within − CCGP  (large gap = context-specific representation)

Per-parity for the MLP (even-MLP on even positions, odd-MLP on odd positions);
unified for OGPT (single residual stream at chosen layer).

Usage:
  # MLP, cross-phase, 4 turn bins, ~30K positions
  python compute_ccgp.py --ckpt <path>.pt --hidden 512 --mode-data mlp \\
      --features wheneven --ccgp-mode phase --n-bins 4 --n 30000

  # MLP, cross-context-cell (diagonal pair)
  python compute_ccgp.py --ckpt <path>.pt --hidden 512 --mode-data mlp \\
      --features wheneven --ccgp-mode context

  # OGPT layer 5
  python compute_ccgp.py --mode-data ogpt --ogpt-ckpt ckpts/gpt_nanda_synthetic.ckpt \\
      --layer 5 --ccgp-mode phase
"""
import argparse
import os
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def _feature_cols(features):
    """Map feature name to column slice in the 180-d precomputed features."""
    N = 60
    return {
        "when":        list(range(N, 2 * N)),
        "played":      list(range(0, N)),
        "wheneven":    list(range(N, 3 * N)),
        "when+even":   list(range(N, 3 * N)),
        "played+even": list(range(0, N)) + list(range(2 * N, 3 * N)),
        "all":         list(range(0, 3 * N)),
    }.get(features, list(range(N, 3 * N)))


def get_mlp_activations(ckpt_path, hidden_dim, features, eval_path, n_sample):
    """Load MLP and return per-parity hidden activations + board labels + positions.

    Returns dict {even: (h, board, pos), odd: (h, board, pos)} where h is
    the post-ReLU hidden vector from the right parity sub-network.
    """
    from train_pattern_simple import DirectMLP
    from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
        _load_features, get_device,
    )

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    input_dim = ckpt.get('input_dim', len(_feature_cols(features)))
    n_patterns = ckpt.get('n_patterns', 960)

    me = DirectMLP(input_dim, hidden_dim, n_patterns).to(device)
    mo = DirectMLP(input_dim, hidden_dim, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); mo.load_state_dict(ckpt['odd'])
    me.eval(); mo.eval()

    X, Y, pos = _load_features(eval_path)
    feat_cols = _feature_cols(features)
    X = X[:, feat_cols]
    if n_sample is not None and n_sample < len(X):
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(len(X), n_sample, replace=False))
        X, Y, pos = X[idx], Y[idx], pos[idx]

    pos_np = pos.numpy() if hasattr(pos, 'numpy') else np.asarray(pos)
    Y_np = Y.numpy() if hasattr(Y, 'numpy') else np.asarray(Y)

    em = (pos_np % 2 == 0)
    om = ~em
    out = {}
    with torch.no_grad():
        for parity, mask, model in [("even", em, me), ("odd", om, mo)]:
            if not mask.any():
                continue
            x_p = X[torch.from_numpy(mask).bool()].to(device) if hasattr(X, 'numpy') else \
                  torch.from_numpy(X[mask]).to(device)
            # net = Linear(input,H) -> ReLU -> Linear(H,960)
            h = model.net[1](model.net[0](x_p))
            out[parity] = (h.cpu().numpy().astype(np.float32),
                           Y_np[mask], pos_np[mask])
    return out


def get_ogpt_activations(ckpt_path, layer, eval_path, n_sample):
    """Load OGPT, run forward, return residual stream at `layer` + board + positions.

    Returns dict {"all": (h, board, pos)}.
    """
    from data import get_othello
    from mingpt.dataset import CharDataset
    from mingpt.model import GPT, GPTConfig
    from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
        _load_features, get_device,
    )
    device = get_device()

    othello = get_othello(ood_num=100)
    dataset = CharDataset(othello)
    mconf = GPTConfig(dataset.vocab_size, dataset.block_size,
                      n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device).eval()

    X, Y, pos = _load_features(eval_path)
    pos_np = pos.numpy() if hasattr(pos, 'numpy') else np.asarray(pos)
    Y_np = Y.numpy() if hasattr(Y, 'numpy') else np.asarray(Y)

    # We need the actual game token sequences, not "when" features. The
    # feature chunks were precomputed from move histories — extracting those
    # back out is non-trivial here. For MVP, sample positions from a small
    # set of synthetic games run through OGPT.
    raise NotImplementedError(
        "OGPT activation extraction needs the game-token sequences for the "
        "sampled positions; for now run --mode-data mlp. Wire this up by "
        "loading the pickled games corresponding to the eval chunk and "
        "running them through `model.blocks[:layer]`.")


# ---------------------------------------------------------------------------
# CCGP probes
# ---------------------------------------------------------------------------

def _balance(idx, y, rng):
    """Subsample to balance positive/negative in idx given binary labels y."""
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    k = min(len(pos), len(neg))
    if k == 0:
        return idx
    pos = rng.choice(pos, k, replace=False)
    neg = rng.choice(neg, k, replace=False)
    return np.concatenate([pos, neg])


def _probe_acc(h_train, y_train, h_test, y_test, C=1.0, max_iter=1000):
    """Train logistic regression, return test accuracy.

    Standardizes features (StandardScaler fit on train) before LR so lbfgs
    converges in tens of iterations instead of thousands.
    """
    if len(h_train) < 20 or len(h_test) < 10:
        return None
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None
    scaler = StandardScaler()
    h_train = scaler.fit_transform(h_train)
    h_test = scaler.transform(h_test)
    clf = LogisticRegression(max_iter=max_iter, C=C, solver='lbfgs')
    clf.fit(h_train, y_train)
    return float(clf.score(h_test, y_test))


def _subsample(idx, n_target, rng):
    """Without-replacement subsample if too many, else return idx unchanged."""
    if len(idx) <= n_target:
        return idx
    return rng.choice(idx, n_target, replace=False)


def ccgp_phase(h, board, pos, n_bins=4, classes=(1, 2),
               cells=range(64), seed=0, min_per_bin=200,
               train_size=None):
    """Option A: leave-one-bin-out CCGP across game-phase bins.

    For each fold, both CCGP and Within use the SAME train_size (after
    class balancing). Within is k-fold CV inside ONE bin, with the same
    train_size taken from the (k-1) train-folds of that bin pooled together.
    If train_size is None, it's auto-set to the smaller of (CCGP train pool,
    Within train pool) so the two are matched per cell × class.
    """
    rng = np.random.RandomState(seed)
    quantiles = np.quantile(pos, np.linspace(0, 1, n_bins + 1))
    quantiles[0] -= 0.5; quantiles[-1] += 0.5
    bins = np.digitize(pos, quantiles[1:-1])

    ccgp_per, within_per = [], []
    for cell in cells:
        for cls in classes:
            y = (board[:, cell] == cls).astype(np.int32)
            if y.sum() < 100 or (1 - y).sum() < 100:
                continue

            # Per-bin balanced index pools
            per_bin = []
            for b in range(n_bins):
                idx = np.where(bins == b)[0]
                if len(idx) < min_per_bin:
                    per_bin.append(None)
                    continue
                per_bin.append(_balance(idx, y, rng))

            valid_bins = [b for b in range(n_bins) if per_bin[b] is not None]
            if len(valid_bins) < 2:
                continue

            # Match train size between CCGP (n-1 bins pooled) and Within
            # (n-1 folds of one bin pooled).
            ccgp_train_pool = min(sum(len(per_bin[b]) for b in valid_bins if b != h_b)
                                  for h_b in valid_bins)
            within_train_pool = min(int(len(per_bin[b]) * (len(valid_bins) - 1) / len(valid_bins))
                                    for b in valid_bins)
            t = train_size or min(ccgp_train_pool, within_train_pool)
            t = max(t, 50)

            # CCGP: leave-one-bin-out, subsample to t for training
            fold_acc = []
            for held_b in valid_bins:
                tr_pool = np.concatenate([per_bin[b] for b in valid_bins if b != held_b])
                te_idx = per_bin[held_b]
                tr_idx = _subsample(tr_pool, t, rng)
                a = _probe_acc(h[tr_idx], y[tr_idx], h[te_idx], y[te_idx])
                if a is not None:
                    fold_acc.append(a)
            if fold_acc:
                ccgp_per.append(np.mean(fold_acc))

            # Within: k-fold CV inside each bin. k = len(valid_bins) so the
            # held-out fraction matches CCGP (1/n of one bin).
            wf = []
            n_folds = len(valid_bins)
            for b in valid_bins:
                idx = per_bin[b].copy()
                rng.shuffle(idx)
                fold_size = len(idx) // n_folds
                if fold_size < 20:
                    continue
                for f in range(n_folds):
                    te = idx[f * fold_size:(f + 1) * fold_size]
                    tr = np.concatenate([idx[:f * fold_size], idx[(f + 1) * fold_size:]])
                    tr = _subsample(tr, t, rng)
                    a = _probe_acc(h[tr], y[tr], h[te], y[te])
                    if a is not None:
                        wf.append(a)
            if wf:
                within_per.append(np.mean(wf))

    return {
        'ccgp':   float(np.mean(ccgp_per))   if ccgp_per else float('nan'),
        'within': float(np.mean(within_per)) if within_per else float('nan'),
        'gap':    float(np.mean(within_per) - np.mean(ccgp_per))
                  if (ccgp_per and within_per) else float('nan'),
        'n_pairs': len(ccgp_per),
        'per_pair_ccgp': ccgp_per,
        'per_pair_within': within_per,
    }


def ccgp_context(h, board, pos, classes=(1, 2),
                 cells=range(64), seed=0, min_per_cond=200,
                 train_size=None):
    """Option B: cross-context-cell CCGP. Pair each cell C with diagonal cell D
    (D = 63 - C). Split positions by D's occupancy. Train on D-empty, test on
    D-occupied (and vice versa). Within = 2-fold CV inside each context
    state, with the same per-fit training set size as CCGP."""
    rng = np.random.RandomState(seed)
    ccgp_per, within_per = [], []
    for cell in cells:
        ctx = 63 - cell
        if cell == ctx:
            continue
        ctx_state = (board[:, ctx] != 0).astype(int)
        for cls in classes:
            y = (board[:, cell] == cls).astype(np.int32)
            if y.sum() < 100 or (1 - y).sum() < 100:
                continue

            # balanced index per context state
            per_state = []
            for s in (0, 1):
                idx = np.where(ctx_state == s)[0]
                if len(idx) < min_per_cond:
                    per_state.append(None)
                    continue
                per_state.append(_balance(idx, y, rng))
            if any(p is None for p in per_state):
                continue

            # Match training size: CCGP trains on one state, tests on the other.
            # Within trains on half of one state, tests on the other half.
            ccgp_train_pool = min(len(per_state[0]), len(per_state[1]))
            within_train_pool = min(len(per_state[0]) // 2, len(per_state[1]) // 2)
            t = train_size or min(ccgp_train_pool, within_train_pool)
            t = max(t, 50)

            # CCGP: train on state 0, test on state 1 (and vice versa)
            fold_acc = []
            for held in (0, 1):
                tr_idx = _subsample(per_state[1 - held], t, rng)
                te_idx = per_state[held]
                a = _probe_acc(h[tr_idx], y[tr_idx], h[te_idx], y[te_idx])
                if a is not None:
                    fold_acc.append(a)
            if fold_acc:
                ccgp_per.append(np.mean(fold_acc))

            # Within: 2-fold CV within each state, same train_size
            wf = []
            for s in (0, 1):
                idx = per_state[s].copy()
                rng.shuffle(idx)
                half = len(idx) // 2
                for tr_part, te_part in [(idx[:half], idx[half:]),
                                          (idx[half:], idx[:half])]:
                    tr = _subsample(tr_part, t, rng)
                    a = _probe_acc(h[tr], y[tr], h[te_part], y[te_part])
                    if a is not None:
                        wf.append(a)
            if wf:
                within_per.append(np.mean(wf))

    return {
        'ccgp':   float(np.mean(ccgp_per))   if ccgp_per else float('nan'),
        'within': float(np.mean(within_per)) if within_per else float('nan'),
        'gap':    float(np.mean(within_per) - np.mean(ccgp_per))
                  if (ccgp_per and within_per) else float('nan'),
        'n_pairs': len(ccgp_per),
    }


def _print_summary(label, res):
    print(f"\n--- {label} ---")
    print(f"  CCGP    = {res['ccgp']:.4f}")
    print(f"  Within  = {res['within']:.4f}")
    print(f"  Gap     = {res['gap']:.4f}")
    print(f"  ({res['n_pairs']} (cell, class) pairs averaged)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None,
                   help="Path to MLP checkpoint (mode-data mlp).")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--features", default="wheneven")
    p.add_argument("--mode-data", choices=["mlp", "ogpt"], default="mlp")
    p.add_argument("--ogpt-ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--n", type=int, default=30000,
                   help="positions to sample from the eval chunk")
    p.add_argument("--ccgp-mode", choices=["phase", "context", "both"],
                   default="both")
    p.add_argument("--n-bins", type=int, default=4)
    p.add_argument("--eval-chunk", default=None,
                   help="path to a feature_chunks/*.npz file")
    args = p.parse_args()

    if args.eval_chunk is None:
        chunk_dir = ("experiments/mathematical_transformation_experiments/"
                     "heuristic_probe_results/feature_chunks")
        files = sorted(f for f in os.listdir(chunk_dir)
                       if f.endswith(".npz") and "_patterns" not in f)
        args.eval_chunk = os.path.join(chunk_dir, files[-1])
    print(f"Eval chunk: {args.eval_chunk}")

    if args.mode_data == "mlp":
        if args.ckpt is None:
            p.error("--ckpt is required when --mode-data mlp")
        per_parity = get_mlp_activations(
            args.ckpt, args.hidden, args.features,
            args.eval_chunk, args.n)
    else:
        per_parity = {"all": get_ogpt_activations(
            args.ogpt_ckpt, args.layer, args.eval_chunk, args.n)}

    print(f"\nLoaded activations:")
    for parity, (h, board, pos) in per_parity.items():
        print(f"  {parity}: {len(h)} positions, hidden_dim={h.shape[1]}, "
              f"turn_range=[{int(pos.min())}, {int(pos.max())}]")

    for parity, (h, board, pos) in per_parity.items():
        print(f"\n========================================")
        print(f"  Parity: {parity}")
        print(f"========================================")

        if args.ccgp_mode in ("phase", "both"):
            r = ccgp_phase(h, board, pos, n_bins=args.n_bins)
            _print_summary(f"Option A: cross-game-phase ({args.n_bins} bins)", r)

        if args.ccgp_mode in ("context", "both"):
            r = ccgp_context(h, board, pos)
            _print_summary("Option B: cross-context-cell (diagonal pair)", r)


if __name__ == "__main__":
    main()
