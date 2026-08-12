"""Cross-Condition Generalization Performance (CCGP) for board state decoding.

Tests whether a board variable ("cell C is mine/yours") is encoded ABSTRACTLY
-- one reusable direction that transfers across contexts -- vs a bag of
context-specific detectors. Train a probe on one region of a condition, test on
a held-out region.

For each (cell, class):
  CCGP    = mean held-out-condition probe accuracy
  Within  = mean within-condition ceiling (matched train size)
  Gap     = Within - CCGP   (large Gap = context-bound, NON-abstract code)

Condition modes (--ccgp-mode):
  phase     game-phase turn bins, leave-one-bin-out (early<->late)
  context   diagonal-opposite cell occupied vs empty
  crowd     local neighbor-occupancy high vs low
  frontier  C adjacent to an empty square vs interior
  spatial   leave-cells-out SHARED decoder (is there one board direction across
            all 64 squares, or 64 per-cell codes?)  [strongest abstraction test]
  null      RANDOM split -- control; CCGP should ~= Within. If not, the estimator
            is biased and real Gaps are inflated. ALWAYS run this.
  both      phase+context     all   every mode incl. null

--nonlinear swaps the logistic-regression probe for an MLP (linear vs non-linear
abstractness). Per-parity for the MLP; unified stream for OGPT.

TODO (next tranche): flip/recency modes (need the raw `when` feature + the
60-move-cell<->64-board-cell map) and parity mode (needs OGPT extraction, which
get_ogpt_activations still stubs).

Usage:
  # MLP, all modes incl. null control, linear probe
  python compute_ccgp.py --ckpt <mlp>.pt --hidden 512 --features playedeven \\
      --ccgp-mode all --n 30000

  # spatial abstraction test with a non-linear probe
  python compute_ccgp.py --ckpt <mlp>.pt --hidden 512 --features playedeven \\
      --ccgp-mode spatial --nonlinear
"""
import argparse
import os
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# 8-neighborhood adjacency on the 8x8 board (used by crowd/frontier contexts).
def _neighbors8(c):
    r, cc = divmod(c, 8)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, ncc = r + dr, cc + dc
            if 0 <= nr < 8 and 0 <= ncc < 8:
                out.append(nr * 8 + ncc)
    return out


_NEI = [_neighbors8(c) for c in range(64)]


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
    from train_pattern_simple import DirectMLP, to_move_grid_input
    from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
        _load_features, get_device,
    )

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    is_movegrid = (features == "move_grid") or (ckpt.get('input_dim') == 3600)
    input_dim = ckpt.get('input_dim',
                         3600 if is_movegrid else len(_feature_cols(features)))
    n_patterns = ckpt.get('n_patterns', 960)

    me = DirectMLP(input_dim, hidden_dim, n_patterns).to(device)
    mo = DirectMLP(input_dim, hidden_dim, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); mo.load_state_dict(ckpt['odd'])
    me.eval(); mo.eval()

    X_raw, Y, pos = _load_features(eval_path)          # X_raw: (N, 180)
    if not is_movegrid:
        X = X_raw[:, _feature_cols(features)]           # column-slice reps
    else:
        X = X_raw                                       # transform per-batch below
    if n_sample is not None and n_sample < len(X):
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(len(X), n_sample, replace=False))
        X, Y, pos = X[idx], Y[idx], pos[idx]

    pos_np = pos.numpy() if hasattr(pos, 'numpy') else np.asarray(pos)
    Y_np = Y.numpy() if hasattr(Y, 'numpy') else np.asarray(Y)

    def to_tensor(a):
        return a if torch.is_tensor(a) else torch.from_numpy(np.asarray(a))

    em = (pos_np % 2 == 0)
    om = ~em
    out = {}
    with torch.no_grad():
        for parity, mask, model in [("even", em, me), ("odd", om, mo)]:
            if not mask.any():
                continue
            x_p = to_tensor(X)[torch.from_numpy(mask)].float()
            if is_movegrid:                             # 180-d -> 3600-d
                x_p = to_move_grid_input(x_p)
            x_p = x_p.to(device)
            # net = Linear(input,H) -> ReLU -> Linear(H,960)
            h = model.net[1](model.net[0](x_p))
            out[parity] = (h.cpu().numpy().astype(np.float32),
                           Y_np[mask], pos_np[mask])
    return out


def get_ogpt_activations(ckpt_path, layer, eval_path, n_sample,
                         ply_lo=5, ply_hi=54, seed=0, batch=200):
    """Load Othello-GPT, run games through it, return per-parity residual-stream
    activations at `layer` + board labels + ply, mirroring get_mlp_activations.

    Board labels are ABSOLUTE color {0=empty, 1=white, 2=black}; split by ply
    parity so that, within a parity group, class 1/2 IS the mine/yours variable
    (same as the per-parity MLP). One position per game (independent test set).

    `eval_path` is ignored for OGPT (we run real game sequences, not the chunk).
    Returns dict {even: (h, board, pos), odd: (h, board, pos)}.
    """
    import pickle
    from mingpt.model import GPT, GPTConfig
    from experiments.mathematical_transformation_experiments.probe_state_pred_for_othello import (
        extract_activations, tokenize_games, _get_state_stack,
        GAME_LEN, SYNTHETIC_DIR, get_device,
    )
    device = get_device()

    # Rebuild the GPT config from the checkpoint tensor shapes (robust to
    # block_size / n_embd differences), then load.
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
        sd = sd['model']
    vocab, n_embd = sd['tok_emb.weight'].shape
    block_size = sd['pos_emb'].shape[1]
    n_layer = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.'))
    mconf = GPTConfig(vocab, block_size, n_layer=n_layer, n_head=8, n_embd=n_embd)
    model = GPT(mconf); model.load_state_dict(sd); model = model.to(device).eval()
    if layer >= n_layer:
        raise ValueError(f"--layer {layer} but model has {n_layer} layers (0..{n_layer-1})")
    print(f"  OGPT: {n_layer} layers, d={n_embd}, block={block_size}, probing resid_post @ layer {layer}")

    # Load enough synthetic games (one probed position each -> need n_sample games).
    files = sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pickle"))
    games = []
    for fn in files:
        with open(os.path.join(SYNTHETIC_DIR, fn), "rb") as f:
            games.extend(g for g in pickle.load(f) if len(g) == GAME_LEN)
        if len(games) >= n_sample:
            break
    rng = np.random.RandomState(seed)
    rng.shuffle(games)
    games = games[:n_sample]
    print(f"  OGPT: probing {len(games)} games, one ply each in [{ply_lo},{ply_hi})")

    H, B, P = [], [], []
    for s in range(0, len(games), batch):
        gb = games[s:s + batch]
        toks = tokenize_games(gb, seq_len=block_size).to(device)     # (b, block)
        resid = extract_activations(model, toks, layer)               # (b, block, d)
        ss = _get_state_stack(gb, 0, block_size).numpy()              # (b, block, 8, 8) in {-1,0,1}
        for i in range(len(gb)):
            t = int(rng.randint(ply_lo, ply_hi))
            H.append(resid[i, t].detach().cpu().numpy())
            st = ss[i, t].reshape(64)
            # absolute color: 0 empty, 1 white(-1), 2 black(+1)
            B.append(np.where(st == 0, 0, np.where(st == -1, 1, 2)).astype(np.int8))
            P.append(t)

    h = np.stack(H).astype(np.float32)
    board = np.stack(B)
    pos = np.asarray(P, dtype=np.int64)
    out = {}
    for parity, mask in (("even", pos % 2 == 0), ("odd", pos % 2 == 1)):
        if mask.any():
            out[parity] = (h[mask], board[mask], pos[mask])
    return out


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


def _probe_acc(h_train, y_train, h_test, y_test, C=1.0, max_iter=1000,
               nonlinear=False):
    """Train a probe (logistic regression, or an MLP if nonlinear), return test acc.

    Standardizes features (StandardScaler fit on train) first. nonlinear=True
    swaps in a small MLPClassifier so we can compare whether the board variable
    is LINEARLY abstract (transfers under LR) vs only non-linearly abstract.
    """
    if len(h_train) < 20 or len(h_test) < 10:
        return None
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None
    scaler = StandardScaler()
    h_train = scaler.fit_transform(h_train)
    h_test = scaler.transform(h_test)
    if nonlinear:
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=200,
                            early_stopping=True, n_iter_no_change=8,
                            random_state=0)
    else:
        clf = LogisticRegression(max_iter=max_iter, C=C, solver='lbfgs')
    clf.fit(h_train, y_train)
    return float(clf.score(h_test, y_test))


def _cap_joint(H, y, n, rng):
    """Cap a pooled (H, y) sample set to at most n rows (without replacement)."""
    if len(H) <= n:
        return H, y
    sel = rng.choice(len(H), n, replace=False)
    return H[sel], y[sel]


def _subsample(idx, n_target, rng):
    """Without-replacement subsample if too many, else return idx unchanged."""
    if len(idx) <= n_target:
        return idx
    return rng.choice(idx, n_target, replace=False)


def ccgp_phase(h, board, pos, n_bins=4, classes=(1, 2),
               cells=range(64), seed=0, min_per_bin=200,
               train_size=None, nonlinear=False):
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
                a = _probe_acc(h[tr_idx], y[tr_idx], h[te_idx], y[te_idx],
                               nonlinear=nonlinear)
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
                    a = _probe_acc(h[tr], y[tr], h[te], y[te],
                                   nonlinear=nonlinear)
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


def _context_state(board, cell, ctx_mode, rng):
    """Binary per-position context state for cell C. A good board code should
    decode C's own class invariant to this context (small Gap)."""
    if ctx_mode == "context":                       # diagonal-opposite cell occupied
        return (board[:, 63 - cell] != 0).astype(int)
    nei = _NEI[cell]
    if ctx_mode == "crowd":                         # local crowding: occupied-neighbor count
        occ = (board[:, nei] != 0).sum(1)
        return (occ >= np.median(occ)).astype(int)
    if ctx_mode == "frontier":                      # C touches an empty square (frontier vs interior)
        return (board[:, nei] == 0).any(1).astype(int)
    if ctx_mode == "null":                          # random split -> control; CCGP should ~= Within
        return rng.randint(0, 2, size=len(board))
    raise ValueError(f"unknown ctx_mode {ctx_mode}")


def ccgp_context(h, board, pos, classes=(1, 2),
                 cells=range(64), seed=0, min_per_cond=200,
                 train_size=None, ctx_mode="context", nonlinear=False):
    """Cross-context CCGP. For each cell C, split positions by a binary context
    (see _context_state / ctx_mode), train the C-class probe on one context
    state, test on the other (both directions), vs a within-state 2-fold ceiling
    at matched train size. Gap = Within - CCGP; large Gap = context-bound code.

    ctx_mode: 'context' (diagonal cell occupied), 'crowd' (neighbor occupancy),
    'frontier' (C adjacent to empty), 'null' (random split; sanity control)."""
    rng = np.random.RandomState(seed)
    ccgp_per, within_per = [], []
    for cell in cells:
        ctx_state = _context_state(board, cell, ctx_mode, rng)
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
                a = _probe_acc(h[tr_idx], y[tr_idx], h[te_idx], y[te_idx],
                               nonlinear=nonlinear)
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
                    a = _probe_acc(h[tr], y[tr], h[te_part], y[te_part],
                                   nonlinear=nonlinear)
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


def ccgp_spatial(h, board, classes=(1, 2), cells=range(64), seed=0,
                 n_folds=8, cap_per_cell=1500, cap_train=12000, nonlinear=False):
    """Cross-SQUARE CCGP: is there ONE reusable 'is-mine'/'is-yours' coding
    direction shared across board locations, or 64 per-cell codes?

    For each class, pool samples (h_i, board_i[C]==cls) across a set of cells and
    train a SINGLE decoder; leave-one-fold-of-cells out and test on the held-out
    cells. High CCGP => a translation-shared board-state direction (abstract).
    Within = per-cell 2-fold ceiling. Large Gap => cell-specific memorization.

    Note: h_i is shared across the 64 cells of a position, so pooling reuses the
    same activation with different per-cell labels -- that's the point: it forces
    a location-agnostic direction."""
    rng = np.random.RandomState(seed)
    cells = list(cells)
    rng.shuffle(cells)
    folds = [cells[i::n_folds] for i in range(n_folds)]
    N = len(board)
    all_idx = np.arange(N)

    def pool(cell_subset, cls, cap_each):
        Hs, ys = [], []
        for c in cell_subset:
            y = (board[:, c] == cls).astype(np.int32)
            if y.sum() < 30 or (1 - y).sum() < 30:
                continue
            idx = _subsample(_balance(all_idx, y, rng), cap_each, rng)
            Hs.append(h[idx]); ys.append(y[idx])
        if not Hs:
            return None, None
        return np.concatenate(Hs), np.concatenate(ys)

    ccgp_per, within_per = [], []
    for cls in classes:
        fold_acc = []
        for f in range(n_folds):
            test_cells = folds[f]
            train_cells = [c for c in cells if c not in set(folds[f])]
            per_each = max(50, cap_train // max(len(train_cells), 1))
            Htr, ytr = pool(train_cells, cls, per_each)
            Hte, yte = pool(test_cells, cls, cap_per_cell)
            if Htr is None or Hte is None:
                continue
            Htr, ytr = _cap_joint(Htr, ytr, cap_train, rng)
            a = _probe_acc(Htr, ytr, Hte, yte, nonlinear=nonlinear)
            if a is not None:
                fold_acc.append(a)
        if fold_acc:
            ccgp_per.append(np.mean(fold_acc))

        # Within ceiling: per-cell 2-fold at matched per-cell cap.
        wf = []
        for c in cells:
            y = (board[:, c] == cls).astype(np.int32)
            if y.sum() < 30 or (1 - y).sum() < 30:
                continue
            idx = _balance(all_idx, y, rng); rng.shuffle(idx)
            half = len(idx) // 2
            for tr_part, te_part in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
                tr = _subsample(tr_part, cap_per_cell, rng)
                a = _probe_acc(h[tr], y[tr], h[te_part], y[te_part], nonlinear=nonlinear)
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
    p.add_argument("--ccgp-mode",
                   choices=["phase", "context", "crowd", "frontier",
                            "spatial", "null", "both", "all"],
                   default="both",
                   help="phase=game-phase bins; context=diagonal cell; "
                        "crowd/frontier=neighborhood; spatial=leave-cells-out "
                        "shared decoder; null=random-split control; "
                        "both=phase+context; all=every mode incl. null.")
    p.add_argument("--nonlinear", action="store_true",
                   help="Use an MLP probe instead of logistic regression "
                        "(tests non-linear vs linear abstractness).")
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
        per_parity = get_ogpt_activations(
            args.ogpt_ckpt, args.layer, args.eval_chunk, args.n)

    print(f"\nLoaded activations:")
    for parity, (h, board, pos) in per_parity.items():
        print(f"  {parity}: {len(h)} positions, hidden_dim={h.shape[1]}, "
              f"turn_range=[{int(pos.min())}, {int(pos.max())}]")

    for parity, (h, board, pos) in per_parity.items():
        print(f"\n========================================")
        print(f"  Parity: {parity}")
        print(f"========================================")

        m = args.ccgp_mode
        nl = args.nonlinear
        tag = " [NONLINEAR probe]" if nl else " [linear probe]"

        if m in ("phase", "both", "all"):
            r = ccgp_phase(h, board, pos, n_bins=args.n_bins, nonlinear=nl)
            _print_summary(f"phase: cross-game-phase ({args.n_bins} bins){tag}", r)

        if m in ("context", "both", "all"):
            r = ccgp_context(h, board, pos, ctx_mode="context", nonlinear=nl)
            _print_summary(f"context: cross-context-cell (diagonal){tag}", r)

        if m in ("crowd", "all"):
            r = ccgp_context(h, board, pos, ctx_mode="crowd", nonlinear=nl)
            _print_summary(f"crowd: neighbor-occupancy context{tag}", r)

        if m in ("frontier", "all"):
            r = ccgp_context(h, board, pos, ctx_mode="frontier", nonlinear=nl)
            _print_summary(f"frontier: C-adjacent-to-empty context{tag}", r)

        if m in ("spatial", "all"):
            r = ccgp_spatial(h, board, nonlinear=nl)
            _print_summary(f"spatial: leave-cells-out shared decoder{tag}", r)

        if m in ("null", "all"):
            r = ccgp_context(h, board, pos, ctx_mode="null", nonlinear=nl)
            _print_summary(f"null: RANDOM-split control (CCGP~=Within expected){tag}", r)


if __name__ == "__main__":
    main()
