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

# 60 move-cells (all squares except the 4 center) -> board-cell indices.
_VALID_MOVES = [c for c in range(64) if c not in (27, 28, 35, 36)]


def _placement_from_raw(Xr):
    """From raw 180-d features [played(60), when(60), even(60)] derive, per BOARD
    cell (64): placement color and placement step.

    Encoding (train_pattern_simple): move_num = round(when*60 - 1); even==1 means
    placed by WHITE (step 0 = white). Board-label convention is 0=empty, 1=white,
    2=black, so place_color uses the same {1=white, 2=black, 0=unplayed}.

    Returns (place_color (N,64) int8, place_step (N,64) int16 with -1 = unplayed).
    Center cells (27,28,35,36) are always unplayed here (initial discs, no 'when').
    """
    played = Xr[:, 0:60] > 0.5
    when = Xr[:, 60:120]
    even = Xr[:, 120:180] > 0.5
    move_num = np.clip(np.round(when * 60.0 - 1.0), 0, 59).astype(np.int16)
    col60 = np.where(played, np.where(even, 1, 2), 0).astype(np.int8)     # 1 white, 2 black
    step60 = np.where(played, move_num, -1).astype(np.int16)
    N = Xr.shape[0]
    place_color = np.zeros((N, 64), np.int8)
    place_step = np.full((N, 64), -1, np.int16)
    place_color[:, _VALID_MOVES] = col60
    place_step[:, _VALID_MOVES] = step60
    return place_color, place_step


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
    Xr = X_raw.numpy() if hasattr(X_raw, 'numpy') else np.asarray(X_raw)
    place_color, place_step = _placement_from_raw(Xr)   # (N,64) each
    if not is_movegrid:
        X = X_raw[:, _feature_cols(features)]           # column-slice reps
    else:
        X = X_raw                                       # transform per-batch below
    if n_sample is not None and n_sample < len(X):
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(len(X), n_sample, replace=False))
        X, Y, pos = X[idx], Y[idx], pos[idx]
        place_color, place_step = place_color[idx], place_step[idx]

    pos_np = pos.numpy() if hasattr(pos, 'numpy') else np.asarray(pos)
    Y_np = Y.numpy() if hasattr(Y, 'numpy') else np.asarray(Y)

    def to_tensor(a):
        return a if torch.is_tensor(a) else torch.from_numpy(np.asarray(a))

    em = (pos_np % 2 == 0)
    om = ~em
    out, aux = {}, {}
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
            aux[parity] = (place_color[mask], place_step[mask])
    return out, aux


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

    H, B, P, PC, PS = [], [], [], [], []
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
            # placement info from the move sequence (black first: step 0 = black).
            pc = np.zeros(64, np.int8); ps = np.full(64, -1, np.int16)
            for j in range(t + 1):
                c = gb[i][j]
                pc[c] = 2 if (j % 2 == 0) else 1        # black(2) on even steps, white(1) on odd
                ps[c] = j
            PC.append(pc); PS.append(ps)

    h = np.stack(H).astype(np.float32)
    board = np.stack(B)
    pos = np.asarray(P, dtype=np.int64)
    place_color = np.stack(PC); place_step = np.stack(PS)
    out, aux = {}, {}
    for parity, mask in (("even", pos % 2 == 0), ("odd", pos % 2 == 1)):
        if mask.any():
            out[parity] = (h[mask], board[mask], pos[mask])
            aux[parity] = (place_color[mask], place_step[mask])
    return out, aux


# ---------------------------------------------------------------------------
# CCGP probes
# ---------------------------------------------------------------------------

def sample_shared_positions(ogpt_ckpt, layer, n_sample, ply_lo=5, ply_hi=54,
                            seed=0, batch=200):
    """Sample (game, ply) positions ONCE and compute the shared, expensive parts:
    the MLP's 180-d features, the board label, placement aux, and the OGPT
    residual at `layer` -- all on the SAME positions. Reuse the returned dict
    across many MLPs (each cheap) via mlp_from_sample() and ogpt_from_sample().

    One probed ply per game (independent). Returns a dict of numpy arrays."""
    import pickle
    from mingpt.model import GPT, GPTConfig
    from experiments.mathematical_transformation_experiments.probe_state_pred_for_othello import (
        extract_activations, tokenize_games, _get_state_stack,
        GAME_LEN, SYNTHETIC_DIR, get_device,
    )
    device = get_device()
    B64_TO_M60 = {c: i for i, c in enumerate(_VALID_MOVES)}

    sd = torch.load(ogpt_ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
        sd = sd['model']
    vocab, n_embd = sd['tok_emb.weight'].shape
    block = sd['pos_emb'].shape[1]
    n_layer = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.'))
    gpt = GPT(GPTConfig(vocab, block, n_layer=n_layer, n_head=8, n_embd=n_embd))
    gpt.load_state_dict(sd); gpt = gpt.to(device).eval()
    if layer >= n_layer:
        raise ValueError(f"--layer {layer} but OGPT has {n_layer} layers")

    files = sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pickle"))
    games = []
    for fn in files:
        with open(os.path.join(SYNTHETIC_DIR, fn), "rb") as f:
            games.extend(g for g in pickle.load(f) if len(g) == GAME_LEN)
        if len(games) >= n_sample:
            break
    rng = np.random.RandomState(seed)
    rng.shuffle(games); games = games[:n_sample]
    nmoves = rng.randint(ply_lo, ply_hi, size=len(games))
    N = len(games)
    print(f"  shared sample: {N} games from {len(files)} pickle(s), one ply each in [{ply_lo},{ply_hi}); OGPT L{layer}/{n_layer}", flush=True)

    feat = np.zeros((N, 180), np.float32)
    board = np.zeros((N, 64), np.int8)
    pc = np.zeros((N, 64), np.int8); ps = np.full((N, 64), -1, np.int16)
    pos = np.asarray(nmoves, dtype=np.int64)
    ogpt_h = np.zeros((N, n_embd), np.float32)

    for s0 in range(0, N, batch):
        gb = games[s0:s0 + batch]
        toks = tokenize_games(gb, seq_len=block).to(device)
        resid = extract_activations(gpt, toks, layer)                # (b, block, d)
        ss = _get_state_stack(gb, 0, block).numpy()                  # (b, block, 8, 8)
        for i in range(len(gb)):
            gi = s0 + i; t = int(nmoves[gi])
            for step in range(t):
                c = gb[i][step]; j = B64_TO_M60[c]
                feat[gi, j] = 1.0                                    # played
                feat[gi, 60 + j] = (step + 1) / 60.0                 # when
                feat[gi, 120 + j] = 1.0 if step % 2 == 0 else 0.0    # even (step-parity)
                ps[gi, c] = step; pc[gi, c] = 2 if step % 2 == 0 else 1
            st = ss[i, t - 1].reshape(64)                            # board after t moves
            board[gi] = np.where(st == 0, 0, np.where(st == -1, 1, 2)).astype(np.int8)
            ogpt_h[gi] = resid[i, t - 1].detach().cpu().numpy()
        if s0 % (batch * 25) == 0:
            print(f"    ...{min(s0 + batch, N)}/{N} positions", flush=True)

    return dict(feat=feat, board=board, pos=pos, place_color=pc, place_step=ps,
                ogpt_h=ogpt_h, n_embd=n_embd, games=games)


def _split_parity(h, sample):
    """Wrap an (N, d) activation array + a sample dict into (per_parity, aux)."""
    board, pos, pc, ps = sample['board'], sample['pos'], sample['place_color'], sample['place_step']
    out, aux = {}, {}
    for parity, mask in (("even", pos % 2 == 0), ("odd", pos % 2 == 1)):
        if mask.any():
            out[parity] = (h[mask], board[mask], pos[mask])
            aux[parity] = (pc[mask], ps[mask])
    return out, aux


def ogpt_from_sample(sample):
    """(per_parity, aux) for the OGPT residual already in the sample."""
    return _split_parity(sample['ogpt_h'], sample)


def mlp_from_sample(sample, mlp_ckpt, mlp_hidden, mlp_features):
    """Run one MLP on the sampled positions -> (per_parity, aux). Cheap; reuse
    the same `sample` across many MLPs."""
    from train_pattern_simple import DirectMLP, to_move_grid_input
    from experiments.mathematical_transformation_experiments.probe_state_pred_for_othello import get_device
    device = get_device()
    ck = torch.load(mlp_ckpt, map_location=device)
    is_mg = (mlp_features == "move_grid") or (ck.get('input_dim') == 3600)
    input_dim = ck.get('input_dim', 3600 if is_mg else len(_feature_cols(mlp_features)))
    me = DirectMLP(input_dim, mlp_hidden, ck.get('n_patterns', 960)).to(device)
    mo = DirectMLP(input_dim, mlp_hidden, ck.get('n_patterns', 960)).to(device)
    me.load_state_dict(ck['even']); mo.load_state_dict(ck['odd']); me.eval(); mo.eval()

    feat = sample['feat']
    X = to_move_grid_input(torch.from_numpy(feat)) if is_mg else torch.from_numpy(feat[:, _feature_cols(mlp_features)])
    pos = sample['pos']
    h_full = np.zeros((len(pos), mlp_hidden), np.float32)
    with torch.no_grad():
        for parity, mask in (("even", pos % 2 == 0), ("odd", pos % 2 == 1)):
            if not mask.any():
                continue
            model = me if parity == "even" else mo
            xp = X[torch.from_numpy(mask)].float().to(device)
            h_full[mask] = model.net[1](model.net[0](xp)).cpu().numpy().astype(np.float32)
    return _split_parity(h_full, sample)


def j1b_from_sample(sample, bank, flanking_patterns, svd_k=2048, batch=1024, seed=0):
    """J1B (interpretable tree-bank) representation on the shared positions:
    the ~47k tree-leaf one-hot (OpeningTreeMLP over 121-d mover-canonicalized
    playedeven features), built SPARSE then TruncatedSVD-reduced to svd_k dims so
    CCGP's per-fold fits are tractable. SVD is a linear reprojection, so the
    linear-probe CCGP is preserved when svd_k captures the board-relevant
    variance -- validate via Within (~J1B's board-decode ceiling, ~0.90). If
    Within is low, raise svd_k.

    Returns (per_parity, aux) on the SAME positions as the rest of the sample."""
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    import train_streaming_probe as tsp
    from opening_tree_mlp import playedeven_features
    from experiments.mathematical_transformation_experiments.probe_state_pred_for_othello import get_device
    device = get_device()

    W_tree, b_tree, meta = tsp.load_trees(bank)
    mlp = tsp.OpeningTreeMLP(W_tree, b_tree, meta, device)
    leaf_build = tsp.load_leaf_build(bank)                 # None for binary J1 bank
    # no_flanking=True -> flanking patterns are unused; load only if present.
    patterns = tsp.load_patterns(flanking_patterns) if os.path.exists(flanking_patterns) else None
    n_leaves = int(mlp.W.shape[0])
    games, pos = sample['games'], sample['pos']
    N = len(games)
    print(f"  J1B: bank leaves={n_leaves}, building leaf one-hots for {N} positions ...", flush=True)

    rows, cols = [], []
    for s0 in range(0, N, batch):
        gg = games[s0:s0 + batch]; tt = pos[s0:s0 + batch]
        X = np.stack([playedeven_features(g[:int(t)], canonicalize_mover=True)
                      for g, t in zip(gg, tt)]).astype(np.float32)
        H = tsp.build_hidden_layer_batch(X, mlp, patterns, None, False, device,
                                         no_flanking=True, leaf_build=leaf_build,
                                         leaf_index=None)
        r, c = np.nonzero(H.cpu().numpy())
        rows.append(r + s0); cols.append(c)
        if s0 % (batch * 20) == 0:
            print(f"    ...{min(s0 + batch, N)}/{N}", flush=True)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    Hs = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(N, n_leaves))
    print(f"  J1B: sparse leaf matrix {Hs.shape}, {Hs.nnz} nnz ({Hs.nnz/N:.0f}/row); "
          f"TruncatedSVD -> {svd_k}", flush=True)
    svd = TruncatedSVD(n_components=min(svd_k, n_leaves - 1), random_state=seed)
    h_red = svd.fit_transform(Hs).astype(np.float32)
    print(f"  J1B: SVD explained variance = {svd.explained_variance_ratio_.sum():.3f}", flush=True)
    return _split_parity(h_red, sample)


def get_shared_activations(mlp_ckpt, mlp_hidden, mlp_features, ogpt_ckpt, layer,
                           n_sample, ply_lo=5, ply_hi=54, seed=0, batch=200):
    """Single MLP-vs-OGPT pair on identical positions (composes the pieces).
    Returns {'mlp': (per_parity, aux), 'ogpt': (per_parity, aux)}."""
    sample = sample_shared_positions(ogpt_ckpt, layer, n_sample, ply_lo, ply_hi, seed, batch)
    return {'mlp': mlp_from_sample(sample, mlp_ckpt, mlp_hidden, mlp_features),
            'ogpt': ogpt_from_sample(sample)}


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


def ccgp_conditioned(h, board, pos, place_color, place_step, cond="flip",
                     cells=None, seed=0, min_per_cond=150, train_size=None,
                     nonlinear=False):
    """Flip / recency CCGP, conditioned on cell C being OCCUPIED.

    Target = C's color dichotomy (white vs black; within a parity that IS
    mine vs yours). Context split:
      flip     : current color != placement color (net-flipped) vs never-flipped
                 -> does 'C is mine' transfer across whether C was just captured?
                 A Markov state code should (small Gap).
      recency  : moves-since-placement >= median (long-settled) vs recent
                 -> do long-settled squares decode like fresh ones?

    Only non-center cells (the 60 move-cells have placement info)."""
    rng = np.random.RandomState(seed)
    cells = list(cells) if cells is not None else list(_VALID_MOVES)
    ccgp_per, within_per = [], []
    for cell in cells:
        occ = np.where(board[:, cell] != 0)[0]
        if len(occ) < 2 * min_per_cond:
            continue
        y = (board[occ, cell] == 1).astype(np.int32)          # white(1) vs black(0)
        if y.sum() < 60 or (1 - y).sum() < 60:
            continue
        if cond == "flip":
            ctx = (board[occ, cell] != place_color[occ, cell]).astype(int)   # 1=flipped
        elif cond == "recency":
            rec = pos[occ].astype(np.int32) - place_step[occ, cell].astype(np.int32)
            ctx = (rec >= np.median(rec)).astype(int)          # 1=long-settled
        else:
            raise ValueError(f"unknown cond {cond}")
        hocc = h[occ]

        per_state = []
        for s in (0, 1):
            si = np.where(ctx == s)[0]
            if len(si) < min_per_cond:
                per_state.append(None); continue
            per_state.append(_balance(si, y, rng))
        if any(p is None for p in per_state):
            continue

        ccgp_pool = min(len(per_state[0]), len(per_state[1]))
        within_pool = min(len(per_state[0]) // 2, len(per_state[1]) // 2)
        t = train_size or min(ccgp_pool, within_pool)
        t = max(t, 50)

        fold_acc = []
        for held in (0, 1):
            tr = _subsample(per_state[1 - held], t, rng)
            te = per_state[held]
            a = _probe_acc(hocc[tr], y[tr], hocc[te], y[te], nonlinear=nonlinear)
            if a is not None:
                fold_acc.append(a)
        if fold_acc:
            ccgp_per.append(np.mean(fold_acc))

        wf = []
        for s in (0, 1):
            si = per_state[s].copy(); rng.shuffle(si)
            half = len(si) // 2
            for tr_part, te_part in [(si[:half], si[half:]), (si[half:], si[:half])]:
                tr = _subsample(tr_part, t, rng)
                a = _probe_acc(hocc[tr], y[tr], hocc[te_part], y[te_part], nonlinear=nonlinear)
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


def run_modes(per_parity, aux, args, model_label=""):
    """Run the selected CCGP modes for one model's per-parity activations.

    Prints a summary per (mode, parity) and RETURNS {mode: {parity: res}} so a
    driver can tabulate Gaps across models."""
    hdr = f" [{model_label}]" if model_label else ""
    print(f"\nLoaded activations{hdr}:")
    for parity, (h, board, pos) in per_parity.items():
        print(f"  {parity}: {len(h)} positions, hidden_dim={h.shape[1]}, "
              f"turn_range=[{int(pos.min())}, {int(pos.max())}]")

    m, nl = args.ccgp_mode, args.nonlinear
    tag = (f"{hdr} " if hdr else " ") + ("[NONLINEAR probe]" if nl else "[linear probe]")
    wanted = (["phase", "context", "crowd", "frontier", "spatial", "flip", "recency", "null"]
              if m == "all" else ["phase", "context"] if m == "both" else [m])

    def compute(mode, h, board, pos, parity):
        if mode == "phase":    return ccgp_phase(h, board, pos, n_bins=args.n_bins, nonlinear=nl)
        if mode == "context":  return ccgp_context(h, board, pos, ctx_mode="context", nonlinear=nl)
        if mode == "crowd":    return ccgp_context(h, board, pos, ctx_mode="crowd", nonlinear=nl)
        if mode == "frontier": return ccgp_context(h, board, pos, ctx_mode="frontier", nonlinear=nl)
        if mode == "spatial":  return ccgp_spatial(h, board, nonlinear=nl)
        if mode == "null":     return ccgp_context(h, board, pos, ctx_mode="null", nonlinear=nl)
        pc, ps = aux[parity]
        return ccgp_conditioned(h, board, pos, pc, ps, cond=mode, nonlinear=nl)

    labels = {"phase": f"phase: cross-game-phase ({args.n_bins} bins)",
              "context": "context: diagonal cell", "crowd": "crowd: neighbor-occupancy",
              "frontier": "frontier: C-adjacent-to-empty", "spatial": "spatial: leave-cells-out shared",
              "flip": "flip: net-flipped vs never-flipped", "recency": "recency: long-settled vs recent",
              "null": "null: RANDOM-split control (CCGP~=Within)"}

    results = {mode: {} for mode in wanted}
    for parity, (h, board, pos) in per_parity.items():
        print(f"\n======================================== {model_label} parity={parity}")
        for mode in wanted:
            res = compute(mode, h, board, pos, parity)
            results[mode][parity] = res
            _print_summary(f"{labels[mode]}{tag}", res)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None,
                   help="Path to MLP checkpoint (mode-data mlp/shared).")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--features", default="wheneven")
    p.add_argument("--mode-data", choices=["mlp", "ogpt", "shared"], default="mlp")
    p.add_argument("--ogpt-ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--n", type=int, default=30000,
                   help="positions to sample from the eval chunk")
    p.add_argument("--ccgp-mode",
                   choices=["phase", "context", "crowd", "frontier",
                            "spatial", "flip", "recency", "null", "both", "all"],
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

    if args.mode_data == "shared":
        if args.ckpt is None:
            p.error("--ckpt (MLP) is required when --mode-data shared")
        shared = get_shared_activations(
            args.ckpt, args.hidden, args.features, args.ogpt_ckpt, args.layer, args.n)
        run_modes(*shared['mlp'], args, model_label="MLP")
        run_modes(*shared['ogpt'], args, model_label="OGPT")
        return

    if args.mode_data == "mlp":
        if args.ckpt is None:
            p.error("--ckpt is required when --mode-data mlp")
        if args.eval_chunk is None:
            chunk_dir = ("experiments/mathematical_transformation_experiments/"
                         "heuristic_probe_results/feature_chunks")
            files = sorted(f for f in os.listdir(chunk_dir)
                           if f.endswith(".npz") and "_patterns" not in f)
            args.eval_chunk = os.path.join(chunk_dir, files[-1])
        print(f"Eval chunk: {args.eval_chunk}")
        per_parity, aux = get_mlp_activations(
            args.ckpt, args.hidden, args.features, args.eval_chunk, args.n)
        run_modes(per_parity, aux, args, model_label="MLP")
    else:
        per_parity, aux = get_ogpt_activations(
            args.ogpt_ckpt, args.layer, args.eval_chunk, args.n)
        run_modes(per_parity, aux, args, model_label="OGPT")


if __name__ == "__main__":
    main()
