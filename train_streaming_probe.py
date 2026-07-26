"""Streaming trainer for legal-move probes on large-scale pre-generated
game data.

The in-memory pipeline (midgame_tree_mlp.py) loads all sampled positions
at once, which caps us at ~6M games (~240M positions, ~130 GB in the
hidden layer even under bool step activation).  For matching the MLP's
6M-game training budget we don't strictly need streaming, but for going
larger AND for the bigger pattern-tree hidden layer (~48K units), we do.

Design (mirrors train_pattern_simple.py's chunk-streaming pattern):

  1. Load trees from a --load-trees-from checkpoint ONCE.
  2. Load hand-crafted flanking-pattern definitions ONCE.
  3. Iterate through --pickle-dir/*.pickle in a random order per epoch.
  4. Per pickle:
     - Load ~100K games from disk (~10 sec).
     - Extract midgame positions in [ply_min, ply_max).
     - Compute played+even+mover_parity features + legal-move mask +
       optional recent bits.
     - Apply the loaded trees to get tree-path activations.
     - Concatenate: [tree_paths, recent_bits, flanking_patterns] → H.
     - Iterate mini-batches, forward+backward on the probe.
     - Discard the chunk.
  5. Optional per-epoch eval on the reserved last pickle.
"""
import argparse
import glob
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# numpy 2.x (e.g. a RunPod pod) pickles arrays under `numpy._core`; numpy 1.x
# (the cluster's py3.8 env) has no such module, so torch.load of a cross-fit
# bank raises ModuleNotFoundError.  Alias the renamed modules so old numpy can
# unpickle new-numpy banks.  No-op when numpy already has _core.
if not hasattr(np, '_core'):
    import importlib as _il
    try:
        sys.modules['numpy._core'] = _il.import_module('numpy.core')
        for _s in ('multiarray', 'numeric', 'umath', 'overrides',
                    '_multiarray_umath', 'fromnumeric', '_methods'):
            try:
                sys.modules['numpy._core.' + _s] = _il.import_module('numpy.core.' + _s)
            except Exception:
                pass
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# OthelloBoardState is only needed for the pickle-replay path (process_pickle_
# chunk); import it lazily there so the chunk-ext / leaf-cache path doesn't drag
# in data.othello's deps (e.g. `pgn`), which a minimal pod may not have.
from opening_tree_mlp import (
    playedeven_features, LinearPatternProbOr, PatternProbOrHead,
    OpeningTreeMLP, BOARD_CELLS, C64_TO_C60,
)
from flanking_patterns import (
    load_patterns, compute_pattern_activations, patterns_by_target,
    true_pattern_activations,
)


# --------------------------------------------------------------------------
# chunk_ext_*.npz fast path: no OthelloBoardState replay
# --------------------------------------------------------------------------
# The 40 chunk_ext_*.npz files under
#   experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks/
# hold ~24M pre-simulated games' worth of positions, with:
#   features  (N, 180) float16 — [played(60), when(60), even(60)]
#   labels    (N, 64)  int8    — board state, {0=empty, 1=white, 2=black}
#   positions (N,)     uint8   — turn index (state AFTER move t)
# The chunk position convention differs from the streaming/pickle convention
# by exactly 1: chunk position t == streaming position t+1 (since streaming
# extracts features "before" the move, chunk stores "after" the move).

_PATTERN_ARRAYS_CACHE = None


def _get_pattern_arrays():
    global _PATTERN_ARRAYS_CACHE
    if _PATTERN_ARRAYS_CACHE is None:
        from hand_crafted_flanking import enumerate_flanking_patterns
        from generate_rule_games import precompute_pattern_arrays
        pats = enumerate_flanking_patterns()
        _PATTERN_ARRAYS_CACHE = precompute_pattern_arrays(pats)
    return _PATTERN_ARRAYS_CACHE


def process_chunk_ext_file(chunk_path, ply_min, ply_max,
                              canonicalize_mover=False,
                              max_positions=None,
                              pat_batch=200_000,
                              needs_ordinal=False):
    """Load one chunk_ext_*.npz file, return (X, S, T, L) as np arrays.

    X: (N, 121) float32 — [played(60), even-or-placed_as_mover(60), mp(1)].
    S: (N, 64)  int64   — mover-relative state labels (1=mine, 2=opp).
    T: (N,)     int32   — streaming-convention ply (chunk_position + 1).
    L: (N, 64)  uint8   — legal-move mask derived from 960 flanking patterns.

    max_positions: hard cap on rows returned; useful for warmup / small
                    trial runs.  Naive first-N slice after ply filtering.
    pat_batch: micro-batch size for compute_pattern_labels_batch to keep
               the (batch, 960) float32 buffer under a few GB.  200K rows
               × 960 × 4 = ~750 MB per batch.
    """
    from train_pattern_simple import compute_pattern_labels_batch

    z = np.load(chunk_path)
    positions = z['positions'].astype(np.int64)
    # chunk position t <=> streaming position t + 1
    stream_lo = ply_min
    stream_hi = ply_max
    chunk_lo = stream_lo - 1
    chunk_hi = stream_hi - 1
    mask = (positions >= chunk_lo) & (positions < chunk_hi)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        z.close()
        return None, None, None, None
    if max_positions is not None and len(idx) > max_positions:
        # chunk_ext rows are stored in position-major order (all games at
        # position 5, then all at 6, ...); a plain first-N slice would give
        # a single-ply subset.  Shuffle first so a capped trial sees the
        # full ply distribution.
        rng = np.random.RandomState(0)
        idx = rng.choice(idx, size=max_positions, replace=False)
        idx.sort()  # sorted access is faster for the subsequent npz reads

    positions = positions[idx]
    features = np.asarray(z['features'][idx]).astype(np.float32)
    labels = np.asarray(z['labels'][idx]).astype(np.int8)
    z.close()

    stream_pos = (positions + 1).astype(np.int32)
    played = features[:, :60]                    # (N, 60)
    even = features[:, 120:180]                  # (N, 60)

    N = len(features)
    X = np.zeros((N, 121), dtype=np.float32)
    X[:, :60] = played
    mover_parity = (stream_pos % 2).astype(np.float32)  # (N,)
    if canonicalize_mover:
        # placed_as_mover: cell placed by whichever side is now to move.
        # Cell was placed at even step  <=>  even[c] == 1  <=>  step parity 0.
        # Streaming position T is played by parity T % 2, so mover_parity = T % 2.
        # placed_as_mover = played AND (step_parity == mover_parity)
        # step_parity == 0  <=>  even == 1  <=>  1 - even == mover_parity when mp == 0
        # Equivalently: placed_as_mover = played * (even == (1 - mover_parity))
        target_even = (1.0 - mover_parity)[:, None]      # (N, 1)
        placed_as_mover = played * (even == target_even)
        X[:, 60:120] = placed_as_mover
        # feat[120] stays 0 for canonical (parity baked into cell bits).
    else:
        X[:, 60:120] = even
        X[:, 120] = mover_parity

    # 960-d pattern legality via the vectorized numpy path — computed in
    # sub-batches to avoid allocating the full (N, 960) float32 tensor
    # (which is ~92 GB for a 24M-row chunk).  Reduce to 64-d L on the fly.
    targets, terminals, opp_cells, opp_mask = _get_pattern_arrays()
    per_cell_pat_ids = [np.where(targets == c)[0] for c in range(BOARD_CELLS)]
    L = np.zeros((N, BOARD_CELLS), dtype=np.uint8)
    for start in range(0, N, pat_batch):
        end = min(start + pat_batch, N)
        pat = compute_pattern_labels_batch(
            labels[start:end], positions[start:end],
            targets, terminals, opp_cells, opp_mask)
        for c in range(BOARD_CELLS):
            pids = per_cell_pat_ids[c]
            if len(pids) > 0:
                L[start:end, c] = (pat[:, pids] > 0).any(axis=1).astype(np.uint8)
        del pat

    # J3 ordinal: append the reconstructed movesago block (cols 121:181).
    # movesago = positions + 2 - when*60  (played cells; -1 unplayed).
    # Validated to match playedeven_features(time_ordinal='movesago') exactly.
    if needs_ordinal:
        when = features[:, 60:120].astype(np.float32)
        # movesago is an integer; round to undo float16 storage error in `when`
        # so it matches the exact-integer values the trees were fit on.
        movesago = np.where(played > 0,
                            np.round(positions[:, None] + 2.0 - when * 60.0),
                            -1.0).astype(np.float32)
        X = np.concatenate([X, movesago], axis=1)     # (N, 181)

    # Return the board-state labels as S (int8, ~1.8 GB) — option B needs them
    # to compute per-batch pattern targets; option A ignores S.
    return X, labels, stream_pos, L


def _chunk_cache_path(cache_dir, path, ply_min, ply_max, canon, cap):
    base = os.path.basename(path)
    cap_s = 'all' if cap is None else str(int(cap))
    return os.path.join(
        cache_dir,
        f'{base}.ply{ply_min}-{ply_max}.cap{cap_s}.'
        f'canon{int(bool(canon))}.v1.npz')


def load_chunk_cached(path, ply_min, ply_max, canonicalize_mover, cap,
                       needs_ordinal, cache_dir=None):
    """Return (X, S, T, L) for a chunk_ext file, backed by an on-disk cache.

    The cache stores the ALWAYS-181-d input (base 121 + movesago 60), so a
    SINGLE cache file serves both base (J0/J1/J2 — X sliced to 121) and ordinal
    (J3 — full 181) configs, and every epoch after the first is a fast npz read
    instead of a re-load + 960-pattern recompute.  The expensive part
    (np.load + compute_pattern_labels_batch for L) happens once per
    (chunk, ply-range, cap, canon).

    Written atomically (os.replace), so parallel configs racing on the same key
    never observe a half-written file; a corrupt/partial read just rebuilds.
    """
    cpath = (None if not cache_dir else
             _chunk_cache_path(cache_dir, path, ply_min, ply_max,
                               canonicalize_mover, cap))
    X = S = T = L = None
    if cpath and os.path.exists(cpath):
        try:
            z = np.load(cpath)
            X = z['X'].astype(np.float32)
            S = z['S'].astype(np.int8)
            T = z['T'].astype(np.int32)
            L = z['L'].astype(np.uint8)
            z.close()
        except Exception as e:                     # partial/corrupt → rebuild
            print(f'  [cache] {os.path.basename(cpath)} unreadable ({e}); '
                   f'rebuilding', flush=True)
            X = S = T = L = None
    if X is None:
        # Build the movesago block (needs_ordinal=True → 181-d) whenever we
        # need it (ordinal config) OR we're persisting a cache — so the cached
        # file is method-agnostic (serves base and ordinal alike).  When we are
        # neither caching nor ordinal, build the lean 121-d directly (old
        # memory profile).  All X columns are integer-valued (0/1 bits +
        # integer movesago in [-1, ~53]), so int8 storage is lossless; cast
        # back to float32 on read.
        build_ordinal = needs_ordinal or bool(cpath)
        X, S, T, L = process_chunk_ext_file(
            path, ply_min, ply_max, canonicalize_mover=canonicalize_mover,
            max_positions=cap, needs_ordinal=build_ordinal)
        if X is None:
            return None, None, None, None
        if cpath:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = f'{cpath}.tmp{os.getpid()}'
            with open(tmp, 'wb') as fh:
                np.savez(fh,
                         X=X.astype(np.int8),
                         S=np.asarray(S, dtype=np.int8),
                         T=np.asarray(T, dtype=np.int16),
                         L=np.asarray(L, dtype=np.uint8))
            os.replace(tmp, cpath)                 # atomic
    if not needs_ordinal:
        # Base configs don't use the movesago block — drop it (contiguous copy
        # so the 181-d buffer is freed).
        X = np.ascontiguousarray(X[:, :121])
    return X, S, T, L


def load_leaf_index_chunk(leaf_cache_dir, path, ply_min, ply_max, cap):
    """J3 fast path: return (leaves, S, T, L) where `leaves` is the prebuilt
    (N, n_trees) int16 leaf-id matrix (from build_leaf_cache.py) in place of X.
    `leaves` is read fully into RAM (random per-batch indexing over an NFS
    memmap would be slow).  S/T/L come from the .stl.npz saved alongside."""
    base = os.path.basename(path)
    stem = f'{base}.ply{ply_min}-{ply_max}.cap{cap}.leaves.i16'
    lpath = os.path.join(leaf_cache_dir, stem)
    meta = np.load(lpath + '.meta.npz')
    N = int(meta['N']); nt = int(meta['n_trees'])
    leaves = np.fromfile(lpath, dtype=np.int16, count=N * nt).reshape(N, nt)
    stl = np.load(lpath + '.stl.npz')
    return leaves, stl['S'], stl['T'].astype(np.int32), stl['L']


def process_pickle_chunk(pickle_path, ply_min, ply_max, recent_Ks=None):
    """Load one pickle file, replay each game, extract midgame positions.

    Returns (X, S, T, L, played, even, mp) numpy arrays.
    """
    from data.othello import OthelloBoardState   # lazy: pickle path only
    with open(pickle_path, 'rb') as f:
        games = pickle.load(f)
    Xs, Ss, Ts, Ls = [], [], [], []
    for game_moves in games:
        board = OthelloBoardState()
        prefix = []
        for move in game_moves:
            valid = board.get_valid_moves()
            if not valid:
                board.update([])
                valid = board.get_valid_moves()
                if not valid:
                    break
            ply = len(prefix)
            if ply_min <= ply < ply_max:
                parity = ply % 2
                mover_color = 1 if parity == 0 else -1
                raw = board.state.flatten().astype(np.int8)
                lbl = np.zeros(BOARD_CELLS, dtype=np.int64)
                lbl[raw == mover_color] = 1
                lbl[raw == -mover_color] = 2
                Xs.append(playedeven_features(prefix, recent_Ks=recent_Ks))
                Ss.append(lbl)
                Ts.append(ply)
                lmask = np.zeros(BOARD_CELLS, dtype=np.uint8)
                for m in valid:
                    lmask[m] = 1
                Ls.append(lmask)
            if move not in valid:
                break
            board.update([move])
            prefix.append(move)
    if not Xs:
        return None, None, None, None
    X = np.stack(Xs).astype(np.float32)
    S = np.stack(Ss)
    T = np.array(Ts, dtype=np.int32)
    L = np.stack(Ls)
    return X, S, T, L


def load_trees(ckpt_path):
    print(f'loading trees from {ckpt_path}...')
    # weights_only=False: our banks embed numpy arrays / sklearn objects; on
    # PyTorch 2.6 the default weights_only=True rejects them.  Trusted checkpoint.
    try:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location='cpu')
    W = ck['W']; b = ck['b']; meta = ck['path_info']
    if isinstance(W, torch.Tensor):
        W = W.numpy()
    if isinstance(b, torch.Tensor):
        b = b.numpy()
    tree_idx = [i for i, m in enumerate(meta)
                 if m.get('kind') in ('tree_path', 'pattern_path',
                                       'pattern_multi')]
    W_tree = W[tree_idx]
    b_tree = b[tree_idx]
    tree_meta = [meta[i] for i in tree_idx]
    print(f'  {len(tree_meta)} tree paths; input_dim={W_tree.shape[1]}')
    return W_tree, b_tree, tree_meta


def load_leaf_build(ckpt_path):
    """For a --hidden-from-leaves bank (J3 ordinal): return leaf_build =
    [(sklearn_tree, node_id), ...] so H can be built via tree.apply (the ±1 W is
    invalid for numeric splits).  Returns None if the bank isn't leaf-based."""
    try:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location='cpu')
    if not ck.get('hidden_from_leaves') or ck.get('sklearn_trees') is None:
        return None
    trees = ck['sklearn_trees']
    return [(trees[ti], nid) for ti, nid in ck['leaf_build_idx']]


def build_H_from_leaves_np(leaf_build, Xnp):
    """TRUE leaf one-hot via tree.apply (honors numeric ordinal splits)."""
    from collections import defaultdict
    N = Xnp.shape[0]
    H = np.zeros((N, len(leaf_build)), dtype=bool)
    cols_by_tree = defaultdict(list); tref = {}
    for col, (tree, nid) in enumerate(leaf_build):
        cols_by_tree[id(tree)].append((col, nid)); tref[id(tree)] = tree
    for tid, colnodes in cols_by_tree.items():
        leaves = tref[tid].apply(np.ascontiguousarray(Xnp))
        for col, nid in colnodes:
            H[:, col] = (leaves == nid)
    return H


def assemble_ordinal_241(X_np):
    """Build J3's 241-d input from an X that has movesago appended (cols
    121:181): [played,placed_as_mover,mp(121), mover_movesago(60), opp(60)],
    split by placed_as_mover — matching the fit's _split_ordinal."""
    X121 = X_np[:, :121]
    ord60 = X_np[:, 121:181]
    pam = X_np[:, 60:120] > 0
    mover = np.where(pam, ord60, -1.0).astype(np.float32)
    opp = np.where(pam, -1.0, ord60).astype(np.float32)
    return np.concatenate([X121, mover, opp], axis=1)


def build_hidden_layer_batch(X_np, mlp, patterns, recent_Ks, use_relu,
                                 device, no_flanking=False, leaf_build=None,
                                 leaf_index=None):
    """Compute [tree_paths | recent_bits | flanking_patterns] for one
    chunk of positions.

    no_flanking=True drops the 960 flanking patterns from H — a TREE-ONLY
    hidden layer (the honest "what do the trees decode by themselves" run).

    Returns bool tensor on GPU (or float32 under use_relu)."""
    dtype = torch.float32 if use_relu else torch.bool
    # J3 FAST path (--leaf-index-cache-dir): X_np is the (b, n_trees) leaf-id
    # matrix from the prebuilt leaf cache; build the one-hot H by gather+compare
    # (H[:,col] = leaves[:,col_tree_idx[col]] == col_nid[col]) — no tree.apply.
    # Exact-equal to the tree.apply path (validated in build_leaf_cache).
    if leaf_index is not None:
        col_tree_idx, col_nid = leaf_index
        H = (X_np[:, col_tree_idx] == col_nid[None, :])
        return torch.from_numpy(np.ascontiguousarray(H)).to(
            device=device, dtype=dtype)
    # J3 (ordinal, --hidden-from-leaves): H = true leaf one-hot via tree.apply
    # on the 241-d input (built from X's appended movesago block).  Tree-only.
    if leaf_build is not None:
        inp241 = assemble_ordinal_241(X_np)
        H = build_H_from_leaves_np(leaf_build, inp241)
        return torch.from_numpy(H).to(device=device, dtype=dtype)
    # Trees were fit on played+even+mover_parity only (input_dim=121);
    # slice X_np to those columns for the tree forward.  Recent bits are
    # concatenated separately from X_np[:, 121:].
    tree_in_dim = mlp.W.shape[1]
    X = torch.from_numpy(np.ascontiguousarray(X_np[:, :tree_in_dim])
                            ).to(device)
    H_tree = mlp(X, out_device=device, out_dtype=dtype, use_relu=use_relu)
    del X
    # Recent bits are already in X_np at cols [121:121 + 60*len(recent_Ks)).
    if recent_Ks:
        n_recent = 60 * len(recent_Ks)
        recent = X_np[:, 121:121 + n_recent].astype(np.uint8)
        recent_t = torch.from_numpy(recent).to(device=device, dtype=dtype)
    else:
        recent_t = None
    parts = [H_tree]
    if recent_t is not None:
        parts.append(recent_t)
    if not no_flanking:
        # Flanking patterns.
        played = X_np[:, :60].astype(np.uint8)
        even = X_np[:, 60:120].astype(np.uint8)
        mp = X_np[:, 120].astype(np.uint8)
        FP_np = compute_pattern_activations(patterns, played, even, mp)
        parts.append(torch.from_numpy(FP_np).to(device=device, dtype=dtype))
    return torch.cat(parts, dim=1)


def evaluate(probe, eval_path, ply_min, ply_max, recent_Ks, mlp,
                patterns, use_relu, device, batch=1024,
                use_chunk_ext=False, canonicalize_mover=False,
                max_positions=None, no_flanking=False,
                leaf_build=None, needs_ordinal=False, cache_dir=None,
                leaf_index=None, leaf_index_cache_dir=None):
    if use_chunk_ext and leaf_index is not None and leaf_index_cache_dir:
        X, S, T, L = load_leaf_index_chunk(
            leaf_index_cache_dir, eval_path, ply_min, ply_max, max_positions)
    elif use_chunk_ext:
        X, S, T, L = load_chunk_cached(
            eval_path, ply_min, ply_max, canonicalize_mover, max_positions,
            needs_ordinal, cache_dir=cache_dir)
    else:
        X, S, T, L = process_pickle_chunk(eval_path, ply_min, ply_max,
                                              recent_Ks=recent_Ks)
    N = X.shape[0]
    correct_total = 0
    tp_t = fp_t = fn_t = 0          # legal-move (positive class) tallies
    am_ok = am_n = 0               # argmax-legality
    correct_by_ply = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for i in range(0, N, batch):
            X_batch = X[i:i + batch]
            H = build_hidden_layer_batch(X_batch, mlp, patterns,
                                             recent_Ks, use_relu, device,
                                             no_flanking=no_flanking,
                                             leaf_build=leaf_build,
                                             leaf_index=leaf_index)
            L_batch = torch.from_numpy(L[i:i + batch]).to(device)
            p = probe(H.float() if not use_relu else H)
            preds = (p > 0.5).to(torch.uint8)
            correct = (preds == L_batch).sum().item()
            correct_total += correct
            Lb = L_batch.to(torch.uint8)
            tp_t += int(((preds == 1) & (Lb == 1)).sum().item())
            fp_t += int(((preds == 1) & (Lb == 0)).sum().item())
            fn_t += int(((preds == 0) & (Lb == 1)).sum().item())
            has = Lb.sum(1) > 0
            if bool(has.any()):
                top = p.argmax(1)
                am_ok += int(Lb[torch.arange(Lb.shape[0]), top][has].sum().item())
                am_n += int(has.sum().item())
            T_batch = T[i:i + batch]
            for j in range(X_batch.shape[0]):
                ply_bucket = int(T_batch[j]) // 10 * 10
                pos_correct = int(((preds[j] == L_batch[j]).sum()).item())
                correct_by_ply[ply_bucket][0] += pos_correct
                correct_by_ply[ply_bucket][1] += 64
    total = N * 64
    acc = correct_total / total
    rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) else 0.0
    prec = tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f'  eval per-cell acc: {100*acc:.4f}%  (N={N} positions)')
    print(f'  eval LEGAL-MOVE: recall={100*rec:.2f}%  precision={100*prec:.2f}%'
           f'  F1={100*f1:.2f}%  argmax-legal='
           f'{100*am_ok/am_n if am_n else float("nan"):.2f}%')
    for lo in sorted(correct_by_ply.keys()):
        c, t = correct_by_ply[lo]
        print(f'    ply [{lo:2d},{lo+10:2d})  '
               f'acc={100 * c / t:.4f}%')
    return acc


# ===================== Nanda mine/yours board-state decoder ==================
# "Nanda-style" = mine/yours, via a parity mode-split (Nanda's tl_probing_v1 /
# our probe_state_pred_for_othello.py).  Two probes over the ABSOLUTE state
# {0=empty,1=white,2=black}: an EVEN-ply probe and an ODD-ply probe.  At eval,
# even-ply rows are decoded by the even probe, odd-ply by the odd probe — each
# learns the color->mine/yours mapping for its parity, which IS mine/yours.
# (Nanda's third "all" mode is just his absolute-color baseline and is never
# used in the reported accuracy, so we omit it.)  Same target/loss/hparams as
# the transformer decoder → tree-H accuracy directly comparable to the ~95%.
def make_state_probe(hidden_dim, device):
    """(modes=2 [even, odd], hidden_dim, 64 cells, 3 options) linear probe."""
    W = torch.randn(2, hidden_dim, BOARD_CELLS, 3, device=device) \
        / (hidden_dim ** 0.5)
    W.requires_grad_(True)
    return W


def _state_forward(W, H):
    # H (b, hidden) float -> (modes, b, cells, options)
    return torch.einsum('bh,mhco->mbco', H, W)


def state_loss(W, H, S_batch, T_batch):
    """Per-cell CE: even-ply rows -> even probe (mode 0), odd -> odd (mode 1)."""
    out = _state_forward(W, H)                      # (2, b, cells, 3)
    even = (T_batch % 2 == 0)
    loss = out.new_zeros(())
    for m, mask in ((0, even), (1, ~even)):
        if bool(mask.any()):
            loss = loss + F.cross_entropy(
                out[m][mask].reshape(-1, 3), S_batch[mask].reshape(-1))
    return loss


def _state_preds(W, H, T_batch):
    out = _state_forward(W, H)                       # (3, b, cells, 3)
    even = (T_batch % 2 == 0)
    preds = torch.zeros(H.shape[0], BOARD_CELLS, dtype=torch.long,
                        device=H.device)
    preds[even] = out[0][even].argmax(-1)
    preds[~even] = out[1][~even].argmax(-1)
    return preds


def evaluate_state(state_probe, eval_path, ply_min, ply_max, recent_Ks, mlp,
                   patterns, use_relu, device, batch, use_chunk_ext,
                   canonicalize_mover, max_positions, no_flanking, leaf_build,
                   needs_ordinal, cache_dir, leaf_index, leaf_index_cache_dir):
    if use_chunk_ext and leaf_index is not None and leaf_index_cache_dir:
        X, S, T, L = load_leaf_index_chunk(
            leaf_index_cache_dir, eval_path, ply_min, ply_max, max_positions)
    elif use_chunk_ext:
        X, S, T, L = load_chunk_cached(
            eval_path, ply_min, ply_max, canonicalize_mover, max_positions,
            needs_ordinal, cache_dir=cache_dir)
    else:
        X, S, T, L = process_pickle_chunk(eval_path, ply_min, ply_max,
                                              recent_Ks=recent_Ks)
    N = X.shape[0]
    correct = total = 0
    by_ply = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for i in range(0, N, batch):
            H = build_hidden_layer_batch(
                X[i:i + batch], mlp, patterns, recent_Ks, use_relu, device,
                no_flanking=no_flanking, leaf_build=leaf_build,
                leaf_index=leaf_index)
            if H.dtype != torch.float32:
                H = H.float()
            Tb = torch.from_numpy(T[i:i + batch].astype(np.int64)).to(device)
            Sb = torch.from_numpy(
                np.ascontiguousarray(S[i:i + batch]).astype(np.int64)).to(device)
            preds = _state_preds(state_probe, H, Tb)
            correct += int((preds == Sb).sum().item()); total += Sb.numel()
            for j in range(preds.shape[0]):
                pb = int(T[i + j]) // 10 * 10
                by_ply[pb][0] += int((preds[j] == Sb[j]).sum().item())
                by_ply[pb][1] += BOARD_CELLS
    acc = correct / max(total, 1)
    print(f'  eval STATE per-cell acc (mine/yours): {100*acc:.4f}%  '
           f'(N={N} positions)')
    for lo in sorted(by_ply):
        c, t = by_ply[lo]
        print(f'    ply [{lo:2d},{lo+10:2d})  acc={100*c/t:.4f}%')
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load-trees-from', default=None,
                    help='Tree bank checkpoint.  Required unless '
                          '--flanking-only.')
    ap.add_argument('--flanking-only', action='store_true',
                    help='Diagnostic: use ZERO tree units — the hidden layer '
                          'is the 960 flanking patterns alone (no trees, no '
                          'recent bits).  Measures what the flanking rules '
                          'decode by themselves.')
    ap.add_argument('--no-flanking', action='store_true',
                    help='Drop the 960 flanking patterns from the hidden layer '
                          '— TREE-ONLY readout (trees [+recent] decode by '
                          'themselves).  Mutually exclusive with '
                          '--flanking-only.')
    ap.add_argument('--data-source', default='pickle',
                    choices=['pickle', 'chunk-ext'],
                    help='pickle: replay games from data/othello_synthetic/*.pickle '
                          '(slow, ~15+ min/pickle).  chunk-ext: load pre-simulated '
                          'boards from experiments/.../chunk_ext_*.npz (fast, '
                          'seconds/chunk).  chunk-ext is preferred unless the '
                          'chunk files are missing.')
    ap.add_argument('--pickle-dir', default='data/othello_synthetic')
    ap.add_argument('--chunk-dir',
                    default=('experiments/mathematical_transformation_experiments/'
                             'heuristic_probe_results/feature_chunks'),
                    help='Directory of chunk_ext_*.npz files (used only when '
                          '--data-source chunk-ext).')
    ap.add_argument('--canonicalize-mover', action='store_true',
                    help='Use mover-relative encoding (placed_as_mover) in '
                          'the 121-d input.  Required when loading trees '
                          'trained with --canonicalize-mover.')
    ap.add_argument('--max-positions-per-file', type=int, default=None,
                    help='Cap on positions loaded from each chunk_ext/'
                          'pickle file.  Useful for trial runs and to bound '
                          'the 960-d pattern-legality memory.  chunk-ext '
                          'files have ~24M rows in [10,50); cap at ~4M for '
                          'a fast trial.')
    ap.add_argument('--flanking-patterns',
                    default='hand_crafted_flanking_patterns.pt')
    ap.add_argument('--num-train-games', type=int, default=6_000_000)
    ap.add_argument('--num-test-games', type=int, default=100_000)
    ap.add_argument('--no-recent', action='store_true',
                    help='Disable recent-K hidden bits entirely (tree-only H). '
                          'Cleaner than --recent-Ks "" in shell scripts.')
    ap.add_argument('--recent-Ks', default='1,2,5,10,20',
                    help='Comma-sep list; empty to disable.')
    ap.add_argument('--use-pattern-bias', action='store_true',
                    help='StruPO only: include the learned 960-d '
                          'per-pattern bias in PatternProbOrHead.  Default '
                          'off (bias = 0 fixed buffer) -- purer '
                          '"weights over leaves" architecture.')
    ap.add_argument('--pattern-bce', action='store_true',
                    help='Option B: train the 960 sigmoids with BCE against the '
                          'TRUE pattern firings (from state), prob-OR is '
                          'inference-only.  Default (off) = option A: prob-OR '
                          'output trained end-to-end against the legal mask.  '
                          'linpo probe-type only.')
    ap.add_argument('--probe-type', default='linpo',
                    choices=['linpo', 'strupo'],
                    help='linpo: LinearPatternProbOr (Linear H->960 + '
                          'prob-OR).  strupo: PatternProbOrHead (per-'
                          'pattern linear over leaves + prob-OR).')
    ap.add_argument('--task', default='legal', choices=['legal', 'state'],
                    help='legal (default): train the legal-move probe (linpo/'
                          'pattern-bce).  state: train a Nanda-style 3-mode '
                          'mine/yours board-state decoder on the hidden layer H '
                          '(per-cell accuracy), instead of the legal probe.')
    ap.add_argument('--ply-min', type=int, default=0)
    ap.add_argument('--ply-max', type=int, default=60)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--use-relu', action='store_true',
                    help='Default is bool step (memory-efficient).')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-seeds', type=int, default=1,
                    help='Train N probe heads (different init) simultaneously '
                         'on the shared hidden layer, then ensemble them at '
                         'eval.  Near-free vs N separate runs, since the chunk '
                         'load and hidden-layer build are shared across seeds.')
    ap.add_argument('--checkpoint-every', type=int, default=5,
                    help='Write a resume checkpoint to <out>.resume every N '
                         'chunks (and at each epoch end).  0 disables.  The '
                         'chunk order is deterministic per epoch, so resume '
                         'continues from the exact (epoch, chunk) reached.')
    ap.add_argument('--resume', action='store_true',
                    help='If a resume checkpoint exists, load probe+optimizer '
                         'and continue from the saved (epoch, chunk).  Safe to '
                         'add to a resubmitted job after a walltime timeout.')
    ap.add_argument('--resume-from', default=None,
                    help='Explicit resume-checkpoint path (default <out>.resume).')
    ap.add_argument('--cache-dir', default=None,
                    help='Directory for on-disk chunk caches (chunk-ext only). '
                         'The first pass builds a method-agnostic 181-d cache '
                         'per (chunk, ply-range, cap, canon); every later epoch '
                         'AND every other config reuses it — turning a '
                         'multi-epoch / multi-config run from N re-loads (+ '
                         '960-pattern recompute) into one.  Strongly '
                         'recommended whenever --epochs>1 or several configs '
                         'share the same data.')
    ap.add_argument('--precompute-cache-only', action='store_true',
                    help='Build the chunk caches for all train + eval files, '
                         'then exit.  Run once (serially, e.g. --flanking-only) '
                         'before launching a parallel grid so every config '
                         'reads the cache instead of racing to rebuild it.')
    ap.add_argument('--leaf-index-cache-dir', default=None,
                    help='J3 FAST path: directory of prebuilt leaf-index caches '
                         '(from build_leaf_cache.py).  When set (ordinal bank '
                         'only), H is built by gather+compare over the cached '
                         '(N,n_trees) leaf-ids instead of walking 960 trees per '
                         'batch — ~50-100x faster training.  Needs colmap.npz + '
                         'per-chunk .leaves.i16/.stl.npz in the dir.')
    args = ap.parse_args()

    if args.flanking_only and args.no_flanking:
        raise ValueError('--flanking-only and --no-flanking are mutually '
                         'exclusive (that would be an empty hidden layer).')

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    if args.flanking_only:
        # Zero tree units -> H = flanking patterns only.  Trees were fit on the
        # 121-d played+even+mover_parity input, so keep input_dim=121 for the
        # (empty) tree forward; recent bits are forced off for a pure
        # flanking-only readout.
        W_tree = np.zeros((0, 121), dtype=np.float32)
        b_tree = np.zeros((0,), dtype=np.float32)
        tree_meta = []
        print('--flanking-only: 0 tree units; H = 960 flanking patterns only '
              '(recent bits forced off)')
    else:
        if not args.load_trees_from:
            raise ValueError('--load-trees-from is required unless '
                             '--flanking-only is set.')
        W_tree, b_tree, tree_meta = load_trees(args.load_trees_from)
    mlp = OpeningTreeMLP(W_tree, b_tree, tree_meta, device)
    input_dim = W_tree.shape[1]

    # J3 ordinal (--hidden-from-leaves): use the saved sklearn trees + tree.apply
    # instead of the (invalid) ±1 W.  needs_ordinal → build the 241-d input.
    leaf_build = (None if args.flanking_only
                  else load_leaf_build(args.load_trees_from))
    needs_ordinal = leaf_build is not None
    if needs_ordinal:
        print(f'  leaf-based bank: {len(leaf_build)} leaf units via tree.apply '
               f'(J3 ordinal; movesago rebuilt from chunks)')

    # J3 FAST path: load the leaf-index colmap; H then comes from the prebuilt
    # leaf cache via gather+compare (no per-batch tree.apply).
    leaf_index = None
    if args.leaf_index_cache_dir and needs_ordinal:
        cm = np.load(os.path.join(args.leaf_index_cache_dir, 'colmap.npz'))
        leaf_index = (cm['col_tree_idx'], cm['col_nid'])
        print(f'  LEAF-INDEX CACHE: {args.leaf_index_cache_dir} '
               f'({len(leaf_index[0])} H cols, {int(cm["n_trees"])} trees) — '
               f'H via gather+compare, no tree.apply')

    recent_Ks = (None if (args.flanking_only or args.no_recent) else
                 (tuple(int(k) for k in args.recent_Ks.split(',')
                        if k.strip()) or None))
    patterns = load_patterns(args.flanking_patterns)
    print(f'loaded {len(patterns)} flanking patterns')

    # Verify input dim matches (skip for leaf-based J3 — W is unused there).
    expected_dim = 121
    if recent_Ks:
        expected_dim += 60 * len(recent_Ks)
    if not needs_ordinal and input_dim > expected_dim:
        raise ValueError(
            f'checkpoint tree input_dim={input_dim} > current featurizer '
            f'expected_dim={expected_dim} — recent-Ks may not match')

    # Data-source selection: pickle (slow replay) or chunk-ext (fast npz).
    use_chunk_ext = (args.data_source == 'chunk-ext')
    if use_chunk_ext:
        files = sorted(glob.glob(os.path.join(args.chunk_dir,
                                                  'chunk_ext_*.npz')))
        if not files:
            raise ValueError(f'no chunk_ext_*.npz files in {args.chunk_dir}')
        # Each chunk file holds ~600K games (~32M positions covering [5,59)).
        games_per_file = 600_000
        print(f'data source: chunk-ext  ({len(files)} chunk_ext files, '
               f'~{games_per_file:,} games each)', flush=True)
    else:
        files = sorted(glob.glob(os.path.join(args.pickle_dir, '*.pickle')))
        if not files:
            raise ValueError(f'no .pickle files in {args.pickle_dir}')
        games_per_file = 100_000
        print(f'data source: pickle  ({len(files)} pickle files, '
               f'~{games_per_file:,} games each)', flush=True)

    test_file = files[-1]
    train_files = files[:-1]
    print(f'{len(train_files)} train files + 1 held-out for eval')

    def load_chunk(path, cap=None):
        """Return (X, S, T, L) for one input file, regardless of source.
        In leaf-index mode X is the (N, n_trees) leaf-id matrix."""
        cap = cap if cap is not None else args.max_positions_per_file
        if use_chunk_ext:
            if leaf_index is not None:
                return load_leaf_index_chunk(
                    args.leaf_index_cache_dir, path, args.ply_min,
                    args.ply_max, cap)
            return load_chunk_cached(
                path, args.ply_min, args.ply_max, args.canonicalize_mover, cap,
                needs_ordinal, cache_dir=args.cache_dir)
        return process_pickle_chunk(path, args.ply_min, args.ply_max,
                                        recent_Ks=recent_Ks)

    # --precompute-cache-only: build the shared cache for every train + eval
    # file (serially), then exit — so a parallel grid launched afterwards reads
    # the cache instead of every config racing to rebuild the same chunks.
    if args.precompute_cache_only:
        if not use_chunk_ext:
            raise ValueError('--precompute-cache-only requires '
                             '--data-source chunk-ext')
        if not args.cache_dir:
            raise ValueError('--precompute-cache-only requires --cache-dir')
        print(f'PRECOMPUTE-CACHE-ONLY → {args.cache_dir}', flush=True)
        for i, f in enumerate(train_files):
            t = time.time()
            X, _, _, _ = load_chunk(f, cap=args.max_positions_per_file)
            n = 0 if X is None else X.shape[0]
            print(f'  [{i + 1}/{len(train_files)}] {os.path.basename(f)}: '
                   f'{n} rows in {time.time() - t:.0f}s', flush=True)
            del X
        t = time.time()
        load_chunk(test_file, cap=500_000)   # match the in-job eval's cap/key
        print(f'  eval {os.path.basename(test_file)} cached in '
               f'{time.time() - t:.0f}s', flush=True)
        print('precompute-cache-only done.', flush=True)
        return

    # Warm up: process one file to figure hidden_dim.  Cap tightly — we
    # only need enough rows to derive shapes.  In leaf-index mode hidden_dim is
    # just the number of H columns (from the colmap) and there is no cap-4096
    # leaf cache, so skip the warmup entirely.
    if leaf_index is not None:
        hidden_dim = len(leaf_index[0])
        print(f'  hidden_dim={hidden_dim} (from leaf-index colmap)')
    else:
        print(f'warmup: processing first file (cap 4096 rows)...', flush=True)
        tw = time.time()
        Xw, _, _, _ = load_chunk(train_files[0], cap=4096)
        print(f'  warmup load: {time.time() - tw:.1f}s  '
               f'positions={Xw.shape[0] if Xw is not None else 0}', flush=True)
        Xw_small = Xw[:64]
        H_small = build_hidden_layer_batch(Xw_small, mlp, patterns,
                                                recent_Ks, args.use_relu, device,
                                                no_flanking=args.no_flanking,
                                                leaf_build=leaf_build)
        hidden_dim = H_small.shape[1]
        print(f'  hidden_dim={hidden_dim}')
        del Xw, Xw_small, H_small

    # Initialize N probe heads (different init) for ensembling.  N=1 is the
    # original single-probe behavior.  All heads share the same hidden layer
    # per batch, so the extra cost is only the (cheap) per-head forward/back.
    def make_probe(seed):
        torch.manual_seed(1000 + seed)
        if args.probe_type == 'linpo':
            return LinearPatternProbOr(hidden_dim, patterns).to(device)
        # StruPO needs full meta list.  We only have tree_meta from
        # the load; pattern-tree meta already has 'pattern' fields.  If
        # tree_meta doesn't have pattern fields, StruPO cannot group.
        if 'pattern' not in tree_meta[0]:
            raise ValueError(
                'StruPO requires tree_target=patterns checkpoint (with '
                'pattern-path meta).  Loaded checkpoint has tree_path '
                'entries — use --probe-type linpo instead.')
        return PatternProbOrHead(
            tree_meta, patterns,
            use_pattern_bias=args.use_pattern_bias).to(device)

    if args.task == 'state':
        # Nanda 3-mode mine/yours board decoder on the hidden layer H.
        torch.manual_seed(1000)
        state_probe = make_state_probe(hidden_dim, device)
        state_opt = torch.optim.AdamW([state_probe], lr=1e-4,
                                       betas=(0.9, 0.99), weight_decay=0.01)
        probes, opts = [], []
        print(f'STATE decoder: Nanda mine/yours linear probe '
               f'(2 x {hidden_dim} x {BOARD_CELLS} x 3), AdamW lr=1e-4 '
               f'betas=(0.9,0.99) wd=0.01; mine/yours via even/odd mode-split; '
               f'params={2 * hidden_dim * BOARD_CELLS * 3:,}')
    else:
        state_probe = state_opt = None
        probes = [make_probe(s) for s in range(args.n_seeds)]
        opts = [torch.optim.AdamW(p.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
                for p in probes]
        print(f'{len(probes)} probe head(s), type={args.probe_type}, '
               f'params/head='
               f'{sum(p.numel() for p in probes[0].parameters()):,}')

    n_train_files = min(len(train_files),
                          (args.num_train_games + games_per_file - 1)
                          // games_per_file)
    train_subset = train_files[:n_train_files]
    print(f'training on ~{n_train_files * games_per_file:,} games '
           f'({n_train_files} files)', flush=True)

    # Option B (--pattern-bce): per-batch 960-pattern targets from state.
    if args.pattern_bce:
        from train_pattern_simple import compute_pattern_labels_batch as _cplb
        _pat_arrays = _get_pattern_arrays()   # targets, terminals, opp_*, mask

    # --- Resume support: restore probe+optimizer and skip completed chunks ---
    resume_path = args.resume_from or (args.out + '.resume')

    def save_resume(epoch, chunk_index):
        if not args.checkpoint_every or args.task == 'state':
            return                       # state runs are short; no resume
        tmp = resume_path + '.tmp'
        torch.save({'probe_states': [p.state_dict() for p in probes],
                    'opt_states': [o.state_dict() for o in opts],
                    'epoch': epoch, 'chunk_index': chunk_index,
                    'args': vars(args)}, tmp)
        os.replace(tmp, resume_path)     # atomic

    start_epoch, start_chunk = 1, 0
    if args.resume and args.task != 'state' and os.path.exists(resume_path):
        rck = torch.load(resume_path, map_location=device)
        # Back-compat: old single-probe checkpoints used 'probe_state'.
        p_states = rck.get('probe_states', [rck['probe_state']])
        o_states = rck.get('opt_states', [rck['opt_state']])
        for p, st in zip(probes, p_states):
            p.load_state_dict(st)
        for o, st in zip(opts, o_states):
            o.load_state_dict(st)
        start_epoch, start_chunk = rck['epoch'], rck['chunk_index']
        if start_chunk >= len(train_subset):      # that epoch finished
            start_epoch += 1
            start_chunk = 0
        print(f'RESUMED from {resume_path}: continuing at epoch '
               f'{start_epoch}, chunk {start_chunk}', flush=True)
    elif args.resume:
        print(f'--resume given but no checkpoint at {resume_path}; '
               f'starting fresh', flush=True)

    acc = None
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        print(f'\n=== epoch {epoch}/{args.epochs} ===')
        rng = np.random.RandomState(epoch)
        order = rng.permutation(len(train_subset))
        resume_skip = start_chunk if epoch == start_epoch else 0
        if resume_skip:
            print(f'  skipping first {resume_skip} chunks (already done '
                   f'this epoch)', flush=True)
        epoch_loss = 0.0
        epoch_batches = 0
        for ci, ci_idx in enumerate(order):
            if ci < resume_skip:
                continue
            pf = train_subset[ci_idx]
            print(f'  [{ci + 1}/{len(order)}] starting {os.path.basename(pf)} '
                   f'(cumulative {time.time() - t0:.0f}s)', flush=True)
            t_load = time.time()
            X, S, T, L = load_chunk(pf)
            if X is None:
                continue
            N = X.shape[0]
            print(f'    loaded {N} positions in {time.time() - t_load:.1f}s',
                    flush=True)
            t_hidden = time.time()
            # Process in mini-batches so we don't materialize the full
            # H matrix for the pickle at once (48K columns × 4M rows
            # would blow memory).
            perm = np.random.RandomState(epoch * 100 + ci).permutation(N)
            n_batches = (N + args.batch_size - 1) // args.batch_size
            batch_idx = 0
            for i in range(0, N, args.batch_size):
                idx = perm[i:i + args.batch_size]
                X_batch = X[idx]
                H_batch = build_hidden_layer_batch(
                    X_batch, mlp, patterns, recent_Ks, args.use_relu,
                    device, no_flanking=args.no_flanking,
                    leaf_build=leaf_build, leaf_index=leaf_index)
                if H_batch.dtype != torch.float32:
                    H_batch = H_batch.float()
                if args.task == 'state':
                    # Nanda 3-mode mine/yours board decoder on H.
                    S_batch = torch.from_numpy(
                        np.ascontiguousarray(S[idx]).astype(np.int64)).to(device)
                    T_batch = torch.from_numpy(
                        T[idx].astype(np.int64)).to(device)
                    loss = state_loss(state_probe, H_batch, S_batch, T_batch)
                    state_opt.zero_grad(); loss.backward(); state_opt.step()
                    last_loss = loss.item()
                    epoch_loss += last_loss; epoch_batches += 1
                    batch_idx += 1
                    if batch_idx % 256 == 0:
                        print(f'      batch {batch_idx}/{n_batches}  '
                               f'({time.time() - t_hidden:.1f}s so far, '
                               f'loss={last_loss:.4f})', flush=True)
                    continue
                L_batch = torch.from_numpy(L[idx]).to(
                    device=device, dtype=torch.float32)
                # Shared H_batch (no grad through it) → each head trains
                # independently on the same batch.
                # Option B: per-batch 960 pattern targets from state (cheap;
                # avoids the ~100 GB full-chunk target matrix).
                if args.pattern_bce:
                    pos_b = (T[idx] - 1).astype(np.int64)   # chunk 0-based ply
                    pt_np = _cplb(S[idx], pos_b, *_pat_arrays)   # (b, 960)
                    pt_batch = torch.from_numpy(pt_np).to(
                        device=device, dtype=torch.float32)
                else:
                    pt_batch = None
                last_loss = 0.0
                for p, o in zip(probes, opts):
                    if pt_batch is not None:
                        # Option B: BCE on the 960 pattern logits vs true firings
                        logits = p.linear(H_batch)
                        loss = F.binary_cross_entropy_with_logits(logits, pt_batch)
                    else:
                        # Option A: prob-OR output vs legal mask (end-to-end)
                        probs = p(H_batch).clamp(1e-6, 1 - 1e-6)
                        loss = F.binary_cross_entropy(probs, L_batch)
                    o.zero_grad(); loss.backward(); o.step()
                    last_loss = loss.item()
                epoch_loss += last_loss; epoch_batches += 1
                batch_idx += 1
                # Intra-chunk heartbeat every 256 batches so we know we're
                # not stuck.
                if batch_idx % 256 == 0:
                    print(f'      batch {batch_idx}/{n_batches}  '
                           f'({time.time() - t_hidden:.1f}s so far, '
                           f'loss={loss.item():.4f})', flush=True)
            del X, S, T, L
            print(f'    trained ({time.time() - t_hidden:.1f}s;   '
                   f'cumulative time {time.time() - t0:.0f}s)',
                    flush=True)
            if args.checkpoint_every and (ci + 1) % args.checkpoint_every == 0:
                save_resume(epoch, ci + 1)
                print(f'    [ckpt] resume saved @ epoch {epoch} chunk {ci + 1}',
                        flush=True)
        save_resume(epoch, len(order))   # epoch-end checkpoint
        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f'  epoch {epoch} avg loss: {avg_loss:.4f}')
        print(f'  eval on {os.path.basename(test_file)}...', flush=True)
        # Cap eval at 500K positions — enough for a stable per-cell
        # accuracy estimate, keeps eval under ~1 min.
        # In-job eval is a single-head diagnostic; the ensemble number comes
        # from reeval over all saved heads.
        if args.task == 'state':
            acc = evaluate_state(
                state_probe, test_file, args.ply_min, args.ply_max, recent_Ks,
                mlp, patterns, args.use_relu, device, args.batch_size,
                use_chunk_ext, args.canonicalize_mover, 500_000,
                args.no_flanking, leaf_build, needs_ordinal, args.cache_dir,
                leaf_index, args.leaf_index_cache_dir)
        else:
            acc = evaluate(probes[0], test_file, args.ply_min, args.ply_max,
                              recent_Ks, mlp, patterns, args.use_relu, device,
                              batch=args.batch_size,
                              use_chunk_ext=use_chunk_ext,
                              canonicalize_mover=args.canonicalize_mover,
                              max_positions=500_000,
                              no_flanking=args.no_flanking,
                              leaf_build=leaf_build, needs_ordinal=needs_ordinal,
                              cache_dir=args.cache_dir,
                              leaf_index=leaf_index,
                              leaf_index_cache_dir=args.leaf_index_cache_dir)

    if args.task == 'state':
        torch.save({'state_probe': state_probe.detach().cpu(),
                    'args': vars(args), 'final_acc': acc}, args.out)
    else:
        torch.save({
            'probe_states': [p.state_dict() for p in probes],
            'probe_state': probes[0].state_dict(),   # back-compat: first head
            'args': vars(args),
            'final_acc': acc,
        }, args.out)
    print(f'\nsaved {args.out}')
    if args.checkpoint_every and os.path.exists(resume_path):
        os.remove(resume_path)           # training complete; drop resume file
        print(f'removed resume checkpoint {resume_path}')


if __name__ == '__main__':
    main()
