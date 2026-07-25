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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.othello import OthelloBoardState
from opening_tree_mlp import (
    playedeven_features, LinearPatternProbOr, PatternProbOrHead,
    OpeningTreeMLP, BOARD_CELLS, C64_TO_C60,
)
from flanking_patterns import (
    load_patterns, compute_pattern_activations, patterns_by_target,
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
                              pat_batch=200_000):
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

    # State labels are not used by the streaming probe training loop
    # (only X, T, L are consumed).  Return None to avoid the cost.
    return X, None, stream_pos, L


def process_pickle_chunk(pickle_path, ply_min, ply_max, recent_Ks=None):
    """Load one pickle file, replay each game, extract midgame positions.

    Returns (X, S, T, L, played, even, mp) numpy arrays.
    """
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


def build_hidden_layer_batch(X_np, mlp, patterns, recent_Ks, use_relu,
                                 device):
    """Compute [tree_paths | recent_bits | flanking_patterns] for one
    chunk of positions.

    Returns bool tensor on GPU (or float32 under use_relu)."""
    dtype = torch.float32 if use_relu else torch.bool
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
    # Flanking patterns.
    played = X_np[:, :60].astype(np.uint8)
    even = X_np[:, 60:120].astype(np.uint8)
    mp = X_np[:, 120].astype(np.uint8)
    FP_np = compute_pattern_activations(patterns, played, even, mp)
    FP_t = torch.from_numpy(FP_np).to(device=device, dtype=dtype)
    parts = [H_tree]
    if recent_t is not None:
        parts.append(recent_t)
    parts.append(FP_t)
    return torch.cat(parts, dim=1)


def evaluate(probe, eval_path, ply_min, ply_max, recent_Ks, mlp,
                patterns, use_relu, device, batch=1024,
                use_chunk_ext=False, canonicalize_mover=False,
                max_positions=None):
    if use_chunk_ext:
        X, S, T, L = process_chunk_ext_file(
            eval_path, ply_min, ply_max,
            canonicalize_mover=canonicalize_mover,
            max_positions=max_positions)
    else:
        X, S, T, L = process_pickle_chunk(eval_path, ply_min, ply_max,
                                              recent_Ks=recent_Ks)
    N = X.shape[0]
    correct_total = 0
    correct_by_ply = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for i in range(0, N, batch):
            X_batch = X[i:i + batch]
            H = build_hidden_layer_batch(X_batch, mlp, patterns,
                                             recent_Ks, use_relu, device)
            L_batch = torch.from_numpy(L[i:i + batch]).to(device)
            p = probe(H.float() if not use_relu else H)
            preds = (p > 0.5).to(torch.uint8)
            correct = (preds == L_batch).sum().item()
            correct_total += correct
            T_batch = T[i:i + batch]
            for j in range(X_batch.shape[0]):
                ply_bucket = int(T_batch[j]) // 10 * 10
                pos_correct = int(((preds[j] == L_batch[j]).sum()).item())
                correct_by_ply[ply_bucket][0] += pos_correct
                correct_by_ply[ply_bucket][1] += 64
    total = N * 64
    acc = correct_total / total
    print(f'  eval per-cell acc: {100*acc:.4f}%  (N={N} positions)')
    for lo in sorted(correct_by_ply.keys()):
        c, t = correct_by_ply[lo]
        print(f'    ply [{lo:2d},{lo+10:2d})  '
               f'acc={100 * c / t:.4f}%')
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load-trees-from', required=True)
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
    ap.add_argument('--recent-Ks', default='1,2,5,10,20',
                    help='Comma-sep list; empty to disable.')
    ap.add_argument('--use-pattern-bias', action='store_true',
                    help='StruPO only: include the learned 960-d '
                          'per-pattern bias in PatternProbOrHead.  Default '
                          'off (bias = 0 fixed buffer) -- purer '
                          '"weights over leaves" architecture.')
    ap.add_argument('--probe-type', default='linpo',
                    choices=['linpo', 'strupo'],
                    help='linpo: LinearPatternProbOr (Linear H->960 + '
                          'prob-OR).  strupo: PatternProbOrHead (per-'
                          'pattern linear over leaves + prob-OR).')
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
    args = ap.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    W_tree, b_tree, tree_meta = load_trees(args.load_trees_from)
    mlp = OpeningTreeMLP(W_tree, b_tree, tree_meta, device)
    input_dim = W_tree.shape[1]

    recent_Ks = tuple(int(k) for k in args.recent_Ks.split(',')
                        if k.strip()) or None
    patterns = load_patterns(args.flanking_patterns)
    print(f'loaded {len(patterns)} flanking patterns')

    # Verify input dim matches.
    expected_dim = 121
    if recent_Ks:
        expected_dim += 60 * len(recent_Ks)
    if input_dim > expected_dim:
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
        """Return (X, S, T, L) for one input file, regardless of source."""
        cap = cap if cap is not None else args.max_positions_per_file
        if use_chunk_ext:
            return process_chunk_ext_file(
                path, args.ply_min, args.ply_max,
                canonicalize_mover=args.canonicalize_mover,
                max_positions=cap)
        return process_pickle_chunk(path, args.ply_min, args.ply_max,
                                        recent_Ks=recent_Ks)

    # Warm up: process one file to figure hidden_dim.  Cap tightly — we
    # only need enough rows to derive shapes.
    print(f'warmup: processing first file (cap 4096 rows)...', flush=True)
    tw = time.time()
    Xw, _, _, _ = load_chunk(train_files[0], cap=4096)
    print(f'  warmup load: {time.time() - tw:.1f}s  '
           f'positions={Xw.shape[0] if Xw is not None else 0}', flush=True)
    Xw_small = Xw[:64]
    H_small = build_hidden_layer_batch(Xw_small, mlp, patterns,
                                            recent_Ks, args.use_relu, device)
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

    probes = [make_probe(s) for s in range(args.n_seeds)]
    opts = [torch.optim.AdamW(p.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay) for p in probes]
    print(f'{len(probes)} probe head(s), type={args.probe_type}, '
           f'params/head={sum(p.numel() for p in probes[0].parameters()):,}')

    n_train_files = min(len(train_files),
                          (args.num_train_games + games_per_file - 1)
                          // games_per_file)
    train_subset = train_files[:n_train_files]
    print(f'training on ~{n_train_files * games_per_file:,} games '
           f'({n_train_files} files)', flush=True)

    # --- Resume support: restore probe+optimizer and skip completed chunks ---
    resume_path = args.resume_from or (args.out + '.resume')

    def save_resume(epoch, chunk_index):
        if not args.checkpoint_every:
            return
        tmp = resume_path + '.tmp'
        torch.save({'probe_states': [p.state_dict() for p in probes],
                    'opt_states': [o.state_dict() for o in opts],
                    'epoch': epoch, 'chunk_index': chunk_index,
                    'args': vars(args)}, tmp)
        os.replace(tmp, resume_path)     # atomic

    start_epoch, start_chunk = 1, 0
    if args.resume and os.path.exists(resume_path):
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
                L_batch = torch.from_numpy(L[idx]).to(
                    device=device, dtype=torch.float32)
                H_batch = build_hidden_layer_batch(
                    X_batch, mlp, patterns, recent_Ks, args.use_relu,
                    device)
                if H_batch.dtype != torch.float32:
                    H_batch = H_batch.float()
                # Shared H_batch (no grad through it) → each head trains
                # independently on the same batch.
                last_loss = 0.0
                for p, o in zip(probes, opts):
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
        acc = evaluate(probes[0], test_file, args.ply_min, args.ply_max,
                          recent_Ks, mlp, patterns, args.use_relu, device,
                          batch=args.batch_size,
                          use_chunk_ext=use_chunk_ext,
                          canonicalize_mover=args.canonicalize_mover,
                          max_positions=500_000)

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
