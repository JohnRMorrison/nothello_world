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
                 if m.get('kind') in ('tree_path', 'pattern_path')]
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


def evaluate(probe, eval_pickle, ply_min, ply_max, recent_Ks, mlp,
                patterns, use_relu, device, batch=1024):
    X, S, T, L = process_pickle_chunk(eval_pickle, ply_min, ply_max,
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
    ap.add_argument('--pickle-dir', required=True)
    ap.add_argument('--flanking-patterns',
                    default='hand_crafted_flanking_patterns.pt')
    ap.add_argument('--num-train-games', type=int, default=6_000_000)
    ap.add_argument('--num-test-games', type=int, default=100_000)
    ap.add_argument('--recent-Ks', default='1,2,5,10,20',
                    help='Comma-sep list; empty to disable.')
    ap.add_argument('--probe-type', default='linpo',
                    choices=['linpo', 'strupo'],
                    help='linpo: LinearPatternProbOr (Linear H->960 + '
                          'prob-OR).  strupo: PatternProbOrHead (per-'
                          'pattern linear over leaves + prob-OR).')
    ap.add_argument('--ply-min', type=int, default=10)
    ap.add_argument('--ply-max', type=int, default=50)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--use-relu', action='store_true',
                    help='Default is bool step (memory-efficient).')
    ap.add_argument('--out', required=True)
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

    # Compute hidden dim by running one small batch through the pipeline.
    files = sorted(glob.glob(os.path.join(args.pickle_dir, '*.pickle')))
    if not files:
        raise ValueError(f'no .pickle files in {args.pickle_dir}')
    test_pickle = files[-1]
    train_files = files[:-1]
    print(f'{len(train_files)} train pickle files + 1 held-out for eval')

    # Warm up: process one pickle to figure hidden_dim.
    print('warmup: processing first pickle to determine hidden dim...')
    Xw, _, _, _ = process_pickle_chunk(train_files[0], args.ply_min,
                                            args.ply_max, recent_Ks=recent_Ks)
    Xw_small = Xw[:64]
    H_small = build_hidden_layer_batch(Xw_small, mlp, patterns,
                                            recent_Ks, args.use_relu, device)
    hidden_dim = H_small.shape[1]
    print(f'  hidden_dim={hidden_dim}')
    del Xw, Xw_small, H_small

    # Initialize probe.
    if args.probe_type == 'linpo':
        probe = LinearPatternProbOr(hidden_dim, patterns).to(device)
        print(f'probe: LinearPatternProbOr  params={sum(p.numel() for p in probe.parameters()):,}')
    else:
        # StruPO needs full meta list.  We only have tree_meta from
        # the load; pattern-tree meta already has 'pattern' fields.  If
        # tree_meta doesn't have pattern fields, StruPO cannot group.
        if 'pattern' not in tree_meta[0]:
            raise ValueError(
                'StruPO requires tree_target=patterns checkpoint (with '
                'pattern-path meta).  Loaded checkpoint has tree_path '
                'entries — use --probe-type linpo instead.')
        probe = PatternProbOrHead(tree_meta, patterns).to(device)
        print(f'probe: PatternProbOrHead    params={sum(p.numel() for p in probe.parameters()):,}')

    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    # How many pickles to use per epoch (100K games each).
    games_per_pickle = 100_000
    n_train_pickles = min(len(train_files),
                            (args.num_train_games + games_per_pickle - 1)
                            // games_per_pickle)
    train_subset = train_files[:n_train_pickles]
    print(f'training on ~{n_train_pickles * games_per_pickle:,} games '
           f'({n_train_pickles} pickle files)')

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f'\n=== epoch {epoch}/{args.epochs} ===')
        rng = np.random.RandomState(epoch)
        order = rng.permutation(len(train_subset))
        epoch_loss = 0.0
        epoch_batches = 0
        for ci, ci_idx in enumerate(order):
            pf = train_subset[ci_idx]
            t_load = time.time()
            X, S, T, L = process_pickle_chunk(pf, args.ply_min, args.ply_max,
                                                  recent_Ks=recent_Ks)
            if X is None:
                continue
            N = X.shape[0]
            print(f'  [{ci + 1}/{len(order)}] {os.path.basename(pf)}: '
                   f'{N} positions  (load+extract '
                   f'{time.time() - t_load:.1f}s)')
            t_hidden = time.time()
            # Process in mini-batches so we don't materialize the full
            # H matrix for the pickle at once (48K columns × 4M rows
            # would blow memory).
            perm = np.random.RandomState(epoch * 100 + ci).permutation(N)
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
                probs = probe(H_batch).clamp(1e-6, 1 - 1e-6)
                loss = F.binary_cross_entropy(probs, L_batch)
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item(); epoch_batches += 1
            del X, S, T, L
            print(f'    trained ({time.time() - t_hidden:.1f}s;   '
                   f'cumulative time {time.time() - t0:.0f}s)')
        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f'  epoch {epoch} avg loss: {avg_loss:.4f}')
        print(f'  eval on {os.path.basename(test_pickle)}...')
        acc = evaluate(probe, test_pickle, args.ply_min, args.ply_max,
                          recent_Ks, mlp, patterns, args.use_relu, device,
                          batch=args.batch_size)

    torch.save({
        'probe_state': probe.state_dict(),
        'args': vars(args),
        'final_acc': acc,
    }, args.out)
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
