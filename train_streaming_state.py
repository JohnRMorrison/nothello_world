"""Streaming state-decoding probe: linear board decoder on the ~48K
tree-leaf activations of a pattern-tree checkpoint.

Analogous to Nanda's linear probe on GPT layer 6 (95.88% state accuracy).
Uses ONLY the tree-leaf hidden layer -- no flanking-pattern activations,
no recent bits.

Target: per-cell state in mover-relative encoding {0=empty, 1=mine, 2=opp}.
Loss: cross-entropy per cell, summed over the 64 cells and averaged over
positions.

Usage:
    python train_streaming_state.py \\
        --load-trees-from ckpts_midgame/.../canonical_g20000_d15_ml10_p10-50.pt \\
        --canonicalize-mover \\
        --num-train-games 6000000 \\
        --ply-min 10 --ply-max 50
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from opening_tree_mlp import OpeningTreeMLP, BOARD_CELLS
from train_streaming_probe import load_trees, playedeven_features


# ------------------------------------------------------------------
# Loader: return (X, S) for one chunk_ext file.
# X = 121-d input to the trees.  S = (N, 64) mover-relative labels.
# ------------------------------------------------------------------

def load_chunk_state(chunk_path, ply_min, ply_max, canonicalize_mover,
                       max_positions=None):
    """Load features, labels, positions from a chunk_ext_*.npz.  Convert
    labels (absolute: 0=empty, 1=white, 2=black) to MOVER-RELATIVE
    (0=empty, 1=mine, 2=opp) at each position's next-mover.

    Applies the same ply-range shift as process_chunk_ext_file:
    chunk position t == streaming position t+1.
    """
    z = np.load(chunk_path)
    positions = z['positions'].astype(np.int64)
    chunk_lo = ply_min - 1
    chunk_hi = ply_max - 1
    mask = (positions >= chunk_lo) & (positions < chunk_hi)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        z.close()
        return None, None, None
    if max_positions is not None and len(idx) > max_positions:
        rng = np.random.RandomState(0)
        idx = rng.choice(idx, size=max_positions, replace=False)
        idx.sort()
    positions = positions[idx]
    features = np.asarray(z['features'][idx]).astype(np.float32)
    labels_abs = np.asarray(z['labels'][idx]).astype(np.int8)
    z.close()

    # Build 121-d X (played + even-or-placed_as_mover + mp).
    stream_pos = (positions + 1).astype(np.int32)
    played = features[:, :60]
    even = features[:, 120:180]
    N = len(features)
    X = np.zeros((N, 121), dtype=np.float32)
    X[:, :60] = played
    mover_parity = (stream_pos % 2).astype(np.float32)
    if canonicalize_mover:
        target_even = (1.0 - mover_parity)[:, None]
        placed_as_mover = played * (even == target_even)
        X[:, 60:120] = placed_as_mover
    else:
        X[:, 60:120] = even
        X[:, 120] = mover_parity

    # Convert labels to mover-relative.
    # Streaming position T -> next mover has parity T % 2.
    # BLACK plays parity 0 (chunk label 2), WHITE plays parity 1
    # (chunk label 1).
    # mine_label per row = 2 if BLACK to move else 1
    # opp_label per row = 3 - mine_label
    is_black = (stream_pos % 2 == 0)          # (N,) bool
    mine_label = np.where(is_black, 2, 1)     # (N,) int8
    opp_label = 3 - mine_label                # (N,) int8
    # Broadcast comparison to (N, 64)
    S = np.zeros_like(labels_abs, dtype=np.int64)
    S[labels_abs == mine_label[:, None]] = 1
    S[labels_abs == opp_label[:, None]] = 2
    return X, S, stream_pos


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load-trees-from', required=True)
    ap.add_argument('--chunk-dir',
                    default=('experiments/mathematical_transformation_experiments/'
                             'heuristic_probe_results/feature_chunks'))
    ap.add_argument('--num-train-games', type=int, default=6_000_000)
    ap.add_argument('--canonicalize-mover', action='store_true')
    ap.add_argument('--ply-min', type=int, default=10)
    ap.add_argument('--ply-max', type=int, default=50)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--use-relu', action='store_true')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-positions-per-file', type=int, default=None)
    ap.add_argument('--eval-max-positions', type=int, default=500_000)
    ap.add_argument('--checkpoint-every', type=int, default=5,
                    help='Write a resume checkpoint to <out>.resume every N '
                         'chunks (and at each epoch end).  0 disables.  Chunk '
                         'order is deterministic per epoch, so resume '
                         'continues from the exact (epoch, chunk) reached.')
    ap.add_argument('--resume', action='store_true',
                    help='If a resume checkpoint exists, load probe+optimizer '
                         'and continue from the saved (epoch, chunk).')
    ap.add_argument('--resume-from', default=None,
                    help='Explicit resume-checkpoint path (default <out>.resume).')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading trees from {args.load_trees_from}...', flush=True)
    W, b, meta = load_trees(args.load_trees_from)
    mlp = OpeningTreeMLP(W, b, meta, device)
    hidden_dim = W.shape[0]
    print(f'  tree hidden_dim = {hidden_dim}', flush=True)

    files = sorted(glob.glob(os.path.join(args.chunk_dir, 'chunk_ext_*.npz')))
    if not files:
        raise ValueError(f'no chunk_ext_*.npz in {args.chunk_dir}')
    games_per_file = 600_000
    n_train_files = min(len(files) - 1,
                          (args.num_train_games + games_per_file - 1)
                          // games_per_file)
    train_files = files[:n_train_files]
    test_file = files[-1]
    print(f'training on {n_train_files} chunks; eval on '
           f'{os.path.basename(test_file)}', flush=True)

    # Linear probe: hidden -> (64 * 3) logits per position.
    probe = torch.nn.Linear(hidden_dim, BOARD_CELLS * 3).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    # --- Resume support: restore probe+optimizer and skip completed chunks ---
    resume_path = args.resume_from or (args.out + '.resume')

    def save_resume(epoch, chunk_index):
        if not args.checkpoint_every:
            return
        tmp = resume_path + '.tmp'
        torch.save({'probe_state': probe.state_dict(),
                    'opt_state': opt.state_dict(),
                    'epoch': epoch, 'chunk_index': chunk_index,
                    'args': vars(args)}, tmp)
        os.replace(tmp, resume_path)     # atomic

    start_epoch, start_chunk = 1, 0
    if args.resume and os.path.exists(resume_path):
        rck = torch.load(resume_path, map_location=device)
        probe.load_state_dict(rck['probe_state'])
        opt.load_state_dict(rck['opt_state'])
        start_epoch, start_chunk = rck['epoch'], rck['chunk_index']
        if start_chunk >= len(train_files):       # that epoch finished
            start_epoch += 1
            start_chunk = 0
        print(f'RESUMED from {resume_path}: continuing at epoch '
               f'{start_epoch}, chunk {start_chunk}', flush=True)
    elif args.resume:
        print(f'--resume given but no checkpoint at {resume_path}; '
               f'starting fresh', flush=True)

    def compute_hidden(X_np):
        dtype = torch.float32 if args.use_relu else torch.bool
        tree_in_dim = mlp.W.shape[1]
        X_t = torch.from_numpy(np.ascontiguousarray(X_np[:, :tree_in_dim])).to(device)
        H = mlp(X_t, out_device=device, out_dtype=dtype,
                use_relu=args.use_relu)
        del X_t
        return H

    acc = None
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        print(f'\n=== epoch {epoch}/{args.epochs} ===', flush=True)
        rng = np.random.RandomState(epoch)
        order = rng.permutation(len(train_files))
        resume_skip = start_chunk if epoch == start_epoch else 0
        if resume_skip:
            print(f'  skipping first {resume_skip} chunks (already done '
                   f'this epoch)', flush=True)
        for ci, ci_idx in enumerate(order):
            if ci < resume_skip:
                continue
            pf = train_files[ci_idx]
            print(f'  [{ci + 1}/{len(train_files)}] {os.path.basename(pf)} '
                   f'(cumulative {time.time()-t0:.0f}s)', flush=True)
            t_load = time.time()
            X, S, _ = load_chunk_state(
                pf, args.ply_min, args.ply_max,
                canonicalize_mover=args.canonicalize_mover,
                max_positions=args.max_positions_per_file)
            if X is None:
                continue
            N = X.shape[0]
            print(f'    loaded {N} positions in {time.time()-t_load:.1f}s',
                    flush=True)
            t_train = time.time()
            perm = np.random.RandomState(epoch * 100 + ci).permutation(N)
            n_batches = (N + args.batch_size - 1) // args.batch_size
            for bi in range(n_batches):
                idx = perm[bi * args.batch_size : (bi + 1) * args.batch_size]
                Xb = X[idx]
                Sb = torch.from_numpy(S[idx]).to(device)     # (B, 64) long
                Hb = compute_hidden(Xb).float()               # (B, hidden)
                logits = probe(Hb).view(-1, BOARD_CELLS, 3)   # (B, 64, 3)
                loss = F.cross_entropy(
                    logits.reshape(-1, 3), Sb.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                if (bi + 1) % 256 == 0:
                    print(f'      batch {bi+1}/{n_batches}  '
                           f'({time.time()-t_train:.1f}s so far, '
                           f'loss={loss.item():.4f})', flush=True)
            print(f'    trained ({time.time()-t_train:.1f}s)', flush=True)
            del X, S
            if args.checkpoint_every and (ci + 1) % args.checkpoint_every == 0:
                save_resume(epoch, ci + 1)
                print(f'    [ckpt] resume saved @ epoch {epoch} chunk {ci + 1}',
                        flush=True)
        save_resume(epoch, len(order))   # epoch-end checkpoint

        # Eval per epoch.
        print(f'\n  eval on {os.path.basename(test_file)}...', flush=True)
        Xe, Se, _ = load_chunk_state(
            test_file, args.ply_min, args.ply_max,
            canonicalize_mover=args.canonicalize_mover,
            max_positions=args.eval_max_positions)
        Ne = Xe.shape[0]
        total_correct = 0
        total_cells = 0
        with torch.no_grad():
            for bi in range(0, Ne, args.batch_size):
                Xb = Xe[bi:bi + args.batch_size]
                Sb = torch.from_numpy(Se[bi:bi + args.batch_size]).to(device)
                Hb = compute_hidden(Xb).float()
                logits = probe(Hb).view(-1, BOARD_CELLS, 3)
                preds = logits.argmax(dim=-1)
                total_correct += (preds == Sb).sum().item()
                total_cells += Sb.numel()
        acc = total_correct / total_cells
        print(f'  epoch {epoch} eval per-cell 3-way acc = {100*acc:.4f}%',
               f'(N={Ne})', flush=True)

    torch.save({
        'probe_state': probe.state_dict(),
        'args': vars(args),
        'final_acc': acc,
    }, args.out)
    print(f'\nsaved {args.out}')
    if args.checkpoint_every and os.path.exists(resume_path):
        os.remove(resume_path)           # training complete; drop resume file
        print(f'removed resume checkpoint {resume_path}')


if __name__ == '__main__':
    main()
