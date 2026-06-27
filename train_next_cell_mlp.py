"""Train an MLP to predict the next CELL (60-way classification) from
move_grid features.

  Input:  3600-d move_grid feature (60 cells × 60 move-numbers, signed)
  Output: 60-d softmax over cells
  Loss:   cross-entropy against the actually-played next move

Training data is streamed directly from the synthetic pickle files
(no precomputed chunks needed for this task because we need the actual
played cell as label, which the existing chunks don't store).

Usage:
    python train_next_cell_mlp.py --hidden 512 --epochs 3 \
        --max-games 6000000 --batch-size 4096
"""
import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')

# 60 movable cells (excluding the 4 initial center cells)
CENTER_CELLS = {27, 28, 35, 36}
VALID_MOVES = sorted(set(range(64)) - CENTER_CELLS)
CELL_TO_IDX = {c: i for i, c in enumerate(VALID_MOVES)}


class NextCellMLP(nn.Module):
    def __init__(self, input_dim=3600, hidden_dim=512, n_cells=60):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_cells),
        )

    def forward(self, x):
        return self.net(x)


def games_iter(pickle_files, max_games=None):
    """Yield games one at a time across pickle files."""
    n = 0
    for pf in pickle_files:
        with open(pf, 'rb') as f:
            batch = pickle.load(f)
        if len(batch) < 9e4:
            continue
        for game in batch:
            yield game
            n += 1
            if max_games is not None and n >= max_games:
                return


def encode_game_samples(game):
    """For a game M_1..M_N, return (N, 3600) features and (N,) cell targets.

    Sample t (0-indexed):
      features[t] = move_grid state after game[0..t-1] (t moves played)
      target[t]   = cell index of game[t] (the next move to predict)

    move_grid convention (matches train_pattern_simple.py:to_move_grid_input):
      grid[cell, move_num] = -1 if played at even-indexed move (0,2,4...)
                             +1 if played at odd-indexed move (1,3,5...)
                             0  if not played at that move number
      Flattened to 3600-d.
    """
    N = len(game)
    if N < 1:
        return None, None

    features = np.zeros((N, 60, 60), dtype=np.float32)
    targets = np.zeros(N, dtype=np.int64)

    cur = np.zeros((60, 60), dtype=np.float32)
    valid_mask = np.ones(N, dtype=bool)
    for t in range(N):
        # Save state BEFORE adding move t (features for predicting game[t])
        features[t] = cur
        cell = game[t]
        if cell not in CELL_TO_IDX:
            valid_mask[t] = False
            targets[t] = -1
            continue
        c60 = CELL_TO_IDX[cell]
        targets[t] = c60
        # Update cur to include move t (for use predicting game[t+1])
        color = -1.0 if t % 2 == 0 else 1.0
        cur[c60, t] = color

    if not valid_mask.any():
        return None, None
    features = features[valid_mask].reshape(-1, 3600)
    targets = targets[valid_mask]
    return features, targets


def stream_batches(pickle_files, max_games, batch_size, device, shuffle_seed=None):
    """Yield (X_batch, y_batch) tensors of size ~batch_size.

    Accumulates samples from games into a buffer, optionally shuffles, and
    flushes batches of size batch_size.  When max_games is reached the
    remainder is also yielded.
    """
    rng = np.random.RandomState(shuffle_seed) if shuffle_seed is not None else None
    buf_X, buf_y = [], []
    buf_size = 0
    chunk_size = max(batch_size * 8, 32_000)  # buffer ~8 batches before flushing

    for game in games_iter(pickle_files, max_games):
        feats, tgts = encode_game_samples(game)
        if feats is None:
            continue
        buf_X.append(feats)
        buf_y.append(tgts)
        buf_size += len(feats)
        if buf_size >= chunk_size:
            X_all = np.concatenate(buf_X, axis=0)
            y_all = np.concatenate(buf_y, axis=0)
            buf_X, buf_y = [], []
            buf_size = 0
            if rng is not None:
                perm = rng.permutation(len(X_all))
                X_all = X_all[perm]
                y_all = y_all[perm]
            for i in range(0, len(X_all), batch_size):
                xb = torch.from_numpy(X_all[i:i + batch_size]).to(device)
                yb = torch.from_numpy(y_all[i:i + batch_size]).to(device)
                yield xb, yb

    # flush remainder
    if buf_X:
        X_all = np.concatenate(buf_X, axis=0)
        y_all = np.concatenate(buf_y, axis=0)
        if rng is not None:
            perm = rng.permutation(len(X_all))
            X_all = X_all[perm]
            y_all = y_all[perm]
        for i in range(0, len(X_all), batch_size):
            xb = torch.from_numpy(X_all[i:i + batch_size]).to(device)
            yb = torch.from_numpy(y_all[i:i + batch_size]).to(device)
            yield xb, yb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--max-games', type=int, default=6_000_000)
    ap.add_argument('--eval-games', type=int, default=50_000,
                    help='Number of games (from end of train pool) to hold out for eval')
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output-dir',
                    default='experiments/mathematical_transformation_experiments/'
                            'heuristic_probe_results/pattern_detector_checkpoints')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, H={args.hidden}, epochs={args.epochs}, "
          f"max_games={args.max_games}, batch_size={args.batch_size}, "
          f"seed={args.seed}")

    pickle_files = sorted(glob.glob(os.path.join(args.data_dir, '*.pickle')))
    print(f"Found {len(pickle_files)} pickle files")
    # Split: training games from start, eval from end (within max_games range)
    # We'll process pickle files sequentially; eval samples come from the LAST
    # games we'd otherwise have used for training.
    train_max = max(0, args.max_games - args.eval_games)

    # Build model
    model = NextCellMLP(input_dim=3600, hidden_dim=args.hidden, n_cells=60).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    seed_tag = "" if args.seed == 0 else f"_seed{args.seed}"
    save_path = os.path.join(save_dir,
        f"next_cell_mlp_H{args.hidden}_movegrid{seed_tag}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        last_log = t0
        for step, (xb, yb) in enumerate(stream_batches(
                pickle_files, train_max, args.batch_size, device,
                shuffle_seed=epoch * 1000 + args.seed)):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            total_n += len(yb)
            total_correct += (logits.argmax(dim=-1) == yb).sum().item()
            now = time.time()
            if now - last_log > 60:
                print(f"  epoch {epoch} step {step}: "
                      f"n={total_n:,} loss={total_loss/total_n:.4f} "
                      f"acc={total_correct/total_n:.4f} "
                      f"elapsed={int(now-t0)}s", flush=True)
                last_log = now

        # End-of-epoch eval
        model.eval()
        ev_loss = 0.0
        ev_correct = 0
        ev_n = 0
        with torch.no_grad():
            for xb, yb in stream_batches(
                    pickle_files[-1:], args.eval_games, args.batch_size, device,
                    shuffle_seed=None):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, reduction='sum')
                ev_loss += loss.item()
                ev_n += len(yb)
                ev_correct += (logits.argmax(dim=-1) == yb).sum().item()

        train_acc = total_correct / max(1, total_n)
        ev_acc = ev_correct / max(1, ev_n)
        print(f"Epoch {epoch}: train n={total_n:,} loss={total_loss/total_n:.4f} "
              f"acc={train_acc:.4f}  |  eval n={ev_n:,} "
              f"loss={ev_loss/max(1,ev_n):.4f} acc={ev_acc:.4f}  "
              f"({int(time.time()-t0)}s)", flush=True)

        torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch,
            'hidden': args.hidden,
            'input_dim': 3600,
            'n_cells': 60,
            'best_train_acc': train_acc,
            'best_eval_acc': ev_acc,
        }, save_path)
        print(f"  Saved {save_path}")


if __name__ == '__main__':
    main()
