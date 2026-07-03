"""Compare ensemble legal-move and board-state probe accuracy across all
board states consistent with a given moveset.

Two experiments in one pass (they share the ensemble forward):

  A. Legal-move top-1 accuracy against each consistent board's actual
     legal moves.  Tests whether the ensemble's ONE prediction generalizes
     across all boards a moveset could correspond to.

  B. Board-state 3-class per-cell accuracy from a hidden-state probe
     against each consistent board.  Tests whether the probe's ONE
     prediction generalizes across the consistent boards.

Uniform accuracy across consistent boards -> ensemble/probe has marginalised
over the ambiguity.  Biased toward one board (usually the training one) ->
memorised specific game trajectories.

Usage:
    python consistent_board_analysis.py \\
        --multi-ckpt experiments/.../multi_seed_N100_H512_playedeven_chunks0-9.pt \\
        --k 25  --num-games 1000  --n-samples 1000 \\
        --output-csv consistent_board_k25.csv
"""
import argparse
import copy
import csv
import glob
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from train_multi_seed_mlp import VectorizedMLP, slice_played_even
from train_pattern_simple import _get_cell_pat_index
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from compare_v4_vs_mlp import (
    load_val_games, played_even_features, C64_TO_C60,
)
from data.othello import OthelloBoardState
from eval_multi_seed_ensemble import load_vectorized_from_multi


# ---------- Board enumeration ----------

def sample_orderings(prefix, n_samples):
    """For a given played prefix, sample n_samples random orderings that
    preserve parity: parity-0 cells shuffled among the even-index slots,
    parity-1 cells shuffled among odd-index slots."""
    even_moves = [m for i, m in enumerate(prefix) if i % 2 == 0]
    odd_moves  = [m for i, m in enumerate(prefix) if i % 2 == 1]
    orderings = []
    for _ in range(n_samples):
        e = random.sample(even_moves, len(even_moves))
        o = random.sample(odd_moves,  len(odd_moves))
        out = []
        for i in range(len(prefix)):
            if i % 2 == 0:
                out.append(e[i // 2])
            else:
                out.append(o[i // 2])
        orderings.append(out)
    return orderings


def replay(moves):
    """Return (is_valid, board_state 8x8, next_hand_color) or (False, None, None)."""
    b = OthelloBoardState()
    try:
        for m in moves:
            valids = b.get_valid_moves()
            if m not in valids:
                return False, None, None
            b.umpire(m)
    except Exception:
        return False, None, None
    return True, b.state.copy(), b.next_hand_color


def enumerate_consistent_boards(game_prefix, n_samples):
    """Return dict {board_hash: (state_8x8, next_hand_color, count)}."""
    orderings = sample_orderings(game_prefix, n_samples)
    boards = {}
    for seq in orderings:
        valid, state, next_c = replay(seq)
        if not valid:
            continue
        h = state.tobytes()
        if h in boards:
            boards[h] = (boards[h][0], boards[h][1], boards[h][2] + 1)
        else:
            boards[h] = (state, next_c, 1)
    return boards


# ---------- Legal moves from an arbitrary board state ----------

def legal_from_state(state_8x8, next_hand_color):
    """Given a board state and whose-turn indicator, return the set of
    legal moves as 60-cell indices."""
    b = OthelloBoardState()
    b.state = state_8x8.copy()
    b.next_hand_color = next_hand_color
    valids_64 = b.get_valid_moves()
    return {C64_TO_C60[c] for c in valids_64 if c in C64_TO_C60}


# ---------- Probe head ----------

N_CELLS_64 = 64
N_CLASSES = 3   # 0=empty, 1=black, 2=white


class ConcatProbe(nn.Module):
    def __init__(self, n_seeds, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(n_seeds * hidden_dim, N_CELLS_64 * N_CLASSES)

    def forward(self, hidden):    # (B, N, hidden)
        return self.linear(hidden.flatten(1)).view(-1, N_CELLS_64, N_CLASSES)


def board_state_target_from_state(state_8x8):
    """Return 64-d int array of the board state.  0=empty, 1=black, 2=white."""
    flat = state_8x8.flatten()
    out = np.zeros(64, dtype=np.int64)
    out[flat == 1]  = 1   # black
    out[flat == -1] = 2   # white
    return out


# ---------- Ensemble forward ----------

def ensemble_hidden_and_scores(feats_batch, ks_batch, weights, idx, mask,
                                 N, device):
    """Return (hidden (B, N, H), cell_scores (B, N, 60))."""
    W1_e, b1_e, W2_e, b2_e, W1_o, b1_o, W2_o, b2_o = weights
    x = feats_batch.to(device)
    ks_t = ks_batch.to(device)
    use_me = (ks_t % 2 == 0); use_mo = ~use_me   # matches train convention
    B = x.shape[0]
    hidden_dim = W1_e.shape[2]
    h_all = torch.zeros(N, B, hidden_dim, device=device)
    logits_all = torch.zeros(N, B, 960, device=device)

    def fwd(W1, b1, W2, b2, xs):
        x_nbi = xs.unsqueeze(0).expand(N, -1, -1)
        h = F.relu(torch.bmm(x_nbi, W1) + b1)
        y = torch.bmm(h, W2) + b2
        return h, y

    if use_me.any():
        h, y = fwd(W1_e, b1_e, W2_e, b2_e, x[use_me])
        h_all[:, use_me] = h; logits_all[:, use_me] = y
    if use_mo.any():
        h, y = fwd(W1_o, b1_o, W2_o, b2_o, x[use_mo])
        h_all[:, use_mo] = h; logits_all[:, use_mo] = y

    log1m = -F.softplus(logits_all)
    gathered = log1m[:, :, idx]
    gathered = gathered.masked_fill(~mask[None, None], 0.0)
    cell_scores = -gathered.sum(dim=-1)
    return h_all.permute(1, 0, 2), cell_scores.permute(1, 0, 2)   # (B, N, H), (B, N, 60)


# ---------- Probe training ----------

def train_probe(weights, N, hidden_dim, games, device, epochs=5,
                 batch_size=512, k_min=5, k_max=53, lr=1e-3):
    """Train ConcatProbe on hidden -> board_state (3-class per cell)."""
    print(f"Building probe training set...", flush=True)
    feats_list, ks_list, boards_list = [], [], []
    for game in games:
        b = OthelloBoardState()
        for k in range(0, k_max + 1):
            if k >= k_min:
                target = board_state_target_from_state(b.state)
                feats_list.append(played_even_features(game[:k]))
                ks_list.append(k)
                boards_list.append(target)
            if k < len(game):
                try:
                    b.umpire(game[k])
                except Exception:
                    break
    n = len(feats_list)
    print(f"  {n:,} positions")

    probe = ConcatProbe(N, hidden_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device)
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n)
        total_loss = 0.0
        t0 = time.time()
        for i in range(0, n, batch_size):
            batch_idx = perm[i:i + batch_size]
            feats = torch.stack([feats_list[j] for j in batch_idx])
            ks    = torch.tensor([ks_list[j] for j in batch_idx])
            board = torch.tensor([boards_list[j] for j in batch_idx],
                                   device=device)
            with torch.no_grad():
                hidden, _ = ensemble_hidden_and_scores(
                    feats, ks, weights, idx, mask, N, device)
            logits = probe(hidden)                    # (B, 64, 3)
            loss = F.cross_entropy(
                logits.view(-1, N_CLASSES), board.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(batch_idx)
        print(f"  epoch {epoch}: loss={total_loss/n:.4f}  "
              f"({int(time.time()-t0)}s)", flush=True)
    return probe, idx, mask


# ---------- Main experiment ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi-ckpt', required=True)
    ap.add_argument('--k', type=int, default=25)
    ap.add_argument('--num-games', type=int, default=1000)
    ap.add_argument('--n-samples', type=int, default=1000,
                    help='Monte Carlo orderings per position for enumeration.')
    ap.add_argument('--probe-train-games', type=int, default=2000)
    ap.add_argument('--probe-epochs', type=int, default=5)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=1)
    ap.add_argument('--output-csv', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading {args.multi_ckpt}")
    me, mo, N, hidden_dim, _ = load_vectorized_from_multi(
        args.multi_ckpt, device)
    print(f"  N={N}, H={hidden_dim}")
    weights = (
        me.W1.detach(), me.b1.detach(), me.W2.detach(), me.b2.detach(),
        mo.W1.detach(), mo.b1.detach(), mo.W2.detach(), mo.b2.detach(),
    )

    games = load_val_games(args.data_dir, args.num_data_files)
    probe_games = games[:args.probe_train_games]
    experiment_games = games[
        args.probe_train_games:
        args.probe_train_games + args.num_games]
    print(f"  probe train: {len(probe_games)} games")
    print(f"  experiment:  {len(experiment_games)} games at k={args.k}")

    probe, idx, mask = train_probe(
        weights, N, hidden_dim, probe_games, device,
        epochs=args.probe_epochs,
    )

    # Run the two experiments per game
    print(f"\nRunning experiment for {len(experiment_games)} games...")
    n_written = 0
    n_positions = 0
    n_with_ambiguity = 0
    with open(args.output_csv, 'w', newline='') as f_out:
        w = csv.writer(f_out)
        w.writerow([
            'game_idx', 'k', 'board_hash', 'board_count',
            'n_distinct_boards', 'n_valid_orderings',
            'is_training_board',
            'top1_cell_60', 'top1_is_legal',
            'probe_cell_accuracy',
        ])

        t0 = time.time()
        probe.eval()
        for g_idx, game in enumerate(experiment_games):
            # Enumerate consistent boards
            prefix = game[:args.k]
            boards = enumerate_consistent_boards(prefix, args.n_samples)
            n_valid = sum(c for _, _, c in boards.values())
            n_distinct = len(boards)
            n_positions += 1
            if n_distinct < 2:
                continue
            n_with_ambiguity += 1

            # Determine which board is the "training" one
            b_actual = OthelloBoardState()
            valid_actual = True
            try:
                for m in prefix:
                    b_actual.umpire(m)
            except Exception:
                valid_actual = False
            training_hash = (b_actual.state.tobytes()
                              if valid_actual else None)

            # Ensemble forward (single call for this position)
            feats = played_even_features(prefix).unsqueeze(0)          # (1, 120)
            ks_t = torch.tensor([args.k], dtype=torch.long)
            with torch.no_grad():
                hidden, cell_scores = ensemble_hidden_and_scores(
                    feats, ks_t, weights, idx, mask, N, device)
                # Aggregate cell scores across seeds (sum_log_prob_or):
                agg = cell_scores.sum(dim=1).squeeze(0)                 # (60,)
                top1_60 = int(agg.argmax().item())
                # Probe forward:
                probe_logits = probe(hidden)                            # (1, 64, 3)
                probe_argmax = probe_logits.argmax(dim=-1).squeeze(0)   # (64,)
                probe_argmax = probe_argmax.cpu().numpy()

            for b_hash, (state, next_c, count) in boards.items():
                legal_set = legal_from_state(state, next_c)
                top1_legal = int(top1_60 in legal_set)
                board_target = board_state_target_from_state(state)
                probe_acc = float((probe_argmax == board_target).mean())
                w.writerow([
                    g_idx, args.k, b_hash.hex()[:16],  # short hash
                    count, n_distinct, n_valid,
                    int(b_hash == training_hash),
                    top1_60, top1_legal,
                    f"{probe_acc:.4f}",
                ])
                n_written += 1

            if (g_idx + 1) % 50 == 0:
                print(f"  {g_idx+1}/{len(experiment_games)}  "
                      f"ambiguous: {n_with_ambiguity}  "
                      f"rows written: {n_written}  "
                      f"({int(time.time()-t0)}s)", flush=True)

    print()
    print(f"Done.  Wrote {n_written} rows over "
          f"{n_with_ambiguity}/{n_positions} positions with ambiguity "
          f"(distinct consistent boards >= 2).")
    print(f"Output: {args.output_csv}")


if __name__ == '__main__':
    main()
