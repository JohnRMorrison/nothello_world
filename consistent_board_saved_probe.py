"""Consistent-board analysis using the SAVED single-MLP + SAVED probe pair
(no training).

For a moveset with multiple consistent board states, evaluate whether the
single MLP's ONE prediction is uniformly accurate across the boards.  Same
question for the probe's ONE board-state prediction.

Uses:
  --mlp-ckpt   pattern_simple_direct_H512_playedeven.pt
                 (DirectMLP:  Linear(120, 512) -> ReLU -> Linear(512, 960),
                  split by turn parity into 'even' and 'odd' state dicts)
  --probe-ckpt probe_direct_H512_playedeven.pt
                 (two Linear(512, 64*3) probes, one per parity)

No probe training - loads the saved probe directly.

Usage:
    python consistent_board_saved_probe.py \\
        --mlp-ckpt experiments/.../pattern_simple_direct_H512_playedeven.pt \\
        --probe-ckpt experiments/.../probe_direct_H512_playedeven.pt \\
        --k 25 --num-games 1000 --n-samples 1000 \\
        --output-csv consistent_board_k25_saved.csv
"""
import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from train_pattern_simple import DirectMLP, _get_cell_pat_index
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from compare_v4_vs_mlp import (
    played_even_features, C64_TO_C60, load_val_games,
)
from data.othello import OthelloBoardState


# ---------- Board enumeration (biased MC) ----------

def sample_one_valid_ordering(p0_cells, p1_cells):
    """See analyze_late_ambiguity.sample_one_valid_ordering."""
    remaining_p0 = list(p0_cells)
    remaining_p1 = list(p1_cells)
    board = OthelloBoardState()
    step = 0
    while remaining_p0 or remaining_p1:
        parity = step % 2
        cands = remaining_p0 if parity == 0 else remaining_p1
        legal = set(board.get_valid_moves())
        available = [c for c in cands if c in legal]
        if not available:
            return False, None, None
        chosen = random.choice(available)
        board.umpire(chosen)
        if parity == 0:
            remaining_p0.remove(chosen)
        else:
            remaining_p1.remove(chosen)
        step += 1
    return True, board.state.copy(), board.next_hand_color


def enumerate_consistent_boards(game_prefix, n_samples, max_retries=5):
    p0 = list(game_prefix[0::2])
    p1 = list(game_prefix[1::2])
    boards = {}
    for _ in range(n_samples):
        for _ in range(max_retries + 1):
            valid, state, next_c = sample_one_valid_ordering(p0, p1)
            if valid:
                break
        if not valid:
            continue
        h = state.tobytes()
        if h in boards:
            boards[h] = (boards[h][0], boards[h][1], boards[h][2] + 1)
        else:
            boards[h] = (state, next_c, 1)
    return boards


def legal_from_state(state_8x8, next_hand_color):
    b = OthelloBoardState()
    b.state = state_8x8.copy()
    b.next_hand_color = next_hand_color
    valids_64 = b.get_valid_moves()
    return {C64_TO_C60[c] for c in valids_64 if c in C64_TO_C60}


def board_state_target_from_state(state_8x8):
    flat = state_8x8.flatten()
    out = np.zeros(64, dtype=np.int64)
    out[flat == 1]  = 1
    out[flat == -1] = 2
    return out


# ---------- MLP + probe loading ----------

def load_mlp_and_probe(mlp_path, probe_path, input_dim, device):
    mlp_ckpt = torch.load(mlp_path, map_location=device)
    probe_ckpt = torch.load(probe_path, map_location=device)
    H = probe_ckpt['hidden_dim']

    model_even = DirectMLP(input_dim, H).to(device)
    model_odd  = DirectMLP(input_dim, H).to(device)
    model_even.load_state_dict(mlp_ckpt['even'])
    model_odd.load_state_dict(mlp_ckpt['odd'])
    model_even.eval(); model_odd.eval()

    probe_even = nn.Linear(H, 64 * 3).to(device)
    probe_odd  = nn.Linear(H, 64 * 3).to(device)
    probe_even.load_state_dict(probe_ckpt['even'])
    probe_odd.load_state_dict(probe_ckpt['odd'])
    probe_even.eval(); probe_odd.eval()

    print(f"  MLP:   H={H}, DirectMLP loaded")
    print(f"  Probe: 2x Linear({H}, 192), best_acc={probe_ckpt.get('best_acc'):.4f}")
    return (model_even, model_odd), (probe_even, probe_odd), H


@torch.no_grad()
def forward_hidden_scores_probe(feats, k_parity, models, probes, idx, mask,
                                  device):
    """Given ONE position (feats, k_parity), return:
      cell_scores 60-d cell prob_or scores
      probe_logits 64x3 board state logits
    """
    model_even, model_odd = models
    probe_even, probe_odd = probes
    x = feats.unsqueeze(0).to(device)  # (1, 120)
    # Convention (matches train_pattern_simple + train_multi_seed): even parity
    # -> model_even.  If chunk positions field: (positions % 2 == 0) -> even.
    is_even = (k_parity % 2 == 0)
    mlp = model_even if is_even else model_odd
    probe = probe_even if is_even else probe_odd

    # DirectMLP.net[0] is Linear(120, H); [1] is ReLU; [2] is Linear(H, 960)
    lin1, relu, lin2 = mlp.net[0], mlp.net[1], mlp.net[2]
    h = relu(lin1(x))                     # (1, H)
    logits = lin2(h)                       # (1, 960)

    log1m = -F.softplus(logits)            # (1, 960)
    gathered = log1m[:, idx]               # (1, 60, K)
    gathered = gathered.masked_fill(~mask[None], 0.0)
    cell_scores = -gathered.sum(dim=-1)    # (1, 60)

    probe_out = probe(h).view(-1, 64, 3)   # (1, 64, 3)
    return cell_scores.squeeze(0), probe_out.squeeze(0)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mlp-ckpt', required=True)
    ap.add_argument('--probe-ckpt', required=True)
    ap.add_argument('--k', type=int, default=25)
    ap.add_argument('--num-games', type=int, default=1000)
    ap.add_argument('--n-samples', type=int, default=1000,
                    help='Biased-MC orderings per position for enumeration.')
    ap.add_argument('--game-offset', type=int, default=2000,
                    help='Skip the first N games (default 2000, disjoint '
                         'from prior probe-training splits).')
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
    print(f"MLP:   {args.mlp_ckpt}")
    print(f"Probe: {args.probe_ckpt}")
    models, probes, H = load_mlp_and_probe(
        args.mlp_ckpt, args.probe_ckpt, input_dim=120, device=device)

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    print(f"Loading games...")
    games = load_val_games(args.data_dir, args.num_data_files)
    experiment_games = games[args.game_offset:
                              args.game_offset + args.num_games]
    print(f"  experiment: {len(experiment_games)} games at k={args.k}")

    print(f"\nRunning experiment for {len(experiment_games)} games...")
    n_written = 0
    n_positions = 0
    n_with_ambiguity = 0
    with open(args.output_csv, 'w', newline='') as f_out:
        w = csv.writer(f_out)
        w.writerow([
            'game_idx', 'k', 'board_hash', 'board_count',
            'n_distinct_boards',
            'is_training_board',
            'top1_cell_60', 'top1_is_legal',
            'probe_cell_accuracy',
        ])

        t0 = time.time()
        for g_idx, game in enumerate(experiment_games):
            prefix = game[:args.k]
            boards = enumerate_consistent_boards(prefix, args.n_samples)
            n_distinct = len(boards)
            n_positions += 1
            if n_distinct < 2:
                continue
            n_with_ambiguity += 1

            # Determine the training-observed board (from replay of original)
            b_actual = OthelloBoardState()
            valid_actual = True
            try:
                for m in prefix:
                    b_actual.umpire(m)
            except Exception:
                valid_actual = False
            training_hash = (b_actual.state.tobytes()
                              if valid_actual else None)

            # ONE forward pass through the MLP + probe for this moveset
            feats = played_even_features(prefix)               # (120,)
            cell_scores, probe_out = forward_hidden_scores_probe(
                feats, args.k, models, probes, idx, mask, device)
            top1_60 = int(cell_scores.argmax().item())
            probe_argmax = probe_out.argmax(dim=-1).cpu().numpy()  # (64,)

            for b_hash, (state, next_c, count) in boards.items():
                legal_set = legal_from_state(state, next_c)
                top1_legal = int(top1_60 in legal_set)
                board_target = board_state_target_from_state(state)
                probe_acc = float((probe_argmax == board_target).mean())
                w.writerow([
                    g_idx, args.k, b_hash.hex()[:16],
                    count, n_distinct,
                    int(b_hash == training_hash),
                    top1_60, top1_legal,
                    f"{probe_acc:.4f}",
                ])
                n_written += 1

            if (g_idx + 1) % 50 == 0:
                print(f"  {g_idx+1}/{len(experiment_games)}  "
                      f"ambiguous: {n_with_ambiguity}  "
                      f"rows: {n_written}  "
                      f"({int(time.time()-t0)}s)", flush=True)

    print()
    print(f"Done.  Wrote {n_written} rows over "
          f"{n_with_ambiguity}/{n_positions} ambiguous positions.")
    print(f"Output: {args.output_csv}")


if __name__ == '__main__':
    main()
