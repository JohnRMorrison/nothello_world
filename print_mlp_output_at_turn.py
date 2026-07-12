"""Print the MLP's 60-cell output + multi-hot legality target at a given
turn of an adversarial game, in board-cell-index order.

The MLP is a DirectMLP taking a 120-d played+even feature vector and
producing 960 pattern logits, aggregated to 60 cell scores via prob_or.
Converted to per-cell probabilities via p = 1 - exp(-cell_score).

The multi-hot target is the training target for this style of MLP:
one bit per cell indicating whether the cell is a currently-legal move
on the actual board.

Usage:
    python print_mlp_output_at_turn.py --game-index 0 --turn 32 \
        --mlp-ckpt experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_direct_H512_playedeven.pt \
        --hidden 512
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from data.othello import OthelloBoardState
from train_pattern_simple import DirectMLP, _get_cell_pat_index
from compare_v4_vs_mlp import played_even_features, C64_TO_C60
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX


MOVABLE_64 = sorted(C64_TO_C60.keys())      # 60 board cells in cell-index order


def alg(cell):
    return f"{'abcdefgh'[cell % 8]}{cell // 8 + 1}"


def load_mlp(mlp_ckpt_path, hidden, device):
    ckpt = torch.load(mlp_ckpt_path, map_location='cpu')
    input_dim = ckpt.get('input_dim', 120)
    n_patterns = ckpt.get('n_patterns', 960)
    me = DirectMLP(input_dim, hidden, n_patterns)
    mo = DirectMLP(input_dim, hidden, n_patterns)
    me.load_state_dict(ckpt['even'])
    mo.load_state_dict(ckpt['odd'])
    me.to(device).eval()
    mo.to(device).eval()
    return me, mo, input_dim, n_patterns


def cell_scores_from_mlp(me, mo, feats, k, device):
    """feats: (1, 120).  k = number of moves played so far.  Returns (60,)."""
    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)
    idx = idx.to(device)
    mask = mask.to(device)
    # Parity routing: use `me` (even model) when k is odd, matching the
    # val-game convention used in experiment1_adversarial_rate_mlp.py.
    with torch.no_grad():
        if k % 2 == 1:
            logits = me(feats)                            # (1, 960)
        else:
            logits = mo(feats)
        log1m = -F.softplus(logits)                       # (1, 960)
        gathered = log1m[:, idx]                          # (1, 60, 16)
        gathered = gathered.masked_fill(~mask[None], 0.0)
        cell_scores = -gathered.sum(dim=-1).squeeze(0)    # (60,)
    return cell_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_by_depth')
    ap.add_argument('--game-index', type=int, default=0)
    ap.add_argument('--turn', type=int, default=-1,
                    help='-1 = the error turn T.')
    ap.add_argument('--mlp-ckpt', required=True)
    ap.add_argument('--hidden', type=int, default=512)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    d = np.load(os.path.join(args.adversarial_dir, 'adversarial_records.npz'),
                 allow_pickle=True)
    games = d['games']; turns = d['turns']; cells = d['illegal_cells']
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(games))
    i = int(idx[args.game_index])
    game = list(games[i])
    T = int(turns[i]); C = int(cells[i])
    turn = args.turn if args.turn >= 0 else T

    me, mo, input_dim, n_patterns = load_mlp(args.mlp_ckpt, args.hidden, device)

    # Features and legality at `turn` -- these come from state AFTER turn+1
    # moves played (matching the OGPT script's convention).
    board = OthelloBoardState()
    for t in range(0, turn + 1):
        board.umpire(game[t])
    legal_set = set(board.get_valid_moves())

    feats = played_even_features(game[:turn + 1]).unsqueeze(0).to(device)
    k = turn + 1
    scores_60 = cell_scores_from_mlp(me, mo, feats, k, device).cpu().numpy()
    probs_60 = 1.0 - np.exp(-scores_60.clip(min=0))

    print(f"Game {args.game_index + 1}: error turn T = {T}, illegal C = {alg(C)}")
    print(f"MLP H={args.hidden} output at turn {turn} ({turn + 1} moves played)")
    print()
    print(f"  idx | cell |  MLP p    | target | legal on actual")
    print(f"  ----+------+-----------+--------+-----------------")
    for out_i, cell in enumerate(MOVABLE_64):
        p = float(probs_60[out_i])
        is_legal = cell in legal_set
        target = 1 if is_legal else 0
        marker = ' <- C' if cell == C else ''
        print(f"  {out_i + 1:>3} | {alg(cell):>4} | {p:9.4f} | {target:>6} | "
              f"{'LEGAL' if is_legal else 'ILLEGAL'}{marker}")

    print()
    print(f"  Multi-hot target (1 = legal):")
    tgt = [1 if c in legal_set else 0 for c in MOVABLE_64]
    print(f"  {tgt}")
    print()
    print(f"  Number of legal cells: {sum(tgt)}")


if __name__ == '__main__':
    main()
