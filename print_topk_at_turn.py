"""Print top-K cells by OGPT probability at a given turn of an adversarial
game.  Marks each cell's actual-board legality and shows P.

Usage:
    python print_topk_at_turn.py --game-index 0 --turn 32 --k 10
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import tokenize_games, VOCAB_SIZE


CENTER_CELLS = {27, 28, 35, 36}
VALID_MOVES = [i for i in range(64) if i not in CENTER_CELLS]


def alg(cell):
    return f"{'abcdefgh'[cell % 8]}{cell // 8 + 1}"


def build_pos_to_token(block_size):
    dummy_game = list(VALID_MOVES)
    toks = tokenize_games([dummy_game], seq_len=block_size)[0].tolist()
    pos_to_token = np.full(64, -1, dtype=np.int64)
    for i, m in enumerate(dummy_game):
        if i < len(toks):
            pos_to_token[m] = toks[i]
    return pos_to_token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_by_depth')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--game-index', type=int, default=0)
    ap.add_argument('--turn', type=int, default=-1,
                    help='-1 = the error turn T.')
    ap.add_argument('--k', type=int, default=10)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    pos_to_token = build_pos_to_token(block_size)

    d = np.load(os.path.join(args.adversarial_dir, 'adversarial_records.npz'),
                allow_pickle=True)
    games = d['games']; turns = d['turns']; cells = d['illegal_cells']
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(games))
    i = int(idx[args.game_index])
    game = list(games[i])
    T = int(turns[i])
    C = int(cells[i])
    turn = args.turn if args.turn >= 0 else T

    # Model forward
    L = min(turn + 1, block_size)
    tokens = tokenize_games([game[:L]], seq_len=block_size).to(device)
    with torch.no_grad():
        logits, _ = model(tokens)
    probs = F.softmax(logits[0, turn, :], dim=-1).detach().cpu().numpy()

    # Marginals over the 60 valid cells
    probs_60 = np.zeros(60, dtype=np.float32)
    for k, m in enumerate(VALID_MOVES):
        tok = int(pos_to_token[m])
        if tok >= 0:
            probs_60[k] = probs[tok]
    probs_60 = probs_60 / max(probs_60.sum(), 1e-9)

    # Legality on actual board at `turn`
    board = OthelloBoardState()
    for t in range(0, turn + 1):
        board.umpire(game[t])
    legal_set = set(board.get_valid_moves())

    # Rank cells
    order = np.argsort(-probs_60)
    print(f"Game {args.game_index + 1}: error turn T = {T}, illegal C = {alg(C)}")
    print(f"Distribution at turn {turn} (after {turn + 1} moves played)")
    print()
    print(f"  rank | cell |  P    | legal on actual")
    print(f"  -----+------+-------+-----------------")
    for r in range(args.k):
        idx60 = int(order[r])
        cell = VALID_MOVES[idx60]
        p = float(probs_60[idx60])
        is_legal = cell in legal_set
        is_C = (cell == C)
        marker = ' <- C' if is_C else ''
        print(f"  {r+1:>4} | {alg(cell):>4} | {p * 100:5.2f}% | "
              f"{'LEGAL' if is_legal else 'ILLEGAL'}{marker}")

    top10_sum = float(probs_60[order[:args.k]].sum())
    print()
    print(f"  Sum of top {args.k}: {top10_sum * 100:.2f}%")
    tail = 1 - top10_sum
    print(f"  Remaining tail:  {tail * 100:.2f}%")


if __name__ == '__main__':
    main()
