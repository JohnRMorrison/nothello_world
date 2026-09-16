"""Shared helpers for the three OGPT behavioral experiments:

  exp1_legal_to_illegal.py   - probability retained on a move that was legal
                               and becomes illegal (flank-loss).
  exp2_forfeit_top1_error.py - top-1 (argmax) illegal-move rate at each ply
                               after a forfeit (skipped turn).
  exp3_opponent_chains.py    - probability on an ILLEGAL empty square vs the
                               length of the opponent chain leading up to it.

All three load the Nanda synthetic Othello-GPT, run batched next-move
inference, and replay each game with OthelloBoardState to get the true
board state + legal-move set at every decision point.

Conventions (aligned with ogpt_recall_by_turn.py):
  - A game is a length-60 list of board cells (0..63, center 4 excluded).
  - Decision point t (t = 0..58): the model has seen moves g[0..t] and
    predicts move t+1.  After umpire(g[0..t]):
        state[t]  = board after t+1 stones placed
        mover[t]  = colour to play move t+1  (+1 black, -1 white)
        legal[t]  = get_valid_moves() for that mover
        probs[t]  = softmax over the 60 movable cells for move t+1
  - Cell indexing: MOVABLE (sorted 0..63 minus center) is the 60-cell order;
    M2I maps a 0..63 board cell to its 0..59 movable index.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import tokenize_games, load_games, VOCAB_SIZE, STOI  # noqa: E402

CENTER = {27, 28, 35, 36}
MOVABLE = [c for c in range(64) if c not in CENTER]        # 60 cells, sorted
M2I = {c: i for i, c in enumerate(MOVABLE)}                # board cell -> 0..59
I2M = {i: c for c, i in M2I.items()}
N_MOVES = 60

# 8 ray directions (dr, dc)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_ogpt(ckpt, device):
    """Load the Nanda synthetic Othello-GPT (mingpt causal LM)."""
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    cell_tokens = torch.tensor([STOI[c] for c in MOVABLE],
                               device=device, dtype=torch.long)
    return model, block_size, cell_tokens


def iter_game_probs(model, games, block_size, cell_tokens, device, batch=64):
    """Yield (game_index, game, probs) one game at a time.

    probs: (len(game)-? , 60) float32 = softmax over the full vocab, gathered
    to the 60 movable cells, for each sequence position.  Row t is the
    distribution predicting move t+1.
    """
    toks = tokenize_games(games, seq_len=block_size).to(device)
    with torch.no_grad():
        for i in range(0, len(games), batch):
            out = model(toks[i:i + batch])
            logits = out[0] if isinstance(out, tuple) else out    # (B,T,vocab)
            p = F.softmax(logits, dim=-1)[:, :, cell_tokens]       # (B,T,60)
            p = p.float().cpu().numpy()
            for j in range(p.shape[0]):
                gi = i + j
                yield gi, games[gi], p[j]


def replay_game(game):
    """Replay a game; return per-decision-point (state, legal, mover).

    states: (T, 64) int8   board after t+1 stones (+1 black, -1 white, 0 empty)
    legal:  (T, 64) bool    legal-move mask for move t+1 (side = mover[t])
    mover:  (T,)   int8      +1 black / -1 white, colour to play move t+1
    """
    b = OthelloBoardState()
    T = len(game)
    states = np.zeros((T, 64), dtype=np.int8)
    legal = np.zeros((T, 64), dtype=bool)
    mover = np.zeros(T, dtype=np.int8)
    for t, mv in enumerate(game):
        b.umpire(mv)
        states[t] = b.state.reshape(64)
        mover[t] = b.next_hand_color
        for m in b.get_valid_moves():
            legal[t, m] = True
    return states, legal, mover


def opponent_chain_lengths(state64, cell, opponent):
    """Length of the contiguous opponent run starting adjacent to `cell` in
    each of the 8 directions.  Returns an int array of length 8 (same order as
    DIRECTIONS).  A run stops at the first non-opponent cell (empty / own /
    edge).  `cell` itself is assumed empty.
    """
    state = state64.reshape(8, 8)
    r0, c0 = cell // 8, cell % 8
    out = np.zeros(8, dtype=np.int64)
    for d, (dr, dc) in enumerate(DIRECTIONS):
        n = 0
        r, c = r0 + dr, c0 + dc
        while 0 <= r < 8 and 0 <= c < 8 and state[r, c] == opponent:
            n += 1
            r += dr
            c += dc
        out[d] = n
    return out


def load_experiment_games(max_files, n_games, seed=None, shuffle=False):
    games = load_games(max_files=max_files)
    if shuffle:
        rng = np.random.RandomState(seed if seed is not None else 0)
        idx = rng.permutation(len(games))
        games = [games[i] for i in idx]
    if n_games is not None and len(games) > n_games:
        games = games[:n_games]
    return games
