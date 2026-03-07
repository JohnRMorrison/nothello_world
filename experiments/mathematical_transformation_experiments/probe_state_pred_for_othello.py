#!/usr/bin/env python
"""
Probe the V_table state predictor for real Othello board state.

The state predictor was trained on an arbitrary non-Othello task (predicting
60-dim binary state from a sparse V_table flip rule). This script tests whether
a linear probe can decode actual Othello board positions from its activations.

If it can, board state decodability is not decisive evidence of a world model.

Usage (from project root):
    python experiments/mathematical_transformation_experiments/probe_state_pred_for_othello.py \
        --ckpt-dir experiments/mathematical_transformation_experiments/ckpts/state_pred_vseed42_8L_512d \
        --layer 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import importlib.util
import types

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import OthelloBoardState without triggering data/__init__.py (which requires pgn)
sys.modules["pgn"] = types.ModuleType("pgn")
_spec = importlib.util.spec_from_file_location(
    "othello_module", os.path.join(PROJECT_ROOT, "data", "othello.py"),
    submodule_search_locations=[],
)
_othello_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_othello_mod)
OthelloBoardState = _othello_mod.OthelloBoardState

from mingpt.model import GPT, GPTConfig

GAME_LEN = 60
VOCAB_SIZE = 61
STATE_DIM = 60
PAD_IDX = 0
ROWS, COLS = 8, 8
OPTIONS = 3  # white, empty, black
MODES = 3    # even, odd, all

_VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})
STOI = {-100: 0}
STOI.update({move: i + 1 for i, move in enumerate(_VALID_MOVES)})
ITOS = {v: k for k, v in STOI.items()}

SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "othello_synthetic")


# ============================= Model ==========================================

class GPTStatePredictor(GPT):
    """State predictor with 60-dim binary output (BCE loss)."""

    def __init__(self, config, state_dim=STATE_DIM):
        super().__init__(config)
        self.state_dim = state_dim
        self.head = nn.Linear(config.n_embd, state_dim, bias=False)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class GPTBracketPredictor(GPT):
    """Bracket predictor with 180-dim output (3-class CE loss)."""

    def __init__(self, config, state_dim=STATE_DIM, n_classes=3):
        super().__init__(config)
        self.state_dim = state_dim
        self.n_classes = n_classes
        self.head = nn.Linear(config.n_embd, state_dim * n_classes, bias=False)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits.view(b, t, self.state_dim, self.n_classes)


# ============================= Helpers ========================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize_games(games, seq_len=None):
    if seq_len is None:
        seq_len = GAME_LEN
    tokens = torch.full((len(games), seq_len), PAD_IDX, dtype=torch.long)
    for i, game in enumerate(games):
        n = min(len(game), seq_len)
        for j in range(n):
            tokens[i, j] = STOI[game[j]]
    return tokens


@torch.no_grad()
def extract_activations(model, x, layer):
    b, t = x.size()
    tok = model.tok_emb(x)
    pos = model.pos_emb[:, :t, :]
    h = model.drop(tok + pos)
    for block in model.blocks[:layer]:
        h = block(h)
    return h


def seq_to_state_normal(raw_moves):
    """Standard Othello board state."""
    board = OthelloBoardState()
    states = []
    for move in raw_moves:
        board.umpire(move)
        states.append(np.copy(board.state).astype(np.int8))
    return np.stack(states, axis=0)  # (T, 8, 8)


def load_games(max_files=None):
    import pickle
    files = sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pickle"))
    if max_files is not None:
        files = files[:max_files]
    games = []
    for fname in files:
        with open(os.path.join(SYNTHETIC_DIR, fname), "rb") as f:
            batch = pickle.load(f)
        games.extend(g for g in batch if len(g) == GAME_LEN)
    return games


# ============================= Board Probe (8x8x3) ===========================

def train_board_probe(model, games, device, layer, block_size,
                      pos_start=5, pos_end_offset=5, batch_size=100,
                      lr=1e-4, epochs=15):
    """Train Nanda-style linear probe for Othello board state."""
    d_model = model.pos_emb.shape[-1]
    pos_end = min(GAME_LEN, block_size) - pos_end_offset

    linear_probe = torch.randn(
        MODES, d_model, ROWS, COLS, OPTIONS, device=device,
    ) / np.sqrt(d_model)
    linear_probe.requires_grad = True

    optimizer = torch.optim.AdamW(
        [linear_probe], lr=lr, betas=(0.9, 0.99), weight_decay=0.01,
    )

    num_games = len(games)
    n_eval = max(int(num_games * 0.1), batch_size)
    n_train = num_games - n_eval
    train_games = games[:n_train]
    eval_games = games[n_train:]

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_train)
        for i in tqdm(range(0, n_train, batch_size), desc=f"  epoch {epoch}", leave=False):
            idx = perm[i:i + batch_size]
            batch_games = [train_games[j] for j in idx]

            tokens = tokenize_games(batch_games, seq_len=block_size).to(device)
            state_stack = torch.stack([
                torch.tensor(seq_to_state_normal(g)) for g in batch_games
            ])
            state_stack = state_stack[:, pos_start:pos_end, :, :]

            # One-hot encode: 0=empty, 1=white, 2=black
            one_hot = torch.zeros(
                MODES, len(batch_games), pos_end - pos_start,
                ROWS, COLS, OPTIONS, device=device, dtype=torch.int,
            )
            one_hot[:, ..., 0] = state_stack == 0
            one_hot[:, ..., 1] = state_stack == -1
            one_hot[:, ..., 2] = state_stack == 1

            acts = extract_activations(model, tokens, layer)
            acts = acts[:, pos_start:pos_end]

            probe_out = torch.einsum("bpd,mdrco->mbprco", acts, linear_probe)
            probe_log_probs = probe_out.log_softmax(-1)
            probe_correct_log_probs = (
                (probe_log_probs * one_hot).mean(dim=1).sum(dim=-1) * OPTIONS
            )
            loss_even = -probe_correct_log_probs[0, 0::2].mean(0).sum()
            loss_odd = -probe_correct_log_probs[1, 1::2].mean(0).sum()
            loss_all = -probe_correct_log_probs[2, :].mean(0).sum()
            loss = loss_even + loss_odd + loss_all

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Eval
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for i in range(0, n_eval, batch_size):
                batch_games = eval_games[i:i + batch_size]
                tokens = tokenize_games(batch_games, seq_len=block_size).to(device)
                state_stack = torch.stack([
                    torch.tensor(seq_to_state_normal(g)) for g in batch_games
                ])
                state_stack = state_stack[:, pos_start:pos_end, :, :]

                acts = extract_activations(model, tokens, layer)
                acts = acts[:, pos_start:pos_end]

                probe_out = torch.einsum("bpd,mdrco->mbprco", acts, linear_probe)
                preds = probe_out[2].argmax(dim=-1)
                # Match training one-hot: 0=empty, 1=white, 2=black
                t_state = state_stack.to(device)
                targets = torch.zeros_like(t_state, dtype=torch.long)
                targets[t_state == 0] = 0
                targets[t_state == -1] = 1
                targets[t_state == 1] = 2
                eval_correct += (preds == targets).sum().item()
                eval_total += targets.numel()

        acc = eval_correct / eval_total
        best_acc = max(best_acc, acc)
        print(f"  Epoch {epoch}: eval acc={acc:.4%}", flush=True)

    return best_acc


# ============================= Main ===========================================

def parse_args():
    p = argparse.ArgumentParser(description="Probe state predictor for Othello board state")
    p.add_argument("--ckpt-dir", type=str, required=True,
                    help="Path to state predictor checkpoint directory")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--layer", type=int, default=6, help="Layer to probe")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--max-games", type=int, default=100000)
    p.add_argument("--probe-epochs", type=int, default=15)
    p.add_argument("--probe-batch-size", type=int, default=100)
    p.add_argument("--probe-lr", type=float, default=1e-4)
    p.add_argument("--pos-start", type=int, default=5)
    p.add_argument("--pos-end-offset", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # Find checkpoint
    ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.ckpt_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.ckpt_dir, "final_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found in {args.ckpt_dir}")

    state_dict = torch.load(ckpt_path, map_location=device)
    block_size = state_dict["pos_emb"].shape[1]
    config = GPTConfig(
        VOCAB_SIZE, block_size,
        n_layer=args.layers, n_head=args.n_head, n_embd=args.n_embd,
    )
    # Auto-detect model type from head weight shape
    head_out_dim = state_dict["head.weight"].shape[0]
    if head_out_dim == STATE_DIM:
        model = GPTStatePredictor(config)
        model_type = "state_pred"
    elif head_out_dim == STATE_DIM * 3:
        model = GPTBracketPredictor(config)
        model_type = "bracket_pred"
    else:
        raise ValueError(f"Unknown head output dim: {head_out_dim}")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_type} from {ckpt_path} (block_size={block_size})")

    # Load games
    print(f"Loading games...")
    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    print(f"Using {len(games)} games, probing layer {args.layer}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Probing layer {args.layer} for Othello board state", flush=True)
    print(f"{'='*60}", flush=True)
    acc = train_board_probe(
        model, games, device, args.layer, block_size,
        pos_start=args.pos_start, pos_end_offset=args.pos_end_offset,
        batch_size=args.probe_batch_size, lr=args.probe_lr,
        epochs=args.probe_epochs,
    )

    above = acc - 1/3
    print(f"\nLayer {args.layer}: acc={acc:.4%} (chance=33.3%, +{above:.4%})", flush=True)

    # Save
    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "state_pred_probe_results")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"results_layer{args.layer}.json")
    with open(out_path, "w") as f:
        json.dump({
            "ckpt_dir": args.ckpt_dir,
            "layer": args.layer,
            "num_games": len(games),
            "accuracy": acc,
            "chance": 1/3,
        }, f, indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
