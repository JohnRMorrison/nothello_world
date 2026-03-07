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
OPTIONS = 3  # white=0, empty=1, black=2 (Li et al. encoding: state + 1)
NUM_SQUARES = 64

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


# ============================= Board Probe (Li et al.) ========================

def _collect_activations_and_labels(model, games, device, layer, block_size):
    """Extract per-position activations and board-state labels for all games.

    Returns:
        acts: list of tensors, each (d_model,)
        labels: list of tensors, each (64,) with values in {0,1,2}
                (white=0, empty=1, black=2  — Li's encoding: state+1)
    """
    acts = []
    labels = []
    game_batch = 64
    for start in tqdm(range(0, len(games), game_batch), desc="  extracting", leave=False):
        batch_games = games[start:start + game_batch]
        tokens = tokenize_games(batch_games, seq_len=block_size).to(device)
        with torch.no_grad():
            h = extract_activations(model, tokens, layer)  # (B, T, d)
        for gi, game in enumerate(batch_games):
            board = OthelloBoardState()
            for t, move in enumerate(game):
                if t >= block_size:
                    break
                board.umpire(move)
                acts.append(h[gi, t].cpu())
                # Li encoding: state + 1 → white=0, empty=1, black=2
                labels.append(torch.tensor(board.state.flatten() + 1, dtype=torch.long))
    return acts, labels


def train_board_probe(model, games, device, layer, block_size,
                      lr=1e-3, epochs=16, batch_size=1024):
    """Train Li et al. linear probe (nn.Linear) for Othello board state."""
    d_model = model.pos_emb.shape[-1]

    num_games = len(games)
    n_eval = max(int(num_games * 0.2), 100)
    n_train = num_games - n_eval
    train_games = games[:n_train]
    eval_games = games[n_train:]

    print(f"  Collecting activations for {n_train} train games...", flush=True)
    train_acts, train_labels = _collect_activations_and_labels(
        model, train_games, device, layer, block_size)
    print(f"  Collecting activations for {n_eval} eval games...", flush=True)
    eval_acts, eval_labels = _collect_activations_and_labels(
        model, eval_games, device, layer, block_size)

    print(f"  Train samples: {len(train_acts)}, Eval samples: {len(eval_acts)}", flush=True)

    # Stack into tensors
    train_X = torch.stack(train_acts)        # (N, d_model)
    train_Y = torch.stack(train_labels)      # (N, 64)
    eval_X = torch.stack(eval_acts)
    eval_Y = torch.stack(eval_labels)

    # Probe: nn.Linear(d_model, 64*3)  — same as BatteryProbeClassification
    probe = nn.Linear(d_model, NUM_SQUARES * OPTIONS, bias=True).to(device)
    nn.init.normal_(probe.weight, mean=0.0, std=0.02)
    nn.init.zeros_(probe.bias)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=0)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # Train
        probe.train()
        perm = torch.randperm(len(train_X))
        for i in range(0, len(train_X), batch_size):
            idx = perm[i:i + batch_size]
            x = train_X[idx].to(device)
            y = train_Y[idx].to(device)
            logits = probe(x).reshape(-1, NUM_SQUARES, OPTIONS)  # (B, 64, 3)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, OPTIONS), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval
        probe.eval()
        eval_correct = 0
        eval_total = 0
        eval_losses = []
        with torch.no_grad():
            for i in range(0, len(eval_X), batch_size):
                x = eval_X[i:i + batch_size].to(device)
                y = eval_Y[i:i + batch_size].to(device)
                logits = probe(x).reshape(-1, NUM_SQUARES, OPTIONS)
                eval_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, OPTIONS), y.reshape(-1))
                eval_losses.append(eval_loss.item())
                preds = logits.argmax(dim=-1)
                eval_correct += (preds == y).sum().item()
                eval_total += y.numel()

        acc = eval_correct / eval_total
        best_acc = max(best_acc, acc)
        scheduler.step(np.mean(eval_losses))
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: eval acc={acc:.4%}  loss={np.mean(eval_losses):.5f}  lr={cur_lr:.2e}",
              flush=True)

    return best_acc


# ============================= Main ===========================================

def parse_args():
    p = argparse.ArgumentParser(description="Probe state predictor for Othello board state")
    p.add_argument("--ckpt-dir", type=str, default=None,
                    help="Path to checkpoint directory")
    p.add_argument("--ckpt-path", type=str, default=None,
                    help="Path to checkpoint file (alternative to --ckpt-dir)")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--layer", type=int, default=6, help="Layer to probe")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--max-games", type=int, default=100000)
    p.add_argument("--probe-epochs", type=int, default=16)
    p.add_argument("--probe-batch-size", type=int, default=1024)
    p.add_argument("--probe-lr", type=float, default=1e-3)
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
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        ckpt_path = args.ckpt_path
    elif args.ckpt_dir:
        ckpt_path = None
        for name in ["best.pt", "best_model.pt", "final_model.pt", "random_init.pt"]:
            p = os.path.join(args.ckpt_dir, name)
            if os.path.exists(p):
                ckpt_path = p
                break
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in {args.ckpt_dir}")
    else:
        raise ValueError("Must specify --ckpt-dir or --ckpt-path")

    state_dict = torch.load(ckpt_path, map_location=device)
    block_size = state_dict["pos_emb"].shape[1]
    config = GPTConfig(
        VOCAB_SIZE, block_size,
        n_layer=args.layers, n_head=args.n_head, n_embd=args.n_embd,
    )
    # Auto-detect model type from head weight shape
    head_out_dim = state_dict["head.weight"].shape[0]
    if head_out_dim == VOCAB_SIZE:
        model = GPT(config)
        model_type = "othello_gpt"
    elif head_out_dim == STATE_DIM:
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
        lr=args.probe_lr, epochs=args.probe_epochs,
        batch_size=args.probe_batch_size,
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
