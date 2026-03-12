#!/usr/bin/env python
"""
Probe Othello-GPT for variant board states.

Tests whether Othello-GPT's activations can decode board states from games
played under different rules. If they can, it shows that board state
decodability is not decisive evidence of a world model.

Variants:
  1. no_flip:       Othello but pieces never flip
  2. row_flip:      Othello but flips only along rows (not cols/diags)
  3. vtable_flip:   Arbitrary flip game (V_table based, 60-dim binary state)
  4. gravity:       Pieces fall down in their column (Connect Four style)
  5. shuffled:      Real Othello board state but shifted in time by an offset

Usage (from project root):
    python experiments/mathematical_transformation_experiments/probe_variant_boards.py \
        --ckpt-path ckpts/gpt_synthetic.ckpt --layer 6

Probes all variants and reports accuracy for each.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib, types
# Import OthelloBoardState without triggering data/__init__.py (which requires pgn)
_spec = importlib.util.spec_from_file_location(
    "othello_module", os.path.join(PROJECT_ROOT, "data", "othello.py"),
    submodule_search_locations=[],
)
_othello_mod = importlib.util.module_from_spec(_spec)
# Stub out pgn, seaborn, matplotlib before executing the module
# (seaborn/matplotlib hang on NFS due to slow directory scans)
for _stub in ["pgn", "seaborn", "matplotlib", "matplotlib.pyplot",
              "matplotlib.patches", "matplotlib.collections", "matplotlib.colors",
              "psutil"]:
    if _stub not in sys.modules:
        sys.modules[_stub] = types.ModuleType(_stub)
_spec.loader.exec_module(_othello_mod)
OthelloBoardState = _othello_mod.OthelloBoardState
from mingpt.model import GPT, GPTConfig

GAME_LEN = 60
VOCAB_SIZE = 61
PAD_IDX = 0
ROWS, COLS = 8, 8
OPTIONS = 3  # white=0, empty=1, black=2 (Li et al. encoding: state + 1)
NUM_SQUARES = 64

_VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})
STOI = {-100: 0}
STOI.update({move: i + 1 for i, move in enumerate(_VALID_MOVES)})
ITOS = {v: k for k, v in STOI.items()}

SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "othello_synthetic")

EIGHTS = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
ROW_DIRS = [[0, 1], [0, -1]]  # only left/right


# ============================= Variant Simulators =============================

def seq_to_state_normal(raw_moves):
    """Standard Othello — baseline."""
    board = OthelloBoardState()
    states = []
    for move in raw_moves:
        board.umpire(move)
        states.append(np.copy(board.state).astype(np.int8))
    return np.stack(states, axis=0)  # (T, 8, 8)


def _find_flips(board_state, r, c, color, directions):
    """Find pieces to flip in given directions."""
    tbf = []
    for direction in directions:
        buffer = []
        cur_r, cur_c = r, c
        while True:
            cur_r += direction[0]
            cur_c += direction[1]
            if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                break
            if board_state[cur_r, cur_c] == 0:
                break
            elif board_state[cur_r, cur_c] == color:
                tbf.extend(buffer)
                break
            else:
                buffer.append((cur_r, cur_c))
    return tbf


def _determine_color(board_state, r, c, next_color):
    """Determine which color plays, handling forfeits like Othello."""
    tbf = _find_flips(board_state, r, c, next_color, EIGHTS)
    if tbf:
        return next_color, tbf
    # Try other color (forfeit)
    other = -next_color
    tbf = _find_flips(board_state, r, c, other, EIGHTS)
    return other, tbf


def seq_to_state_no_flip(raw_moves):
    """Variant 1: Place pieces, never flip. Alternating colors."""
    board = np.zeros((8, 8), dtype=np.int8)
    # Initial Othello setup
    board[3, 4] = 1; board[3, 3] = -1
    board[4, 3] = 1; board[4, 4] = -1
    next_color = 1
    states = []
    for move in raw_moves:
        r, c = move // 8, move % 8
        if board[r, c] == 0:
            # Determine color (use normal Othello logic for color assignment)
            # but don't actually flip
            color, _ = _determine_color(board, r, c, next_color)
            board[r, c] = color
            next_color = -color
        states.append(np.copy(board))
    return np.stack(states, axis=0)


def seq_to_state_row_flip(raw_moves):
    """Variant 2: Only flip along rows (horizontal), not columns or diagonals."""
    board = np.zeros((8, 8), dtype=np.int8)
    board[3, 4] = 1; board[3, 3] = -1
    board[4, 3] = 1; board[4, 4] = -1
    next_color = 1
    states = []
    for move in raw_moves:
        r, c = move // 8, move % 8
        if board[r, c] != 0:
            states.append(np.copy(board))
            continue
        # Try with next_color using ALL directions (for color determination)
        color, _ = _determine_color(board, r, c, next_color)
        # But only flip along rows
        tbf = _find_flips(board, r, c, color, ROW_DIRS)
        for fr, fc in tbf:
            board[fr, fc] = color
        board[r, c] = color
        next_color = -color
        states.append(np.copy(board))
    return np.stack(states, axis=0)


def seq_to_state_gravity(raw_moves):
    """Variant 4: Pieces fall to the bottom of their column (like Connect Four).
    Flips still happen normally based on the landed position."""
    board = np.zeros((8, 8), dtype=np.int8)
    board[3, 4] = 1; board[3, 3] = -1
    board[4, 3] = 1; board[4, 4] = -1
    next_color = 1
    states = []
    for move in raw_moves:
        _, c = move // 8, move % 8
        # Find lowest empty row in this column
        land_r = -1
        for row in range(7, -1, -1):
            if board[row, c] == 0:
                land_r = row
                break
        if land_r == -1:
            # Column full, just place at original position
            r = move // 8
            if board[r, c] == 0:
                color, _ = _determine_color(board, r, c, next_color)
                board[r, c] = color
                next_color = -color
            states.append(np.copy(board))
            continue
        # Determine color using original Othello logic at the landed position
        color, _ = _determine_color(board, land_r, c, next_color)
        # Flip based on landed position
        tbf = _find_flips(board, land_r, c, color, EIGHTS)
        for fr, fc in tbf:
            board[fr, fc] = color
        board[land_r, c] = color
        next_color = -color
        states.append(np.copy(board))
    return np.stack(states, axis=0)


def make_shuffled_simulator(offset=7):
    """Variant 5: Real Othello board state but shifted in time."""
    def seq_to_state_shuffled(raw_moves):
        normal = seq_to_state_normal(raw_moves)  # (T, 8, 8)
        T = len(normal)
        shifted = np.zeros_like(normal)
        for t in range(T):
            src = (t + offset) % T
            shifted[t] = normal[src]
        return shifted
    return seq_to_state_shuffled


# V_table variant (60-dim binary, not 8x8 board)
def make_vtable_simulator(vtable_seed=42):
    """Variant 3: Arbitrary V_table flip game. Returns (T, 60) binary states."""
    from experiments.mathematical_transformation_experiments.train_state_predictor import (
        generate_vtable, STATE_DIM,
    )
    vtable = generate_vtable(vtable_seed)

    def seq_to_state_vtable(raw_moves):
        s = np.ones(STATE_DIM, dtype=np.float32)
        states = []
        for move in raw_moves:
            token = STOI[move]
            v = vtable[token]
            flip = v == 1
            s = np.where(flip, -s, s)
            states.append(((s + 1) / 2).astype(np.uint8))
        return np.stack(states, axis=0)  # (T, 60)
    return seq_to_state_vtable


# ============================= Shared Probing ================================

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

def _collect_board_activations_and_labels(model, games, device, layer, simulator, block_size):
    """Extract per-position activations and board-state labels."""
    acts = []
    labels = []
    game_batch = 64
    for start in tqdm(range(0, len(games), game_batch), desc="  extracting", leave=False):
        batch_games = games[start:start + game_batch]
        tokens = tokenize_games(batch_games, seq_len=block_size).to(device)
        with torch.no_grad():
            h = extract_activations(model, tokens, layer)
        for gi, game in enumerate(batch_games):
            states = simulator(game)  # (T, 8, 8)
            for t in range(min(len(game), block_size)):
                acts.append(h[gi, t].cpu())
                # Li encoding: state + 1 → white=0, empty=1, black=2
                labels.append(torch.tensor(states[t].flatten() + 1, dtype=torch.long))
    return acts, labels


def train_board_probe(model, games, device, layer, simulator,
                      lr=1e-3, epochs=16, batch_size=1024, block_size=59):
    """Train Li et al. linear probe for an 8x8 board state variant."""
    d_model = model.pos_emb.shape[-1]

    num_games = len(games)
    n_eval = max(int(num_games * 0.2), 100)
    n_train = num_games - n_eval
    train_games = games[:n_train]
    eval_games = games[n_train:]

    print(f"  Collecting activations for {n_train} train games...", flush=True)
    train_acts, train_labels = _collect_board_activations_and_labels(
        model, train_games, device, layer, simulator, block_size)
    print(f"  Collecting activations for {n_eval} eval games...", flush=True)
    eval_acts, eval_labels = _collect_board_activations_and_labels(
        model, eval_games, device, layer, simulator, block_size)

    print(f"  Train samples: {len(train_acts)}, Eval samples: {len(eval_acts)}", flush=True)

    train_X = torch.stack(train_acts)
    train_Y = torch.stack(train_labels)
    eval_X = torch.stack(eval_acts)
    eval_Y = torch.stack(eval_labels)

    probe = nn.Linear(d_model, NUM_SQUARES * OPTIONS, bias=True).to(device)
    nn.init.normal_(probe.weight, mean=0.0, std=0.02)
    nn.init.zeros_(probe.bias)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=0)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(len(train_X))
        for i in range(0, len(train_X), batch_size):
            idx = perm[i:i + batch_size]
            x = train_X[idx].to(device)
            y = train_Y[idx].to(device)
            logits = probe(x).reshape(-1, NUM_SQUARES, OPTIONS)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, OPTIONS), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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


# ============================= V_table Probe (60-dim binary) ==================

def _collect_vtable_activations_and_labels(model, games, device, layer, simulator, block_size):
    """Extract per-position activations and vtable binary labels."""
    acts = []
    labels = []
    game_batch = 64
    for start in tqdm(range(0, len(games), game_batch), desc="  extracting", leave=False):
        batch_games = games[start:start + game_batch]
        tokens = tokenize_games(batch_games, seq_len=block_size).to(device)
        with torch.no_grad():
            h = extract_activations(model, tokens, layer)
        for gi, game in enumerate(batch_games):
            states = simulator(game)  # (T, 60)
            for t in range(min(len(game), block_size)):
                acts.append(h[gi, t].cpu())
                labels.append(torch.tensor(states[t], dtype=torch.float32))
    return acts, labels


def train_vtable_probe(model, games, device, layer, simulator,
                       lr=1e-3, epochs=16, batch_size=1024, block_size=59):
    """Train a linear probe for the 60-dim binary V_table state."""
    d_model = model.pos_emb.shape[-1]
    state_dim = 60

    num_games = len(games)
    n_eval = max(int(num_games * 0.2), 100)
    n_train = num_games - n_eval
    train_games = games[:n_train]
    eval_games = games[n_train:]

    print(f"  Collecting activations for {n_train} train games...", flush=True)
    train_acts, train_labels = _collect_vtable_activations_and_labels(
        model, train_games, device, layer, simulator, block_size)
    print(f"  Collecting activations for {n_eval} eval games...", flush=True)
    eval_acts, eval_labels = _collect_vtable_activations_and_labels(
        model, eval_games, device, layer, simulator, block_size)

    print(f"  Train samples: {len(train_acts)}, Eval samples: {len(eval_acts)}", flush=True)

    train_X = torch.stack(train_acts)
    train_Y = torch.stack(train_labels)
    eval_X = torch.stack(eval_acts)
    eval_Y = torch.stack(eval_labels)

    probe = nn.Linear(d_model, state_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=0)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(len(train_X))
        for i in range(0, len(train_X), batch_size):
            idx = perm[i:i + batch_size]
            x = train_X[idx].to(device)
            y = train_Y[idx].to(device)
            logits = probe(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        eval_correct = 0
        eval_total = 0
        eval_losses = []
        with torch.no_grad():
            for i in range(0, len(eval_X), batch_size):
                x = eval_X[i:i + batch_size].to(device)
                y = eval_Y[i:i + batch_size].to(device)
                logits = probe(x)
                eval_loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
                eval_losses.append(eval_loss.item())
                preds = (logits > 0).float()
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
    p = argparse.ArgumentParser(description="Probe Othello-GPT for variant board states")
    p.add_argument("--ckpt-path", type=str, default="ckpts/gpt_synthetic.ckpt",
                    help="Path to Othello-GPT checkpoint")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--layer", type=int, default=6, help="Layer to probe")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--max-games", type=int, default=100000)
    p.add_argument("--probe-epochs", type=int, default=16)
    p.add_argument("--probe-batch-size", type=int, default=1024)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--time-offset", type=int, default=7,
                    help="Time offset for shuffled variant")
    p.add_argument("--vtable-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", type=str, nargs="+",
                    default=["normal", "no_flip", "row_flip", "vtable_flip",
                             "gravity", "shuffled"],
                    help="Which variants to probe")
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # Load Othello-GPT
    # Othello-GPT uses block_size=59 (next-token prediction: 59 inputs -> 59 outputs)
    state_dict = torch.load(args.ckpt_path, map_location=device)
    # Auto-detect block_size from checkpoint
    block_size = state_dict["pos_emb"].shape[1]
    config = GPTConfig(
        VOCAB_SIZE, block_size,
        n_layer=args.layers, n_head=args.n_head, n_embd=args.n_embd,
    )
    model = GPT(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded Othello-GPT from {args.ckpt_path}")

    # Load games
    print(f"Loading games (max_files={args.max_files})...")
    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    print(f"Using {len(games)} games, probing layer {args.layer}, block_size={block_size}")

    # Define variants
    board_variants = {
        "normal": ("Standard Othello (baseline)", seq_to_state_normal),
        "no_flip": ("No Flip Othello", seq_to_state_no_flip),
        "row_flip": ("Row-Only Flip Othello", seq_to_state_row_flip),
        "gravity": ("Gravity Othello", seq_to_state_gravity),
        "shuffled": (f"Shuffled Othello (offset={args.time_offset})",
                     make_shuffled_simulator(args.time_offset)),
    }

    results = {}

    for name in args.variants:
        if name == "vtable_flip":
            desc = "V_table Flip Game (60-dim binary)"
            print(f"\n{'='*60}")
            print(f"Probing: {desc}")
            print(f"{'='*60}")
            simulator = make_vtable_simulator(args.vtable_seed)
            acc = train_vtable_probe(
                model, games, device, args.layer, simulator,
                lr=args.probe_lr, epochs=args.probe_epochs,
                batch_size=args.probe_batch_size, block_size=block_size,
            )
            results[name] = {"description": desc, "accuracy": acc, "chance": 0.5}
        elif name in board_variants:
            desc, simulator = board_variants[name]
            print(f"\n{'='*60}")
            print(f"Probing: {desc}")
            print(f"{'='*60}")
            acc = train_board_probe(
                model, games, device, args.layer, simulator,
                lr=args.probe_lr, epochs=args.probe_epochs,
                batch_size=args.probe_batch_size, block_size=block_size,
            )
            results[name] = {"description": desc, "accuracy": acc, "chance": 1/3}
        else:
            print(f"Unknown variant: {name}, skipping")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY (layer {args.layer})")
    print(f"{'='*60}")
    for name, r in results.items():
        above = r["accuracy"] - r["chance"]
        print(f"  {r['description']:40s}  acc={r['accuracy']:.4%}  "
              f"(chance={r['chance']:.1%}, +{above:.4%})")

    # Save
    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "variant_probe_results")
    os.makedirs(args.output_dir, exist_ok=True)
    for name, r in results.items():
        out_path = os.path.join(args.output_dir, f"results_{name}_layer{args.layer}.json")
        with open(out_path, "w") as f:
            json.dump({
                "layer": args.layer,
                "variant": name,
                "num_games": len(games),
                "ckpt_path": args.ckpt_path,
                **r,
            }, f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
