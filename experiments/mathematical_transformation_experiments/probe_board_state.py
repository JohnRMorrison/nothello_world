#!/usr/bin/env python
"""
Train a probe to recover board state from GPTClassifier activations.

Supports two probe types:
  - "linear"    : Neel Nanda's 3-mode linear probe (even/odd/all positions)
  - "nonlinear" : Original authors' two-layer MLP probe (BatteryProbeClassificationTwoLayer)

Works with any layer count. Run from the project root:

    python experiments/mathematical_transformation_experiments/probe_board_state.py \
        --ckpt-path <path_to_best.pt> --layer 2 --probe-type linear

    python experiments/mathematical_transformation_experiments/probe_board_state.py \
        --ckpt-path <path_to_best.pt> --layer 2 --probe-type nonlinear --mid-dim 128
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.othello import OthelloBoardState
from experiments.mathematical_transformation_experiments.dataset import (
    STOI, ITOS, GAME_LEN, PAD_IDX, VOCAB_SIZE,
)
from experiments.mathematical_transformation_experiments.train_boolean_classifier import (
    GPTClassifier, get_device,
)
from mingpt.model import GPTConfig
from mingpt.probe_model import BatteryProbeClassification, BatteryProbeClassificationTwoLayer

ROWS, COLS, OPTIONS = 8, 8, 3
MODES = 3  # even, odd, all (Nanda probe)
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "othello_synthetic")


# ============================= Activation Extraction ==========================

@torch.no_grad()
def extract_activations(model, x, layer):
    """Run model embedding + first `layer` blocks, return residual stream.

    Args:
        model: GPTClassifier (or any GPT subclass with .tok_emb, .pos_emb, .drop, .blocks)
        x: (B, T) long tensor of tokenized moves
        layer: int, number of blocks to run (0 = embedding only)
    Returns:
        (B, T, d_model) activations
    """
    b, t = x.size()
    tok = model.tok_emb(x)
    pos = model.pos_emb[:, :t, :]
    h = model.drop(tok + pos)
    for block in model.blocks[:layer]:
        h = block(h)
    return h


# ============================= Ground Truth ===================================

def seq_to_state_stack(raw_moves):
    """Replay a game, return (num_moves, 8, 8) array with -1/0/1 values."""
    if isinstance(raw_moves, torch.Tensor):
        raw_moves = raw_moves.tolist()
    board = OthelloBoardState()
    states = []
    for move in raw_moves:
        board.umpire(move)
        states.append(np.copy(board.state).astype(np.int8))
    return np.stack(states, axis=0)


def seq_to_state_flat(raw_moves):
    """Replay a game, return list of 64-element arrays with 0/1/2 values.
    (0=white, 1=blank, 2=black — matches get_state() encoding)"""
    board = OthelloBoardState()
    return board.get_gt(raw_moves, "get_state")


def seq_to_age_flat(raw_moves):
    """Replay a game, return list of 64-element age arrays."""
    board = OthelloBoardState()
    return board.get_gt(raw_moves, "get_age")


def state_stack_to_one_hot(state_stack):
    """Convert (B, T, 8, 8) state stack to one-hot (MODES, B, T, 8, 8, 3)."""
    one_hot = torch.zeros(
        MODES, state_stack.shape[0], state_stack.shape[1],
        ROWS, COLS, OPTIONS,
        device=state_stack.device, dtype=torch.int,
    )
    one_hot[:, ..., 0] = state_stack == 0   # empty
    one_hot[:, ..., 1] = state_stack == -1  # white
    one_hot[:, ..., 2] = state_stack == 1   # black
    return one_hot


# ============================= Data Loading ===================================

def load_games(max_files=None):
    """Load raw game sequences from othello_synthetic pickle files.
    Only keeps full-length (60 move) games for uniform tensor shapes."""
    files = sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pickle"))
    if max_files is not None:
        files = files[:max_files]
    games = []
    for fname in files:
        with open(os.path.join(SYNTHETIC_DIR, fname), "rb") as f:
            batch = pickle.load(f)
        games.extend(g for g in batch if len(g) == GAME_LEN)
    return games


def tokenize_games(games):
    """Convert raw game sequences to token tensor (N, GAME_LEN)."""
    tokens = torch.full((len(games), GAME_LEN), PAD_IDX, dtype=torch.long)
    for i, game in enumerate(games):
        n = min(len(game), GAME_LEN)
        for j in range(n):
            tokens[i, j] = STOI[game[j]]
    return tokens


# ============================= Nanda Linear Probe =============================

def train_nanda_probe(model, games, device, args):
    """Train Nanda's 3-mode linear probe on residual stream activations."""
    d_model = args.n_embd
    layer = args.layer
    pos_start = args.pos_start
    pos_end = GAME_LEN - args.pos_end_offset
    batch_size = args.probe_batch_size

    linear_probe = torch.randn(
        MODES, d_model, ROWS, COLS, OPTIONS, device=device,
    ) / np.sqrt(d_model)
    linear_probe.requires_grad = True

    optimizer = torch.optim.AdamW(
        [linear_probe], lr=args.probe_lr, betas=(0.9, 0.99), weight_decay=0.01,
    )

    num_games = len(games)
    # Split: use last 10% for eval
    n_eval = max(int(num_games * 0.1), batch_size)
    n_train = num_games - n_eval
    train_games = games[:n_train]
    eval_games = games[n_train:]

    print(f"Nanda probe: {n_train} train, {n_eval} eval games")
    print(f"Probing layer {layer}, positions {pos_start}:{pos_end}")

    for epoch in range(1, args.probe_epochs + 1):
        # Train
        perm = torch.randperm(n_train)
        train_losses = []
        for i in tqdm(range(0, n_train, batch_size), desc=f"  epoch {epoch} train", leave=False):
            idx = perm[i:i + batch_size]
            batch_games = [train_games[j] for j in idx]
            actual_bs = len(batch_games)

            tokens = tokenize_games(batch_games).to(device)
            state_stack = torch.stack([
                torch.tensor(seq_to_state_stack(g)) for g in batch_games
            ])
            state_stack = state_stack[:, pos_start:pos_end, :, :]
            state_stack_one_hot = state_stack_to_one_hot(state_stack).to(device)

            with torch.no_grad():
                acts = extract_activations(model, tokens, layer)
                acts = acts[:, pos_start:pos_end]

            probe_out = torch.einsum(
                "bpd,mdrco->mbprco", acts, linear_probe,
            )
            probe_log_probs = probe_out.log_softmax(-1)
            probe_correct_log_probs = (
                (probe_log_probs * state_stack_one_hot).mean(dim=1).sum(dim=-1) * OPTIONS
            )

            loss_even = -probe_correct_log_probs[0, 0::2].mean(0).sum()
            loss_odd = -probe_correct_log_probs[1, 1::2].mean(0).sum()
            loss_all = -probe_correct_log_probs[2, :].mean(0).sum()
            loss = loss_even + loss_odd + loss_all

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

        # Eval
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for i in range(0, n_eval, batch_size):
                batch_games = eval_games[i:i + batch_size]
                tokens = tokenize_games(batch_games).to(device)
                state_stack = torch.stack([
                    torch.tensor(seq_to_state_stack(g)) for g in batch_games
                ])
                state_stack = state_stack[:, pos_start:pos_end, :, :]

                acts = extract_activations(model, tokens, layer)
                acts = acts[:, pos_start:pos_end]

                probe_out = torch.einsum("bpd,mdrco->mbprco", acts, linear_probe)
                # Use mode 2 (all positions) for accuracy
                preds = probe_out[2].argmax(dim=-1)  # (B, T, 8, 8)
                # Map -1/0/1 to 0/1/2 for comparison
                # Match training one-hot: 0=empty, 1=white, 2=black
                t_state = state_stack.to(device)
                targets = torch.zeros_like(t_state, dtype=torch.long)
                targets[t_state == 0] = 0
                targets[t_state == -1] = 1
                targets[t_state == 1] = 2
                eval_correct += (preds == targets).sum().item()
                eval_total += targets.numel()

        acc = eval_correct / eval_total
        mean_loss = np.mean(train_losses)
        print(f"  Epoch {epoch}: train loss={mean_loss:.4f}, eval acc={acc:.4%}")

    return linear_probe, acc


# ============================= Original Authors' Probe ========================

class ProbingDataset(Dataset):
    """Flat dataset of (activation_vector, board_state, cell_ages) triples."""
    def __init__(self, activations, states, ages):
        self.act = activations
        self.y = states
        self.age = ages

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.act[idx],
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(self.age[idx], dtype=torch.long),
        )


def build_probing_dataset(model, games, device, layer, batch_size=64):
    """Extract activations and ground truth for all positions in all games."""
    act_list = []
    state_list = []
    age_list = []

    tokens_all = tokenize_games(games)

    for i in tqdm(range(0, len(games), batch_size), desc="  extracting activations"):
        batch_games = games[i:i + batch_size]
        batch_tokens = tokens_all[i:i + batch_size].to(device)

        with torch.no_grad():
            acts = extract_activations(model, batch_tokens, layer).cpu()

        for j, game in enumerate(batch_games):
            n = min(len(game), GAME_LEN)
            states = seq_to_state_flat(game[:n])
            ages = seq_to_age_flat(game[:n])
            for pos in range(n):
                act_list.append(acts[j, pos])
                state_list.append(states[pos])
                age_list.append(ages[pos])

    return ProbingDataset(act_list, state_list, age_list)


def train_original_probe(model, games, device, args):
    """Train the original authors' probe (linear or two-layer MLP)."""
    layer = args.layer

    print(f"Building probing dataset (layer {layer})...")
    dataset = build_probing_dataset(model, games, device, layer, batch_size=args.probe_batch_size)
    print(f"Dataset: {len(dataset)} (activation, board_state) pairs")

    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    train_ds, test_ds = random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    if args.nonlinear:
        probe = BatteryProbeClassificationTwoLayer(
            device, probe_class=3, num_task=64,
            mid_dim=args.mid_dim, input_dim=args.n_embd,
        )
    else:
        probe = BatteryProbeClassification(
            device, probe_class=3, num_task=64, input_dim=args.n_embd,
        )
    probe = probe.to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.75, patience=0,
    )

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=1024, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=1024, num_workers=0, pin_memory=False)

    best_acc = 0.0
    history = []

    for epoch in range(1, args.probe_epochs + 1):
        # Train
        probe.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        for act, y, age in tqdm(train_loader, desc=f"  epoch {epoch} train", leave=False):
            act, y = act.to(device), y.to(device)
            logits, loss = probe(act, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
            preds = logits.argmax(dim=-1)
            train_correct += (preds == y).sum().item()
            train_total += y.numel()

        # Eval
        probe.eval()
        test_losses = []
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for act, y, age in test_loader:
                act, y = act.to(device), y.to(device)
                logits, loss = probe(act, y)
                test_losses.append(loss.item())
                preds = logits.argmax(dim=-1)
                test_correct += (preds == y).sum().item()
                test_total += y.numel()

        test_loss = np.mean(test_losses)
        scheduler.step(test_loss)
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total

        print(f"  Epoch {epoch}: train loss={np.mean(train_losses):.4f} acc={train_acc:.4%}, "
              f"test loss={test_loss:.4f} acc={test_acc:.4%}")

        history.append({
            "epoch": epoch, "train_loss": float(np.mean(train_losses)),
            "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc,
        })

        if test_acc > best_acc:
            best_acc = test_acc

    return probe, best_acc, history


# ============================= Main ===========================================

def parse_args():
    p = argparse.ArgumentParser(description="Probe GPTClassifier for board state")
    # Model
    p.add_argument("--ckpt-path", type=str, required=True,
                   help="Path to GPTClassifier checkpoint (best.pt or epoch_XXX.pt)")
    p.add_argument("--layers", type=int, default=8, help="Number of transformer layers in the model")
    p.add_argument("--n-embd", type=int, default=512, help="Embedding dimension of the model")
    p.add_argument("--n-head", type=int, default=8, help="Number of attention heads in the model")
    p.add_argument("--n-transforms", type=int, default=1,
                   help="Number of model outputs (auto-detected from args.json if available)")
    # Probe
    p.add_argument("--layer", type=int, required=True,
                   help="Which layer to probe (e.g. 2 for after 2nd block)")
    p.add_argument("--probe-type", type=str, default="linear", choices=["linear", "nonlinear"],
                   help="'linear' = Nanda 3-mode probe, 'nonlinear' = original authors' MLP probe")
    p.add_argument("--nonlinear", action="store_true",
                   help="For original probe: use two-layer MLP instead of single linear")
    p.add_argument("--mid-dim", type=int, default=128,
                   help="Hidden dim for nonlinear probe (only used with --probe-type nonlinear)")
    # Data
    p.add_argument("--max-files", type=int, default=1,
                   help="Number of data pickle files to load")
    p.add_argument("--max-games", type=int, default=50000,
                   help="Max games to use for probing")
    # Training
    p.add_argument("--probe-epochs", type=int, default=4)
    p.add_argument("--probe-batch-size", type=int, default=100)
    p.add_argument("--probe-lr", type=float, default=1e-4,
                   help="Learning rate (Nanda probe only; original uses 1e-3)")
    p.add_argument("--pos-start", type=int, default=5,
                   help="Skip first N positions (Nanda probe)")
    p.add_argument("--pos-end-offset", type=int, default=5,
                   help="Skip last N positions (Nanda probe)")
    # Output
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # ---- Auto-detect model config from args.json if available ----
    ckpt_dir = os.path.dirname(args.ckpt_path)
    args_json_path = os.path.join(ckpt_dir, "args.json")
    n_transforms = args.n_transforms
    if os.path.exists(args_json_path):
        with open(args_json_path) as f:
            train_args = json.load(f)
        # Override model architecture from training config unless user explicitly set them
        args.layers = train_args.get("layers", args.layers)
        args.n_embd = train_args.get("n_embd", args.n_embd)
        args.n_head = train_args.get("n_head", args.n_head)
        n_transforms = train_args.get("n_transforms", n_transforms)
        print(f"Auto-detected model config from {args_json_path}")

    # ---- Load model ----
    print(f"Loading model: {args.layers}L/{args.n_embd}d/{args.n_head}h/{n_transforms} outputs")
    config = GPTConfig(
        VOCAB_SIZE, GAME_LEN,
        n_layer=args.layers, n_head=args.n_head, n_embd=args.n_embd,
    )
    model = GPTClassifier(config, n_outputs=n_transforms)

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt_path}")

    assert args.layer <= args.layers, (
        f"Cannot probe layer {args.layer} of a {args.layers}-layer model"
    )

    # ---- Load games ----
    print(f"Loading games (max_files={args.max_files})...")
    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    print(f"Using {len(games)} games")

    # ---- Output dir ----
    if args.output_dir is None:
        ckpt_parent = os.path.dirname(args.ckpt_path)
        args.output_dir = os.path.join(
            ckpt_parent, f"probe_L{args.layer}_{args.probe_type}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Train probe ----
    t0 = time.time()
    if args.probe_type == "linear":
        probe, acc = train_nanda_probe(model, games, device, args)
        probe_path = os.path.join(args.output_dir, "linear_probe.pth")
        torch.save(probe, probe_path)
        print(f"\nSaved linear probe to {probe_path}")
        history = None
    else:
        probe, acc, history = train_original_probe(model, games, device, args)
        probe_path = os.path.join(args.output_dir, "probe.pt")
        torch.save(probe.state_dict(), probe_path)
        print(f"\nSaved nonlinear probe to {probe_path}")

    elapsed = time.time() - t0
    print(f"Best accuracy: {acc:.4%} ({elapsed:.0f}s)")

    # ---- Save metadata ----
    meta = {
        "probe_type": args.probe_type,
        "layer": args.layer,
        "model_layers": args.layers,
        "n_embd": args.n_embd,
        "n_transforms": n_transforms,
        "ckpt_path": args.ckpt_path,
        "num_games": len(games),
        "best_acc": acc,
    }
    if args.probe_type == "nonlinear":
        meta["nonlinear"] = args.nonlinear
        meta["mid_dim"] = args.mid_dim
    if history:
        meta["history"] = history
    with open(os.path.join(args.output_dir, "probe_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
