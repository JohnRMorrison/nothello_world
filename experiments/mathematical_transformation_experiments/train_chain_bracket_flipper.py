#!/usr/bin/env python
"""
Train a causal transformer on the Chain Bracket Flipping Game.

Like the bracket flipping game, but with cascading flips: after the initial
round of bracket flips, newly-flipped moves can trigger further brackets
(scanning both forward and backward). Cascades continue until no new flips
occur or 5 rounds are reached.

Chain Bracket Flipping Game rules:
  - Each move ID is assigned to one of 5 groups (fixed random table).
  - Odd steps are black (+1), even steps are white (-1).
  - Round 1: scan backward for brackets (same as regular bracket game).
  - Rounds 2+: each newly-flipped move scans both forward and backward for
    new brackets. A move can only flip once per step.
  - Max 5 cascade rounds per step.

The model predicts 3-class state per dimension at each position (cross-entropy).

Run from the project root:

    python -m experiments.mathematical_transformation_experiments.train_chain_bracket_flipper [OPTIONS]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.mathematical_transformation_experiments.dataset import (
    GAME_LEN,
    PAD_IDX,
    STOI,
    VOCAB_SIZE,
)
from experiments.mathematical_transformation_experiments.generate_labels import (
    SYNTHETIC_DIR,
    list_synthetic_files,
    load_pickle,
)
from mingpt.model import GPT, GPTConfig

STATE_DIM = 60
N_CLASSES = 3
N_GROUPS = 5
MAX_CHAIN_ROUNDS = 5
GTABLE_DIR = os.path.join(SCRIPT_DIR, "vtables")


# ============================= Group Table ====================================

def generate_group_table(seed=42):
    rng = np.random.default_rng(seed)
    table = np.full(VOCAB_SIZE, -1, dtype=np.int8)
    table[1:] = rng.integers(0, N_GROUPS, size=VOCAB_SIZE - 1).astype(np.int8)
    return table


def get_group_table(seed=42):
    os.makedirs(GTABLE_DIR, exist_ok=True)
    path = os.path.join(GTABLE_DIR, f"group_table_seed{seed}.npy")
    if os.path.exists(path):
        table = np.load(path)
        print(f"Loaded group table from {path}")
    else:
        table = generate_group_table(seed)
        np.save(path, table)
        print(f"Generated and saved group table to {path}")
    return table


# ============================= State computation ==============================

def _apply_brackets_from(pos, color_c, color, tokens, group_table,
                         last_valid, flipped_this_step, backward_only):
    """Scan from position pos for brackets and apply flips.

    Args:
        pos: position to scan from
        color_c: the color to match (current color of the move at pos)
        color: mutable array of current colors
        tokens: token array
        group_table: group assignment table
        last_valid: last non-pad position index
        flipped_this_step: set of positions already flipped this step
        backward_only: if True, only scan backward

    Returns:
        set of newly flipped positions
    """
    newly_flipped = set()
    opposite = -color_c
    group_p = group_table[tokens[pos]]

    # Backward scan: j from pos-1 down to 0
    for j in range(pos - 1, -1, -1):
        if group_table[tokens[j]] != group_p:
            continue
        if j in flipped_this_step:
            continue
        if color[j] != color_c:
            continue
        # Check for opposite-color move between j and pos
        has_opposite = False
        for k in range(j + 1, pos):
            if color[k] == opposite:
                has_opposite = True
                break
        if not has_opposite:
            continue
        # Bracket fires
        for k in range(j + 1, pos):
            if color[k] == opposite and k not in flipped_this_step:
                color[k] = color_c
                newly_flipped.add(k)

    # Forward scan (rounds 2+ only)
    if not backward_only:
        for j in range(pos + 1, last_valid + 1):
            if group_table[tokens[j]] != group_p:
                continue
            if j in flipped_this_step:
                continue
            if color[j] != color_c:
                continue
            has_opposite = False
            for k in range(pos + 1, j):
                if color[k] == opposite:
                    has_opposite = True
                    break
            if not has_opposite:
                continue
            for k in range(pos + 1, j):
                if color[k] == opposite and k not in flipped_this_step:
                    color[k] = color_c
                    newly_flipped.add(k)

    return newly_flipped


def _compute_single_game_chain(tokens, group_table):
    """Compute chain bracket flipping game state for a single game.

    Args:
        tokens: (T,) int array of token indices (0=pad, 1-60=moves)
        group_table: (VOCAB_SIZE,) int8 array

    Returns:
        states: (T, STATE_DIM) int8 array, values in {-1, 0, +1}
    """
    T = len(tokens)
    color = np.zeros(T, dtype=np.int8)
    states = np.zeros((T, STATE_DIM), dtype=np.int8)

    last_valid = -1
    for t in range(T):
        tok = tokens[t]
        if tok == PAD_IDX:
            break
        last_valid = t

        current_player = np.int8(1) if (t + 1) % 2 == 1 else np.int8(-1)
        color[t] = current_player

        flipped_this_step = set()

        # Round 1: scan backward from t
        newly_flipped = _apply_brackets_from(
            t, current_player, color, tokens, group_table,
            last_valid, flipped_this_step, backward_only=True,
        )
        flipped_this_step |= newly_flipped

        # Rounds 2-5: cascade
        for _round in range(2, MAX_CHAIN_ROUNDS + 1):
            if not newly_flipped:
                break
            next_flipped = set()
            for pos in sorted(newly_flipped):
                new_color = color[pos]
                round_flips = _apply_brackets_from(
                    pos, new_color, color, tokens, group_table,
                    last_valid, flipped_this_step, backward_only=False,
                )
                next_flipped |= round_flips
                flipped_this_step |= round_flips
            newly_flipped = next_flipped

        # Record state
        states[t, :t + 1] = color[:t + 1]

    # Pad positions: copy last valid state
    if last_valid >= 0:
        for t in range(last_valid + 1, T):
            states[t] = states[last_valid]

    return states


def compute_chain_bracket_states(token_seqs, group_table):
    N, T = token_seqs.shape
    states = np.zeros((N, T, STATE_DIM), dtype=np.int8)
    for i in range(N):
        states[i] = _compute_single_game_chain(token_seqs[i], group_table)
    return states


# ============================= Dataset ========================================

def tokenize_game(game_raw):
    tokens = np.full(GAME_LEN, PAD_IDX, dtype=np.int64)
    n = min(len(game_raw), GAME_LEN)
    for j in range(n):
        tokens[j] = STOI[game_raw[j]]
    return tokens


class ChainBracketStateDataset(Dataset):
    """Othello token sequences paired with chain bracket flipping game state targets."""

    def __init__(self, group_table, max_files=None):
        all_tokens = []
        all_states = []

        files = list_synthetic_files()
        if max_files is not None:
            files = files[:max_files]

        for fname in tqdm(files, desc="Loading games"):
            games_raw = load_pickle(os.path.join(SYNTHETIC_DIR, fname))
            tokens = np.array(
                [tokenize_game(g) for g in games_raw], dtype=np.int64
            )
            states = compute_chain_bracket_states(tokens, group_table)

            all_tokens.append(tokens)
            all_states.append(states)

        self.tokens = np.concatenate(all_tokens, axis=0)
        self.states = np.concatenate(all_states, axis=0)

        print(
            f"Dataset: {len(self)} games, "
            f"tokens {self.tokens.shape}, states {self.states.shape}"
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.tokens[idx].copy())
        y = torch.from_numpy((self.states[idx] + 1).astype(np.int64))
        return x, y

    def split(self, train_frac=0.8, seed=42):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self))
        n_train = int(len(self) * train_frac)
        return (
            Subset(self, perm[:n_train].tolist()),
            Subset(self, perm[n_train:].tolist()),
        )


# ============================= Model ==========================================

class GPTChainBracketPredictor(GPT):
    """Causal transformer predicting 3-class state per dimension at each position."""

    def __init__(self, config, state_dim=STATE_DIM, n_classes=N_CLASSES):
        super().__init__(config)
        self.state_dim = state_dim
        self.n_classes = n_classes
        self.head = nn.Linear(config.n_embd, state_dim * n_classes, bias=False)
        self.head.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.head(x)
        logits = logits.view(b, t, self.state_dim, self.n_classes)
        return logits


# ============================= Device =========================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================= Training =======================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_elements = 0

    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)

        non_pad = (x != PAD_IDX)
        mask = non_pad.unsqueeze(-1).expand_as(y)

        logits_flat = logits.reshape(-1, N_CLASSES)
        targets_flat = y.reshape(-1)
        mask_flat = mask.reshape(-1)

        loss = nn.functional.cross_entropy(
            logits_flat[mask_flat], targets_flat[mask_flat]
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n = mask_flat.sum().item()
        total_loss += loss.item() * n
        preds = logits_flat[mask_flat].argmax(dim=-1)
        total_correct += (preds == targets_flat[mask_flat]).sum().item()
        total_elements += n

    return total_loss / total_elements, total_correct / total_elements


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_elements = 0

    for x, y in tqdm(loader, desc="  eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)

        non_pad = (x != PAD_IDX)
        mask = non_pad.unsqueeze(-1).expand_as(y)

        logits_flat = logits.reshape(-1, N_CLASSES)
        targets_flat = y.reshape(-1)
        mask_flat = mask.reshape(-1)

        loss = nn.functional.cross_entropy(
            logits_flat[mask_flat], targets_flat[mask_flat]
        )

        n = mask_flat.sum().item()
        total_loss += loss.item() * n
        preds = logits_flat[mask_flat].argmax(dim=-1)
        total_correct += (preds == targets_flat[mask_flat]).sum().item()
        total_elements += n

    return total_loss / total_elements, total_correct / total_elements


# ============================= Main ===========================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train causal transformer on chain bracket flipping game"
    )
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-files", type=int, default=5)
    p.add_argument("--group-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--save-random-init", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # ---- Group table ----
    group_table = get_group_table(seed=args.group_seed)
    for g in range(N_GROUPS):
        count = (group_table[1:] == g).sum()
        print(f"  Group {g}: {count} members")

    # ---- Checkpoint dir ----
    if args.ckpt_dir is None:
        name = f"chain_bracket_pred_gseed{args.group_seed}_{args.layers}L_{args.n_embd}d"
        args.ckpt_dir = os.path.join(SCRIPT_DIR, "ckpts", name)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- Save random init and exit if requested ----
    if args.save_random_init:
        config = GPTConfig(
            VOCAB_SIZE, GAME_LEN,
            n_layer=args.layers, n_head=args.n_head, n_embd=args.n_embd,
        )
        model = GPTChainBracketPredictor(config)
        init_path = os.path.join(args.ckpt_dir, "random_init.pt")
        torch.save(model.state_dict(), init_path)
        with open(os.path.join(args.ckpt_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Saved random init to {init_path}")
        print(f"Model: {args.layers}L / {args.n_embd}d / {args.n_head}h — {n_params:,} params")
        return

    # ---- Eval-only mode ----
    if args.eval_only:
        ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
        if not os.path.exists(ckpt_path):
            pt_files = [
                os.path.join(args.ckpt_dir, f)
                for f in os.listdir(args.ckpt_dir) if f.endswith(".pt")
            ]
            ckpt_path = max(pt_files, key=os.path.getmtime) if pt_files else None
        if ckpt_path is None:
            print(f"ERROR: No checkpoint found in {args.ckpt_dir}")
            sys.exit(1)

        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        config = GPTConfig(
            VOCAB_SIZE, GAME_LEN,
            n_layer=args.layers, n_head=args.n_head, n_embd=args.n_embd,
        )
        model = GPTChainBracketPredictor(config).to(device)
        model.load_state_dict(state_dict)

        print("Loading dataset...")
        dataset = ChainBracketStateDataset(group_table=group_table, max_files=args.max_files)
        _, test_ds = dataset.split(train_frac=args.train_frac, seed=args.seed)
        test_loader = DataLoader(
            test_ds, shuffle=False, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"\nEval-only: ce={test_loss:.4f}  acc={test_acc:.4f}")
        return

    # ---- Data ----
    print("Loading dataset...")
    t0 = time.time()
    dataset = ChainBracketStateDataset(group_table=group_table, max_files=args.max_files)
    train_ds, test_ds = dataset.split(train_frac=args.train_frac, seed=args.seed)
    print(
        f"Loaded {len(dataset)} games in {time.time() - t0:.1f}s "
        f"(train={len(train_ds)}, test={len(test_ds)})"
    )

    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, shuffle=False, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    config = GPTConfig(
        VOCAB_SIZE, GAME_LEN,
        n_layer=args.layers, n_head=args.n_head, n_embd=args.n_embd,
    )
    model = GPTChainBracketPredictor(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.layers}L / {args.n_embd}d / {args.n_head}h — {n_params:,} params")

    class OptimConfig:
        learning_rate = args.lr
        weight_decay = 0.1
        betas = (0.9, 0.95)

    optimizer = model.configure_optimizers(OptimConfig())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Save args ----
    args_dict = vars(args)
    args_dict["group_table_path"] = os.path.join(
        GTABLE_DIR, f"group_table_seed{args.group_seed}.npy"
    )
    with open(os.path.join(args.ckpt_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    # ---- Train ----
    history = []
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.2e})")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        print(f"  train ce={train_loss:.4f}  acc={train_acc:.4f}", flush=True)
        print(f"  test  ce={test_loss:.4f}  acc={test_acc:.4f}", flush=True)

        record = {
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
        }
        history.append(record)

        ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_loss": test_loss,
            "test_acc": test_acc,
        }, ckpt_path)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_path = os.path.join(args.ckpt_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  ** new best test acc: {best_test_acc:.4f} — saved to {best_path}")

    # ---- Save history ----
    hist_path = os.path.join(args.ckpt_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best test accuracy: {best_test_acc:.4f}")
    print(f"Checkpoints: {args.ckpt_dir}")


if __name__ == "__main__":
    main()
