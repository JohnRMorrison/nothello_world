#!/usr/bin/env python
"""
Train a transformer to predict boolean labels from Othello game sequences.

Supports CUDA, MPS (Apple Silicon), and CPU. Run from the project root:

    python experiments/mathematical_transformation_experiments/train_boolean_classifier.py [OPTIONS]

Options:
    --layers 4              Number of transformer layers (default: 4)
    --n-embd 256            Embedding dimension (default: 256)
    --n-head 8              Number of attention heads (default: 8)
    --epochs 20             Training epochs (default: 20)
    --batch-size 256        Batch size (default: 256)
    --lr 3e-4               Learning rate (default: 3e-4)
    --max-files 5           Number of data pickle files to load (default: 5, None=all)
    --transform dot_product Transform name (default: dot_product)
    --seed 42               Random seed (default: 42)
    --ckpt-dir ...          Checkpoint directory (default: auto)
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.mathematical_transformation_experiments.dataset import (
    BooleanLabelTokenDataset,
    GAME_LEN,
    PAD_IDX,
    VOCAB_SIZE,
)
from mingpt.model import GPT, GPTConfig  # noqa: F401 — GPTConfig used in main()


# ============================= Model ==========================================

class GPTClassifier(GPT):
    """Same architecture as OthelloGPT (causal transformer), with a binary
    classification head instead of next-token prediction.

    Uses the last non-padding position's hidden state for classification
    (in causal attention, this position has attended to all prior real tokens).
    """

    def __init__(self, config):
        super().__init__(config)
        # Replace the next-token prediction head with binary classification
        self.head = nn.Linear(config.n_embd, 2, bias=False)
        self.head.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)  # (B, T, n_embd)

        # Use last non-padding position per sample
        non_pad = (idx != PAD_IDX)  # (B, T)
        lengths = non_pad.sum(dim=1)  # (B,)
        last_idx = (lengths - 1).clamp(min=0)  # (B,)
        last_hidden = x[torch.arange(b, device=x.device), last_idx]  # (B, n_embd)

        logits = self.head(last_hidden)  # (B, 2)
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
    total_samples = 0

    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in tqdm(loader, desc="  eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ============================= Main ===========================================

def parse_args():
    p = argparse.ArgumentParser(description="Train transformer boolean classifier on Othello games")
    p.add_argument("--layers", type=int, default=8, help="Number of transformer layers")
    p.add_argument("--n-embd", type=int, default=512, help="Embedding dimension")
    p.add_argument("--n-head", type=int, default=8, help="Number of attention heads")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-files", type=int, default=5, help="Data files to load (None=all)")
    p.add_argument("--transform", type=str, default="dot_product")
    p.add_argument("--label-seed", type=int, default=42, help="Seed used for label generation")
    p.add_argument("--seed", type=int, default=42, help="Training random seed")
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--save-random-init", action="store_true",
                   help="Save a randomly initialized (untrained) model checkpoint and exit")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # ---- Checkpoint dir ----
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(
            SCRIPT_DIR, "ckpts",
            f"{args.transform}_seed{args.label_seed}_{args.layers}L_{args.n_embd}d"
        )
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- Save random init and exit if requested ----
    if args.save_random_init:
        config = GPTConfig(
            VOCAB_SIZE, GAME_LEN,
            n_layer=args.layers, n_head=args.n_head, n_embd=args.n_embd,
        )
        model = GPTClassifier(config)
        init_path = os.path.join(args.ckpt_dir, "random_init.pt")
        torch.save(model.state_dict(), init_path)
        args_path = os.path.join(args.ckpt_dir, "args.json")
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Saved randomly initialized model to {init_path}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {args.layers}L / {args.n_embd}d / {args.n_head}h — {n_params:,} params")
        return

    # ---- Data ----
    print("Loading dataset...")
    t0 = time.time()
    dataset = BooleanLabelTokenDataset(
        transform_name=args.transform, seed=args.label_seed, max_files=args.max_files
    )
    train_ds, test_ds = dataset.split(train_frac=args.train_frac, seed=args.seed)
    print(f"Loaded {len(dataset)} games in {time.time() - t0:.1f}s "
          f"(train={len(train_ds)}, test={len(test_ds)})")

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
    model = GPTClassifier(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.layers}L / {args.n_embd}d / {args.n_head}h — {n_params:,} params")

    class OptimConfig:
        learning_rate = args.lr
        weight_decay = 0.1
        betas = (0.9, 0.95)

    optimizer = model.configure_optimizers(OptimConfig())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Save args ----
    args_path = os.path.join(args.ckpt_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Train ----
    history = []
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.2e})")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        print(f"  train loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  test  loss={test_loss:.4f}  acc={test_acc:.4f}")

        record = {
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
        }
        history.append(record)

        # Save checkpoint every epoch
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
