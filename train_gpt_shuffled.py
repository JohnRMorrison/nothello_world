"""Train Othello-GPT on RANDOMIZED-ORDER move sequences.

Each game's moves are reshuffled fresh every epoch (handled inside the
CharDataset via shuffle_moves=True), forcing the network to learn
heuristics that are invariant to move order.  Compare the final loss
against the trivial baseline of `log(60 - k)` averaged over positions
to gauge how much order-free structure the network can recover.

Saves:
  ckpts/gpt_shuffled_<timestamp>.ckpt        — final model weights
  logs/shuffled_<timestamp>_final_loss.txt   — final train + val loss

Usage:
  python train_gpt_shuffled.py
Optional env vars (with defaults):
  EPOCHS=250  BATCH_SIZE=512  NUM_WORKERS=8  CKPT_TAG=  (suffix)
"""
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import get_othello
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import set_seed


def main():
    set_seed(44)

    epochs = int(os.environ.get('EPOCHS', '250'))
    batch_size = int(os.environ.get('BATCH_SIZE', '512'))
    num_workers = int(os.environ.get('NUM_WORKERS', '8'))
    ckpt_tag = os.environ.get('CKPT_TAG', '')

    print(f"epochs={epochs}  batch_size={batch_size}  num_workers={num_workers}")

    print("Loading Othello dataset...")
    othello = get_othello(ood_num=-1, data_root=None, wthor=True)
    print(f"  {len(othello.sequences)} train games, {len(othello.val)} val games")

    print("Building train CharDataset (shuffle_moves=True)...")
    train_dataset = CharDataset(othello, shuffle_moves=True)

    # Build a val dataset that reuses the train vocab/block_size; we still
    # shuffle val sequences so we evaluate the same regime we trained on.
    class _ValGames:
        def __init__(self, games):
            self.games = games
        def __len__(self): return len(self.games)
        def __getitem__(self, i): return self.games[i]
    print("Building val CharDataset (shuffle_moves=True)...")
    val_games = _ValGames(othello.val)
    val_dataset = CharDataset(val_games, shuffle_moves=True)
    # Sanity-check vocab match
    if val_dataset.vocab_size != train_dataset.vocab_size:
        print(f"WARN: val vocab {val_dataset.vocab_size} != train vocab {train_dataset.vocab_size}")

    print("Building model (8L, 8H, 512d)...")
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                      n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)

    t_start = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{ckpt_tag}" if ckpt_tag else ""
    ckpt_path = f"./ckpts/gpt_shuffled{tag}_{t_start}.ckpt"
    os.makedirs('./ckpts', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    tconf = TrainerConfig(
        max_epochs=epochs,
        batch_size=batch_size,
        learning_rate=5e-4,
        lr_decay=True,
        warmup_tokens=len(train_dataset) * train_dataset.block_size * 5,
        final_tokens=len(train_dataset) * train_dataset.block_size * epochs,
        num_workers=num_workers,
        ckpt_path=ckpt_path,
    )

    # Train (no per-epoch val to save time; we evaluate val once after training)
    trainer = Trainer(model, train_dataset, None, tconf)
    print(f"Training on device {trainer.device} for {epochs} epochs...")
    trainer.train()

    # Final eval: shuffled-order val loss
    print("Computing final validation loss...")
    raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    raw_model.eval()
    loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True)
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            _, loss = raw_model(x, y)
            total_loss += loss.mean().item()
            total_batches += 1
    val_loss = total_loss / max(1, total_batches)
    print(f"FINAL VAL LOSS (shuffled): {val_loss:.6f}")

    out_path = f"./logs/shuffled{tag}_{t_start}_final_loss.txt"
    with open(out_path, 'w') as f:
        f.write(f"epochs {epochs}\nbatch_size {batch_size}\n")
        f.write(f"ckpt {ckpt_path}\n")
        f.write(f"final_val_loss_shuffled {val_loss:.6f}\n")
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
