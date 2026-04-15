"""
Fine-tune OthelloGPT on corruption-generated games and record per-batch losses.

Usage:
  python finetune_corruption.py --games-dir experiments/corruption/games/type1_alpha010 \
                                --output-dir experiments/corruption/losses \
                                --label type1_alpha010
"""

import argparse
import json
import os
import pickle
import sys
import math
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mingpt.model import GPT, GPTConfig
from mingpt.dataset import CharDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-ckpt", action="store_true")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load games
    games_path = os.path.join(args.games_dir, "games.pickle")
    print(f"Loading games from {games_path}...")
    with open(games_path, 'rb') as f:
        games = pickle.load(f)
    print(f"Loaded {len(games)} games")

    # Filter out very short games
    games = [g for g in games if len(g) >= 5]
    print(f"After filtering: {len(games)} games with >= 5 moves")

    # Create dataset
    train_dataset = CharDataset(games)
    print(f"Vocab size: {train_dataset.vocab_size}, Block size: {train_dataset.block_size}")

    # Build model
    mconf = GPTConfig(
        train_dataset.vocab_size,
        train_dataset.block_size,
        n_layer=8, n_head=8, n_embd=512
    )
    model = GPT(mconf)

    # Load pre-trained weights
    print(f"Loading checkpoint: {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Optimizer (same as original training)
    optimizer = model.configure_optimizers(
        argparse.Namespace(
            learning_rate=args.lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
    )

    # DataLoader
    loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Training loop — record per-batch losses
    all_losses = []
    batch_count = 0
    t0 = time.time()
    loss_path = os.path.join(args.output_dir, f"{args.label}.json")

    def save_logs():
        with open(loss_path, 'w') as f:
            json.dump({
                'label': args.label,
                'losses': all_losses,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'num_games': len(games),
                'total_batches': batch_count,
                'elapsed_seconds': time.time() - t0,
            }, f)

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, y)
            loss = loss.mean()

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            all_losses.append(loss_val)
            batch_count += 1

            recent_avg = np.mean(all_losses[-50:])
            pbar.set_postfix(loss=f"{loss_val:.4f}", avg50=f"{recent_avg:.4f}")

            if batch_count % 100 == 0:
                save_logs()

        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}: mean_loss={mean_loss:.4f} ({len(epoch_losses)} batches)")
        save_logs()

    elapsed = time.time() - t0
    print(f"\nDone: {batch_count} total batches in {elapsed:.0f}s")
    print(f"Final loss: {np.mean(all_losses[-100:]):.4f} (last 100 batches)")

    save_logs()
    print(f"Saved losses to {loss_path}")

    # Optionally save checkpoint
    if args.save_ckpt:
        ckpt_dir = os.path.join(args.output_dir, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{args.label}.ckpt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == '__main__':
    main()
