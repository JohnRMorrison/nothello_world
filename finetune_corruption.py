"""
Fine-tune OthelloGPT on corruption-generated games and record metrics.

Records per-eval-step: loss, top-1 accuracy, mean rank of correct move.
Eval is on a held-out 10% test split, run every --eval-every batches.

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

sys.path.insert(0, os.path.dirname(__file__))
from mingpt.model import GPT, GPTConfig
from mingpt.dataset import CharDataset


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on test set. Returns loss, top-1 accuracy, mean rank."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_rank_sum = 0.0
    total_tokens = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, loss = model(x, y)
        loss = loss.mean()

        # logits: (B, T, vocab), y: (B, T)
        B, T, V = logits.shape

        # Flatten for metrics
        logits_flat = logits.reshape(-1, V)  # (B*T, V)
        y_flat = y.reshape(-1)              # (B*T,)

        # Top-1 accuracy
        preds = logits_flat.argmax(dim=-1)
        total_correct += (preds == y_flat).sum().item()

        # Mean rank: rank of correct token (1-indexed)
        # Sort descending, find position of correct answer
        sorted_indices = logits_flat.argsort(dim=-1, descending=True)
        ranks = (sorted_indices == y_flat.unsqueeze(-1)).nonzero(as_tuple=True)[1] + 1
        total_rank_sum += ranks.float().sum().item()

        total_loss += loss.item() * (B * T)
        total_tokens += B * T

    model.train()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    mean_rank = total_rank_sum / total_tokens
    return avg_loss, accuracy, mean_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=200)
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

    # 90/10 train/test split
    n_test = len(games) // 10
    n_train = len(games) - n_test
    train_games = games[:n_train]
    test_games = games[n_train:]
    print(f"Train: {len(train_games)}, Test: {len(test_games)}")

    # Create datasets
    train_dataset = CharDataset(train_games)
    test_dataset = CharDataset(test_games)
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

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=args.batch_size,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        pin_memory=True,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Eval before any training
    print("Evaluating before training...", flush=True)
    eval_loss, eval_acc, eval_rank = evaluate(model, test_loader, device)
    print(f"  Initial: loss={eval_loss:.4f}, acc={eval_acc:.4f}, mean_rank={eval_rank:.2f}")

    eval_steps = [0]
    eval_losses = [eval_loss]
    eval_accs = [eval_acc]
    eval_ranks = [eval_rank]

    # Training loop
    batch_count = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        for it, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, y)
            loss = loss.mean()

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_count += 1

            if batch_count % args.eval_every == 0:
                eval_loss, eval_acc, eval_rank = evaluate(model, test_loader, device)
                eval_steps.append(batch_count)
                eval_losses.append(eval_loss)
                eval_accs.append(eval_acc)
                eval_ranks.append(eval_rank)

                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1}, batch {it+1}/{len(train_loader)}, "
                      f"step={batch_count}: loss={eval_loss:.4f}, acc={eval_acc:.4f}, "
                      f"rank={eval_rank:.2f}, elapsed={elapsed:.0f}s", flush=True)

        # End-of-epoch eval
        eval_loss, eval_acc, eval_rank = evaluate(model, test_loader, device)
        eval_steps.append(batch_count)
        eval_losses.append(eval_loss)
        eval_accs.append(eval_acc)
        eval_ranks.append(eval_rank)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}: loss={eval_loss:.4f}, acc={eval_acc:.4f}, "
              f"rank={eval_rank:.2f} ({batch_count} batches, {elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {batch_count} total batches in {elapsed:.0f}s")
    print(f"Final: loss={eval_losses[-1]:.4f}, acc={eval_accs[-1]:.4f}, rank={eval_ranks[-1]:.2f}")

    # Save results
    out_path = os.path.join(args.output_dir, f"{args.label}.json")
    with open(out_path, 'w') as f:
        json.dump({
            'label': args.label,
            'eval_steps': eval_steps,
            'eval_losses': eval_losses,
            'eval_accs': eval_accs,
            'eval_ranks': eval_ranks,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'eval_every': args.eval_every,
            'num_train': len(train_games),
            'num_test': len(test_games),
            'total_batches': batch_count,
            'elapsed_seconds': elapsed,
        }, f)
    print(f"Saved results to {out_path}")

    # Optionally save checkpoint
    if args.save_ckpt:
        ckpt_dir = os.path.join(args.output_dir, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{args.label}.ckpt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == '__main__':
    main()
