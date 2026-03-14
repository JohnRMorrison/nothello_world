"""
Fine-tune OthelloGPT on corruption-generated games and record metrics.

Records per-eval-step: loss, legal-move accuracy (is top-1 prediction a move
the corrupted rules consider legal?), and mean rank of the played move.
Eval is on a held-out 10% test split, run every --eval-every batches.

Usage:
  python finetune_corruption.py --games-dir experiments/corruption_v2/games_100k/alpha010 \
                                --output-dir experiments/corruption_v2/losses_100k \
                                --label alpha010
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from mingpt.model import GPT, GPTConfig
from mingpt.dataset import CharDataset


def build_legal_mask(games, legal_moves, stoi, block_size, vocab_size):
    """Build a boolean mask (N_tokens, vocab_size) of legal moves.

    The dataset produces x = tokens[:-1], y = tokens[1:].  Position t in y
    is the prediction target for move (t+1) in the game.  The legal moves at
    that point are legal_moves[game_idx][t+1].

    Returns np.ndarray of shape (total_non_padding_positions, vocab_size).
    """
    # Count total positions first
    total = sum(min(len(g) - 1, block_size) for g in games)
    mask = np.zeros((total, vocab_size), dtype=np.bool_)

    pos = 0
    for gi, game in enumerate(games):
        T = min(len(game) - 1, block_size)
        for t in range(T):
            move_idx = t + 1
            if move_idx < len(legal_moves[gi]):
                for p in legal_moves[gi][move_idx]:
                    if p in stoi:
                        mask[pos, stoi[p]] = True
            pos += 1
    return mask


@torch.no_grad()
def evaluate(model, loader, device, legal_mask=None):
    """Evaluate on test set.

    Returns (loss, legal_acc, mean_rank).
    legal_mask: np.ndarray (N, V) boolean mask from build_legal_mask().
    If None, legal_acc falls back to exact-match accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_legal = 0
    total_rank = 0.0
    n_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        B, T, V = logits.shape

        # Loss
        n_valid = (y != -100).sum().item()
        total_loss += loss.mean().item() * n_valid

        # Flatten non-padding positions
        mask = (y != -100)  # (B, T)
        preds_flat = logits.argmax(dim=-1)[mask].cpu().numpy()  # (n_valid,)
        y_flat = y[mask].cpu().numpy()

        # Legal-move accuracy
        if legal_mask is not None:
            end = n_tokens + len(preds_flat)
            if end <= len(legal_mask):
                legal_slice = legal_mask[n_tokens:end]
                total_legal += legal_slice[np.arange(len(preds_flat)), preds_flat].sum()
        else:
            total_legal += (preds_flat == y_flat).sum()

        # Mean rank of the played move (vectorized)
        logits_flat = logits[mask]  # (n_valid, V)
        y_tok = y[mask]  # (n_valid,)
        correct_logit = logits_flat[torch.arange(len(y_tok)), y_tok].unsqueeze(-1)
        ranks = (logits_flat > correct_logit).sum(dim=-1) + 1  # 1-indexed
        total_rank += ranks.float().sum().item()

        n_tokens += len(preds_flat)

    model.train()
    return (total_loss / n_tokens,
            total_legal / n_tokens,
            total_rank / n_tokens)


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
    parser.add_argument("--random-init", action="store_true",
                        help="Skip loading checkpoint; train from random init")
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

    # Load legal moves (saved by generate_rule_games.py)
    legal_path = os.path.join(args.games_dir, "legal_moves.pickle")
    has_legal = os.path.exists(legal_path)
    if has_legal:
        print(f"Loading legal moves from {legal_path}...")
        with open(legal_path, 'rb') as f:
            legal_moves = pickle.load(f)
        print(f"Loaded legal moves for {len(legal_moves)} games")
    else:
        print("No legal_moves.pickle found — will use exact-match accuracy")
        legal_moves = None

    # Filter out very short games
    if has_legal:
        keep = [(g, l) for g, l in zip(games, legal_moves) if len(g) >= 5]
        games = [g for g, _ in keep]
        legal_moves = [l for _, l in keep]
    else:
        games = [g for g in games if len(g) >= 5]
    print(f"After filtering: {len(games)} games with >= 5 moves")

    # 90/10 train/test split
    n_test = len(games) // 10
    n_train = len(games) - n_test
    train_games = games[:n_train]
    test_games = games[n_train:]
    test_legal = legal_moves[n_train:] if has_legal else None
    print(f"Train: {len(train_games)}, Test: {len(test_games)}")

    # Create datasets
    train_dataset = CharDataset(train_games)
    test_dataset = CharDataset(test_games)
    print(f"Vocab size: {train_dataset.vocab_size}, Block size: {train_dataset.block_size}")

    # Build legal-move mask for test data
    if has_legal:
        print("Building legal move mask for test set...", flush=True)
        test_legal_mask = build_legal_mask(
            test_games, test_legal, train_dataset.stoi,
            train_dataset.block_size, train_dataset.vocab_size)
        print(f"  Mask shape: {test_legal_mask.shape}")
    else:
        test_legal_mask = None

    # Build model
    mconf = GPTConfig(
        train_dataset.vocab_size,
        train_dataset.block_size,
        n_layer=8, n_head=8, n_embd=512
    )
    model = GPT(mconf)

    # Load pre-trained weights (or skip for random init)
    if args.random_init:
        print("Using random initialization (no checkpoint)")
    else:
        print(f"Loading checkpoint: {args.ckpt}")
        state_dict = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state_dict)
    model = model.to(device)

    # Optimizer
    optimizer = model.configure_optimizers(
        argparse.Namespace(
            learning_rate=args.lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, shuffle=True, pin_memory=True,
        batch_size=args.batch_size, num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, pin_memory=True,
        batch_size=args.batch_size, num_workers=4,
    )

    # Eval before any training
    print("Evaluating before training...", flush=True)
    eval_loss, eval_acc, eval_rank = evaluate(
        model, test_loader, device, test_legal_mask)
    print(f"  Initial: loss={eval_loss:.4f}, legal_acc={eval_acc:.4f}, mean_rank={eval_rank:.2f}")

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
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss.mean()

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_count += 1

            if batch_count % args.eval_every == 0:
                eval_loss, eval_acc, eval_rank = evaluate(
                    model, test_loader, device, test_legal_mask)
                eval_steps.append(batch_count)
                eval_losses.append(eval_loss)
                eval_accs.append(eval_acc)
                eval_ranks.append(eval_rank)

                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1}, batch {it+1}/{len(train_loader)}, "
                      f"step={batch_count}: loss={eval_loss:.4f}, legal_acc={eval_acc:.4f}, "
                      f"rank={eval_rank:.2f}, elapsed={elapsed:.0f}s", flush=True)

        # End-of-epoch eval
        eval_loss, eval_acc, eval_rank = evaluate(
            model, test_loader, device, test_legal_mask)
        eval_steps.append(batch_count)
        eval_losses.append(eval_loss)
        eval_accs.append(eval_acc)
        eval_ranks.append(eval_rank)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}: loss={eval_loss:.4f}, legal_acc={eval_acc:.4f}, "
              f"rank={eval_rank:.2f} ({batch_count} batches, {elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {batch_count} total batches in {elapsed:.0f}s")
    print(f"Final: loss={eval_losses[-1]:.4f}, legal_acc={eval_accs[-1]:.4f}, "
          f"rank={eval_ranks[-1]:.2f}")

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

    if args.save_ckpt:
        ckpt_dir = os.path.join(args.output_dir, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{args.label}.ckpt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == '__main__':
    main()
