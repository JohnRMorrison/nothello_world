"""Stage 1: Collect model probabilities on games.

For each shard of 100K games, runs batched inference to get model probabilities
at every position, computes 120-d move-history features, and computes legal
move masks. Saves compressed .npz files.

Usage:
    python behavioral_collect.py --shard 0 --output-dir behavioral_data
    python behavioral_collect.py --shard 0 --num-games 100  # small test
"""

import argparse
import os
import sys
import time
import numpy as np

import torch
import torch.nn.functional as F

from behavioral_utils import (
    build_120d_features, load_model, build_vocab_to_pos_map,
    extract_probs_60d, compute_legal_masks_batch, load_shard_games,
    POS_START, POS_END, N_MOVES, MOVE_TO_IDX
)


def collect_probabilities(games, model, dataset, device, pos_start, pos_end,
                          batch_size=64):
    """Run batched inference on games, return probabilities at each position.

    One forward pass per batch gives logits at all 59 positions simultaneously.

    Args:
        games: list of game sequences (board positions 0-63)
        model: GPT model in eval mode
        dataset: CharDataset with stoi/itos
        device: torch device
        pos_start: first position to extract (inclusive)
        pos_end: last position to extract (exclusive)
        batch_size: number of games per forward pass

    Returns:
        all_probs: (n_games, length, 60) float32 numpy array
    """
    vocab_to_pos, _ = build_vocab_to_pos_map(dataset)
    n_games = len(games)
    length = pos_end - pos_start

    # Pre-tokenize all games
    block_size = dataset.block_size  # 59
    all_tokens = np.zeros((n_games, block_size), dtype=np.int64)
    for i, game in enumerate(games):
        seq_len = min(len(game), block_size)
        for s in range(seq_len):
            all_tokens[i, s] = dataset.stoi[game[s]]

    all_probs = np.zeros((n_games, length, N_MOVES), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_games, batch_size):
            end = min(start + batch_size, n_games)
            tokens_batch = torch.tensor(all_tokens[start:end],
                                        dtype=torch.long).to(device)

            # Forward pass: (B, 59, 61) logits
            logits, _ = model(tokens_batch)
            probs = F.softmax(logits, dim=-1)  # (B, 59, 61)

            # Extract 60-d probability vectors
            probs_60d = extract_probs_60d(probs, vocab_to_pos)  # (B, 59, 60)

            # Position t in the game corresponds to logits index t-1
            # (model at index t-1 predicts the move at position t)
            # For pos_start=4: we want logits[:, 3, :] through logits[:, pos_end-2, :]
            for ti, t in enumerate(range(pos_start, pos_end)):
                logit_idx = t - 1
                if logit_idx < probs_60d.shape[1]:
                    all_probs[start:end, ti, :] = probs_60d[:, logit_idx, :]

            if (start // batch_size) % 50 == 0:
                print(f"  Inference: {end}/{n_games} games", flush=True)

    return all_probs


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Collect model probabilities")
    parser.add_argument("--shard", type=int, required=True, help="Shard index (0-19)")
    parser.add_argument("--ckpt", type=str, default="./ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--output-dir", type=str, default="behavioral_data")
    parser.add_argument("--games-dir", type=str, default="data/othello_synthetic")
    parser.add_argument("--num-games", type=int, default=100000,
                        help="Games per shard (default 100K)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pos-start", type=int, default=POS_START)
    parser.add_argument("--pos-end", type=int, default=POS_END)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # Load games
    print(f"Loading {args.num_games} games for shard {args.shard}...", flush=True)
    games = load_shard_games(args.shard, args.num_games, args.games_dir)
    print(f"  Loaded {len(games)} games", flush=True)

    # Load model
    print("Loading model...", flush=True)
    model, dataset, device = load_model(args.ckpt)
    print(f"  Device: {device}", flush=True)

    # Compute features
    print("Computing 120-d features...", flush=True)
    features, positions = build_120d_features(games, args.pos_start, args.pos_end)
    print(f"  Features shape: {features.shape}", flush=True)

    # Run inference
    print("Running batched inference...", flush=True)
    probs_3d = collect_probabilities(
        games, model, dataset, device,
        args.pos_start, args.pos_end, args.batch_size
    )
    # Reshape to (n_samples, 60)
    n_games = len(games)
    length = args.pos_end - args.pos_start
    probs = probs_3d.reshape(n_games * length, N_MOVES)
    print(f"  Probs shape: {probs.shape}", flush=True)

    # Compute legal masks
    print("Computing legal masks...", flush=True)
    legal = compute_legal_masks_batch(games, args.pos_start, args.pos_end)
    print(f"  Legal shape: {legal.shape}", flush=True)

    # Save
    out_path = os.path.join(args.output_dir, f"shard_{args.shard:02d}.npz")
    print(f"Saving to {out_path}...", flush=True)
    np.savez_compressed(
        out_path,
        features=features.astype(np.float16),
        probs=probs.astype(np.float16),
        legal=legal.astype(np.uint8),
        positions=positions,
    )

    elapsed = time.time() - t0
    file_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done. {features.shape[0]} rows, {file_size:.0f} MB, {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
