"""Stage 2: Per-cell adversarial beam search.

For each target cell, finds games where the model assigns high probability
to that cell when it's illegal. Uses beam search with batched inference.

Usage:
    python behavioral_adversarial.py --cell 0 --output-dir behavioral_data
    python behavioral_adversarial.py --cell 0 --num-starts 5 --beam-width 5  # test
"""

import argparse
import os
import sys
import time
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F

from behavioral_utils import (
    build_120d_features, load_model, build_vocab_to_pos_map,
    N_MOVES, VALID_MOVES, MOVE_TO_IDX, IDX_TO_MOVE, POS_START
)
from data.othello import OthelloBoardState


def fast_copy_board(board):
    """Fast copy of OthelloBoardState without deepcopy overhead."""
    new = OthelloBoardState.__new__(OthelloBoardState)
    new.board_size = board.board_size
    new.state = board.state.copy()
    new.next_hand_color = board.next_hand_color
    new.history = list(board.history)
    new.age = board.age.copy()
    return new


def per_cell_beam_search(model, dataset, device, target_cell,
                         num_starts=500, beam_width=50, prefix_len=5,
                         max_depth=59):
    """Beam search maximizing probability on target cell when it's illegal.

    Args:
        target_cell: cell index 0-59 (index into VALID_MOVES)
        num_starts: number of diverse random openings
        beam_width: beams to keep at each depth
        prefix_len: length of random opening prefix
        max_depth: maximum game length

    Returns:
        list of (game_sequence, cumulative_score) tuples, sorted by score
    """
    target_board_pos = IDX_TO_MOVE[target_cell]
    vocab_to_pos, pos_to_vocab = build_vocab_to_pos_map(dataset)
    target_token_idx = pos_to_vocab[target_board_pos]

    # Build vectorized stoi lookup for fast tokenization
    stoi_arr = np.zeros(64, dtype=np.int64)
    for pos in VALID_MOVES:
        stoi_arr[pos] = dataset.stoi[pos]

    all_results = []

    for start_idx in range(num_starts):
        if start_idx % 50 == 0:
            print(f"  Start {start_idx}/{num_starts}", flush=True)

        # Generate random opening prefix
        prefix = []
        board = OthelloBoardState()
        for _ in range(prefix_len):
            valid = board.get_valid_moves()
            if not valid:
                break
            move = random.choice(valid)
            prefix.append(move)
            board.umpire(move)

        if not board.get_valid_moves():
            continue

        # Beam search from prefix
        # Each beam: (game_seq, board_state, cumulative_score)
        beams = [(list(prefix), fast_copy_board(board), 0.0)]

        for depth in range(prefix_len, max_depth):
            candidates = []
            for game_seq, brd, cum_score in beams:
                valid_moves = brd.get_valid_moves()
                if not valid_moves:
                    candidates.append((game_seq, brd, cum_score, True))
                    continue

                for move in valid_moves:
                    new_seq = game_seq + [move]
                    new_board = fast_copy_board(brd)
                    new_board.umpire(move)
                    candidates.append((new_seq, new_board, cum_score, False))

            # Filter out finished games
            active = [(s, b, c) for s, b, c, done in candidates if not done]
            finished = [(s, b, c) for s, b, c, done in candidates if done]

            if not active:
                all_results.extend([(s, c) for s, b, c in finished])
                break

            # Batch inference on all active candidates (vectorized tokenization)
            n_cands = len(active)
            seq_len = min(len(active[0][0]), dataset.block_size)
            seqs_arr = np.zeros((n_cands, seq_len), dtype=np.int64)
            for i, (seq, _, _) in enumerate(active):
                t_len = min(len(seq), seq_len)
                for s in range(t_len):
                    seqs_arr[i, s] = seq[s]
            tokens = torch.tensor(stoi_arr[seqs_arr], dtype=torch.long,
                                  device=device)

            with torch.no_grad():
                # Process in sub-batches if too many candidates
                batch_size = 256
                all_scores = []
                for bs in range(0, n_cands, batch_size):
                    be = min(bs + batch_size, n_cands)
                    logits, _ = model(tokens[bs:be])
                    probs = F.softmax(logits[:, -1, :], dim=-1)
                    target_probs = probs[:, target_token_idx].cpu().numpy()
                    all_scores.append(target_probs)
                target_scores = np.concatenate(all_scores)

            # Score: add target cell prob only if target cell is illegal
            scored = []
            for i, (seq, brd, cum) in enumerate(active):
                valid_moves = brd.get_valid_moves()
                if target_board_pos not in valid_moves:
                    # Target cell is illegal — count this probability
                    new_score = cum + target_scores[i]
                else:
                    new_score = cum
                scored.append((seq, brd, new_score))

            # Keep top beam_width
            scored.sort(key=lambda x: x[2], reverse=True)
            beams = scored[:beam_width]

        # Keep the best game from this start
        if beams:
            best_seq, _, best_score = beams[0]
            all_results.append((best_seq, best_score))

    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results


def extract_adversarial_positions(games, model, dataset, device, target_cell,
                                  prob_threshold=0.005):
    """Extract positions where model promotes target cell while it's illegal.

    Args:
        games: list of game sequences
        target_cell: cell index 0-59
        prob_threshold: minimum probability to include (default 0.5%)

    Returns:
        features: (n_pos, 120) float32
        target_probs: (n_pos,) float32
        all_probs: (n_pos, 60) float32
        legal_masks: (n_pos, 60) uint8
    """
    target_board_pos = IDX_TO_MOVE[target_cell]
    vocab_to_pos, pos_to_vocab = build_vocab_to_pos_map(dataset)
    target_token_idx = pos_to_vocab[target_board_pos]

    feat_list = []
    tprob_list = []
    prob_list = []
    legal_list = []

    for game in games:
        board = OthelloBoardState()
        for t in range(len(game)):
            if t >= POS_START:
                valid_moves = board.get_valid_moves()

                if target_board_pos not in valid_moves:
                    # Target is illegal — check model's probability
                    context = game[:t]
                    x = torch.tensor([dataset.stoi[s] for s in context],
                                     dtype=torch.long)[None, ...].to(device)
                    with torch.no_grad():
                        logits, _ = model(x)
                    probs = F.softmax(logits[0, -1, :], dim=-1)
                    target_prob = probs[target_token_idx].item()

                    if target_prob >= prob_threshold:
                        # Compute features for this position
                        f, _ = build_120d_features([game], t, t + 1)

                        # Extract 60-d probs
                        p60 = np.zeros(N_MOVES, dtype=np.float32)
                        for tok_idx in range(len(probs)):
                            bp = vocab_to_pos[tok_idx]
                            if bp >= 0:
                                p60[MOVE_TO_IDX[bp]] = probs[tok_idx].item()

                        # Legal mask
                        lm = np.zeros(N_MOVES, dtype=np.uint8)
                        for m in valid_moves:
                            if m in MOVE_TO_IDX:
                                lm[MOVE_TO_IDX[m]] = 1

                        feat_list.append(f[0])
                        tprob_list.append(target_prob)
                        prob_list.append(p60)
                        legal_list.append(lm)

            if t < len(game):
                board.umpire(game[t])

    if not feat_list:
        return (np.zeros((0, 120), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros((0, N_MOVES), dtype=np.float32),
                np.zeros((0, N_MOVES), dtype=np.uint8))

    return (np.stack(feat_list),
            np.array(tprob_list, dtype=np.float32),
            np.stack(prob_list),
            np.stack(legal_list))


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Per-cell adversarial beam search")
    parser.add_argument("--cell", type=int, required=True, help="Target cell index (0-59)")
    parser.add_argument("--ckpt", type=str, default="./ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--output-dir", type=str, default="behavioral_data")
    parser.add_argument("--num-starts", type=int, default=200)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--prefix-len", type=int, default=5)
    parser.add_argument("--num-save", type=int, default=500,
                        help="Number of top games to extract positions from")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed + args.cell)
    np.random.seed(args.seed + args.cell)
    t0 = time.time()

    cell_name = f"{VALID_MOVES[args.cell]}"
    print(f"Cell {args.cell} (board pos {cell_name})", flush=True)

    # Load model
    print("Loading model...", flush=True)
    model, dataset, device = load_model(args.ckpt)
    print(f"  Device: {device}", flush=True)

    # Run beam search
    print(f"Beam search: {args.num_starts} starts, width {args.beam_width}...", flush=True)
    results = per_cell_beam_search(
        model, dataset, device, args.cell,
        num_starts=args.num_starts,
        beam_width=args.beam_width,
        prefix_len=args.prefix_len,
    )
    print(f"  Found {len(results)} games", flush=True)
    if results:
        print(f"  Top scores: {[f'{s:.4f}' for _, s in results[:5]]}", flush=True)

    # Extract adversarial positions from top games
    top_games = [seq for seq, _ in results[:args.num_save]]
    print(f"Extracting adversarial positions from {len(top_games)} games...", flush=True)
    features, target_probs, probs, legal = extract_adversarial_positions(
        top_games, model, dataset, device, args.cell
    )
    print(f"  Found {len(features)} adversarial positions", flush=True)

    # Save
    adv_dir = os.path.join(args.output_dir, "adversarial")
    os.makedirs(adv_dir, exist_ok=True)
    out_path = os.path.join(adv_dir, f"cell_{args.cell:02d}.npz")
    np.savez_compressed(
        out_path,
        features=features.astype(np.float16),
        target_prob=target_probs,
        probs=probs.astype(np.float16),
        legal=legal,
    )

    elapsed = time.time() - t0
    print(f"Saved {out_path} ({len(features)} positions, {elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
