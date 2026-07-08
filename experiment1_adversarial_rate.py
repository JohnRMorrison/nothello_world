"""Experiment 1: adversarial success rate per beam start (exact, enumerated).

Enumerates ALL 1,396 unique 5-move Othello openings.  For each opening,
runs a beam-width-10 search that maximizes model probability on a target
cell when that cell is illegal.  At every beam step, checks the current
board state: if the target cell is illegal AND OGPT's top-1 argmax IS the
target cell, this start counts as an "adversarial success."

Reports:
  - Fraction of 1,396 starts that yield an adversarial success (per cell)
  - Sample adversarial games (top-5 by target-cell probability)

Usage:
    python experiment1_adversarial_rate.py --cell 0 \\
        --ckpt ckpts/gpt_nanda_synthetic.ckpt \\
        --beam-width 10 --max-depth 40 --prefix-len 5
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from behavioral_utils import (
    load_model, build_vocab_to_pos_map,
    N_MOVES, VALID_MOVES, MOVE_TO_IDX, IDX_TO_MOVE,
)
from data.othello import OthelloBoardState


def fast_copy_board(board):
    new = OthelloBoardState.__new__(OthelloBoardState)
    new.board_size = board.board_size
    new.state = board.state.copy()
    new.next_hand_color = board.next_hand_color
    new.history = list(board.history)
    new.age = board.age.copy()
    return new


def enumerate_prefixes(prefix_len):
    """Return all game-legal move sequences of the given length."""
    prefixes = []
    def rec(board, depth, seq):
        if depth == 0:
            prefixes.append(list(seq))
            return
        for m in board.get_valid_moves():
            nb = fast_copy_board(board)
            nb.umpire(m)
            seq.append(m)
            rec(nb, depth - 1, seq)
            seq.pop()
    rec(OthelloBoardState(), prefix_len, [])
    return prefixes


def batch_predict(model, dataset, device, seqs, stoi_arr):
    """Feed each sequence to the model, return (top1_token_idx, all_probs) arrays.

    seqs: list of game sequences (variable length, up to block_size).
    Returns:
        top1: (N,) int64 token idx of argmax (from the LAST-position logits)
        probs: (N, vocab_size) float32 probabilities
    """
    n = len(seqs)
    seq_len = max(len(s) for s in seqs)
    seq_len = min(seq_len, dataset.block_size)
    arr = np.zeros((n, seq_len), dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(seqs):
        L = min(len(s), seq_len)
        lengths[i] = L
        for j in range(L):
            arr[i, j] = s[j]
    tokens = torch.tensor(stoi_arr[arr], dtype=torch.long, device=device)

    batch_size = 256
    all_top1, all_probs = [], []
    with torch.no_grad():
        for bs in range(0, n, batch_size):
            be = min(bs + batch_size, n)
            logits, _ = model(tokens[bs:be])
            # Gather the LAST valid position's logits per row.
            L = lengths[bs:be]
            row_idx = torch.arange(be - bs, device=device)
            last_logits = logits[row_idx, torch.tensor(L - 1, device=device)]
            probs = F.softmax(last_logits, dim=-1)
            all_top1.append(probs.argmax(dim=-1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_top1), np.concatenate(all_probs)


def check_adversarial_and_score(model, dataset, device, active_beams,
                                  target_board_pos, target_token_idx,
                                  stoi_arr):
    """For each active beam, check adversarial condition and get target prob.

    Returns:
        target_probs: (N,) probability model assigns to target token
        adversarial: (N,) bool — True if target illegal AND top-1 == target
    """
    seqs = [b[0] for b in active_beams]
    top1, probs = batch_predict(model, dataset, device, seqs, stoi_arr)
    target_probs = probs[:, target_token_idx]
    adv = np.zeros(len(active_beams), dtype=bool)
    for i, (seq, brd, cum) in enumerate(active_beams):
        valid = brd.get_valid_moves()
        if target_board_pos in valid:
            continue
        if int(top1[i]) == target_token_idx:
            adv[i] = True
    return target_probs, adv


def run_start(model, dataset, device, target_cell, prefix, beam_width,
              max_depth, target_board_pos, target_token_idx, stoi_arr):
    """Run beam search from one prefix.  Return (adversarial_found, best_seq, best_score)."""
    board = OthelloBoardState()
    for m in prefix:
        board.umpire(m)
    if not board.get_valid_moves():
        return False, list(prefix), 0.0

    beams = [(list(prefix), fast_copy_board(board), 0.0)]
    adversarial_found = False
    best_seq, best_score = list(prefix), 0.0

    for depth in range(len(prefix), max_depth):
        # Expand each beam by every legal next move
        candidates = []
        for game_seq, brd, cum in beams:
            valid = brd.get_valid_moves()
            if not valid:
                continue  # this beam's game ended
            for m in valid:
                new_seq = game_seq + [m]
                nb = fast_copy_board(brd)
                nb.umpire(m)
                candidates.append((new_seq, nb, cum))
        if not candidates:
            break

        # Score all candidates + check adversarial for each
        target_probs, adv = check_adversarial_and_score(
            model, dataset, device, candidates,
            target_board_pos, target_token_idx, stoi_arr,
        )
        if adv.any():
            adversarial_found = True

        # Score: cumulative + target-cell prob when illegal (as behavioral_adversarial)
        new_scored = []
        for i, (seq, brd, cum) in enumerate(candidates):
            valid = brd.get_valid_moves()
            score = cum + (target_probs[i] if target_board_pos not in valid else 0.0)
            new_scored.append((seq, brd, float(score)))
            if score > best_score:
                best_score = float(score)
                best_seq = list(seq)

        new_scored.sort(key=lambda x: x[2], reverse=True)
        beams = new_scored[:beam_width]

        # Early exit once we've established the flag AND explored a few deep
        # levels; keeps the loop cheap.  Comment out to always run to max_depth.
        if adversarial_found and depth >= len(prefix) + 8:
            break

    return adversarial_found, best_seq, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', type=int, required=True,
                    help='Target cell index 0..59')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--output-dir', default='experiment1_data')
    ap.add_argument('--beam-width', type=int, default=10)
    ap.add_argument('--prefix-len', type=int, default=5)
    ap.add_argument('--max-depth', type=int, default=40)
    ap.add_argument('--top-save', type=int, default=5,
                    help='Save the top-N successful games per cell.')
    args = ap.parse_args()

    t0 = time.time()
    print(f"Cell {args.cell} (board pos {VALID_MOVES[args.cell]})", flush=True)
    print(f"Loading model {args.ckpt}...", flush=True)
    model, dataset, device = load_model(args.ckpt)
    model.eval()
    print(f"  Device: {device}", flush=True)

    target_board_pos = IDX_TO_MOVE[args.cell]
    _, pos_to_vocab = build_vocab_to_pos_map(dataset)
    target_token_idx = pos_to_vocab[target_board_pos]
    stoi_arr = np.zeros(64, dtype=np.int64)
    for pos in VALID_MOVES:
        stoi_arr[pos] = dataset.stoi[pos]

    print(f"Enumerating {args.prefix_len}-move prefixes...", flush=True)
    prefixes = enumerate_prefixes(args.prefix_len)
    print(f"  {len(prefixes)} unique prefixes", flush=True)

    n_success = 0
    successes = []  # (score, seq)
    success_mask = np.zeros(len(prefixes), dtype=bool)
    for pi, prefix in enumerate(prefixes):
        adv, seq, score = run_start(
            model, dataset, device, args.cell, prefix,
            args.beam_width, args.max_depth,
            target_board_pos, target_token_idx, stoi_arr,
        )
        if adv:
            n_success += 1
            success_mask[pi] = True
            successes.append((score, seq))
        if (pi + 1) % 100 == 0:
            print(f"  {pi+1}/{len(prefixes)}  "
                  f"success: {n_success}  ({int(time.time()-t0)}s)",
                  flush=True)

    rate = n_success / len(prefixes) if prefixes else 0.0
    print()
    print(f"=== Cell {args.cell} results ===")
    print(f"  Prefixes:              {len(prefixes)}")
    print(f"  Adversarial successes: {n_success}")
    print(f"  Success rate:          {rate:.4f}")
    print(f"  Elapsed:               {int(time.time()-t0)}s")

    os.makedirs(args.output_dir, exist_ok=True)
    successes.sort(key=lambda x: x[0], reverse=True)
    top_games = [s for _, s in successes[:args.top_save]]
    np.savez_compressed(
        os.path.join(args.output_dir, f"cell_{args.cell:02d}.npz"),
        cell=args.cell,
        n_prefixes=len(prefixes),
        n_success=n_success,
        rate=rate,
        success_mask=success_mask,
        top_scores=np.array([sc for sc, _ in successes[:args.top_save]],
                             dtype=np.float32),
        top_games=np.array(top_games, dtype=object),
    )


if __name__ == '__main__':
    main()
