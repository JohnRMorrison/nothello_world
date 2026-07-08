"""Establish "for all 1,396 5-move openings, at least one adversarial game
exists" — without per-cell decomposition.

Enumerates all 1,396 unique 5-move Othello openings.  From each, runs a
beam-width-10 search whose score is the total probability OGPT assigns
to illegal cells (no specific target cell).  Success = at any position
along the beam, OGPT's top-1 argmax is illegal.

Same scoring / success rule as experiment1_adversarial_rate_by_depth.py,
but the sample source is exhaustive enumeration at depth 5 rather than
random val-game truncation.

Usage:
    python experiment1_any_cell_from_enumeration.py \\
        --ckpt ckpts/gpt_nanda_synthetic.ckpt \\
        --beam-width 10 --max-depth 40
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
    """All valid game-legal move sequences of the given length."""
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
    """Return top1 token + probs vector per sequence."""
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

    all_top1, all_probs = [], []
    with torch.no_grad():
        for bs in range(0, n, 256):
            be = min(bs + 256, n)
            logits, _ = model(tokens[bs:be])
            L = lengths[bs:be]
            row_idx = torch.arange(be - bs, device=device)
            last_logits = logits[row_idx, torch.tensor(L - 1, device=device)]
            probs = F.softmax(last_logits, dim=-1)
            all_top1.append(probs.argmax(dim=-1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_top1), np.concatenate(all_probs)


def score_and_check(active_beams, model, dataset, device, stoi_arr,
                     vocab_to_pos):
    """For each beam, return (illegal_prob_mass, is_adversarial).

    illegal_prob_mass = sum of softmax over cells currently illegal.
    is_adversarial = True iff model top-1 argmax lands on illegal cell.
    """
    seqs = [b[0] for b in active_beams]
    top1, probs = batch_predict(model, dataset, device, seqs, stoi_arr)
    illegal_mass = np.zeros(len(active_beams), dtype=np.float32)
    is_adv = np.zeros(len(active_beams), dtype=bool)
    for i, (seq, brd, cum) in enumerate(active_beams):
        valid_set = set(brd.get_valid_moves())
        for tok_idx in range(probs.shape[1]):
            board_pos = vocab_to_pos[tok_idx]
            if board_pos >= 0 and board_pos not in valid_set:
                illegal_mass[i] += probs[i, tok_idx]
        top1_pos = vocab_to_pos[int(top1[i])]
        if top1_pos >= 0 and top1_pos not in valid_set:
            is_adv[i] = True
    return illegal_mass, is_adv


def run_from_prefix(model, dataset, device, prefix, beam_width, max_depth,
                     stoi_arr, vocab_to_pos):
    board = OthelloBoardState()
    for m in prefix:
        board.umpire(m)
    if not board.get_valid_moves():
        return False

    beams = [(list(prefix), fast_copy_board(board), 0.0)]
    adversarial_found = False
    for depth in range(len(prefix), max_depth):
        candidates = []
        for game_seq, brd, cum in beams:
            valid = brd.get_valid_moves()
            if not valid:
                continue
            for m in valid:
                new_seq = game_seq + [m]
                nb = fast_copy_board(brd)
                nb.umpire(m)
                candidates.append((new_seq, nb, cum))
        if not candidates:
            break

        illegal_mass, is_adv = score_and_check(
            candidates, model, dataset, device, stoi_arr, vocab_to_pos)
        if is_adv.any():
            adversarial_found = True

        new_scored = [
            (seq, brd, cum + float(illegal_mass[i]))
            for i, (seq, brd, cum) in enumerate(candidates)
        ]
        new_scored.sort(key=lambda x: x[2], reverse=True)
        beams = new_scored[:beam_width]

        if adversarial_found and depth >= len(prefix) + 6:
            break

    return adversarial_found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--output-dir', default='experiment1_any_cell')
    ap.add_argument('--beam-width', type=int, default=10)
    ap.add_argument('--prefix-len', type=int, default=5)
    ap.add_argument('--max-depth', type=int, default=40)
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading model {args.ckpt}...", flush=True)
    model, dataset, device = load_model(args.ckpt)
    model.eval()
    print(f"  Device: {device}", flush=True)

    vocab_to_pos, _ = build_vocab_to_pos_map(dataset)
    stoi_arr = np.zeros(64, dtype=np.int64)
    for pos in VALID_MOVES:
        stoi_arr[pos] = dataset.stoi[pos]

    print(f"Enumerating {args.prefix_len}-move prefixes...", flush=True)
    prefixes = enumerate_prefixes(args.prefix_len)
    print(f"  {len(prefixes)} unique prefixes", flush=True)

    n_success = 0
    success_mask = np.zeros(len(prefixes), dtype=bool)
    for pi, prefix in enumerate(prefixes):
        if run_from_prefix(
            model, dataset, device, prefix,
            args.beam_width, args.max_depth, stoi_arr, vocab_to_pos,
        ):
            n_success += 1
            success_mask[pi] = True
        if (pi + 1) % 100 == 0:
            print(f"  {pi+1}/{len(prefixes)}  "
                  f"success: {n_success}  ({int(time.time()-t0)}s)",
                  flush=True)

    rate = n_success / len(prefixes)
    print()
    print(f"=== Results ===")
    print(f"  Prefixes:              {len(prefixes)}")
    print(f"  Adversarial successes: {n_success}")
    print(f"  Success rate:          {rate:.4f}")
    print(f"  Elapsed:               {int(time.time()-t0)}s")

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.output_dir, 'enumeration.npz'),
        n_prefixes=len(prefixes),
        n_success=n_success,
        rate=rate,
        success_mask=success_mask,
    )


if __name__ == '__main__':
    main()
