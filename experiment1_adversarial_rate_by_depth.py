"""Experiment 1 variant: adversarial success rate by starting depth (no per-cell).

Complements experiment1_adversarial_rate.py.  Rather than enumerating all
1,396 five-move openings and targeting each of the 60 cells, this script:

  1. Randomly samples N valid game sequences of a given depth D from val games
  2. From each sample, runs beam-width-10 search extending the sequence
  3. Uses "probability mass OGPT assigns to illegal cells" as the beam score
     (no specific target cell)
  4. Success = OGPT's top-1 argmax is illegal at any position along the beam
  5. Reports the fraction of samples that yield adversarial success, per depth

Usage:
    python experiment1_adversarial_rate_by_depth.py \\
        --depth 10 --n-samples 1000 \\
        --ckpt ckpts/gpt_nanda_synthetic.ckpt \\
        --beam-width 10 --max-depth 40
"""
import argparse
import os
import sys
import time
import random
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


def batch_predict(model, dataset, device, seqs, stoi_arr):
    """Return top1 token per sample + probs vector per sample."""
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


def score_and_check_adversarial(active_beams, model, dataset, device,
                                 stoi_arr, vocab_to_pos):
    """For each beam, return (illegal_prob_mass, is_adversarial).

    illegal_prob_mass = sum of model probabilities over cells that are
    currently illegal — the beam-search steering signal.
    is_adversarial = True iff model's top-1 argmax lands on an illegal cell.
    """
    seqs = [b[0] for b in active_beams]
    top1, probs = batch_predict(model, dataset, device, seqs, stoi_arr)
    illegal_mass = np.zeros(len(active_beams), dtype=np.float32)
    is_adv = np.zeros(len(active_beams), dtype=bool)
    top1_illegal_pos = np.full(len(active_beams), -1, dtype=np.int64)
    for i, (seq, brd, cum) in enumerate(active_beams):
        valid_set = set(brd.get_valid_moves())
        for tok_idx in range(probs.shape[1]):
            board_pos = vocab_to_pos[tok_idx]
            if board_pos >= 0 and board_pos not in valid_set:
                illegal_mass[i] += probs[i, tok_idx]
        top1_pos = vocab_to_pos[int(top1[i])]
        if top1_pos >= 0 and top1_pos not in valid_set:
            is_adv[i] = True
            top1_illegal_pos[i] = top1_pos
    return illegal_mass, is_adv, top1_illegal_pos


def run_sample(model, dataset, device, prefix, beam_width, max_depth,
               stoi_arr, vocab_to_pos, save_positions=False):
    board = OthelloBoardState()
    for m in prefix:
        board.umpire(m)
    if not board.get_valid_moves():
        return False, []

    beams = [(list(prefix), fast_copy_board(board), 0.0)]
    adversarial_found = False
    records = []

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

        illegal_mass, is_adv, top1_illegal_pos = score_and_check_adversarial(
            candidates, model, dataset, device, stoi_arr, vocab_to_pos,
        )
        if is_adv.any():
            adversarial_found = True
            if save_positions:
                for i in range(len(candidates)):
                    if is_adv[i]:
                        seq = candidates[i][0]
                        records.append((tuple(seq), len(seq) - 1,
                                          int(top1_illegal_pos[i])))

        new_scored = [(seq, brd, cum + float(illegal_mass[i]))
                       for i, (seq, brd, cum) in enumerate(candidates)]
        new_scored.sort(key=lambda x: x[2], reverse=True)
        beams = new_scored[:beam_width]

        if adversarial_found and depth >= len(prefix) + 6:
            break

    return adversarial_found, records


def load_val_games(data_dir, max_files, min_length=59):
    """Load games from val pickle files that are at least min_length moves long."""
    import pickle
    import os
    files = sorted(os.listdir(data_dir))[-max_files:]
    games = []
    for fname in files:
        p = os.path.join(data_dir, fname)
        try:
            with open(p, 'rb') as f:
                batch = pickle.load(f)
        except Exception:
            continue
        if len(batch) < 9e4:
            continue
        for g in batch:
            if len(g) >= min_length:
                games.append(g)
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, required=True,
                    help='Starting sequence length (moves already played).')
    ap.add_argument('--n-samples', type=int, default=1000)
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--output-dir', default='experiment1_by_depth')
    ap.add_argument('--beam-width', type=int, default=10)
    ap.add_argument('--max-depth', type=int, default=40)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save-positions', action='store_true',
                    help='Save every (game, turn, illegal_cell) triple '
                         'to adversarial_records.npz for downstream causal '
                         'analysis.')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    print(f"Depth {args.depth}  n_samples={args.n_samples}", flush=True)
    print(f"Loading model {args.ckpt}...", flush=True)
    model, dataset, device = load_model(args.ckpt)
    model.eval()
    print(f"  Device: {device}", flush=True)

    vocab_to_pos, _ = build_vocab_to_pos_map(dataset)
    stoi_arr = np.zeros(64, dtype=np.int64)
    for pos in VALID_MOVES:
        stoi_arr[pos] = dataset.stoi[pos]

    print(f"Loading val games from {args.num_data_files} pickle files...", flush=True)
    games = load_val_games(args.data_dir, args.num_data_files,
                            min_length=args.depth)
    print(f"  {len(games)} val games with length >= {args.depth}", flush=True)

    # Sample n_samples games and truncate each to depth
    idx = np.random.choice(len(games), size=min(args.n_samples, len(games)),
                            replace=False)
    prefixes = [games[i][:args.depth] for i in idx]
    print(f"Sampled {len(prefixes)} prefixes at depth {args.depth}", flush=True)

    n_success = 0
    all_records = []
    for pi, prefix in enumerate(prefixes):
        adv, records = run_sample(
            model, dataset, device, prefix,
            args.beam_width, args.max_depth,
            stoi_arr, vocab_to_pos, save_positions=args.save_positions,
        )
        if adv:
            n_success += 1
        if args.save_positions:
            all_records.extend(records)
        if (pi + 1) % 100 == 0:
            print(f"  {pi+1}/{len(prefixes)}  "
                  f"success: {n_success}  "
                  f"records: {len(all_records)}  "
                  f"({int(time.time()-t0)}s)", flush=True)

    rate = n_success / len(prefixes) if prefixes else 0.0
    print()
    print(f"=== Depth {args.depth} results ===")
    print(f"  Samples:               {len(prefixes)}")
    print(f"  Adversarial successes: {n_success}")
    print(f"  Success rate:          {rate:.4f}")
    print(f"  Elapsed:               {int(time.time()-t0)}s")

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.output_dir, f"depth_{args.depth:02d}.npz"),
        depth=args.depth,
        n_samples=len(prefixes),
        n_success=n_success,
        rate=rate,
    )
    if args.save_positions:
        games_arr = np.array([r[0] for r in all_records], dtype=object)
        turns_arr = np.array([r[1] for r in all_records], dtype=np.int64)
        cells_arr = np.array([r[2] for r in all_records], dtype=np.int64)
        np.savez_compressed(
            os.path.join(args.output_dir,
                          f"adversarial_records_depth_{args.depth:02d}.npz"),
            games=games_arr,
            turns=turns_arr,
            illegal_cells=cells_arr,
        )
        print(f"  Saved {len(all_records)} adversarial records to "
              f"adversarial_records_depth_{args.depth:02d}.npz")


if __name__ == '__main__':
    main()
