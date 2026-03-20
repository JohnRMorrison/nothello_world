"""Screen board-state triples for heuristic signal on natural adversarial positions.

Replays games to get board state (black/white/empty per cell), runs model
inference to get probabilities, then compares triple firing rates between
adversarial (illegal but model promotes) and phase-matched control positions.

Usage:
    python behavioral_boardstate_screen.py --cells 10,20,30,40,50 --n-games 50000
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
from itertools import combinations

import torch
import torch.nn.functional as F

from behavioral_utils import (
    load_model, build_vocab_to_pos_map, extract_probs_60d,
    N_MOVES, VALID_MOVES, MOVE_TO_IDX, IDX_TO_MOVE, POS_START, POS_END,
    load_shard_games
)
from data.othello import OthelloBoardState


def collect_board_states_and_probs(games, model, dataset, device, batch_size=64):
    """Replay games to get board states, run inference for probs.

    Returns:
        board_states: (n_positions, 64) int8 array (1=black, -1=white, 0=empty)
        probs_60: (n_positions, 60) float32 array
        legal_60: (n_positions, 60) uint8 array
        move_numbers: (n_positions,) int8 array
    """
    vocab_to_pos, pos_to_vocab = build_vocab_to_pos_map(dataset)
    block_size = dataset.block_size

    # Pre-tokenize for batched inference
    n_games = len(games)
    all_tokens = np.zeros((n_games, block_size), dtype=np.int64)
    stoi_arr = np.zeros(64, dtype=np.int64)
    for pos in VALID_MOVES:
        stoi_arr[pos] = dataset.stoi[pos]
    for i, game in enumerate(games):
        seq_len = min(len(game), block_size)
        all_tokens[i, :seq_len] = stoi_arr[np.array(game[:seq_len])]

    # Batched inference
    all_probs = np.zeros((n_games, block_size, N_MOVES), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_games, batch_size):
            end = min(start + batch_size, n_games)
            tokens_batch = torch.tensor(all_tokens[start:end],
                                        dtype=torch.long).to(device)
            logits, _ = model(tokens_batch)
            probs = F.softmax(logits, dim=-1)
            all_probs[start:end] = extract_probs_60d(probs, vocab_to_pos)

            if (start // batch_size) % 100 == 0:
                print(f"    Inference: {end}/{n_games}", flush=True)

    # Replay games for board states and legal moves
    board_list = []
    prob_list = []
    legal_list = []
    move_list = []

    for gi, game in enumerate(games):
        board = OthelloBoardState()
        for t in range(len(game)):
            if POS_START <= t < min(POS_END, block_size + 1):
                # Board state
                bs = board.state.flatten().astype(np.int8)  # (64,)

                # Legal moves
                valid = board.get_valid_moves()
                lm = np.zeros(N_MOVES, dtype=np.uint8)
                for m in valid:
                    if m in MOVE_TO_IDX:
                        lm[MOVE_TO_IDX[m]] = 1

                # Model probs at position t are at logits index t-1
                p = all_probs[gi, t - 1, :]

                board_list.append(bs)
                prob_list.append(p)
                legal_list.append(lm)
                move_list.append(t)

            board.umpire(game[t])

        if (gi + 1) % 10000 == 0:
            print(f"    Replayed {gi+1}/{n_games} games", flush=True)

    return (np.array(board_list, dtype=np.int8),
            np.array(prob_list, dtype=np.float32),
            np.array(legal_list, dtype=np.uint8),
            np.array(move_list, dtype=np.int8))


def screen_board_triples(adv_board, ctrl_board, target_cell, top_k=20):
    """Screen triples of board-state conditions.

    Features: for each of 64 cells, check if it's black (state==1),
    white (state==-1), or occupied (state!=0).

    We test triples of (cell_i is color_i) conditions.
    To keep it tractable, test:
      - All triples of "cell X is occupied" (64 choose 3 = 41664)
      - All triples of "cell X is black/white" for cells near target
    """
    # Occupied triples (most general)
    adv_occ = (adv_board != 0)  # (n, 64) bool
    ctrl_occ = (ctrl_board != 0)

    # For color-aware: black (1) and white (-1)
    adv_black = (adv_board == 1)
    adv_white = (adv_board == -1)
    ctrl_black = (ctrl_board == 1)
    ctrl_white = (ctrl_board == -1)

    target_pos = IDX_TO_MOVE[target_cell]
    other_positions = [p for p in range(64) if p != target_pos]

    # Screen color-aware pairs first (more informative than triples)
    # For each pair of cells, test: both black, both white, one black + one white
    print("  --- Color-aware pair screening ---", flush=True)
    pair_results = []
    for a, b in combinations(other_positions, 2):
        # Test: a is black AND b is white (flanking-like pattern)
        for (a_feat, a_name), (b_feat, b_name) in [
            ((adv_black, "B"), (ctrl_black, "B")),  # skip, need both sides
        ]:
            pass

        # Test all 4 color combos for the pair
        for a_color, a_label, a_adv, a_ctrl in [
            (1, "B", adv_black, ctrl_black), (-1, "W", adv_white, ctrl_white)
        ]:
            for b_color, b_label, b_adv, b_ctrl in [
                (1, "B", adv_black, ctrl_black), (-1, "W", adv_white, ctrl_white)
            ]:
                af = float((a_adv[:, a] & b_adv[:, b]).mean())
                if af < 0.03:
                    continue
                cf = float((a_ctrl[:, a] & b_ctrl[:, b]).mean())
                ratio = af / max(cf, 1e-6)
                if ratio > 1.3:
                    ra, ca = a // 8, a % 8
                    rb, cb = b // 8, b % 8
                    pair_results.append({
                        'cells': f"{chr(65+ra)}{ca+1}={a_label}, {chr(65+rb)}{cb+1}={b_label}",
                        'adv_rate': af, 'ctrl_rate': cf, 'ratio': ratio,
                    })

    pair_results.sort(key=lambda x: x['ratio'], reverse=True)
    print(f"  Color pairs with ratio > 1.3: {len(pair_results)}", flush=True)
    for p in pair_results[:15]:
        print(f"    {p['cells']}: adv={p['adv_rate']:.3f} "
              f"ctrl={p['ctrl_rate']:.3f} ratio={p['ratio']:.2f}x", flush=True)

    # Color-aware triples: a=color1, b=color2, c=color3
    print("\n  --- Color-aware triple screening ---", flush=True)
    triple_results = []
    # Limit to nearby cells (within distance 3) to keep tractable
    tr, tc = target_pos // 8, target_pos % 8
    nearby = [p for p in other_positions
              if abs(p // 8 - tr) <= 3 and abs(p % 8 - tc) <= 3]
    print(f"  Testing triples from {len(nearby)} nearby cells", flush=True)

    color_feats = [
        (1, "B", adv_black, ctrl_black),
        (-1, "W", adv_white, ctrl_white),
    ]

    for a, b, c in combinations(nearby, 3):
        for _, al, a_adv, a_ctrl in color_feats:
            for _, bl, b_adv, b_ctrl in color_feats:
                for _, cl, c_adv, c_ctrl in color_feats:
                    af = float((a_adv[:, a] & b_adv[:, b] & c_adv[:, c]).mean())
                    if af < 0.03:
                        continue
                    cf = float((a_ctrl[:, a] & b_ctrl[:, b] & c_ctrl[:, c]).mean())
                    ratio = af / max(cf, 1e-6)
                    if ratio > 1.3:
                        ra, ca = a // 8, a % 8
                        rb, cb = b // 8, b % 8
                        rc, cc = c // 8, c % 8
                        triple_results.append({
                            'cells': (f"{chr(65+ra)}{ca+1}={al}, "
                                      f"{chr(65+rb)}{cb+1}={bl}, "
                                      f"{chr(65+rc)}{cc+1}={cl}"),
                            'adv_rate': af, 'ctrl_rate': cf, 'ratio': ratio,
                        })

    triple_results.sort(key=lambda x: x['ratio'], reverse=True)
    print(f"  Color triples with ratio > 1.3: {len(triple_results)}", flush=True)
    for t in triple_results[:15]:
        print(f"    {t['cells']}: adv={t['adv_rate']:.3f} "
              f"ctrl={t['ctrl_rate']:.3f} ratio={t['ratio']:.2f}x", flush=True)

    return pair_results, triple_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=str, default="10,20,30,40,50")
    parser.add_argument("--n-games", type=int, default=50000)
    parser.add_argument("--ckpt", type=str, default="./ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--games-dir", type=str, default="data/othello_synthetic")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    cells = [int(c) for c in args.cells.split(",")]

    print("Loading model...", flush=True)
    model, dataset, device = load_model(args.ckpt)
    print(f"  Device: {device}", flush=True)

    print(f"Loading {args.n_games} games...", flush=True)
    games = load_shard_games(0, args.n_games, args.games_dir)
    print(f"  Loaded {len(games)} games", flush=True)

    print("Computing board states and model probs...", flush=True)
    board_states, probs, legal, move_nums = collect_board_states_and_probs(
        games, model, dataset, device, args.batch_size)
    print(f"  {len(board_states)} positions total", flush=True)

    for cell in cells:
        pos = VALID_MOVES[cell]
        name = chr(65 + pos // 8) + str(pos % 8 + 1)
        print(f"\n{'=' * 60}", flush=True)
        print(f"Cell {cell} ({name}, board pos {pos})", flush=True)
        print(f"{'=' * 60}", flush=True)

        cell_prob = probs[:, cell]
        cell_legal = legal[:, cell]

        # Natural adversarial: illegal but model assigns > 0.5%
        adv_mask = (cell_legal == 0) & (cell_prob > 0.005)
        # Control: illegal and model assigns < 0.1%
        ctrl_mask = (cell_legal == 0) & (cell_prob < 0.001)

        adv_board = board_states[adv_mask]
        adv_moves = move_nums[adv_mask]
        ctrl_board_all = board_states[ctrl_mask]
        ctrl_moves_all = move_nums[ctrl_mask]

        print(f"  Natural adv: {len(adv_board)}, Control pool: {len(ctrl_board_all)}",
              flush=True)
        print(f"  Adv move range: {adv_moves.min()}-{adv_moves.max()}, "
              f"mean={adv_moves.mean():.1f}", flush=True)

        # Phase match
        selected = []
        for mv in range(60):
            adv_at = int((adv_moves == mv).sum())
            if adv_at == 0:
                continue
            ctrl_at = np.where(ctrl_moves_all == mv)[0]
            if len(ctrl_at) == 0:
                continue
            n = min(len(ctrl_at), adv_at * 15)
            chosen = np.random.choice(ctrl_at, n, replace=False)
            selected.extend(chosen.tolist())

        ctrl_board = ctrl_board_all[selected]
        print(f"  Phase-matched control: {len(ctrl_board)}", flush=True)

        t0 = time.time()
        screen_board_triples(adv_board, ctrl_board, cell)
        print(f"  Elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
