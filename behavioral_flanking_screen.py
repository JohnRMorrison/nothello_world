"""Screen partial flanking patterns as heuristic candidates.

For each target cell and each of its ~16 flanking patterns, check whether
partial matches (some but not all cells correct) fire more often in
adversarial positions than phase-matched control. Partial matches that
the model relies on but that don't guarantee legality are heuristics.

Usage:
    python behavioral_flanking_screen.py --cells 10,20,30,40,50 --n-games 50000
"""

import argparse
import os
import sys
import time
import numpy as np

import torch
import torch.nn.functional as F

from behavioral_utils import (
    load_model, build_vocab_to_pos_map, extract_probs_60d,
    N_MOVES, VALID_MOVES, MOVE_TO_IDX, IDX_TO_MOVE, POS_START, POS_END,
    load_shard_games
)
from hand_crafted_flanking import enumerate_flanking_patterns
from data.othello import OthelloBoardState


def collect_board_states_and_probs(games, model, dataset, device, batch_size=64):
    """Replay games to get board states, run inference for probs."""
    vocab_to_pos, _ = build_vocab_to_pos_map(dataset)
    block_size = dataset.block_size
    n_games = len(games)

    # Vectorized tokenization
    stoi_arr = np.zeros(64, dtype=np.int64)
    for pos in VALID_MOVES:
        stoi_arr[pos] = dataset.stoi[pos]
    all_tokens = np.zeros((n_games, block_size), dtype=np.int64)
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

    # Replay for board states and legal masks
    board_list, prob_list, legal_list, move_list = [], [], [], []
    for gi, game in enumerate(games):
        board = OthelloBoardState()
        for t in range(len(game)):
            if POS_START <= t < min(POS_END, block_size + 1):
                bs = board.state.flatten().astype(np.int8)
                valid = board.get_valid_moves()
                lm = np.zeros(N_MOVES, dtype=np.uint8)
                for m in valid:
                    if m in MOVE_TO_IDX:
                        lm[MOVE_TO_IDX[m]] = 1
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


def get_patterns_for_cell(cell_idx):
    """Get all flanking patterns where the target is this cell."""
    target_pos = IDX_TO_MOVE[cell_idx]
    patterns = enumerate_flanking_patterns()
    return [p for p in patterns if p['target'] == target_pos]


def check_pattern_match(board_state, pattern, player_color):
    """Check how many cells in the pattern match the expected colors.

    A full flanking pattern requires:
      - target cell is empty
      - each opponent cell has opponent's color
      - terminal cell has player's color

    Returns:
        full_match: bool (all conditions met = legal move via this pattern)
        n_opponent_match: how many opponent cells have opponent color
        terminal_match: whether terminal cell has player color
        n_total: total cells to check (len(opponents) + 1 for terminal)
    """
    opponent_color = -player_color
    target = pattern['target']
    opponents = pattern['opponents']
    terminal = pattern['terminal']

    # Target must be empty
    if board_state[target] != 0:
        return False, 0, False, len(opponents) + 1

    n_opp_match = sum(1 for opp in opponents
                      if board_state[opp] == opponent_color)
    term_match = board_state[terminal] == player_color

    full = (n_opp_match == len(opponents)) and term_match
    return full, n_opp_match, term_match, len(opponents) + 1


def screen_partial_flanking(adv_boards, ctrl_boards, cell_idx,
                            adv_moves, ctrl_moves):
    """Screen partial flanking patterns for this cell.

    For each flanking pattern, compute partial match statistics for
    adversarial vs phase-matched control positions.
    """
    patterns = get_patterns_for_cell(cell_idx)
    if not patterns:
        print(f"  No flanking patterns for cell {cell_idx}", flush=True)
        return []

    target_pos = IDX_TO_MOVE[cell_idx]
    print(f"  {len(patterns)} flanking patterns for this cell", flush=True)

    results = []

    for pi, pattern in enumerate(patterns):
        dr, dc = pattern['direction']
        dir_name = {(-1,0): 'up', (1,0): 'down', (0,-1): 'left', (0,1): 'right',
                    (-1,-1): 'up-left', (-1,1): 'up-right',
                    (1,-1): 'down-left', (1,1): 'down-right'}[pattern['direction']]
        opp_cells = pattern['opponents']
        term_cell = pattern['terminal']
        length = pattern['length']

        # For both colors (player could be black=1 or white=-1)
        for player_color, color_label in [(1, 'black'), (-1, 'white')]:
            opponent_color = -player_color

            # Vectorized: check each condition across all positions
            # Target must be empty
            adv_target_empty = adv_boards[:, target_pos] == 0
            ctrl_target_empty = ctrl_boards[:, target_pos] == 0

            # Opponent cells have opponent color
            adv_opp = np.ones(len(adv_boards), dtype=bool)
            ctrl_opp = np.ones(len(ctrl_boards), dtype=bool)
            for opp in opp_cells:
                adv_opp &= (adv_boards[:, opp] == opponent_color)
                ctrl_opp &= (ctrl_boards[:, opp] == opponent_color)

            # Terminal has player color
            adv_term = adv_boards[:, term_cell] == player_color
            ctrl_term = ctrl_boards[:, term_cell] == player_color

            # Full match (all conditions)
            adv_full = adv_target_empty & adv_opp & adv_term
            ctrl_full = ctrl_target_empty & ctrl_opp & ctrl_term

            # Partial: opponents match but terminal doesn't
            adv_partial_no_term = adv_target_empty & adv_opp & ~adv_term
            ctrl_partial_no_term = ctrl_target_empty & ctrl_opp & ~ctrl_term

            # Partial: terminal matches but not all opponents
            # (check each subset of opponents)
            # For simplicity, check: at least (length-1) opponents match
            if length > 1:
                adv_opp_count = np.zeros(len(adv_boards), dtype=int)
                ctrl_opp_count = np.zeros(len(ctrl_boards), dtype=int)
                for opp in opp_cells:
                    adv_opp_count += (adv_boards[:, opp] == opponent_color).astype(int)
                    ctrl_opp_count += (ctrl_boards[:, opp] == opponent_color).astype(int)
                adv_most_opp = adv_target_empty & (adv_opp_count >= length - 1) & adv_term
                ctrl_most_opp = ctrl_target_empty & (ctrl_opp_count >= length - 1) & ctrl_term
            else:
                adv_most_opp = adv_full
                ctrl_most_opp = ctrl_full

            # Compute rates
            metrics = {
                'full': (adv_full.mean(), ctrl_full.mean()),
                'opp_no_term': (adv_partial_no_term.mean(), ctrl_partial_no_term.mean()),
            }
            if length > 1:
                metrics['most_opp_with_term'] = (adv_most_opp.mean(), ctrl_most_opp.mean())

            for match_type, (adv_rate, ctrl_rate) in metrics.items():
                if adv_rate < 0.01:
                    continue
                ratio = adv_rate / max(ctrl_rate, 1e-6)
                if ratio > 1.2 or match_type == 'full':
                    opp_names = [f"{chr(65+o//8)}{o%8+1}" for o in opp_cells]
                    term_name = f"{chr(65+term_cell//8)}{term_cell%8+1}"

                    results.append({
                        'direction': dir_name,
                        'length': length,
                        'opponents': opp_names,
                        'terminal': term_name,
                        'player': color_label,
                        'match_type': match_type,
                        'adv_rate': float(adv_rate),
                        'ctrl_rate': float(ctrl_rate),
                        'ratio': float(ratio),
                    })

    results.sort(key=lambda x: x['ratio'], reverse=True)
    return results


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

        adv_mask = (cell_legal == 0) & (cell_prob > 0.005)
        ctrl_mask = (cell_legal == 0) & (cell_prob < 0.001)

        adv_boards = board_states[adv_mask]
        adv_moves = move_nums[adv_mask]
        ctrl_boards_all = board_states[ctrl_mask]
        ctrl_moves_all = move_nums[ctrl_mask]

        print(f"  Natural adv: {len(adv_boards)}, Control pool: {len(ctrl_boards_all)}",
              flush=True)

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

        ctrl_boards = ctrl_boards_all[selected]
        ctrl_moves = ctrl_moves_all[selected]
        print(f"  Phase-matched control: {len(ctrl_boards)}", flush=True)

        t0 = time.time()
        results = screen_partial_flanking(adv_boards, ctrl_boards, cell,
                                          adv_moves, ctrl_moves)

        # Print results grouped by match type
        for match_type in ['opp_no_term', 'most_opp_with_term', 'full']:
            typed = [r for r in results if r['match_type'] == match_type]
            if not typed:
                continue
            print(f"\n  --- {match_type} ---", flush=True)
            for r in typed[:10]:
                print(f"    {r['direction']} len={r['length']} "
                      f"opp={r['opponents']} term={r['terminal']} "
                      f"player={r['player']}: "
                      f"adv={r['adv_rate']:.3f} ctrl={r['ctrl_rate']:.3f} "
                      f"ratio={r['ratio']:.2f}x", flush=True)

        print(f"\n  Total patterns with ratio > 1.2: {len(results)}", flush=True)
        print(f"  Elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
