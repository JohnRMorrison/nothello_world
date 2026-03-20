"""Compute sensitivity of OthelloGPT to each of the 960 flanking rules.

For each rule, sensitivity = excess probability the model assigns to the
target cell when the rule is satisfied vs not, controlling for move number
and filtering to positions where the target cell is empty.

Usage:
    python behavioral_sensitivity.py --n-games 50000
    python behavioral_sensitivity.py --n-games 50000 --output sensitivity.json
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from collections import defaultdict

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
    """Replay games for board states, run inference for probs."""
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
                print(f"  Inference: {end}/{n_games}", flush=True)

    # Replay for board states
    board_list, prob_list, move_list, color_list = [], [], [], []
    for gi, game in enumerate(games):
        board = OthelloBoardState()
        for t in range(len(game)):
            if POS_START <= t < min(POS_END, block_size + 1):
                bs = board.state.flatten().astype(np.int8)  # (64,)
                p = all_probs[gi, t - 1, :]  # (60,)
                # Current player: 1=black, -1=white
                player = board.next_hand_color
                board_list.append(bs)
                prob_list.append(p)
                move_list.append(t)
                color_list.append(player)
            board.umpire(game[t])
        if (gi + 1) % 10000 == 0:
            print(f"  Replayed {gi+1}/{n_games} games", flush=True)

    return (np.array(board_list, dtype=np.int8),
            np.array(prob_list, dtype=np.float32),
            np.array(move_list, dtype=np.int8),
            np.array(color_list, dtype=np.int8))


def compute_sensitivity(board_states, probs, move_nums, player_colors,
                        patterns, min_samples=500):
    """Compute sensitivity for each of the 960 flanking rules.

    Returns list of dicts with sensitivity scores and metadata.
    """
    n_positions = len(board_states)
    n_patterns = len(patterns)

    # Group patterns by target cell for efficiency
    patterns_by_target = defaultdict(list)
    for pi, pat in enumerate(patterns):
        patterns_by_target[pat['target']].append((pi, pat))

    # For each rule, accumulate stats by move number
    # stats[rule_id][move_num] = {'sat_sum': ..., 'sat_count': ...,
    #                              'nosat_sum': ..., 'nosat_count': ...}
    stats = [defaultdict(lambda: {'sat_sum': 0.0, 'sat_count': 0,
                                   'nosat_sum': 0.0, 'nosat_count': 0})
             for _ in range(n_patterns)]

    print(f"  Processing {n_positions} positions...", flush=True)

    for pos_idx in range(n_positions):
        bs = board_states[pos_idx]
        player = player_colors[pos_idx]
        opponent = -player
        move_num = int(move_nums[pos_idx])

        # For each target cell that is empty at this position
        for target_pos in range(64):
            if bs[target_pos] != 0:
                continue

            if target_pos not in patterns_by_target:
                continue

            # Get model prob for this cell
            if target_pos not in MOVE_TO_IDX:
                continue
            cell_idx = MOVE_TO_IDX[target_pos]
            prob = float(probs[pos_idx, cell_idx])

            # Check each rule for this target
            for pi, pat in patterns_by_target[target_pos]:
                # Check if rule is satisfied for current player
                opp_cells = pat['opponents']
                term_cell = pat['terminal']

                all_opp = all(bs[opp] == opponent for opp in opp_cells)
                term_ok = (bs[term_cell] == player)

                satisfied = all_opp and term_ok

                if satisfied:
                    stats[pi][move_num]['sat_sum'] += prob
                    stats[pi][move_num]['sat_count'] += 1
                else:
                    stats[pi][move_num]['nosat_sum'] += prob
                    stats[pi][move_num]['nosat_count'] += 1

        if (pos_idx + 1) % 500000 == 0:
            print(f"    {pos_idx+1}/{n_positions} positions", flush=True)

    # Compute sensitivity per rule
    results = []
    for pi, pat in enumerate(patterns):
        rule_stats = stats[pi]

        total_sat = sum(v['sat_count'] for v in rule_stats.values())
        total_nosat = sum(v['nosat_count'] for v in rule_stats.values())

        # Weighted average of excess across move numbers
        weighted_excess = 0.0
        total_weight = 0
        excess_by_move = {}

        for move_num, s in sorted(rule_stats.items()):
            sc = s['sat_count']
            nc = s['nosat_count']
            if sc == 0 or nc == 0:
                continue

            sat_mean = s['sat_sum'] / sc
            nosat_mean = s['nosat_sum'] / nc
            excess = sat_mean - nosat_mean
            weight = min(sc, nc)  # weight by smaller group

            weighted_excess += excess * weight
            total_weight += weight
            excess_by_move[int(move_num)] = {
                'excess': float(excess),
                'sat_mean': float(sat_mean),
                'nosat_mean': float(nosat_mean),
                'sat_count': sc,
                'nosat_count': nc,
            }

        sensitivity = weighted_excess / total_weight if total_weight > 0 else 0.0

        # Base rate: overall prob for target when empty
        base_sum = sum(v['sat_sum'] + v['nosat_sum'] for v in rule_stats.values())
        base_count = total_sat + total_nosat
        base_rate = base_sum / base_count if base_count > 0 else 0.0

        opp_names = [f"{chr(65+o//8)}{o%8+1}" for o in pat['opponents']]
        term_name = f"{chr(65+pat['terminal']//8)}{pat['terminal']%8+1}"
        target_name = f"{chr(65+pat['target']//8)}{pat['target']%8+1}"
        dir_name = {(-1,0): 'up', (1,0): 'down', (0,-1): 'left', (0,1): 'right',
                    (-1,-1): 'up-left', (-1,1): 'up-right',
                    (1,-1): 'down-left', (1,1): 'down-right'}[pat['direction']]

        results.append({
            'rule_id': pi,
            'target': pat['target'],
            'target_name': target_name,
            'opponents': [int(o) for o in pat['opponents']],
            'opponent_names': opp_names,
            'terminal': pat['terminal'],
            'terminal_name': term_name,
            'direction': dir_name,
            'direction_vec': list(pat['direction']),
            'length': pat['length'],
            'sensitivity': float(sensitivity),
            'n_satisfied': total_sat,
            'n_not_satisfied': total_nosat,
            'base_rate': float(base_rate),
            'reliable': total_sat >= min_samples,
            'excess_by_move': excess_by_move,
        })

    return results


def print_summary(results):
    """Print summary of sensitivity computation."""
    sensitivities = [r['sensitivity'] for r in results]
    n_reliable = sum(1 for r in results if r['reliable'])
    n_positive = sum(1 for r in results if r['sensitivity'] > 0)
    n_underpowered = sum(1 for r in results if not r['reliable'])

    print(f"\n{'=' * 60}", flush=True)
    print("SENSITIVITY SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Total rules: {len(results)}", flush=True)
    print(f"  Reliable (n_sat >= 500): {n_reliable}", flush=True)
    print(f"  Underpowered (n_sat < 500): {n_underpowered}", flush=True)
    print(f"  Positive sensitivity: {n_positive}", flush=True)
    print(f"  Mean sensitivity: {np.mean(sensitivities):.5f}", flush=True)
    print(f"  Median sensitivity: {np.median(sensitivities):.5f}", flush=True)
    print(f"  Max sensitivity: {np.max(sensitivities):.5f}", flush=True)
    print(f"  Min sensitivity: {np.min(sensitivities):.5f}", flush=True)

    # Top 20 most sensitive rules
    sorted_results = sorted(results, key=lambda x: x['sensitivity'], reverse=True)
    print(f"\n  Top 20 most sensitive rules:", flush=True)
    for r in sorted_results[:20]:
        flag = "" if r['reliable'] else " [LOW N]"
        print(f"    {r['target_name']} {r['direction']} len={r['length']} "
              f"opp={r['opponent_names']} term={r['terminal_name']}: "
              f"sens={r['sensitivity']:.5f} "
              f"(n_sat={r['n_satisfied']}, base={r['base_rate']:.4f}){flag}",
              flush=True)

    # Bottom 20 (least sensitive, among reliable)
    reliable_sorted = sorted([r for r in results if r['reliable']],
                             key=lambda x: x['sensitivity'])
    print(f"\n  Bottom 20 least sensitive (reliable only):", flush=True)
    for r in reliable_sorted[:20]:
        print(f"    {r['target_name']} {r['direction']} len={r['length']} "
              f"opp={r['opponent_names']} term={r['terminal_name']}: "
              f"sens={r['sensitivity']:.5f} "
              f"(n_sat={r['n_satisfied']}, base={r['base_rate']:.4f})",
              flush=True)

    # By direction
    print(f"\n  Mean sensitivity by direction:", flush=True)
    by_dir = defaultdict(list)
    for r in results:
        if r['reliable']:
            by_dir[r['direction']].append(r['sensitivity'])
    for d in sorted(by_dir.keys()):
        vals = by_dir[d]
        print(f"    {d:10s}: mean={np.mean(vals):.5f} "
              f"(n={len(vals)} reliable rules)", flush=True)

    # By length
    print(f"\n  Mean sensitivity by length:", flush=True)
    by_len = defaultdict(list)
    for r in results:
        if r['reliable']:
            by_len[r['length']].append(r['sensitivity'])
    for l in sorted(by_len.keys()):
        vals = by_len[l]
        print(f"    length {l}: mean={np.mean(vals):.5f} "
              f"(n={len(vals)} reliable rules)", flush=True)

    # Sensitivity concentration (Lorenz-like)
    reliable_sens = sorted([r['sensitivity'] for r in results if r['reliable']],
                           reverse=True)
    if reliable_sens:
        total = sum(reliable_sens)
        cumsum = np.cumsum(reliable_sens) / total if total > 0 else np.zeros(len(reliable_sens))
        for pct in [10, 25, 50]:
            n = max(1, len(reliable_sens) * pct // 100)
            print(f"    Top {pct}% of rules account for "
                  f"{cumsum[n-1]*100:.1f}% of total sensitivity", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute flanking rule sensitivity for OthelloGPT")
    parser.add_argument("--n-games", type=int, default=50000)
    parser.add_argument("--ckpt", type=str, default="./ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--games-dir", type=str, default="data/othello_synthetic")
    parser.add_argument("--output", type=str, default="behavioral_data/sensitivity.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    t0 = time.time()

    print("Loading model...", flush=True)
    model, dataset, device = load_model(args.ckpt)
    print(f"  Device: {device}", flush=True)

    print(f"Loading {args.n_games} games...", flush=True)
    games = load_shard_games(0, args.n_games, args.games_dir)
    print(f"  Loaded {len(games)} games", flush=True)

    print("Computing board states and model probs...", flush=True)
    board_states, probs, move_nums, player_colors = \
        collect_board_states_and_probs(games, model, dataset, device,
                                       args.batch_size)
    print(f"  {len(board_states)} positions total", flush=True)

    print("Enumerating flanking patterns...", flush=True)
    patterns = enumerate_flanking_patterns()
    print(f"  {len(patterns)} patterns", flush=True)

    print("Computing sensitivity...", flush=True)
    results = compute_sensitivity(board_states, probs, move_nums,
                                  player_colors, patterns, args.min_samples)

    print_summary(results)

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    # Strip excess_by_move for compact output (save separately if needed)
    compact = []
    for r in results:
        rc = {k: v for k, v in r.items() if k != 'excess_by_move'}
        compact.append(rc)

    with open(args.output, 'w') as f:
        json.dump({
            'n_games': args.n_games,
            'n_positions': len(board_states),
            'n_patterns': len(patterns),
            'rules': compact,
        }, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nSaved to {args.output} ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
