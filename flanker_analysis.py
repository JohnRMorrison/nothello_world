"""Flanker Analysis: Does OthelloGPT check for terminal pieces?

For each flanking rule with 3+ opponent pieces, find positions where the
opponent chain is complete and compare the model's probability for the target
cell when:
  - "satisfied": terminal cell has friendly piece (flanker present)
  - "unsatisfied": terminal cell does NOT have friendly piece (flanker absent)

If the model checks for the flanker: satisfied_prob >> unsatisfied_prob
If the model uses partial heuristic: satisfied_prob ≈ unsatisfied_prob

Usage:
    python flanker_analysis.py --n-games 100000 --output experiments/flanker_analysis.json
"""

import argparse
import json
import os
import sys
import time
import pickle
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from behavioral_utils import (
    load_model, build_vocab_to_pos_map, extract_probs_60d,
    N_MOVES, VALID_MOVES, MOVE_TO_IDX, IDX_TO_MOVE, POS_START, POS_END,
    load_shard_games
)
from hand_crafted_flanking import enumerate_flanking_patterns
from data.othello import OthelloBoardState


def run_inference(games, model, dataset, device, batch_size=64):
    """Run batched inference on games, return (n_games, block_size, 60) probs."""
    vocab_to_pos, pos_to_vocab = build_vocab_to_pos_map(dataset)
    block_size = dataset.block_size
    n_games = len(games)

    stoi_arr = np.zeros(64, dtype=np.int64)
    for pos in VALID_MOVES:
        stoi_arr[pos] = dataset.stoi[pos]

    all_tokens = np.zeros((n_games, block_size), dtype=np.int64)
    for i, game in enumerate(games):
        seq_len = min(len(game), block_size)
        all_tokens[i, :seq_len] = stoi_arr[np.array(game[:seq_len])]

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

    return all_probs


def analyze_flanker(games, all_probs, patterns, max_per_rule=10000):
    """For each long-chain pattern, find satisfied and unsatisfied positions.

    Returns dict with per-rule and per-length aggregates.
    """
    long_patterns = [(i, p) for i, p in enumerate(patterns) if p['length'] >= 3]
    print(f"Analyzing {len(long_patterns)} patterns with length >= 3", flush=True)

    # Per-rule accumulators
    rule_results = {}
    for pid, p in long_patterns:
        rule_results[pid] = {
            'target': p['target'],
            'opponents': p['opponents'],
            'terminal': p['terminal'],
            'direction': p['direction'],
            'length': p['length'],
            'satisfied_probs': [],
            'unsatisfied_probs': [],
            'satisfied_moves': [],
            'unsatisfied_moves': [],
        }

    # Baseline accumulators: average prob per (target_cell, move_number)
    baseline_counts = defaultdict(lambda: [0, 0.0])  # (count, sum_prob)

    n_games = len(games)
    block_size = all_probs.shape[1]

    for gi, game in enumerate(games):
        if (gi + 1) % 10000 == 0:
            print(f"    Replayed {gi+1}/{n_games} games", flush=True)

        board = OthelloBoardState()
        for t in range(len(game)):
            if t < POS_START or t >= min(POS_END, block_size + 1):
                board.umpire(game[t])
                continue

            state = board.state.flatten()  # (64,) with {-1, 0, 1}
            player_color = board.next_hand_color  # 1=black, -1=white
            opponent_color = -player_color

            # Model probs at position t are at logits index t-1
            probs_60 = all_probs[gi, t - 1, :]  # (60,)

            # Update baseline for all target cells
            for cell_idx in range(N_MOVES):
                cell = VALID_MOVES[cell_idx]
                key = (cell, t)
                baseline_counts[key][0] += 1
                baseline_counts[key][1] += probs_60[cell_idx]

            # Check each long-chain pattern
            for pid, p in long_patterns:
                rr = rule_results[pid]

                # Check if we already have enough for this rule
                if (len(rr['satisfied_probs']) >= max_per_rule and
                        len(rr['unsatisfied_probs']) >= max_per_rule):
                    continue

                target = p['target']
                opponents = p['opponents']
                terminal = p['terminal']

                # Target must be empty
                r_t, c_t = target // 8, target % 8
                if state[target] != 0:
                    continue

                # All opponents must have opponent's color
                all_opp = True
                for opp in opponents:
                    if state[opp] != opponent_color:
                        all_opp = False
                        break
                if not all_opp:
                    continue

                # Get model probability for target cell
                if target not in MOVE_TO_IDX:
                    continue
                target_prob = float(probs_60[MOVE_TO_IDX[target]])

                # Check terminal
                if state[terminal] == player_color:
                    # Flanker present → satisfied
                    if len(rr['satisfied_probs']) < max_per_rule:
                        rr['satisfied_probs'].append(target_prob)
                        rr['satisfied_moves'].append(int(t))
                else:
                    # Flanker absent → unsatisfied
                    if len(rr['unsatisfied_probs']) < max_per_rule:
                        rr['unsatisfied_probs'].append(target_prob)
                        rr['unsatisfied_moves'].append(int(t))

            board.umpire(game[t])

    # Compute baseline averages
    baseline_avg = {}
    for key, (count, total) in baseline_counts.items():
        if count > 0:
            baseline_avg[key] = total / count

    return rule_results, baseline_avg


def summarize_results(rule_results, baseline_avg, patterns):
    """Compute per-rule and per-length summaries."""

    per_rule = []
    by_length = defaultdict(lambda: {
        'satisfied_probs': [], 'unsatisfied_probs': [],
        'baseline_probs': [], 'n_rules': 0,
        'diffs': [],
    })

    for pid, rr in rule_results.items():
        n_sat = len(rr['satisfied_probs'])
        n_unsat = len(rr['unsatisfied_probs'])

        if n_sat < 10 or n_unsat < 10:
            continue  # too few samples

        mean_sat = np.mean(rr['satisfied_probs'])
        mean_unsat = np.mean(rr['unsatisfied_probs'])
        diff = mean_sat - mean_unsat

        # Compute baseline for this rule's positions
        target = rr['target']
        sat_baselines = [baseline_avg.get((target, m), 0)
                         for m in rr['satisfied_moves']]
        unsat_baselines = [baseline_avg.get((target, m), 0)
                           for m in rr['unsatisfied_moves']]
        mean_baseline = np.mean(sat_baselines + unsat_baselines) if sat_baselines or unsat_baselines else 0

        length = rr['length']
        target_name = chr(65 + target // 8) + str(target % 8 + 1)
        dir_name = str(rr['direction'])

        rule_entry = {
            'rule_id': pid,
            'target': target,
            'target_name': target_name,
            'chain_length': length,
            'direction': dir_name,
            'n_satisfied': n_sat,
            'n_unsatisfied': n_unsat,
            'mean_prob_satisfied': float(mean_sat),
            'mean_prob_unsatisfied': float(mean_unsat),
            'prob_difference': float(diff),
            'mean_baseline_prob': float(mean_baseline),
        }
        per_rule.append(rule_entry)

        # Aggregate by length
        bl = by_length[length]
        bl['satisfied_probs'].append(mean_sat)
        bl['unsatisfied_probs'].append(mean_unsat)
        bl['baseline_probs'].append(mean_baseline)
        bl['diffs'].append(diff)
        bl['n_rules'] += 1

    # Compute per-length summaries
    per_length = {}
    for length in sorted(by_length.keys()):
        bl = by_length[length]
        per_length[length] = {
            'n_rules': bl['n_rules'],
            'mean_prob_satisfied': float(np.mean(bl['satisfied_probs'])),
            'mean_prob_unsatisfied': float(np.mean(bl['unsatisfied_probs'])),
            'mean_baseline': float(np.mean(bl['baseline_probs'])),
            'mean_difference': float(np.mean(bl['diffs'])),
            'std_difference': float(np.std(bl['diffs'])),
            'median_difference': float(np.median(bl['diffs'])),
        }

    return per_rule, per_length


def plot_results(per_length, output_path):
    """Plot mean probability difference by chain length."""
    import matplotlib.pyplot as plt

    lengths = sorted(per_length.keys())
    diffs = [per_length[l]['mean_difference'] for l in lengths]
    stds = [per_length[l]['std_difference'] for l in lengths]
    sat_probs = [per_length[l]['mean_prob_satisfied'] for l in lengths]
    unsat_probs = [per_length[l]['mean_prob_unsatisfied'] for l in lengths]
    baselines = [per_length[l]['mean_baseline'] for l in lengths]
    n_rules = [per_length[l]['n_rules'] for l in lengths]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Probability difference by chain length
    ax = axes[0]
    ax.bar(lengths, diffs, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Chain Length (# opponent pieces)', fontsize=14)
    ax.set_ylabel('Prob Difference\n(satisfied - unsatisfied)', fontsize=14)
    ax.set_title('Does the Flanker Matter?', fontsize=16)
    ax.set_xticks(lengths)
    for i, l in enumerate(lengths):
        ax.text(l, diffs[i] + stds[i] + 0.002, f'n={n_rules[i]}',
                ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Absolute probabilities
    ax = axes[1]
    x = np.array(lengths)
    width = 0.25
    ax.bar(x - width, sat_probs, width, label='Satisfied (flanker present)',
           color='green', alpha=0.7)
    ax.bar(x, unsat_probs, width, label='Unsatisfied (no flanker)',
           color='red', alpha=0.7)
    ax.bar(x + width, baselines, width, label='Baseline (all positions)',
           color='gray', alpha=0.5)
    ax.set_xlabel('Chain Length (# opponent pieces)', fontsize=14)
    ax.set_ylabel('Mean Model Probability', fontsize=14)
    ax.set_title('Model Probability by Condition', fontsize=16)
    ax.set_xticks(lengths)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Flanker Analysis")
    parser.add_argument("--n-games", type=int, default=100000)
    parser.add_argument("--max-per-rule", type=int, default=10000)
    parser.add_argument("--output", type=str,
                        default="experiments/flanker_analysis.json")
    parser.add_argument("--plot", type=str,
                        default="experiments/flanker_analysis.png")
    parser.add_argument("--ckpt", type=str,
                        default="ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    t0 = time.time()

    # Load model
    print("Loading model...", flush=True)
    model, dataset, device = load_model(args.ckpt)

    # Load games
    print(f"Loading {args.n_games} games...", flush=True)
    games = []
    shard = 0
    while len(games) < args.n_games:
        shard_games = load_shard_games(shard_id=shard, games_per_shard=100000,
                                       pickle_dir="data/othello_synthetic")
        games.extend(shard_games)
        shard += 1
        if shard > 20:
            break
    games = games[:args.n_games]
    print(f"  Loaded {len(games)} games", flush=True)

    # Run inference
    print("Running inference...", flush=True)
    all_probs = run_inference(games, model, dataset, device)

    # Get patterns
    patterns = enumerate_flanking_patterns()

    # Analyze
    print("Analyzing flanker patterns...", flush=True)
    rule_results, baseline_avg = analyze_flanker(
        games, all_probs, patterns, max_per_rule=args.max_per_rule
    )

    # Summarize
    per_rule, per_length = summarize_results(rule_results, baseline_avg, patterns)

    # Print summary
    print(f"\n{'='*60}")
    print("FLANKER ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"\nPer chain length:")
    print(f"{'Length':>6} {'N_rules':>8} {'Sat_prob':>9} {'Unsat_prob':>10} "
          f"{'Baseline':>9} {'Diff':>8} {'Std':>8}")
    for length in sorted(per_length.keys()):
        pl = per_length[length]
        print(f"{length:>6} {pl['n_rules']:>8} {pl['mean_prob_satisfied']:>9.4f} "
              f"{pl['mean_prob_unsatisfied']:>10.4f} {pl['mean_baseline']:>9.4f} "
              f"{pl['mean_difference']:>8.4f} {pl['std_difference']:>8.4f}")

    print(f"\nTop 10 rules by probability difference:")
    sorted_rules = sorted(per_rule, key=lambda x: x['prob_difference'], reverse=True)
    for r in sorted_rules[:10]:
        print(f"  {r['target_name']} {r['direction']} len={r['chain_length']}: "
              f"sat={r['mean_prob_satisfied']:.4f} unsat={r['mean_prob_unsatisfied']:.4f} "
              f"diff={r['prob_difference']:.4f} (n={r['n_satisfied']}/{r['n_unsatisfied']})")

    print(f"\nBottom 10 rules by probability difference:")
    for r in sorted_rules[-10:]:
        print(f"  {r['target_name']} {r['direction']} len={r['chain_length']}: "
              f"sat={r['mean_prob_satisfied']:.4f} unsat={r['mean_prob_unsatisfied']:.4f} "
              f"diff={r['prob_difference']:.4f} (n={r['n_satisfied']}/{r['n_unsatisfied']})")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        'n_games': len(games),
        'max_per_rule': args.max_per_rule,
        'per_rule': per_rule,
        'per_length': {str(k): v for k, v in per_length.items()},
        'elapsed_seconds': time.time() - t0,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output}")

    # Plot
    try:
        plot_results(per_length, args.plot)
    except ImportError:
        print("matplotlib not available, skipping plot")

    print(f"\nTotal elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
