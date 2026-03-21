"""Generate games with sensitivity-ranked corruption.

3x3 factorial design:
  Corruption types: full, terminal_only, drop_opponent
  Sensitivity ranks: high, low, random

All conditions matched by legal-move divergence via binary search on
number of rules corrupted.

Usage:
    python generate_sensitivity_games.py \
        --corruption-type full --sensitivity-rank high \
        --target-divergence 0.20 \
        --sensitivity-file behavioral_data/sensitivity.json \
        --num-games 2000000 --output-dir behavioral_data/games/full_high/
"""

import argparse
import json
import os
import pickle
import sys
import time
import numpy as np
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hand_crafted_flanking import enumerate_flanking_patterns
from generate_rule_games import (
    precompute_pattern_arrays, generate_games, evaluate_rules_vec
)
from data.othello import OthelloBoardState


# =============================================================================
# Corruption functions
# =============================================================================

def corrupt_full(patterns, rule_ids, rng):
    """Replace opponents + terminal with random cells."""
    patterns = deepcopy(patterns)
    for rid in rule_ids:
        p = patterns[rid]
        p['opponents'] = [int(rng.randint(0, 64)) for _ in p['opponents']]
        p['terminal'] = int(rng.randint(0, 64))
    return patterns


def corrupt_terminal_only(patterns, rule_ids, rng):
    """Keep opponents, replace terminal with random cell."""
    patterns = deepcopy(patterns)
    for rid in rule_ids:
        p = patterns[rid]
        p['terminal'] = int(rng.randint(0, 64))
    return patterns


def corrupt_drop_opponent(patterns, rule_ids, rng):
    """Remove one opponent from the chain (shorten by 1).
    Only affects rules with length >= 2. Rules with length 1 are skipped.
    """
    patterns = deepcopy(patterns)
    for rid in rule_ids:
        p = patterns[rid]
        if len(p['opponents']) >= 2:
            drop_idx = int(rng.randint(0, len(p['opponents'])))
            p['opponents'] = [o for i, o in enumerate(p['opponents'])
                              if i != drop_idx]
            p['length'] = len(p['opponents'])
    return patterns


CORRUPTION_FNS = {
    'full': corrupt_full,
    'terminal_only': corrupt_terminal_only,
    'drop_opponent': corrupt_drop_opponent,
}


# =============================================================================
# Rule selection by sensitivity rank
# =============================================================================

def select_rules(sensitivity_data, rank, n_rules, rng,
                  frequency_matched=False):
    """Select rule IDs based on sensitivity ranking.

    rank: 'high' (top of ranking), 'low' (bottom), 'random'
    n_rules: how many to select
    frequency_matched: if True, select from high/low pools with matched
        frequency distributions (only for 'high' and 'low' ranks)
    """
    rules = sensitivity_data['rules']
    sorted_rules = sorted(rules, key=lambda r: r['sensitivity'], reverse=True)
    all_ids = [r['rule_id'] for r in sorted_rules]

    if frequency_matched and rank in ('high', 'low'):
        return _select_frequency_matched(sorted_rules, rank, n_rules, rng)

    n_rules = min(n_rules, len(all_ids))

    if rank == 'high':
        return all_ids[:n_rules]
    elif rank == 'low':
        return all_ids[-n_rules:]
    elif rank == 'random':
        return list(rng.choice(all_ids, n_rules, replace=False))
    else:
        raise ValueError(f"Unknown rank: {rank}")


def _select_frequency_matched(sorted_rules, rank, n_rules, rng):
    """Select rules with matched frequency from high/low sensitivity pools.

    Bins rules by frequency, samples equal numbers from each bin to ensure
    the high and low selections have similar frequency distributions.
    """
    n_pool = len(sorted_rules) // 3
    high_pool = sorted_rules[:n_pool]
    low_pool = sorted_rules[-n_pool:]

    pool = high_pool if rank == 'high' else low_pool
    other = low_pool if rank == 'high' else high_pool

    # Find frequency overlap
    pool_freqs = [r['n_satisfied'] for r in pool]
    other_freqs = [r['n_satisfied'] for r in other]
    overlap_min = max(min(pool_freqs), min(other_freqs))
    overlap_max = min(max(pool_freqs), max(other_freqs))

    # Bin by log-frequency and sample
    freq_bins = np.logspace(np.log10(max(overlap_min, 1)),
                            np.log10(overlap_max), 11)
    selected = []
    per_bin = max(1, n_rules // (len(freq_bins) - 1) + 1)

    for i in range(len(freq_bins) - 1):
        lo, hi = freq_bins[i], freq_bins[i + 1]
        candidates = [r for r in pool if lo <= r['n_satisfied'] < hi]
        other_count = sum(1 for r in other if lo <= r['n_satisfied'] < hi)
        n_take = min(len(candidates), other_count, per_bin)
        if n_take > 0:
            chosen = rng.choice(candidates, n_take, replace=False)
            selected.extend(chosen.tolist())

    # Trim to target count
    if len(selected) > n_rules:
        idx = rng.choice(len(selected), n_rules, replace=False)
        selected = [selected[i] for i in idx]

    return [r['rule_id'] for r in selected]


# =============================================================================
# Divergence computation
# =============================================================================

def compute_divergence_sample(corrupted_patterns, n_sample=1000, seed=0):
    """Compute legal-move Jaccard divergence from standard Othello.

    Generates n_sample games under both standard and corrupted rules,
    compares legal move sets at each position.
    """
    rng = np.random.RandomState(seed)

    # Standard patterns
    std_patterns = enumerate_flanking_patterns()
    std_targets, std_terminals, std_opp, std_mask = \
        precompute_pattern_arrays(std_patterns)

    # Corrupted patterns
    cor_targets, cor_terminals, cor_opp, cor_mask = \
        precompute_pattern_arrays(corrupted_patterns)

    # Generate games under standard rules
    std_games = generate_games(std_targets, std_terminals, std_opp, std_mask,
                               n_sample, rng, chunk_size=n_sample,
                               save_legal=True)
    std_games, std_legal = std_games

    total_jaccard = 0.0
    n_positions = 0

    for gi in range(n_sample):
        game = std_games[gi]
        # Replay under both rule sets
        flat_std = np.zeros(64, dtype=np.int8)
        flat_cor = np.zeros(64, dtype=np.int8)
        # Initialize center
        for pos, val in [(27, 1), (28, -1), (35, -1), (36, 1)]:
            flat_std[pos] = val
            flat_cor[pos] = val

        is_black = True
        for t, move in enumerate(game):
            if t >= 4:  # skip first few moves for meaningful comparison
                # Get legal moves under both rule sets
                std_legal_set = set()
                cor_legal_set = set()

                std_result = evaluate_rules_vec(flat_std, is_black,
                                                std_targets, std_terminals,
                                                std_opp, std_mask)
                cor_result = evaluate_rules_vec(flat_std, is_black,
                                                cor_targets, cor_terminals,
                                                cor_opp, cor_mask)

                if std_result is not None:
                    std_legal_set = set(std_result.tolist())
                if cor_result is not None:
                    cor_legal_set = set(cor_result.tolist())

                if std_legal_set or cor_legal_set:
                    union = len(std_legal_set | cor_legal_set)
                    inter = len(std_legal_set & cor_legal_set)
                    if union > 0:
                        jaccard = 1.0 - inter / union
                        total_jaccard += jaccard
                        n_positions += 1

            # Apply move to board (simplified — just place piece)
            color = 1 if is_black else -1
            flat_std[move] = color
            flat_cor[move] = color
            is_black = not is_black

    return total_jaccard / max(n_positions, 1)


# =============================================================================
# Divergence-matched rule selection
# =============================================================================

def find_n_rules_for_divergence(sensitivity_data, rank, corruption_type,
                                target_div, tolerance=0.02, seed=42):
    """Binary search for number of rules to corrupt to reach target divergence."""
    rng = np.random.RandomState(seed)
    base_patterns = enumerate_flanking_patterns()
    corrupt_fn = CORRUPTION_FNS[corruption_type]

    # For drop_opponent, only rules with length >= 2 are affected
    if corruption_type == 'drop_opponent':
        eligible = [r for r in sensitivity_data['rules'] if r['length'] >= 2]
        max_rules = len(eligible)
    else:
        max_rules = len(sensitivity_data['rules'])

    lo, hi = 1, max_rules
    best_n, best_div = lo, 0.0

    print(f"  Binary search for divergence {target_div:.2f} "
          f"({corruption_type}, {rank})...", flush=True)

    for iteration in range(20):
        mid = (lo + hi) // 2
        if mid == best_n and iteration > 0:
            break

        rule_ids = select_rules(sensitivity_data, rank, mid, rng)
        corrupted = corrupt_fn(base_patterns, rule_ids, rng)
        div = compute_divergence_sample(corrupted, n_sample=500, seed=seed)

        print(f"    n={mid}: divergence={div:.4f}", flush=True)

        if abs(div - target_div) < tolerance:
            return mid, div
        elif div < target_div:
            lo = mid + 1
        else:
            hi = mid - 1

        if abs(div - target_div) < abs(best_div - target_div):
            best_n, best_div = mid, div

    return best_n, best_div


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate sensitivity-ranked corruption games")
    parser.add_argument("--corruption-type", type=str, required=True,
                        choices=['full', 'terminal_only', 'drop_opponent'])
    parser.add_argument("--sensitivity-rank", type=str, required=True,
                        choices=['high', 'low', 'random'])
    parser.add_argument("--target-divergence", type=float, default=0.20)
    parser.add_argument("--fixed-count", type=int, default=None,
                        help="Corrupt exactly this many rules (skip divergence matching)")
    parser.add_argument("--frequency-matched", action="store_true",
                        help="Match frequency distributions between high/low selections")
    parser.add_argument("--sensitivity-file", type=str,
                        default="behavioral_data/sensitivity.json")
    parser.add_argument("--num-games", type=int, default=2000000)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()

    t0 = time.time()
    rng = np.random.RandomState(args.seed)

    print(f"Condition: {args.corruption_type} × {args.sensitivity_rank}",
          flush=True)

    # Load sensitivity scores
    with open(args.sensitivity_file) as f:
        sensitivity_data = json.load(f)
    print(f"Loaded {len(sensitivity_data['rules'])} rule sensitivities",
          flush=True)

    if args.fixed_count is not None:
        # Fixed count mode: corrupt exactly N rules, measure divergence
        n_rules = args.fixed_count
        print(f"Fixed count mode: corrupting {n_rules} rules", flush=True)
    else:
        # Divergence matching mode
        print(f"Target divergence: {args.target_divergence}", flush=True)
        n_rules, achieved_div = find_n_rules_for_divergence(
            sensitivity_data, args.sensitivity_rank, args.corruption_type,
            args.target_divergence, args.tolerance, args.seed)
        print(f"\nCorrupting {n_rules} rules → divergence {achieved_div:.4f}",
              flush=True)

    if args.frequency_matched:
        print("Frequency-matched mode", flush=True)

    # Select and corrupt rules
    rule_ids = select_rules(sensitivity_data, args.sensitivity_rank,
                            n_rules, rng,
                            frequency_matched=args.frequency_matched)
    base_patterns = enumerate_flanking_patterns()
    corrupt_fn = CORRUPTION_FNS[args.corruption_type]
    corrupted_patterns = corrupt_fn(base_patterns, rule_ids, rng)

    # Measure divergence if using fixed count
    if args.fixed_count is not None:
        achieved_div = compute_divergence_sample(corrupted_patterns,
                                                  n_sample=500, seed=args.seed)
        print(f"  Achieved divergence: {achieved_div:.4f}", flush=True)

    # Compute sensitivity stats for corrupted vs uncorrupted
    rule_sens = {r['rule_id']: r['sensitivity']
                 for r in sensitivity_data['rules']}
    rule_freq = {r['rule_id']: r['n_satisfied']
                 for r in sensitivity_data['rules']}
    corrupted_sens = [rule_sens[rid] for rid in rule_ids]
    uncorrupted_sens = [rule_sens[rid] for rid in range(len(base_patterns))
                        if rid not in set(rule_ids)]
    total_impact = sum(rule_sens[rid] * rule_freq[rid] for rid in rule_ids)

    print(f"  Corrupted rules: n={len(rule_ids)}, "
          f"mean_sens={np.mean(corrupted_sens):.5f}", flush=True)
    print(f"  Uncorrupted rules: n={len(uncorrupted_sens)}, "
          f"mean_sens={np.mean(uncorrupted_sens):.5f}", flush=True)
    print(f"  Total impact: {total_impact:.1f}", flush=True)

    # Generate games
    targets, terminals, opp_cells, opp_mask = \
        precompute_pattern_arrays(corrupted_patterns)

    print(f"\nGenerating {args.num_games} games...", flush=True)
    result = generate_games(targets, terminals, opp_cells, opp_mask,
                            args.num_games, rng, save_legal=True)
    games, legal_moves = result

    # Filter short games
    valid_games = []
    valid_legal = []
    for g, l in zip(games, legal_moves):
        if len(g) >= 5:
            valid_games.append(g)
            valid_legal.append(l)
    print(f"  {len(valid_games)} valid games (>= 5 moves)", flush=True)

    game_lengths = [len(g) for g in valid_games]
    print(f"  Mean length: {np.mean(game_lengths):.1f}, "
          f"min: {np.min(game_lengths)}, max: {np.max(game_lengths)}",
          flush=True)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'games.pickle'), 'wb') as f:
        pickle.dump(valid_games, f)
    with open(os.path.join(args.output_dir, 'legal_moves.pickle'), 'wb') as f:
        pickle.dump(valid_legal, f)

    metadata = {
        'corruption_type': args.corruption_type,
        'sensitivity_rank': args.sensitivity_rank,
        'matching_mode': 'freq_matched' if args.frequency_matched else ('fixed_count' if args.fixed_count else 'divergence'),
        'frequency_matched': args.frequency_matched,
        'fixed_count': args.fixed_count,
        'target_divergence': args.target_divergence if args.fixed_count is None else None,
        'achieved_divergence': float(achieved_div),
        'n_rules_corrupted': len(rule_ids),
        'n_rules_total': len(base_patterns),
        'total_impact': float(total_impact),
        'corrupted_rule_ids': rule_ids,
        'mean_sensitivity_corrupted': float(np.mean(corrupted_sens)),
        'mean_sensitivity_uncorrupted': float(np.mean(uncorrupted_sens)),
        'num_games': len(valid_games),
        'mean_game_length': float(np.mean(game_lengths)),
        'seed': args.seed,
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nSaved to {args.output_dir} ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
