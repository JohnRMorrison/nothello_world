"""Suppression vs Promotion: Paired Contrasts

Tests whether the model is better at suppression (learning to stop predicting
a cell) than promotion (learning to start predicting a cell).

3 paired contrasts × 2 rule counts = 12 conditions.
Each pair uses the SAME rules — only the direction differs.

Pairs:
  1. drop_opponent (permissive, IL) vs add_opponent (restrictive, LI)
  2. eliminate (restrictive, LI) vs duplicate (permissive, IL)
  3. relax_color (permissive, IL) vs tighten_color (restrictive, LI)

Usage:
    python suppression_vs_promotion.py --condition-id 0 --output-dir experiments/supp_vs_prom
"""

import argparse
import json
import os
import sys
import time
import pickle
import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from hand_crafted_flanking import enumerate_flanking_patterns
from sensitivity_param_search import (
    precompute_pattern_arrays_extended,
    generate_games_extended,
    collect_three_test_sets,
    evaluate_on_test_sets,
    build_standard_lpm_test,
    prepare_lpm_test,
    CENTER_CELLS,
)
from behavioral_utils import load_model, N_MOVES, VALID_MOVES
from mingpt.dataset import CharDataset
from finetune_corruption import evaluate, build_legal_mask
from torch.utils.data import DataLoader

# Condition mapping
# 0-1: 50 rules, drop_opponent / add_opponent
# 2-3: 50 rules, eliminate / duplicate
# 4-5: 50 rules, relax_color / tighten_color
# 6-7: 100 rules, drop_opponent / add_opponent
# 8-9: 100 rules, eliminate / duplicate
# 10-11: 100 rules, relax_color / tighten_color

CONDITIONS = [
    {'n_rules': 50,  'corruption': 'drop_opponent'},
    {'n_rules': 50,  'corruption': 'add_opponent'},
    {'n_rules': 50,  'corruption': 'eliminate'},
    {'n_rules': 50,  'corruption': 'duplicate'},
    {'n_rules': 50,  'corruption': 'relax_color'},
    {'n_rules': 50,  'corruption': 'tighten_color'},
    {'n_rules': 100, 'corruption': 'drop_opponent'},
    {'n_rules': 100, 'corruption': 'add_opponent'},
    {'n_rules': 100, 'corruption': 'eliminate'},
    {'n_rules': 100, 'corruption': 'duplicate'},
    {'n_rules': 100, 'corruption': 'relax_color'},
    {'n_rules': 100, 'corruption': 'tighten_color'},
]

EVAL_SCHEDULE = [0, 5, 25, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000]


def corrupt_drop_opponent(patterns, rule_ids, rng):
    """Remove one opponent from the chain (more permissive → IL)."""
    patterns = deepcopy(patterns)
    n_modified = 0
    for rid in rule_ids:
        p = patterns[rid]
        if len(p['opponents']) >= 2:
            drop_idx = rng.randint(len(p['opponents']))
            p['opponents'] = [o for i, o in enumerate(p['opponents']) if i != drop_idx]
            p['length'] = len(p['opponents'])
            n_modified += 1
    return patterns, n_modified


def corrupt_add_opponent(patterns, rule_ids, rng):
    """Add one opponent requirement at adjacent cell (more restrictive → LI)."""
    patterns = deepcopy(patterns)
    n_modified = 0
    for rid in rule_ids:
        p = patterns[rid]
        used = {p['target'], p['terminal']} | set(p['opponents'])
        neighbors = set()
        for cell in p['opponents']:
            r, c = cell // 8, cell % 8
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        nc_cell = nr * 8 + nc
                        if nc_cell not in used and nc_cell not in CENTER_CELLS:
                            neighbors.add(nc_cell)
        if neighbors:
            new_opp = rng.choice(list(neighbors))
            p['opponents'].append(new_opp)
            p['length'] = len(p['opponents'])
            n_modified += 1
    return patterns, n_modified


def corrupt_eliminate(patterns, rule_ids, rng):
    """Replace pattern so it never fires (more restrictive → LI).
    Set all opponent cells to impossible positions (-1)."""
    patterns = deepcopy(patterns)
    n_modified = 0
    for rid in rule_ids:
        p = patterns[rid]
        # Make terminal require a cell that's the target itself
        # (target is empty, terminal requires player piece → can't both be true)
        p['terminal'] = p['target']
        n_modified += 1
    return patterns, n_modified


def corrupt_duplicate(patterns, rule_ids, rng):
    """Replace pattern with an existing pattern from a different cell (more permissive → IL).
    The target cell gains a new way to be legal."""
    patterns = deepcopy(patterns)
    n_modified = 0
    all_pattern_indices = list(range(len(patterns)))
    for rid in rule_ids:
        p = patterns[rid]
        target = p['target']
        # Find a pattern from a different cell
        candidates = [i for i in all_pattern_indices
                      if patterns[i]['target'] != target and i not in rule_ids]
        if candidates:
            donor = rng.choice(candidates)
            dp = patterns[donor]
            # Keep the target but use donor's opponents and terminal
            p['opponents'] = list(dp['opponents'])
            p['terminal'] = dp['terminal']
            p['direction'] = dp['direction']
            p['length'] = dp['length']
            n_modified += 1
    return patterns, n_modified


def corrupt_relax_color(patterns, rule_ids, rng):
    """One opponent cell now accepts ANY piece (more permissive → IL).
    Implemented by removing that cell from the opponents list — the check
    is no longer required."""
    patterns = deepcopy(patterns)
    n_modified = 0
    for rid in rule_ids:
        p = patterns[rid]
        if len(p['opponents']) >= 2:
            relax_idx = rng.randint(len(p['opponents']))
            p['opponents'] = [o for i, o in enumerate(p['opponents']) if i != relax_idx]
            p['length'] = len(p['opponents'])
            n_modified += 1
    return patterns, n_modified


def corrupt_tighten_color(patterns, rule_ids, rng):
    """One previously unchecked adjacent cell must now be opponent (more restrictive → LI)."""
    patterns = deepcopy(patterns)
    n_modified = 0
    for rid in rule_ids:
        p = patterns[rid]
        used = {p['target'], p['terminal']} | set(p['opponents'])
        # Find adjacent cells to target that aren't already checked
        r, c = p['target'] // 8, p['target'] % 8
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    nc_cell = nr * 8 + nc
                    if nc_cell not in used and nc_cell not in CENTER_CELLS:
                        neighbors.append(nc_cell)
        if neighbors:
            new_opp = rng.choice(neighbors)
            p['opponents'].append(new_opp)
            p['length'] = len(p['opponents'])
            n_modified += 1
    return patterns, n_modified


def apply_corruption(corruption_type, patterns, rule_ids, rng):
    """Apply the specified corruption."""
    if corruption_type == 'drop_opponent':
        return corrupt_drop_opponent(patterns, rule_ids, rng)
    elif corruption_type == 'add_opponent':
        return corrupt_add_opponent(patterns, rule_ids, rng)
    elif corruption_type == 'eliminate':
        return corrupt_eliminate(patterns, rule_ids, rng)
    elif corruption_type == 'duplicate':
        return corrupt_duplicate(patterns, rule_ids, rng)
    elif corruption_type == 'relax_color':
        return corrupt_relax_color(patterns, rule_ids, rng)
    elif corruption_type == 'tighten_color':
        return corrupt_tighten_color(patterns, rule_ids, rng)
    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")


def determine_skip_sets(corruption_type):
    """Determine which test sets to skip based on corruption direction."""
    permissive = {'drop_opponent', 'duplicate', 'relax_color'}
    restrictive = {'add_opponent', 'eliminate', 'tighten_color'}
    if corruption_type in permissive:
        return {'LI'}  # no LI positions for permissive corruptions
    elif corruption_type in restrictive:
        return {'IL'}  # no IL positions for restrictive corruptions
    return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="experiments/supp_vs_prom")
    parser.add_argument("--n-train", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ckpt", type=str, default="ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cid = args.condition_id
    cfg = CONDITIONS[cid]
    n_rules = cfg['n_rules']
    corruption_type = cfg['corruption']

    print(f"Condition {cid}: {n_rules} rules, {corruption_type}")
    print(f"Started at: {time.strftime('%c')}", flush=True)

    rng = np.random.RandomState(args.seed)

    # Load model
    model, dataset, device = load_model(args.ckpt)
    print(f"Device: {device}", flush=True)

    # Select random rules (same seed for paired conditions)
    # Pairs share the same n_rules, so use n_rules as part of seed
    pair_seed = args.seed + n_rules
    pair_rng = np.random.RandomState(pair_seed)
    rule_ids = pair_rng.choice(960, n_rules, replace=False).tolist()
    print(f"Selected {len(rule_ids)} rules", flush=True)

    # Apply corruption
    base_patterns = enumerate_flanking_patterns()
    corrupted_patterns, n_modified = apply_corruption(
        corruption_type, base_patterns, rule_ids, rng)
    print(f"Modified {n_modified} rules (type={corruption_type})", flush=True)

    # Generate or load games
    games_dir = os.path.join(args.output_dir, "games", f"cond_{cid:03d}")
    os.makedirs(games_dir, exist_ok=True)
    games_path = os.path.join(games_dir, "train_games.pickle")
    legal_path = os.path.join(games_dir, "train_legal.pickle")

    if os.path.exists(games_path):
        print(f"Loading existing games from {games_dir}...", flush=True)
        with open(games_path, 'rb') as f:
            train_games = pickle.load(f)
        with open(legal_path, 'rb') as f:
            train_legal = pickle.load(f)
        print(f"  Loaded {len(train_games)} existing games", flush=True)
    else:
        print(f"Generating {args.n_train} games...", flush=True)
        corrupted_arrays = precompute_pattern_arrays_extended(corrupted_patterns)
        train_games, train_legal = generate_games_extended(
            corrupted_arrays, n_games=args.n_train, seed=args.seed)
        with open(games_path, 'wb') as f:
            pickle.dump(train_games, f)
        with open(legal_path, 'wb') as f:
            pickle.dump(train_legal, f)
        print(f"  Saved {len(train_games)} games", flush=True)

    train_games = [g for g in train_games if len(g) >= 5]
    train_legal = train_legal[:len(train_games)]
    n_train = int(len(train_games) * 0.95)
    print(f"  {len(train_games)} games after filtering", flush=True)

    # Collect test sets
    print("Collecting test positions...", flush=True)
    skip_sets = determine_skip_sets(corruption_type)
    corrupted_arrays = precompute_pattern_arrays_extended(corrupted_patterns)
    std_arrays = precompute_pattern_arrays_extended(base_patterns)
    test_sets = collect_three_test_sets(
        corrupted_arrays, std_arrays, n_per_set=5000,
        skip_sets=skip_sets, n_games_max=200000, seed=args.seed + 1)

    counts = {k: len(test_sets.get(k, [])) for k in ['LL', 'IL', 'LI']}
    print(f"  Test sets: LL={counts['LL']}, IL={counts['IL']}, "
          f"LI={counts['LI']}", flush=True)

    # Build LPM test sets
    print("Building standard LPM test set...", flush=True)
    std_games_test, std_legal_test = build_standard_lpm_test(
        n_games=10000, seed=args.seed+2)
    ref_dataset = CharDataset(train_games[:100])
    std_loader, std_mask = prepare_lpm_test(
        std_games_test, std_legal_test, ref_dataset, device)

    print("Building corrupted LPM test set...", flush=True)
    cor_sample = rng.choice(len(train_games), min(10000, len(train_games)), replace=False)
    cor_games = [train_games[i] for i in cor_sample]
    cor_legal = [train_legal[i] for i in cor_sample]
    cor_loader, cor_mask = prepare_lpm_test(
        cor_games, cor_legal, ref_dataset, device)

    # Train
    print("Training...", flush=True)
    train_dataset = CharDataset(train_games[:n_train])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=0, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    total_steps = len(train_loader)
    eval_at = set(EVAL_SCHEDULE)
    eval_at.add(total_steps - 1)

    results = {
        'eval_steps': [], 'LL_prob': [], 'LL_acc': [],
        'IL_prob': [], 'IL_acc': [], 'LI_prob': [], 'LI_acc': [],
        'std_lpm': [], 'cor_lpm': [],
    }
    t0 = time.time()

    model.train()
    for step, (x, y) in enumerate(train_loader):
        if step in eval_at:
            model.eval()
            metrics = evaluate_on_test_sets(model, test_sets, train_dataset, device)
            std_l, std_a, std_r, std_lpm = evaluate(model, std_loader, device, std_mask)
            cor_l, cor_a, cor_r, cor_lpm = evaluate(model, cor_loader, device, cor_mask)

            results['eval_steps'].append(step)
            for k in ['LL', 'IL', 'LI']:
                results[f'{k}_prob'].append(metrics.get(f'{k}_prob', 0.0))
                results[f'{k}_acc'].append(metrics.get(f'{k}_acc', 0.0))
            results['std_lpm'].append(float(std_lpm))
            results['cor_lpm'].append(float(cor_lpm))

            print(f"  Step {step}: LL={metrics.get('LL_prob',0):.4f} "
                  f"IL={metrics.get('IL_prob',0):.4f} "
                  f"LI={metrics.get('LI_prob',0):.4f} "
                  f"std={std_lpm:.4f} cor={cor_lpm:.4f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
            model.train()

        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final eval
    model.eval()
    metrics = evaluate_on_test_sets(model, test_sets, train_dataset, device)
    std_l, std_a, std_r, std_lpm = evaluate(model, std_loader, device, std_mask)
    cor_l, cor_a, cor_r, cor_lpm = evaluate(model, cor_loader, device, cor_mask)
    results['eval_steps'].append(total_steps)
    for k in ['LL', 'IL', 'LI']:
        results[f'{k}_prob'].append(metrics.get(f'{k}_prob', 0.0))
        results[f'{k}_acc'].append(metrics.get(f'{k}_acc', 0.0))
    results['std_lpm'].append(float(std_lpm))
    results['cor_lpm'].append(float(cor_lpm))

    # Save
    output = {
        'condition_id': cid,
        'corruption_type': corruption_type,
        'n_rules': n_rules,
        'n_rules_modified': n_modified,
        'rule_ids': rule_ids,
        'n_train_games': len(train_games),
        'total_steps': total_steps,
        'elapsed_seconds': time.time() - t0,
        **results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"cond_{cid:03d}.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved {out_path}", flush=True)
    print(f"Finished at: {time.strftime('%c')}")


if __name__ == "__main__":
    main()
