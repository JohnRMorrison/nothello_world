"""Coherent vs Incoherent at Multiple Scales (No Additional Corruption)

Tests whether the model benefits from spatial coherence of corrupted rules.
The spatial transformation (shift or cross-wire) IS the corruption — no
flip_color, drop_third, etc. on top.

12 conditions: 6 n_rules × 2 (coherent/incoherent)
  0-5: coherent at [25, 75, 100, 125, 150, 175] rules
  6-11: incoherent at [25, 75, 100, 125, 150, 175] rules

Usage:
    python coherent_scale_experiment.py --condition-id 0 --output-dir experiments/coherent_scale
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
    evaluate_lpm,
    CENTER_CELLS,
)
from behavioral_utils import load_model, N_MOVES, VALID_MOVES
from mingpt.dataset import CharDataset
from torch.utils.data import DataLoader

# Condition mapping
N_RULES_LEVELS = [25, 75, 100, 125, 150, 175]
EVAL_SCHEDULE = [0, 5, 25, 50, 100, 200, 500, 1000, 2000, 5000,
                 10000, 20000, 50000, 100000]


def _shift_cell_no_wrap(cell, dr, dc):
    """Shift a cell by (dr, dc) WITHOUT wrapping. Returns None if off-board or center."""
    r, c = cell // 8, cell % 8
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= 8 or nc < 0 or nc >= 8:
        return None
    result = nr * 8 + nc
    if result in CENTER_CELLS:
        return None
    return result


def _shift_pattern_no_wrap(pattern, dr, dc):
    """Try to shift all cells in a pattern by (dr, dc). Returns shifted pattern or None."""
    new_target = _shift_cell_no_wrap(pattern['target'], dr, dc)
    if new_target is None:
        return None

    new_opponents = []
    for opp in pattern['opponents']:
        shifted = _shift_cell_no_wrap(opp, dr, dc)
        if shifted is None:
            return None
        new_opponents.append(shifted)

    new_terminal = _shift_cell_no_wrap(pattern['terminal'], dr, dc)
    if new_terminal is None:
        return None

    return {
        'target': new_target,
        'opponents': new_opponents,
        'terminal': new_terminal,
        'direction': pattern['direction'],
        'length': pattern['length'],
    }


def apply_spatial_only(group_name, patterns, rule_ids, rng):
    """Apply spatial transformation only — no corruption on top.

    coherent: shift each rule's pattern by a diagonal offset, trying
      (+1,+1), (-1,-1), (+1,-1), (-1,+1) in order. No wrapping.
      If no shift works, skip the rule.
    incoherent: cross-wire opponents/terminal between distant rules.
    """
    patterns = deepcopy(patterns)
    n_modified = 0

    if group_name == 'coherent':
        shifts = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
        for rid in rule_ids:
            p = patterns[rid]
            shifted = None
            for dr, dc in shifts:
                shifted = _shift_pattern_no_wrap(p, dr, dc)
                if shifted is not None:
                    break
            if shifted is not None:
                patterns[rid] = shifted
                n_modified += 1

    elif group_name == 'incoherent':
        ids = list(rule_ids)
        rng.shuffle(ids)
        for i in range(0, len(ids) - 1, 2):
            a, b = ids[i], ids[i + 1]
            pa, pb = patterns[a], patterns[b]
            pa['opponents'], pb['opponents'] = pb['opponents'], pa['opponents']
            pa['terminal'], pb['terminal'] = pb['terminal'], pa['terminal']
            pa['length'] = len(pa['opponents'])
            pb['length'] = len(pb['opponents'])
            n_modified += 2

    return patterns, n_modified


def train_and_evaluate_scale(model, train_games, train_legal, test_sets,
                             device, std_loader, std_mask, cor_loader, cor_mask,
                             lr=5e-5, bs=16):
    """Train for 1 epoch and evaluate at schedule points."""
    from finetune_corruption import evaluate, build_legal_mask

    train_dataset = CharDataset(train_games)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              num_workers=0, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_steps = len(train_loader)

    eval_at = set(EVAL_SCHEDULE)
    eval_at.add(total_steps - 1)

    results = {
        'eval_steps': [], 'LL_prob': [], 'LL_acc': [],
        'IL_prob': [], 'IL_acc': [], 'LI_prob': [], 'LI_acc': [],
        'std_lpm': [], 'cor_lpm': [],
    }

    def do_eval(step):
        model.eval()
        metrics = evaluate_on_test_sets(model, test_sets, train_dataset, device)
        std_loss, std_acc, std_rank, std_lpm = evaluate(model, std_loader, device, std_mask)
        cor_loss, cor_acc, cor_rank, cor_lpm = evaluate(model, cor_loader, device, cor_mask)

        results['eval_steps'].append(step)
        for k in ['LL', 'IL', 'LI']:
            results[f'{k}_prob'].append(metrics.get(f'{k}_prob', 0.0))
            results[f'{k}_acc'].append(metrics.get(f'{k}_acc', 0.0))
        results['std_lpm'].append(float(std_lpm))
        results['cor_lpm'].append(float(cor_lpm))

        print(f"  Step {step}: LL_prob={metrics.get('LL_prob',0):.4f} "
              f"IL_prob={metrics.get('IL_prob',0):.4f} "
              f"LI_prob={metrics.get('LI_prob',0):.4f} "
              f"std_lpm={std_lpm:.4f} cor_lpm={cor_lpm:.4f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
        model.train()

    model.train()
    t0 = time.time()

    for step, (x, y) in enumerate(train_loader):
        if step in eval_at:
            do_eval(step)

        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final eval
    do_eval(total_steps - 1)

    results['total_steps'] = total_steps
    results['elapsed_seconds'] = time.time() - t0
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="experiments/coherent_scale")
    parser.add_argument("--n-train", type=int, default=2000000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ckpt", type=str, default="ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cid = args.condition_id
    if cid < 6:
        group_name = 'coherent'
        n_rules = N_RULES_LEVELS[cid]
    else:
        group_name = 'incoherent'
        n_rules = N_RULES_LEVELS[cid - 6]

    print(f"Condition {cid}: {group_name} with {n_rules} rules")
    print(f"Started at: {time.strftime('%c')}", flush=True)

    rng = np.random.RandomState(args.seed)

    # Load model
    model, dataset, device = load_model(args.ckpt)
    print(f"Device: {device}", flush=True)

    # Select random rules
    rule_ids = rng.choice(960, n_rules, replace=False).tolist()
    print(f"Selected {len(rule_ids)} rules", flush=True)

    # Apply spatial transformation only
    base_patterns = enumerate_flanking_patterns()
    corrupted_patterns, n_modified = apply_spatial_only(
        group_name, base_patterns, rule_ids, rng)
    print(f"Modified {n_modified} rules (type={group_name})", flush=True)

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
        std_arrays = precompute_pattern_arrays_extended(base_patterns)
        train_games, train_legal = generate_games_extended(
            corrupted_arrays, num_games=args.n_train, rng=rng, save_legal=True)
        with open(games_path, 'wb') as f:
            pickle.dump(train_games, f)
        with open(legal_path, 'wb') as f:
            pickle.dump(train_legal, f)
        print(f"  Saved {len(train_games)} games to {games_dir}", flush=True)

    # Filter short games
    train_games = [g for g in train_games if len(g) >= 5]
    train_legal = train_legal[:len(train_games)]
    print(f"  {len(train_games)} games after filtering", flush=True)

    # Collect test sets
    print("Collecting test positions...", flush=True)
    corrupted_arrays = precompute_pattern_arrays_extended(corrupted_patterns)
    std_arrays = precompute_pattern_arrays_extended(base_patterns)
    test_sets = collect_three_test_sets(
        corrupted_arrays, std_arrays, n_per_set=5000,
        max_games=200000, rng=np.random.RandomState(args.seed + 1))
    print(f"  Test sets: LL={len(test_sets.get('LL',[]))}, "
          f"IL={len(test_sets.get('IL',[]))}, "
          f"LI={len(test_sets.get('LI',[]))} ", flush=True)

    # Build LPM test sets
    print("Building standard LPM test set...", flush=True)
    from finetune_corruption import build_legal_mask
    std_games_test, std_legal_test = build_standard_lpm_test(n_games=10000, seed=args.seed+2)
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
    results = train_and_evaluate_scale(
        model, train_games, train_legal, test_sets, device,
        std_loader, std_mask, cor_loader, cor_mask,
        lr=args.lr, bs=16)

    # Save
    output = {
        'condition_id': cid,
        'group_name': group_name,
        'n_rules': n_rules,
        'n_rules_modified': n_modified,
        'rule_ids': rule_ids,
        'n_train_games': len(train_games),
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
