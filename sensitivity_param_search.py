"""
Sensitivity Parameter Search: 84 conditions (14 rule groups × 6 corruption types)

Each condition:
  - Selects 100 flanking rules (frequency-matched within pairs)
  - Applies a specific corruption type
  - Generates 50K training games + 5K targeted test positions
  - Fine-tunes pretrained OthelloGPT (bs=16, full 50K games)
  - Evaluates at steps 0, 50, 100, 200, 300, final

Usage:
    python sensitivity_param_search.py --condition-id 0 --output-dir experiments/param_search
"""

import argparse
import json
import os
import pickle
import sys
import time
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data.dataloader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hand_crafted_flanking import (
    enumerate_flanking_patterns, VALID_MOVES, MOVE_TO_IDX, N_MOVES,
    CENTER_CELLS, DIRECTIONS
)
from data.othello import OthelloBoardState
from mingpt.model import GPT, GPTConfig
from mingpt.dataset import CharDataset

CENTER_SET = np.array(sorted(CENTER_CELLS))

# ============================================================================
# Condition mapping
# ============================================================================

RULE_GROUPS = [
    'high_sens', 'low_sens',       # (a) sensitivity
    'diagonal', 'random_dir',      # (b) direction
    'left_board', 'full_board',    # (c) left vs whole board
    'center', 'periphery',        # (d) center vs periphery
    'coherent', 'incoherent',     # (e) spatial coherence
    'one_color', 'both_colors',   # (f) color restriction
    'after30', 'all_moves',       # (g) temporal restriction
]

CORRUPTION_TYPES = [
    'drop_third',
    'corrupt_nearby',
    'add_adjacent',
    'flip_color',
    'extend_chain',
    'play_occupied',
]


def condition_id_to_names(cid):
    group_idx = cid // 6
    corr_idx = cid % 6
    return RULE_GROUPS[group_idx], CORRUPTION_TYPES[corr_idx]


# ============================================================================
# Rule selection
# ============================================================================

def _frequency_match(pool, other_pool, n_rules, rng):
    """Select n_rules from pool with frequency matched to other_pool."""
    pool_freqs = [r['n_satisfied'] for r in pool]
    other_freqs = [r['n_satisfied'] for r in other_pool]
    overlap_min = max(min(pool_freqs), min(other_freqs))
    overlap_max = min(max(pool_freqs), max(other_freqs))

    if overlap_min >= overlap_max:
        # No overlap — just take n_rules from pool
        chosen = list(rng.choice(pool, min(n_rules, len(pool)), replace=False))
        return [r['rule_id'] for r in chosen]

    freq_bins = np.logspace(np.log10(max(overlap_min, 1)),
                            np.log10(overlap_max), 11)
    selected = []
    per_bin = max(1, n_rules // (len(freq_bins) - 1) + 1)

    for i in range(len(freq_bins) - 1):
        lo, hi = freq_bins[i], freq_bins[i + 1]
        candidates = [r for r in pool if lo <= r['n_satisfied'] < hi]
        other_count = sum(1 for r in other_pool if lo <= r['n_satisfied'] < hi)
        n_take = min(len(candidates), other_count, per_bin)
        if n_take > 0:
            chosen = rng.choice(candidates, n_take, replace=False)
            selected.extend(chosen.tolist())

    if len(selected) > n_rules:
        idx = rng.choice(len(selected), n_rules, replace=False)
        selected = [selected[i] for i in idx]

    return [r['rule_id'] for r in selected]


def select_rules_for_group(group_name, sensitivity_data, rng):
    """Select 100 rule IDs based on the rule group."""
    rules = sensitivity_data['rules']
    sorted_rules = sorted(rules, key=lambda r: r['sensitivity'], reverse=True)
    n_pool = len(sorted_rules) // 3

    if group_name == 'high_sens':
        high_pool = sorted_rules[:n_pool]
        low_pool = sorted_rules[-n_pool:]
        return _frequency_match(high_pool, low_pool, 100, rng)

    elif group_name == 'low_sens':
        high_pool = sorted_rules[:n_pool]
        low_pool = sorted_rules[-n_pool:]
        return _frequency_match(low_pool, high_pool, 100, rng)

    elif group_name == 'diagonal':
        diag_dirs = {(-1, -1), (-1, 1), (1, -1), (1, 1)}
        pool = [r for r in rules if tuple(r['direction_vec']) in diag_dirs]
        other = [r for r in rules if tuple(r['direction_vec']) not in diag_dirs]
        return _frequency_match(pool, other, 100, rng)

    elif group_name == 'random_dir':
        diag_dirs = {(-1, -1), (-1, 1), (1, -1), (1, 1)}
        pool = [r for r in rules if tuple(r['direction_vec']) not in diag_dirs]
        diag_pool = [r for r in rules if tuple(r['direction_vec']) in diag_dirs]
        # Match to diagonal selection's frequency
        diag_ids = _frequency_match(diag_pool, pool, 100, rng)
        diag_selected = [r for r in diag_pool if r['rule_id'] in set(diag_ids)]
        return _frequency_match(pool, diag_selected, 100, rng)

    elif group_name == 'left_board':
        pool = [r for r in rules if r['target'] % 8 < 4]
        other = [r for r in rules if r['target'] % 8 >= 4]
        return _frequency_match(pool, other, 100, rng)

    elif group_name == 'full_board':
        left_pool = [r for r in rules if r['target'] % 8 < 4]
        right_pool = [r for r in rules if r['target'] % 8 >= 4]
        left_ids = _frequency_match(left_pool, right_pool, 100, rng)
        left_selected = [r for r in left_pool if r['rule_id'] in set(left_ids)]
        return _frequency_match(rules, left_selected, 100, rng)

    elif group_name == 'center':
        center_cells = set()
        for r in range(2, 6):
            for c in range(2, 6):
                cell = r * 8 + c
                if cell not in CENTER_CELLS:
                    center_cells.add(cell)
        pool = [r for r in rules if r['target'] in center_cells]
        other = [r for r in rules if r['target'] not in center_cells]
        return _frequency_match(pool, other, 100, rng)

    elif group_name == 'periphery':
        center_cells = set()
        for r in range(2, 6):
            for c in range(2, 6):
                cell = r * 8 + c
                if cell not in CENTER_CELLS:
                    center_cells.add(cell)
        pool = [r for r in rules if r['target'] not in center_cells]
        center_pool = [r for r in rules if r['target'] in center_cells]
        center_ids = _frequency_match(center_pool, pool, 100, rng)
        center_selected = [r for r in center_pool if r['rule_id'] in set(center_ids)]
        return _frequency_match(pool, center_selected, 100, rng)

    elif group_name in ('coherent', 'incoherent'):
        # Both use the same 100 random rules; corruption differs
        return list(rng.choice([r['rule_id'] for r in rules], 100, replace=False))

    elif group_name in ('one_color', 'both_colors'):
        # Same 100 rules; the difference is in how corruption is applied
        return list(rng.choice([r['rule_id'] for r in rules], 100, replace=False))

    elif group_name in ('after30', 'all_moves'):
        # Same 100 rules; the difference is temporal application
        return list(rng.choice([r['rule_id'] for r in rules], 100, replace=False))

    else:
        raise ValueError(f"Unknown group: {group_name}")


# ============================================================================
# Corruption functions
# ============================================================================

def corrupt_drop_third(patterns, rule_ids, rng):
    """Remove floor(len(opp)/3) opponents from chain (min 1 if len >= 2)."""
    patterns = deepcopy(patterns)
    n_modified = 0
    for rid in rule_ids:
        p = patterns[rid]
        if len(p['opponents']) >= 2:
            n_drop = max(1, len(p['opponents']) // 3)
            drop_idx = sorted(rng.choice(len(p['opponents']), n_drop, replace=False),
                              reverse=True)
            for di in drop_idx:
                p['opponents'].pop(di)
            p['length'] = len(p['opponents'])
            n_modified += 1
    return patterns, n_modified


def corrupt_nearby(patterns, rule_ids, rng):
    """Replace floor(len(opp)/3) opponents with nearby (dist 2-3) cells."""
    patterns = deepcopy(patterns)
    n_modified = 0
    for rid in rule_ids:
        p = patterns[rid]
        n_replace = max(1, len(p['opponents']) // 3)
        replace_idx = rng.choice(len(p['opponents']), min(n_replace, len(p['opponents'])),
                                 replace=False)
        for idx in replace_idx:
            cell = p['opponents'][idx]
            candidates = _get_nearby_cells(cell)
            if candidates:
                p['opponents'][idx] = int(rng.choice(candidates))
                n_modified += 1
    return patterns, n_modified


def _get_nearby_cells(cell):
    """Get cells at Chebyshev distance 2-3 from cell."""
    r, c = cell // 8, cell % 8
    candidates = []
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                dist = max(abs(dr), abs(dc))
                if 2 <= dist <= 3:
                    candidates.append(nr * 8 + nc)
    return candidates


def corrupt_add_adjacent(patterns, rule_ids, rng):
    """Add floor(len(opp)/3) adjacent cells as extra opponent requirements."""
    patterns = deepcopy(patterns)
    n_modified = 0
    for rid in rule_ids:
        p = patterns[rid]
        n_add = max(1, len(p['opponents']) // 3)
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
                        if nc_cell not in used:
                            neighbors.add(nc_cell)
        neighbors = list(neighbors)
        if neighbors:
            n_add = min(n_add, len(neighbors))
            chosen = rng.choice(neighbors, n_add, replace=False)
            p['opponents'].extend([int(c) for c in chosen])
            p['length'] = len(p['opponents'])
            n_modified += 1
    return patterns, n_modified


def corrupt_flip_color(patterns, rule_ids, rng):
    """Flip color check for floor(len(opp)/3) opponents (opponent→friendly)."""
    patterns = deepcopy(patterns)
    n_modified = 0
    for rid in rule_ids:
        p = patterns[rid]
        n_flip = max(1, len(p['opponents']) // 3)
        flip_idx = rng.choice(len(p['opponents']), min(n_flip, len(p['opponents'])),
                              replace=False)
        p['flipped_indices'] = sorted(flip_idx.tolist())
        n_modified += 1
    return patterns, n_modified


def corrupt_extend_chain(patterns, rule_ids, rng):
    """Add one more opponent at end, shift terminal one cell further."""
    patterns = deepcopy(patterns)
    n_modified = 0
    for rid in rule_ids:
        p = patterns[rid]
        dr, dc = p['direction']
        old_term = p['terminal']
        tr, tc = old_term // 8, old_term % 8
        new_tr, new_tc = tr + dr, tc + dc
        if 0 <= new_tr < 8 and 0 <= new_tc < 8:
            p['opponents'].append(old_term)
            p['terminal'] = new_tr * 8 + new_tc
            p['length'] = len(p['opponents'])
            n_modified += 1
    return patterns, n_modified


def corrupt_play_occupied(patterns, rule_ids, rng):
    """Allow play on occupied squares for these rules' targets."""
    patterns = deepcopy(patterns)
    n_modified = 0
    for rid in rule_ids:
        patterns[rid]['skip_target_check'] = True
        n_modified += 1
    return patterns, n_modified


def apply_corruption(corruption_type, patterns, rule_ids, rng):
    """Apply the specified corruption to the given rules."""
    if corruption_type == 'drop_third':
        return corrupt_drop_third(patterns, rule_ids, rng)
    elif corruption_type == 'corrupt_nearby':
        return corrupt_nearby(patterns, rule_ids, rng)
    elif corruption_type == 'add_adjacent':
        return corrupt_add_adjacent(patterns, rule_ids, rng)
    elif corruption_type == 'flip_color':
        return corrupt_flip_color(patterns, rule_ids, rng)
    elif corruption_type == 'extend_chain':
        return corrupt_extend_chain(patterns, rule_ids, rng)
    elif corruption_type == 'play_occupied':
        return corrupt_play_occupied(patterns, rule_ids, rng)
    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")


def apply_spatial_corruption(group_name, patterns, rule_ids, corruption_type, rng):
    """For coherent/incoherent groups, apply spatial transformation THEN corruption."""
    patterns = deepcopy(patterns)

    if group_name == 'coherent':
        # Shift all cell references by (+1, +1), wrapping mod 8
        for rid in rule_ids:
            p = patterns[rid]
            p['target'] = _shift_cell(p['target'], 1, 1)
            p['opponents'] = [_shift_cell(c, 1, 1) for c in p['opponents']]
            p['terminal'] = _shift_cell(p['terminal'], 1, 1)

    elif group_name == 'incoherent':
        # Cross-wire: swap opponents/terminal between distant rules
        ids = list(rule_ids)
        rng.shuffle(ids)
        for i in range(0, len(ids) - 1, 2):
            a, b = ids[i], ids[i + 1]
            pa, pb = patterns[a], patterns[b]
            pa['opponents'], pb['opponents'] = pb['opponents'], pa['opponents']
            pa['terminal'], pb['terminal'] = pb['terminal'], pa['terminal']
            pa['length'] = len(pa['opponents'])
            pb['length'] = len(pb['opponents'])

    # Now apply the actual corruption type on top
    return apply_corruption(corruption_type, patterns, rule_ids, rng)


def _shift_cell(cell, dr, dc):
    """Shift a cell by (dr, dc), wrapping around the 8x8 board.
    If result lands on a center cell, shift again until it doesn't."""
    r, c = cell // 8, cell % 8
    for attempt in range(8):
        nr = (r + dr + attempt) % 8
        nc = (c + dc + attempt) % 8
        result = nr * 8 + nc
        if result not in CENTER_CELLS:
            return result
    return cell  # fallback: no shift


# ============================================================================
# Extended pattern evaluation (supports flip_color, play_occupied, player_only)
# ============================================================================

def precompute_pattern_arrays_extended(patterns):
    """Like precompute_pattern_arrays but with extra fields for extended eval."""
    n = len(patterns)
    max_opp = max(len(p['opponents']) for p in patterns)

    targets = np.array([p['target'] for p in patterns], dtype=np.int32)
    terminals = np.array([p['terminal'] for p in patterns], dtype=np.int32)
    opp_lens = np.array([len(p['opponents']) for p in patterns], dtype=np.int32)

    opp_cells = np.zeros((n, max_opp), dtype=np.int32)
    for i, p in enumerate(patterns):
        for j, o in enumerate(p['opponents']):
            opp_cells[i, j] = o

    opp_mask = np.arange(max_opp)[None, :] < opp_lens[:, None]

    # flip_mask: True where opponent check should be friendly instead
    flip_mask = np.zeros((n, max_opp), dtype=np.bool_)
    for i, p in enumerate(patterns):
        if 'flipped_indices' in p:
            for fi in p['flipped_indices']:
                if fi < max_opp:
                    flip_mask[i, fi] = True

    # skip_target_check: True where target doesn't need to be empty
    skip_target = np.array([p.get('skip_target_check', False) for p in patterns],
                           dtype=np.bool_)

    # player_only: -1 = any player, 0 = black only, 1 = white only
    player_only = np.array([p.get('player_restrict', -1) for p in patterns],
                           dtype=np.int32)

    return targets, terminals, opp_cells, opp_mask, flip_mask, skip_target, player_only


def evaluate_rules_extended(flat, is_black_turn, targets, terminals, opp_cells,
                            opp_mask, flip_mask, skip_target, player_only):
    """Extended vectorized rule evaluation."""
    my_val = 1 if is_black_turn else -1
    opp_val = -my_val

    # Target check
    target_empty = (flat[targets] == 0) | skip_target

    # Terminal check
    terminal_friendly = (flat[terminals] == my_val)

    # Opponent check with color flipping
    opp_vals = flat[opp_cells]
    opp_is_opponent = (opp_vals == opp_val)
    opp_is_friendly = (opp_vals == my_val)
    opp_match = np.where(flip_mask, opp_is_friendly, opp_is_opponent) | ~opp_mask
    opp_all = opp_match.all(axis=1)

    # Player restriction
    player_color = 0 if is_black_turn else 1
    player_ok = (player_only == -1) | (player_only == player_color)

    fires = target_empty & opp_all & terminal_friendly & player_ok

    if not fires.any():
        return None

    return np.unique(targets[fires])


# ============================================================================
# Game generation
# ============================================================================

def place_piece_no_flip(board, pos):
    """Place a piece without applying Othello flip rules."""
    r, c = pos // 8, pos % 8
    board.state[r, c] = board.next_hand_color
    board.next_hand_color *= -1


def generate_single_game_extended(arrays, rng, save_legal=False,
                                  arrays_late=None, phase_boundary=0):
    """Generate a single game. If arrays_late provided, switch at phase_boundary."""
    targets, terminals, opp_cells, opp_mask, flip_mask, skip_target, player_only = arrays
    if arrays_late is not None:
        t2, te2, oc2, om2, fm2, st2, po2 = arrays_late

    board = OthelloBoardState()
    moves = []
    legal_per_turn = [] if save_legal else None

    for turn in range(60):
        flat = board.state.flatten()
        is_black = (board.next_hand_color == 1)

        # Choose rule set based on phase
        if arrays_late is not None and turn >= phase_boundary:
            legal_cells = evaluate_rules_extended(
                flat, is_black, t2, te2, oc2, om2, fm2, st2, po2)
        else:
            legal_cells = evaluate_rules_extended(
                flat, is_black, targets, terminals, opp_cells, opp_mask,
                flip_mask, skip_target, player_only)

        if legal_cells is None:
            empty = np.where(flat == 0)[0]
            empty = np.setdiff1d(empty, CENTER_SET)
            if len(empty) == 0:
                break
            if save_legal:
                legal_per_turn.append(empty.tolist())
            board_pos = int(empty[rng.randint(len(empty))])
        else:
            if save_legal:
                legal_per_turn.append(legal_cells.tolist())
            board_pos = int(legal_cells[rng.randint(len(legal_cells))])

        if board.tentative_move(board_pos) != 0:
            board.update([board_pos])
        else:
            place_piece_no_flip(board, board_pos)

        moves.append(board_pos)

    if save_legal:
        return moves, legal_per_turn
    return moves


def generate_games_extended(arrays, num_games, rng, save_legal=False,
                            arrays_late=None, phase_boundary=0):
    """Generate multiple games with extended rules."""
    all_games = []
    all_legal = [] if save_legal else None

    for i in range(num_games):
        result = generate_single_game_extended(
            arrays, rng, save_legal=save_legal,
            arrays_late=arrays_late, phase_boundary=phase_boundary)
        if save_legal:
            game, legal = result
            all_games.append(game)
            all_legal.append(legal)
        else:
            all_games.append(result)

        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1}/{num_games} games...", flush=True)

    if save_legal:
        return all_games, all_legal
    return all_games


# ============================================================================
# Targeted test position collection
# ============================================================================

def _count_corrupted_rules_per_cell(flat, is_black, corrupted_patterns,
                                     rule_ids):
    """For each cell, count how many corrupted rules fire at this position."""
    my_val = 1 if is_black else -1
    opp_val = -my_val
    counts = {}
    for rid in rule_ids:
        p = corrupted_patterns[rid]
        target = p['target']
        # Check if this rule fires
        if flat[target] != 0 and not p.get('skip_target_check', False):
            continue
        if flat[p['terminal']] != my_val:
            continue
        all_opp = True
        flipped = p.get('flipped_indices', [])
        for j, opp_cell in enumerate(p['opponents']):
            expected = my_val if j in flipped else opp_val
            if flat[opp_cell] != expected:
                all_opp = False
                break
        if all_opp:
            counts[target] = counts.get(target, 0) + 1
    return counts


def collect_three_test_sets(corrupted_arrays, std_arrays, n_per_set=5000,
                            max_games=200000, rng=None,
                            arrays_late=None, phase_boundary=0,
                            corrupted_patterns=None, rule_ids=None):
    """Collect three test sets based on standard vs corrupted legality.

    For each position, categorize each cell into:
      - legal_legal (LL): legal under both standard and corrupted rules
      - illegal_legal (IL): illegal under standard, legal under corrupted
      - legal_illegal (LI): legal under standard, illegal under corrupted

    Also records n_corrupted_rules: how many corrupted rules fire for each
    target cell, for filtering in analysis.

    Returns dict with three lists of test positions. Each position has:
      game_prefix, move_idx, target_cells, n_corrupted_rules (per cell)
    """
    if rng is None:
        rng = np.random.RandomState(99)

    test_sets = {'LL': [], 'IL': [], 'LI': []}
    games_tried = 0

    while games_tried < max_games:
        # Check if we have enough
        if all(len(test_sets[k]) >= n_per_set for k in test_sets):
            break

        board = OthelloBoardState()
        game_moves = []

        for turn in range(60):
            flat = board.state.flatten()
            is_black = (board.next_hand_color == 1)

            std_legal = evaluate_rules_extended(flat, is_black, *std_arrays)
            std_set = set(std_legal.tolist()) if std_legal is not None else set()

            if arrays_late is not None and turn >= phase_boundary:
                cor_legal = evaluate_rules_extended(flat, is_black, *arrays_late)
            else:
                cor_legal = evaluate_rules_extended(flat, is_black, *corrupted_arrays)
            cor_set = set(cor_legal.tolist()) if cor_legal is not None else set()

            # Categorize cells
            ll_cells = sorted(std_set & cor_set)
            il_cells = sorted(cor_set - std_set)  # newly legal
            li_cells = sorted(std_set - cor_set)  # no longer legal

            # Count corrupted rules per cell (if patterns provided)
            if corrupted_patterns is not None and rule_ids is not None:
                n_cor_rules = _count_corrupted_rules_per_cell(
                    flat, is_black, corrupted_patterns, rule_ids)
            else:
                n_cor_rules = {}

            pos_info = {
                'game_prefix': list(game_moves),
                'move_idx': turn,
            }

            if ll_cells and len(test_sets['LL']) < n_per_set:
                test_sets['LL'].append({**pos_info, 'target_cells': ll_cells,
                                        'std_legal': sorted(std_set),
                                        'cor_legal': sorted(cor_set),
                                        'n_corrupted_rules': {c: n_cor_rules.get(c, 0) for c in ll_cells}})
            if il_cells and len(test_sets['IL']) < n_per_set:
                test_sets['IL'].append({**pos_info, 'target_cells': il_cells,
                                        'std_legal': sorted(std_set),
                                        'cor_legal': sorted(cor_set),
                                        'n_corrupted_rules': {c: n_cor_rules.get(c, 0) for c in il_cells}})
            if li_cells and len(test_sets['LI']) < n_per_set:
                test_sets['LI'].append({**pos_info, 'target_cells': li_cells,
                                        'std_legal': sorted(std_set),
                                        'cor_legal': sorted(cor_set),
                                        'n_corrupted_rules': {c: n_cor_rules.get(c, 0) for c in li_cells}})

            # Play under corrupted rules
            if cor_legal is not None and len(cor_legal) > 0:
                move = int(cor_legal[rng.randint(len(cor_legal))])
            else:
                empty = np.where(flat == 0)[0]
                empty = np.setdiff1d(empty, CENTER_SET)
                if len(empty) == 0:
                    break
                move = int(empty[rng.randint(len(empty))])

            if board.tentative_move(move) != 0:
                board.update([move])
            else:
                place_piece_no_flip(board, move)
            game_moves.append(move)

        games_tried += 1

    for k in test_sets:
        test_sets[k] = test_sets[k][:n_per_set]

    print(f"  Test sets: LL={len(test_sets['LL'])}, IL={len(test_sets['IL'])}, "
          f"LI={len(test_sets['LI'])} (from {games_tried} games)", flush=True)
    return test_sets


# ============================================================================
# Training and evaluation
# ============================================================================

def evaluate_on_test_sets(model, test_sets, dataset, device):
    """Evaluate model on three test sets.

    For each test set (LL, IL, LI), measures:
      - For LL: fraction of model probability on LL cells (should stay high)
      - For IL: fraction of model probability on IL cells (should increase)
      - For LI: fraction of model probability on LI cells (should decrease)

    Returns dict with metrics per test set.
    """
    model.eval()
    stoi = dataset.stoi

    results = {}
    with torch.no_grad():
        for set_name, positions in test_sets.items():
            total_prob_on_targets = 0.0
            total_acc = 0
            n = 0

            for pos in positions:
                prefix = pos['game_prefix']
                if len(prefix) < 1:
                    continue

                tokens = [stoi[m] for m in prefix]
                x = torch.tensor([tokens], dtype=torch.long, device=device)

                logits, _ = model(x)
                probs = torch.softmax(logits[0, -1, :], dim=-1)

                # Probability mass on target cells
                target_prob = 0.0
                for cell in pos['target_cells']:
                    if cell in stoi:
                        target_prob += probs[stoi[cell]].item()
                total_prob_on_targets += target_prob

                # Is argmax one of the target cells?
                pred_token = logits[0, -1, :].argmax().item()
                pred_cell = dataset.itos[pred_token]
                if pred_cell in set(pos['target_cells']):
                    total_acc += 1

                n += 1

            if n > 0:
                results[f'{set_name}_prob'] = total_prob_on_targets / n
                results[f'{set_name}_acc'] = total_acc / n
                results[f'{set_name}_n'] = n
            else:
                results[f'{set_name}_prob'] = 0.0
                results[f'{set_name}_acc'] = 0.0
                results[f'{set_name}_n'] = 0

    return results


EVAL_SCHEDULE = [0, 5, 25, 50, 100, 200, 300]  # plus final step


def build_standard_lpm_test(n_games=10000, seed=123):
    """Load standard Othello games and build legal mask for LPM evaluation."""
    from data.othello import OthelloBoardState
    import random as _random

    _random.seed(seed)
    rng = np.random.RandomState(seed)

    # Generate standard Othello games
    games = []
    legal_moves_all = []
    for _ in range(n_games):
        board = OthelloBoardState()
        game = []
        legal_per_turn = []
        for turn in range(60):
            valid = board.get_valid_moves()
            if not valid:
                break
            legal_per_turn.append(valid)
            move = valid[rng.randint(len(valid))]
            game.append(move)
            board.umpire(move)
        if len(game) >= 5:
            games.append(game)
            legal_moves_all.append(legal_per_turn)

    return games, legal_moves_all


def evaluate_standard_lpm(model, std_games, std_legal, dataset, device, bs=64):
    """Evaluate LPM on standard Othello games (catastrophic forgetting test)."""
    from finetune_corruption import evaluate, build_legal_mask

    std_dataset = CharDataset(std_games)
    std_mask = build_legal_mask(std_games, std_legal, dataset.stoi,
                                dataset.block_size, dataset.vocab_size)
    std_loader = DataLoader(std_dataset, batch_size=bs, shuffle=False,
                            num_workers=0)
    loss, acc, rank, lpm = evaluate(model, std_loader, device, std_mask)
    return {'std_loss': loss, 'std_acc': acc, 'std_rank': rank, 'std_lpm': lpm}


def train_and_evaluate(model, train_games, train_legal, test_sets,
                       device, std_games=None, std_legal=None,
                       cor_test_games=None, cor_test_legal=None,
                       bs=16, lr=3e-4):
    """Fine-tune model and evaluate at scheduled steps."""
    train_dataset = CharDataset(train_games)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              num_workers=0, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1,
                                 betas=(0.9, 0.95))

    total_steps = len(train_loader)
    eval_set = set(EVAL_SCHEDULE)

    # Store per-step results for each test set
    results = {'eval_steps': []}
    for k in ['LL', 'IL', 'LI']:
        results[f'{k}_prob'] = []
        results[f'{k}_acc'] = []
    results['std_lpm'] = []
    results['cor_lpm'] = []
    t0 = time.time()

    def do_eval(step):
        metrics = evaluate_on_test_sets(model, test_sets, train_dataset, device)
        results['eval_steps'].append(step)
        for k in ['LL', 'IL', 'LI']:
            results[f'{k}_prob'].append(metrics.get(f'{k}_prob', 0.0))
            results[f'{k}_acc'].append(metrics.get(f'{k}_acc', 0.0))

        extra_str = ""

        # Standard LPM (catastrophic forgetting check)
        if std_games is not None:
            std_metrics = evaluate_standard_lpm(model, std_games, std_legal,
                                                train_dataset, device)
            results['std_lpm'].append(std_metrics['std_lpm'])
            extra_str += f" std_lpm={std_metrics['std_lpm']:.4f}"
        else:
            results['std_lpm'].append(None)

        # Corrupted game LPM (learning check)
        if cor_test_games is not None:
            cor_metrics = evaluate_standard_lpm(model, cor_test_games,
                                                cor_test_legal,
                                                train_dataset, device)
            results['cor_lpm'].append(cor_metrics['std_lpm'])
            extra_str += f" cor_lpm={cor_metrics['std_lpm']:.4f}"
        else:
            results['cor_lpm'].append(None)

        elapsed = time.time() - t0
        print(f"  Step {step}: LL_prob={metrics.get('LL_prob',0):.4f} "
              f"IL_prob={metrics.get('IL_prob',0):.4f} "
              f"LI_prob={metrics.get('LI_prob',0):.4f}"
              f"{extra_str} elapsed={elapsed:.0f}s", flush=True)

    # Step 0 evaluation
    do_eval(0)

    batch_count = 0
    model.train()
    for it, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss = loss.mean()

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_count += 1

        if batch_count in eval_set:
            do_eval(batch_count)

    # Final evaluation
    do_eval(batch_count)

    results['total_steps'] = batch_count
    results['elapsed_seconds'] = time.time() - t0

    # Record test set sizes
    for k in ['LL', 'IL', 'LI']:
        results[f'{k}_n'] = len(test_sets.get(k, []))

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="experiments/param_search")
    parser.add_argument("--ckpt", type=str, default="ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=200000)
    parser.add_argument("--n-test", type=int, default=5000,
                        help="Number of test positions per test set (LL, IL, LI)")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    group_name, corruption_type = condition_id_to_names(args.condition_id)
    print(f"Condition {args.condition_id}: {group_name} × {corruption_type}")
    print(f"Device: {device}")

    # Load sensitivity data
    sens_path = os.path.join(os.path.dirname(__file__), 'behavioral_data/sensitivity.json')
    with open(sens_path) as f:
        sensitivity_data = json.load(f)

    # Select 100 rules
    rule_ids = select_rules_for_group(group_name, sensitivity_data, rng)
    print(f"Selected {len(rule_ids)} rules")

    # Get base patterns
    base_patterns = enumerate_flanking_patterns()

    # Apply corruption
    if group_name in ('coherent', 'incoherent'):
        corrupted_patterns, n_mod = apply_spatial_corruption(
            group_name, base_patterns, rule_ids, corruption_type, rng)
    elif group_name == 'one_color':
        # Apply corruption, then set player restriction
        corrupted_patterns, n_mod = apply_corruption(
            corruption_type, base_patterns, rule_ids, rng)
        for rid in rule_ids:
            corrupted_patterns[rid]['player_restrict'] = 0  # black only
    else:
        corrupted_patterns, n_mod = apply_corruption(
            corruption_type, base_patterns, rule_ids, rng)

    print(f"Corrupted {n_mod} rules (type={corruption_type})")

    # Build pattern arrays
    corrupted_arrays = precompute_pattern_arrays_extended(corrupted_patterns)
    std_arrays = precompute_pattern_arrays_extended(base_patterns)

    # Handle phased conditions (after30)
    arrays_late = None
    phase_boundary = 0
    if group_name == 'after30':
        # Standard rules for early, corrupted for late
        arrays_late = corrupted_arrays
        corrupted_arrays = std_arrays  # early phase uses standard
        phase_boundary = 30
    elif group_name == 'all_moves':
        # Corrupted rules for all moves (no phase)
        pass

    # Generate training games (load existing if available, generate more if needed)
    games_dir = os.path.join(args.output_dir, f"games/cond_{args.condition_id:03d}")
    os.makedirs(games_dir, exist_ok=True)
    existing_games_path = os.path.join(games_dir, "train_games.pickle")
    existing_legal_path = os.path.join(games_dir, "train_legal.pickle")

    train_games, train_legal = [], []
    if os.path.exists(existing_games_path) and os.path.exists(existing_legal_path):
        print(f"Loading existing games from {games_dir}...")
        with open(existing_games_path, 'rb') as f:
            train_games = pickle.load(f)
        with open(existing_legal_path, 'rb') as f:
            train_legal = pickle.load(f)
        print(f"  Loaded {len(train_games)} existing games")

    n_remaining = args.n_train - len(train_games)
    if n_remaining > 0:
        print(f"Generating {n_remaining} additional training games...")
        new_games, new_legal = generate_games_extended(
            corrupted_arrays, n_remaining, rng, save_legal=True,
            arrays_late=arrays_late, phase_boundary=phase_boundary)
        train_games.extend(new_games)
        train_legal.extend(new_legal)

    # Filter short games
    valid = [(g, l) for g, l in zip(train_games, train_legal) if len(g) >= 5]
    train_games = [g for g, _ in valid]
    train_legal = [l for _, l in valid]
    print(f"  {len(train_games)} games after filtering")

    # Collect three test sets
    print(f"Collecting test positions ({args.n_test} per set)...")
    test_sets = collect_three_test_sets(
        corrupted_arrays, std_arrays, n_per_set=args.n_test, rng=rng,
        arrays_late=arrays_late, phase_boundary=phase_boundary,
        corrupted_patterns=corrupted_patterns, rule_ids=rule_ids)

    # Save games and test sets for reproducibility
    with open(os.path.join(games_dir, "train_games.pickle"), 'wb') as f:
        pickle.dump(train_games, f)
    with open(os.path.join(games_dir, "train_legal.pickle"), 'wb') as f:
        pickle.dump(train_legal, f)
    with open(os.path.join(games_dir, "test_sets.pickle"), 'wb') as f:
        pickle.dump(test_sets, f)
    with open(os.path.join(games_dir, "corrupted_patterns.pickle"), 'wb') as f:
        pickle.dump(corrupted_patterns, f)
    print(f"  Saved games to {games_dir}")

    # Load model
    print("Loading model...")
    from data import get_othello
    othello = get_othello(ood_num=100)
    dummy_dataset = CharDataset(othello)

    mconf = GPTConfig(dummy_dataset.vocab_size, dummy_dataset.block_size,
                      n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    state_dict = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Build standard Othello test set for catastrophic forgetting measurement
    print("Building standard Othello test set (10K games)...")
    std_games, std_legal = build_standard_lpm_test(n_games=10000, seed=123)
    print(f"  {len(std_games)} standard games")

    # Sample 10K from training data for corrupted-game LPM
    print("Sampling 10K training games for corrupted LPM test...")
    cor_test_idx = rng.choice(len(train_games), min(10000, len(train_games)), replace=False)
    cor_test_games = [train_games[i] for i in cor_test_idx]
    cor_test_legal = [train_legal[i] for i in cor_test_idx]
    print(f"  {len(cor_test_games)} corrupted test games")

    # Train and evaluate
    print("Training...")
    results = train_and_evaluate(
        model, train_games, train_legal, test_sets, device,
        std_games=std_games, std_legal=std_legal,
        cor_test_games=cor_test_games, cor_test_legal=cor_test_legal)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        'condition_id': args.condition_id,
        'rule_group': group_name,
        'corruption_type': corruption_type,
        'n_rules_selected': len(rule_ids),
        'n_rules_modified': n_mod,
        'rule_ids': [int(r) for r in rule_ids],
        'n_train_games': len(train_games),
        'seed': args.seed,
        **results,
    }

    # Add sensitivity stats for selected rules
    rule_sens = [sensitivity_data['rules'][rid]['sensitivity'] for rid in rule_ids]
    rule_freq = [sensitivity_data['rules'][rid]['n_satisfied'] for rid in rule_ids]
    output['mean_sensitivity'] = float(np.mean(rule_sens))
    output['mean_frequency'] = float(np.mean(rule_freq))
    output['total_impact'] = float(np.sum([s * f for s, f in zip(rule_sens, rule_freq)]))

    out_path = os.path.join(args.output_dir, f"cond_{args.condition_id:03d}.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
