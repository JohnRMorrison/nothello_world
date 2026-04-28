"""Pruned helpers used by incoherent_rules_experiment.py.

Extracted from sensitivity_param_search.py (the larger 84-condition
parameter sweep) — kept just the ~10 functions Task B actually calls.
"""
import numpy as np
import torch
from copy import deepcopy

from hand_crafted_flanking import CENTER_CELLS
from data.othello import OthelloBoardState

CENTER_SET = np.array(sorted(CENTER_CELLS))


def build_legal_mask(games, legal_moves, stoi, block_size, vocab_size):
    """Boolean mask (N_tokens, vocab_size) of legal moves.

    Inlined from finetune_corruption.py (parent repo).
    """
    max_key = max(k for k in stoi.keys() if k >= 0)
    stoi_map = np.full(max_key + 1, -1, dtype=np.int32)
    for k, v in stoi.items():
        if k >= 0:
            stoi_map[k] = v
    rows = []
    cols = []
    pos = 0
    for gi, game in enumerate(games):
        T = min(len(game) - 1, block_size)
        for t in range(T):
            move_idx = t + 1
            if move_idx < len(legal_moves[gi]):
                for p in legal_moves[gi][move_idx]:
                    if p <= max_key and stoi_map[p] >= 0:
                        rows.append(pos)
                        cols.append(stoi_map[p])
            pos += 1
    total = pos
    mask = np.zeros((total, vocab_size), dtype=np.bool_)
    mask[np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)] = True
    return mask


def place_piece_no_flip(board, pos):
    """Place a piece without applying Othello flip rules."""
    r, c = pos // 8, pos % 8
    board.state[r, c] = board.next_hand_color
    board.next_hand_color *= -1


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
                            corrupted_patterns=None, rule_ids=None,
                            skip_sets=None):
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

    if skip_sets is None:
        skip_sets = set()
    active_sets = {'LL', 'IL', 'LI'} - set(skip_sets)
    test_sets = {k: [] for k in active_sets}
    games_tried = 0

    while games_tried < max_games:
        if all(len(test_sets[k]) >= n_per_set for k in active_sets):
            break
        # Stop early if no positions found after 20K games for any set
        if games_tried >= 20000:
            empty = [k for k in active_sets if len(test_sets[k]) == 0]
            if empty:
                print(f"  Warning: no positions found for {empty} after {games_tried} games, stopping",
                      flush=True)
                active_sets -= set(empty)
                if not active_sets:
                    break

        board = OthelloBoardState()
        game_moves = []
        # Track which sets got a position from this game (at most 1 per set per game)
        game_used = set()

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

            if 'LL' in active_sets and ll_cells and len(test_sets['LL']) < n_per_set and 'LL' not in game_used and turn >= 4:
                test_sets['LL'].append({**pos_info, 'target_cells': ll_cells,
                                        'std_legal': sorted(std_set),
                                        'cor_legal': sorted(cor_set),
                                        'n_corrupted_rules': {c: n_cor_rules.get(c, 0) for c in ll_cells}})
                game_used.add('LL')
            if 'IL' in active_sets and il_cells and len(test_sets['IL']) < n_per_set and 'IL' not in game_used and turn >= 4:
                test_sets['IL'].append({**pos_info, 'target_cells': il_cells,
                                        'std_legal': sorted(std_set),
                                        'cor_legal': sorted(cor_set),
                                        'n_corrupted_rules': {c: n_cor_rules.get(c, 0) for c in il_cells}})
                game_used.add('IL')
            if 'LI' in active_sets and li_cells and len(test_sets['LI']) < n_per_set and 'LI' not in game_used and turn >= 4:
                test_sets['LI'].append({**pos_info, 'target_cells': li_cells,
                                        'std_legal': sorted(std_set),
                                        'cor_legal': sorted(cor_set),
                                        'n_corrupted_rules': {c: n_cor_rules.get(c, 0) for c in li_cells}})
                game_used.add('LI')

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

    counts = {k: len(test_sets.get(k, [])) for k in ['LL', 'IL', 'LI']}
    print(f"  Test sets: LL={counts['LL']}, IL={counts['IL']}, "
          f"LI={counts['LI']} (from {games_tried} games)", flush=True)
    return test_sets


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

                # Is argmax legal under corrupted rules?
                pred_token = logits[0, -1, :].argmax().item()
                pred_cell = dataset.itos[pred_token]
                if pred_cell in set(pos['cor_legal']):
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


def prepare_lpm_test(games, legal_moves, ref_dataset, device, bs=64):
    """Pre-build dataset, mask, and loader for LPM evaluation.

    Creates CharDataset with overridden stoi/itos to match the model's
    vocabulary. Returns (loader, mask) ready for evaluate().
    """
    
    ds = CharDataset(games)
    ds.stoi = ref_dataset.stoi
    ds.itos = ref_dataset.itos
    ds.vocab_size = ref_dataset.vocab_size

    mask = build_legal_mask(games, legal_moves, ref_dataset.stoi,
                            ref_dataset.block_size, ref_dataset.vocab_size)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    return loader, mask


def evaluate_lpm(model, loader, mask, device):
    """Evaluate LPM using pre-built loader and mask."""
    from finetune_corruption import evaluate
    loss, acc, rank, lpm = evaluate(model, loader, device, mask)
    return {'loss': loss, 'acc': acc, 'rank': rank, 'lpm': lpm}


