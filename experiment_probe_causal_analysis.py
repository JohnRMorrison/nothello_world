"""Deep causal analysis of probe errors on adversarial positions.

Implements the 7-step plan to distinguish "world model corruption creates
hallucinated flanks at C" (World C) from "world model is diffuse noise that
happens to make C legal" (World A).

For each adversarial position:
  - Identify flank-providing directions under probe-decoded board
  - Find "critical errors": probe errors necessary for a flank
  - Minimal correction test: does fixing only critical errors make C illegal?
  - Classify each ray error as flank-creating vs flank-irrelevant
  - Shuffled baseline: random errors at same positions -> flank rate
  - Compare probe error rate on flank-providing rays vs others

Outputs a per-position CSV plus aggregate stats.

Usage:
    python experiment_probe_causal_analysis.py \\
        --adversarial-dir experiment1_data \\
        --ckpt ckpts/gpt_nanda_synthetic.ckpt \\
        --probe mechanistic_interpretability/main_linear_probe.pth \\
        --output-csv experiment_probe_causal.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, load_games, extract_activations, VOCAB_SIZE, GAME_LEN,
)

from experiment_probe_on_adversarial import (
    state_to_gt, _build_token_to_board_pos, find_adversarial_positions,
    get_hidden_and_state, probe_predict,
)


# 8 direction vectors for flanking
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)]


def probe_to_nanda_state(pred_flat):
    """probe pred (64,) w/ 0=empty, 1=white, 2=black -> nanda state (8,8) w/ 1=black,-1=white,0=empty."""
    st = np.zeros((8, 8), dtype=np.int8)
    for c in range(64):
        r_, c_ = c // 8, c % 8
        p = int(pred_flat[c])
        if p == 1:
            st[r_, c_] = -1
        elif p == 2:
            st[r_, c_] = 1
    return st


def nanda_state_to_probe(state_8x8):
    """Inverse: (8,8) nanda -> (64,) probe convention."""
    flat = np.zeros(64, dtype=np.int64)
    for cell in range(64):
        r_, c_ = cell // 8, cell % 8
        v = state_8x8[r_, c_]
        if v == -1:
            flat[cell] = 1
        elif v == 1:
            flat[cell] = 2
    return flat


def next_hand_color_at_turn(t):
    k = t + 1
    return 1 if (k % 2 == 0) else -1


def is_flank_in_direction(state_8x8, C_cell, direction, next_player):
    """Return True if a valid Othello flank exists from C_cell in `direction`
    on the given state, for `next_player` (1=black, -1=white)."""
    if state_8x8[C_cell // 8, C_cell % 8] != 0:
        return False  # C must be empty
    dr, dc = direction
    r, c = C_cell // 8 + dr, C_cell % 8 + dc
    # First step must be opponent piece
    if not (0 <= r < 8 and 0 <= c < 8):
        return False
    if state_8x8[r, c] != -next_player:
        return False
    # Continue while opponent, terminate on friendly
    r += dr; c += dc
    while 0 <= r < 8 and 0 <= c < 8:
        v = state_8x8[r, c]
        if v == next_player:
            return True  # terminal found
        if v == 0:
            return False  # empty breaks the flank
        r += dr; c += dc
    return False


def flank_providing_directions(state_8x8, C_cell, next_player):
    return [d for d in DIRS
             if is_flank_in_direction(state_8x8, C_cell, d, next_player)]


def ray_cells_in_direction(C_cell, direction):
    """Cells starting from C+direction, up to board edge (list of cell indices)."""
    dr, dc = direction
    r, c = C_cell // 8 + dr, C_cell % 8 + dc
    out = []
    while 0 <= r < 8 and 0 <= c < 8:
        out.append(r * 8 + c)
        r += dr; c += dc
    return out


def critical_errors_for_direction(probe_state, gt_state, C_cell, direction,
                                    next_player):
    """Cells in this ray where probe disagrees with gt AND swapping to gt
    breaks the flank."""
    if not is_flank_in_direction(probe_state, C_cell, direction, next_player):
        return []
    ray = ray_cells_in_direction(C_cell, direction)
    critical = []
    for cell in ray:
        r_, c_ = cell // 8, cell % 8
        pv = probe_state[r_, c_]
        gv = gt_state[r_, c_]
        if pv == gv:
            continue
        # Try swapping probe->gt for this cell only, see if flank breaks
        swapped = probe_state.copy()
        swapped[r_, c_] = gv
        if not is_flank_in_direction(swapped, C_cell, direction, next_player):
            critical.append(cell)
    return critical


def classify_ray_error(probe_state, gt_state, C_cell, cell_i, direction,
                        next_player):
    """Return 'creating' if the error contributes to a valid flank via that
    direction, else 'irrelevant'."""
    # Test flank with the current probe (with error)
    with_err = is_flank_in_direction(probe_state, C_cell, direction, next_player)
    # Test flank if this specific cell were correct
    fixed = probe_state.copy()
    r_, c_ = cell_i // 8, cell_i % 8
    fixed[r_, c_] = gt_state[r_, c_]
    without_err = is_flank_in_direction(fixed, C_cell, direction, next_player)
    if with_err and not without_err:
        return 'creating'
    return 'irrelevant'


def shuffled_baseline(probe_state, gt_state, C_cell, all_ray_cells,
                       next_player, n_shuffles=100, seed=0):
    """Count how many random error patterns (same # errors on ray, random
    which ray cells + random wrong values) make C legal."""
    # Identify current probe errors on ray
    err_cells = [ci for ci in all_ray_cells
                  if probe_state[ci // 8, ci % 8] != gt_state[ci // 8, ci % 8]]
    n_err = len(err_cells)
    if n_err == 0 or len(all_ray_cells) < n_err:
        return 0.0
    rng = np.random.RandomState(seed)
    n_legal = 0
    for _ in range(n_shuffles):
        # Pick n_err random ray cells, assign random wrong values
        idx = rng.choice(len(all_ray_cells), size=n_err, replace=False)
        shuf = gt_state.copy()  # start from ground truth
        for j in idx:
            cell = all_ray_cells[j]
            r_, c_ = cell // 8, cell % 8
            true_v = gt_state[r_, c_]
            # Pick a WRONG value
            choices = [v for v in (-1, 0, 1) if v != true_v]
            shuf[r_, c_] = rng.choice(choices)
        # Check if C is legal under shuffled state
        try:
            b = OthelloBoardState()
            b.state = shuf.copy()
            b.next_hand_color = next_player
            if C_cell in b.get_valid_moves():
                n_legal += 1
        except Exception:
            pass
    return n_legal / n_shuffles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_data',
                    help='Directory of experiment1 output .npz files.  Ignored '
                         'when --natural-source is set.')
    ap.add_argument('--natural-source', action='store_true',
                    help='Instead of loading beam-search adversarial games '
                         'from experiment1_data, walk val games directly and '
                         'find EVERY position where OGPT top-1 is illegal. '
                         'Larger sample, matches natural failure distribution.')
    ap.add_argument('--max-files', type=int, default=3,
                    help='For --natural-source: how many val pickle files to '
                         'load (each ~100K games).')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe',
                    default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--max-adversarial', type=int, default=5000,
                    help='Cap on number of adversarial positions to analyze.')
    ap.add_argument('--n-shuffles', type=int, default=100)
    ap.add_argument('--output-csv', default='experiment_probe_causal.csv')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    probe = torch.load(args.probe, map_location='cpu')
    print(f"Probe shape: {tuple(probe.shape)}")
    assert probe.shape == (3, 512, 8, 8, 3)

    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    print(f"Loaded OGPT (block_size={block_size}), probing layer {args.layer}")

    token_to_pos = _build_token_to_board_pos(block_size, device)

    if args.natural_source:
        print(f"Loading val games (up to {args.max_files} pickle files)...")
        val_games = load_games(max_files=args.max_files)
        print(f"  {len(val_games):,} val games loaded")
        print("Finding adversarial (game, turn, C) triples from val games "
              "(every position with illegal top-1)...")
        adv_positions = find_adversarial_positions(
            model, device, val_games, block_size, token_to_pos,
            max_positions=args.max_adversarial,
        )
    else:
        print(f"Loading adversarial games from {args.adversarial_dir}...")
        adv_games = []
        for c in range(60):
            p = os.path.join(args.adversarial_dir, f'cell_{c:02d}.npz')
            if not os.path.exists(p):
                continue
            d = np.load(p, allow_pickle=True)
            for g in d['top_games']:
                adv_games.append(list(g))
        print(f"  {len(adv_games)} candidate games")
        print("Finding adversarial (game, turn, C) triples...")
        adv_positions = find_adversarial_positions(
            model, device, adv_games, block_size, token_to_pos,
            max_positions=args.max_adversarial,
        )
    print(f"  {len(adv_positions):,} adversarial positions")

    # Per-position analysis
    rows = []
    dist_critical = []
    n_became_illegal = 0
    n_analyzed_c = 0
    n_creating = 0
    n_irrelevant = 0
    err_rates_flank = []
    err_rates_other = []
    shuffled_frac_legal = []

    t0 = time.time()
    for i, (game_tuple, turn, C) in enumerate(adv_positions):
        hidden, gt_state = get_hidden_and_state(
            model, device, game_tuple, turn, args.layer, block_size)
        pred = probe_predict(hidden, turn, probe)                   # (8, 8)
        pred_flat = pred.flatten()
        probe_state = probe_to_nanda_state(pred_flat)
        color_next = next_hand_color_at_turn(turn)

        # Only analyze positions where C is legal under probe's board
        flank_dirs = flank_providing_directions(probe_state, C, color_next)
        if not flank_dirs:
            continue
        n_analyzed_c += 1

        # Critical errors: union across all flank-providing directions
        crit_cells = set()
        for d in flank_dirs:
            crit_cells.update(critical_errors_for_direction(
                probe_state, gt_state, C, d, color_next))
        dist_critical.append(len(crit_cells))

        # Minimal correction: replace critical cells with gt, see if C illegal
        corrected = probe_state.copy()
        for cell in crit_cells:
            r_, c_ = cell // 8, cell % 8
            corrected[r_, c_] = gt_state[r_, c_]
        try:
            b = OthelloBoardState()
            b.state = corrected.copy()
            b.next_hand_color = color_next
            became_illegal = (C not in b.get_valid_moves())
        except Exception:
            became_illegal = False
        if became_illegal:
            n_became_illegal += 1

        # Classify ray errors
        all_ray_cells = []
        for d in DIRS:
            all_ray_cells.extend(ray_cells_in_direction(C, d))
        ray_err_cells = [ci for ci in all_ray_cells
                          if probe_state[ci // 8, ci % 8] !=
                             gt_state[ci // 8, ci % 8]]
        n_ray_creating = 0
        n_ray_irrelevant = 0
        for cell in ray_err_cells:
            # Find which direction cell belongs to (may be multiple? — rays are disjoint)
            cell_dirs = [d for d in DIRS
                          if cell in ray_cells_in_direction(C, d)]
            # A cell can only belong to at most one direction from C
            # (the 8 rays are disjoint), so cell_dirs has length 1
            if not cell_dirs:
                continue
            d = cell_dirs[0]
            klass = classify_ray_error(probe_state, gt_state, C, cell,
                                        d, color_next)
            if klass == 'creating':
                n_ray_creating += 1
            else:
                n_ray_irrelevant += 1
        n_creating += n_ray_creating
        n_irrelevant += n_ray_irrelevant

        # Direction-specific error rate
        flank_ray_err = 0; flank_ray_total = 0
        other_ray_err = 0; other_ray_total = 0
        for d in DIRS:
            ray = ray_cells_in_direction(C, d)
            n_err = sum(1 for ci in ray if probe_state[ci // 8, ci % 8]
                         != gt_state[ci // 8, ci % 8])
            if d in flank_dirs:
                flank_ray_err += n_err
                flank_ray_total += len(ray)
            else:
                other_ray_err += n_err
                other_ray_total += len(ray)
        flank_rate = flank_ray_err / max(flank_ray_total, 1)
        other_rate = other_ray_err / max(other_ray_total, 1)
        err_rates_flank.append(flank_rate)
        err_rates_other.append(other_rate)

        # Shuffled baseline
        p_shuffle = shuffled_baseline(
            probe_state, gt_state, C, all_ray_cells, color_next,
            n_shuffles=args.n_shuffles, seed=i)
        shuffled_frac_legal.append(p_shuffle)

        rows.append({
            'i': i,
            'turn': int(turn),
            'C': int(C),
            'n_flank_dirs': len(flank_dirs),
            'n_critical_errors': len(crit_cells),
            'became_illegal_after_min_correction': int(became_illegal),
            'n_ray_creating': n_ray_creating,
            'n_ray_irrelevant': n_ray_irrelevant,
            'err_rate_flank_rays': f"{flank_rate:.4f}",
            'err_rate_other_rays': f"{other_rate:.4f}",
            'shuffled_p_legal': f"{p_shuffle:.4f}",
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(adv_positions)}  "
                  f"({int(time.time()-t0)}s)", flush=True)

    # Write CSV
    with open(args.output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved {args.output_csv}")

    # Aggregate stats
    print()
    print("=" * 66)
    print("AGGREGATE RESULTS")
    print("=" * 66)
    n_C_legal = n_analyzed_c
    print(f"Positions where C is legal under probe's board: {n_C_legal}")
    if n_C_legal > 0:
        # Step 3
        print(f"\nSTEP 3 (Minimal correction):")
        print(f"  Critical error count distribution:")
        crit = np.array(dist_critical)
        for q in (0, 25, 50, 75, 100):
            print(f"    p{q:>3d}: {int(np.percentile(crit, q))}")
        print(f"    mean: {crit.mean():.2f}")
        print(f"  After correcting only critical errors, "
              f"{n_became_illegal}/{n_C_legal} = "
              f"{n_became_illegal/n_C_legal:.4f} became ILLEGAL.")

        # Step 4
        total_ray_err = n_creating + n_irrelevant
        if total_ray_err > 0:
            print(f"\nSTEP 4 (Ray error classification):")
            print(f"  Flank-creating errors: "
                  f"{n_creating}/{total_ray_err} = "
                  f"{n_creating/total_ray_err:.4f}")
            print(f"  Flank-irrelevant errors: "
                  f"{n_irrelevant}/{total_ray_err} = "
                  f"{n_irrelevant/total_ray_err:.4f}")

        # Step 5
        shuf = np.array(shuffled_frac_legal)
        print(f"\nSTEP 5 (Shuffled baseline):")
        print(f"  P(C legal under randomly-permuted same-count ray errors) "
              f"mean: {shuf.mean():.4f}")
        print(f"  Compared to observed P(C legal under probe): 0.9005")

        # Step 6
        f_arr = np.array(err_rates_flank)
        o_arr = np.array(err_rates_other)
        print(f"\nSTEP 6 (Direction-specific error rate):")
        print(f"  Mean error rate on FLANK-providing rays: {f_arr.mean():.4f}")
        print(f"  Mean error rate on OTHER 7 rays:         {o_arr.mean():.4f}")
        diff = f_arr - o_arr
        print(f"  Paired diff (flank - other):  mean = {diff.mean():+.4f}, "
              f"std = {diff.std():.4f}")


if __name__ == '__main__':
    main()
