"""Per-position table: probability on the affected moves after each
intervention (Nanda et al.'s push) and in natural play, on the same 5000
positions as the Section 5.4 numbers (generate_paper_numbers.py, seed 42).

Outputs (CSV):
  <out>_positions.csv     one row per position: pos_index, move_number,
                          next_color, move_history, legal_moves
  <out>_probabilities.csv one row per (position, square, kind):
    kind = natural    : the current player already has a piece on the square
                        (no intervention). moves_affected = moves that are
                        legal only because of that piece; P_clean = model's
                        total probability on them (the 12.4% statistic).
    kind = add_mine / add_yours / flip / remove : intervention.
                        newly_legal / newly_illegal moves on the edited board,
                        their total probability before (clean) and after the
                        intervention at each size.

Intervention: x <- x + alpha * unit(probe vector of the target state) after
every block from Li layer 4 onward (blocks 3..7), each block its own probe,
one fixed alpha for every intervention (1x = --fixed-alpha).

Squares: r, c are 0-7 row/column indices into board.state; flat = 8*r + c.
Moves are flat indices 0-63, space-separated.

Usage:
    python build_position_table.py --fixed-alpha 1.530237
"""
import argparse
import collections
import csv
import glob
import os
import pickle
import random
import sys

import torch

sys.path.insert(0, '.')
from src import config, intervention, metrics
from mechanistic_interpretability.multi_intervention import compute_counterfactual_legal
from data.othello import OthelloBoardState
from generate_paper_numbers import bv2cls, classify_change

START_BLOCK = 3            # Li layer 4
PROBE_MODE = 0
CENTER = {27, 28, 35, 36}
VALID64 = [c for c in range(64) if c not in CENTER]
STOI = {c: i + 1 for i, c in enumerate(config.STOI_INDICES)}
fmt_moves = lambda s: ' '.join(str(m) for m in sorted(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-positions', type=int, default=5000)
    ap.add_argument('--pos-lo', type=int, default=10)
    ap.add_argument('--pos-hi', type=int, default=50)
    ap.add_argument('--num-pickle-files', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fixed-alpha', type=float, required=True)
    ap.add_argument('--alpha-mults', type=float, nargs='+', default=[1, 2.5])
    ap.add_argument('--out', default='Intervention Results/position_table/position_table')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = intervention.load_model(device=device)
    layer_probes = intervention.load_layer_probes(range(START_BLOCK, 8), device=device)
    sizes = [(m, args.fixed_alpha * m) for m in args.alpha_mults]
    print(f'device {device}; fixed 1x = {args.fixed_alpha}; sizes {sizes}')

    # Same position sampling as generate_paper_numbers.py.
    files = sorted(glob.glob('./data/othello_synthetic/*.pickle'))[-args.num_pickle_files:]
    games = []
    for fp in files:
        with open(fp, 'rb') as fh:
            games.extend(pickle.load(fh))
    rng.shuffle(games)
    games = games[:max(args.n_positions * 3, 10000)]

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    pos_f = open(args.out + '_positions.csv', 'w', newline='')
    prob_f = open(args.out + '_probabilities.csv', 'w', newline='')
    pos_w, prob_w = csv.writer(pos_f), csv.writer(prob_f)
    pos_w.writerow(['pos_index', 'move_number', 'next_color', 'move_history', 'legal_moves'])
    size_cols = []
    for m, _ in sizes:
        size_cols += [f'P_after_newly_legal_{m:g}x', f'P_after_newly_illegal_{m:g}x']
    prob_w.writerow(['pos_index', 'r', 'c', 'flat', 'square_state', 'kind', 'effect',
                     'moves_affected', 'P_clean', 'newly_legal', 'newly_illegal',
                     'P_before_newly_legal', 'P_before_newly_illegal'] + size_cols)

    counts = collections.Counter()
    n_processed = 0
    for g in games:
        if n_processed >= args.n_positions:
            break
        if len(g) <= args.pos_lo:
            continue
        board = OthelloBoardState()
        prefix = []
        target_pos = rng.randint(args.pos_lo, min(args.pos_hi, len(g) - 1))
        ok = True
        for t in range(target_pos):
            valid = board.get_valid_moves()
            if not valid:
                board.update([]); valid = board.get_valid_moves()
                if not valid: ok = False; break
            if g[t] not in valid: ok = False; break
            board.update([g[t]]); prefix.append(g[t])
        if not ok or len(prefix) != target_pos:
            continue
        valid = board.get_valid_moves()
        if not valid:
            continue

        board2d = board.state.copy()
        color = int(board.next_hand_color)
        orig_legal = set(int(m) for m in valid)
        pos = len(prefix) - 1
        pi = n_processed
        pos_w.writerow([pi, len(prefix), color, ' '.join(str(m) for m in prefix),
                        fmt_moves(orig_legal)])

        token_ids = torch.tensor([[STOI[m] for m in prefix]], dtype=torch.long, device=device)
        with torch.no_grad():
            prefix_acts = intervention.compute_prefix(model, token_ids, START_BLOCK + 1)
            clean_logits = intervention.forward_from_prefix(
                model, prefix_acts.clone(), START_BLOCK + 1)
        clean_p60 = metrics.cell_probs(clean_logits[0, -1].detach().cpu())

        for flat in VALID64:
            r, c = flat // 8, flat % 8
            orig = int(board2d[r, c])
            state = 'empty' if orig == 0 else ('mine' if orig == color else 'yours')

            # Natural play, exactly as run_ogpt_vs_natural_all_squares.py.
            if orig == color:
                remove_cf = compute_counterfactual_legal(board2d, [(r, c, 0)], color)
                needs_x = orig_legal - remove_cf
                if needs_x:
                    prob_w.writerow([pi, r, c, flat, state, 'natural', '',
                                     fmt_moves(needs_x),
                                     f'{metrics.mass_on(clean_p60, needs_x):.6g}',
                                     '', '', '', ''] + [''] * len(size_cols))
                    counts['natural'] += 1

            conditions = ([('add_mine', color), ('add_yours', -color)] if orig == 0
                          else [('remove', 0), ('flip', -orig)])
            for kind, tgt in conditions:
                cf = compute_counterfactual_legal(board2d, [(r, c, tgt)], color)
                sub = classify_change(orig_legal, cf)
                if sub is None:
                    continue
                cc, tc = bv2cls(orig, color), bv2cls(tgt, color)
                if cc == tc:
                    continue
                nl, ni = cf - orig_legal, orig_legal - cf
                after = []
                for _, alpha in sizes:
                    logits, _ = intervention.run_cascade(
                        model, prefix_acts, pos, (r, c), cc, tc, START_BLOCK, alpha,
                        layer_probes, probe_mode=PROBE_MODE,
                        later_layers='always', push='add')
                    p = metrics.cell_probs(logits[0, -1].detach().cpu())
                    after += [f'{metrics.mass_on(p, nl):.6g}', f'{metrics.mass_on(p, ni):.6g}']
                prob_w.writerow([pi, r, c, flat, state, kind, sub, '', '',
                                 fmt_moves(nl), fmt_moves(ni),
                                 f'{metrics.mass_on(clean_p60, nl):.6g}',
                                 f'{metrics.mass_on(clean_p60, ni):.6g}'] + after)
                counts[f'{kind}/{sub}'] += 1

        n_processed += 1
        if n_processed % 200 == 0:
            print(f'  {n_processed}/{args.n_positions}', flush=True)
            prob_f.flush(); pos_f.flush()

    pos_f.close(); prob_f.close()
    print(f'Done: {n_processed} positions')
    print('Row counts (interventions should match the Section 5.4 "Per-condition n"):')
    for k in sorted(counts):
        print(f'  {k}: n={counts[k]}')
    print(f'Saved -> {args.out}_positions.csv and {args.out}_probabilities.csv')


if __name__ == '__main__':
    main()
