"""Three-way classification of adversarial positions.

For each position (game, T, C), classify as sticky-cell, bias-first,
or corruption via a two-question decision tree:

  Q1: Was C legal at some earlier same-parity turn with P(C) > threshold?
      If yes, take t_L = last such legal turn walking backward from
      t_bias.  t_transition = the same-parity turn immediately after
      t_L (i.e. t_L + 2), which must be illegal (that is the transition
      point).  If P(C, t_transition) > 0.5 * P(C, t_L), classify as
      STICKY-CELL and stop.

  Q2: (Only if Q1 didn't classify.)  At t_bias, is the probe correct on
      all critical cells identified at T?  YES -> BIAS-FIRST.
      NO -> CORRUPTION.  If T has no critical cells at all (no
      hallucinated flank), that is also BIAS-FIRST -- OGPT rationalises
      C without any world-model support.

Coverage: 100% (any position with t_bias defined).

Sweeps thresholds and reports the fraction in each category per
threshold.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, VOCAB_SIZE, extract_activations,
)
from experiment_probe_causal_analysis import (
    next_hand_color_at_turn, flank_providing_directions,
    critical_errors_for_direction,
)


CENTER_CELLS = {27, 28, 35, 36}
VALID_MOVES = [i for i in range(64) if i not in CENTER_CELLS]

PERSISTENCE_THRESHOLD = 0.5


def build_pos_to_token(block_size):
    dummy_game = list(VALID_MOVES)
    toks = tokenize_games([dummy_game], seq_len=block_size)[0].tolist()
    pos_to_token = np.full(64, -1, dtype=np.int64)
    for i, m in enumerate(dummy_game):
        if i < len(toks):
            pos_to_token[m] = toks[i]
    return pos_to_token


def probe_argmax_state(hidden_512, turn, probe):
    mode = 0 if turn % 2 == 1 else 1
    W = probe[mode]
    h = torch.from_numpy(hidden_512).float()
    logits = torch.einsum('d,drco->rco', h, W)
    cls = logits.argmax(dim=-1).detach().cpu().numpy()
    st = np.zeros((8, 8), dtype=np.int8)
    st[cls == 1] = -1
    st[cls == 2] = 1
    return st


def probe_logits_at_cell(hidden_512, turn, probe, cell):
    mode = 0 if turn % 2 == 1 else 1
    W = probe[mode]
    r, c = cell // 8, cell % 8
    h = torch.from_numpy(hidden_512).float()
    return torch.einsum('d,do->o', h, W[:, r, c, :]).detach().cpu().numpy()


def gt_probe_class(state_8x8, cell):
    v = state_8x8[cell // 8, cell % 8]
    if v == 0: return 0
    if v == -1: return 1
    return 2


def classify_position(game, T, C, model, probe, layer, block_size,
                       pos_to_token, device, thresholds):
    """Return {threshold -> category}."""
    L = min(T + 1, block_size)
    tokens = tokenize_games([game[:L]], seq_len=block_size).to(device)
    with torch.no_grad():
        logits, _ = model(tokens)
        acts = extract_activations(model, tokens, layer)
    acts_np = acts.cpu().numpy()[0]

    board = OthelloBoardState()
    state_at, legal_at = {}, {}
    for t in range(0, T + 1):
        try:
            board.umpire(game[t])
        except Exception:
            return None
        state_at[t] = np.asarray(board.state, dtype=np.int8).copy()
        legal_at[t] = set(board.get_valid_moves())

    color_next = next_hand_color_at_turn(T)
    target_parity = T % 2
    parity_turns = [t for t in range(0, T + 1) if t % 2 == target_parity]

    # Precompute P(C, t) for all same-parity turns
    C_idx60 = VALID_MOVES.index(C)
    p_C_at = {}
    for t in parity_turns:
        probs = F.softmax(logits[0, t, :], dim=-1).detach().cpu().numpy()
        probs_60 = np.zeros(60, dtype=np.float32)
        for k, m in enumerate(VALID_MOVES):
            tok = int(pos_to_token[m])
            if tok >= 0:
                probs_60[k] = probs[tok]
        probs_60 = probs_60 / max(probs_60.sum(), 1e-9)
        p_C_at[t] = float(probs_60[C_idx60])

    # Critical cells at T
    probe_state_T = probe_argmax_state(acts_np[T], T, probe)
    flank_dirs = flank_providing_directions(probe_state_T, C, color_next)
    crit_cells = set()
    for dr in flank_dirs:
        crit_cells.update(critical_errors_for_direction(
            probe_state_T, state_at[T], C, dr, color_next))
    crit_cells = sorted(crit_cells)
    n_crit = len(crit_cells)

    result = {}
    for th in thresholds:
        # Find t_bias
        t_bias = None
        for t in parity_turns:
            if (C not in legal_at[t]) and p_C_at[t] > th:
                t_bias = t
                break
        if t_bias is None:
            result[th] = {'category': 'no_bias_turn', 'n_crit': n_crit}
            continue

        # Q1: sticky-cell test.  Walk backward from t_bias on same parity
        # to find the most recent LEGAL same-parity turn with P > threshold.
        t_L = None
        for t in reversed(parity_turns):
            if t >= t_bias:
                continue
            if (C in legal_at[t]) and p_C_at[t] > th:
                t_L = t
                break

        sticky = False
        persistence = None
        t_transition = None
        if t_L is not None:
            # t_transition is the same-parity turn immediately after t_L
            # (must be illegal for the transition to exist)
            t_transition = t_L + 2
            if t_transition > T:
                # No such turn in-range; can't test persistence
                sticky = False
            elif t_transition not in p_C_at:
                sticky = False
            elif C in legal_at.get(t_transition, set()):
                # t_transition is legal (edge case: L->L, no transition here)
                sticky = False
            else:
                persistence = p_C_at[t_transition] / max(p_C_at[t_L], 1e-12)
                sticky = (persistence > PERSISTENCE_THRESHOLD)

        if sticky:
            result[th] = {
                'category': 'sticky_cell',
                'n_crit': n_crit,
                't_bias': int(t_bias), 't_L': int(t_L),
                't_transition': int(t_transition),
                'p_L': p_C_at[t_L], 'p_transition': p_C_at[t_transition],
                'persistence': persistence,
            }
            continue

        # Q2: probe correct on ALL critical cells at t_bias?
        if n_crit == 0:
            # No hallucinated flank at T -- pure readout failure = bias-first
            result[th] = {
                'category': 'bias_first',
                'n_crit': 0,
                't_bias': int(t_bias),
                'subtype': 'no_crit',
            }
            continue

        # Compute min margin at t_bias
        margins = []
        for k in crit_cells:
            lg = probe_logits_at_cell(acts_np[t_bias], t_bias, probe, k)
            gt_cls = gt_probe_class(state_at[t_bias], k)
            other = [lg[j] for j in range(3) if j != gt_cls]
            margins.append(float(lg[gt_cls] - max(other)))
        min_margin_bias = min(margins)

        if min_margin_bias > 0:
            result[th] = {
                'category': 'bias_first',
                'n_crit': n_crit,
                't_bias': int(t_bias),
                'min_margin': min_margin_bias,
                'subtype': 'probe_intact',
            }
        else:
            result[th] = {
                'category': 'corruption',
                'n_crit': n_crit,
                't_bias': int(t_bias),
                'min_margin': min_margin_bias,
            }

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_by_depth')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe',
                    default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--thresholds', type=str,
                    default='0.001,0.003,0.005,0.01,0.03,0.05')
    ap.add_argument('--output-csv', default='three_way.csv')
    ap.add_argument('--output-summary', default='three_way.txt')
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(',')]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    probe = torch.load(args.probe, map_location='cpu')
    pos_to_token = build_pos_to_token(block_size)

    d = np.load(os.path.join(args.adversarial_dir, 'adversarial_records.npz'),
                allow_pickle=True)
    games = d['games']; turns = d['turns']; cells = d['illegal_cells']
    N = len(games) if args.limit == 0 else min(args.limit, len(games))
    print(f"Processing {N} positions")

    categories = ['sticky_cell', 'bias_first', 'corruption', 'no_bias_turn']
    counts = {th: {c: 0 for c in categories} for th in thresholds}

    csv_rows = [['idx', 'T', 'C'] + [f'th{th}' for th in thresholds]]

    t0 = time.time()
    for i in range(N):
        game = list(games[i]); T = int(turns[i]); C = int(cells[i])
        res = classify_position(game, T, C, model, probe, args.layer,
                                 block_size, pos_to_token, device, thresholds)
        row = [i, T, C]
        for th in thresholds:
            if res is None:
                row.append('')
                continue
            cat = res[th]['category']
            counts[th][cat] += 1
            row.append(cat)
        csv_rows.append(row)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (N - i - 1) / (i + 1)
            print(f"  {i+1}/{N}  ({elapsed:.0f}s, ~{eta:.0f}s remaining)",
                   flush=True)

    with open(args.output_csv, 'w') as f:
        for row in csv_rows:
            f.write(','.join(str(x) for x in row) + '\n')
    print(f"Wrote per-position CSV to {args.output_csv}")

    lines = [f"Adversarial positions processed: {N}",
              f"Persistence threshold (sticky-cell criterion): "
              f"P(t_transition) > {PERSISTENCE_THRESHOLD} * P(t_L)", ""]
    for th in thresholds:
        c = counts[th]
        total = sum(c.values())
        lines.append(f"=== Threshold P(C) > {th * 100:.2f}% ===")
        lines.append(f"  Total positions classified:      {total}")
        for cat in categories:
            pct = c[cat] / total * 100 if total > 0 else 0.0
            lines.append(f"    {cat:<20s}: {c[cat]:>6d}  ({pct:5.1f}%)")
        # Combined bias-side vs corruption side
        bias_total = c['sticky_cell'] + c['bias_first']
        lines.append(f"    -- bias total (sticky + bias-first):    "
                      f"{bias_total}  ({bias_total / total * 100:5.1f}%)")
        lines.append(f"    -- corruption:                          "
                      f"{c['corruption']}  ({c['corruption'] / total * 100:5.1f}%)")
        lines.append("")

    with open(args.output_summary, 'w') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote summary to {args.output_summary}")


if __name__ == '__main__':
    main()
