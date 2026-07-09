"""Aggregate precedence metrics across all adversarial positions.

For each position (game, error_turn T, illegal cell C):
  1. Identify critical probe-error cells at T (require |crit| >= 1).
  2. Walk C's-parity turns t <= T.  At each turn, compute:
       P(C, t)                         (softmax marginal on 60 valid cells)
       C_illegal(t)                    (bool, from actual board)
       min_margin(t) = min over critical cells k of logit_margin(k, t)
  3. For threshold in {1%, 3%, 5%}:
       t_bias = first t on C's parity with C illegal AND P(C, t) > threshold
       quiet  = {t' on C's parity, t' < t_bias, C illegal, P(C, t') < threshold}
       Qualify if |quiet| >= 1.
       Metric 1: min_margin(t_bias) > 0  (probe correct on all critical cells)
       Metric 2: Δ = min_margin(t_bias) - mean_{t' in quiet} min_margin(t')

Reports coverage (fraction qualified) and Metric-1 / Metric-2 aggregates
per threshold.  Also dumps per-position rows to a CSV for follow-up.
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


def build_pos_to_token(block_size):
    dummy_game = list(VALID_MOVES)
    toks = tokenize_games([dummy_game], seq_len=block_size)[0].tolist()
    pos_to_token = np.full(64, -1, dtype=np.int64)
    for i, m in enumerate(dummy_game):
        if i < len(toks):
            pos_to_token[m] = toks[i]
    return pos_to_token


def gt_probe_class(state_8x8, cell):
    """Nanda-state (1=BLACK, -1=WHITE, 0=empty) -> probe class {0, 1, 2}."""
    v = state_8x8[cell // 8, cell % 8]
    if v == 0: return 0
    if v == -1: return 1
    return 2


def process_position(game, T, C, model, probe, layer, block_size,
                       pos_to_token, device):
    """Return dict with per-turn arrays (or None if no critical cells)."""
    L = min(T + 1, block_size)
    tokens = tokenize_games([game[:L]], seq_len=block_size).to(device)
    with torch.no_grad():
        logits, _ = model(tokens)                                # (1, block_size, V)
        acts = extract_activations(model, tokens, layer)          # (1, block_size, 512)
    acts_np = acts.cpu().numpy()[0]                              # (block_size, 512)

    # Play the game & cache per-turn state + legality
    board = OthelloBoardState()
    state_at = {}
    legal_at = {}
    for t in range(0, T + 1):
        try:
            board.umpire(game[t])
        except Exception:
            return None
        state_at[t] = np.asarray(board.state, dtype=np.int8).copy()
        legal_at[t] = set(board.get_valid_moves())

    # Critical cells at T (from probe argmax vs actual)
    color_next = next_hand_color_at_turn(T)
    W_T = probe[0 if T % 2 == 1 else 1]                          # (512, 8, 8, 3)
    h_T = torch.from_numpy(acts_np[T]).float()
    logits_T = torch.einsum('d,drco->rco', h_T, W_T)
    cls_T = logits_T.argmax(dim=-1).detach().cpu().numpy()       # (8, 8)
    probe_state_T = np.zeros((8, 8), dtype=np.int8)
    probe_state_T[cls_T == 1] = -1
    probe_state_T[cls_T == 2] = 1

    flank_dirs = flank_providing_directions(probe_state_T, C, color_next)
    crit_cells = set()
    for dr in flank_dirs:
        crit_cells.update(critical_errors_for_direction(
            probe_state_T, state_at[T], C, dr, color_next))
    if not crit_cells:
        return None
    crit_cells = sorted(crit_cells)

    # Precompute crit-cell ground-truth class + row/col indices
    crit_rc = np.array([[k // 8, k % 8] for k in crit_cells], dtype=np.int64)

    # Walk C's-parity turns t <= T
    target_parity = T % 2
    P_C = []
    is_illegal = []
    min_margin = []
    turn_list = []
    C_idx60 = VALID_MOVES.index(C)

    for t in range(0, T + 1):
        if t % 2 != target_parity:
            continue

        # P(C, t)
        probs = F.softmax(logits[0, t, :], dim=-1).detach().cpu().numpy()
        probs_60 = np.zeros(60, dtype=np.float32)
        for k, m in enumerate(VALID_MOVES):
            tok = int(pos_to_token[m])
            if tok >= 0:
                probs_60[k] = probs[tok]
        probs_60 = probs_60 / max(probs_60.sum(), 1e-9)
        P_C.append(float(probs_60[C_idx60]))

        is_illegal.append(C not in legal_at[t])

        # min margin across critical cells
        mode = 0 if t % 2 == 1 else 1
        W = probe[mode]                                            # (512, 8, 8, 3)
        h = torch.from_numpy(acts_np[t]).float()
        cell_logits = torch.einsum('d,do->o', h,
                                     W[:, crit_rc[0, 0], crit_rc[0, 1], :])
        margins = []
        for i, k in enumerate(crit_cells):
            r_, c_ = k // 8, k % 8
            lg = torch.einsum('d,do->o', h, W[:, r_, c_, :]).detach().cpu().numpy()
            gt_cls = gt_probe_class(state_at[t], k)
            other = [lg[j] for j in range(3) if j != gt_cls]
            margins.append(float(lg[gt_cls] - max(other)))
        min_margin.append(min(margins))
        turn_list.append(t)

    return {
        'turns': np.array(turn_list, dtype=np.int64),
        'P_C': np.array(P_C, dtype=np.float64),
        'is_illegal': np.array(is_illegal, dtype=bool),
        'min_margin': np.array(min_margin, dtype=np.float64),
        'n_crit': len(crit_cells),
    }


def compute_metrics(traj, threshold):
    turns = traj['turns']
    P_C = traj['P_C']
    ill = traj['is_illegal']
    mm = traj['min_margin']

    # First bias turn: first t on C's parity where C illegal AND P(C) > threshold
    mask_bias = ill & (P_C > threshold)
    if not mask_bias.any():
        return None
    j_bias = np.argmax(mask_bias)                # first True index

    # Quiet turns: t' < t_bias, C illegal, P(C, t') < threshold
    mask_quiet = ill & (P_C < threshold)
    mask_quiet[j_bias:] = False
    if not mask_quiet.any():
        return None

    min_margin_bias = float(mm[j_bias])
    min_margin_quiet_mean = float(mm[mask_quiet].mean())
    return {
        'qualified': True,
        'metric1_correct': min_margin_bias > 0,      # probe correct on all critical cells
        'metric1_conf_correct': min_margin_bias > 1, # confidently correct
        'delta': min_margin_bias - min_margin_quiet_mean,
        'mm_bias': min_margin_bias,
        'mm_quiet': min_margin_quiet_mean,
        't_bias': int(turns[j_bias]),
        'n_quiet': int(mask_quiet.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_by_depth')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe',
                    default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0,
                    help='If > 0, process only this many positions (for debug).')
    ap.add_argument('--thresholds', type=str, default='0.01,0.03,0.05')
    ap.add_argument('--output-csv', default='precedence_aggregate.csv')
    ap.add_argument('--output-summary', default='precedence_aggregate.txt')
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(',')]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    print(f"Loaded OGPT (block_size={block_size})")

    probe = torch.load(args.probe, map_location='cpu')
    assert probe.shape == (3, 512, 8, 8, 3)
    print(f"Loaded probe {tuple(probe.shape)}")

    pos_to_token = build_pos_to_token(block_size)

    records_path = os.path.join(args.adversarial_dir, 'adversarial_records.npz')
    d = np.load(records_path, allow_pickle=True)
    games = d['games']; turns = d['turns']; cells = d['illegal_cells']
    N = len(games) if args.limit == 0 else min(args.limit, len(games))
    print(f"Processing {N} / {len(games)} positions")

    # Per-position records
    csv_rows = [['idx', 'T', 'C', 'n_crit', 'threshold',
                 'qualified', 't_bias', 'n_quiet',
                 'mm_bias', 'mm_quiet', 'delta',
                 'metric1_correct', 'metric1_conf_correct']]
    per_thresh_counts = {th: {
        'total': 0, 'no_crit': 0, 'no_bias': 0, 'no_quiet': 0,
        'qualified': 0, 'm1_correct': 0, 'm1_conf': 0,
        'deltas': [], 'mm_bias': [], 'mm_quiet': [],
    } for th in thresholds}

    t0 = time.time()
    for i in range(N):
        game = list(games[i])
        T = int(turns[i]); C = int(cells[i])
        traj = process_position(game, T, C, model, probe,
                                 args.layer, block_size, pos_to_token, device)
        for th in thresholds:
            per_thresh_counts[th]['total'] += 1
            if traj is None:
                per_thresh_counts[th]['no_crit'] += 1
                csv_rows.append([i, T, C, 0, th, 'no_crit',
                                  '', '', '', '', '', '', ''])
                continue
            m = compute_metrics(traj, th)
            if m is None:
                # Distinguish no-bias-turn vs no-quiet-turn
                mask_bias = traj['is_illegal'] & (traj['P_C'] > th)
                if not mask_bias.any():
                    per_thresh_counts[th]['no_bias'] += 1
                    reason = 'no_bias'
                else:
                    per_thresh_counts[th]['no_quiet'] += 1
                    reason = 'no_quiet'
                csv_rows.append([i, T, C, traj['n_crit'], th, reason,
                                  '', '', '', '', '', '', ''])
                continue
            per_thresh_counts[th]['qualified'] += 1
            if m['metric1_correct']:
                per_thresh_counts[th]['m1_correct'] += 1
            if m['metric1_conf_correct']:
                per_thresh_counts[th]['m1_conf'] += 1
            per_thresh_counts[th]['deltas'].append(m['delta'])
            per_thresh_counts[th]['mm_bias'].append(m['mm_bias'])
            per_thresh_counts[th]['mm_quiet'].append(m['mm_quiet'])
            csv_rows.append([i, T, C, traj['n_crit'], th, 'qualified',
                              m['t_bias'], m['n_quiet'],
                              f"{m['mm_bias']:.4f}",
                              f"{m['mm_quiet']:.4f}",
                              f"{m['delta']:.4f}",
                              int(m['metric1_correct']),
                              int(m['metric1_conf_correct'])])

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (N - i - 1) / (i + 1)
            print(f"  {i+1}/{N}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                   flush=True)

    # Write CSV
    with open(args.output_csv, 'w') as f:
        for row in csv_rows:
            f.write(','.join(str(x) for x in row) + '\n')
    print(f"Wrote per-position CSV to {args.output_csv}")

    # Write summary
    summary_lines = []
    summary_lines.append(f"Adversarial positions processed: {N}")
    summary_lines.append("")
    for th in thresholds:
        c = per_thresh_counts[th]
        total = c['total']
        qual = c['qualified']
        summary_lines.append(f"=== Threshold: P(C) > {th * 100:.0f}% ===")
        summary_lines.append(f"  Total positions:                {total}")
        summary_lines.append(f"  No critical cells at T:         {c['no_crit']}  "
                              f"({c['no_crit'] / total * 100:5.1f}%)")
        summary_lines.append(f"  No bias turn (P(C) never > th): {c['no_bias']}  "
                              f"({c['no_bias'] / total * 100:5.1f}%)")
        summary_lines.append(f"  No quiet illegal turn before:   {c['no_quiet']}  "
                              f"({c['no_quiet'] / total * 100:5.1f}%)")
        summary_lines.append(f"  QUALIFIED:                      {qual}  "
                              f"({qual / total * 100:5.1f}%)")
        if qual > 0:
            deltas = np.array(c['deltas'])
            mm_bias = np.array(c['mm_bias'])
            mm_quiet = np.array(c['mm_quiet'])
            summary_lines.append("")
            summary_lines.append(f"  Metric 1: probe correct on ALL critical cells at t_bias "
                                  f"(min-margin > 0)")
            summary_lines.append(f"    Fraction:                     "
                                  f"{c['m1_correct'] / qual * 100:5.1f}%")
            summary_lines.append(f"  Metric 1 strict: min-margin > +1")
            summary_lines.append(f"    Fraction:                     "
                                  f"{c['m1_conf'] / qual * 100:5.1f}%")
            summary_lines.append("")
            summary_lines.append(f"  Metric 2: Δ = min_margin(t_bias) - "
                                  f"mean_{{t' in quiet}} min_margin(t')")
            summary_lines.append(f"    Δ median:                     {np.median(deltas):+7.3f}")
            summary_lines.append(f"    Δ mean:                       {deltas.mean():+7.3f}")
            summary_lines.append(f"    Δ std:                        {deltas.std():+7.3f}")
            summary_lines.append(f"    Δ percentiles (10/25/75/90):  "
                                  f"{np.percentile(deltas, 10):+.2f} / "
                                  f"{np.percentile(deltas, 25):+.2f} / "
                                  f"{np.percentile(deltas, 75):+.2f} / "
                                  f"{np.percentile(deltas, 90):+.2f}")
            summary_lines.append(f"    min_margin at t_bias  (median): {np.median(mm_bias):+.3f}")
            summary_lines.append(f"    min_margin at quiet   (median): {np.median(mm_quiet):+.3f}")
        summary_lines.append("")

    with open(args.output_summary, 'w') as f:
        f.write("\n".join(summary_lines))
    print("\n".join(summary_lines))
    print(f"Wrote summary to {args.output_summary}")


if __name__ == '__main__':
    main()
