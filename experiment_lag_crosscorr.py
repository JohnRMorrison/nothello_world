"""Lag cross-correlation between P(C) and probe corruption metrics
across the C's-parity turn trajectory of each adversarial position.

For each (game, T, C):
  1. Play the game.  On every C's-parity turn t, compute:
       - P(C, t)
       - margin at the critical cell (cell with worst probe margin at T
         among cells identified as flank-critical by the causal analysis).
         Undefined when there are no critical cells (~18% of positions).
       - margin averaged over all cells on C's 8 rays
       - CE loss at the critical cell:  -log p(true class)
       - CE loss averaged over all cells on C's 8 rays
     Sign convention: margin metrics are FLIPPED (`-margin`) so that
     higher = more corrupt, matching the loss metrics.

  2. Two turn slices per position:
       (a) ALL same-parity turns t in [0, T]
       (b) ONLY same-parity turns where C is illegal on the actual board

  3. For each (metric, slice), compute Pearson cross-correlation between
     P(C, t) and metric(t + Delta) for Delta in {-3, -2, -1, 0, +1, +2, +3}.

Per-position CSV wide-format: idx, T, C, n_all, n_ill, then columns
`{slice}_{metric}_lag{d}` giving each of the 7 correlations for each of
4 metrics x 2 slices.  Positions with too-short series or degenerate
variance write empty strings.

Summary aggregates for each (metric, slice, Delta): median and IQR of
rho_p across positions, plus the asymmetry statistic
S = median_p [rho_p(+1) - rho_p(-1)].
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
    critical_errors_for_direction, DIRS, ray_cells_in_direction,
)


CENTER_CELLS = {27, 28, 35, 36}
VALID_MOVES = [i for i in range(64) if i not in CENTER_CELLS]
LAGS = list(range(-3, 4))
METRICS = ['margin_crit', 'margin_mean', 'loss_crit', 'loss_mean']
SLICES = ['all', 'ill']


def build_pos_to_token(block_size):
    dummy_game = list(VALID_MOVES)
    toks = tokenize_games([dummy_game], seq_len=block_size)[0].tolist()
    pos_to_token = np.full(64, -1, dtype=np.int64)
    for i, m in enumerate(dummy_game):
        if i < len(toks):
            pos_to_token[m] = toks[i]
    return pos_to_token


def gt_probe_class(state_8x8, cell):
    v = state_8x8[cell // 8, cell % 8]
    if v == 0: return 0
    if v == -1: return 1
    return 2


def compute_probe_metrics(hidden_512, turn, probe, C, state,
                           ray_cells, critical_cell):
    """Return dict of 4 corruption metrics at this turn.

    `ray_cells` is a precomputed list of all cells on C's 8 rays.
    `critical_cell` is the fixed cell to track across the trajectory
    (may be None if no critical cells exist for this position).
    Convention: higher = more corrupt.
    """
    mode = 0 if turn % 2 == 1 else 1
    W = probe[mode]
    h = torch.from_numpy(hidden_512).float()
    logits = torch.einsum('d,drco->rco', h, W).detach().cpu().numpy()  # (8, 8, 3)

    margins = []
    losses = []
    for cell in ray_cells:
        r_, c_ = cell // 8, cell % 8
        lg = logits[r_, c_]
        gt_cls = gt_probe_class(state, cell)
        # softmax
        s = lg - lg.max()
        exp_s = np.exp(s)
        p = exp_s / exp_s.sum()
        other = [lg[j] for j in range(3) if j != gt_cls]
        margin = float(lg[gt_cls] - max(other))
        margins.append(margin)
        losses.append(float(-np.log(p[gt_cls] + 1e-12)))

    out = {
        'margin_mean': -float(np.mean(margins)),   # flip sign: higher = more corrupt
        'loss_mean':    float(np.mean(losses)),
        'margin_crit': np.nan,
        'loss_crit':   np.nan,
    }
    if critical_cell is not None:
        r_, c_ = critical_cell // 8, critical_cell % 8
        lg = logits[r_, c_]
        gt_cls = gt_probe_class(state, critical_cell)
        s = lg - lg.max()
        exp_s = np.exp(s)
        p = exp_s / exp_s.sum()
        other = [lg[j] for j in range(3) if j != gt_cls]
        margin = float(lg[gt_cls] - max(other))
        out['margin_crit'] = -margin
        out['loss_crit'] = float(-np.log(p[gt_cls] + 1e-12))
    return out


def lag_correlations(x, y, lags):
    """Return {lag -> Pearson rho} for shifting y by lag turns.

    Positive lag: pair x[t] with y[t+lag] (y in the future).
    Negative lag: pair x[t] with y[t+lag] (y in the past).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = {}
    n = len(x)
    for lag in lags:
        if lag >= 0:
            xs = x[: n - lag] if lag > 0 else x
            ys = y[lag:]
        else:
            xs = x[-lag:]
            ys = y[: n + lag]
        # NaN filter (a metric might be undefined at some turns)
        m = np.isfinite(xs) & np.isfinite(ys)
        if m.sum() < 3 or np.std(xs[m]) == 0 or np.std(ys[m]) == 0:
            out[lag] = np.nan
        else:
            out[lag] = float(np.corrcoef(xs[m], ys[m])[0, 1])
    return out


def process_position(game, T, C, model, probe, layer, block_size,
                      pos_to_token, device):
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

    # Ray cells (fixed for this position)
    ray_cells = []
    for d in DIRS:
        ray_cells.extend(ray_cells_in_direction(C, d))

    # Identify critical cell to track (may be None).  Pick the cell with
    # WORST margin at T among the causal-analysis critical cells.
    probe_state_T = np.zeros((8, 8), dtype=np.int8)
    mode_T = 0 if T % 2 == 1 else 1
    hT = torch.from_numpy(acts_np[T]).float()
    logits_T = torch.einsum('d,drco->rco', hT, probe[mode_T]).detach().cpu().numpy()
    cls_T = logits_T.argmax(axis=-1)
    probe_state_T[cls_T == 1] = -1
    probe_state_T[cls_T == 2] = 1

    flank_dirs = flank_providing_directions(probe_state_T, C, color_next)
    crit_set = set()
    for dr in flank_dirs:
        crit_set.update(critical_errors_for_direction(
            probe_state_T, state_at[T], C, dr, color_next))
    critical_cell = None
    if crit_set:
        # Pick worst-margin cell at T
        worst = None
        worst_m = np.inf
        for k in crit_set:
            r_, c_ = k // 8, k % 8
            lg = logits_T[r_, c_]
            gt = gt_probe_class(state_at[T], k)
            other = [lg[j] for j in range(3) if j != gt]
            m = float(lg[gt] - max(other))
            if m < worst_m:
                worst_m = m
                worst = k
        critical_cell = int(worst)

    # Walk parity turns, compute metrics
    p_series = []
    metric_series = {m: [] for m in METRICS}
    illegal_mask = []
    for t in parity_turns:
        p_series.append(p_C_at[t])
        illegal_mask.append(C not in legal_at[t])
        mets = compute_probe_metrics(acts_np[t], t, probe, C, state_at[t],
                                       ray_cells, critical_cell)
        for m in METRICS:
            metric_series[m].append(mets[m])

    p_series = np.array(p_series)
    ill_mask = np.array(illegal_mask, dtype=bool)

    result = {
        'critical_cell': critical_cell,
        'n_all': len(p_series),
        'n_ill': int(ill_mask.sum()),
    }

    for slice_name in SLICES:
        if slice_name == 'all':
            mask = np.ones(len(p_series), dtype=bool)
        else:
            mask = ill_mask
        p_slice = p_series[mask]
        for m in METRICS:
            y = np.array(metric_series[m], dtype=np.float64)[mask]
            if len(p_slice) < 5:
                # too short for reliable lag correlations
                result[(slice_name, m)] = {lag: np.nan for lag in LAGS}
            else:
                result[(slice_name, m)] = lag_correlations(
                    p_slice, y, LAGS)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_by_depth')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe',
                    default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--output-csv', default='lag_crosscorr.csv')
    ap.add_argument('--output-summary', default='lag_crosscorr.txt')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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

    header = ['idx', 'T', 'C', 'critical_cell', 'n_all', 'n_ill']
    for slice_name in SLICES:
        for m in METRICS:
            for lag in LAGS:
                header.append(f"{slice_name}_{m}_lag{lag:+d}")
    csv_rows = [header]

    # Accumulators for aggregation
    all_rhos = {(sl, m, lag): [] for sl in SLICES for m in METRICS for lag in LAGS}

    t0 = time.time()
    for i in range(N):
        game = list(games[i]); T = int(turns[i]); C = int(cells[i])
        res = process_position(game, T, C, model, probe, args.layer,
                                 block_size, pos_to_token, device)
        row = [i, T, C]
        if res is None:
            row.extend([''] * (len(header) - 3))
            csv_rows.append(row)
            continue
        row.extend([res['critical_cell'] if res['critical_cell'] is not None else '',
                     res['n_all'], res['n_ill']])
        for sl in SLICES:
            for m in METRICS:
                for lag in LAGS:
                    v = res[(sl, m)][lag]
                    row.append(f"{v:.4f}" if np.isfinite(v) else '')
                    if np.isfinite(v):
                        all_rhos[(sl, m, lag)].append(v)
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

    lines = [f"Adversarial positions processed: {N}", ""]
    for sl in SLICES:
        for m in METRICS:
            lines.append(f"=== slice={sl}  metric={m} ===")
            arr_by_lag = {lag: np.array(all_rhos[(sl, m, lag)]) for lag in LAGS}
            n = len(arr_by_lag[0])
            lines.append(f"  n positions with defined rho: {n}")
            if n == 0:
                lines.append("  (no data)"); lines.append(""); continue
            lines.append(f"  Lag |  median rho  |   IQR (p25..p75)")
            for lag in LAGS:
                a = arr_by_lag[lag]
                if len(a) < 2:
                    lines.append(f"   {lag:+d} |  (insufficient)")
                    continue
                lines.append(f"   {lag:+d} |  {np.median(a):+7.4f}   |  "
                              f"{np.percentile(a, 25):+7.4f} .. "
                              f"{np.percentile(a, 75):+7.4f}")
            # Asymmetry: median per-position of rho(+1) - rho(-1)
            pos_ids = [i for i in range(len(all_rhos[(sl, m, 1)]))
                       if i < len(all_rhos[(sl, m, -1)])]
            # Align the two lag arrays (same order of positions since
            # both use the same filtering path)
            r_plus  = np.array(all_rhos[(sl, m, 1)])
            r_minus = np.array(all_rhos[(sl, m, -1)])
            # Restrict to overlapping length (they should be the same)
            L_ = min(len(r_plus), len(r_minus))
            if L_ >= 2:
                diffs = r_plus[:L_] - r_minus[:L_]
                lines.append(f"  Asymmetry S = median [rho(+1) - rho(-1)]: "
                              f"{np.median(diffs):+7.4f}   "
                              f"(mean {diffs.mean():+.4f}, "
                              f"IQR {np.percentile(diffs, 25):+.4f} .. "
                              f"{np.percentile(diffs, 75):+.4f})")
                lines.append(f"  S > 0 => bias-first (P(C) predicts future "
                              f"corruption)")
                lines.append(f"  S < 0 => corruption-first")
            lines.append("")

    with open(args.output_summary, 'w') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote summary to {args.output_summary}")


if __name__ == '__main__':
    main()
