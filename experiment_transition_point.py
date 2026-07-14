"""Transition-point analysis: P(C) and board corruption B(C) at the
first illegal same-parity turn of the final illegal episode, compared to
the immediately-preceding legal same-parity turn.

For each adversarial position (game, T, C):
  1. Walk C's-parity turns.  Find the final illegal episode = contiguous
     run of illegal same-parity turns ending at T.
  2. t_transition = first turn of that episode.
  3. t_L         = t_transition - 2 (immediately preceding same-parity
     turn; must be legal since it's just outside the final illegal
     episode).  Skip positions where t_L < 0.
  4. Record at each turn: P(C, t), B_margin(C, t) = mean over C's 8-ray
     cells of -logit_margin, B_loss(C, t) = mean over ray cells of the
     probe CE loss -log p(true).  Higher = more corrupt.

Reports:
  - Distributions of P_I, P_L, B_I, B_L
  - Persistence ratio P_I / P_L
  - Quadrant counts on the P_I vs B_I scatter (using medians as
    split points)
  - Correlation between P_I and B_I across positions

Per-position CSV: idx, T, C, t_L, t_transition, P_L, P_I, B_margin_L,
B_margin_I, B_loss_L, B_loss_I.
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
    next_hand_color_at_turn, DIRS, ray_cells_in_direction,
    flank_providing_directions, critical_errors_for_direction,
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
    v = state_8x8[cell // 8, cell % 8]
    if v == 0: return 0
    if v == -1: return 1
    return 2


def compute_ray_corruption(hidden_512, turn, probe, C, state, ray_cells):
    """Return (mean(-margin), mean(CE loss)) over C's 8 rays.
    Higher = more corrupt for both."""
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
        s = lg - lg.max()
        exp_s = np.exp(s)
        p = exp_s / exp_s.sum()
        other = [lg[j] for j in range(3) if j != gt_cls]
        margins.append(float(lg[gt_cls] - max(other)))
        losses.append(float(-np.log(p[gt_cls] + 1e-12)))
    return -float(np.mean(margins)), float(np.mean(losses))


def compute_single_cell_corruption(hidden_512, turn, probe, cell, state):
    """Return (-margin, CE loss) at a specific cell."""
    mode = 0 if turn % 2 == 1 else 1
    W = probe[mode]
    r_, c_ = cell // 8, cell % 8
    h = torch.from_numpy(hidden_512).float()
    lg = torch.einsum('d,do->o', h, W[:, r_, c_, :]).detach().cpu().numpy()
    gt_cls = gt_probe_class(state, cell)
    s = lg - lg.max()
    exp_s = np.exp(s)
    p = exp_s / exp_s.sum()
    other = [lg[j] for j in range(3) if j != gt_cls]
    margin = float(lg[gt_cls] - max(other))
    loss = float(-np.log(p[gt_cls] + 1e-12))
    return -margin, loss


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

    target_parity = T % 2
    parity_turns = [t for t in range(0, T + 1) if t % 2 == target_parity]

    # Final illegal episode: walk backward from T until we hit a legal turn
    episode = []
    for t in reversed(parity_turns):
        if C in legal_at[t]:
            break
        episode.append(t)
    episode = list(reversed(episode))
    if not episode:
        return None
    t_transition = episode[0]
    t_L = t_transition - 2
    if t_L < 0 or t_L not in legal_at or C not in legal_at[t_L]:
        # Either the episode starts at the very beginning of parity turns
        # (no legal predecessor exists) or an earlier same-parity turn
        # was also illegal (episode wasn't preceded by a legal turn).
        return None

    C_idx60 = VALID_MOVES.index(C)

    def p_C_at(t):
        probs = F.softmax(logits[0, t, :], dim=-1).detach().cpu().numpy()
        p60 = np.zeros(60, dtype=np.float32)
        for k, m in enumerate(VALID_MOVES):
            tok = int(pos_to_token[m])
            if tok >= 0:
                p60[k] = probs[tok]
        p60 = p60 / max(p60.sum(), 1e-9)
        return float(p60[C_idx60])

    ray_cells = []
    for d in DIRS:
        ray_cells.extend(ray_cells_in_direction(C, d))

    P_L = p_C_at(t_L)
    P_I = p_C_at(t_transition)
    B_margin_L, B_loss_L = compute_ray_corruption(
        acts_np[t_L], t_L, probe, C, state_at[t_L], ray_cells)
    B_margin_I, B_loss_I = compute_ray_corruption(
        acts_np[t_transition], t_transition, probe, C,
        state_at[t_transition], ray_cells)

    # Critical cell: the min-margin cell at T among cells that break the
    # hallucinated flank when reverted.  May not exist for all positions.
    color_next = next_hand_color_at_turn(T)
    mode_T = 0 if T % 2 == 1 else 1
    hT = torch.from_numpy(acts_np[T]).float()
    logits_T = torch.einsum('d,drco->rco', hT, probe[mode_T]).detach().cpu().numpy()
    cls_T = logits_T.argmax(axis=-1)
    probe_state_T = np.zeros((8, 8), dtype=np.int8)
    probe_state_T[cls_T == 1] = -1
    probe_state_T[cls_T == 2] = 1
    flank_dirs = flank_providing_directions(probe_state_T, C, color_next)
    crit_set = set()
    for dr in flank_dirs:
        crit_set.update(critical_errors_for_direction(
            probe_state_T, state_at[T], C, dr, color_next))

    critical_cell = None
    crit_margin_L = crit_loss_L = None
    crit_margin_I = crit_loss_I = None
    if crit_set:
        worst_m, worst = np.inf, None
        for k in crit_set:
            r_, c_ = k // 8, k % 8
            lg = logits_T[r_, c_]
            gt = gt_probe_class(state_at[T], k)
            other = [lg[j] for j in range(3) if j != gt]
            m = float(lg[gt] - max(other))
            if m < worst_m:
                worst_m = m
                worst = int(k)
        critical_cell = worst
        crit_margin_L, crit_loss_L = compute_single_cell_corruption(
            acts_np[t_L], t_L, probe, critical_cell, state_at[t_L])
        crit_margin_I, crit_loss_I = compute_single_cell_corruption(
            acts_np[t_transition], t_transition, probe, critical_cell,
            state_at[t_transition])

    return {
        't_L': t_L, 't_transition': t_transition,
        'P_L': P_L, 'P_I': P_I,
        'B_margin_L': B_margin_L, 'B_margin_I': B_margin_I,
        'B_loss_L': B_loss_L, 'B_loss_I': B_loss_I,
        'critical_cell': critical_cell,
        'B_margin_crit_L': crit_margin_L, 'B_margin_crit_I': crit_margin_I,
        'B_loss_crit_L': crit_loss_L,     'B_loss_crit_I': crit_loss_I,
    }


def _pct(k, tot):
    return f"{k / tot * 100:.1f}%" if tot > 0 else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_by_depth')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe',
                    default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--output-csv', default='transition_point.csv')
    ap.add_argument('--output-summary', default='transition_point.txt')
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

    header = ['idx', 'T', 'C', 't_L', 't_transition',
               'P_L', 'P_I', 'B_margin_L', 'B_margin_I',
               'B_loss_L', 'B_loss_I',
               'critical_cell',
               'B_margin_crit_L', 'B_margin_crit_I',
               'B_loss_crit_L', 'B_loss_crit_I']
    csv_rows = [header]
    skipped = 0

    t0 = time.time()
    for i in range(N):
        game = list(games[i]); T = int(turns[i]); C = int(cells[i])
        res = process_position(game, T, C, model, probe, args.layer,
                                 block_size, pos_to_token, device)
        if res is None:
            skipped += 1
            continue
        def _f(x, w=4):
            return f"{x:.{w}f}" if x is not None else ''
        csv_rows.append([
            i, T, C, res['t_L'], res['t_transition'],
            f"{res['P_L']:.6f}", f"{res['P_I']:.6f}",
            f"{res['B_margin_L']:.4f}", f"{res['B_margin_I']:.4f}",
            f"{res['B_loss_L']:.4f}", f"{res['B_loss_I']:.4f}",
            res['critical_cell'] if res['critical_cell'] is not None else '',
            _f(res['B_margin_crit_L']), _f(res['B_margin_crit_I']),
            _f(res['B_loss_crit_L']),   _f(res['B_loss_crit_I']),
        ])
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (N - i - 1) / (i + 1)
            print(f"  {i+1}/{N}  ({elapsed:.0f}s, ~{eta:.0f}s remaining)",
                   flush=True)

    with open(args.output_csv, 'w') as f:
        for row in csv_rows:
            f.write(','.join(str(x) for x in row) + '\n')
    print(f"Wrote per-position CSV to {args.output_csv}")

    # Load into arrays for summary stats
    data = np.array([
        [float(r[5]), float(r[6]), float(r[7]),
         float(r[8]), float(r[9]), float(r[10])]
        for r in csv_rows[1:]
    ])
    if len(data) == 0:
        print("No data to summarize.")
        return
    P_L = data[:, 0]; P_I = data[:, 1]
    Bm_L = data[:, 2]; Bm_I = data[:, 3]
    Bl_L = data[:, 4]; Bl_I = data[:, 5]

    lines = [f"Adversarial positions processed: {N}",
              f"Qualified (episode has preceding legal turn on parity): {len(data)}",
              f"Skipped: {skipped}",
              ""]

    def dist(name, arr):
        lines.append(f"  {name}:")
        lines.append(f"    median={np.median(arr):+.4f}  mean={arr.mean():+.4f}")
        for q in [10, 25, 50, 75, 90]:
            lines.append(f"    p{q}={np.percentile(arr, q):+.4f}")

    lines.append("=== At last-legal turn (t_L) ===")
    dist("P_L (should be moderate)", P_L)
    dist("B_margin_L (should be near zero -- probe clean)", Bm_L)
    dist("B_loss_L", Bl_L)
    lines.append("")
    lines.append("=== At first-illegal turn (t_transition) ===")
    dist("P_I (0 = correctly dropped, high = sticky)", P_I)
    dist("B_margin_I", Bm_I)
    dist("B_loss_I", Bl_I)
    lines.append("")

    # Persistence
    ratio = P_I / np.maximum(P_L, 1e-12)
    lines.append("=== Persistence P_I / P_L ===")
    dist("ratio", ratio)
    frac_stick_50 = float((ratio > 0.5).mean())
    frac_stick_25 = float((ratio > 0.25).mean())
    lines.append(f"  Fraction with P_I > 0.50 * P_L (sticky): {frac_stick_50 * 100:.1f}%")
    lines.append(f"  Fraction with P_I > 0.25 * P_L:         {frac_stick_25 * 100:.1f}%")
    lines.append("")

    # Quadrant analysis on the P_I vs B_loss_I scatter, split by median
    def quadrants(P_arr, B_arr, name_p, name_b):
        p_med = float(np.median(P_arr))
        b_med = float(np.median(B_arr))
        hp = P_arr >= p_med
        hb = B_arr >= b_med
        tot = len(P_arr)
        counts = {
            'HP_HB': int((hp & hb).sum()),
            'HP_LB': int((hp & ~hb).sum()),
            'LP_HB': int((~hp & hb).sum()),
            'LP_LB': int((~hp & ~hb).sum()),
        }
        lines.append(f"=== Quadrants: {name_p} vs {name_b} ===")
        lines.append(f"  Split at medians: {name_p}={p_med:.4f}  {name_b}={b_med:.4f}")
        lines.append(f"                   |  Low {name_b}  |  High {name_b}")
        lines.append(f"    High {name_p}     |  {counts['HP_LB']:>5} ({_pct(counts['HP_LB'], tot):>6})  |  "
                      f"{counts['HP_HB']:>5} ({_pct(counts['HP_HB'], tot):>6})")
        lines.append(f"    Low  {name_p}     |  {counts['LP_LB']:>5} ({_pct(counts['LP_LB'], tot):>6})  |  "
                      f"{counts['LP_HB']:>5} ({_pct(counts['LP_HB'], tot):>6})")
        rho_p = float(np.corrcoef(P_arr, B_arr)[0, 1])
        # Spearman via rank
        def _rank(a):
            order = a.argsort()
            r = np.empty_like(order, dtype=np.float64)
            r[order] = np.arange(len(a))
            return r
        rho_s = float(np.corrcoef(_rank(P_arr), _rank(B_arr))[0, 1])
        lines.append(f"  Correlation across positions: Pearson r={rho_p:+.4f}  "
                      f"Spearman rho={rho_s:+.4f}")
        lines.append("")
        return counts

    quadrants(P_I, Bm_I, 'P_I', 'B_margin_I')
    quadrants(P_I, Bl_I, 'P_I', 'B_loss_I')

    with open(args.output_summary, 'w') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote summary to {args.output_summary}")


if __name__ == '__main__':
    main()
