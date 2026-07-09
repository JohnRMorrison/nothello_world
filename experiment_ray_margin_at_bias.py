"""Probe logit margin on ALL of C's 8 rays at t_bias.

For each adversarial position (game, T, C):
  1. Identify the final illegal episode -- contiguous run of same-parity
     turns ending at T where C is actually illegal (same as
     experiment_within_episode_precedence.py).
  2. Within the episode, t_bias = first turn where P(C) > threshold.
  3. At t_bias, walk each of C's 8 rays (up/down/left/right/diagonals).
     For every cell along every ray, compute the probe logit margin
     (gt-class logit - max other-class logit) using probe mode 0/1
     chosen by t_bias parity.
  4. min_ray_margin = min over all cells on all rays.
       > 0  -> probe correctly represents EVERY cell that could
               contribute to a flank from C
       > +1 -> probe confidently correct on every such cell
       <= 0 -> at least one cell wrong

Sweeps thresholds {1%, 3%, 5%}.  Reports:
  - Fraction of qualified positions with min_ray_margin > 0
  - Fraction with min_ray_margin > +1
  - Per-position CSV with (idx, T, C, threshold, t_bias, min_ray_margin)
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


def compute_ray_min_margin(hidden_512, turn, probe, C, state):
    """Return min over all cells on C's 8 rays of the probe logit margin."""
    mode = 0 if turn % 2 == 1 else 1
    W = probe[mode]                                          # (512, 8, 8, 3)
    h = torch.from_numpy(hidden_512).float()
    # Compute all 64-cell logits at once, then index into ray cells.
    logits = torch.einsum('d,drco->rco', h, W).detach().cpu().numpy()  # (8,8,3)

    min_margin = np.inf
    for direction in DIRS:
        for cell in ray_cells_in_direction(C, direction):
            r_, c_ = cell // 8, cell % 8
            lg = logits[r_, c_]
            gt_cls = gt_probe_class(state, cell)
            other = [lg[j] for j in range(3) if j != gt_cls]
            m = float(lg[gt_cls] - max(other))
            if m < min_margin:
                min_margin = m
    return min_margin


def process_position(game, T, C, model, probe, layer, block_size,
                       pos_to_token, device, thresholds):
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

    # P(C, t) on same-parity turns
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

    # Final illegal episode: contiguous illegal same-parity turns ending at T
    episode = []
    for t in reversed(parity_turns):
        if C in legal_at[t]:
            break
        episode.append(t)
    episode = list(reversed(episode))

    result = {}
    for th in thresholds:
        t_bias = None
        for t in episode:
            if p_C_at[t] > th:
                t_bias = t
                break
        if t_bias is None:
            result[th] = {'t_bias': -1, 'min_ray_margin': None}
            continue
        m = compute_ray_min_margin(acts_np[t_bias], t_bias, probe, C,
                                     state_at[t_bias])
        result[th] = {'t_bias': int(t_bias), 'min_ray_margin': m}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_by_depth')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe',
                    default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--thresholds', type=str, default='0.01,0.03,0.05')
    ap.add_argument('--output-csv', default='ray_margin.csv')
    ap.add_argument('--output-summary', default='ray_margin.txt')
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

    counts = {th: {'qualified': 0, 'no_bias': 0,
                    'min_gt_0': 0, 'min_gt_1': 0,
                    'margins': []}
              for th in thresholds}

    csv_rows = [['game_id', 'T', 'C', 'threshold', 't_bias', 'min_ray_margin']]

    t0 = time.time()
    for i in range(N):
        game = list(games[i]); T = int(turns[i]); C = int(cells[i])
        res = process_position(game, T, C, model, probe, args.layer,
                                 block_size, pos_to_token, device, thresholds)
        for th in thresholds:
            if res is None or res[th]['t_bias'] == -1:
                counts[th]['no_bias'] += 1
                csv_rows.append([i, T, C, th, '', ''])
                continue
            counts[th]['qualified'] += 1
            m = res[th]['min_ray_margin']
            counts[th]['margins'].append(m)
            if m > 0:
                counts[th]['min_gt_0'] += 1
            if m > 1:
                counts[th]['min_gt_1'] += 1
            csv_rows.append([i, T, C, th, res[th]['t_bias'], f"{m:.4f}"])

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
    for th in thresholds:
        c = counts[th]
        total = c['qualified'] + c['no_bias']
        qual = c['qualified']
        lines.append(f"=== Threshold P(C) > {th * 100:.2f}% ===")
        lines.append(f"  Total positions:              {total}")
        lines.append(f"  Qualified (t_bias exists):    {qual}  "
                      f"({qual / total * 100:5.1f}%)")
        lines.append(f"  No bias turn in episode:      {c['no_bias']}  "
                      f"({c['no_bias'] / total * 100:5.1f}%)")
        if qual > 0:
            lines.append(f"  Fraction min_ray_margin > 0:  "
                          f"{c['min_gt_0']}  ({c['min_gt_0'] / qual * 100:5.1f}%)")
            lines.append(f"  Fraction min_ray_margin > +1: "
                          f"{c['min_gt_1']}  ({c['min_gt_1'] / qual * 100:5.1f}%)")
            arr = np.array(c['margins'])
            lines.append(f"  Distribution of min_ray_margin:")
            lines.append(f"    median={np.median(arr):+.3f}  mean={arr.mean():+.3f}")
            for q in [10, 25, 50, 75, 90]:
                lines.append(f"    p{q}={np.percentile(arr, q):+.3f}")
        lines.append("")

    with open(args.output_summary, 'w') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote summary to {args.output_summary}")


if __name__ == '__main__':
    main()
