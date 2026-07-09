"""2x2 classification of adversarial positions.

Two orthogonal axes at a chosen P(C) threshold:

  Sticky axis: Was C legal at the immediately preceding same-parity
               turn where P(C) > threshold?
    - Yes: sticky candidate (OGPT had elevated P(C) during a nearby
           legal window)
    - No: not-sticky (either no prior elevated turn, or the prior
          elevated turn was already illegal)

  Corruption axis: Is C legal under the probe's decoded board at t_bias?
    - Yes: corruption (world model rationalises C)
    - No: not-corruption (world model correctly says C is illegal)

Produces a 2x2 count table per threshold, over the full 23k positions.
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


def classify_position(game, T, C, model, probe, layer, block_size,
                       pos_to_token, device, thresholds):
    L = min(T + 1, block_size)
    tokens = tokenize_games([game[:L]], seq_len=block_size).to(device)
    with torch.no_grad():
        logits, _ = model(tokens)
        acts = extract_activations(model, tokens, layer)
    acts_np = acts.cpu().numpy()[0]

    board = OthelloBoardState()
    legal_at = {}
    for t in range(0, T + 1):
        try:
            board.umpire(game[t])
        except Exception:
            return None
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

    result = {}
    for th in thresholds:
        # t_bias: first illegal same-parity turn with P > threshold
        t_bias = None
        for t in parity_turns:
            if (C not in legal_at[t]) and p_C_at[t] > th:
                t_bias = t
                break
        if t_bias is None:
            result[th] = {'category': 'no_bias_turn'}
            continue

        # Sticky axis: the immediately preceding same-parity turn where
        # P(C) > threshold.  Walk backward from t_bias, take the first
        # such turn.
        t_prev = None
        for t in reversed(parity_turns):
            if t >= t_bias:
                continue
            if p_C_at[t] > th:
                t_prev = t
                break

        if t_prev is None:
            sticky = False
        else:
            sticky = (C in legal_at[t_prev])

        # Corruption axis: is C legal under probe at t_bias?
        probe_state = probe_argmax_state(acts_np[t_bias], t_bias, probe)
        flank_dirs = flank_providing_directions(probe_state, C, color_next)
        corrupt = len(flank_dirs) > 0

        if sticky and corrupt:
            cat = 'sticky_corrupt'
        elif sticky and not corrupt:
            cat = 'sticky_notcorrupt'
        elif not sticky and corrupt:
            cat = 'notsticky_corrupt'
        else:
            cat = 'notsticky_notcorrupt'

        result[th] = {
            'category': cat,
            't_bias': int(t_bias),
            't_prev': int(t_prev) if t_prev is not None else -1,
            'p_bias': p_C_at[t_bias],
            'p_prev': p_C_at[t_prev] if t_prev is not None else 0.0,
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
                    default='0.0005,0.001,0.003,0.005,0.01,0.03,0.05')
    ap.add_argument('--output-csv', default='two_by_two.csv')
    ap.add_argument('--output-summary', default='two_by_two.txt')
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

    categories = ['sticky_corrupt', 'sticky_notcorrupt',
                   'notsticky_corrupt', 'notsticky_notcorrupt',
                   'no_bias_turn']
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

    lines = [f"Adversarial positions processed: {N}", ""]
    for th in thresholds:
        c = counts[th]
        total = sum(c.values())
        classified = total - c['no_bias_turn']
        lines.append(f"=== Threshold P(C) > {th * 100:.2f}% ===")
        lines.append(f"  Total:              {total}")
        lines.append(f"  No bias turn:       {c['no_bias_turn']}  "
                     f"({c['no_bias_turn'] / total * 100:5.1f}%)")
        lines.append(f"  Classified:         {classified}  "
                     f"({classified / total * 100:5.1f}%)")
        if classified > 0:
            def pct(k):
                return c[k] / classified * 100
            lines.append("")
            lines.append("  2x2 table (% of classified):")
            lines.append(f"                           |  CORRUPT     |  NOT CORRUPT")
            lines.append(f"    -----------------------+--------------+-----------------")
            lines.append(f"    STICKY (prev L, P>th)  |  {pct('sticky_corrupt'):>5.1f}%      |  {pct('sticky_notcorrupt'):>5.1f}%")
            lines.append(f"    NOT STICKY             |  {pct('notsticky_corrupt'):>5.1f}%      |  {pct('notsticky_notcorrupt'):>5.1f}%")
            lines.append("")
            lines.append(f"    Row sums:  sticky = {pct('sticky_corrupt') + pct('sticky_notcorrupt'):.1f}%  |  "
                          f"not-sticky = {pct('notsticky_corrupt') + pct('notsticky_notcorrupt'):.1f}%")
            lines.append(f"    Col sums:  corrupt = {pct('sticky_corrupt') + pct('notsticky_corrupt'):.1f}%  |  "
                          f"not-corrupt = {pct('sticky_notcorrupt') + pct('notsticky_notcorrupt'):.1f}%")
        lines.append("")

    with open(args.output_summary, 'w') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote summary to {args.output_summary}")


if __name__ == '__main__':
    main()
