"""Bias-corruption precedence within the final illegal episode.

For each adversarial position (game, T, C):
  1. Identify the "final illegal episode" -- the contiguous run of same-
     parity turns ending at T where C is actually illegal.
  2. Within that episode:
       t_bias    = first turn where P(C) > threshold
       t_corrupt = first turn where C is legal under the probe's
                   decoded board (any flanking direction)
  3. Classify:
       bias-first        : t_bias < t_corrupt, or t_corrupt undefined
       simultaneous      : t_bias == t_corrupt
       corruption-first  : t_corrupt < t_bias

Also reports the distribution of (t_corrupt - t_bias) in same-parity
turn units, plus episode length as a diagnostic (short episodes force
simultaneous classification).
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

    # Identify final illegal episode.  Walk BACKWARD from T on same-parity
    # turns until we hit a legal turn (or run out of turns).  Episode is
    # the contiguous run of illegal turns ending at T.
    episode = []
    for t in reversed(parity_turns):
        if C in legal_at[t]:
            break
        episode.append(t)
    episode = list(reversed(episode))  # chronological order

    if len(episode) == 0:
        # Shouldn't happen -- T should be illegal by construction of
        # adversarial records.
        return None

    # For each turn in the episode, compute whether C is legal per probe.
    # (Only decode as needed for the first-corrupt check.)
    result = {}
    # Precompute c-legal-per-probe once per episode turn (used across all
    # thresholds).
    c_legal_probe = {}
    for t in episode:
        probe_state = probe_argmax_state(acts_np[t], t, probe)
        c_legal_probe[t] = len(flank_providing_directions(
            probe_state, C, color_next)) > 0

    # First turn where c_legal_probe -- constant across thresholds.
    t_corrupt = None
    for t in episode:
        if c_legal_probe[t]:
            t_corrupt = t
            break

    for th in thresholds:
        # First turn in episode where P(C) > threshold.
        t_bias = None
        for t in episode:
            if p_C_at[t] > th:
                t_bias = t
                break

        if t_bias is None:
            result[th] = {'category': 'no_bias_in_episode',
                          'episode_len': len(episode)}
            continue

        if t_corrupt is None:
            cat = 'bias_first'
            gap = None
        else:
            if t_bias < t_corrupt:
                cat = 'bias_first'
            elif t_bias == t_corrupt:
                cat = 'simultaneous'
            else:
                cat = 'corruption_first'
            gap = (t_corrupt - t_bias) // 2   # same-parity turns

        result[th] = {
            'category': cat,
            'episode_len': len(episode),
            't_bias': int(t_bias),
            't_corrupt': int(t_corrupt) if t_corrupt is not None else -1,
            'gap_parity_turns': gap,
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
    ap.add_argument('--thresholds', type=str, default='0.01,0.03,0.05')
    ap.add_argument('--output-csv', default='within_episode.csv')
    ap.add_argument('--output-summary', default='within_episode.txt')
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

    cats = ['bias_first', 'simultaneous', 'corruption_first',
             'no_bias_in_episode']
    counts = {th: {c: 0 for c in cats} for th in thresholds}
    gaps  = {th: [] for th in thresholds}    # gap_parity_turns
    episode_lens = []

    csv_rows = [['idx', 'T', 'C', 'episode_len'] +
                sum([[f'th{th}_cat', f'th{th}_gap'] for th in thresholds], [])]

    t0 = time.time()
    for i in range(N):
        game = list(games[i]); T = int(turns[i]); C = int(cells[i])
        res = classify_position(game, T, C, model, probe, args.layer,
                                 block_size, pos_to_token, device, thresholds)
        row = [i, T, C, res[thresholds[0]]['episode_len'] if res else 0]
        if res is not None:
            episode_lens.append(res[thresholds[0]]['episode_len'])
        for th in thresholds:
            if res is None:
                row.extend(['', ''])
                continue
            r = res[th]
            counts[th][r['category']] += 1
            if r.get('gap_parity_turns') is not None:
                gaps[th].append(r['gap_parity_turns'])
            row.append(r['category'])
            row.append(r.get('gap_parity_turns', ''))
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
    lines.append("Episode length distribution (same-parity turns):")
    if episode_lens:
        el = np.array(episode_lens)
        lines.append(f"  median={np.median(el):.1f}  mean={el.mean():.2f}  "
                      f"min={el.min()}  max={el.max()}")
        for L_ in [1, 2, 3, 4, 5]:
            frac = (el == L_).sum() / len(el) * 100
            lines.append(f"  length {L_}: {(el == L_).sum()} "
                          f"({frac:.1f}%)")
        frac_ge6 = (el >= 6).sum() / len(el) * 100
        lines.append(f"  length >=6: {(el >= 6).sum()} ({frac_ge6:.1f}%)")
    lines.append("")

    for th in thresholds:
        c = counts[th]
        total = sum(c.values())
        classified = total - c['no_bias_in_episode']
        lines.append(f"=== Threshold P(C) > {th * 100:.2f}% ===")
        lines.append(f"  Total:                {total}")
        lines.append(f"  No bias in episode:   {c['no_bias_in_episode']}  "
                      f"({c['no_bias_in_episode'] / total * 100:5.1f}%)")
        lines.append(f"  Classified:           {classified}  "
                      f"({classified / total * 100:5.1f}%)")
        if classified > 0:
            def pct(k): return c[k] / classified * 100
            lines.append(f"    bias-first:         {c['bias_first']}  ({pct('bias_first'):5.1f}%)")
            lines.append(f"    simultaneous:       {c['simultaneous']}  ({pct('simultaneous'):5.1f}%)")
            lines.append(f"    corruption-first:   {c['corruption_first']}  ({pct('corruption_first'):5.1f}%)")
        g = gaps[th]
        if g:
            g = np.array(g)
            lines.append(f"  Gap (t_corrupt - t_bias) in same-parity turns:")
            lines.append(f"    median={np.median(g):+.1f}  mean={g.mean():+.2f}")
            for q in [10, 25, 50, 75, 90]:
                lines.append(f"    p{q}={np.percentile(g, q):+.1f}")
        lines.append("")

    with open(args.output_summary, 'w') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote summary to {args.output_summary}")


if __name__ == '__main__':
    main()
