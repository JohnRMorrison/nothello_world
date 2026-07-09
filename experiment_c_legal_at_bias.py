"""For each adversarial position, at the first same-parity turn where
P(C) > threshold and C is illegal, ask: is C legal under the probe's
decoded board at that turn?

  Fraction C legal per probe at t_bias == world-model / corruption story
  Fraction C illegal per probe at t_bias == bias / readout story

100% coverage (no critical-cell or quiet-turn filters).  Sweeps over
thresholds {1%, 3%, 5%}.
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
    """Nanda-mode probe argmax -> 8x8 state (1=black, -1=white, 0=empty)."""
    mode = 0 if turn % 2 == 1 else 1
    W = probe[mode]
    h = torch.from_numpy(hidden_512).float()
    logits = torch.einsum('d,drco->rco', h, W)
    cls = logits.argmax(dim=-1).detach().cpu().numpy()
    st = np.zeros((8, 8), dtype=np.int8)
    st[cls == 1] = -1
    st[cls == 2] = 1
    return st


def process_position(game, T, C, model, probe, layer, block_size,
                      pos_to_token, device, thresholds):
    """Return dict: {threshold -> {'t_bias': int or None, 'c_legal_per_probe': bool}}."""
    L = min(T + 1, block_size)
    tokens = tokenize_games([game[:L]], seq_len=block_size).to(device)
    with torch.no_grad():
        logits, _ = model(tokens)
        acts = extract_activations(model, tokens, layer)
    acts_np = acts.cpu().numpy()[0]

    board = OthelloBoardState()
    legal_at = {}
    state_at = {}
    for t in range(0, T + 1):
        try:
            board.umpire(game[t])
        except Exception:
            return None
        legal_at[t] = set(board.get_valid_moves())
        state_at[t] = np.asarray(board.state, dtype=np.int8).copy()

    target_parity = T % 2
    parity_turns = [t for t in range(0, T + 1) if t % 2 == target_parity]

    # Precompute P(C, t) at each parity turn
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

    color_next = next_hand_color_at_turn(T)  # same for all C-parity turns

    result = {}
    for th in thresholds:
        t_bias = None
        for t in parity_turns:
            if (C not in legal_at[t]) and p_C_at[t] > th:
                t_bias = t
                break
        if t_bias is None:
            result[th] = {'t_bias': None, 'c_legal_per_probe': None}
            continue
        probe_state = probe_argmax_state(acts_np[t_bias], t_bias, probe)
        # Note: color_next at t_bias = color_next at T (same parity)
        flank_dirs = flank_providing_directions(probe_state, C, color_next)
        result[th] = {
            't_bias': int(t_bias),
            'c_legal_per_probe': len(flank_dirs) > 0,
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
    ap.add_argument('--output-csv', default='c_legal_at_bias.csv')
    ap.add_argument('--output-summary', default='c_legal_at_bias.txt')
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

    probe = torch.load(args.probe, map_location='cpu')
    assert probe.shape == (3, 512, 8, 8, 3)

    pos_to_token = build_pos_to_token(block_size)

    records_path = os.path.join(args.adversarial_dir, 'adversarial_records.npz')
    d = np.load(records_path, allow_pickle=True)
    games = d['games']; turns = d['turns']; cells = d['illegal_cells']
    N = len(games) if args.limit == 0 else min(args.limit, len(games))
    print(f"Processing {N} positions")

    counts = {th: {'no_bias': 0, 'c_legal': 0, 'c_illegal': 0, 'total': 0}
              for th in thresholds}
    csv_rows = [['idx', 'T', 'C'] + [f'th{th}_c_legal' for th in thresholds]]

    t0 = time.time()
    for i in range(N):
        game = list(games[i]); T = int(turns[i]); C = int(cells[i])
        res = process_position(game, T, C, model, probe,
                                args.layer, block_size, pos_to_token,
                                device, thresholds)
        row = [i, T, C]
        for th in thresholds:
            counts[th]['total'] += 1
            if res is None or res[th]['t_bias'] is None:
                counts[th]['no_bias'] += 1
                row.append('')
                continue
            if res[th]['c_legal_per_probe']:
                counts[th]['c_legal'] += 1
                row.append('1')
            else:
                counts[th]['c_illegal'] += 1
                row.append('0')
        csv_rows.append(row)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (N - i - 1) / (i + 1)
            print(f"  {i+1}/{N}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                   flush=True)

    with open(args.output_csv, 'w') as f:
        for row in csv_rows:
            f.write(','.join(str(x) for x in row) + '\n')
    print(f"Wrote per-position CSV to {args.output_csv}")

    lines = [f"Adversarial positions processed: {N}", ""]
    for th in thresholds:
        c = counts[th]
        total = c['total']
        classified = c['c_legal'] + c['c_illegal']
        lines.append(f"=== Threshold P(C) > {th * 100:.0f}% ===")
        lines.append(f"  Total positions:              {total}")
        lines.append(f"  No bias turn:                 {c['no_bias']}  "
                     f"({c['no_bias'] / total * 100:5.1f}%)")
        lines.append(f"  Classified positions:         {classified}  "
                     f"({classified / total * 100:5.1f}%)")
        if classified > 0:
            frac_legal = c['c_legal'] / classified
            frac_illegal = c['c_illegal'] / classified
            lines.append(f"    C LEGAL per probe (corruption/world-model): "
                          f"{c['c_legal']}  ({frac_legal * 100:5.1f}%)")
            lines.append(f"    C ILLEGAL per probe (bias/readout):         "
                          f"{c['c_illegal']}  ({frac_illegal * 100:5.1f}%)")
        lines.append("")

    with open(args.output_summary, 'w') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote summary to {args.output_summary}")


if __name__ == '__main__':
    main()
