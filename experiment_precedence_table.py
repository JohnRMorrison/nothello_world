"""Precedence table: for each of the first 5 adversarial games (shuffle seed 0),
identify the critical probe-error cells at the error turn T, then trace the
last (delta+1) turns of:
  - P(C, t): OGPT's probability on the illegal cell C at turn t
  - margin(k, t): probe logit margin at critical cell k, per turn
                  = logit[gt_class] - max(logit[other 2 classes])
                  > 0 = probe correct;  < 0 = probe confidently wrong.

Prints one table per game.
"""
import argparse
import os
import sys

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
from experiment_probe_on_adversarial import state_to_gt
from experiment_probe_causal_analysis import (
    probe_to_nanda_state, next_hand_color_at_turn,
    flank_providing_directions, critical_errors_for_direction,
)


CENTER_CELLS = {27, 28, 35, 36}
VALID_MOVES = [i for i in range(64) if i not in CENTER_CELLS]


def alg(cell):
    return f"{'abcdefgh'[cell % 8]}{cell // 8 + 1}"


def build_pos_to_token(block_size):
    dummy_game = list(VALID_MOVES)
    toks = tokenize_games([dummy_game], seq_len=block_size)[0].tolist()
    pos_to_token = np.full(64, -1, dtype=np.int64)
    for i, m in enumerate(dummy_game):
        if i < len(toks):
            pos_to_token[m] = toks[i]
    return pos_to_token


def gt_probe_class_at_cell(state_8x8, cell):
    """Nanda-state -> probe class {0=empty, 1=WHITE, 2=BLACK}."""
    r, c = cell // 8, cell % 8
    v = state_8x8[r, c]
    if v == 0: return 0
    if v == -1: return 1
    return 2


def probe_logits_at_cell(hidden_512, turn, probe, cell):
    """3 logits at the cell under the correct mode."""
    mode = 0 if turn % 2 == 1 else 1
    W = probe[mode]  # (512, 8, 8, 3)
    r, c = cell // 8, cell % 8
    h = torch.from_numpy(hidden_512).float()
    return torch.einsum('d,do->o', h, W[:, r, c, :]).numpy()


def probe_argmax_state(hidden_512, turn, probe):
    """Full 8x8 argmax decode -> nanda state."""
    mode = 0 if turn % 2 == 1 else 1
    W = probe[mode]
    h = torch.from_numpy(hidden_512).float()
    logits = torch.einsum('d,drco->rco', h, W)  # (8, 8, 3)
    cls = logits.argmax(dim=-1).numpy()
    st = np.zeros((8, 8), dtype=np.int8)
    st[cls == 1] = -1
    st[cls == 2] = 1
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_by_depth')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe',
                    default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--n-games', type=int, default=5)
    ap.add_argument('--delta', type=int, default=5,
                    help="Number of C's-parity turns to look back before the "
                         "error turn.  delta=5 -> 6 rows covering turns "
                         "[T-10, T-8, ..., T].")
    ap.add_argument('--output-txt', default='precedence_tables.txt')
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
    assert probe.shape == (3, 512, 8, 8, 3)

    pos_to_token = build_pos_to_token(block_size)

    records_path = os.path.join(args.adversarial_dir, 'adversarial_records.npz')
    d = np.load(records_path, allow_pickle=True)
    games = d['games']; turns = d['turns']; cells = d['illegal_cells']
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(games))[:args.n_games]

    lines = []

    for gi, i in enumerate(idx):
        game = list(games[i])
        T = int(turns[i])          # error turn
        C = int(cells[i])
        color_next = next_hand_color_at_turn(T)

        # Feed tokens through error_turn; grab logits + layer-K activations
        L = min(T + 1, block_size)
        tokens = tokenize_games([game[:L]], seq_len=block_size).to(device)
        with torch.no_grad():
            logits, _ = model(tokens)                             # (1, block_size, V)
            acts = extract_activations(model, tokens, args.layer)  # (1, block_size, 512)
        acts_np = acts.cpu().numpy()[0]                           # (block_size, 512)

        # Play the game & cache per-turn ground-truth state
        board = OthelloBoardState()
        state_at_turn = {}
        legal_at_turn = {}
        for t in range(0, T + 1):
            board.umpire(game[t])
            state_at_turn[t] = np.asarray(board.state, dtype=np.int8).copy()
            legal_at_turn[t] = set(board.get_valid_moves())

        # --- Identify critical cells at error turn T ---
        probe_state_T = probe_argmax_state(acts_np[T], T, probe)
        gt_state_T = state_at_turn[T]
        flank_dirs = flank_providing_directions(probe_state_T, C, color_next)
        crit_cells = set()
        for dr in flank_dirs:
            crit_cells.update(critical_errors_for_direction(
                probe_state_T, gt_state_T, C, dr, color_next))
        crit_cells = sorted(crit_cells)

        header = (
            f"\n{'=' * 76}\n"
            f"GAME {gi + 1} / {args.n_games}    "
            f"error turn T = {T}    "
            f"illegal cell C = {alg(C)}    "
            f"to move at T: {'BLACK' if color_next == 1 else 'WHITE'}\n"
            f"Flank-providing directions at T (per probe board): {len(flank_dirs)}\n"
            f"Critical cells at T: "
            f"{', '.join(alg(k) for k in crit_cells) if crit_cells else '(none)'}\n"
            f"{'=' * 76}"
        )
        lines.append(header)

        if not crit_cells:
            lines.append("  (no critical cells — no hallucinated flank; skipping)")
            continue

        # --- Table header (rows are C's-parity turns only) ---
        col_hdr = ("  turn |   P(C)   |  probe-error cells (any)"
                   "                       |  "
                   + " | ".join(f"m({alg(k)})" for k in crit_cells))
        lines.append(col_hdr)
        lines.append("  " + "-" * (len(col_hdr) - 2))

        # --- Rows: last delta+1 turns on C's-player's parity ---
        target_parity = T % 2
        c_parity_turns = [t for t in range(0, T + 1) if t % 2 == target_parity]
        c_parity_turns = c_parity_turns[-(args.delta + 1):]
        for t in c_parity_turns:
            # P(C) at turn t: OGPT's prob at position t (over 60 valid cells)
            probs = F.softmax(logits[0, t, :], dim=-1).cpu().numpy()
            probs_60 = np.zeros(60, dtype=np.float32)
            for k, m in enumerate(VALID_MOVES):
                tok = int(pos_to_token[m])
                if tok >= 0:
                    probs_60[k] = probs[tok]
            probs_60 = probs_60 / max(probs_60.sum(), 1e-9)
            p_C = float(probs_60[VALID_MOVES.index(C)])

            # Probe errors: cells where probe argmax != actual
            probe_state_t = probe_argmax_state(acts_np[t], t, probe)
            err_cells = []
            for cell in range(64):
                r_, c_ = cell // 8, cell % 8
                if probe_state_t[r_, c_] != state_at_turn[t][r_, c_]:
                    err_cells.append(cell)
            err_str = ",".join(alg(k) for k in err_cells) if err_cells else "-"

            # Logit margins for each critical cell
            margins = []
            for k in crit_cells:
                lg = probe_logits_at_cell(acts_np[t], t, probe, k)   # (3,)
                gt_cls = gt_probe_class_at_cell(state_at_turn[t], k)
                other = [lg[j] for j in range(3) if j != gt_cls]
                margins.append(float(lg[gt_cls] - max(other)))

            margin_strs = " | ".join(f"{m:+6.2f}" for m in margins)
            lines.append(f"   {t:>3d} | {p_C:7.4f}  |  "
                         f"{err_str:<45s}|  {margin_strs}")

    with open(args.output_txt, 'w') as f:
        f.write("\n".join(lines))
    print(f"Wrote precedence tables to {args.output_txt}")
    print("\n".join(lines))


if __name__ == '__main__':
    main()
