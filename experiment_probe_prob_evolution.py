"""Move-by-move probability heatmaps for the first N adversarial games.

For each adversarial (game, error_turn, C), feed the move sequence to
Othello-GPT one move at a time.  At each turn, print an 8x8 text heatmap
showing OGPT's probability distribution over cells, with:

  - 2-digit percentage per cell (0-99)
  - Suffix char per cell:
       X = cell has a BLACK piece (played)
       O = cell has a WHITE piece (played)
       ! = empty AND currently ILLEGAL to play there
       (space) = empty AND legal
       # = this is cell C (the eventually-illegal top-1 pick)

Also records per-turn stats to a companion CSV and generates a matplotlib
line plot of P(C) over turns for each game.

Usage:
    python experiment_probe_prob_evolution.py \\
        --adversarial-dir experiment1_by_depth \\
        --ckpt ckpts/gpt_nanda_synthetic.ckpt \\
        --n-games 5 \\
        --last-n-turns 15 \\
        --output-txt prob_evolution.txt \\
        --output-plot prob_evolution.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

sys.path.insert(0, '.')
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, VOCAB_SIZE,
)


CENTER_CELLS = {27, 28, 35, 36}
VALID_MOVES = [i for i in range(64) if i not in CENTER_CELLS]


def alg(cell):
    return f"{'abcdefgh'[cell % 8]}{cell // 8 + 1}"


def build_token_to_pos_and_pos_to_token(model_block_size):
    """Return (token_to_pos: (V,), pos_to_token: (64,) w/ -1 for invalid)."""
    dummy_game = list(VALID_MOVES)  # play each valid cell once
    toks = tokenize_games([dummy_game], seq_len=model_block_size)[0].tolist()
    token_to_pos = np.full(VOCAB_SIZE, -1, dtype=np.int64)
    pos_to_token = np.full(64, -1, dtype=np.int64)
    for i, m in enumerate(dummy_game):
        if i < len(toks):
            token_to_pos[toks[i]] = m
            pos_to_token[m] = toks[i]
    return token_to_pos, pos_to_token


def render_heatmap(probs_60, state_8x8, legal_set, C_cell):
    """Text heatmap.  probs_60: (60,) softmax marginal over valid cells.

    Returns list of strings (rows).
    """
    # Reindex to 8x8 (probs on invalid cells = 0)
    probs_64 = np.zeros(64, dtype=np.float32)
    for i, cell in enumerate(VALID_MOVES):
        probs_64[cell] = probs_60[i]

    rows = []
    header = "     " + "".join(f" {c}  " for c in "abcdefgh")
    rows.append(header)
    rows.append("    " + "-" * 34)
    for r in range(8):
        cells = []
        for c in range(8):
            idx = r * 8 + c
            p = probs_64[idx]
            pct = int(round(p * 100))
            pct = max(0, min(99, pct))
            # Determine suffix
            if idx == C_cell:
                suf = "#"
            else:
                v = state_8x8[r, c]
                if v == 1:
                    suf = "X"
                elif v == -1:
                    suf = "O"
                elif idx not in legal_set:
                    suf = "!"
                else:
                    suf = " "
            cells.append(f"{pct:>2d}{suf} ")
        rows.append(f" {r + 1} |" + "".join(cells))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_by_depth')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--n-games', type=int, default=5)
    ap.add_argument('--last-n-turns', type=int, default=15,
                    help='For each game, show the last N turns before the '
                         'error (0 = all turns).')
    ap.add_argument('--output-txt', default='prob_evolution.txt')
    ap.add_argument('--output-plot', default='prob_evolution.png')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    print(f"Loaded OGPT (block_size={block_size})")

    token_to_pos, pos_to_token = build_token_to_pos_and_pos_to_token(block_size)

    records_path = os.path.join(args.adversarial_dir,
                                  'adversarial_records.npz')
    d = np.load(records_path, allow_pickle=True)
    games = d['games']; turns = d['turns']; cells = d['illegal_cells']
    print(f"Loaded {len(games)} adversarial records")

    # First N records after the same shuffle we've been using
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(games))[:args.n_games]

    lines = []
    game_plot_data = []   # list of (turn_list, p_C_list, first_illegal_turn, error_turn)

    for gi, i in enumerate(idx):
        game = list(games[i])
        error_turn = int(turns[i])   # position index in the sequence
        C = int(cells[i])
        n_moves_played_at_error = error_turn + 1

        lines.append("")
        lines.append("=" * 72)
        lines.append(f"GAME {gi + 1} / {args.n_games}")
        lines.append(f"Error turn: {error_turn} (after move index {error_turn}); "
                      f"illegal cell C: {alg(C)} (index {C})")
        lines.append(f"Game length: {len(game)}")
        lines.append("=" * 72)

        # Feed tokens to model ONCE (through error_turn), then read per-position logits
        L = min(error_turn + 1, block_size)
        tokens = tokenize_games([game[:L]], seq_len=block_size).to(device)
        with torch.no_grad():
            logits, _ = model(tokens)                             # (1, L, V)

        # For each turn t (0..error_turn), extract prediction
        # logits[0, t-1] gives prediction of move at position t (given tokens[0..t-1])
        # Track per-turn:
        # - probability on cell C, C's rank, C legal or not, total P(illegal)
        turns_seen = []
        p_C_seen = []
        rank_C_seen = []
        C_legal_seen = []
        p_illegal_total_seen = []
        legal_sets_at_turn = {}
        state_at_turn = {}

        board = OthelloBoardState()
        for t in range(0, error_turn + 1):
            try:
                board.umpire(game[t])
            except Exception:
                break
            # Board state is now AFTER move t; predict move at t+1
            state_at_turn[t] = np.asarray(board.state, dtype=np.int8)
            legal_set = set(board.get_valid_moves())
            legal_sets_at_turn[t] = legal_set

            # Prediction: use logits at position t (0-indexed), which is the
            # prediction given tokens[0..t] (t+1 tokens fed)
            if t < 0 or t >= L:
                continue
            # Extract 60-d probability marginal
            probs_60 = np.zeros(60, dtype=np.float32)
            probs = F.softmax(logits[0, t, :], dim=-1).cpu().numpy()
            for k, m in enumerate(VALID_MOVES):
                tok = int(pos_to_token[m])
                if tok >= 0:
                    probs_60[k] = float(probs[tok])

            # Renormalize (probability might not sum to 1 over just the 60 cells)
            total = float(probs_60.sum())
            probs_60 = probs_60 / max(total, 1e-9)

            # Track P(C), rank of C, C's legality, P(illegal)
            C_idx60 = VALID_MOVES.index(C)
            p_C = float(probs_60[C_idx60])
            rank_C = int((probs_60 > probs_60[C_idx60]).sum()) + 1
            C_legal = C in legal_set
            legal_mask = np.array([m in legal_set for m in VALID_MOVES], dtype=bool)
            p_illegal = float(probs_60[~legal_mask].sum())

            turns_seen.append(t)
            p_C_seen.append(p_C)
            rank_C_seen.append(rank_C)
            C_legal_seen.append(int(C_legal))
            p_illegal_total_seen.append(p_illegal)

        # Emit per-turn heatmap for the last-N-turns window
        keep_from = 0
        if args.last_n_turns > 0:
            keep_from = max(0, len(turns_seen) - args.last_n_turns)
        for j in range(keep_from, len(turns_seen)):
            t = turns_seen[j]
            probs_60 = np.zeros(60, dtype=np.float32)
            # Recompute (cheap because logits already cached above)
            probs = F.softmax(logits[0, t, :], dim=-1).cpu().numpy()
            for k, m in enumerate(VALID_MOVES):
                tok = int(pos_to_token[m])
                if tok >= 0:
                    probs_60[k] = probs[tok]
            total = probs_60.sum()
            probs_60 = probs_60 / max(total, 1e-9)

            legal_set = legal_sets_at_turn[t]
            state = state_at_turn[t]
            hm = render_heatmap(probs_60, state, legal_set, C)
            lines.append("")
            lines.append(f"--- Turn {t}  moves played: {t + 1}  "
                          f"P(C={alg(C)})={p_C_seen[j]:.4f}  "
                          f"rank(C)={rank_C_seen[j]:>2d}/60  "
                          f"C {'LEGAL' if C_legal_seen[j] else 'ILLEGAL'}  "
                          f"P(all illegal)={p_illegal_total_seen[j]:.4f} ---")
            lines.extend(hm)

        # Bookkeeping for plot
        first_illegal_turn = None
        for j, val in enumerate(C_legal_seen):
            if val == 0:
                first_illegal_turn = turns_seen[j]
                break
        game_plot_data.append((
            list(turns_seen), list(p_C_seen), list(C_legal_seen),
            list(p_illegal_total_seen), first_illegal_turn,
            error_turn, alg(C),
        ))

    # Write txt
    with open(args.output_txt, 'w') as f:
        f.write("\n".join(lines))
    print(f"Wrote text heatmaps to {args.output_txt}")

    # Line plot: P(C) over turns for each game
    try:
        fig = Figure(figsize=(11, 3 * args.n_games))
        for gi, (ts, pc, legal, p_ill, first_ill, err_turn, C_name) \
                in enumerate(game_plot_data):
            ax = fig.add_subplot(args.n_games, 1, gi + 1)
            ax.plot(ts, pc, 'o-', color='C0', label=f"P(C={C_name})")
            ax.plot(ts, p_ill, 's--', color='C3', alpha=0.6,
                     label="Total P(illegal)")
            # Mean prob over legal cells for reference
            # (we don't have that pre-computed here; skip for simplicity)
            ax.axvline(err_turn, color='k', linestyle=':',
                        label=f"error turn = {err_turn}")
            if first_ill is not None and first_ill != err_turn:
                ax.axvline(first_ill, color='r', linestyle='--',
                            label=f"C first illegal at turn {first_ill}")
            ax.set_xlabel("Turn (moves played after)")
            ax.set_ylabel("Probability")
            ax.set_title(f"Game {gi + 1}: illegal cell {C_name}, "
                          f"error turn {err_turn}")
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output_plot, dpi=150, bbox_inches='tight')
        print(f"Saved line plot to {args.output_plot}")
    except Exception as e:
        print(f"[warning] plot render failed: {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
