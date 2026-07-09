"""Game-1 illustration for the presentation:

  1. Text table over C's-parity turns (12, 14, ..., 32) with columns:
       turn | P(C) | probe-error cells (any)
  2. Line plot of P(C) over ALL turns 12..32
  3. Line plot of the probe's logit margin at cell F2 over ALL turns 12..32
     (logit of ground-truth class minus max of the other two class logits;
     probe mode chosen per turn parity as elsewhere)

Cell for margin plot is configurable (--margin-cell f2 by default).

Usage:
    python plot_game1_trajectory.py \\
        --game-index 0 --start-turn 12 --end-turn 32 \\
        --margin-cell f2 \\
        --out-prefix plots/game1_traj
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, VOCAB_SIZE, extract_activations,
)


CENTER_CELLS = {27, 28, 35, 36}
VALID_MOVES = [i for i in range(64) if i not in CENTER_CELLS]


def alg(cell):
    return f"{'abcdefgh'[cell % 8]}{cell // 8 + 1}"


def cell_from_alg(name):
    """'f2' -> cell index."""
    name = name.strip().lower()
    col = 'abcdefgh'.index(name[0])
    row = int(name[1]) - 1
    return row * 8 + col


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
    """Nanda state (1=BLACK, -1=WHITE, 0=empty) -> probe class {0,1,2}
    (0=empty, 1=WHITE, 2=BLACK)."""
    v = state_8x8[cell // 8, cell % 8]
    if v == 0: return 0
    if v == -1: return 1
    return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_by_depth')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe',
                    default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--game-index', type=int, default=0,
                    help='Starting game index (or the only game if --n-games=1).')
    ap.add_argument('--n-games', type=int, default=1,
                    help='Number of consecutive games to process starting at '
                         '--game-index.  With N>1, plots are skipped by default '
                         '(pass --plots to enable).')
    ap.add_argument('--start-turn', type=int, default=0)
    ap.add_argument('--end-turn', type=int, default=-1,
                    help='-1 = the error turn T.')
    ap.add_argument('--margin-cell', type=str, default='f2')
    ap.add_argument('--out-prefix', default='plots/game_traj')
    ap.add_argument('--plots', action='store_true',
                    help='Also produce P(C) and margin plots for each game.')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    probe = torch.load(args.probe, map_location='cpu')
    assert probe.shape == (3, 512, 8, 8, 3)

    pos_to_token = build_pos_to_token(block_size)

    d = np.load(os.path.join(args.adversarial_dir, 'adversarial_records.npz'),
                allow_pickle=True)
    games = d['games']; turns = d['turns']; cells = d['illegal_cells']
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(games))

    all_tables = []
    for gi in range(args.game_index, args.game_index + args.n_games):
        i = int(idx[gi])
        game = list(games[i])
        T = int(turns[i])
        C = int(cells[i])
        end_turn = args.end_turn if args.end_turn >= 0 else T

        margin_cell = cell_from_alg(args.margin_cell)
        print(f"Game {gi + 1}: error turn T = {T}, C = {alg(C)}, "
              f"margin cell = {alg(margin_cell)}, turn range [{args.start_turn}, {end_turn}]")

        # Feed to model, get logits + activations for all turns up to end_turn
        L = min(end_turn + 1, block_size)
        tokens = tokenize_games([game[:L]], seq_len=block_size).to(device)
        with torch.no_grad():
            logits, _ = model(tokens)
            acts = extract_activations(model, tokens, args.layer)
        acts_np = acts.cpu().numpy()[0]

        # Play the game, collect per-turn data
        board = OthelloBoardState()
        per_turn = {}
        C_idx60 = VALID_MOVES.index(C)
        target_parity = T % 2

        for t in range(0, end_turn + 1):
            board.umpire(game[t])
            state = np.asarray(board.state, dtype=np.int8).copy()
            legal_set = set(board.get_valid_moves())

            probs = F.softmax(logits[0, t, :], dim=-1).detach().cpu().numpy()
            probs_60 = np.zeros(60, dtype=np.float32)
            for k, m in enumerate(VALID_MOVES):
                tok = int(pos_to_token[m])
                if tok >= 0:
                    probs_60[k] = probs[tok]
            probs_60 = probs_60 / max(probs_60.sum(), 1e-9)
            p_C = float(probs_60[C_idx60])

            probe_state = probe_argmax_state(acts_np[t], t, probe)
            err_cells = []
            for cell in range(64):
                r_, c_ = cell // 8, cell % 8
                if probe_state[r_, c_] != state[r_, c_]:
                    err_cells.append(cell)

            lg = probe_logits_at_cell(acts_np[t], t, probe, margin_cell)
            gt_cls = gt_probe_class(state, margin_cell)
            other = [lg[j] for j in range(3) if j != gt_cls]
            margin = float(lg[gt_cls] - max(other))

            per_turn[t] = {
                'p_C': p_C, 'err_cells': err_cells, 'margin': margin,
                'C_illegal': C not in legal_set,
                'is_c_parity': (t % 2 == target_parity),
            }

        # --- Text table (C's-parity turns only) ---
        lines = []
        lines.append("=" * 68)
        lines.append(f"Game {gi + 1}: C = {alg(C)}, error turn T = {T}")
        lines.append("=" * 68)
        lines.append("  turn |   P(C)   | C status  |  probe-error cells (any)")
        lines.append("  " + "-" * 62)
        for t in range(args.start_turn, end_turn + 1):
            if not per_turn[t]['is_c_parity']:
                continue
            err_str = ",".join(alg(k) for k in per_turn[t]['err_cells']) \
                        if per_turn[t]['err_cells'] else '-'
            c_status = 'ILLEGAL' if per_turn[t]['C_illegal'] else 'LEGAL  '
            lines.append(f"   {t:>3d} | {per_turn[t]['p_C']:7.4f}  | {c_status}   |  {err_str}")
        all_tables.append("\n".join(lines))
        print("\n".join(lines))

        if not args.plots:
            continue

        # --- C's-player turns only ---
        ts_c = [t for t in range(args.start_turn, end_turn + 1)
                 if per_turn[t]['is_c_parity']]
        p_C_c = [per_turn[t]['p_C'] for t in ts_c]
        m_c = [per_turn[t]['margin'] for t in ts_c]
        c_illegal = [per_turn[t]['C_illegal'] for t in ts_c]
        # margin > 0 => probe correct on this cell at that turn
        m_correct = [per_turn[t]['margin'] > 0 for t in ts_c]

        GREEN = '#2ca02c'
        RED   = '#d1341a'

        # --- Plot 1: P(C) with green/red dots for C's-player legality ---
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(ts_c, p_C_c, '-', color='#888888', linewidth=1, alpha=0.5)
        colors_p = [RED if ill else GREEN for ill in c_illegal]
        ax1.scatter(ts_c, p_C_c, c=colors_p, s=80, zorder=3,
                     edgecolor='black', linewidth=0.5)
        ax1.axvline(T, color='black', linestyle=':', linewidth=1,
                    label=f'error turn T={T}')
        # Custom legend for dot colors
        from matplotlib.lines import Line2D
        legend_dots = [
            Line2D([0], [0], marker='o', color='w', label=f'{alg(C)} legal',
                    markerfacecolor=GREEN, markersize=9,
                    markeredgecolor='black', markeredgewidth=0.5),
            Line2D([0], [0], marker='o', color='w', label=f'{alg(C)} illegal',
                    markerfacecolor=RED, markersize=9,
                    markeredgecolor='black', markeredgewidth=0.5),
        ]
        ax1.set_xlabel('turn')
        ax1.set_ylabel(f'P({alg(C)})')
        ax1.set_title(f"Game {gi + 1}: P({alg(C)}) on "
                       f"{alg(C)}-player turns "
                       f"[{args.start_turn}-{end_turn}]")
        ax1.grid(True, alpha=0.3)
        ax1.legend(handles=legend_dots + [
            Line2D([0], [0], color='black', linestyle=':',
                    label=f'error turn T={T}')
        ], loc='upper left', fontsize=9)
        fig1.tight_layout()
        p1_path = f"{args.out_prefix}_g{gi + 1}_p_C.png"
        os.makedirs(os.path.dirname(p1_path) or '.', exist_ok=True)
        fig1.savefig(p1_path, dpi=200, bbox_inches='tight')
        plt.close(fig1)
        print(f"Wrote plot to {p1_path}")

        # --- Plot 2: probe logit margin at margin_cell with green/red dots
        #            indicating probe correctness on this cell ---
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.axhline(0, color='#666666', linewidth=1, linestyle='-')
        ax2.plot(ts_c, m_c, '-', color='#888888', linewidth=1, alpha=0.5)
        colors_m = [GREEN if ok else RED for ok in m_correct]
        ax2.scatter(ts_c, m_c, c=colors_m, s=80, zorder=3,
                     edgecolor='black', linewidth=0.5)
        ax2.axvline(T, color='black', linestyle=':', linewidth=1)
        legend_dots2 = [
            Line2D([0], [0], marker='o', color='w',
                    label=f'probe correct on {alg(margin_cell)}',
                    markerfacecolor=GREEN, markersize=9,
                    markeredgecolor='black', markeredgewidth=0.5),
            Line2D([0], [0], marker='o', color='w',
                    label=f'probe wrong on {alg(margin_cell)}',
                    markerfacecolor=RED, markersize=9,
                    markeredgecolor='black', markeredgewidth=0.5),
        ]
        ax2.set_xlabel('turn')
        ax2.set_ylabel(f'logit margin at {alg(margin_cell)}\n'
                        f'(gt class − max other)')
        ax2.set_title(f"Game {gi + 1}: probe logit margin at "
                       f"{alg(margin_cell)} on {alg(C)}-player turns "
                       f"[{args.start_turn}-{end_turn}]")
        ax2.grid(True, alpha=0.3)
        ax2.legend(handles=legend_dots2 + [
            Line2D([0], [0], color='black', linestyle=':',
                    label=f'error turn T={T}')
        ], loc='upper right', fontsize=9)
        fig2.tight_layout()
        p2_path = f"{args.out_prefix}_g{gi + 1}_margin_{args.margin_cell}.png"
        fig2.savefig(p2_path, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f"Wrote plot to {p2_path}")

    tbl_path = f"{args.out_prefix}_tables.txt"
    os.makedirs(os.path.dirname(tbl_path) or '.', exist_ok=True)
    with open(tbl_path, 'w') as f:
        f.write("\n\n".join(all_tables))
    print(f"\nWrote {len(all_tables)} tables to {tbl_path}")


if __name__ == '__main__':
    main()
