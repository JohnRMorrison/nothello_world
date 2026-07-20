"""Run Othello-GPT inference on selected positions; save results for the
presentation notebook (notebooks/presentation_boards.ipynb, Section 3).

Outputs notebooks/talk_data/performance_data.npz with:
  # Adversarial single-position figure --------------------------------
  adv_state         (8, 8) int8   board at t_I
  adv_probs         (8, 8) float  per-cell prob at t_I
  adv_illegal_cell  (2,)   int    (row, col) of the chosen-illegal cell
  adv_turn          scalar        the turn index (t_I)

  # Adversarial triptych (t_L, t_I, t_next same-parity) --------------
  trip_states       (3, 8, 8) int8
  trip_probs        (3, 8, 8) float
  trip_turns        (3,)      int    turn indices
  trip_labels       (3,) str object 'legal', 'illegal chosen', 'after'
  trip_illegal_cell (2,)      int    the C being tracked

  # No-flanker-but-high-P example -----------------------------------
  flank_state          (8, 8) int8
  flank_probs          (8, 8) float
  flank_target_cell    (2,)   int    the cell with high prob but no flank
  flank_line_cells     (M, 2) int    the opp-piece line cells
  flank_direction      (2,)   int    (dr, dc) of the opp line

Usage:
    python notebooks/prep_performance_data.py \\
        [--adv-index N] [--search-pickles K] [--min-prob P] [--min-line L]

Everything is deterministic given `--adv-index` and `--search-seed`.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT,
                                 'experiments/mathematical_transformation_experiments'))

from mingpt.model import GPT, GPTConfig  # noqa: E402
from data.othello import OthelloBoardState  # noqa: E402
from probe_state_pred_for_othello import tokenize_games, VOCAB_SIZE  # noqa: E402


CENTER_CELLS = {27, 28, 35, 36}
VALID_MOVES = [i for i in range(64) if i not in CENTER_CELLS]
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1),  (1, -1), (1, 0), (1, 1)]


# --------------------------------------------------------------------------
# Model + inference helpers
# --------------------------------------------------------------------------

def load_model(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    block_size = sd['pos_emb'].shape[1]
    cfg = GPTConfig(VOCAB_SIZE, block_size,
                     n_layer=8, n_head=8, n_embd=512)
    model = GPT(cfg)
    model.load_state_dict(sd)
    return model.to(device).eval(), block_size


def build_pos_to_token(block_size):
    dummy = list(VALID_MOVES)
    toks = tokenize_games([dummy], seq_len=block_size)[0].tolist()
    pos_to_token = np.full(64, -1, dtype=np.int64)
    for i, m in enumerate(dummy):
        if i < len(toks):
            pos_to_token[m] = toks[i]
    return pos_to_token


@torch.no_grad()
def probs_at_turn(model, game, turn, block_size, pos_to_token, device):
    """Model's per-cell probabilities for the move about to be made at `turn`.

    game[:turn] is the prefix already played.  Returns (64,) float32 renormalised
    to sum to 1 over the 60 non-center cells.
    """
    if turn == 0:
        # No context yet -- return uniform over legal opening moves
        p = np.zeros(64, dtype=np.float32)
        return p
    tokens = tokenize_games([game[:turn]],
                              seq_len=block_size).to(device)
    logits, _ = model(tokens)
    probs = F.softmax(logits[0, turn - 1, :], dim=-1).detach().cpu().numpy()
    p = np.zeros(64, dtype=np.float32)
    for cell in VALID_MOVES:
        tok = int(pos_to_token[cell])
        if tok >= 0:
            p[cell] = probs[tok]
    s = p.sum()
    if s > 1e-9:
        p /= s
    return p


def state_at_turn(game, turn):
    """Board state after moves 0..turn-1 have been played."""
    board = OthelloBoardState()
    for m in game[:turn]:
        board.umpire(m)
    return np.asarray(board.state, dtype=np.int8).copy()


def legal_at_turn(game, turn):
    board = OthelloBoardState()
    for m in game[:turn]:
        board.umpire(m)
    return set(board.get_valid_moves())


# --------------------------------------------------------------------------
# Figure 1 & 2: adversarial position + triptych
# --------------------------------------------------------------------------

def pick_adversarial_triptych(game, T_illegal, C_illegal, model, block_size,
                                pos_to_token, device):
    """Pick a (t_prev, t_I, t_next) triptych around the adversarial choice.

    t_prev  = T_illegal - 2 (same-parity previous turn).  If <0, uses T-1.
    t_I     = T_illegal (turn at which model chose illegal C)
    t_next  = T_illegal + 2 (same-parity next turn).  If >=len(game), T+1.

    Also tries to find C's last-legal same-parity turn as t_L (bonus:
    a more meaningful 'before' if one exists).  Falls back to t_prev
    otherwise.

    Returns dict with per-turn state, probs, label; or None if T is too
    close to game boundaries.
    """
    legals = {t: legal_at_turn(game, t) for t in range(T_illegal + 1)}

    # Preferred 'before' turn: last same-parity turn where C was legal.
    t_before = None
    for t in range(T_illegal - 2, -1, -2):
        if C_illegal in legals[t]:
            t_before = t
            break
    # Fallback: same-parity previous turn (regardless of C's legality).
    if t_before is None:
        for cand in (T_illegal - 2, T_illegal - 1):
            if cand >= 0:
                t_before = cand
                break
    if t_before is None:
        return None

    # 'After' turn: prefer same parity.
    t_next = None
    for cand in (T_illegal + 2, T_illegal + 1):
        if cand < len(game):
            t_next = cand
            break
    if t_next is None:
        return None

    turns = [t_before, T_illegal, t_next]
    labels = ['before', 'illegal chosen', 'after']
    states = np.stack([state_at_turn(game, t).reshape(8, 8) for t in turns])
    probs = np.stack([probs_at_turn(model, game, t, block_size,
                                       pos_to_token, device).reshape(8, 8)
                        for t in turns])
    return {
        'turns': np.array(turns, dtype=np.int32),
        'labels': np.array(labels, dtype=object),
        'states': states.astype(np.int8),
        'probs': probs.astype(np.float32),
    }


# --------------------------------------------------------------------------
# Figure 3: no-flanker-but-high-P
# --------------------------------------------------------------------------

def has_valid_flank(state_8x8, r, c, mover_color):
    """Standard Othello legality: cell (r, c) is legal if there exists a
    direction with >=1 opponent piece(s) capped by a mover piece."""
    opp = -mover_color
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        opps = 0
        while 0 <= rr < 8 and 0 <= cc < 8 and state_8x8[rr, cc] == opp:
            opps += 1
            rr += dr
            cc += dc
        if opps > 0 and 0 <= rr < 8 and 0 <= cc < 8 and \
                state_8x8[rr, cc] == mover_color:
            return True
    return False


def longest_opp_line(state_8x8, r, c, mover_color):
    """Return (length, (dr, dc), [cells]) for the longest opp-piece run
    starting from (r+dr, c+dc) in some direction, where the terminating
    cell is NOT a mover piece (so it's not a flank).  If multiple
    directions tie, take the first one found."""
    opp = -mover_color
    best_len = 0
    best_dir = None
    best_cells = []
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        cells = []
        while 0 <= rr < 8 and 0 <= cc < 8 and state_8x8[rr, cc] == opp:
            cells.append((rr, cc))
            rr += dr
            cc += dc
        # If terminating cell is a mover piece, this is a valid flank; skip.
        if 0 <= rr < 8 and 0 <= cc < 8 and state_8x8[rr, cc] == mover_color:
            continue
        if len(cells) > best_len:
            best_len = len(cells)
            best_dir = (dr, dc)
            best_cells = cells
    return best_len, best_dir, best_cells


def find_no_flanker_example(model, pickle_paths, n_games, min_prob,
                              min_line, block_size, pos_to_token, device,
                              seed=0):
    """Search regular games for a position where the model puts >= min_prob
    on an EMPTY cell C that has an opponent line of length >= min_line in
    some direction, with NO valid flank in any direction (C is illegal).

    Returns dict with state, probs, target_cell, line_cells, direction,
    or None.  Picks the highest-prob such case found.
    """
    rng = np.random.RandomState(seed)
    best = None
    best_score = -1.0
    n_scanned = 0

    for pkl in pickle_paths:
        with open(pkl, 'rb') as f:
            games = pickle.load(f)
        # Shuffle order but deterministic
        order = rng.permutation(len(games))
        for gi in order:
            game = tuple(games[gi])
            if n_scanned >= n_games:
                break
            n_scanned += 1
            # Scan mid/late-game positions; early game rarely has long lines
            for turn in range(15, min(len(game), 55)):
                st = state_at_turn(game, turn).reshape(8, 8)
                mover = 1 if turn % 2 == 0 else -1
                # candidate cells: empty, not legal, with a long opp line
                # Fetch probs only if we find a candidate to avoid wasted
                # forward passes.
                candidates = []
                for cell in VALID_MOVES:
                    r, c = cell // 8, cell % 8
                    if st[r, c] != 0:
                        continue
                    if has_valid_flank(st, r, c, mover):
                        continue
                    L, d, cells = longest_opp_line(st, r, c, mover)
                    if L >= min_line:
                        candidates.append((cell, r, c, L, d, cells))
                if not candidates:
                    continue
                probs = probs_at_turn(model, game, turn, block_size,
                                        pos_to_token, device).reshape(8, 8)
                for cell, r, c, L, d, cells in candidates:
                    p = float(probs[r, c])
                    if p < min_prob:
                        continue
                    score = p * L
                    if score > best_score:
                        best_score = score
                        best = {
                            'state': st.astype(np.int8),
                            'probs': probs.astype(np.float32),
                            'target_cell': np.array([r, c], dtype=np.int32),
                            'line_cells': np.array(cells, dtype=np.int32),
                            'direction': np.array(d, dtype=np.int32),
                            'game_prefix_len': turn,
                            'p_target': p,
                            'line_len': L,
                        }
            if n_scanned >= n_games:
                break
        if n_scanned >= n_games:
            break
    if best is not None:
        print(f'  best: line_len={best["line_len"]} '
               f'p_target={best["p_target"]:.4f}  (scanned {n_scanned} games)')
    return best


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--adv-npz',
                    default='experiment1_by_depth/adversarial_records.npz')
    ap.add_argument('--adv-index', type=int, default=None,
                    help='Which adversarial record to use.  Default: first '
                          'one with a valid triptych.')
    ap.add_argument('--pickle-dir',
                    default='data/othello_synthetic')
    ap.add_argument('--search-pickles', type=int, default=2,
                    help='How many pickle files to scan for figure 3.')
    ap.add_argument('--search-games', type=int, default=200,
                    help='How many games (per pickle) to scan.')
    ap.add_argument('--min-prob', type=float, default=0.02)
    ap.add_argument('--min-line', type=int, default=3)
    ap.add_argument('--search-seed', type=int, default=0)
    ap.add_argument('--out',
                    default='notebooks/talk_data/performance_data.npz')
    ap.add_argument('--device', default=None,
                    help='cuda or cpu; auto if unset.')
    args = ap.parse_args()

    device = torch.device(args.device or
                            ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    print(f'Loading model from {args.ckpt}...')
    model, block_size = load_model(args.ckpt, device)
    pos_to_token = build_pos_to_token(block_size)
    print(f'  block_size={block_size}')

    print(f'Loading adversarial records from {args.adv_npz}...')
    adv = np.load(args.adv_npz, allow_pickle=True)
    adv_games = adv['games']
    adv_turns = adv['turns']
    adv_illegal = adv['illegal_cells']
    print(f'  {len(adv_games)} records loaded')

    # --- pick an adversarial index that gives a valid triptych ---
    picked_idx = None
    trip = None
    idx_iter = ([args.adv_index] if args.adv_index is not None
                else range(len(adv_games)))
    for i in idx_iter:
        game = tuple(adv_games[i])
        T = int(adv_turns[i])
        C = int(adv_illegal[i])
        try:
            trip = pick_adversarial_triptych(game, T, C, model, block_size,
                                                 pos_to_token, device)
        except Exception as e:
            print(f'  adv[{i}] failed: {e}')
            continue
        if trip is not None:
            picked_idx = i
            break
    if trip is None:
        raise SystemExit('could not build a valid triptych for any '
                          'adversarial record')
    print(f'Picked adv index {picked_idx}: '
           f'turns={trip["turns"].tolist()} '
           f'C={int(adv_illegal[picked_idx])}')

    C_illegal = int(adv_illegal[picked_idx])
    Cr, Cc = C_illegal // 8, C_illegal % 8

    # Figure 1 data = middle of triptych (t_I)
    fig1 = {
        'adv_state': trip['states'][1],
        'adv_probs': trip['probs'][1],
        'adv_illegal_cell': np.array([Cr, Cc], dtype=np.int32),
        'adv_turn': trip['turns'][1],
        'adv_index': np.int64(picked_idx),
    }

    # Triptych data
    fig2 = {
        'trip_states': trip['states'],
        'trip_probs': trip['probs'],
        'trip_turns': trip['turns'],
        'trip_labels': trip['labels'],
        'trip_illegal_cell': np.array([Cr, Cc], dtype=np.int32),
    }

    # --- figure 3: search regular games ---
    pkls = sorted(glob.glob(os.path.join(REPO_ROOT, args.pickle_dir,
                                              '*.pickle')))
    if not pkls:
        raise SystemExit(f'no pickle files in {args.pickle_dir}')
    pkls = pkls[:args.search_pickles]
    print(f'Searching {len(pkls)} pickle file(s), up to '
           f'{args.search_games} games each, for no-flanker-high-P case...')
    flank = find_no_flanker_example(
        model, pkls,
        n_games=args.search_games,
        min_prob=args.min_prob,
        min_line=args.min_line,
        block_size=block_size,
        pos_to_token=pos_to_token,
        device=device,
        seed=args.search_seed,
    )
    if flank is None:
        print('  no matching case found; figure 3 will be missing')
        fig3 = {}
    else:
        fig3 = {
            'flank_state': flank['state'],
            'flank_probs': flank['probs'],
            'flank_target_cell': flank['target_cell'],
            'flank_line_cells': flank['line_cells'],
            'flank_direction': flank['direction'],
            'flank_p_target': np.float32(flank['p_target']),
            'flank_line_len': np.int32(flank['line_len']),
        }

    # --- save ---
    out_path = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **fig1, **fig2, **fig3)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
