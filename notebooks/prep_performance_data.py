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

def find_t_L(game, T_illegal, C_illegal):
    """Return the last same-parity turn t <= T-2 where C was legal, or None."""
    legals_before = OthelloBoardState()
    # We only need legality per turn; walk once and cache same-parity legality.
    legal_history = {}
    for t in range(T_illegal + 1):
        legal_history[t] = C_illegal in set(legals_before.get_valid_moves())
        if t < T_illegal:
            try:
                legals_before.umpire(game[t])
            except Exception:
                return None
    for t in range(T_illegal - 2, -1, -2):
        if legal_history.get(t, False):
            return t
    return None


def score_persistence_record(game, T, C, model, block_size, pos_to_token,
                               device, min_p_I=0.05):
    """Return (score, t_L, P_L, P_I) for one adversarial record.

    Score rewards 'C stays highly probable even though C became illegal':
      score = P_I * (P_I / max(P_L, 1e-6))
    Returns None if this record isn't a valid candidate (no t_L, or P_I
    below threshold).
    """
    if T < 2 or T + 1 >= len(game):
        return None
    t_L = find_t_L(game, T, C)
    if t_L is None:
        return None
    Cr, Cc = C // 8, C % 8
    P_I_full = probs_at_turn(model, game, T, block_size,
                                pos_to_token, device).reshape(8, 8)
    P_I = float(P_I_full[Cr, Cc])
    if P_I < min_p_I:
        return None
    P_L_full = probs_at_turn(model, game, t_L, block_size,
                                pos_to_token, device).reshape(8, 8)
    P_L = float(P_L_full[Cr, Cc])
    retention = P_I / max(P_L, 1e-6)
    score = P_I * retention
    return score, t_L, P_L, P_I


def build_triptych_for_index(game, T_illegal, C_illegal, t_L,
                                model, block_size, pos_to_token, device):
    """Build the (t_L, T_illegal, t_next) triptych, computing states + probs."""
    t_next = None
    for cand in (T_illegal + 2, T_illegal + 1):
        if cand < len(game):
            t_next = cand
            break
    if t_next is None:
        return None

    turns = [t_L, T_illegal, t_next]
    labels = ['C legal', 'C illegal chosen', 'after']
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
                    help='Which adversarial record to use.  Default: search '
                          'and pick the best-persistence example.')
    ap.add_argument('--adv-search-k', type=int, default=300,
                    help='How many adversarial records to score before '
                          'picking the best.  Default 300.')
    ap.add_argument('--adv-min-p-i', type=float, default=0.05,
                    help='Minimum P(C) at t_I to consider a candidate.')
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

    # --- pick an adversarial record via persistence search ---
    if args.adv_index is not None:
        picked_idx = args.adv_index
        game = tuple(adv_games[picked_idx])
        T = int(adv_turns[picked_idx])
        C = int(adv_illegal[picked_idx])
        t_L = find_t_L(game, T, C)
        if t_L is None:
            raise SystemExit(f'record {picked_idx} has no valid t_L')
        picked_P_L = float(probs_at_turn(model, game, t_L, block_size,
                                             pos_to_token, device
                                            ).reshape(8, 8)[C // 8, C % 8])
        picked_P_I = float(probs_at_turn(model, game, T, block_size,
                                             pos_to_token, device
                                            ).reshape(8, 8)[C // 8, C % 8])
        picked_t_L = t_L
    else:
        print(f'Searching {args.adv_search_k} adversarial records for a '
               f'high-persistence example (P_I >= {args.adv_min_p_i})...')
        best = None
        best_score = -1.0
        for i in range(min(args.adv_search_k, len(adv_games))):
            game = tuple(adv_games[i])
            T = int(adv_turns[i])
            C = int(adv_illegal[i])
            try:
                r = score_persistence_record(game, T, C, model, block_size,
                                                  pos_to_token, device,
                                                  min_p_I=args.adv_min_p_i)
            except Exception as e:
                continue
            if r is None:
                continue
            score, t_L, P_L, P_I = r
            if score > best_score:
                best_score = score
                best = (i, t_L, P_L, P_I)
                print(f'  new best @ i={i}: P_L={P_L:.3f} '
                       f'P_I={P_I:.3f} retention={P_I/max(P_L,1e-6):.2f}')
        if best is None:
            raise SystemExit('no adversarial record met the P_I threshold; '
                              'try lowering --adv-min-p-i')
        picked_idx, picked_t_L, picked_P_L, picked_P_I = best
        game = tuple(adv_games[picked_idx])
        T = int(adv_turns[picked_idx])
        C = int(adv_illegal[picked_idx])

    print(f'Picked adv[{picked_idx}]: T={T} C={C} t_L={picked_t_L} '
           f'P_L={picked_P_L:.4f} P_I={picked_P_I:.4f} '
           f'retention={picked_P_I / max(picked_P_L, 1e-6):.3f}')

    trip = build_triptych_for_index(game, T, C, picked_t_L, model,
                                          block_size, pos_to_token, device)
    if trip is None:
        raise SystemExit(f'record {picked_idx} has no valid t_next')

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
