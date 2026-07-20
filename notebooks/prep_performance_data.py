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
def probs_at_turn(model, game, t, block_size, pos_to_token, device):
    """Model's per-cell probabilities for the NEXT move given that moves
    0..t have already been played.  Matches the convention used when the
    adversarial records were generated (feed t+1 tokens, read logits[t]).

    Returns (64,) float32 renormalised to sum to 1 over the 60 non-center
    cells.
    """
    L = min(t + 1, block_size)
    tokens = tokenize_games([game[:L]], seq_len=block_size).to(device)
    logits, _ = model(tokens)
    probs = F.softmax(logits[0, t, :], dim=-1).detach().cpu().numpy()
    p = np.zeros(64, dtype=np.float32)
    for cell in VALID_MOVES:
        tok = int(pos_to_token[cell])
        if tok >= 0:
            p[cell] = probs[tok]
    s = p.sum()
    if s > 1e-9:
        p /= s
    return p


def state_at_turn(game, t):
    """Board state AFTER moves 0..t have been played (i.e., the state
    from which the model would predict move t+1).  Matches probs_at_turn.
    """
    board = OthelloBoardState()
    for m in game[:t + 1]:
        board.umpire(int(m))
    return np.asarray(board.state, dtype=np.int8).copy()


def legal_at_turn(game, t):
    board = OthelloBoardState()
    for m in game[:t + 1]:
        board.umpire(int(m))
    return set(board.get_valid_moves())


# --------------------------------------------------------------------------
# Figure 1 & 2: adversarial position + triptych
# --------------------------------------------------------------------------

def find_t_L(game, T, C_illegal, verbose=False):
    """Return the last same-parity 'prediction point' t <= T-2 where the
    NEXT move (t+1) could have been C.

    In the adversarial-record convention, T is the last-played turn index
    and the model's illegal argmax was for move T+1.  A same-parity
    earlier prediction point t (t < T, t == T mod 2) is one where the
    model was predicting a move of the same parity as C.  We want C to
    have been in the legal-moves-after-game[:t+1] at that point.
    """
    board = OthelloBoardState()
    legal_after = {}      # legal_after[t] = legals after playing game[0..t]
    for t in range(T + 1):
        try:
            board.umpire(int(game[t]))
        except Exception as e:
            if verbose:
                print(f'    umpire failed at t={t} mv={game[t]}: {e}')
            return None
        legal_after[t] = set(board.get_valid_moves())
    for t in range(T - 2, -1, -2):
        if C_illegal in legal_after.get(t, set()):
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
    # Adversarial records always have T == len(game) - 1 (the beam
    # search stopped at the illegal argmax so there is no game[T+1]);
    # do NOT require a valid t_next here.
    if T < 2:
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
    """Build the (t_L, T_illegal[, t_next]) panels + probs.  Falls back
    to two panels if the game ends at T_illegal (typical for
    adversarial records, where the beam search halted at the illegal)."""
    t_next = None
    for cand in (T_illegal + 2, T_illegal + 1):
        if cand < len(game):
            t_next = cand
            break

    turns = [t_L, T_illegal]
    labels = ['C legal', 'C illegal chosen']
    if t_next is not None:
        turns.append(t_next)
        labels.append('after')
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
    ap.add_argument('--adv-search-k', type=int, default=1000,
                    help='How many adversarial records to score before '
                          'picking the best.')
    ap.add_argument('--adv-min-p-i', type=float, default=0.01,
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
        # Full inference sanity check on adv[0].
        _game0 = tuple(adv_games[0])
        _T0 = int(adv_turns[0])
        _C0 = int(adv_illegal[0])
        _p_full = probs_at_turn(model, _game0, _T0, block_size,
                                    pos_to_token, device)
        _top5 = np.argsort(_p_full)[::-1][:5]
        print(f'DIAG adv[0]: game_len={len(_game0)}, T={_T0}, C={_C0}')
        print(f'  P(cell {_C0}) [claimed illegal argmax] = {_p_full[_C0]:.4f}')
        print(f'  Top-5 cells by our P: '
               f'{[(int(c), float(_p_full[c])) for c in _top5]}')
        print(f'  game[T]           = {_game0[_T0]!r}  '
               f'(actual next move in the recorded game)')
        # Also test the alternate convention: feed game[:T+1], read logits[T]
        # (in case mingpt output at position i predicts token i, not i+1).
        _alt_tokens = tokenize_games(
            [_game0[:min(_T0 + 1, block_size)]], seq_len=block_size).to(device)
        with torch.no_grad():
            _alt_logits, _ = model(_alt_tokens)
        _alt_probs = F.softmax(_alt_logits[0, _T0, :], dim=-1).cpu().numpy()
        _alt_p = np.zeros(64, dtype=np.float32)
        for cell in VALID_MOVES:
            tok = int(pos_to_token[cell])
            if tok >= 0:
                _alt_p[cell] = _alt_probs[tok]
        _alt_p /= max(_alt_p.sum(), 1e-9)
        _alt_top5 = np.argsort(_alt_p)[::-1][:5]
        print(f'  ALT (feed T+1 tokens, read logits[T]):')
        print(f'    P(cell {_C0}) = {_alt_p[_C0]:.4f}')
        print(f'    Top-5:        {[(int(c), float(_alt_p[c])) for c in _alt_top5]}')
        print(f'Searching {args.adv_search_k} adversarial records for a '
               f'high-persistence example (P_I >= {args.adv_min_p_i})...')
        best = None
        best_score = -1.0
        n_have_tL = 0
        max_p_i_seen = 0.0
        top10 = []  # keep the 10 best (score, i, t_L, P_L, P_I)
        for i in range(min(args.adv_search_k, len(adv_games))):
            game = tuple(adv_games[i])
            T = int(adv_turns[i])
            C = int(adv_illegal[i])
            if T < 2 or T + 1 >= len(game):
                continue
            try:
                t_L = find_t_L(game, T, C)
            except Exception:
                continue
            if t_L is None:
                continue
            n_have_tL += 1
            Cr, Cc = C // 8, C % 8
            try:
                P_I = float(probs_at_turn(model, game, T, block_size,
                                              pos_to_token, device
                                             ).reshape(8, 8)[Cr, Cc])
            except Exception:
                continue
            if P_I > max_p_i_seen:
                max_p_i_seen = P_I
            if P_I < args.adv_min_p_i:
                continue
            P_L = float(probs_at_turn(model, game, t_L, block_size,
                                          pos_to_token, device
                                         ).reshape(8, 8)[Cr, Cc])
            retention = P_I / max(P_L, 1e-6)
            score = P_I * retention
            top10.append((score, i, t_L, P_L, P_I))
            top10.sort(reverse=True)
            top10 = top10[:10]
            if score > best_score:
                best_score = score
                best = (i, t_L, P_L, P_I)
                print(f'  new best @ i={i}: P_L={P_L:.3f} '
                       f'P_I={P_I:.3f} retention={retention:.2f}')
            if (i + 1) % 100 == 0:
                print(f'  ...{i+1}/{args.adv_search_k} scanned, '
                       f'{n_have_tL} with t_L, max P_I={max_p_i_seen:.3f}',
                       flush=True)
        print(f'\nSearch summary: scanned {min(args.adv_search_k, len(adv_games))}, '
               f'{n_have_tL} had t_L, max P_I seen = {max_p_i_seen:.3f}')
        print('Top 10 by score:')
        for sc, ii, tl, pl, pi in top10:
            print(f'  i={ii}  P_L={pl:.4f}  P_I={pi:.4f}  '
                   f'retention={pi/max(pl,1e-6):.3f}  score={sc:.4f}')
        if best is None:
            raise SystemExit(f'no adversarial record met P_I >= {args.adv_min_p_i}. '
                              f'Max P_I seen among {n_have_tL} candidates with t_L '
                              f'was {max_p_i_seen:.4f}. Try --adv-min-p-i {max_p_i_seen/2:.3f}')
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
