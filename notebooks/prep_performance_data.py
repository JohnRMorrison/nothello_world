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

def find_t_L_and_transition(game, T, C_illegal, verbose=False):
    """Return (t_L, t_transition, legal_after) or (None, None, None).

    t_L: last same-parity prediction point <= T-2 where C was legal (in
         legal_after[t_L]).
    t_transition: earliest same-parity prediction point t > t_L where C
         is NOT in legal_after[t] -- i.e., C 'became illegal' at
         t_transition and stayed illegal through T.
    """
    board = OthelloBoardState()
    legal_after = {}
    for t in range(T + 1):
        try:
            board.umpire(int(game[t]))
        except Exception as e:
            if verbose:
                print(f'    umpire failed at t={t} mv={game[t]}: {e}')
            return None, None, None
        legal_after[t] = set(board.get_valid_moves())
    t_L = None
    for t in range(T - 2, -1, -2):
        if C_illegal in legal_after.get(t, set()):
            t_L = t
            break
    if t_L is None:
        return None, None, None
    # Walk FORWARD from t_L in same-parity steps, find first turn where
    # C is illegal.
    t_transition = None
    for t in range(t_L + 2, T + 1, 2):
        if C_illegal not in legal_after.get(t, set()):
            t_transition = t
            break
    return t_L, t_transition, legal_after


def find_t_L(game, T, C_illegal, verbose=False):
    """Back-compat wrapper for callers that only need t_L."""
    t_L, _, _ = find_t_L_and_transition(game, T, C_illegal, verbose)
    return t_L


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


def line_len_from_cell(state_8x8, r, c, dr, dc, mover_color):
    """How many consecutive opp pieces from (r+dr, c+dc) in direction (dr, dc)?"""
    opp = -mover_color
    rr, cc = r + dr, c + dc
    n = 0
    while 0 <= rr < 8 and 0 <= cc < 8 and state_8x8[rr, cc] == opp:
        n += 1
        rr += dr
        cc += dc
    return n


def scan_monotonic_growth(game, target_cell, direction, mover_parity,
                             min_ply=5):
    """For a fixed (target_cell, direction, mover_parity), return a list of
    (turn, opp_line_length) for same-parity turns where the target is
    empty AND has NO valid flank in ANY direction AND the line length is
    STRICTLY GREATER than at the previously-kept turn.

    Does NOT invoke the model -- caller runs inference on the returned
    turns.
    """
    r, c = target_cell // 8, target_cell % 8
    dr, dc = direction
    board = OthelloBoardState()
    prev_len = 0
    kept = []
    for t in range(len(game)):
        try:
            board.umpire(int(game[t]))
        except Exception:
            break
        if t < min_ply:
            continue
        # We look at prediction points where the NEXT move is by
        # mover_parity's mover.  Next move parity = (t+1) % 2.
        if ((t + 1) % 2) != mover_parity:
            continue
        st = np.asarray(board.state, dtype=np.int8).reshape(8, 8)
        if st[r, c] != 0:
            continue
        mover_color = 1 if mover_parity == 0 else -1
        if has_valid_flank(st, r, c, mover_color):
            continue
        L = line_len_from_cell(st, r, c, dr, dc, mover_color)
        if L > prev_len:
            kept.append((t, L))
            prev_len = L
    return kept


def find_monotonic_growth_example(pickle_paths, n_games, min_line_final,
                                     min_seq_len, model, block_size,
                                     pos_to_token, device,
                                     min_final_p=0.05, require_contiguous=False,
                                     seed=0, verbose=True):
    """Search for a game/target/direction where a monotonically increasing
    opp-line toward `target_cell` unfolds over same-parity prediction
    points, AND the model puts substantial probability on the target at
    the final turn (>= min_final_p).

    Returns dict with game, target_cell, direction, mover_parity, and
    (turn, line_len, p_target) triples.

    Score = p_final * sequence_length -- prefers cases with high P at
    the longest line while also spanning many turns.
    """
    rng = np.random.RandomState(seed)
    best = None
    best_score = -1
    n_scanned = 0
    n_candidates = 0

    for pkl in pickle_paths:
        with open(pkl, 'rb') as f:
            games = pickle.load(f)
        order = rng.permutation(len(games))
        for gi in order:
            game = tuple(games[gi])
            if n_scanned >= n_games:
                break
            n_scanned += 1
            for target_cell in VALID_MOVES:
                r, c = target_cell // 8, target_cell % 8
                for direction in DIRS:
                    for mover_parity in (0, 1):
                        seq = scan_monotonic_growth(
                            game, target_cell, direction, mover_parity)
                        if len(seq) < min_seq_len:
                            continue
                        if seq[-1][1] < min_line_final:
                            continue
                        if require_contiguous:
                            lens = [L for (_, L) in seq]
                            expected = list(range(lens[0], lens[-1] + 1))
                            if lens != expected:
                                continue
                        n_candidates += 1
                        # Compute P at the FINAL turn only (cheap first-pass
                        # filter).  If below threshold, skip full sequence.
                        t_final, L_final = seq[-1]
                        p_final = float(probs_at_turn(
                            model, game, t_final, block_size,
                            pos_to_token, device).reshape(8, 8)[r, c])
                        if p_final < min_final_p:
                            continue
                        # Now compute P at every turn in the sequence.
                        seq_with_p = []
                        for (t, L) in seq:
                            p = float(probs_at_turn(
                                model, game, t, block_size,
                                pos_to_token, device).reshape(8, 8)[r, c])
                            seq_with_p.append((t, L, p))
                        # Score prefers longer sequences AND high final P.
                        # seq_len^2 rewards games that cover many distinct
                        # line lengths (denser story), not just the ones
                        # where line jumps from 2 to 7.
                        score = p_final * (len(seq) ** 2)
                        if score > best_score:
                            best_score = score
                            best = {
                                'game': game,
                                'target_cell': target_cell,
                                'direction': direction,
                                'mover_parity': mover_parity,
                                'sequence': seq_with_p,
                                'pickle': pkl,
                                'game_index': int(gi),
                            }
                            if verbose:
                                print(f'  new best game={gi} '
                                       f'target={target_cell} '
                                       f'dir={direction}: seq_len='
                                       f'{len(seq)} final_L='
                                       f'{L_final} p_final={p_final:.3f}',
                                       flush=True)
            if n_scanned >= n_games:
                break
        if n_scanned >= n_games:
            break
    print(f'  scanned {n_scanned} games; '
           f'{n_candidates} candidates met (line+seq_len) filter')
    return best


def find_no_flanker_example(model, pickle_paths, n_games, min_prob,
                              min_line, block_size, pos_to_token, device,
                              seed=0, top_k=5):
    """Search regular games for positions where the model puts >= min_prob
    on an EMPTY cell C that has an opponent line of length >= min_line in
    some direction, with NO valid flank in any direction (C is illegal).

    Returns list of top_k dicts (best-scoring first) with state, probs,
    target_cell, line_cells, direction.  Score = p_target * line_length.
    """
    rng = np.random.RandomState(seed)
    top = []   # list of (score, dict) — keep top_k
    n_scanned = 0

    def maybe_push(score, entry):
        top.append((score, entry))
        top.sort(key=lambda x: x[0], reverse=True)
        del top[top_k:]

    def already_have_target(target_cell, game_id):
        """Deduplicate: don't keep multiple entries from the same game +
        same target cell (they'd all be the same position)."""
        for _, e in top:
            if (int(e['target_cell'][0]) == target_cell[0]
                    and int(e['target_cell'][1]) == target_cell[1]
                    and e['_game_id'] == game_id):
                return True
        return False

    for pkl in pickle_paths:
        with open(pkl, 'rb') as f:
            games = pickle.load(f)
        order = rng.permutation(len(games))
        for gi in order:
            game = tuple(games[gi])
            game_id = (pkl, int(gi))
            if n_scanned >= n_games:
                break
            n_scanned += 1
            for turn in range(15, min(len(game), 55)):
                st = state_at_turn(game, turn).reshape(8, 8)
                mover = 1 if (turn + 1) % 2 == 0 else -1
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
                    if already_have_target((r, c), game_id):
                        continue
                    score = p * L
                    entry = {
                        'state': st.astype(np.int8),
                        'probs': probs.astype(np.float32),
                        'target_cell': np.array([r, c], dtype=np.int32),
                        'line_cells': np.array(cells, dtype=np.int32),
                        'direction': np.array(d, dtype=np.int32),
                        'game_prefix_len': turn,
                        'p_target': p,
                        'line_len': L,
                        '_game': game,
                        '_game_id': game_id,
                    }
                    maybe_push(score, entry)
            if n_scanned >= n_games:
                break
        if n_scanned >= n_games:
            break
    print(f'  scanned {n_scanned} games; kept top {len(top)}')
    for k, (sc, e) in enumerate(top):
        tc = e['target_cell'].tolist()
        print(f'    #{k+1}: line_len={e["line_len"]} '
               f'p_target={e["p_target"]:.4f} '
               f'target_cell=({tc[0]},{tc[1]}) '
               f'turn={e["game_prefix_len"]} score={sc:.4f}')
    return [entry for _, entry in top]


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
    ap.add_argument('--flank-top-k', type=int, default=5,
                    help='How many top no-flanker examples to keep.')
    ap.add_argument('--growth-n-games', type=int, default=500,
                    help='How many games to scan for a monotonic '
                          'line-growth game.')
    ap.add_argument('--growth-min-seq', type=int, default=3,
                    help='Minimum length of the monotonic (turn, L) '
                          'sequence.')
    ap.add_argument('--growth-min-final', type=int, default=3,
                    help='Minimum final line length in the sequence.')
    ap.add_argument('--growth-min-final-p', type=float, default=0.05,
                    help='Minimum P(target) at the final turn of a '
                          'monotonic-growth sequence to consider it a '
                          'valid candidate.')
    ap.add_argument('--growth-require-contiguous', action='store_true',
                    help='Require the sequence of line lengths to be '
                          'contiguous (no gaps).  If the game jumps from '
                          'length 2 to 7, discard.')
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
            # Adversarial records always have T == len(game) - 1;
            # only require T >= 2 so we can walk back to a same-parity
            # earlier prediction point.
            if T < 2:
                continue
            try:
                t_L, t_trans, _ = find_t_L_and_transition(game, T, C)
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
            top10.append((score, i, t_L, t_trans, P_L, P_I))
            top10.sort(key=lambda x: x[0], reverse=True)
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
        for sc, ii, tl, tt, pl, pi in top10:
            print(f'  i={ii}  t_L={tl}  t_transition={tt}  '
                   f'P_L={pl:.4f}  P_I={pi:.4f}  '
                   f'retention={pi/max(pl,1e-6):.3f}  score={sc:.4f}')
        if best is None or not top10:
            raise SystemExit(f'no adversarial record met P_I >= {args.adv_min_p_i}. '
                              f'Max P_I seen among {n_have_tL} candidates with t_L '
                              f'was {max_p_i_seen:.4f}. Try --adv-min-p-i {max_p_i_seen/2:.3f}')

    # --- Build the top-5 payload: 3 boards per record (t_L, t_trans, T) ---
    top_k = min(5, len(top10)) if args.adv_index is None else 1
    top_states = np.zeros((top_k, 3, 8, 8), dtype=np.int8)
    top_probs = np.zeros((top_k, 3, 8, 8), dtype=np.float32)
    top_turns = np.zeros((top_k, 3), dtype=np.int32)
    top_legal_at = np.zeros((top_k, 3), dtype=np.uint8)   # is C legal at each moment
    top_illegal_cells = np.zeros((top_k, 2), dtype=np.int32)
    top_indices = np.zeros(top_k, dtype=np.int64)
    top_P_L = np.zeros(top_k, dtype=np.float32)
    top_P_I = np.zeros(top_k, dtype=np.float32)

    if args.adv_index is not None:
        selection = [(0, picked_idx, picked_t_L, picked_t_L, picked_P_L, picked_P_I)]
    else:
        selection = [(k, top10[k][1], top10[k][2],
                       top10[k][3] if top10[k][3] is not None else top10[k][2] + 2,
                       top10[k][4], top10[k][5]) for k in range(top_k)]

    for k, i, tl, tt, pl, pi in selection:
        game_k = tuple(adv_games[i])
        Tk = int(adv_turns[i])
        Ck = int(adv_illegal[i])
        Cr, Cc = Ck // 8, Ck % 8

        # Compute legal-after for legality checks per moment
        _, _, legal_after = find_t_L_and_transition(game_k, Tk, Ck)
        if legal_after is None:
            continue

        # Ensure t_trans is bounded to <= T
        if tt is None or tt > Tk:
            tt = Tk

        moments = [tl, tt, Tk]
        for m, t in enumerate(moments):
            top_states[k, m] = state_at_turn(game_k, t).reshape(8, 8)
            top_probs[k, m] = probs_at_turn(model, game_k, t, block_size,
                                                pos_to_token, device).reshape(8, 8)
            top_turns[k, m] = t
            top_legal_at[k, m] = int(Ck in legal_after.get(t, set()))
        top_illegal_cells[k] = np.array([Cr, Cc], dtype=np.int32)
        top_indices[k] = i
        top_P_L[k] = pl
        top_P_I[k] = pi

    # Figure 1 kept for back-compat (middle moment of #1)
    fig1 = {
        'adv_state': top_states[0, 1],
        'adv_probs': top_probs[0, 1],
        'adv_illegal_cell': top_illegal_cells[0],
        'adv_turn': top_turns[0, 1],
        'adv_index': top_indices[0],
    }

    # Top-5 payload (3 boards each) -- this is what the notebook uses
    fig2 = {
        'top_states': top_states,
        'top_probs': top_probs,
        'top_turns': top_turns,
        'top_legal_at': top_legal_at,
        'top_illegal_cells': top_illegal_cells,
        'top_indices': top_indices,
        'top_P_L': top_P_L,
        'top_P_I': top_P_I,
        'top_labels': np.array(['t_L (C legal)', 't_transition (C becomes illegal)',
                                 'T (adversarial move)'], dtype=object),
    }

    # --- figure 3: search regular games ---
    pkls = sorted(glob.glob(os.path.join(REPO_ROOT, args.pickle_dir,
                                              '*.pickle')))
    if not pkls:
        raise SystemExit(f'no pickle files in {args.pickle_dir}')
    pkls = pkls[:args.search_pickles]
    print(f'Searching {len(pkls)} pickle file(s), up to '
           f'{args.search_games} games each, for no-flanker-high-P case...')
    flank_list = find_no_flanker_example(
        model, pkls,
        n_games=args.search_games,
        min_prob=args.min_prob,
        min_line=args.min_line,
        block_size=block_size,
        pos_to_token=pos_to_token,
        device=device,
        seed=args.search_seed,
        top_k=args.flank_top_k,
    )
    if not flank_list:
        print('  no matching case found; figure 3 will be missing')
        fig3 = {}
    else:
        flank = flank_list[0]   # winner used for the single-figure back-compat

        # Search for a game with a MONOTONIC opp-line-growth sequence
        # (target cell always empty AND never has a valid flank).
        print(f'Searching {len(pkls)} pickle(s), {args.growth_n_games} games '
               f'each, for a monotonic line-growth game...')
        mono = find_monotonic_growth_example(
            pkls,
            n_games=args.growth_n_games,
            min_line_final=args.growth_min_final,
            min_seq_len=args.growth_min_seq,
            model=model,
            block_size=block_size,
            pos_to_token=pos_to_token,
            device=device,
            min_final_p=args.growth_min_final_p,
            require_contiguous=args.growth_require_contiguous,
            seed=args.search_seed,
        )
        if mono is None:
            print('  no monotonic growth game found; growth_* will be empty.')
            growth_states = np.zeros((0, 8, 8), dtype=np.int8)
            growth_probs = np.zeros((0, 8, 8), dtype=np.float32)
            growth_turns = np.zeros(0, dtype=np.int32)
            growth_line_lens = np.zeros(0, dtype=np.int32)
            growth_p_target = np.zeros(0, dtype=np.float32)
            growth_meta = np.array([], dtype=object)
        else:
            print(f'  best: game_idx={mono["game_index"]} target={mono["target_cell"]} '
                   f'dir={mono["direction"]} sequence={mono["sequence"]}')
            mono_game = mono['game']
            tr = mono['target_cell'] // 8
            tc = mono['target_cell'] % 8
            states = []
            probs_list = []
            turns_list = []
            lens_list = []
            p_target_list = []
            for (t, L, _p) in mono['sequence']:
                st = state_at_turn(mono_game, t).reshape(8, 8)
                pr = probs_at_turn(model, mono_game, t, block_size,
                                       pos_to_token, device).reshape(8, 8)
                states.append(st)
                probs_list.append(pr.astype(np.float32))
                turns_list.append(t)
                lens_list.append(L)
                p_target_list.append(float(pr[tr, tc]))
            growth_states = np.stack(states).astype(np.int8)
            growth_probs = np.stack(probs_list).astype(np.float32)
            growth_turns = np.array(turns_list, dtype=np.int32)
            growth_line_lens = np.array(lens_list, dtype=np.int32)
            growth_p_target = np.array(p_target_list, dtype=np.float32)
            growth_meta = np.array([{
                'target_cell': mono['target_cell'],
                'direction': mono['direction'],
                'mover_parity': mono['mover_parity'],
                'game_index': mono['game_index'],
                'game': list(mono['game']),
            }], dtype=object)
        # Also pack the full top-K flank list.
        K = len(flank_list)
        top_flank_states = np.stack([e['state'] for e in flank_list]).astype(np.int8)
        top_flank_probs = np.stack([e['probs'] for e in flank_list]).astype(np.float32)
        top_flank_target_cells = np.stack([e['target_cell'] for e in flank_list])
        top_flank_directions = np.stack([e['direction'] for e in flank_list])
        top_flank_p = np.array([e['p_target'] for e in flank_list], dtype=np.float32)
        top_flank_line_lens = np.array([e['line_len'] for e in flank_list], dtype=np.int32)
        top_flank_turns = np.array([e['game_prefix_len'] for e in flank_list],
                                        dtype=np.int32)

        fig3 = {
            # single winner (back-compat)
            'flank_state': flank['state'],
            'flank_probs': flank['probs'],
            'flank_target_cell': flank['target_cell'],
            'flank_line_cells': flank['line_cells'],
            'flank_direction': flank['direction'],
            'flank_p_target': np.float32(flank['p_target']),
            'flank_line_len': np.int32(flank['line_len']),
            # top-K flanks
            'top_flank_states': top_flank_states,
            'top_flank_probs': top_flank_probs,
            'top_flank_target_cells': top_flank_target_cells,
            'top_flank_directions': top_flank_directions,
            'top_flank_p': top_flank_p,
            'top_flank_line_lens': top_flank_line_lens,
            'top_flank_turns': top_flank_turns,
            'growth_states': growth_states,
            'growth_probs': growth_probs,
            'growth_turns': growth_turns,
            'growth_line_lens': growth_line_lens,
            'growth_p_target': growth_p_target,
            'growth_meta': growth_meta,
        }
        # Add growth_target_cell / direction from mono if present (else fall
        # back to the winning-flank target cell so old notebook code doesn't
        # crash).
        if mono is not None:
            fig3['growth_target_cell'] = np.array(
                [mono['target_cell'] // 8, mono['target_cell'] % 8],
                dtype=np.int32)
            fig3['growth_direction'] = np.array(mono['direction'],
                                                     dtype=np.int32)
        else:
            fig3['growth_target_cell'] = flank['target_cell']
            fig3['growth_direction'] = flank['direction']

    # --- save ---
    out_path = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **fig1, **fig2, **fig3)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
