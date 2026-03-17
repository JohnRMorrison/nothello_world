"""
Generate games of Othello variants with modified rules.

Variants:
  1. no_same_quadrant   - Illegal to move in the same quadrant as the last move
  2. no_diagonal_flips  - Flips only along rows and columns, not diagonals
  3. no_row_flips       - Flips only along columns and diagonals, not rows
  4. locked_flips       - Once a piece has been flipped twice, it's permanently locked
  5. max_three_flips    - A move is only legal if it flips at most 3 pieces
  6. self_flanking      - Flanking your OWN pieces flips them to opponent's color
  7. delayed_flips      - Flips happen one turn later (at the start of your next move)
  8. adjacent_legal     - Any empty square adjacent to any piece is legal (flips if bracket)
  9. skip_empty_flips   - Rays skip empty squares when checking for flanking
 10. capture_any        - Adjacent to any opponent piece is legal (flips if bracket)
 11. wrap_flips         - Flanking rays wrap around board edges (torus topology)

Usage:
  python generate_variant_games.py --variant no_diagonal_flips --num-games 100000 \
      --output-dir experiments/variants/no_diagonal_flips --seed 42
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(f=None, **kwargs):
        if f is not None:
            return f
        return lambda fn: fn

# Direction vectors: [row_delta, col_delta]
EIGHTS = [
    [-1,  0],  # up
    [-1,  1],  # up-right
    [ 0,  1],  # right
    [ 1,  1],  # down-right
    [ 1,  0],  # down
    [ 1, -1],  # down-left
    [ 0, -1],  # left
    [-1, -1],  # up-left
]

# ---------------------------------------------------------------------------
# Precomputed ray tables for vectorized move-legality checks
# ---------------------------------------------------------------------------

def _precompute_rays():
    """Precompute ray indices for all (square, direction) pairs.

    RAYS[sq, di, k] = flat board index of the k-th cell along the ray,
    or 64 (sentinel) if the ray is shorter than k+1.
    RAY_LENS[sq, di] = actual length of the ray.
    """
    rays = np.full((64, 8, 7), 64, dtype=np.int32)
    ray_lens = np.zeros((64, 8), dtype=np.int32)
    for sq in range(64):
        r, c = sq // 8, sq % 8
        for di, (dr, dc) in enumerate(EIGHTS):
            cur_r, cur_c = r + dr, c + dc
            idx = 0
            while 0 <= cur_r <= 7 and 0 <= cur_c <= 7:
                rays[sq, di, idx] = cur_r * 8 + cur_c
                idx += 1
                cur_r += dr
                cur_c += dc
            ray_lens[sq, di] = idx
    return rays, ray_lens

RAYS, RAY_LENS = _precompute_rays()

# Quadrant for each square (0-3)
QUADRANTS = np.array([((sq // 8) >= 4) * 2 + ((sq % 8) >= 4) for sq in range(64)])

# Direction masks for variants
DIR_MASK_ALL = np.ones(8, dtype=bool)
DIR_MASK_NO_DIAG = np.array([True, False, True, False, True, False, True, False])
DIR_MASK_NO_ROW = np.array([True, True, False, True, True, True, False, True])

# Adjacency table for each square (for adjacent_legal / capture_any)
def _precompute_adjacency():
    """Precompute adjacent squares for each cell."""
    adj = []
    for sq in range(64):
        r, c = sq // 8, sq % 8
        nbrs = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr <= 7 and 0 <= nc <= 7:
                    nbrs.append(nr * 8 + nc)
        adj.append(np.array(nbrs, dtype=np.int32))
    return adj

ADJACENCY = _precompute_adjacency()

# Precomputed wrap-around rays (torus topology)
def _precompute_wrap_rays():
    """Precompute rays that wrap around board edges."""
    rays = np.full((64, 8, 7), 64, dtype=np.int32)
    ray_lens = np.zeros((64, 8), dtype=np.int32)
    for sq in range(64):
        r, c = sq // 8, sq % 8
        for di, (dr, dc) in enumerate(EIGHTS):
            cur_r, cur_c = r + dr, c + dc
            idx = 0
            while idx < 7:
                wr, wc = cur_r % 8, cur_c % 8
                cell = wr * 8 + wc
                if cell == sq:  # wrapped all the way around
                    break
                rays[sq, di, idx] = cell
                idx += 1
                cur_r += dr
                cur_c += dc
            ray_lens[sq, di] = idx
    return rays, ray_lens

WRAP_RAYS, WRAP_RAY_LENS = _precompute_wrap_rays()


# ---------------------------------------------------------------------------
# Vectorized helpers — process all 64 squares at once
# ---------------------------------------------------------------------------

def _flips_vec(flat, color, dir_mask):
    """Vectorized: for each square, count flips and check validity.

    Args:
        flat: int8 array (64,) — board state
        color: +1 or -1
        dir_mask: bool array (8,) — which directions to consider

    Returns:
        (n_flips, valid_dir) where:
            n_flips: int array (64,) — total flip count per square
            valid_dir: bool array (64, 8) — which directions have valid flips
    """
    # Pad with sentinel at index 64 (reads as 0 = empty)
    flat_pad = np.empty(65, dtype=np.int8)
    flat_pad[:64] = flat
    flat_pad[64] = 0

    vals = flat_pad[RAYS]  # (64, 8, 7)
    is_opp = (vals == -color) & dir_mask[None, :, None]  # (64, 8, 7)

    # First non-opponent position in each ray
    not_opp = ~is_opp
    first_break = not_opp.argmax(axis=2)  # (64, 8)
    has_break = not_opp.any(axis=2)       # (64, 8)

    # Value at the first break position
    gather = np.take_along_axis(RAYS, first_break[:, :, None], axis=2).squeeze(2)
    terminator = flat_pad[gather]  # (64, 8)

    # Valid flip: >=1 opponent then own color
    valid_dir = (first_break > 0) & (terminator == color) & has_break
    n_flips_per_dir = np.where(valid_dir, first_break, 0)
    n_flips = n_flips_per_dir.sum(axis=1)  # (64,)

    return n_flips, valid_dir


def _self_flank_vec(flat, color, dir_mask):
    """Check which squares would self-flank (own run terminated by opponent).

    Returns bool array (64,) — True if placing here causes a self-flank.
    """
    flat_pad = np.empty(65, dtype=np.int8)
    flat_pad[:64] = flat
    flat_pad[64] = 0

    vals = flat_pad[RAYS]
    is_own = (vals == color) & dir_mask[None, :, None]

    not_own = ~is_own
    first_break = not_own.argmax(axis=2)
    has_break = not_own.any(axis=2)

    gather = np.take_along_axis(RAYS, first_break[:, :, None], axis=2).squeeze(2)
    terminator = flat_pad[gather]

    self_flank_dir = (first_break > 0) & (terminator == -color) & has_break
    return self_flank_dir.any(axis=1)  # (64,)


def _flips_locked_vec(flat, color, dir_mask, fc_flat):
    """Like _flips_vec but only counts unlocked pieces (flip_count < 2).

    Returns n_unlocked_flips: int array (64,).
    """
    flat_pad = np.empty(65, dtype=np.int8)
    flat_pad[:64] = flat
    flat_pad[64] = 0

    fc_pad = np.empty(65, dtype=np.int8)
    fc_pad[:64] = fc_flat
    fc_pad[64] = 99  # sentinel: always "locked"

    vals = flat_pad[RAYS]
    is_opp = (vals == -color) & dir_mask[None, :, None]

    not_opp = ~is_opp
    first_break = not_opp.argmax(axis=2)
    has_break = not_opp.any(axis=2)

    gather = np.take_along_axis(RAYS, first_break[:, :, None], axis=2).squeeze(2)
    terminator = flat_pad[gather]
    valid_dir = (first_break > 0) & (terminator == color) & has_break  # (64, 8)

    # Check which ray positions are in the flanked region AND unlocked
    fc_vals = fc_pad[RAYS]  # (64, 8, 7)
    positions = np.arange(7)[None, None, :]  # (1, 1, 7)
    in_flank = positions < first_break[:, :, None]  # (64, 8, 7)
    is_unlocked = (fc_vals < 2)

    n_unlocked = (is_opp & in_flank & is_unlocked & valid_dir[:, :, None]).sum(axis=2)
    return n_unlocked.sum(axis=1)  # (64,)


@njit(cache=True)
def _flips_skip_empty_inner(flat, color, dir_mask, rays, ray_lens):
    """Numba-accelerated inner loop for skip-empty flips."""
    n_flips_total = np.zeros(64, dtype=np.int32)
    for sq in range(64):
        if flat[sq] != 0:
            continue
        for di in range(8):
            if not dir_mask[di]:
                continue
            ray_len = ray_lens[sq, di]
            if ray_len == 0:
                continue
            opp_count = 0
            for k in range(ray_len):
                idx = rays[sq, di, k]
                v = flat[idx] if idx < 64 else 0
                if v == 0:
                    continue  # skip empty
                elif v == -color:
                    opp_count += 1
                elif v == color:
                    n_flips_total[sq] += opp_count
                    break
                else:
                    break
    return n_flips_total


def _flips_skip_empty_vec(flat, color, dir_mask):
    """Flips that skip empty squares in rays (numba-accelerated).

    A ray like B-W-_-W-B would flip both W's (the empty is ignored).
    Only counts flips; returns (n_flips, None).
    """
    n_flips = _flips_skip_empty_inner(flat, color, dir_mask, RAYS, RAY_LENS)
    return n_flips, None


def _flips_wrap_vec(flat, color, dir_mask):
    """Vectorized flips using wrap-around (torus) rays."""
    flat_pad = np.empty(65, dtype=np.int8)
    flat_pad[:64] = flat
    flat_pad[64] = 0

    vals = flat_pad[WRAP_RAYS]  # (64, 8, 7)
    is_opp = (vals == -color) & dir_mask[None, :, None]

    not_opp = ~is_opp
    first_break = not_opp.argmax(axis=2)
    has_break = not_opp.any(axis=2)

    gather = np.take_along_axis(WRAP_RAYS, first_break[:, :, None], axis=2).squeeze(2)
    terminator = flat_pad[gather]

    valid_dir = (first_break > 0) & (terminator == color) & has_break
    n_flips_per_dir = np.where(valid_dir, first_break, 0)
    n_flips = n_flips_per_dir.sum(axis=1)

    return n_flips, valid_dir


# ---------------------------------------------------------------------------
# Scalar find_flips (used only in make_move — called once per turn)
# ---------------------------------------------------------------------------

def find_flips(state, move, color, directions=None):
    """Find all pieces that would be flipped by placing `color` at `move`."""
    if directions is None:
        directions = range(8)
    r, c = move // 8, move % 8
    tbf = []
    for di in directions:
        dr, dc = EIGHTS[di]
        buffer = []
        cur_r, cur_c = r + dr, c + dc
        while 0 <= cur_r <= 7 and 0 <= cur_c <= 7:
            val = state[cur_r, cur_c]
            if val == 0:
                break
            elif val == color:
                tbf.extend(buffer)
                break
            else:
                buffer.append((cur_r, cur_c))
            cur_r += dr
            cur_c += dc
    return tbf


def find_self_flank_flips(state, move, color, directions=None):
    """Find own pieces that would be self-flanked by placing `color` at `move`.

    Self-flanking: a run of own-color pieces terminated by an opponent piece.
    These own pieces get flipped to the opponent's color.
    """
    if directions is None:
        directions = range(8)
    r, c = move // 8, move % 8
    tbf = []
    for di in directions:
        dr, dc = EIGHTS[di]
        buffer = []
        cur_r, cur_c = r + dr, c + dc
        while 0 <= cur_r <= 7 and 0 <= cur_c <= 7:
            val = state[cur_r, cur_c]
            if val == 0:
                break
            elif val == color:
                buffer.append((cur_r, cur_c))
            elif val == -color:
                # Opponent terminates a run of own pieces → self-flank
                tbf.extend(buffer)
                break
            cur_r += dr
            cur_c += dc
    return tbf


@njit(cache=True)
def _find_flips_skip_empty_inner(flat, move, color, dir_mask, rays, ray_lens):
    """Numba-accelerated flip finding for skip-empty variant.

    Returns flat indices of pieces to flip (max 6*8=48 possible).
    Uses a fixed-size array; count indicates how many are valid.
    """
    result = np.empty(48, dtype=np.int32)
    count = 0
    for di in range(8):
        if not dir_mask[di]:
            continue
        ray_len = ray_lens[move, di]
        if ray_len == 0:
            continue
        buf_start = count
        buf_count = 0
        for k in range(ray_len):
            idx = rays[move, di, k]
            v = flat[idx] if idx < 64 else 0
            if v == 0:
                continue
            elif v == -color:
                result[count] = idx
                count += 1
                buf_count += 1
            elif v == color:
                break  # keep the buffer entries
            else:
                # shouldn't happen, but safety
                count -= buf_count  # discard buffer
                break
        else:
            # ray ended without finding own color — discard buffer
            count -= buf_count
    return result, count


def find_flips_skip_empty(state, move, color, directions=None):
    """Find flips skipping over empty squares in rays."""
    flat = state.flatten().astype(np.int8)
    if directions is None:
        dir_mask = DIR_MASK_ALL
    else:
        dir_mask = np.zeros(8, dtype=np.bool_)
        for d in directions:
            dir_mask[d] = True
    result, count = _find_flips_skip_empty_inner(flat, move, color, dir_mask, RAYS, RAY_LENS)
    # Convert flat indices back to (r, c) tuples
    return [(int(idx) // 8, int(idx) % 8) for idx in result[:count]]


def find_flips_wrap(state, move, color, directions=None):
    """Find flips using wrap-around (torus) rays."""
    if directions is None:
        directions = range(8)
    r, c = move // 8, move % 8
    tbf = []
    for di in directions:
        dr, dc = EIGHTS[di]
        buffer = []
        cur_r, cur_c = (r + dr) % 8, (c + dc) % 8
        steps = 0
        while steps < 7:
            if cur_r == r and cur_c == c:
                break
            val = state[cur_r, cur_c]
            if val == 0:
                break
            elif val == color:
                tbf.extend(buffer)
                break
            else:
                buffer.append((cur_r, cur_c))
            cur_r = (cur_r + dr) % 8
            cur_c = (cur_c + dc) % 8
            steps += 1
    return tbf


# ---------------------------------------------------------------------------
# Board class
# ---------------------------------------------------------------------------

class VariantBoard:
    """Othello board that supports variant rules."""

    def __init__(self, variant="normal"):
        self.state = np.zeros((8, 8), dtype=np.int8)
        self.state[3, 4] = 1    # black
        self.state[3, 3] = -1   # white
        self.state[4, 3] = 1    # black
        self.state[4, 4] = -1   # white
        self.next_hand_color = 1  # black first
        self.history = []
        self.variant = variant

        # For locked_flips: track how many times each piece has been flipped
        self.flip_count = np.zeros((8, 8), dtype=np.int8)

        # For delayed_flips: pending flips to apply at start of next move
        self.pending_flips = []  # list of (r, c) to flip

        # Direction mask for this variant
        if variant == "no_diagonal_flips":
            self._dir_mask = DIR_MASK_NO_DIAG
        elif variant == "no_row_flips":
            self._dir_mask = DIR_MASK_NO_ROW
        else:
            self._dir_mask = DIR_MASK_ALL

        # Direction list for scalar find_flips (used in make_move)
        if variant == "no_diagonal_flips":
            self._dirs = [0, 2, 4, 6]
        elif variant == "no_row_flips":
            self._dirs = [0, 1, 3, 4, 5, 7]
        else:
            self._dirs = list(range(8))

    def get_valid_moves(self):
        """Return list of legal moves for current player (vectorized)."""
        color = self.next_hand_color
        flat = self.state.ravel()
        empty = (flat == 0)

        # --- adjacent_legal: any empty square adjacent to any piece ---
        if self.variant == "adjacent_legal":
            occupied = (flat != 0)
            adj_mask = np.zeros(64, dtype=bool)
            for sq in np.where(occupied)[0]:
                for nbr in ADJACENCY[sq]:
                    if flat[nbr] == 0:
                        adj_mask[nbr] = True
            regular = np.where(adj_mask)[0].tolist()
            return regular if regular else []

        # --- capture_any: any empty square adjacent to opponent piece ---
        if self.variant == "capture_any":
            opp_occupied = (flat == -color)
            adj_mask = np.zeros(64, dtype=bool)
            for sq in np.where(opp_occupied)[0]:
                for nbr in ADJACENCY[sq]:
                    if flat[nbr] == 0:
                        adj_mask[nbr] = True
            regular = np.where(adj_mask)[0].tolist()
            return regular if regular else []

        # --- skip_empty_flips: uses skip-empty vectorized flip counts ---
        if self.variant == "skip_empty_flips":
            n_flips, _ = _flips_skip_empty_vec(flat, color, self._dir_mask)
            valid_mask = empty & (n_flips > 0)
            regular = np.where(valid_mask)[0].tolist()
            if regular:
                return regular
            # Forfeit
            opp_flips, _ = _flips_skip_empty_vec(flat, -color, self._dir_mask)
            forfeit_mask = empty & (opp_flips > 0)
            return np.where(forfeit_mask)[0].tolist()

        # --- wrap_flips: uses wrap-around rays ---
        if self.variant == "wrap_flips":
            n_flips, _ = _flips_wrap_vec(flat, color, self._dir_mask)
            valid_mask = empty & (n_flips > 0)
            regular = np.where(valid_mask)[0].tolist()
            if regular:
                return regular
            # Forfeit
            opp_flips, _ = _flips_wrap_vec(flat, -color, self._dir_mask)
            forfeit_mask = empty & (opp_flips > 0)
            return np.where(forfeit_mask)[0].tolist()

        # --- Compute flip counts for all squares at once ---
        if self.variant == "locked_flips":
            n_flips = _flips_locked_vec(flat, color, self._dir_mask,
                                        self.flip_count.ravel())
        else:
            n_flips, _ = _flips_vec(flat, color, self._dir_mask)

        # --- Apply variant-specific filters ---
        if self.variant == "no_same_quadrant" and self.history:
            last_q = QUADRANTS[self.history[-1]]
            empty = empty & (QUADRANTS != last_q)

        # self_flanking: no move restriction — flips happen in make_move

        if self.variant == "max_three_flips":
            valid_mask = empty & (n_flips >= 1) & (n_flips <= 3)
            regular = np.where(valid_mask)[0].tolist()
            if regular:
                return regular
            # Forfeit: opponent's moves with ≤3 flips
            opp_flips, _ = _flips_vec(flat, -color, self._dir_mask)
            forfeit_mask = empty & (opp_flips >= 1) & (opp_flips <= 3)
            return np.where(forfeit_mask)[0].tolist()

        # Regular moves: empty squares with flips
        valid_mask = empty & (n_flips > 0)
        regular = np.where(valid_mask)[0].tolist()
        if regular:
            return regular

        # Forfeit: squares where opponent could flip
        if self.variant == "locked_flips":
            opp_flips = _flips_locked_vec(flat, -color, self._dir_mask,
                                          self.flip_count.ravel())
        else:
            opp_flips, _ = _flips_vec(flat, -color, self._dir_mask)

        forfeit_mask = empty & (opp_flips > 0)
        return np.where(forfeit_mask)[0].tolist()

    def make_move(self, move):
        """Execute a move, updating board state."""
        r, c = move // 8, move % 8
        color = self.next_hand_color

        # delayed_flips: apply pending flips from opponent's last move
        if self.variant == "delayed_flips" and self.pending_flips:
            for fr, fc in self.pending_flips:
                if self.state[fr, fc] != 0:
                    self.state[fr, fc] *= -1
            self.pending_flips = []

        # Find flips for current move (variant-specific)
        if self.variant == "skip_empty_flips":
            flips = find_flips_skip_empty(self.state, move, color, self._dirs)
        elif self.variant == "wrap_flips":
            flips = find_flips_wrap(self.state, move, color, self._dirs)
        elif self.variant in ("adjacent_legal", "capture_any"):
            # Normal flanking flips — if no bracket, piece is just placed
            flips = find_flips(self.state, move, color, self._dirs)
        else:
            flips = find_flips(self.state, move, color, self._dirs)

        if len(flips) == 0 and self.variant not in ("adjacent_legal", "capture_any"):
            # Forfeit — switch color and retry
            color *= -1
            self.next_hand_color *= -1
            if self.variant == "skip_empty_flips":
                flips = find_flips_skip_empty(self.state, move, color, self._dirs)
            elif self.variant == "wrap_flips":
                flips = find_flips_wrap(self.state, move, color, self._dirs)
            else:
                flips = find_flips(self.state, move, color, self._dirs)

        # Variant 4: locked_flips — filter out pieces flipped twice already
        if self.variant == "locked_flips":
            flips = [(r2, c2) for r2, c2 in flips if self.flip_count[r2, c2] < 2]

        # Variant: self_flanking — find own pieces to flip BEFORE applying normal flips
        sf_flips = []
        if self.variant == "self_flanking":
            sf_flips = find_self_flank_flips(self.state, move, color, self._dirs)

        # Apply flips
        if self.variant == "delayed_flips":
            self.pending_flips = list(flips)
        else:
            for fr, fc in flips:
                self.state[fr, fc] *= -1
                if self.variant == "locked_flips":
                    self.flip_count[fr, fc] += 1

        # Apply self-flank flips (own pieces flipped to opponent's color)
        for fr, fc in sf_flips:
            self.state[fr, fc] *= -1

        # Place piece
        self.state[r, c] = color
        self.next_hand_color *= -1
        self.history.append(move)


def generate_game(variant, rng):
    """Generate a single game of the given variant.

    Returns (moves_list, legal_moves_per_turn).
    """
    board = VariantBoard(variant)
    moves = []
    legal_per_turn = []

    for turn in range(60):
        valid = board.get_valid_moves()
        if not valid:
            break
        legal_per_turn.append(valid)
        move = int(valid[rng.randint(len(valid))])
        board.make_move(move)
        moves.append(move)

    return moves, legal_per_turn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True,
                        choices=["no_same_quadrant", "no_diagonal_flips",
                                 "no_row_flips", "locked_flips",
                                 "max_three_flips", "self_flanking",
                                 "delayed_flips", "adjacent_legal",
                                 "skip_empty_flips", "capture_any",
                                 "wrap_flips"])
    parser.add_argument("--num-games", type=int, default=100000)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Variant: {args.variant}", flush=True)
    print(f"Generating {args.num_games} games...", flush=True)
    t0 = time.time()

    all_games = []
    all_legal = []
    lengths = []

    for i in range(args.num_games):
        moves, legal = generate_game(args.variant, rng)
        all_games.append(moves)
        all_legal.append(legal)
        lengths.append(len(moves))

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            avg_len = np.mean(lengths[-10000:])
            print(f"  {i+1}/{args.num_games} games, avg_len={avg_len:.1f}, "
                  f"elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {args.num_games} games in {elapsed:.0f}s")
    print(f"Average game length: {np.mean(lengths):.1f} moves")
    print(f"Min/Max length: {min(lengths)}/{max(lengths)}")

    # Save
    games_path = os.path.join(args.output_dir, "games.pickle")
    with open(games_path, 'wb') as f:
        pickle.dump(all_games, f)
    print(f"Saved games to {games_path}")

    legal_path = os.path.join(args.output_dir, "legal_moves.pickle")
    with open(legal_path, 'wb') as f:
        pickle.dump(all_legal, f)
    print(f"Saved legal moves to {legal_path}")


if __name__ == "__main__":
    main()
