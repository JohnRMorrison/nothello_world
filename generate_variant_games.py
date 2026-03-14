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

Usage:
  python generate_variant_games.py --variant no_diagonal_flips --num-games 100000 \
      --output-dir experiments/variants/no_diagonal_flips --seed 42
"""

import argparse
import os
import pickle
import time

import numpy as np

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

# Indices into EIGHTS
ROWS_ONLY = [2, 6]           # right, left
COLS_ONLY = [0, 4]           # up, down
DIAG_ONLY = [1, 3, 5, 7]    # diagonals
NON_DIAG  = [0, 2, 4, 6]    # rows + cols
NON_ROW   = [0, 1, 3, 4, 5, 7]  # cols + diags

CENTER = {27, 28, 35, 36}


def quadrant_of(move):
    """Return quadrant index 0-3 for a board position 0-63."""
    r, c = move // 8, move % 8
    return (0 if r < 4 else 2) + (0 if c < 4 else 1)


def find_flips(state, move, color, directions=None):
    """Find all pieces that would be flipped by placing `color` at `move`.

    Returns list of (r, c) pairs to flip.
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
                tbf.extend(buffer)
                break
            else:
                buffer.append((cur_r, cur_c))
            cur_r += dr
            cur_c += dc
    return tbf


def find_self_flips(state, move, color, directions=None):
    """Find own pieces flanked by placing `color` at `move`.

    Flanking own pieces means: color ... color ... -color (or edge).
    Actually: we look for lines where we place `color`, then see a run
    of `color` (own pieces) terminated by `-color` (opponent).
    Those own pieces get flipped to opponent.
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
                # own piece — add to buffer (these might get flipped)
                buffer.append((cur_r, cur_c))
            else:
                # opponent piece terminates the line — flip the buffer
                tbf.extend(buffer)
                break
            cur_r += dr
            cur_c += dc
    return tbf


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

    def _get_flip_directions(self):
        """Return which direction indices to use for flipping."""
        if self.variant == "no_diagonal_flips":
            return NON_DIAG
        elif self.variant == "no_row_flips":
            return NON_ROW
        else:
            return range(8)

    def get_valid_moves(self):
        """Return list of legal moves for current player."""
        color = self.next_hand_color
        dirs = self._get_flip_directions()
        regular = []
        forfeit = []

        for move in range(64):
            r, c = move // 8, move % 8
            if self.state[r, c] != 0:
                continue

            # Variant 1: no_same_quadrant
            if self.variant == "no_same_quadrant" and len(self.history) > 0:
                if quadrant_of(move) == quadrant_of(self.history[-1]):
                    continue

            # Variant 6: self_flanking — must avoid flanking own pieces
            if self.variant == "self_flanking":
                own_flips = find_self_flips(self.state, move, color)
                if len(own_flips) > 0:
                    continue  # illegal — would flank own pieces

            # Find flips
            flips = find_flips(self.state, move, color, dirs)

            # Variant 4: locked_flips — remove flips on pieces flipped twice already
            if self.variant == "locked_flips":
                flips = [(r2, c2) for r2, c2 in flips if self.flip_count[r2, c2] < 2]

            # Variant 5: max_three_flips — only legal if flips <= 3 pieces
            if self.variant == "max_three_flips":
                if 1 <= len(flips) <= 3:
                    regular.append(move)
                # Don't add forfeit moves with wrong flip count
                continue

            if len(flips) > 0:
                regular.append(move)
            else:
                # Check if opponent could use this square (forfeit)
                opp_flips = find_flips(self.state, move, -color, dirs)
                if self.variant == "locked_flips":
                    opp_flips = [(r2, c2) for r2, c2 in opp_flips
                                 if self.flip_count[r2, c2] < 2]
                if len(opp_flips) > 0:
                    forfeit.append(move)

        if regular:
            return regular
        elif forfeit:
            return forfeit
        return []

    def make_move(self, move):
        """Execute a move, updating board state."""
        r, c = move // 8, move % 8
        color = self.next_hand_color
        dirs = self._get_flip_directions()

        # Variant 7: apply pending flips from last turn
        if self.variant == "delayed_flips" and self.pending_flips:
            for fr, fc in self.pending_flips:
                # Only flip if the piece is still the same color it was when captured
                # (it might have been overwritten by a new placement)
                if self.state[fr, fc] != 0:
                    self.state[fr, fc] *= -1
            self.pending_flips = []

        # Find flips for current move
        flips = find_flips(self.state, move, color, dirs)

        if len(flips) == 0:
            # Forfeit — switch color and retry
            color *= -1
            self.next_hand_color *= -1
            flips = find_flips(self.state, move, color, dirs)

        # Variant 4: locked_flips — filter out pieces flipped twice already
        if self.variant == "locked_flips":
            flips = [(r2, c2) for r2, c2 in flips if self.flip_count[r2, c2] < 2]

        # Apply flips
        if self.variant == "delayed_flips":
            # Don't flip now — queue them for next turn
            self.pending_flips = list(flips)
        else:
            for fr, fc in flips:
                self.state[fr, fc] *= -1
                if self.variant == "locked_flips":
                    self.flip_count[fr, fc] += 1

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
                                 "delayed_flips"])
    parser.add_argument("--num-games", type=int, default=100000)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Variant: {args.variant}")
    print(f"Generating {args.num_games} games...")
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
                  f"elapsed={elapsed:.0f}s")

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
