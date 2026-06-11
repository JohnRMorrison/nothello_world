"""
Hand-crafted flanking network for Othello legal move prediction.

Each hidden unit corresponds to one specific flanking pattern (target cell,
direction, flank length). Weights are set analytically, not learned.

Input: 128-d vector (2 features per cell: is_mine, is_opponent)
Hidden: one unit per valid flanking pattern
Output: 60-d (one per playable cell), sigmoid probability of legality
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from data.othello import OthelloBoardState


# Board conventions
CENTER_CELLS = {27, 28, 35, 36}  # (3,3), (3,4), (4,3), (4,4)
VALID_MOVES = sorted(set(range(64)) - CENTER_CELLS)
MOVE_TO_IDX = {pos: i for i, pos in enumerate(VALID_MOVES)}
N_MOVES = 60

# 8 directions: (dr, dc)
DIRECTIONS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),   # up, down, left, right
    (-1, -1), (-1, 1), (1, -1), (1, 1),  # diagonals
]


def enumerate_flanking_patterns_color_specific():
    """Enumerate 1920 color-specific flanking patterns (2 * the standard 960).

    Returns the same patterns as `enumerate_flanking_patterns()` but each
    pattern is duplicated for the two possible mover colors. Each pattern is
    tagged with a `mover` field in {+1, -1}:
      - mover = +1 (black is moving): opponent cells contain WHITE pieces (-1),
                                       terminal contains a BLACK piece (+1).
      - mover = -1 (white is moving): opponent cells contain BLACK pieces (+1),
                                       terminal contains a WHITE piece (-1).

    Patterns fire based on absolute spatial color configurations only — no
    parity (whose-turn-it-is) interpretation needed. This is the comparison
    point for "is color-relative encoding actually harder to learn?"

    Order: first 960 patterns are the mover=+1 (black) versions, then 960
    mover=-1 (white) versions, both in the same order as the original 960
    pattern enumeration.
    """
    base = enumerate_flanking_patterns()
    out = []
    for mover in (+1, -1):
        for p in base:
            out.append({
                'target': p['target'],
                'opponents': p['opponents'],
                'terminal': p['terminal'],
                'direction': p['direction'],
                'length': p['length'],
                'mover': mover,
            })
    return out


def enumerate_flanking_patterns():
    """Enumerate all valid flanking patterns on the board.

    Each pattern is: (target_cell, [opponent_cells], terminal_cell, direction, length)
    A pattern fires when:
      - target cell is empty
      - all opponent cells contain opponent pieces
      - terminal cell contains a friendly piece
    """
    patterns = []
    for target in range(64):
        if target in CENTER_CELLS:
            continue
        row, col = target // 8, target % 8
        for dr, dc in DIRECTIONS:
            for length in range(1, 7):  # 1 to 6 opponent pieces
                opponents = []
                valid = True
                for k in range(1, length + 1):
                    r, c = row + k * dr, col + k * dc
                    if not (0 <= r < 8 and 0 <= c < 8):
                        valid = False
                        break
                    opponents.append(r * 8 + c)
                if not valid:
                    continue
                # Terminal cell (friendly piece)
                tr, tc = row + (length + 1) * dr, col + (length + 1) * dc
                if not (0 <= tr < 8 and 0 <= tc < 8):
                    continue
                terminal = tr * 8 + tc
                patterns.append({
                    'target': target,
                    'opponents': opponents,
                    'terminal': terminal,
                    'direction': (dr, dc),
                    'length': length,
                })
    return patterns


class HandCraftedFlanking(nn.Module):
    """Hand-crafted network that detects all Othello flanking patterns.

    Input: 128-d (64 cells × 2 features: is_mine, is_opponent)
    Hidden: one ReLU unit per flanking pattern
    Output: 60-d sigmoid (one per playable cell)
    """

    def __init__(self, patterns):
        super().__init__()
        num_patterns = len(patterns)
        self.patterns = patterns

        # Hidden layer: 128-d input → num_patterns hidden units
        self.hidden = nn.Linear(128, num_patterns)
        self.hidden.weight.data.zero_()
        self.hidden.bias.data.zero_()

        # Output layer: num_patterns → 60 playable cells
        self.output = nn.Linear(num_patterns, N_MOVES)
        self.output.weight.data.zero_()
        self.output.bias.data.fill_(-0.1)

        PENALTY = 10.0  # large enough to kill the unit

        for j, pat in enumerate(patterns):
            target = pat['target']
            opponents = pat['opponents']
            terminal = pat['terminal']
            length = len(opponents)

            # Target cell must be empty: penalize both is_mine and is_opponent
            self.hidden.weight.data[j, 2 * target] = -PENALTY      # is_mine
            self.hidden.weight.data[j, 2 * target + 1] = -PENALTY  # is_opponent

            # Each opponent cell must have opponent piece
            for opp in opponents:
                self.hidden.weight.data[j, 2 * opp + 1] = 1.0  # is_opponent

            # Terminal cell must have friendly piece
            self.hidden.weight.data[j, 2 * terminal] = 1.0  # is_mine

            # Bias: fires only when all conditions met
            self.hidden.bias.data[j] = -(length + 0.5)

            # Output: connect this pattern to its target cell
            playable_idx = MOVE_TO_IDX[target]
            self.output.weight.data[playable_idx, j] = 1.0

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """x: (batch, 128) — is_mine/is_opponent encoding"""
        h = torch.relu(self.hidden(x))
        return torch.sigmoid(self.output(h))


def encode_board(board_state, is_even_turn):
    """Encode an OthelloBoardState as 128-d vector.

    board_state: (8, 8) numpy array with -1=white, 0=empty, 1=black
    is_even_turn: True if it's even turn (white's turn to move)

    Returns: (128,) tensor with (is_mine, is_opponent) per cell
    """
    flat = board_state.flatten()  # 64 values
    encoding = np.zeros(128, dtype=np.float32)

    if is_even_turn:
        # Black's turn (black moves first): black=mine, white=opponent
        my_val, opp_val = 1, -1
    else:
        # White's turn: white=mine, black=opponent
        my_val, opp_val = -1, 1

    for i in range(64):
        if flat[i] == my_val:
            encoding[2 * i] = 1.0      # is_mine
        elif flat[i] == opp_val:
            encoding[2 * i + 1] = 1.0  # is_opponent

    return torch.tensor(encoding)


def validate(num_games=1000, seed=42):
    """Validate the hand-crafted network on random Othello games."""
    patterns = enumerate_flanking_patterns()
    print(f"Total flanking patterns: {len(patterns)}")

    even_net = HandCraftedFlanking(patterns)
    odd_net = HandCraftedFlanking(patterns)
    even_net.eval()
    odd_net.eval()

    # Stats
    total_positions = 0
    total_correct = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    rng = np.random.RandomState(seed)

    for game_idx in range(num_games):
        board = OthelloBoardState()
        for turn in range(60):
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                # Check if other player can move
                board.update([])  # pass
                valid_moves = board.get_valid_moves()
                if not valid_moves:
                    break  # game over

            is_black_turn = (board.next_hand_color == 1)

            # Ground truth: 60-d binary
            gt_legal = np.zeros(N_MOVES, dtype=np.float32)
            for mv in valid_moves:
                if mv in MOVE_TO_IDX:
                    gt_legal[MOVE_TO_IDX[mv]] = 1.0

            # Prediction (even=black's turn, odd=white's turn)
            x = encode_board(board.state, is_black_turn).unsqueeze(0)
            net = even_net if is_black_turn else odd_net
            with torch.no_grad():
                pred_probs = net(x).squeeze(0).numpy()
            pred_legal = (pred_probs > 0.5).astype(np.float32)

            # Metrics
            tp = ((pred_legal == 1) & (gt_legal == 1)).sum()
            fp = ((pred_legal == 1) & (gt_legal == 0)).sum()
            fn = ((pred_legal == 0) & (gt_legal == 1)).sum()
            tn = ((pred_legal == 0) & (gt_legal == 0)).sum()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            total_positions += 1
            total_correct += (pred_legal == gt_legal).all()

            # Play a random valid move
            move = valid_moves[rng.randint(len(valid_moves))]
            board.update([move])

        if (game_idx + 1) % 100 == 0:
            print(f"  {game_idx + 1}/{num_games} games...")

    # Report
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)

    print(f"\n{'='*60}")
    print(f"Hand-Crafted Flanking Network Validation")
    print(f"{'='*60}")
    print(f"Games: {num_games}")
    print(f"Positions evaluated: {total_positions}")
    print(f"Positions with perfect prediction: {total_correct}/{total_positions} "
          f"({100*total_correct/total_positions:.2f}%)")
    print(f"\nPer-square metrics:")
    print(f"  Accuracy: {100*accuracy:.4f}%")
    print(f"  Precision (P): {precision:.4f}")
    print(f"  Recall (R): {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  True positives (TP): {total_tp}")
    print(f"  False positives (FP): {total_fp}")
    print(f"  False negatives (FN): {total_fn}")
    print(f"  True negatives (TN): {total_tn}")
    print(f"\nNetwork structure:")
    print(f"  Input: 128-d (64 cells × 2: is_mine, is_opponent)")
    print(f"  Hidden: {len(patterns)} units (one per flanking pattern)")
    print(f"  Output: {N_MOVES}-d (one per playable cell)")

    return patterns, even_net, odd_net


if __name__ == '__main__':
    patterns, even_net, odd_net = validate(num_games=1000)

    # Save patterns and network
    save_path = 'hand_crafted_flanking_patterns.pt'
    torch.save({
        'patterns': patterns,
        'state_dict_even': even_net.state_dict(),
        'state_dict_odd': odd_net.state_dict(),
    }, save_path)
    print(f"\nSaved to {save_path}")
