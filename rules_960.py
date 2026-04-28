"""
Enumeration of all 960 valid Othello flanking patterns and shared board conventions used by the transfer-task experiments.

Each pattern is (target_cell, [opponent_cells], terminal_cell, direction, length): a flanking line that legalizes `target_cell` for the side to move. Task B (`incoherent_rules_experiment.py`) selects subsets of these patterns and applies spatial corruptions to them.
"""

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


def enumerate_flanking_patterns():
    """Enumerate all valid flanking patterns on the board.

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


if __name__ == '__main__':
    patterns = enumerate_flanking_patterns()
    print(f"Total flanking patterns: {len(patterns)}")
