"""
Heatmap Analysis - Uses the pickle file containing the dictionary and builds
an 8x8 heatmap containing squares where the model predicts illegally most often.

Captures all illegal probability at every position. (previously only counted positions where illegal prob >5%)
"""

import pickle
import numpy as np

print("Loading axon_full_data.pkl...")
with open('axon_full_data.pkl', 'rb') as f:
    data = pickle.load(f)

game_data = data['game_data']
itos = data['itos']
total_games = len(game_data)
print(f"Loaded {total_games} games")

# analyze illegal predictions
square_counts = np.zeros(64)   # count each time a square is predicted illegally (>1% prob)
square_probs = np.zeros(64)    # total probability assigned illegally to each square
total_positions = 0
all_illegal_probs = []         # illegal prob for every position (removed previous 5% threshold)

for gi, game_idx in enumerate(game_data):
    for move_idx in game_data[game_idx]:
        total_positions += 1
        position_illegal = 0.0

        for token_idx in game_data[game_idx][move_idx]:
            entry = game_data[game_idx][move_idx][token_idx]

            if not entry['legal']:
                position_illegal += entry['prob']

                # only count the times where the model assigned >1%
                if entry['prob'] > 0.01:
                    board_pos = itos.get(token_idx, -100)
                    if board_pos >= 0:
                        square_counts[board_pos] += 1
                        square_probs[board_pos] += entry['prob']

        all_illegal_probs.append(position_illegal)

    if (gi + 1) % 10000 == 0:
        print(f"  {gi+1}/{total_games} games...")

all_illegal_probs = np.array(all_illegal_probs)

# === SUMMARY ===
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total games: {total_games}")
print(f"Total positions: {total_positions:,}")
print(f"Total illegal probability across all positions: {all_illegal_probs.sum():.2f}")
print(f"Average illegal prob per position: {all_illegal_probs.mean()*100:.4f}%")
print(f"Median illegal prob per position: {np.median(all_illegal_probs)*100:.4f}%")
print(f"Max illegal prob at any position: {all_illegal_probs.max()*100:.2f}%")
print(f"Std dev: {all_illegal_probs.std()*100:.4f}%")

# distribution breakdown - how many positions fall into each bucket
print(f"\n{'='*60}")
print("ILLEGAL PROBABILITY DISTRIBUTION")
print("How many positions fall into each illegal probability range")
print("="*60)
buckets = [0, 0.001, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.01]
labels = ["0% (perfect)", "0-0.1%", "0.1-1%", "1-2%", "2-5%", "5-10%", "10-20%", "20-50%", "50%+"]
for i in range(len(buckets) - 1):
    count = np.sum((all_illegal_probs >= buckets[i]) & (all_illegal_probs < buckets[i+1]))
    pct = count / total_positions * 100
    print(f"  {labels[i]:>12s}: {count:>10,} positions ({pct:6.2f}%)")
# exact zero
exact_zero = np.sum(all_illegal_probs == 0)
print(f"  {'Exact zero':>12s}: {exact_zero:>10,} positions ({exact_zero/total_positions*100:6.2f}%)")

# percentiles
print(f"\n{'='*60}")
print("PERCENTILES")
print("="*60)
for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
    val = np.percentile(all_illegal_probs, p)
    print(f"  {p:5.1f}th percentile: {val*100:.4f}%")

# reshaping into 8x8 board
count_board = square_counts.reshape(8, 8)
prob_board = square_probs.reshape(8, 8)

# print count heatmap
print(f"\n{'='*60}")
print("ILLEGAL PREDICTION COUNTS")
print("How many times each square was predicted illegally (>1% prob)")
print("="*60)
print("      A     B     C     D     E     F     G     H")
for row in range(8):
    cells = [f"{int(count_board[row][col]):5d}" for col in range(8)]
    print(f"  {row+1}  {' '.join(cells)}")

# print probability heatmap
print(f"\n{'='*60}")
print("TOTAL ILLEGAL PROBABILITY ASSIGNED")
print("Sum of all probability incorrectly assigned to each square")
print("="*60)
print("      A     B     C     D     E     F     G     H")
for row in range(8):
    cells = [f"{prob_board[row][col]:5.1f}" for col in range(8)]
    print(f"  {row+1}  {' '.join(cells)}")

# top 10 worst squares
print(f"\n{'='*60}")
print("TOP 10 WORST SQUARES")
print("="*60)

top_squares = np.argsort(square_counts)[::-1][:10]

for rank, pos in enumerate(top_squares):
    row = pos // 8
    col = pos % 8
    col_letter = "ABCDEFGH"[col]
    print(f"  {rank+1:2d}. {col_letter}{row+1}: "
          f"{int(square_counts[pos]):,} times, "
          f"{square_probs[pos]:.1f} total prob")

# classify top squares
print(f"\n{'='*60}")
print("TOP 10 BY POSITION TYPE")
print("="*60)

center = 0
edge = 0
corner = 0

for pos in top_squares:
    row = pos // 8
    col = pos % 8
    if (row in [0, 7]) and (col in [0, 7]):
        corner += 1
    elif row in [0, 7] or col in [0, 7]:
        edge += 1
    else:
        center += 1

print(f"  Center: {center}")
print(f"  Edge:   {edge}")
print(f"  Corner: {corner}")
