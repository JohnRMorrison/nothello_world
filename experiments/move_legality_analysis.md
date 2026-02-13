# Move Legality Analysis for Othello-GPT

This notebook analyzes how often the Othello-GPT model predicts illegal moves. Unlike
simply checking if the top-1 prediction is legal, we examine **all predictions above
a probability threshold** to understand the model's understanding of game rules.

## Key Questions
1. What percentage of high-probability predictions are illegal moves?
2. When does the model make illegal predictions (early/mid/late game)?
3. How "confident" is the model when predicting illegal moves?
4. Which specific positions cause the most errors?

## Setup
This notebook uses:
- `teo_mingpt/` - Modified mingpt utilities that return probability distributions
- `mingpt/` - Original GPT model architecture
- `data/` - Othello game data and board state utilities


```python
%load_ext autoreload
%autoreload 2
```


```python
import os
import sys

# Find project root by looking for the data folder
# Start from current directory and search upward
def find_project_root():
    """Find project root by looking for data/othello_synthetic marker."""
    # First, try common locations
    candidates = [
        os.getcwd(),
        os.path.dirname(os.getcwd()),  # Parent of cwd
        os.path.expanduser("~/phil_proj/nothello_world"),  # Hardcoded fallback
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "data", "othello_synthetic")):
            return path
    raise RuntimeError("Could not find project root. Please run from the nothello_world directory.")

PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Set
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# Our modules
from data import get_othello
from data.othello import OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig
from teo_mingpt.utils import set_seed
```

## Configuration

Adjust these parameters to control the analysis:


```python
# === CONFIGURATION ===

# Dataset: "synthetic" or "championship"
DATASET = "synthetic"

# Model checkpoint: "synthetic" or "championship"
CHECKPOINT = "synthetic"

# Probability threshold - only consider moves with P > threshold
THRESHOLD = 0.001

# Number of games to analyze (set to None for all)
NUM_GAMES = 100

# Random seed for reproducibility
SEED = 42

# Whether to show detailed per-game breakdown
SHOW_FULL_GAME_DETAILS = False
```


```python
set_seed(SEED)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## Load Dataset and Model


```python
# Load dataset
print(f"Loading {DATASET} dataset...")
if DATASET == "synthetic":
    othello = get_othello(ood_num=-1, data_root=None, wthor=True)
    games = othello.val  # Use validation set for synthetic
else:
    othello = get_othello(data_root="data/othello_championship", wthor=True)
    games = othello.sequences  # Championship has no separate val set

train_dataset = CharDataset(othello)
print(f"Dataset has {len(games)} games available")

# Select games to analyze
if NUM_GAMES is not None:
    game_indices = list(range(min(NUM_GAMES, len(games))))
else:
    game_indices = list(range(len(games)))
print(f"Will analyze {len(game_indices)} games")
```


```python
# Load model
print(f"Loading {CHECKPOINT} model...")
mconf = GPTConfig(
    train_dataset.vocab_size,
    train_dataset.block_size,
    n_layer=8,
    n_head=8,
    n_embd=512
)
model = GPT(mconf)

ckpt_path = f"./ckpts/gpt_{CHECKPOINT}.ckpt"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
model = model.to(device)
model.eval()
print("Model loaded successfully")
```

## Data Structures for Tracking Statistics


```python
@dataclass
class IllegalMove:
    """Record of an illegal move prediction."""
    move: str
    probability: float
    game_idx: int
    position: int  # move number in game (1-indexed)
    rank: int = -1  # rank among all predictions (1 = highest probability)


@dataclass
class LegalityStats:
    """Aggregated statistics about move legality."""
    total_positions: int = 0
    total_candidates: int = 0
    illegal_moves: List[IllegalMove] = field(default_factory=list)

    # Per-position tracking
    candidates_by_position: dict = field(default_factory=lambda: defaultdict(int))
    illegal_by_position: dict = field(default_factory=lambda: defaultdict(int))

    # Per-game tracking
    candidates_by_game: dict = field(default_factory=lambda: defaultdict(int))
    illegal_by_game: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def num_illegal(self) -> int:
        return len(self.illegal_moves)

    @property
    def illegal_rate(self) -> float:
        if self.total_candidates == 0:
            return 0.0
        return self.num_illegal / self.total_candidates

    def add_candidates(self, position: int, game_idx: int, count: int):
        self.total_candidates += count
        self.candidates_by_position[position] += count
        self.candidates_by_game[game_idx] += count

    def add_illegal(self, move: IllegalMove):
        self.illegal_moves.append(move)
        self.illegal_by_position[move.position] += 1
        self.illegal_by_game[move.game_idx] += 1
```

## Core Analysis Functions


```python
@torch.no_grad()
def get_candidate_moves(model, x, threshold):
    """
    Get all moves with probability above threshold.

    Args:
        model: The GPT model
        x: Input tensor of shape (1, seq_len)
        threshold: Minimum probability to include a move

    Returns:
        Tuple of (indices, probabilities, all_probs) for moves above threshold.
        all_probs contains probabilities for all tokens (for rank calculation).
    """
    block_size = model.get_block_size()
    x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
    logits, _ = model(x_cond)
    logits = logits[:, -1, :]  # last position
    probs = F.softmax(logits, dim=-1)

    mask = probs[0] > threshold
    indices = torch.where(mask)[0]
    probs_above = probs[0][mask]

    return indices, probs_above, probs[0]


def analyze_game(model, game, game_idx, dataset, device, threshold, stats):
    """
    Analyze all positions in a single game.
    
    For each position, we:
    1. Get all move predictions with P > threshold
    2. Check which ones are illegal according to Othello rules
    3. Record statistics
    """
    position_results = []
    
    for position in range(1, len(game)):
        context = game[:position]
        x = torch.tensor(
            [dataset.stoi[s] for s in context],
            dtype=torch.long
        )[None, ...].to(device)

        # Get candidate moves
        indices, probs, all_probs = get_candidate_moves(model, x, threshold)
        stats.total_positions += 1
        stats.add_candidates(position, game_idx, len(indices))

        # Get valid moves for current board state
        board = OthelloBoardState()
        board.update(context, prt=False)
        valid_moves = set(board.get_valid_moves())

        # Compute ranks: for each move, rank is 1 + number of moves with higher probability
        sorted_indices = torch.argsort(all_probs, descending=True)
        rank_map = {int(idx): rank + 1 for rank, idx in enumerate(sorted_indices)}

        # Check each candidate
        for idx, prob in zip(indices, probs):
            move_int = dataset.itos[int(idx)]
            if move_int == -100:  # padding token
                continue
            if move_int not in valid_moves:
                move_str = permit_reverse(move_int)
                move_rank = rank_map.get(int(idx), -1)
                stats.add_illegal(IllegalMove(
                    move=move_str,
                    probability=prob.item(),
                    game_idx=game_idx,
                    position=position,
                    rank=move_rank,
                ))
```

## Run the Analysis


```python
stats = LegalityStats()

print(f"Analyzing {len(game_indices)} games with threshold={THRESHOLD}...")

for game_idx in tqdm(game_indices, desc="Games"):
    game = games[game_idx]
    analyze_game(
        model=model,
        game=game,
        game_idx=game_idx,
        dataset=train_dataset,
        device=device,
        threshold=THRESHOLD,
        stats=stats,
    )

print("Analysis complete!")
```

## Results: Overall Statistics


```python
print("=" * 60)
print("MOVE LEGALITY ANALYSIS RESULTS")
print("=" * 60)

print(f"\nConfiguration:")
print(f"  Dataset: {DATASET}")
print(f"  Model: {CHECKPOINT}")
print(f"  Probability threshold: {THRESHOLD}")
print(f"  Games analyzed: {len(game_indices)}")

print(f"\nOverall Statistics:")
print(f"  Total positions analyzed: {stats.total_positions:,}")
print(f"  Total candidate moves (P > {THRESHOLD}): {stats.total_candidates:,}")
print(f"  Illegal moves: {stats.num_illegal:,}")
print(f"  Illegal rate: {stats.illegal_rate * 100:.4f}%")
```

## Results: Illegal Moves by Probability Range

This shows how "confident" the model is when making illegal predictions.
High-probability illegal moves are more concerning than low-probability ones.


```python
if stats.num_illegal > 0:
    prob_ranges = [
        (0.5, 1.0, "≥50%"),
        (0.1, 0.5, "10-50%"),
        (0.01, 0.1, "1-10%"),
        (0.001, 0.01, "0.1-1%"),
        (0.0001, 0.001, "0.01-0.1%"),
        (0.0, 0.0001, "<0.01%"),
    ]

    print("\nIllegal Moves by Probability Range:")
    range_counts = []
    for low, high, label in prob_ranges:
        count = sum(1 for m in stats.illegal_moves if low <= m.probability < high)
        if count > 0:
            print(f"  {label}: {count}")
            range_counts.append((label, count))
    
    # Visualization
    if range_counts:
        labels, counts = zip(*range_counts)
        plt.figure(figsize=(10, 5))
        plt.bar(labels, counts)
        plt.xlabel("Probability Range")
        plt.ylabel("Number of Illegal Moves")
        plt.title("Illegal Move Predictions by Confidence Level")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
else:
    print("\nNo illegal moves found! The model perfectly predicts only legal moves.")
```

## Results: Illegal Moves by Game Phase

Does the model make more errors early, mid, or late game?


```python
if stats.num_illegal > 0:
    early = sum(1 for m in stats.illegal_moves if m.position <= 20)
    mid = sum(1 for m in stats.illegal_moves if 20 < m.position <= 40)
    late = sum(1 for m in stats.illegal_moves if m.position > 40)

    print("Illegal Moves by Game Phase:")
    print(f"  Early game (moves 1-20): {early}")
    print(f"  Mid game (moves 21-40): {mid}")
    print(f"  Late game (moves 41+): {late}")

    # Visualization
    phases = ['Early (1-20)', 'Mid (21-40)', 'Late (41+)']
    phase_counts = [early, mid, late]
    
    plt.figure(figsize=(8, 5))
    plt.bar(phases, phase_counts, color=['green', 'orange', 'red'])
    plt.xlabel("Game Phase")
    plt.ylabel("Number of Illegal Moves")
    plt.title("Illegal Move Predictions by Game Phase")
    plt.tight_layout()
    plt.show()
```

## Results: Rank Statistics

How highly does the model rank its illegal predictions?
A rank of 1 means the illegal move was the model's TOP prediction.


```python
if stats.num_illegal > 0:
    all_ranks = [m.rank for m in stats.illegal_moves if m.rank > 0]
    
    if all_ranks:
        rank_1_count = sum(1 for r in all_ranks if r == 1)
        rank_top3_count = sum(1 for r in all_ranks if r <= 3)
        
        print("Rank Statistics (how high illegal moves were ranked):")
        print(f"  Rank 1 (top prediction was illegal): {rank_1_count} ({rank_1_count/len(all_ranks)*100:.1f}%)")
        print(f"  Rank 1-3 (illegal in top 3): {rank_top3_count} ({rank_top3_count/len(all_ranks)*100:.1f}%)")
        print(f"  Average rank of illegal moves: {sum(all_ranks)/len(all_ranks):.1f}")
        print(f"  Median rank of illegal moves: {sorted(all_ranks)[len(all_ranks)//2]}")
        print(f"  Best (lowest) rank: {min(all_ranks)}")

        # Rank distribution
        rank_buckets = [(1, 1), (2, 3), (4, 10), (11, 20), (21, 100)]
        print(f"\n  Rank distribution:")
        for low, high in rank_buckets:
            count = sum(1 for r in all_ranks if low <= r <= high)
            if count > 0:
                label = f"#{low}" if low == high else f"#{low}-{high}"
                print(f"    {label}: {count} ({count/len(all_ranks)*100:.1f}%)")
```

## Results: Top Illegal Moves by Probability

The most "confident" wrong predictions:


```python
if stats.num_illegal > 0:
    sorted_illegal = sorted(
        stats.illegal_moves,
        key=lambda m: m.probability,
        reverse=True
    )

    print("Top 10 Highest-Probability Illegal Moves:")
    for m in sorted_illegal[:10]:
        print(f"  {m.move}: {m.probability:.4f} (game {m.game_idx}, move {m.position}, rank #{m.rank})")
```

## Results: Per-Game Statistics


```python
if stats.num_illegal > 0:
    games_with_illegal = len(stats.illegal_by_game)
    games_without_illegal = len(game_indices) - games_with_illegal

    print("Per-Game Statistics:")
    print(f"  Games with illegal moves: {games_with_illegal}/{len(game_indices)} ({games_with_illegal/len(game_indices)*100:.1f}%)")
    print(f"  Games without illegal moves: {games_without_illegal}/{len(game_indices)} ({games_without_illegal/len(game_indices)*100:.1f}%)")

    if games_with_illegal > 0:
        # Distribution of illegal moves per game
        illegal_counts = list(stats.illegal_by_game.values())
        avg_illegal = sum(illegal_counts) / len(illegal_counts)
        max_illegal = max(illegal_counts)
        print(f"\nIllegal Moves per Game (among games with errors):")
        print(f"  Average: {avg_illegal:.1f}")
        print(f"  Max: {max_illegal}")

        # Histogram
        plt.figure(figsize=(10, 5))
        plt.hist(illegal_counts, bins=range(0, max(illegal_counts)+2), edgecolor='black', alpha=0.7)
        plt.xlabel("Number of Illegal Moves per Game")
        plt.ylabel("Number of Games")
        plt.title("Distribution of Illegal Moves per Game")
        plt.tight_layout()
        plt.show()
```

## Results: Worst Games by Error Rate


```python
if stats.num_illegal > 0 and len(stats.illegal_by_game) > 0:
    # Calculate per-game error rates
    game_error_rates = []
    for game_idx in stats.illegal_by_game:
        illegal_count = stats.illegal_by_game[game_idx]
        candidate_count = stats.candidates_by_game[game_idx]
        rate = illegal_count / candidate_count if candidate_count > 0 else 0
        game_error_rates.append((game_idx, illegal_count, candidate_count, rate))

    # Sort by error rate descending
    game_error_rates.sort(key=lambda x: x[3], reverse=True)

    print("Top 10 Games by Error Rate:")
    for game_idx, illegal, candidates, rate in game_error_rates[:10]:
        print(f"  Game {game_idx}: {illegal}/{candidates} illegal ({rate*100:.2f}%)")
```

## Detailed Per-Position Analysis (Optional)

Set `SHOW_FULL_GAME_DETAILS = True` in the configuration to see detailed breakdowns.


```python
if SHOW_FULL_GAME_DETAILS and stats.num_illegal > 0:
    # Group illegal moves by game
    illegal_by_game_detail = defaultdict(list)
    for m in stats.illegal_moves:
        illegal_by_game_detail[m.game_idx].append(m)
    
    print("\n" + "=" * 60)
    print("DETAILED PER-GAME BREAKDOWN")
    print("=" * 60)
    
    for game_idx in sorted(illegal_by_game_detail.keys()):
        illegal_list = illegal_by_game_detail[game_idx]
        print(f"\nGame {game_idx} ({len(illegal_list)} illegal predictions):")
        for m in sorted(illegal_list, key=lambda x: x.position):
            print(f"  Position {m.position}: {m.move} (P={m.probability:.4f}, rank #{m.rank})")
```

## Summary

This analysis reveals:
1. **Illegal rate**: What percentage of high-probability predictions are illegal
2. **Confidence level**: Whether the model is "confident" about illegal predictions
3. **Game phase**: When errors occur most frequently
4. **Rank analysis**: Whether illegal moves are top predictions or low-probability alternatives

A well-trained model should have:
- Very low illegal rate (< 1%)
- No high-probability illegal predictions (> 10%)
- No rank-1 illegal predictions (top prediction should always be legal)
