"""
Script to analyze if the model assigns non-trivial probability to illegal moves
"""
import torch
import torch.nn.functional as F
from mingpt.model import GPT, GPTConfig
from mingpt.dataset import CharDataset
from data.othello import OthelloBoardState, permit_reverse
import numpy as np

# Load model
print("Loading model...")
mconf = GPTConfig(vocab_size=61, block_size=59, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

# Load checkpoint (map to CPU since we're on Mac)
device = 'cpu'
model.load_state_dict(torch.load("./ckpts/gpt_championship.ckpt", map_location=device))
model.eval()
print("Model loaded successfully!")

# Example: Analyze a simple game
print("\n" + "="*60)
print("Example Analysis: Testing on a simple game sequence")
print("="*60)

# Start a game
board = OthelloBoardState()

# Get the initial valid moves
initial_valid = board.get_valid_moves()
print(f"Initial valid moves: {[permit_reverse(m) for m in initial_valid]}")

# Play a few legal moves - let's pick from valid moves
game_moves = [initial_valid[0]]  # Play first valid move
board.update([game_moves[0]])

# Continue with a few more moves
for i in range(3):
    valid = board.get_valid_moves()
    if valid:
        next_move = valid[0]
        game_moves.append(next_move)
        board.update([next_move])

# Get valid moves for next position
valid_moves = board.get_valid_moves()
print(f"\nAfter moves: {[permit_reverse(m) for m in game_moves]}")
print(f"Valid next moves: {[permit_reverse(m) for m in valid_moves]}")

# Get model predictions
# Need to convert moves to vocab indices
# (This requires the stoi mapping - we'll use a simplified version)
input_tensor = torch.tensor([game_moves]).long()

with torch.no_grad():
    logits, _ = model(input_tensor)

# Get probabilities for the last position (next move prediction)
last_logits = logits[0, -1, :]  # Shape: [61]
probs = F.softmax(last_logits, dim=-1)

print(f"\nModel's probability distribution:")
print(f"  Total probability mass: {probs.sum().item():.4f}")
print(f"\nTop 5 predicted tokens:")
top_probs, top_indices = torch.topk(probs, 5)
for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
    print(f"  {i+1}. Token {idx.item()}: {prob.item():.4f} ({prob.item()*100:.2f}%)")

# Note: To properly map these back to board positions, we need the stoi/itos mappings
print("\n" + "="*60)
print("NOTE: To complete the analysis, we need to:")
print("1. Map vocab indices back to board positions (0-63)")
print("2. Calculate probability mass on illegal vs legal moves")
print("3. Run this across many game positions")
print("="*60)
