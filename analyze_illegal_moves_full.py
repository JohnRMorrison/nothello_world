"""
Full analysis script to determine if the model assigns non-trivial probability to illegal moves
"""
import torch
import torch.nn.functional as F
from mingpt.model import GPT, GPTConfig
from mingpt.dataset import CharDataset
from data.othello import OthelloBoardState, permit_reverse
import numpy as np

# Vocabulary mappings (token index -> board position)
itos = {
    0: -100, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18,
    20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 26: 25, 27: 26, 28: 29,
    29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 37, 35: 38, 36: 39, 37: 40,
    38: 41, 39: 42, 40: 43, 41: 44, 42: 45, 43: 46, 44: 47, 45: 48, 46: 49,
    47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 56, 54: 57, 55: 58,
    56: 59, 57: 60, 58: 61, 59: 62, 60: 63,
}

# Board position -> token index
stoi = {
    -100: 0, -1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9,
    9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18,
    18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27,
    29: 28, 30: 29, 31: 30, 32: 31, 33: 32, 34: 33, 37: 34, 38: 35, 39: 36,
    40: 37, 41: 38, 42: 39, 43: 40, 44: 41, 45: 42, 46: 43, 47: 44, 48: 45,
    49: 46, 50: 47, 51: 48, 52: 49, 53: 50, 54: 51, 55: 52, 56: 53, 57: 54,
    58: 55, 59: 56, 60: 57, 61: 58, 62: 59, 63: 60,
}

print("="*70)
print("ILLEGAL MOVE PROBABILITY ANALYSIS")
print("="*70)

# Load model
print("\n[1/5] Loading model...")
mconf = GPTConfig(vocab_size=61, block_size=59, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)
device = 'cpu'
model.load_state_dict(torch.load("./ckpts/gpt_championship.ckpt", map_location=device))
model.eval()
print("✓ Model loaded successfully!")

def analyze_position(board, game_moves):
    """
    Analyze a single board position:
    - Get model's predictions
    - Map to board positions
    - Separate legal vs illegal move probabilities
    """
    # Get valid moves at this position
    valid_moves = board.get_valid_moves()

    # Convert game moves to token indices for model input
    token_sequence = [stoi[move] for move in game_moves]
    input_tensor = torch.tensor([token_sequence]).long()

    # Get model predictions
    with torch.no_grad():
        logits, _ = model(input_tensor)

    # Get probabilities for next move (last position in sequence)
    last_logits = logits[0, -1, :]  # Shape: [61]
    probs = F.softmax(last_logits, dim=-1)

    # Map token probabilities back to board positions
    board_probs = {}
    for token_idx in range(1, 61):  # Skip token 0 (padding)
        board_pos = itos[token_idx]
        if board_pos != -100:  # Skip padding marker
            board_probs[board_pos] = probs[token_idx].item()

    # Separate legal and illegal move probabilities
    legal_probs = {pos: board_probs.get(pos, 0.0) for pos in valid_moves}
    illegal_probs = {pos: prob for pos, prob in board_probs.items() if pos not in valid_moves}

    # Calculate statistics
    total_legal_prob = sum(legal_probs.values())
    total_illegal_prob = sum(illegal_probs.values())

    # Find top predictions
    sorted_board_probs = sorted(board_probs.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_board_probs[:5]

    return {
        'valid_moves': valid_moves,
        'legal_probs': legal_probs,
        'illegal_probs': illegal_probs,
        'total_legal_prob': total_legal_prob,
        'total_illegal_prob': total_illegal_prob,
        'top_5_predictions': top_5,
        'num_legal_moves': len(valid_moves),
        'num_illegal_moves': len(illegal_probs),
    }

# Example 1: Analyze a single position
print("\n[2/5] Analyzing example game position...")
board = OthelloBoardState()

# Play a few moves
initial_valid = board.get_valid_moves()
game_moves = [initial_valid[0]]
board.update([game_moves[0]])

for i in range(3):
    valid = board.get_valid_moves()
    if valid:
        next_move = valid[0]
        game_moves.append(next_move)
        board.update([next_move])

print(f"Game sequence: {[permit_reverse(m) for m in game_moves]}")

# Analyze this position
result = analyze_position(board, game_moves)

print(f"\n{'='*70}")
print(f"RESULTS FOR THIS POSITION:")
print(f"{'='*70}")
print(f"Legal moves ({result['num_legal_moves']}): {[permit_reverse(m) for m in result['valid_moves']]}")
print(f"\nProbability Distribution:")
print(f"  ├─ Total probability on LEGAL moves:   {result['total_legal_prob']:.4f} ({result['total_legal_prob']*100:.2f}%)")
print(f"  └─ Total probability on ILLEGAL moves: {result['total_illegal_prob']:.4f} ({result['total_illegal_prob']*100:.2f}%)")

print(f"\nTop 5 Predictions:")
for i, (pos, prob) in enumerate(result['top_5_predictions'], 1):
    is_legal = "✓ LEGAL" if pos in result['valid_moves'] else "✗ ILLEGAL"
    print(f"  {i}. {permit_reverse(pos).upper():4s} : {prob:.4f} ({prob*100:6.2f}%) {is_legal}")

# Calculate if top prediction is legal
top_pred_pos = result['top_5_predictions'][0][0]
top_is_legal = top_pred_pos in result['valid_moves']
print(f"\nTop prediction is: {'LEGAL ✓' if top_is_legal else 'ILLEGAL ✗'}")

print("\n" + "="*70)
print("[3/5] Running analysis across multiple random games...")
print("="*70)

# Analyze multiple games
num_games = 20
num_moves_per_game = 10
all_results = []

for game_idx in range(num_games):
    board = OthelloBoardState()
    game_moves = []

    for move_idx in range(num_moves_per_game):
        valid = board.get_valid_moves()
        if not valid:
            break

        # Random legal move
        import random
        next_move = random.choice(valid)
        game_moves.append(next_move)
        board.update([next_move])

        # Analyze this position (predict next move after this one)
        if len(game_moves) >= 4:  # Need some history
            result = analyze_position(board, game_moves)
            all_results.append(result)

print(f"✓ Analyzed {len(all_results)} positions across {num_games} games")

# Aggregate statistics
print("\n[4/5] Computing aggregate statistics...")
avg_legal_prob = np.mean([r['total_legal_prob'] for r in all_results])
avg_illegal_prob = np.mean([r['total_illegal_prob'] for r in all_results])
max_illegal_prob = np.max([r['total_illegal_prob'] for r in all_results])
min_illegal_prob = np.min([r['total_illegal_prob'] for r in all_results])

# How often is top prediction legal?
top_pred_legal_count = sum(1 for r in all_results if r['top_5_predictions'][0][0] in r['valid_moves'])
top_pred_accuracy = top_pred_legal_count / len(all_results)

print("\n" + "="*70)
print(f"AGGREGATE RESULTS ACROSS {len(all_results)} POSITIONS:")
print("="*70)
print(f"\nProbability Mass:")
print(f"  ├─ Average probability on LEGAL moves:   {avg_legal_prob:.4f} ({avg_legal_prob*100:.2f}%)")
print(f"  └─ Average probability on ILLEGAL moves: {avg_illegal_prob:.4f} ({avg_illegal_prob*100:.2f}%)")
print(f"\nIllegal Move Probability Range:")
print(f"  ├─ Maximum: {max_illegal_prob:.4f} ({max_illegal_prob*100:.2f}%)")
print(f"  └─ Minimum: {min_illegal_prob:.4f} ({min_illegal_prob*100:.2f}%)")
print(f"\nTop Prediction Analysis:")
print(f"  └─ Top prediction is legal: {top_pred_legal_count}/{len(all_results)} ({top_pred_accuracy*100:.1f}%)")

print("\n[5/5] Final Assessment...")
print("="*70)

# Determine if illegal move probability is "non-trivial"
threshold = 0.01  # 1% threshold for "non-trivial"
if avg_illegal_prob > threshold:
    print(f"⚠️  FINDING: Model assigns NON-TRIVIAL probability to illegal moves!")
    print(f"   Average {avg_illegal_prob*100:.2f}% probability on illegal moves (threshold: {threshold*100}%)")
else:
    print(f"✓ FINDING: Model assigns minimal probability to illegal moves")
    print(f"   Average {avg_illegal_prob*100:.2f}% probability on illegal moves (threshold: {threshold*100}%)")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
