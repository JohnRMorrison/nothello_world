"""
Analyze failure cases where models assign high probability to illegal moves
"""
import torch
import torch.nn.functional as F
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState, permit, permit_reverse
import numpy as np
import glob
import re
from collections import defaultdict

# Vocabulary mappings
itos = {
    0: -100, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18,
    20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 26: 25, 27: 26, 28: 29,
    29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 37, 35: 38, 36: 39, 37: 40,
    38: 41, 39: 42, 40: 43, 41: 44, 42: 45, 43: 46, 44: 47, 45: 48, 46: 49,
    47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 56, 54: 57, 55: 58,
    56: 59, 57: 60, 58: 61, 59: 62, 60: 63,
}

stoi = {
    -100: 0, -1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9,
    9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18,
    18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27,
    29: 28, 30: 29, 31: 30, 32: 31, 33: 32, 34: 33, 37: 34, 38: 35, 39: 36,
    40: 37, 41: 38, 42: 39, 43: 40, 44: 41, 45: 42, 46: 43, 47: 44, 48: 45,
    49: 46, 50: 47, 51: 48, 52: 49, 53: 50, 54: 51, 55: 52, 56: 53, 57: 54,
    58: 55, 59: 56, 60: 57, 61: 58, 62: 59, 63: 60,
}

def parse_pgn_file(filepath):
    """Parse a PGN file and extract game move sequences"""
    games = []
    with open(filepath, 'r') as f:
        content = f.read()

    game_blocks = re.split(r'\[Event', content)

    for block in game_blocks[1:]:
        moves_match = re.search(r'\n((?:\d+\.\s+\w+\s+\w+\s*)+)', block)
        if moves_match:
            moves_text = moves_match.group(1)
            moves = re.findall(r'[A-H][1-8]', moves_text)
            move_positions = []
            for move_str in moves:
                pos = permit(move_str)
                if pos != -1:
                    move_positions.append(pos)
            if len(move_positions) >= 10:
                games.append(move_positions)

    return games

def load_championship_games(data_dir, max_games=None):
    """Load games from championship PGN files"""
    pgn_files = glob.glob(f"{data_dir}/othello_championship/*.pgn")
    all_games = []

    for pgn_file in pgn_files:
        games = parse_pgn_file(pgn_file)
        all_games.extend(games)
        if max_games and len(all_games) >= max_games:
            break

    if max_games:
        all_games = all_games[:max_games]

    return all_games

def visualize_board(board_state):
    """Pretty print the board state"""
    rows = "ABCDEFGH"
    print("\n    1 2 3 4 5 6 7 8")
    print("  ┌─────────────────┐")
    for i, row in enumerate(board_state.state):
        pieces = []
        for cell in row:
            if cell == -1:
                pieces.append("⚪")
            elif cell == 1:
                pieces.append("⚫")
            else:
                pieces.append("·")
        print(f"{rows[i]} │ {' '.join(pieces)} │")
    print("  └─────────────────┘")

def analyze_position_detailed(model, board, game_moves, device='cpu'):
    """Get detailed analysis of a position"""
    valid_moves = board.get_valid_moves()

    # Convert to tokens
    token_sequence = [stoi[move] for move in game_moves]
    input_tensor = torch.tensor([token_sequence]).long().to(device)

    with torch.no_grad():
        logits, _ = model(input_tensor)

    last_logits = logits[0, -1, :]
    probs = F.softmax(last_logits, dim=-1)

    # Map to board positions
    board_probs = {}
    for token_idx in range(1, 61):
        board_pos = itos[token_idx]
        if board_pos != -100:
            board_probs[board_pos] = probs[token_idx].item()

    # Separate legal and illegal
    legal_probs = [(pos, board_probs.get(pos, 0.0)) for pos in valid_moves]
    illegal_probs = [(pos, prob) for pos, prob in board_probs.items() if pos not in valid_moves]

    # Sort by probability
    legal_probs.sort(key=lambda x: x[1], reverse=True)
    illegal_probs.sort(key=lambda x: x[1], reverse=True)

    total_illegal = sum(p for _, p in illegal_probs)

    return {
        'valid_moves': valid_moves,
        'legal_probs': legal_probs,
        'illegal_probs': illegal_probs,
        'total_illegal_prob': total_illegal,
        'board_state': board.state.copy(),
        'game_moves': game_moves.copy(),
    }

def find_failure_cases(model_path, games, model_name, num_failures=10, device='cpu'):
    """Find positions where model assigns high probability to illegal moves"""
    print(f"\n{'='*70}")
    print(f"FINDING FAILURE CASES: {model_name}")
    print(f"{'='*70}")

    # Load model
    print(f"Loading {model_name}...")
    mconf = GPTConfig(vocab_size=61, block_size=59, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Model loaded")

    failure_cases = []
    positions_checked = 0

    print(f"\nSearching for worst failures...")

    for game_idx, game in enumerate(games):
        board = OthelloBoardState()

        for move_idx, move in enumerate(game):
            if move_idx < 4:
                board.update([move])
                continue

            game_so_far = game[:move_idx]
            result = analyze_position_detailed(model, board, game_so_far, device)

            # Store if high illegal probability
            if result['total_illegal_prob'] > 0.05:  # 5% threshold
                failure_cases.append({
                    'game_idx': game_idx,
                    'move_idx': move_idx,
                    'illegal_prob': result['total_illegal_prob'],
                    'result': result,
                })

            positions_checked += 1
            board.update([move])

        if positions_checked >= 5000:
            break

    # Sort by worst failures
    failure_cases.sort(key=lambda x: x['illegal_prob'], reverse=True)

    print(f"✓ Found {len(failure_cases)} positions with >5% illegal probability")
    print(f"  (out of {positions_checked} positions checked)")

    return failure_cases[:num_failures]

def analyze_failure_patterns(failure_cases):
    """Look for common patterns in failure cases"""
    print(f"\n{'='*70}")
    print(f"ANALYZING FAILURE PATTERNS")
    print(f"{'='*70}")

    # Categorize failures
    stages = defaultdict(int)
    legal_move_counts = defaultdict(int)

    for case in failure_cases:
        move_idx = case['move_idx']
        num_legal = len(case['result']['valid_moves'])

        # Stage
        if move_idx < 15:
            stages['opening'] += 1
        elif move_idx < 40:
            stages['midgame'] += 1
        else:
            stages['endgame'] += 1

        # Legal move count
        if num_legal <= 3:
            legal_move_counts['few (≤3)'] += 1
        elif num_legal <= 8:
            legal_move_counts['medium (4-8)'] += 1
        else:
            legal_move_counts['many (≥9)'] += 1

    print(f"\nFailure Distribution by Game Stage:")
    for stage, count in stages.items():
        pct = count / len(failure_cases) * 100
        print(f"  {stage:10s}: {count:3d} ({pct:5.1f}%)")

    print(f"\nFailure Distribution by Legal Move Count:")
    for cat, count in legal_move_counts.items():
        pct = count / len(failure_cases) * 100
        print(f"  {cat:15s}: {count:3d} ({pct:5.1f}%)")

def display_failure_case(case, case_num):
    """Display a single failure case in detail"""
    result = case['result']

    print(f"\n{'='*70}")
    print(f"FAILURE CASE #{case_num}")
    print(f"{'='*70}")
    print(f"Game #{case['game_idx']}, Move #{case['move_idx']}")
    print(f"Total Illegal Probability: {case['illegal_prob']:.2%}")

    # Reconstruct board
    board = OthelloBoardState()
    for move in result['game_moves']:
        board.update([move])

    # Show game sequence
    move_strs = [permit_reverse(m) for m in result['game_moves']]
    print(f"\nGame Sequence: {' → '.join(move_strs)}")

    # Show board
    print(f"\nBoard State:")
    visualize_board(board)

    # Show legal moves
    print(f"\nLegal Moves ({len(result['valid_moves'])}):")
    legal_str = [permit_reverse(m).upper() for m in result['valid_moves']]
    print(f"  {', '.join(legal_str)}")

    # Show top legal predictions
    print(f"\nTop 5 Legal Move Predictions:")
    for i, (pos, prob) in enumerate(result['legal_probs'][:5], 1):
        print(f"  {i}. {permit_reverse(pos).upper():4s} : {prob:7.2%}")

    # Show top illegal predictions (the failures!)
    print(f"\nTop 5 ILLEGAL Move Predictions (THE PROBLEM!):")
    for i, (pos, prob) in enumerate(result['illegal_probs'][:5], 1):
        # Check why it's illegal
        test_board = OthelloBoardState()
        for move in result['game_moves']:
            test_board.update([move])

        # Try to figure out why illegal
        row, col = pos // 8, pos % 8
        if test_board.state[row, col] != 0:
            reason = "Square already occupied"
        else:
            reason = "No pieces would be flipped"

        print(f"  {i}. {permit_reverse(pos).upper():4s} : {prob:7.2%} - {reason}")

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("FAILURE CASE ANALYSIS")
    print("="*70)

    # Load games
    print("\n[1/3] Loading championship games...")
    games = load_championship_games("./data", max_games=500)
    print(f"✓ Loaded {len(games)} games")

    # Find failure cases for Championship model
    print("\n[2/3] Finding failure cases for Championship Model...")
    champ_failures = find_failure_cases(
        "./ckpts/gpt_championship.ckpt",
        games,
        "Championship Model",
        num_failures=10
    )

    # Analyze patterns
    analyze_failure_patterns(champ_failures)

    # Display top 5 worst cases
    print("\n[3/3] Displaying Worst Failure Cases...")
    for i, case in enumerate(champ_failures[:5], 1):
        display_failure_case(case, i)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
