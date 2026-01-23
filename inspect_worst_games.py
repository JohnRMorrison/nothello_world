"""
Quick inspection of worst games from Phase 1 results
Shows move sequences, failure positions, and board states
"""
import torch
import torch.nn.functional as F
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState, permit, permit_reverse
import numpy as np
import glob
import re

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

def load_championship_games(data_dir, max_games=1000):
    """Load games from championship PGN files"""
    pgn_files = glob.glob(f"{data_dir}/othello_championship/*.pgn")
    all_games = []

    for pgn_file in pgn_files:
        games = parse_pgn_file(pgn_file)
        all_games.extend(games)
        if len(all_games) >= max_games:
            break

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

def classify_position(pos):
    """Classify a board position as corner, edge, or center"""
    row, col = pos // 8, pos % 8
    if (row in [0, 7]) and (col in [0, 7]):
        return "corner"
    elif row in [0, 7] or col in [0, 7]:
        return "edge"
    else:
        return "center"

def analyze_game_detailed(model, game, game_idx, device='cpu'):
    """Analyze a game and find all failure positions"""
    board = OthelloBoardState()
    failures = []

    for move_idx, move in enumerate(game):
        if move_idx < 4:
            board.update([move])
            continue

        # Analyze position
        game_so_far = game[:move_idx]
        valid_moves = board.get_valid_moves()

        # Get model predictions
        token_sequence = [stoi[m] for m in game_so_far]
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

        # Calculate illegal probability
        illegal_prob = sum(prob for pos, prob in board_probs.items() if pos not in valid_moves)

        # If it's a failure, record details
        if illegal_prob > 0.05:
            # Get top illegal predictions
            illegal_probs = [(pos, prob) for pos, prob in board_probs.items() if pos not in valid_moves]
            illegal_probs.sort(key=lambda x: x[1], reverse=True)

            # Get top legal predictions
            legal_probs = [(pos, board_probs.get(pos, 0.0)) for pos in valid_moves]
            legal_probs.sort(key=lambda x: x[1], reverse=True)

            failures.append({
                'move_idx': move_idx,
                'illegal_prob': illegal_prob,
                'num_legal_moves': len(valid_moves),
                'board_state': board.state.copy(),
                'game_so_far': game_so_far.copy(),
                'top_illegal': illegal_probs[:3],
                'top_legal': legal_probs[:3],
            })

        board.update([move])

    # Game-level characteristics
    corner_moves = sum(1 for move in game if classify_position(move) == "corner")
    edge_moves = sum(1 for move in game if classify_position(move) == "edge")
    center_moves = len(game) - corner_moves - edge_moves

    return {
        'game_idx': game_idx,
        'game': game,
        'game_length': len(game),
        'failures': failures,
        'corner_rate': corner_moves / len(game),
        'edge_rate': edge_moves / len(game),
        'center_rate': center_moves / len(game),
        'opening': game[:4],
    }

def display_game_analysis(analysis):
    """Display detailed analysis of a game"""
    print(f"\n{'='*70}")
    print(f"GAME #{analysis['game_idx']} ANALYSIS")
    print(f"{'='*70}")

    # Game characteristics
    print(f"\nGame Characteristics:")
    print(f"  ├─ Length: {analysis['game_length']} moves")
    print(f"  ├─ Failures: {len(analysis['failures'])} positions")
    print(f"  ├─ Opening: {' → '.join([permit_reverse(m) for m in analysis['opening']])}")
    print(f"  └─ Move distribution:")
    print(f"      ├─ Corner: {analysis['corner_rate']*100:.1f}%")
    print(f"      ├─ Edge:   {analysis['edge_rate']*100:.1f}%")
    print(f"      └─ Center: {analysis['center_rate']*100:.1f}%")

    # Show each failure
    for i, failure in enumerate(analysis['failures'], 1):
        print(f"\n{'-'*70}")
        print(f"FAILURE #{i} at move {failure['move_idx']} (illegal prob: {failure['illegal_prob']*100:.1f}%)")
        print(f"{'-'*70}")

        # Game sequence up to this point
        move_strs = [permit_reverse(m) for m in failure['game_so_far']]
        print(f"\nGame sequence: {' → '.join(move_strs)}")

        # Board state
        board = OthelloBoardState()
        for move in failure['game_so_far']:
            board.update([move])
        visualize_board(board)

        # Legal moves
        print(f"\nLegal moves ({failure['num_legal_moves']}):")
        legal_str = [f"{permit_reverse(pos).upper()} ({prob*100:.1f}%)"
                     for pos, prob in failure['top_legal']]
        print(f"  {', '.join(legal_str)}")

        # Top illegal predictions
        print(f"\nTop ILLEGAL predictions (the problem!):")
        for pos, prob in failure['top_illegal']:
            pos_type = classify_position(pos)
            print(f"  • {permit_reverse(pos).upper():4s} : {prob*100:.1f}% [{pos_type}]")

    # Pattern summary for this game
    print(f"\n{'='*70}")
    print(f"PATTERN SUMMARY FOR GAME #{analysis['game_idx']}")
    print(f"{'='*70}")

    # When do failures occur?
    failure_stages = []
    for failure in analysis['failures']:
        if failure['move_idx'] < 15:
            failure_stages.append("opening")
        elif failure['move_idx'] < 40:
            failure_stages.append("midgame")
        else:
            failure_stages.append("endgame")

    print(f"\nFailure timing:")
    for stage in ["opening", "midgame", "endgame"]:
        count = failure_stages.count(stage)
        if count > 0:
            print(f"  • {stage}: {count} failures")

    # What positions are being predicted incorrectly?
    all_illegal_preds = []
    for failure in analysis['failures']:
        for pos, prob in failure['top_illegal']:
            all_illegal_preds.append((pos, classify_position(pos)))

    pos_types = [t for _, t in all_illegal_preds]
    print(f"\nIllegal predictions by position type:")
    for ptype in ["corner", "edge", "center"]:
        count = pos_types.count(ptype)
        if count > 0:
            print(f"  • {ptype}: {count} predictions")

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("INSPECTING TOP 5 WORST GAMES")
    print("="*70)

    # Load model
    print("\n[1/3] Loading model...")
    mconf = GPTConfig(vocab_size=61, block_size=59, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    device = 'cpu'
    model.load_state_dict(torch.load("./ckpts/gpt_championship.ckpt", map_location=device))
    model.eval()
    print("✓ Model loaded")

    # Load games
    print("\n[2/3] Loading games...")
    games = load_championship_games("./data", max_games=1000)
    print(f"✓ Loaded {len(games)} games")

    # Top 5 worst games from Phase 1 results
    worst_game_indices = [103, 494, 287, 682, 321]

    print(f"\n[3/3] Analyzing top 5 worst games...")
    all_analyses = []

    for game_idx in worst_game_indices:
        print(f"  Analyzing game #{game_idx}...")
        game = games[game_idx]
        analysis = analyze_game_detailed(model, game, game_idx, device)
        all_analyses.append(analysis)

    # Display each analysis
    for analysis in all_analyses:
        display_game_analysis(analysis)

    # Cross-game pattern summary
    print(f"\n\n{'='*70}")
    print(f"CROSS-GAME PATTERN SUMMARY")
    print(f"{'='*70}")

    all_failures = []
    for analysis in all_analyses:
        all_failures.extend(analysis['failures'])

    print(f"\nTotal failures across 5 games: {len(all_failures)}")

    # Average characteristics
    avg_corner_rate = np.mean([a['corner_rate'] for a in all_analyses])
    avg_edge_rate = np.mean([a['edge_rate'] for a in all_analyses])
    avg_center_rate = np.mean([a['center_rate'] for a in all_analyses])

    print(f"\nAverage move distribution in worst games:")
    print(f"  ├─ Corner: {avg_corner_rate*100:.1f}%")
    print(f"  ├─ Edge:   {avg_edge_rate*100:.1f}%")
    print(f"  └─ Center: {avg_center_rate*100:.1f}%")

    # Failure characteristics
    failure_legal_counts = [f['num_legal_moves'] for f in all_failures]
    print(f"\nLegal moves available at failure positions:")
    print(f"  ├─ Average: {np.mean(failure_legal_counts):.1f}")
    print(f"  ├─ Min: {np.min(failure_legal_counts)}")
    print(f"  └─ Max: {np.max(failure_legal_counts)}")

    # What types of positions are predicted incorrectly?
    all_illegal_types = []
    for failure in all_failures:
        for pos, prob in failure['top_illegal']:
            all_illegal_types.append(classify_position(pos))

    print(f"\nIllegal prediction types across all failures:")
    for ptype in ["corner", "edge", "center"]:
        count = all_illegal_types.count(ptype)
        pct = count / len(all_illegal_types) * 100 if all_illegal_types else 0
        print(f"  • {ptype}: {count} ({pct:.1f}%)")

    print(f"\n{'='*70}")
    print("INSPECTION COMPLETE")
    print(f"{'='*70}")
