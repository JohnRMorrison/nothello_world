"""
Comprehensive analysis of illegal move probabilities using championship data
Includes:
- Analysis on real championship games
- Stratification by game stage and legal move count
- Comparison between championship and synthetic models
"""
import torch
import torch.nn.functional as F
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState, permit, permit_reverse
import numpy as np
import glob
from collections import defaultdict
import re

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

    # Split by games (each game starts with [Event ...])
    game_blocks = re.split(r'\[Event', content)

    for block in game_blocks[1:]:  # Skip first empty block
        # Extract moves (format: "1. F5 D6 2. C3 F3 ...")
        moves_match = re.search(r'\n((?:\d+\.\s+\w+\s+\w+\s*)+)', block)
        if moves_match:
            moves_text = moves_match.group(1)
            # Extract just the move coordinates (F5, D6, C3, etc.)
            moves = re.findall(r'[A-H][1-8]', moves_text)
            # Convert to board positions (0-63)
            move_positions = []
            for move_str in moves:
                pos = permit(move_str)
                if pos != -1:
                    move_positions.append(pos)
            if len(move_positions) >= 10:  # Only keep games with at least 10 moves
                games.append(move_positions)

    return games

def load_championship_games(data_dir, max_games=None):
    """Load games from championship PGN files"""
    pgn_files = glob.glob(f"{data_dir}/othello_championship/*.pgn")
    all_games = []

    print(f"Found {len(pgn_files)} PGN files")

    for pgn_file in pgn_files:
        games = parse_pgn_file(pgn_file)
        all_games.extend(games)
        if max_games and len(all_games) >= max_games:
            break

    if max_games:
        all_games = all_games[:max_games]

    return all_games

def analyze_position(model, board, game_moves, device='cpu'):
    """Analyze a single board position"""
    valid_moves = board.get_valid_moves()

    # Convert to token indices
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

    # Calculate statistics
    legal_prob = sum(board_probs.get(pos, 0.0) for pos in valid_moves)
    illegal_prob = sum(prob for pos, prob in board_probs.items() if pos not in valid_moves)

    # Top prediction
    sorted_probs = sorted(board_probs.items(), key=lambda x: x[1], reverse=True)
    top_pred = sorted_probs[0][0]
    top_is_legal = top_pred in valid_moves

    return {
        'num_legal_moves': len(valid_moves),
        'legal_prob': legal_prob,
        'illegal_prob': illegal_prob,
        'top_is_legal': top_is_legal,
        'top_pred_prob': sorted_probs[0][1],
    }

def analyze_model(model_path, games, model_name, num_positions=5000, device='cpu'):
    """Analyze a model on championship games"""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {model_name}")
    print(f"{'='*70}")

    # Load model
    print(f"Loading {model_name}...")
    mconf = GPTConfig(vocab_size=61, block_size=59, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Model loaded")

    # Collect results by category
    results_by_stage = defaultdict(list)  # opening, midgame, endgame
    results_by_legal_count = defaultdict(list)  # few (<=3), medium (4-8), many (9+)
    all_results = []

    positions_analyzed = 0

    for game_idx, game in enumerate(games):
        if positions_analyzed >= num_positions:
            break

        board = OthelloBoardState()

        for move_idx, move in enumerate(game):
            if positions_analyzed >= num_positions:
                break

            # Skip first few moves (need history)
            if move_idx < 4:
                board.update([move])
                continue

            # Analyze position BEFORE making this move
            game_so_far = game[:move_idx]
            result = analyze_position(model, board, game_so_far, device)

            # Categorize by game stage
            if move_idx < 15:
                stage = 'opening'
            elif move_idx < 40:
                stage = 'midgame'
            else:
                stage = 'endgame'
            results_by_stage[stage].append(result)

            # Categorize by number of legal moves
            num_legal = result['num_legal_moves']
            if num_legal <= 3:
                legal_cat = 'few (≤3)'
            elif num_legal <= 8:
                legal_cat = 'medium (4-8)'
            else:
                legal_cat = 'many (≥9)'
            results_by_legal_count[legal_cat].append(result)

            all_results.append(result)
            positions_analyzed += 1

            # Apply the move for next iteration
            board.update([move])

        if (game_idx + 1) % 100 == 0:
            print(f"  Processed {game_idx + 1} games, {positions_analyzed} positions...")

    print(f"✓ Analyzed {positions_analyzed} positions from {game_idx + 1} games")

    # Compute overall statistics
    avg_illegal_prob = np.mean([r['illegal_prob'] for r in all_results])
    std_illegal_prob = np.std([r['illegal_prob'] for r in all_results])
    max_illegal_prob = np.max([r['illegal_prob'] for r in all_results])
    top_accuracy = np.mean([r['top_is_legal'] for r in all_results])

    print(f"\n{'='*70}")
    print(f"OVERALL RESULTS ({model_name})")
    print(f"{'='*70}")
    print(f"Probability on Illegal Moves:")
    print(f"  ├─ Mean:   {avg_illegal_prob:.4f} ({avg_illegal_prob*100:.2f}%)")
    print(f"  ├─ Std:    {std_illegal_prob:.4f}")
    print(f"  └─ Max:    {max_illegal_prob:.4f} ({max_illegal_prob*100:.2f}%)")
    print(f"\nTop Prediction Accuracy: {top_accuracy*100:.1f}% legal")

    # Results by game stage
    print(f"\n{'='*70}")
    print(f"RESULTS BY GAME STAGE")
    print(f"{'='*70}")
    for stage in ['opening', 'midgame', 'endgame']:
        if stage in results_by_stage:
            results = results_by_stage[stage]
            avg_illegal = np.mean([r['illegal_prob'] for r in results])
            top_acc = np.mean([r['top_is_legal'] for r in results])
            print(f"\n{stage.upper()} (n={len(results)}):")
            print(f"  ├─ Avg illegal prob: {avg_illegal:.4f} ({avg_illegal*100:.2f}%)")
            print(f"  └─ Top pred accuracy: {top_acc*100:.1f}%")

    # Results by legal move count
    print(f"\n{'='*70}")
    print(f"RESULTS BY NUMBER OF LEGAL MOVES")
    print(f"{'='*70}")
    for cat in ['few (≤3)', 'medium (4-8)', 'many (≥9)']:
        if cat in results_by_legal_count:
            results = results_by_legal_count[cat]
            avg_illegal = np.mean([r['illegal_prob'] for r in results])
            top_acc = np.mean([r['top_is_legal'] for r in results])
            print(f"\n{cat} legal moves (n={len(results)}):")
            print(f"  ├─ Avg illegal prob: {avg_illegal:.4f} ({avg_illegal*100:.2f}%)")
            print(f"  └─ Top pred accuracy: {top_acc*100:.1f}%")

    return {
        'model_name': model_name,
        'all_results': all_results,
        'avg_illegal_prob': avg_illegal_prob,
        'std_illegal_prob': std_illegal_prob,
        'top_accuracy': top_accuracy,
        'by_stage': results_by_stage,
        'by_legal_count': results_by_legal_count,
    }

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("COMPREHENSIVE ILLEGAL MOVE PROBABILITY ANALYSIS")
    print("="*70)

    # Load championship games
    print("\n[1/4] Loading championship games...")
    games = load_championship_games("./data", max_games=1000)
    print(f"✓ Loaded {len(games)} games")
    print(f"  Average game length: {np.mean([len(g) for g in games]):.1f} moves")

    # Analyze both models
    print("\n[2/4] Analyzing Championship Model...")
    champ_results = analyze_model(
        "./ckpts/gpt_championship.ckpt",
        games,
        "Championship Model",
        num_positions=5000
    )

    print("\n[3/4] Analyzing Synthetic Model...")
    synth_results = analyze_model(
        "./ckpts/gpt_synthetic.ckpt",
        games,
        "Synthetic Model",
        num_positions=5000
    )

    # Compare models
    print("\n[4/4] Model Comparison...")
    print("="*70)
    print("CHAMPIONSHIP vs SYNTHETIC MODEL COMPARISON")
    print("="*70)
    print(f"\nAverage Illegal Move Probability:")
    print(f"  ├─ Championship: {champ_results['avg_illegal_prob']:.4f} ({champ_results['avg_illegal_prob']*100:.2f}%)")
    print(f"  └─ Synthetic:    {synth_results['avg_illegal_prob']:.4f} ({synth_results['avg_illegal_prob']*100:.2f}%)")

    diff = champ_results['avg_illegal_prob'] - synth_results['avg_illegal_prob']
    print(f"\nDifference: {abs(diff):.4f} ({abs(diff)*100:.2f}%)")
    if abs(diff) < 0.01:
        print("→ Models perform similarly")
    elif champ_results['avg_illegal_prob'] < synth_results['avg_illegal_prob']:
        print("→ Championship model performs BETTER (less illegal probability)")
    else:
        print("→ Synthetic model performs BETTER (less illegal probability)")

    print(f"\nTop Prediction Accuracy:")
    print(f"  ├─ Championship: {champ_results['top_accuracy']*100:.1f}%")
    print(f"  └─ Synthetic:    {synth_results['top_accuracy']*100:.1f}%")

    print("\n" + "="*70)
    print("FINAL RESEARCH FINDINGS")
    print("="*70)

    threshold = 0.05  # 5% threshold
    for results in [champ_results, synth_results]:
        print(f"\n{results['model_name']}:")
        if results['avg_illegal_prob'] > threshold:
            print(f"  ⚠️  Assigns NON-TRIVIAL probability to illegal moves")
            print(f"     ({results['avg_illegal_prob']*100:.2f}% avg, threshold: {threshold*100}%)")
        else:
            print(f"  ✓ Assigns minimal probability to illegal moves")
            print(f"     ({results['avg_illegal_prob']*100:.2f}% avg)")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
