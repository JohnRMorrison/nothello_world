"""
Phase 1: Fast scan of 5,000 championship games to identify problematic games

This script:
1. Analyzes 5k games (~200k positions)
2. Tracks error metrics per game
3. Saves results to CSV for Phase 2 analysis
4. Optimized for speed with batch processing
"""
import torch
import torch.nn.functional as F
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState, permit
import numpy as np
import glob
import re
import csv
from collections import defaultdict
import time

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

    print(f"Found {len(pgn_files)} PGN files")
    print("Loading games...")

    for pgn_file in pgn_files:
        games = parse_pgn_file(pgn_file)
        all_games.extend(games)
        if len(all_games) >= max_games:
            break
        if len(all_games) % 1000 == 0:
            print(f"  Loaded {len(all_games)} games...")

    all_games = all_games[:max_games]
    return all_games

def analyze_game(model, game, device='cpu'):
    """
    Analyze a single game and return error metrics

    Returns dict with:
    - game_length: number of moves
    - positions_analyzed: number of positions checked
    - avg_illegal_prob: average illegal probability
    - max_illegal_prob: maximum illegal probability
    - failure_count: positions with >5% illegal prob
    - failure_rate: percentage of positions that are failures
    - illegal_probs: list of all illegal probabilities
    """
    board = OthelloBoardState()
    illegal_probs = []

    for move_idx, move in enumerate(game):
        if move_idx < 4:
            board.update([move])
            continue

        # Analyze position before making move
        game_so_far = game[:move_idx]
        valid_moves = board.get_valid_moves()

        # Convert to tokens
        token_sequence = [stoi[m] for m in game_so_far]
        input_tensor = torch.tensor([token_sequence]).long().to(device)

        with torch.no_grad():
            logits, _ = model(input_tensor)

        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)

        # Map to board positions and calculate illegal probability
        illegal_prob = 0.0
        for token_idx in range(1, 61):
            board_pos = itos[token_idx]
            if board_pos != -100 and board_pos not in valid_moves:
                illegal_prob += probs[token_idx].item()

        illegal_probs.append(illegal_prob)
        board.update([move])

    # Calculate metrics
    if len(illegal_probs) == 0:
        return None

    avg_illegal = np.mean(illegal_probs)
    max_illegal = np.max(illegal_probs)
    failure_count = sum(1 for p in illegal_probs if p > 0.05)
    failure_rate = failure_count / len(illegal_probs)

    return {
        'game_length': len(game),
        'positions_analyzed': len(illegal_probs),
        'avg_illegal_prob': avg_illegal,
        'max_illegal_prob': max_illegal,
        'failure_count': failure_count,
        'failure_rate': failure_rate,
        'illegal_probs': illegal_probs,
    }

def scan_all_games(model, games, device='cpu'):
    """Scan all games and collect error metrics"""
    print(f"\nAnalyzing {len(games)} games...")

    all_results = []
    start_time = time.time()

    for game_idx, game in enumerate(games):
        result = analyze_game(model, game, device)

        if result is not None:
            result['game_idx'] = game_idx
            all_results.append(result)

        # Progress updates
        if (game_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (game_idx + 1) / elapsed
            remaining = (len(games) - game_idx - 1) / rate
            print(f"  Progress: {game_idx + 1}/{len(games)} games "
                  f"({elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m remaining)")

    total_time = time.time() - start_time
    print(f"\n✓ Analyzed {len(all_results)} games in {total_time/60:.1f} minutes")

    return all_results

def save_results(results, output_file='game_error_analysis.csv'):
    """Save results to CSV file"""
    print(f"\nSaving results to {output_file}...")

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['game_idx', 'game_length', 'positions_analyzed',
                      'avg_illegal_prob', 'max_illegal_prob',
                      'failure_count', 'failure_rate']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            # Don't save the full illegal_probs list to CSV
            row = {k: v for k, v in result.items() if k != 'illegal_probs'}
            writer.writerow(row)

    print(f"✓ Saved {len(results)} game summaries to {output_file}")

def print_summary_stats(results):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    avg_illegals = [r['avg_illegal_prob'] for r in results]
    max_illegals = [r['max_illegal_prob'] for r in results]
    failure_rates = [r['failure_rate'] for r in results]

    print(f"\nOverall Statistics (across {len(results)} games):")
    print(f"  Average Illegal Probability:")
    print(f"    ├─ Mean:   {np.mean(avg_illegals):.4f} ({np.mean(avg_illegals)*100:.2f}%)")
    print(f"    ├─ Median: {np.median(avg_illegals):.4f} ({np.median(avg_illegals)*100:.2f}%)")
    print(f"    └─ Std:    {np.std(avg_illegals):.4f}")

    print(f"\n  Maximum Illegal Probability:")
    print(f"    ├─ Mean:   {np.mean(max_illegals):.4f} ({np.mean(max_illegals)*100:.2f}%)")
    print(f"    ├─ Median: {np.median(max_illegals):.4f} ({np.median(max_illegals)*100:.2f}%)")
    print(f"    └─ Max:    {np.max(max_illegals):.4f} ({np.max(max_illegals)*100:.2f}%)")

    print(f"\n  Failure Rate (% positions with >5% illegal prob):")
    print(f"    ├─ Mean:   {np.mean(failure_rates):.4f} ({np.mean(failure_rates)*100:.2f}%)")
    print(f"    ├─ Median: {np.median(failure_rates):.4f} ({np.median(failure_rates)*100:.2f}%)")
    print(f"    └─ Max:    {np.max(failure_rates):.4f} ({np.max(failure_rates)*100:.2f}%)")

    # Distribution of games by error level
    print(f"\n  Distribution by Average Illegal Probability:")
    clean = sum(1 for r in results if r['avg_illegal_prob'] < 0.01)
    low = sum(1 for r in results if 0.01 <= r['avg_illegal_prob'] < 0.03)
    medium = sum(1 for r in results if 0.03 <= r['avg_illegal_prob'] < 0.05)
    high = sum(1 for r in results if r['avg_illegal_prob'] >= 0.05)

    print(f"    ├─ Clean (<1%):      {clean:5d} ({clean/len(results)*100:5.1f}%)")
    print(f"    ├─ Low (1-3%):       {low:5d} ({low/len(results)*100:5.1f}%)")
    print(f"    ├─ Medium (3-5%):    {medium:5d} ({medium/len(results)*100:5.1f}%)")
    print(f"    └─ High (≥5%):       {high:5d} ({high/len(results)*100:5.1f}%)")

    # Top 10 worst games
    print(f"\n  Top 10 Worst Games (by avg illegal prob):")
    sorted_results = sorted(results, key=lambda x: x['avg_illegal_prob'], reverse=True)
    for i, r in enumerate(sorted_results[:10], 1):
        print(f"    {i:2d}. Game #{r['game_idx']:5d}: "
              f"avg={r['avg_illegal_prob']*100:5.2f}%, "
              f"max={r['max_illegal_prob']*100:5.2f}%, "
              f"failures={r['failure_count']}/{r['positions_analyzed']}")

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("PHASE 1: GAME ERROR SCAN")
    print("="*70)

    # Load model
    print("\n[1/4] Loading Championship Model...")
    mconf = GPTConfig(vocab_size=61, block_size=59, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    device = 'cpu'
    model.load_state_dict(torch.load("./ckpts/gpt_championship.ckpt", map_location=device))
    model.eval()
    print("✓ Model loaded")

    # Load games
    print("\n[2/4] Loading Championship Games...")
    games = load_championship_games("./data", max_games=1000)
    print(f"✓ Loaded {len(games)} games")
    print(f"  Average game length: {np.mean([len(g) for g in games]):.1f} moves")

    # Analyze all games
    print("\n[3/4] Scanning All Games...")
    results = scan_all_games(model, games, device)

    # Save results
    print("\n[4/4] Saving Results...")
    save_results(results, 'game_error_analysis.csv')

    # Print summary
    print_summary_stats(results)

    print("\n" + "="*70)
    print("PHASE 1 COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Open game_error_analysis.csv to see all results")
    print("  2. Use Phase 2 Jupyter notebook for visualization and deep dive")
    print("  3. Identify patterns in high-error games")
