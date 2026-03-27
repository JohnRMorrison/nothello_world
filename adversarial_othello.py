"""
Adversarial Games for Othello-GPT

Four approaches to finding game sequences that cause the model to assign
high probability to illegal moves:

1. Search-based, legal games only (beam search)
2. Search-based, illegal inputs
3. Gradient-based, legal games
4. Gradient-based, unrestricted inputs (embedding space optimization)

Usage:
    python adversarial_othello.py --approach [1|2|3|4|all] [options]
"""

import argparse
import random
import json
import time
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns

from data import get_othello
from data.othello import OthelloBoardState, permit_reverse, start_hands
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig
from mingpt.utils import set_seed


# =============================================================================
# Shared Utilities
# =============================================================================

# 8 directions for scanning Othello flips (same as data/othello.py)
EIGHTS = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]


def relaxed_play(board_state, move, color):
    """
    Play a move on the board with the given color, applying any flips
    required by Othello rules. Unlike the standard umpire, this allows
    moves that don't flip any pieces (normally illegal in Othello).

    Args:
        board_state: 8x8 numpy array (1=black, -1=white, 0=empty)
        move: board position (0-63)
        color: 1 for black, -1 for white

    Returns:
        True if the piece was placed, False if the square was occupied.
    """
    r, c = move // 8, move % 8
    if board_state[r, c] != 0:
        return False

    # Scan all 8 directions for pieces to flip
    tbf = []
    for direction in EIGHTS:
        buffer = []
        cur_r, cur_c = r, c
        while True:
            cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
            if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                break
            if board_state[cur_r, cur_c] == 0:
                break
            elif board_state[cur_r, cur_c] == color:
                tbf.extend(buffer)
                break
            else:
                buffer.append([cur_r, cur_c])

    # Apply flips (may be empty — that's OK, we still place the piece)
    for ff in tbf:
        board_state[ff[0], ff[1]] *= -1

    # Place the piece
    board_state[r, c] = color
    return True


def load_model_and_dataset(ckpt_path="./ckpts/gpt_synthetic.ckpt", synthetic=True, num_games=1000):
    """Load the trained Othello-GPT model and dataset.
    Generates synthetic games on the fly if pre-generated data is not available.
    """
    try:
        othello = get_othello(ood_num=-1, data_root=None if synthetic else "data/othello_championship", wthor=True)
    except FileNotFoundError:
        print(f"Pre-generated data not found. Generating {num_games} synthetic games...")
        othello = get_othello(ood_num=num_games)
    train_dataset = CharDataset(othello)
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, train_dataset, othello, device


def forward_from_embeddings(model, x):
    """Forward pass starting from embeddings, bypassing token/position lookup.
    x: [B, T, n_embd] tensor of embeddings
    Returns logits: [B, T, vocab_size]
    """
    x = model.blocks(x)
    x = model.ln_f(x)
    logits = model.head(x)
    return logits


def get_model_probs(model, token_sequence, dataset, device):
    """Get the model's probability distribution for the next move after token_sequence.
    Returns probs: [vocab_size] tensor of probabilities.
    """
    x = torch.tensor([dataset.stoi[s] for s in token_sequence], dtype=torch.long)[None, ...].to(device)
    with torch.no_grad():
        logits, _ = model(x)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    return probs


def compute_illegal_mass(probs, valid_moves, dataset):
    """Compute the total probability assigned to illegal moves.
    probs: [vocab_size] tensor of probabilities
    valid_moves: list of legal board positions (0-63)
    dataset: CharDataset with itos mapping
    Returns: (illegal_mass, list of (board_pos, prob) for illegal moves with prob > 0.001)
    """
    valid_set = set(valid_moves)
    illegal_mass = 0.0
    illegal_moves = []
    for token_idx in range(len(probs)):
        board_pos = dataset.itos[token_idx]
        if board_pos == -100:
            illegal_mass += probs[token_idx].item()
            continue
        if board_pos not in valid_set:
            p = probs[token_idx].item()
            illegal_mass += p
            if p > 0.001:
                illegal_moves.append((board_pos, p))
    illegal_moves.sort(key=lambda x: x[1], reverse=True)
    return illegal_mass, illegal_moves


def score_position(model, game_sequence, move_idx, dataset, device):
    """Score a single position in a game for illegal probability mass.
    game_sequence: list of board positions (0-63) for the full game
    move_idx: evaluate after this many moves have been played (1-indexed)
    Returns: (illegal_mass, illegal_moves, valid_moves)
    """
    context = game_sequence[:move_idx]
    board = OthelloBoardState()
    board.update(context)
    valid_moves = board.get_valid_moves()
    if not valid_moves:
        return 0.0, [], []
    probs = get_model_probs(model, context, dataset, device)
    illegal_mass, illegal_moves = compute_illegal_mass(probs, valid_moves, dataset)
    return illegal_mass, illegal_moves, valid_moves


def score_full_game(model, game_sequence, dataset, device):
    """Score every position in a game. Returns list of (move_idx, illegal_mass)."""
    results = []
    board = OthelloBoardState()
    for move_idx in range(1, len(game_sequence)):
        context = game_sequence[:move_idx]
        valid_moves = board.get_valid_moves() if move_idx > 0 else []
        # Replay up to this point
        board_copy = OthelloBoardState()
        board_copy.update(context)
        valid_moves = board_copy.get_valid_moves()
        if not valid_moves:
            results.append((move_idx, 0.0))
            continue
        probs = get_model_probs(model, context, dataset, device)
        illegal_mass, _ = compute_illegal_mass(probs, valid_moves, dataset)
        results.append((move_idx, illegal_mass))
    return results


def visualize_position(model, game_sequence, move_idx, dataset, device,
                       save_path=None, show=True):
    """
    Visualize the board state and the model's probability distribution at a
    specific position in a game.

    Creates a figure with 3 panels:
    1. Board state with pieces (X=black, O=white)
    2. Model's probability heatmap (legal moves underlined)
    3. Illegal probability heatmap (red = high illegal prob)

    game_sequence: list of board positions (0-63)
    move_idx: show position after this many moves (1-indexed)
    """
    context = game_sequence[:move_idx]
    board = OthelloBoardState()
    board.update(context)
    valid_moves = set(board.get_valid_moves())

    probs = get_model_probs(model, context, dataset, device)

    # Build 8x8 probability grid and illegal grid
    prob_grid = np.zeros(64)
    illegal_grid = np.zeros(64)
    for token_idx in range(len(probs)):
        board_pos = dataset.itos[token_idx]
        if board_pos == -100:
            continue
        p = probs[token_idx].item()
        prob_grid[board_pos] = p
        if board_pos not in valid_moves:
            illegal_grid[board_pos] = p

    prob_grid = prob_grid.reshape(8, 8)
    illegal_grid = illegal_grid.reshape(8, 8)
    board_state = board.state.copy()

    total_illegal = illegal_grid.sum()
    total_legal = prob_grid.sum() - total_illegal

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: Board State ---
    ax = axes[0]
    # Color: white pieces = -1, black pieces = 1, empty = 0
    display_board = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            pos = r * 8 + c
            if board_state[r, c] == 1:
                display_board[r, c] = 1  # black
            elif board_state[r, c] == -1:
                display_board[r, c] = -1  # white
            elif pos in valid_moves:
                display_board[r, c] = 0.3  # legal move available

    sns.heatmap(display_board, ax=ax, cbar=False, square=True, linewidths=0.5,
                xticklabels=list(range(1, 9)), yticklabels=list("ABCDEFGH"),
                cmap=sns.color_palette("RdYlGn", as_cmap=True), vmin=-1.5, vmax=1.5,
                center=0)

    # Add piece markers
    for r in range(8):
        for c in range(8):
            if board_state[r, c] == 1:
                ax.add_collection(PatchCollection(
                    [mpatches.Circle((c + 0.5, r + 0.5), 0.3, facecolor='black')],
                    match_original=True))
            elif board_state[r, c] == -1:
                ax.add_collection(PatchCollection(
                    [mpatches.Circle((c + 0.5, r + 0.5), 0.3, facecolor='white',
                                     edgecolor='black', linewidth=0.5)],
                    match_original=True))
            elif r * 8 + c in valid_moves:
                ax.add_collection(PatchCollection(
                    [mpatches.Circle((c + 0.5, r + 0.5), 0.1, facecolor='green')],
                    match_original=True))

    next_color = "Black" if board.next_hand_color == 1 else "White"
    ax.set_title(f"Board State (move {move_idx})\n{next_color} to play, "
                 f"{len(valid_moves)} legal moves")

    # --- Panel 2: Model Probabilities ---
    ax = axes[1]
    annot = np.array([f"{v:.2f}" if v > 0.01 else "" for v in prob_grid.flatten()])
    # Underline legal moves
    for pos in valid_moves:
        r, c = pos // 8, pos % 8
        val = prob_grid[r, c]
        if val > 0.01:
            annot[pos] = f"*{val:.2f}"

    sns.heatmap(prob_grid, ax=ax, cbar=True, square=True, linewidths=0.5,
                xticklabels=list(range(1, 9)), yticklabels=list("ABCDEFGH"),
                cmap=sns.color_palette("Blues", as_cmap=True), vmin=0, vmax=max(0.2, prob_grid.max()),
                annot=annot.reshape(8, 8), fmt="")
    ax.set_title(f"Model Probabilities\n(* = legal move, total legal prob: {total_legal:.3f})")

    # --- Panel 3: Illegal Probability ---
    ax = axes[2]
    annot_ill = np.array([f"{v:.2f}" if v > 0.005 else "" for v in illegal_grid.flatten()])

    sns.heatmap(illegal_grid, ax=ax, cbar=True, square=True, linewidths=0.5,
                xticklabels=list(range(1, 9)), yticklabels=list("ABCDEFGH"),
                cmap=sns.color_palette("Reds", as_cmap=True), vmin=0, vmax=max(0.1, illegal_grid.max()),
                annot=annot_ill.reshape(8, 8), fmt="")
    ax.set_title(f"Illegal Move Probabilities\n(total illegal mass: {total_illegal:.3f})")

    plt.suptitle(f"Game position after move {move_idx}: "
                 f"{', '.join(permit_reverse(m) for m in context[-5:])}...",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

    return fig


def visualize_game_results(results_file, model, dataset, device, output_dir="adversarial_figures",
                           num_games=3, show=True):
    """
    Visualize the worst positions from the results file.
    Handles results from error-prone search, approach 1 (beam/random search),
    and approach 3 (gradient-guided legal games).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(results_file, "r") as f:
        data = json.load(f)

    results = data.get("results", {})
    fig_count = 0

    # --- Approach 1: beam search and random search ---
    approach1 = results.get("1", {})
    for label, key in [("beam", "beam_search"), ("random", "random_search")]:
        games = approach1.get(key, [])
        if not games:
            continue
        # Sort by worst_position_illegal_mass descending
        games_sorted = sorted(games, key=lambda g: g.get("worst_position_illegal_mass", 0), reverse=True)
        print(f"\n--- Approach 1 ({label} search): top {min(num_games, len(games_sorted))} worst games ---")
        for i, game_data in enumerate(games_sorted[:num_games]):
            game_int = game_data["game_int"]
            worst_idx = game_data["worst_position_move_idx"]
            worst_mass = game_data.get("worst_position_illegal_mass", 0)
            print(f"  Game {i+1}: worst position move {worst_idx}, illegal mass {worst_mass:.4f}")

            save_path = os.path.join(output_dir, f"approach1_{label}_game{i+1}_move{worst_idx}.png")
            visualize_position(model, game_int, worst_idx, dataset, device,
                              save_path=save_path, show=show)
            fig_count += 1

    # --- Approach 3: gradient-guided legal games ---
    approach3 = results.get("3", {})
    opt_games = approach3.get("optimized_games", [])
    if opt_games:
        games_sorted = sorted(opt_games, key=lambda g: g.get("worst_position_illegal_mass",
                              g.get("best_position_illegal_mass", 0)), reverse=True)
        print(f"\n--- Approach 3 (gradient-guided): top {min(num_games, len(games_sorted))} worst games ---")
        for i, game_data in enumerate(games_sorted[:num_games]):
            game_int = game_data.get("game_int", [])
            if not game_int:
                continue
            worst_idx = game_data.get("worst_position_move_idx",
                                      game_data.get("best_position_move_idx", len(game_int)))
            worst_mass = game_data.get("worst_position_illegal_mass",
                                       game_data.get("best_position_illegal_mass", 0))
            print(f"  Game {i+1}: worst position move {worst_idx}, illegal mass {worst_mass:.4f}")

            save_path = os.path.join(output_dir, f"approach3_game{i+1}_move{worst_idx}.png")
            visualize_position(model, game_int, worst_idx, dataset, device,
                              save_path=save_path, show=show)
            fig_count += 1

    # --- Approach 2: corrupted games ---
    approach2 = results.get("2", {})
    corrupted = approach2.get("corrupted_games", [])
    if corrupted:
        # Filter to entries that have stored game sequences
        with_games = [g for g in corrupted if g.get("game_int")]
        if with_games:
            games_sorted = sorted(with_games, key=lambda g: g.get("illegal_mass", 0), reverse=True)
            print(f"\n--- Approach 2 (corrupted games): top {min(num_games, len(games_sorted))} ---")
            for i, game_data in enumerate(games_sorted[:num_games]):
                game_int = game_data["game_int"]
                prefix_len = game_data["legal_prefix_len"]
                mass = game_data.get("illegal_mass", 0)
                print(f"  Game {i+1}: corrupted at move {prefix_len}, illegal mass {mass:.4f}")

                # Replay legal prefix to get board state and valid moves
                legal_prefix = game_int[:prefix_len]
                board = OthelloBoardState()
                board.update(legal_prefix)
                valid_moves = board.get_valid_moves()

                # Feed full corrupted sequence to model, visualize against legal board
                seq_tokens = [dataset.stoi[s] for s in game_int]
                description = f"Corrupted at move {prefix_len}, {game_data['num_random_suffix']} random moves appended"
                save_path = os.path.join(output_dir, f"approach2_corrupted_game{i+1}.png")
                visualize_impossible_board(
                    model, dataset, device, board.state, valid_moves,
                    game_int, f"corrupted_{i+1}", description,
                    save_path=save_path, show=show,
                )
                fig_count += 1

    # --- Approach 4: decoded optimized embeddings ---
    approach4 = results.get("4", {})
    opt_embs = approach4.get("optimized_embeddings", [])
    if opt_embs:
        # Filter to entries that have stored game sequences
        with_games = [g for g in opt_embs if g.get("context_int")]
        if with_games:
            games_sorted = sorted(with_games, key=lambda g: g.get("optimized_illegal_mass", 0), reverse=True)
            print(f"\n--- Approach 4 (embedding optimization): top {min(num_games, len(games_sorted))} ---")
            for i, game_data in enumerate(games_sorted[:num_games]):
                context_int = game_data["context_int"]
                decoded_int = game_data["decoded_moves_int"]
                mass = game_data.get("optimized_illegal_mass", 0)
                baseline = game_data.get("baseline_illegal_mass", 0)
                print(f"  Trial {i+1}: illegal mass {mass:.4f} (baseline {baseline:.4f})")

                # Replay original legal context to get board state and valid moves
                board = OthelloBoardState()
                board.update(context_int)
                valid_moves = board.get_valid_moves()

                # Feed decoded token sequence to model, visualize against legal board
                description = f"Optimized embeddings decoded to tokens (baseline {baseline:.4f})"
                save_path = os.path.join(output_dir, f"approach4_trial{i+1}.png")
                visualize_impossible_board(
                    model, dataset, device, board.state, valid_moves,
                    decoded_int, f"emb_opt_{i+1}", description,
                    save_path=save_path, show=show,
                )
                fig_count += 1

    # --- Error-prone games ---
    error_prone = results.get("error_prone", {})
    games = error_prone.get("games", [])
    if games:
        print(f"\n--- Error-prone games: top {min(num_games, len(games))} ---")
        for i, game_data in enumerate(games[:num_games]):
            game_int = game_data["game_int"]
            worst_idx = game_data["worst_position_move_idx"]
            mistake_positions = game_data.get("mistake_positions", [])

            print(f"  Game {i+1}: {game_data['num_mistakes']} mistakes, "
                  f"max illegal mass: {game_data['max_illegal_mass']:.4f}")

            # Visualize worst position
            save_path = os.path.join(output_dir, f"error_prone_game{i+1}_worst_move{worst_idx}.png")
            visualize_position(model, game_int, worst_idx, dataset, device,
                              save_path=save_path, show=show)
            fig_count += 1

            # Visualize up to 3 other mistake positions
            for j, (midx, mass) in enumerate(mistake_positions[:3]):
                if midx == worst_idx:
                    continue
                save_path = os.path.join(output_dir, f"error_prone_game{i+1}_mistake_move{midx}.png")
                visualize_position(model, game_int, midx, dataset, device,
                                  save_path=save_path, show=show)
                fig_count += 1

    if fig_count == 0:
        print("No visualizable game data found in results file.")
    else:
        print(f"\nGenerated {fig_count} figures in {output_dir}/")


# =============================================================================
# Find Games with Many Mistakes
# =============================================================================

def find_error_prone_games(model, dataset, device, num_games=500,
                           threshold=0.1, verbose=True):
    """
    Search for legal games where the model makes mistakes at many positions.

    Metrics per game:
    - num_mistakes: number of positions where illegal mass > threshold
    - avg_illegal_mass: average illegal mass across all positions
    - cumulative_illegal_mass: sum of illegal mass across all positions
    - max_illegal_mass: worst single position

    Returns results sorted by num_mistakes (most mistakes first).
    """
    if verbose:
        print("=" * 60)
        print(f"FINDING GAMES WITH MANY MISTAKES (threshold={threshold})")
        print("=" * 60)

    game_results = []

    for _ in tqdm(range(num_games), desc="Scoring games", disable=not verbose):
        # Generate a random legal game
        game = []
        board = OthelloBoardState()
        valid = board.get_valid_moves()
        while valid:
            move = random.choice(valid)
            game.append(move)
            board.umpire(move)
            valid = board.get_valid_moves()

        # Score every position
        scores = score_full_game(model, game, dataset, device)
        if not scores:
            continue

        masses = [s[1] for s in scores]
        num_mistakes = sum(1 for m in masses if m > threshold)
        avg_mass = np.mean(masses)
        cum_mass = sum(masses)
        max_mass = max(masses)
        worst_idx = scores[np.argmax(masses)][0]

        # Positions above threshold
        mistake_positions = [(idx, mass) for idx, mass in scores if mass > threshold]

        game_results.append({
            "game": [permit_reverse(m) for m in game],
            "game_int": game,
            "num_mistakes": num_mistakes,
            "num_positions": len(scores),
            "mistake_rate": num_mistakes / len(scores),
            "avg_illegal_mass": avg_mass,
            "cumulative_illegal_mass": cum_mass,
            "max_illegal_mass": max_mass,
            "worst_position_move_idx": worst_idx,
            "mistake_positions": mistake_positions,
            "per_position_illegal_mass": scores,
        })

    # Sort by number of mistakes
    game_results.sort(key=lambda x: x["num_mistakes"], reverse=True)

    if verbose and game_results:
        print(f"\nTop 10 most error-prone games:")
        print(f"{'Rank':>4} {'Mistakes':>8} {'Rate':>6} {'Avg Mass':>9} {'Max Mass':>9}")
        print("-" * 42)
        for i, g in enumerate(game_results[:10]):
            print(f"{i+1:>4} {g['num_mistakes']:>8} {g['mistake_rate']:>6.1%} "
                  f"{g['avg_illegal_mass']:>9.4f} {g['max_illegal_mass']:>9.4f}")

        print(f"\nOverall stats across {len(game_results)} games:")
        all_mistakes = [g["num_mistakes"] for g in game_results]
        all_avg = [g["avg_illegal_mass"] for g in game_results]
        print(f"  Games with at least 1 mistake: {sum(1 for m in all_mistakes if m > 0)}/{len(game_results)}")
        print(f"  Avg mistakes per game: {np.mean(all_mistakes):.2f}")
        print(f"  Max mistakes in a game: {max(all_mistakes)}")
        print(f"  Avg illegal mass across all games: {np.mean(all_avg):.4f}")

    return {
        "threshold": threshold,
        "num_games_searched": num_games,
        "games": game_results[:50],  # Save top 50
        "summary": {
            "games_with_mistakes": sum(1 for g in game_results if g["num_mistakes"] > 0),
            "avg_mistakes_per_game": float(np.mean([g["num_mistakes"] for g in game_results])),
            "max_mistakes_in_game": max(g["num_mistakes"] for g in game_results) if game_results else 0,
            "avg_illegal_mass_all_games": float(np.mean([g["avg_illegal_mass"] for g in game_results])),
        }
    }


# =============================================================================
# Approach 1: Search-Based, Legal Games Only
# =============================================================================

def approach1_beam_search(model, dataset, device, beam_width=10, max_depth=59,
                          num_random_seeds=100, num_save=50, num_diverse_starts=50,
                          prefix_len=5, beam_only=False, verbose=True):
    """
    Find legal Othello game sequences that maximize illegal probability mass.

    Strategy: Run beam search from multiple different random openings to
    ensure diversity. Each run starts from a random prefix of prefix_len
    moves, then uses beam search to maximize illegal mass from there.
    The best game from each start is kept, producing truly different games.

    Also runs random games and keeps the worst positions found.
    """
    if verbose:
        print("=" * 60)
        print("APPROACH 1: Search-Based, Legal Games Only")
        print("=" * 60)

    results = {
        "approach": 1,
        "beam_search": [],
        "random_search": [],
    }

    # --- Diverse Beam Search ---
    if verbose:
        print(f"\nDiverse beam search: {num_diverse_starts} starts, "
              f"width {beam_width}, prefix length {prefix_len}...")

    all_beam_results = []

    for start_idx in tqdm(range(num_diverse_starts), desc="Diverse starts", disable=not verbose):
        # Generate a random opening prefix
        prefix = []
        board = OthelloBoardState()
        for _ in range(prefix_len):
            valid = board.get_valid_moves()
            if not valid:
                break
            move = random.choice(valid)
            prefix.append(move)
            board.umpire(move)

        if not board.get_valid_moves():
            continue

        # Run beam search from this prefix
        beams = [(list(prefix), deepcopy(board), 0.0)]

        for depth in range(prefix_len, max_depth):
            candidates = []
            for game_seq, brd, cum_mass in beams:
                valid_moves = brd.get_valid_moves()
                if not valid_moves:
                    candidates.append((game_seq, brd, cum_mass, True))
                    continue
                for move in valid_moves:
                    new_seq = game_seq + [move]
                    new_board = deepcopy(brd)
                    new_board.umpire(move)
                    new_valid = new_board.get_valid_moves()
                    if not new_valid:
                        candidates.append((new_seq, new_board, cum_mass, True))
                        continue
                    probs = get_model_probs(model, new_seq, dataset, device)
                    illegal_mass, _ = compute_illegal_mass(probs, new_valid, dataset)
                    candidates.append((new_seq, new_board, cum_mass + illegal_mass, False))

            # Keep top beam_width by cumulative illegal mass
            candidates.sort(key=lambda x: x[2], reverse=True)
            beams = []
            for game_seq, brd, cum_mass, done in candidates[:beam_width]:
                if not done:
                    beams.append((game_seq, brd, cum_mass))
            if not beams:
                break

        # Take the single best game from this start
        if beams:
            best_seq, best_board, best_cum = beams[0]
            all_beam_results.append((best_seq, best_cum))

    # Sort all results by cumulative illegal mass, take top num_save
    all_beam_results.sort(key=lambda x: x[1], reverse=True)

    for game_seq, cum_mass in all_beam_results[:num_save]:
        game_scores = score_full_game(model, game_seq, dataset, device)
        worst_pos = max(game_scores, key=lambda x: x[1])
        result = {
            "game": [permit_reverse(m) for m in game_seq],
            "game_int": game_seq,
            "cumulative_illegal_mass": cum_mass,
            "worst_position_move_idx": worst_pos[0],
            "worst_position_illegal_mass": worst_pos[1],
            "per_position_illegal_mass": game_scores,
        }
        results["beam_search"].append(result)
        if verbose:
            print(f"  Cum mass: {cum_mass:.4f}, worst position: move {worst_pos[0]} "
                  f"({worst_pos[1]:.4f} illegal mass)")

    # --- Random Search ---
    if beam_only:
        if verbose:
            print("\nSkipping random search (--beam-only)")
        return results

    if verbose:
        print(f"\nRandom search over {num_random_seeds} games...")

    worst_random = []
    for _ in tqdm(range(num_random_seeds), desc="Random games", disable=not verbose):
        game = []
        board = OthelloBoardState()
        valid = board.get_valid_moves()
        while valid:
            move = random.choice(valid)
            game.append(move)
            board.umpire(move)
            valid = board.get_valid_moves()
        game_scores = score_full_game(model, game, dataset, device)
        if game_scores:
            worst_pos = max(game_scores, key=lambda x: x[1])
            worst_random.append((game, worst_pos[0], worst_pos[1], game_scores))

    worst_random.sort(key=lambda x: x[2], reverse=True)
    for game, move_idx, mass, scores in worst_random[:num_save]:
        result = {
            "game": [permit_reverse(m) for m in game],
            "game_int": game,
            "worst_position_move_idx": move_idx,
            "worst_position_illegal_mass": mass,
            "per_position_illegal_mass": scores,
        }
        results["random_search"].append(result)
        if verbose:
            print(f"  Worst position: move {move_idx} ({mass:.4f} illegal mass)")

    return results


# =============================================================================
# Approach 2: Search-Based, Illegal Inputs
# =============================================================================

def approach2_illegal_inputs(model, dataset, device, num_trials=100, verbose=True):
    """
    Feed the model sequences it never saw during training and measure
    how much probability it assigns to illegal moves.

    Strategies:
    - Random token sequences
    - Repeated moves
    - Reversed legal games
    - Partially corrupted legal games
    """
    if verbose:
        print("=" * 60)
        print("APPROACH 2: Search-Based, Illegal Inputs")
        print("=" * 60)

    results = {
        "approach": 2,
        "random_tokens": [],
        "repeated_moves": [],
        "reversed_games": [],
        "corrupted_games": [],
    }

    # Build list of all valid token indices (excluding padding)
    valid_tokens = [k for k, v in dataset.itos.items() if v != -100]

    # --- Strategy A: Random Token Sequences ---
    if verbose:
        print(f"\nRandom token sequences ({num_trials} trials)...")

    for _ in tqdm(range(num_trials), desc="Random tokens", disable=not verbose):
        seq_len = random.randint(5, 59)
        # Random board positions from valid tokens
        token_indices = [random.choice(valid_tokens) for _ in range(seq_len)]
        board_positions = [dataset.itos[t] for t in token_indices]

        x = torch.tensor(token_indices, dtype=torch.long)[None, ...].to(device)
        with torch.no_grad():
            logits, _ = model(x)
        probs = F.softmax(logits[0, -1, :], dim=-1)

        # For random inputs we don't have a real board state, so we define
        # "illegal" as: try to construct a board from the legal prefix, and if
        # the sequence is fully illegal from the start, report total non-padding prob
        # as the metric (since no move can be verified as legal)
        # Instead, just measure how spread or concentrated the distribution is
        top_prob = probs.max().item()
        entropy = -(probs * (probs + 1e-10).log()).sum().item()

        results["random_tokens"].append({
            "sequence_len": seq_len,
            "top_prob": top_prob,
            "entropy": entropy,
            "top5_probs": probs.topk(5).values.tolist(),
            "top5_tokens": [dataset.itos[i.item()] for i in probs.topk(5).indices],
        })

    # --- Strategy B: Repeated Moves ---
    if verbose:
        print("\nRepeated move sequences...")

    for token_idx in valid_tokens[:20]:
        board_pos = dataset.itos[token_idx]
        for seq_len in [5, 10, 20, 40]:
            token_seq = [token_idx] * seq_len
            x = torch.tensor(token_seq, dtype=torch.long)[None, ...].to(device)
            with torch.no_grad():
                logits, _ = model(x)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum().item()

            results["repeated_moves"].append({
                "board_pos": board_pos,
                "board_pos_name": permit_reverse(board_pos),
                "seq_len": seq_len,
                "top_prob": probs.max().item(),
                "entropy": entropy,
                "top5_probs": probs.topk(5).values.tolist(),
            })

    # --- Strategy C: Reversed Legal Games ---
    if verbose:
        print(f"\nReversed legal games ({num_trials} trials)...")

    for _ in tqdm(range(num_trials), desc="Reversed games", disable=not verbose):
        game = []
        board = OthelloBoardState()
        valid = board.get_valid_moves()
        while valid:
            move = random.choice(valid)
            game.append(move)
            board.umpire(move)
            valid = board.get_valid_moves()

        reversed_game = list(reversed(game))
        # Measure at several positions
        for check_pos in [5, 10, 20, len(reversed_game) - 1]:
            if check_pos >= len(reversed_game):
                continue
            context = reversed_game[:check_pos]
            x = torch.tensor([dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(device)
            with torch.no_grad():
                logits, _ = model(x)
            probs = F.softmax(logits[0, -1, :], dim=-1)

            # Compare against the board state of the legal prefix of the original game
            legal_board = OthelloBoardState()
            try:
                legal_board.update(game[:check_pos])
                legal_valid = legal_board.get_valid_moves()
                illegal_mass, illegal_moves = compute_illegal_mass(probs, legal_valid, dataset)
            except Exception:
                illegal_mass = -1  # Can't evaluate
                illegal_moves = []

            results["reversed_games"].append({
                "check_position": check_pos,
                "illegal_mass_vs_original": illegal_mass,
                "top_illegal": illegal_moves[:5],
            })

    # --- Strategy D: Partially Corrupted Games ---
    if verbose:
        print(f"\nPartially corrupted games ({num_trials} trials)...")

    for _ in tqdm(range(num_trials), desc="Corrupted games", disable=not verbose):
        game = []
        board = OthelloBoardState()
        valid = board.get_valid_moves()
        while valid:
            move = random.choice(valid)
            game.append(move)
            board.umpire(move)
            valid = board.get_valid_moves()

        if len(game) < 10:
            continue

        # Play legal prefix, then inject random moves
        corruption_point = random.randint(5, min(len(game) - 2, 30))
        legal_prefix = game[:corruption_point]
        num_random = random.randint(3, 10)
        all_board_positions = [dataset.itos[t] for t in valid_tokens]
        random_suffix = [random.choice(all_board_positions) for _ in range(num_random)]

        corrupted = legal_prefix + random_suffix
        if len(corrupted) > 59:
            corrupted = corrupted[:59]

        # Get board state at corruption point
        eval_board = OthelloBoardState()
        eval_board.update(legal_prefix)
        valid_at_corruption = eval_board.get_valid_moves()

        # Get model probs after the full corrupted sequence
        x = torch.tensor([dataset.stoi[s] for s in corrupted], dtype=torch.long)[None, ...].to(device)
        with torch.no_grad():
            logits, _ = model(x)
        probs = F.softmax(logits[0, -1, :], dim=-1)

        # We can't easily determine what's "legal" after corruption, so measure
        # against the legal state at the corruption point
        illegal_mass, illegal_moves = compute_illegal_mass(probs, valid_at_corruption, dataset)

        results["corrupted_games"].append({
            "corruption_point": corruption_point,
            "num_random_suffix": num_random,
            "illegal_mass": illegal_mass,
            "top_illegal": illegal_moves[:5],
            "game_int": corrupted,
            "legal_prefix_len": corruption_point,
        })

    if verbose:
        avg_random = np.mean([r["entropy"] for r in results["random_tokens"]])
        avg_corrupt = np.mean([r["illegal_mass"] for r in results["corrupted_games"]]) if results["corrupted_games"] else 0
        print(f"\n  Avg entropy (random tokens): {avg_random:.4f}")
        print(f"  Avg illegal mass (corrupted): {avg_corrupt:.4f}")

    return results


# =============================================================================
# Approach 3: Gradient-Based, Legal Games
# =============================================================================

def approach3_gradient_legal(model, dataset, device, num_games=50, num_swaps=10,
                             verbose=True):
    """
    Use gradients to find which legal game sequences maximize illegal predictions.

    Algorithm:
    1. Start with a random legal game
    2. At each position, compute gradient of illegal mass w.r.t. input embeddings
    3. For each position, find which alternative legal move's embedding is most
       aligned with the gradient direction
    4. Swap to that move if it creates a fully legal game
    5. Repeat
    """
    if verbose:
        print("=" * 60)
        print("APPROACH 3: Gradient-Based, Legal Games")
        print("=" * 60)

    results = {
        "approach": 3,
        "optimized_games": [],
    }

    for game_idx in tqdm(range(num_games), desc="Gradient-guided games", disable=not verbose):
        # Generate a random legal game
        game = []
        board = OthelloBoardState()
        valid = board.get_valid_moves()
        while valid:
            move = random.choice(valid)
            game.append(move)
            board.umpire(move)
            valid = board.get_valid_moves()

        # Score baseline
        baseline_scores = score_full_game(model, game, dataset, device)
        baseline_worst = max(baseline_scores, key=lambda x: x[1])
        baseline_total = sum(s[1] for s in baseline_scores)

        best_game = list(game)
        best_total = baseline_total

        for swap_iter in range(num_swaps):
            # Pick a random position to try swapping
            swap_pos = random.randint(0, len(game) - 2)

            # Replay up to swap_pos to get the board state
            pre_board = OthelloBoardState()
            if swap_pos > 0:
                pre_board.update(game[:swap_pos])
            legal_at_pos = pre_board.get_valid_moves()
            if len(legal_at_pos) <= 1:
                continue

            # Get gradient of illegal mass at position swap_pos+1 w.r.t. embeddings
            context = game[:swap_pos + 1]
            token_indices = torch.tensor(
                [dataset.stoi[s] for s in context], dtype=torch.long
            )[None, ...].to(device)

            # Get embeddings with gradients
            model.eval()
            tok_emb = model.tok_emb(token_indices)
            pos_emb = model.pos_emb[:, :tok_emb.size(1), :]
            x = tok_emb + pos_emb
            x = x.detach().requires_grad_(True)

            logits = forward_from_embeddings(model, x)
            probs = F.softmax(logits[0, -1, :], dim=-1)

            # Compute illegal mass at the next position
            post_board = deepcopy(pre_board)
            post_board.umpire(game[swap_pos])
            next_valid = post_board.get_valid_moves()
            if not next_valid:
                continue

            valid_set = set(next_valid)
            illegal_mask = torch.zeros(probs.shape[0], device=device)
            for token_idx in range(len(probs)):
                board_pos = dataset.itos[token_idx]
                if board_pos == -100 or board_pos not in valid_set:
                    illegal_mask[token_idx] = 1.0

            illegal_mass = (probs * illegal_mask).sum()
            illegal_mass.backward()

            # Get gradient at the swap position
            grad = x.grad[0, swap_pos, :]  # [512]

            # Find which legal move's embedding is most aligned with gradient
            best_move = None
            best_dot = -float('inf')
            current_emb = model.tok_emb.weight[dataset.stoi[game[swap_pos]]].detach()

            for candidate in legal_at_pos:
                if candidate == game[swap_pos]:
                    continue
                cand_emb = model.tok_emb.weight[dataset.stoi[candidate]].detach()
                direction = cand_emb - current_emb
                dot = (grad * direction).sum().item()
                if dot > best_dot:
                    best_dot = dot
                    best_move = candidate

            if best_move is None:
                continue

            # Try the swap - rebuild the game from swap_pos with the new move
            new_game = list(game[:swap_pos]) + [best_move]
            try:
                test_board = OthelloBoardState()
                test_board.update(new_game)
                # Continue with random legal moves
                valid = test_board.get_valid_moves()
                while valid:
                    move = random.choice(valid)
                    new_game.append(move)
                    test_board.umpire(move)
                    valid = test_board.get_valid_moves()
            except Exception:
                continue

            # Score the new game
            new_scores = score_full_game(model, new_game, dataset, device)
            new_total = sum(s[1] for s in new_scores)

            if new_total > best_total:
                best_game = new_game
                best_total = new_total
                game = new_game  # Continue optimizing the improved game

        # Record results
        final_scores = score_full_game(model, best_game, dataset, device)
        final_worst = max(final_scores, key=lambda x: x[1]) if final_scores else (0, 0)

        result = {
            "game": [permit_reverse(m) for m in best_game],
            "game_int": best_game,
            "baseline_total_illegal_mass": baseline_total,
            "optimized_total_illegal_mass": best_total,
            "improvement": best_total - baseline_total,
            "worst_position_move_idx": final_worst[0],
            "worst_position_illegal_mass": final_worst[1],
            "per_position_illegal_mass": final_scores,
        }
        results["optimized_games"].append(result)

    # Sort by improvement
    results["optimized_games"].sort(key=lambda x: x["optimized_total_illegal_mass"], reverse=True)

    if verbose and results["optimized_games"]:
        best = results["optimized_games"][0]
        print(f"\n  Best game: total illegal mass {best['optimized_total_illegal_mass']:.4f} "
              f"(baseline {best['baseline_total_illegal_mass']:.4f})")
        print(f"  Worst position: move {best['worst_position_move_idx']} "
              f"({best['worst_position_illegal_mass']:.4f} illegal mass)")

    return results


# =============================================================================
# Approach 4: Gradient-Based, Unrestricted Inputs
# =============================================================================

def approach4_gradient_unrestricted(model, dataset, device, num_trials=20,
                                     seq_len=20, num_steps=200, lr=0.01,
                                     verbose=True):
    """
    Directly optimize in embedding space to maximize illegal probability mass.
    No legality constraints on the input.

    Algorithm:
    1. Start with a legal game's embeddings
    2. Make embeddings a learnable parameter
    3. Use gradient ascent to maximize illegal mass
    4. Optionally decode back to nearest tokens
    """
    if verbose:
        print("=" * 60)
        print("APPROACH 4: Gradient-Based, Unrestricted Inputs")
        print("=" * 60)

    results = {
        "approach": 4,
        "optimized_embeddings": [],
    }

    for trial in tqdm(range(num_trials), desc="Embedding optimization", disable=not verbose):
        # Generate a random legal game as starting point
        game = []
        board = OthelloBoardState()
        valid = board.get_valid_moves()
        while valid:
            move = random.choice(valid)
            game.append(move)
            board.umpire(move)
            valid = board.get_valid_moves()

        use_len = min(seq_len, len(game))
        context = game[:use_len]

        # Get the board state at this point for defining legal/illegal
        eval_board = OthelloBoardState()
        eval_board.update(context)
        eval_valid = eval_board.get_valid_moves()
        if not eval_valid:
            continue

        # Create illegal mask
        valid_set = set(eval_valid)
        illegal_mask = torch.zeros(dataset.vocab_size, device=device)
        for token_idx in range(dataset.vocab_size):
            board_pos = dataset.itos[token_idx]
            if board_pos == -100 or board_pos not in valid_set:
                illegal_mask[token_idx] = 1.0

        # Get initial embeddings
        token_indices = torch.tensor(
            [dataset.stoi[s] for s in context], dtype=torch.long
        )[None, ...].to(device)

        with torch.no_grad():
            tok_emb = model.tok_emb(token_indices)
            pos_emb = model.pos_emb[:, :tok_emb.size(1), :]
            init_emb = tok_emb + pos_emb

        # Measure baseline
        with torch.no_grad():
            baseline_logits = forward_from_embeddings(model, init_emb)
            baseline_probs = F.softmax(baseline_logits[0, -1, :], dim=-1)
            baseline_illegal = (baseline_probs * illegal_mask).sum().item()

        # Optimize embeddings
        adv_emb = init_emb.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([adv_emb], lr=lr)

        loss_history = []
        for step in range(num_steps):
            optimizer.zero_grad()
            logits = forward_from_embeddings(model, adv_emb)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            illegal_mass = (probs * illegal_mask).sum()

            # We want to maximize illegal mass, so minimize negative
            loss = -illegal_mass
            loss.backward()
            optimizer.step()
            loss_history.append(illegal_mass.item())

        # Final measurement
        with torch.no_grad():
            final_logits = forward_from_embeddings(model, adv_emb)
            final_probs = F.softmax(final_logits[0, -1, :], dim=-1)
            final_illegal = (final_probs * illegal_mask).sum().item()

        # Decode optimized embeddings back to nearest tokens
        decoded_tokens = []
        with torch.no_grad():
            for pos in range(adv_emb.shape[1]):
                emb_at_pos = adv_emb[0, pos, :] - model.pos_emb[0, pos, :]
                # Find nearest token embedding by cosine similarity
                tok_weights = model.tok_emb.weight  # [vocab_size, 512]
                cos_sim = F.cosine_similarity(
                    emb_at_pos.unsqueeze(0), tok_weights, dim=-1
                )
                nearest_token = cos_sim.argmax().item()
                decoded_tokens.append(nearest_token)

        decoded_moves = [dataset.itos[t] for t in decoded_tokens]
        decoded_names = []
        for m in decoded_moves:
            if m == -100:
                decoded_names.append("PAD")
            else:
                decoded_names.append(permit_reverse(m))

        # L2 distance from original
        with torch.no_grad():
            l2_dist = (adv_emb - init_emb).norm().item()

        result = {
            "original_game": [permit_reverse(m) for m in context],
            "decoded_sequence": decoded_names,
            "context_int": context,
            "decoded_moves_int": decoded_moves,
            "baseline_illegal_mass": baseline_illegal,
            "optimized_illegal_mass": final_illegal,
            "improvement": final_illegal - baseline_illegal,
            "l2_distance": l2_dist,
            "loss_history_start": loss_history[:5],
            "loss_history_end": loss_history[-5:],
            "num_valid_moves": len(eval_valid),
        }
        results["optimized_embeddings"].append(result)

    results["optimized_embeddings"].sort(key=lambda x: x["optimized_illegal_mass"], reverse=True)

    if verbose and results["optimized_embeddings"]:
        best = results["optimized_embeddings"][0]
        print(f"\n  Best result: illegal mass {best['optimized_illegal_mass']:.4f} "
              f"(baseline {best['baseline_illegal_mass']:.4f})")
        print(f"  L2 distance from original: {best['l2_distance']:.4f}")
        print(f"  Decoded sequence: {best['decoded_sequence'][:10]}...")

    return results


# =============================================================================
# Approach 5: Impossible Board States
# =============================================================================

def create_impossible_boards():
    """
    Define board states that cannot be reached through any sequence of legal
    Othello moves. Each board is an 8x8 numpy array: 1=black, -1=white, 0=empty.

    Returns dict of {name: (board_state, next_hand_color, description)}.
    """
    boards = {}

    # 1. Two Islands - two disconnected clusters of pieces
    b = np.zeros((8, 8), dtype=int)
    # Island 1: top-left corner
    b[0, 0], b[0, 1], b[1, 0], b[1, 1] = 1, -1, -1, 1
    # Island 2: bottom-right corner
    b[6, 6], b[6, 7], b[7, 6], b[7, 7] = -1, 1, 1, -1
    boards["two_islands"] = (b, 1, "Two disconnected 2x2 clusters in opposite corners")

    # 2. Empty Center - the 4 starting squares empty, pieces around edges
    b = np.zeros((8, 8), dtype=int)
    # Ring of alternating pieces around center but center itself empty
    for r, c in [(2, 2), (2, 3), (2, 4), (2, 5),
                 (3, 2), (3, 5), (4, 2), (4, 5),
                 (5, 2), (5, 3), (5, 4), (5, 5)]:
        b[r, c] = 1 if (r + c) % 2 == 0 else -1
    # Center 4 squares explicitly empty
    b[3, 3], b[3, 4], b[4, 3], b[4, 4] = 0, 0, 0, 0
    boards["empty_center"] = (b, 1, "Pieces around center but 4 starting squares empty")

    # 3. Overwhelmingly One Color - black block with thin white row
    # Black at (6,c) flanks white at (5,c) via black at (4,c)
    b = np.zeros((8, 8), dtype=int)
    b[:5, :] = 1    # 40 black (rows 0-4)
    b[5, :] = -1    # 8 white (row 5)
    boards["all_black"] = (b, 1, "40 black + 8 white, rows 6-7 empty")

    # 4. Almost One Color (sparse) - 20 black, 1 white
    # Black at (5,2) flanks white at (4,2) via black at (3,2)
    b = np.zeros((8, 8), dtype=int)
    for r in range(4):
        for c in range(5):
            b[r, c] = 1
    b[4, 2] = -1
    boards["only_black_sparse"] = (b, 1, "20 black + 1 white piece")

    # 5. Half-and-Half - clean vertical divide with one bridge piece
    # Black at (6,7) flanks white along col 7 via black at (0,7)
    b = np.zeros((8, 8), dtype=int)
    b[:6, :4] = 1    # 24 black (left cols, rows 0-5)
    b[:6, 4:] = -1   # 24 white (right cols, rows 0-5)
    b[0, 7] = 1      # black bridge in far corner
    boards["half_and_half"] = (b, 1, "Left black, right white, one bridge piece, rows 6-7 empty")

    # 6. Checkerboard - alternating with gap at (5,5)-(6,6)
    # Black at (5,5) flanks white at (4,5) via black at (3,5)
    b = np.zeros((8, 8), dtype=int)
    for r in range(8):
        for c in range(8):
            if 5 <= r <= 6 and 5 <= c <= 6:
                continue
            b[r, c] = 1 if (r + c) % 2 == 0 else -1
    boards["checkerboard"] = (b, 1, "Alternating black/white, 2x2 gap at rows 5-6 cols 5-6")

    # 7. Corners - pieces near all 4 corners, vast empty middle
    # Black at (0,2) flanks white at (0,1) via black at (0,0)
    b = np.zeros((8, 8), dtype=int)
    b[0, 0], b[0, 1] = 1, -1
    b[0, 6], b[0, 7] = -1, 1
    b[7, 0], b[7, 1] = -1, 1
    b[7, 6], b[7, 7] = 1, -1
    boards["corners_only"] = (b, 1, "8 pieces in corner pairs, rest empty")

    # 8. Double Ring - outer + inner ring, empty core
    # Black at (2,2) flanks white at (1,2) via black at (0,2)
    b = np.zeros((8, 8), dtype=int)
    for r in range(8):
        for c in range(8):
            if r == 0 or r == 7 or c == 0 or c == 7:
                b[r, c] = 1 if (r + c) % 2 == 0 else -1
            elif r == 1 or r == 6 or c == 1 or c == 6:
                b[r, c] = 1 if (r + c) % 2 == 0 else -1
    boards["ring"] = (b, 1, "Double ring (outer + inner), empty 4x4 core")

    # 9. Diagonal with flanking piece
    # Black at (0,2) flanks white at (0,1) via black at (0,0) (diagonal)
    b = np.zeros((8, 8), dtype=int)
    for i in range(8):
        b[i, i] = 1 if i % 2 == 0 else -1
    b[0, 1] = -1  # white piece creating flanking opportunity
    boards["diagonal"] = (b, 1, "Diagonal pieces + one off-diagonal white")

    # 10. Random scatter with guaranteed flanking
    # Black at (7,6) flanks white at (7,5) via black at (7,4)
    rng = np.random.RandomState(42)
    b = np.zeros((8, 8), dtype=int)
    positions = rng.choice(64, size=16, replace=False)
    for i, pos in enumerate(positions):
        r, c = pos // 8, pos % 8
        b[r, c] = 1 if i % 2 == 0 else -1
    b[7, 4] = 1    # force B-W-empty flanking triple
    b[7, 5] = -1
    b[7, 6] = 0    # ensure empty target
    boards["random_scatter"] = (b, 1, "16 random pieces + flanking triple")

    return boards


def board_to_fake_sequence(board_state, dataset):
    """
    Convert a board state to a fake move sequence for feeding to the model.

    Strategy: list all occupied positions in row-major order, interleaving
    black and white pieces (since Othello alternates turns).

    As each move is played, Othello flip rules are applied: if placing a
    piece flanks opponent pieces, those pieces are flipped. Moves that don't
    flip anything are still allowed (relaxing the normal legality constraint).

    Positions not in the model's vocabulary (the 4 starting squares) are
    skipped, since they were never played as moves during training.

    Returns:
        (sequence, resulting_board_state): the token sequence and the 8x8
        numpy board that results from playing the sequence with flips applied
        (starting from the standard initial position).
    """
    black_positions = []
    white_positions = []
    for r in range(8):
        for c in range(8):
            pos = r * 8 + c
            if pos not in dataset.stoi:
                continue  # skip positions not in vocabulary (starting squares)
            if board_state[r, c] == 1:
                black_positions.append(pos)
            elif board_state[r, c] == -1:
                white_positions.append(pos)

    # Interleave: black goes first in Othello
    sequence = []
    bi, wi = 0, 0
    while bi < len(black_positions) or wi < len(white_positions):
        if bi < len(black_positions):
            sequence.append(black_positions[bi])
            bi += 1
        if wi < len(white_positions):
            sequence.append(white_positions[wi])
            wi += 1

    # Simulate playing the sequence from the standard initial board,
    # applying any flips required by Othello rules
    resulting_board = OthelloBoardState()
    for i, move in enumerate(sequence):
        color = 1 if i % 2 == 0 else -1  # black first, alternating
        relaxed_play(resulting_board.state, move, color)

    return sequence, resulting_board


def simulate_sequence(sequence, max_len=None):
    """
    Simulate playing a token sequence from the standard initial board,
    applying Othello flips where applicable (relaxed legality).

    Returns the resulting OthelloBoardState.
    """
    if max_len is not None:
        sequence = sequence[:max_len]
    board = OthelloBoardState()
    for i, move in enumerate(sequence):
        color = 1 if i % 2 == 0 else -1
        relaxed_play(board.state, move, color)
    # Set next_hand_color based on sequence length
    board.next_hand_color = 1 if len(sequence) % 2 == 0 else -1
    return board


def score_impossible_board(model, dataset, device, board_state, next_color=1):
    """
    Score the model on an impossible board state.

    board_state: 8x8 numpy array (1=black, -1=white, 0=empty)
    next_color: 1 for black's turn, -1 for white's turn

    The board is converted to a fake move sequence, then played through the
    Othello engine with relaxed legality (flips applied, but moves that don't
    flip are still allowed). Valid moves are computed from the resulting board.

    Returns dict with illegal mass, valid moves, model predictions, and
    the resulting board state after flips.
    """
    # Create fake move sequence (truncate to model's block_size)
    sequence, _ = board_to_fake_sequence(board_state, dataset)
    if len(sequence) > dataset.block_size:
        sequence = sequence[:dataset.block_size]

    # Simulate the (possibly truncated) sequence to get the actual board
    resulting_board = simulate_sequence(sequence)
    result_state = resulting_board.state

    # Get valid moves on the resulting board, filtered to model vocabulary
    valid_moves = [m for m in resulting_board.get_valid_moves() if m in dataset.stoi]

    if not sequence:
        return {
            "illegal_mass": 0.0,
            "illegal_moves": [],
            "valid_moves": valid_moves,
            "num_valid": len(valid_moves),
            "num_empty": int(np.sum(result_state == 0)),
            "num_black": int(np.sum(result_state == 1)),
            "num_white": int(np.sum(result_state == -1)),
            "sequence_len": 0,
            "top_predictions": [],
            "resulting_board": result_state.copy(),
        }

    # Get model predictions
    probs = get_model_probs(model, sequence, dataset, device)

    # Compute illegal mass
    if valid_moves:
        illegal_mass, illegal_moves = compute_illegal_mass(probs, valid_moves, dataset)
    else:
        # No valid moves - all probability is "illegal" in some sense
        illegal_mass = 1.0
        illegal_moves = []

    # Top 10 predictions
    top_probs, top_indices = probs.topk(10)
    top_predictions = []
    for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
        board_pos = dataset.itos[idx]
        if board_pos == -100:
            name = "PAD"
        else:
            name = permit_reverse(board_pos)
        is_legal = board_pos in set(valid_moves) if valid_moves else False
        top_predictions.append({
            "move": name,
            "board_pos": board_pos,
            "prob": p,
            "is_legal": is_legal,
        })

    return {
        "illegal_mass": illegal_mass,
        "illegal_moves": illegal_moves[:10],
        "valid_moves": valid_moves,
        "num_valid": len(valid_moves),
        "num_empty": int(np.sum(result_state == 0)),
        "num_black": int(np.sum(result_state == 1)),
        "num_white": int(np.sum(result_state == -1)),
        "sequence_len": len(sequence),
        "top_predictions": top_predictions,
        "resulting_board": result_state.copy(),
    }


def visualize_impossible_board(model, dataset, device, board_state, valid_moves,
                                sequence, name, description,
                                save_path=None, show=True):
    """
    Visualize an impossible board state and the model's response.

    Similar to visualize_position() but takes a pre-computed board state
    instead of replaying moves.
    """
    probs = get_model_probs(model, sequence, dataset, device)

    valid_set = set(valid_moves)

    # Build 8x8 grids
    prob_grid = np.zeros(64)
    illegal_grid = np.zeros(64)
    for token_idx in range(len(probs)):
        board_pos = dataset.itos[token_idx]
        if board_pos == -100:
            continue
        p = probs[token_idx].item()
        prob_grid[board_pos] = p
        if board_pos not in valid_set:
            illegal_grid[board_pos] = p

    prob_grid = prob_grid.reshape(8, 8)
    illegal_grid = illegal_grid.reshape(8, 8)

    total_illegal = illegal_grid.sum()
    total_legal = prob_grid.sum() - total_illegal

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: Board State ---
    ax = axes[0]
    display_board = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            pos = r * 8 + c
            if board_state[r, c] == 1:
                display_board[r, c] = 1
            elif board_state[r, c] == -1:
                display_board[r, c] = -1
            elif pos in valid_set:
                display_board[r, c] = 0.3

    sns.heatmap(display_board, ax=ax, cbar=False, square=True, linewidths=0.5,
                xticklabels=list(range(1, 9)), yticklabels=list("ABCDEFGH"),
                cmap=sns.color_palette("RdYlGn", as_cmap=True), vmin=-1.5, vmax=1.5,
                center=0)

    for r in range(8):
        for c in range(8):
            if board_state[r, c] == 1:
                ax.add_collection(PatchCollection(
                    [mpatches.Circle((c + 0.5, r + 0.5), 0.3, facecolor='black')],
                    match_original=True))
            elif board_state[r, c] == -1:
                ax.add_collection(PatchCollection(
                    [mpatches.Circle((c + 0.5, r + 0.5), 0.3, facecolor='white',
                                     edgecolor='black', linewidth=0.5)],
                    match_original=True))
            elif r * 8 + c in valid_set:
                ax.add_collection(PatchCollection(
                    [mpatches.Circle((c + 0.5, r + 0.5), 0.1, facecolor='green')],
                    match_original=True))

    num_pieces = int(np.sum(board_state != 0))
    ax.set_title(f"Board (after flips): {name}\n{num_pieces} pieces, "
                 f"{len(valid_moves)} legal moves")

    # --- Panel 2: Model Probabilities ---
    ax = axes[1]
    annot = np.array([f"{v:.2f}" if v > 0.01 else "" for v in prob_grid.flatten()])
    for pos in valid_set:
        r, c = pos // 8, pos % 8
        val = prob_grid[r, c]
        if val > 0.01:
            annot[pos] = f"*{val:.2f}"

    sns.heatmap(prob_grid, ax=ax, cbar=True, square=True, linewidths=0.5,
                xticklabels=list(range(1, 9)), yticklabels=list("ABCDEFGH"),
                cmap=sns.color_palette("Blues", as_cmap=True),
                vmin=0, vmax=max(0.2, prob_grid.max()),
                annot=annot.reshape(8, 8), fmt="")
    ax.set_title(f"Model Probabilities\n(* = legal, total legal: {total_legal:.3f})")

    # --- Panel 3: Illegal Probability ---
    ax = axes[2]
    annot_ill = np.array([f"{v:.2f}" if v > 0.005 else "" for v in illegal_grid.flatten()])

    sns.heatmap(illegal_grid, ax=ax, cbar=True, square=True, linewidths=0.5,
                xticklabels=list(range(1, 9)), yticklabels=list("ABCDEFGH"),
                cmap=sns.color_palette("Reds", as_cmap=True),
                vmin=0, vmax=max(0.1, illegal_grid.max()),
                annot=annot_ill.reshape(8, 8), fmt="")
    ax.set_title(f"Illegal Move Probabilities\n(total illegal mass: {total_illegal:.3f})")

    plt.suptitle(f"Impossible Board: {description}", fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

    return fig


def impossible_boards_experiment(model, dataset, device, verbose=True,
                                  output_dir="adversarial_figures", show=False):
    """
    Test the model on board states that cannot be reached through legal play.

    For each impossible board:
    1. Set the board state directly (bypass umpire)
    2. Compute valid moves from the impossible board using Othello rules
    3. Construct a fake move sequence from the piece positions
    4. Feed the sequence to the model and get predictions
    5. Measure illegal probability mass
    6. Generate visualization

    Returns results dict.
    """
    import os

    if verbose:
        print("=" * 60)
        print("IMPOSSIBLE BOARD STATES EXPERIMENT")
        print("=" * 60)

    boards = create_impossible_boards()
    results = {
        "approach": "impossible_boards",
        "boards": {},
    }

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\nTesting {len(boards)} impossible board configurations...\n")
        print(f"{'Board':>25s} {'Pieces':>6s} {'Empty':>5s} {'Valid':>5s} "
              f"{'Illegal Mass':>12s} {'Top Prediction':>20s}")
        print("-" * 80)

    for name, (board_state, next_color, description) in boards.items():
        score = score_impossible_board(model, dataset, device, board_state, next_color)
        results["boards"][name] = {
            "description": description,
            "next_color": "black" if next_color == 1 else "white",
            **score,
            "valid_moves_names": [permit_reverse(m) for m in score["valid_moves"]],
        }

        if verbose:
            top_pred = score["top_predictions"][0] if score["top_predictions"] else {"move": "N/A", "prob": 0}
            legal_tag = "L" if top_pred.get("is_legal") else "X"
            print(f"{name:>25s} {score['num_black'] + score['num_white']:>6d} "
                  f"{score['num_empty']:>5d} {score['num_valid']:>5d} "
                  f"{score['illegal_mass']:>12.4f} "
                  f"{top_pred['move']:>6s} ({top_pred['prob']:.3f}) [{legal_tag}]")

        # Generate visualization using the resulting board (after flips)
        sequence, _ = board_to_fake_sequence(board_state, dataset)
        if len(sequence) > dataset.block_size:
            sequence = sequence[:dataset.block_size]
        if sequence:
            result_board = score["resulting_board"]
            save_path = os.path.join(output_dir, f"impossible_{name}.png")
            visualize_impossible_board(
                model, dataset, device, result_board, score["valid_moves"],
                sequence, name, description,
                save_path=save_path, show=show,
            )

    if verbose:
        print("\n" + "-" * 80)
        masses = [r["illegal_mass"] for r in results["boards"].values()]
        print(f"Average illegal mass across all boards: {np.mean(masses):.4f}")
        print(f"Max illegal mass: {max(masses):.4f}")
        worst_name = max(results["boards"].items(), key=lambda x: x[1]["illegal_mass"])[0]
        print(f"Worst board: {worst_name}")

    return results


# =============================================================================
# Analysis Script
# =============================================================================

def run_analysis(all_results, verbose=True):
    """Summarize and compare results from all approaches."""
    if verbose:
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)

    summary = {}

    # Approach 1
    if 1 in all_results:
        r = all_results[1]
        beam_worst = [g["worst_position_illegal_mass"] for g in r.get("beam_search", [])]
        random_worst = [g["worst_position_illegal_mass"] for g in r.get("random_search", [])]
        summary["approach_1"] = {
            "beam_search_best_illegal_mass": max(beam_worst) if beam_worst else 0,
            "beam_search_avg_illegal_mass": np.mean(beam_worst) if beam_worst else 0,
            "random_search_best_illegal_mass": max(random_worst) if random_worst else 0,
            "random_search_avg_illegal_mass": np.mean(random_worst) if random_worst else 0,
        }
        if verbose:
            s = summary["approach_1"]
            print(f"\nApproach 1 (Search, Legal):")
            print(f"  Beam search - best: {s['beam_search_best_illegal_mass']:.4f}, "
                  f"avg: {s['beam_search_avg_illegal_mass']:.4f}")
            print(f"  Random search - best: {s['random_search_best_illegal_mass']:.4f}, "
                  f"avg: {s['random_search_avg_illegal_mass']:.4f}")

    # Approach 2
    if 2 in all_results:
        r = all_results[2]
        corrupt_masses = [g["illegal_mass"] for g in r.get("corrupted_games", [])]
        summary["approach_2"] = {
            "avg_entropy_random_tokens": np.mean([g["entropy"] for g in r.get("random_tokens", [])]) if r.get("random_tokens") else 0,
            "corrupted_best_illegal_mass": max(corrupt_masses) if corrupt_masses else 0,
            "corrupted_avg_illegal_mass": np.mean(corrupt_masses) if corrupt_masses else 0,
        }
        if verbose:
            s = summary["approach_2"]
            print(f"\nApproach 2 (Search, Illegal Inputs):")
            print(f"  Avg entropy (random tokens): {s['avg_entropy_random_tokens']:.4f}")
            print(f"  Corrupted games - best: {s['corrupted_best_illegal_mass']:.4f}, "
                  f"avg: {s['corrupted_avg_illegal_mass']:.4f}")

    # Approach 3
    if 3 in all_results:
        r = all_results[3]
        games = r.get("optimized_games", [])
        improvements = [g["improvement"] for g in games]
        total_masses = [g["optimized_total_illegal_mass"] for g in games]
        summary["approach_3"] = {
            "best_total_illegal_mass": max(total_masses) if total_masses else 0,
            "avg_total_illegal_mass": np.mean(total_masses) if total_masses else 0,
            "avg_improvement_over_baseline": np.mean(improvements) if improvements else 0,
            "best_single_position": max(
                (g["worst_position_illegal_mass"] for g in games), default=0
            ),
        }
        if verbose:
            s = summary["approach_3"]
            print(f"\nApproach 3 (Gradient, Legal):")
            print(f"  Best total illegal mass: {s['best_total_illegal_mass']:.4f}")
            print(f"  Avg improvement: {s['avg_improvement_over_baseline']:.4f}")
            print(f"  Best single position: {s['best_single_position']:.4f}")

    # Approach 4
    if 4 in all_results:
        r = all_results[4]
        embs = r.get("optimized_embeddings", [])
        final_masses = [g["optimized_illegal_mass"] for g in embs]
        improvements = [g["improvement"] for g in embs]
        summary["approach_4"] = {
            "best_illegal_mass": max(final_masses) if final_masses else 0,
            "avg_illegal_mass": np.mean(final_masses) if final_masses else 0,
            "avg_improvement": np.mean(improvements) if improvements else 0,
            "avg_l2_distance": np.mean([g["l2_distance"] for g in embs]) if embs else 0,
        }
        if verbose:
            s = summary["approach_4"]
            print(f"\nApproach 4 (Gradient, Unrestricted):")
            print(f"  Best illegal mass: {s['best_illegal_mass']:.4f}")
            print(f"  Avg illegal mass: {s['avg_illegal_mass']:.4f}")
            print(f"  Avg L2 distance: {s['avg_l2_distance']:.4f}")

    # Cross-approach comparison
    if verbose:
        print(f"\n{'='*60}")
        print("CROSS-APPROACH COMPARISON (best single-position illegal mass)")
        print(f"{'='*60}")
        bests = {}
        if 1 in all_results:
            bests["Approach 1 (Search, Legal)"] = summary["approach_1"]["beam_search_best_illegal_mass"]
        if 2 in all_results:
            bests["Approach 2 (Search, Illegal)"] = summary["approach_2"]["corrupted_best_illegal_mass"]
        if 3 in all_results:
            bests["Approach 3 (Gradient, Legal)"] = summary["approach_3"]["best_single_position"]
        if 4 in all_results:
            bests["Approach 4 (Gradient, Unrestricted)"] = summary["approach_4"]["best_illegal_mass"]
        for name, val in sorted(bests.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {val:.4f}")

    return summary


# =============================================================================
# Serialization Helpers
# =============================================================================

def _make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def serialize_results(obj):
    """Recursively convert numpy/torch types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: serialize_results(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_results(v) for v in obj]
    elif isinstance(obj, tuple):
        return [serialize_results(v) for v in obj]
    else:
        return _make_serializable(obj)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Adversarial Games for Othello-GPT")
    parser.add_argument("--approach", type=str, default="all",
                        help="Which approach to run: 1, 2, 3, 4, all, or none (just error-prone search)")
    parser.add_argument("--ckpt", type=str, default="./ckpts/gpt_synthetic.ckpt",
                        help="Path to model checkpoint")
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Use synthetic dataset (default)")
    parser.add_argument("--championship", action="store_true",
                        help="Use championship dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="adversarial_results.json",
                        help="Output file for results")
    # Approach 1 params
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--num-random-seeds", type=int, default=100)
    parser.add_argument("--num-save", type=int, default=50,
                        help="Number of top games to save per search method")
    parser.add_argument("--num-diverse-starts", type=int, default=50,
                        help="Number of different random openings for beam search")
    parser.add_argument("--prefix-len", type=int, default=5,
                        help="Length of random opening prefix for diverse beam search")
    # Approach 2 params
    parser.add_argument("--num-trials", type=int, default=100)
    # Approach 3 params
    parser.add_argument("--num-games", type=int, default=50)
    parser.add_argument("--num-swaps", type=int, default=10)
    # Approach 4 params
    parser.add_argument("--num-emb-trials", type=int, default=20)
    parser.add_argument("--emb-seq-len", type=int, default=20)
    parser.add_argument("--emb-steps", type=int, default=200)
    parser.add_argument("--emb-lr", type=float, default=0.01)
    # Error-prone games params
    parser.add_argument("--error-games", type=int, default=500,
                        help="Number of games to search for error-prone ones")
    parser.add_argument("--error-threshold", type=float, default=0.1,
                        help="Illegal mass threshold to count as a mistake")
    # Visualization params
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of worst positions after running")
    parser.add_argument("--visualize-only", type=str, default=None,
                        help="Skip approaches, just visualize results from this JSON file")
    parser.add_argument("--fig-dir", type=str, default="adversarial_figures",
                        help="Directory to save figures")
    parser.add_argument("--num-vis-games", type=int, default=3,
                        help="Number of worst games to visualize")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures without displaying (for headless environments)")
    # Impossible boards params
    parser.add_argument("--impossible", action="store_true",
                        help="Run impossible board states experiment")
    parser.add_argument("--impossible-only", action="store_true",
                        help="Run only the impossible board states experiment (skip all others)")
    parser.add_argument("--beam-only", action="store_true",
                        help="Run only the beam search (skip error-prone search and random baseline)")

    args = parser.parse_args()
    set_seed(args.seed)

    synthetic = not args.championship
    print(f"Loading model from {args.ckpt}...")
    model, dataset, othello, device = load_model_and_dataset(args.ckpt, synthetic)
    print(f"Model loaded on {device}\n")

    show_figs = not args.no_show

    # If --visualize-only, skip all approaches and just visualize existing results
    if args.visualize_only:
        print(f"Visualizing results from {args.visualize_only}...")
        visualize_game_results(args.visualize_only, model, dataset, device,
                               output_dir=args.fig_dir, num_games=args.num_vis_games,
                               show=show_figs)
        print(f"Figures saved to {args.fig_dir}/")
        return

    # If --impossible-only, just run the impossible boards experiment
    if args.impossible_only:
        impossible_results = impossible_boards_experiment(
            model, dataset, device, verbose=True,
            output_dir=args.fig_dir, show=show_figs,
        )
        output = {"results": {"impossible_boards": serialize_results(impossible_results)}}
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
        return

    if args.approach == "all":
        approaches_to_run = [1, 2, 3, 4]
    elif args.approach == "none":
        approaches_to_run = []
    else:
        approaches_to_run = [int(x) for x in args.approach.split(",")]
    all_results = {}

    # Run error-prone game search (unless --beam-only)
    if not args.beam_only:
        error_prone_results = find_error_prone_games(
            model, dataset, device,
            num_games=args.error_games,
            threshold=args.error_threshold,
        )
        all_results["error_prone"] = error_prone_results

    if 1 in approaches_to_run:
        all_results[1] = approach1_beam_search(
            model, dataset, device,
            beam_width=args.beam_width,
            num_random_seeds=args.num_random_seeds,
            num_save=args.num_save,
            num_diverse_starts=args.num_diverse_starts,
            prefix_len=args.prefix_len,
            beam_only=args.beam_only,
        )

    if 2 in approaches_to_run:
        all_results[2] = approach2_illegal_inputs(
            model, dataset, device,
            num_trials=args.num_trials,
        )

    if 3 in approaches_to_run:
        all_results[3] = approach3_gradient_legal(
            model, dataset, device,
            num_games=args.num_games,
            num_swaps=args.num_swaps,
        )

    if 4 in approaches_to_run:
        all_results[4] = approach4_gradient_unrestricted(
            model, dataset, device,
            num_trials=args.num_emb_trials,
            seq_len=args.emb_seq_len,
            num_steps=args.emb_steps,
            lr=args.emb_lr,
        )

    # Run impossible boards experiment if requested
    if args.impossible:
        impossible_results = impossible_boards_experiment(
            model, dataset, device, verbose=True,
            output_dir=args.fig_dir, show=show_figs,
        )
        all_results["impossible_boards"] = impossible_results

    # Run analysis
    summary = run_analysis(all_results)

    output = {
        "summary": summary,
        "results": serialize_results({str(k): v for k, v in all_results.items()}),
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Generate visualizations if requested
    if args.visualize:
        print(f"\nGenerating visualizations...")
        visualize_game_results(args.output, model, dataset, device,
                               output_dir=args.fig_dir, num_games=args.num_vis_games,
                               show=show_figs)
        print(f"Figures saved to {args.fig_dir}/")


if __name__ == "__main__":
    main()
