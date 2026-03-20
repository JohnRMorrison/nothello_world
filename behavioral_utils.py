"""Shared utilities for behavioral heuristic extraction pipeline.

Used by behavioral_collect.py (Stage 1), behavioral_adversarial.py (Stage 2),
behavioral_fit.py (Stage 3), and behavioral_validate.py (Stage 4).
"""

import os
import sys
import numpy as np
from multiprocessing import Pool, cpu_count

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# Constants
# =============================================================================

# 60 valid board positions (exclude center 4 starting squares)
VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})
MOVE_TO_IDX = {m: i for i, m in enumerate(VALID_MOVES)}
IDX_TO_MOVE = {i: m for m, i in MOVE_TO_IDX.items()}
N_MOVES = 60

# Position range for analysis (0-indexed move number)
POS_START = 4   # first position with meaningful context
POS_END = 59    # last position (inclusive in game, exclusive in range)


# =============================================================================
# Feature computation
# =============================================================================

def build_120d_features(games, pos_start=POS_START, pos_end=POS_END):
    """Build 120-d move-history features for a batch of games.

    Features per position t:
      when[i]  (60): normalized step at which move i was played: (step+1)/60.
                     0 if not yet played at position t.
      even[i]  (60): 1 if move i was played on an even step (black's turn),
                     0 if played on odd step or not yet played.

    Together, when[i] and even[i] encode both the timing and color of each
    piece on the board. when[i] > 0 means the cell is occupied; even[i]
    tells you which player occupies it.

    Args:
        games: list of lists of board positions (0-63), each game ~60 moves
        pos_start: first position to include (inclusive)
        pos_end: last position to include (exclusive)

    Returns:
        features: (n_samples, 120) float32 numpy array
        positions: (n_samples,) int8 numpy array
    """
    n_games = len(games)
    length = pos_end - pos_start
    n_samples = n_games * length

    # Convert games to (N, max_len) array of move indices
    max_len = max(len(g) for g in games)
    game_arr = np.zeros((n_games, max_len), dtype=np.int32)
    game_lens = np.array([len(g) for g in games], dtype=np.int32)
    for i, game in enumerate(games):
        for s, move in enumerate(game):
            game_arr[i, s] = MOVE_TO_IDX[move]

    # Build per-game lookup: step_of_move[g, m] = step when move m was played
    # -1 if never played
    step_of_move = np.full((n_games, N_MOVES), -1, dtype=np.int32)
    for s in range(max_len):
        mask = s < game_lens
        move_indices = game_arr[mask, s]
        step_of_move[np.where(mask)[0], move_indices] = s

    # Pre-allocate output
    features = np.zeros((n_samples, 120), dtype=np.float32)
    positions = np.zeros(n_samples, dtype=np.int8)

    for ti, t in enumerate(range(pos_start, pos_end)):
        start = ti * n_games
        end = start + n_games

        # played: move was played at or before step t
        played = (step_of_move >= 0) & (step_of_move <= t)  # (N, 60)
        # when: normalized step (0 if not played)
        when = np.where(played, (step_of_move + 1) / 60.0, 0.0)  # (N, 60)
        # even: played on even step (black's turn)
        even = np.where(played, (step_of_move % 2 == 0).astype(np.float32), 0.0)

        features[start:end, :N_MOVES] = when
        features[start:end, N_MOVES:] = even
        positions[start:end] = t

    return features, positions


# =============================================================================
# Model loading
# =============================================================================

def load_model(ckpt_path="./ckpts/gpt_synthetic.ckpt"):
    """Load OthelloGPT model and dataset.

    Returns:
        model: GPT model in eval mode
        dataset: CharDataset with stoi/itos mappings
        device: torch device
    """
    import torch
    from data import get_othello
    from mingpt.dataset import CharDataset
    from mingpt.model import GPT, GPTConfig

    # Load a small number of games just to build the vocabulary
    # (ood_num=-1 loads all 20M games which is wasteful here)
    othello = get_othello(ood_num=100)

    dataset = CharDataset(othello)
    mconf = GPTConfig(dataset.vocab_size, dataset.block_size,
                      n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, dataset, device


# =============================================================================
# Vocab-to-position mapping
# =============================================================================

def build_vocab_to_pos_map(dataset):
    """Build mapping between model vocab indices and board positions.

    The CharDataset vocab has 61 entries:
      index 0 -> -100 (padding)
      indices 1-60 -> the 60 valid board positions in sorted order

    Returns:
        vocab_to_pos: (61,) int array, vocab_to_pos[token_idx] = board position
                      (-1 for padding)
        pos_to_vocab: (64,) int array, pos_to_vocab[board_pos] = token index
                      (-1 for center squares not in vocab)
    """
    vocab_to_pos = np.full(dataset.vocab_size, -1, dtype=np.int32)
    pos_to_vocab = np.full(64, -1, dtype=np.int32)

    for token_idx in range(dataset.vocab_size):
        board_pos = dataset.itos[token_idx]
        if board_pos == -100:
            vocab_to_pos[token_idx] = -1
        else:
            vocab_to_pos[token_idx] = board_pos
            pos_to_vocab[board_pos] = token_idx

    return vocab_to_pos, pos_to_vocab


def extract_probs_60d(softmax_probs, vocab_to_pos):
    """Extract 60-d probability vectors from model output.

    Args:
        softmax_probs: (B, T, 61) tensor or numpy array of softmax probabilities
        vocab_to_pos: (61,) mapping from vocab index to board position

    Returns:
        probs_60d: (B, T, 60) numpy float32 array, ordered by MOVE_TO_IDX
    """
    if hasattr(softmax_probs, 'cpu'):
        softmax_probs = softmax_probs.cpu().numpy()

    B, T, V = softmax_probs.shape
    probs_60d = np.zeros((B, T, N_MOVES), dtype=np.float32)

    for token_idx in range(V):
        board_pos = vocab_to_pos[token_idx]
        if board_pos < 0:
            continue
        move_idx = MOVE_TO_IDX[board_pos]
        probs_60d[:, :, move_idx] = softmax_probs[:, :, token_idx]

    return probs_60d


# =============================================================================
# Legal mask computation
# =============================================================================

def _compute_legal_masks_worker(args):
    """Worker function for parallel legal mask computation."""
    games_chunk, pos_start, pos_end = args
    from data.othello import OthelloBoardState

    length = pos_end - pos_start
    n_games = len(games_chunk)
    legal = np.zeros((n_games, length, N_MOVES), dtype=np.uint8)

    for gi, game in enumerate(games_chunk):
        board = OthelloBoardState()
        for t in range(max(pos_end, len(game))):
            if pos_start <= t < pos_end:
                ti = t - pos_start
                valid_moves = board.get_valid_moves()
                for m in valid_moves:
                    if m in MOVE_TO_IDX:
                        legal[gi, ti, MOVE_TO_IDX[m]] = 1
            if t < len(game):
                board.umpire(game[t])

    return legal


def compute_legal_masks_batch(games, pos_start=POS_START, pos_end=POS_END,
                              n_workers=None):
    """Compute legal move masks for a batch of games in parallel.

    Args:
        games: list of game sequences
        pos_start: first position (inclusive)
        pos_end: last position (exclusive)
        n_workers: number of parallel workers (default: min(cpu_count(), 8))

    Returns:
        legal: (n_games * length, 60) uint8 numpy array, flattened
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    n_games = len(games)
    chunk_size = max(1, n_games // n_workers)
    chunks = []
    for i in range(0, n_games, chunk_size):
        chunks.append((games[i:i + chunk_size], pos_start, pos_end))

    if n_workers > 1 and n_games > 100:
        with Pool(n_workers) as pool:
            results = pool.map(_compute_legal_masks_worker, chunks)
    else:
        results = [_compute_legal_masks_worker(c) for c in chunks]

    # Concatenate along game axis, then reshape to (n_samples, 60)
    legal = np.concatenate(results, axis=0)  # (n_games, length, 60)
    length = pos_end - pos_start
    return legal.reshape(n_games * length, N_MOVES)


# =============================================================================
# Game loading
# =============================================================================

def load_games_from_pickles(pickle_dir, file_pattern="gen10e5*.pickle",
                            max_games=None):
    """Load games from pickle files in a directory.

    Args:
        pickle_dir: directory containing pickle files
        file_pattern: glob pattern for pickle files
        max_games: maximum number of games to load (None = all)

    Returns:
        games: list of game sequences
    """
    import pickle
    import glob

    files = sorted(glob.glob(os.path.join(pickle_dir, file_pattern)))
    games = []
    for f in files:
        with open(f, 'rb') as fh:
            batch = pickle.load(fh)
        games.extend(batch)
        if max_games and len(games) >= max_games:
            games = games[:max_games]
            break

    return games


def load_shard_games(shard_id, games_per_shard=100000,
                     pickle_dir="data/othello_synthetic"):
    """Load games for a specific shard.

    Each shard loads a different subset of the available pickle files.
    With ~237 files of ~100K games each, we have ~23.7M games total.
    20 shards x 100K = 2M games used.

    Args:
        shard_id: 0-19
        games_per_shard: number of games per shard
        pickle_dir: directory with pickle files

    Returns:
        games: list of game sequences
    """
    import pickle
    import glob

    files = sorted(glob.glob(os.path.join(pickle_dir, "gen10e5*.pickle")))
    if not files:
        raise FileNotFoundError(f"No gen10e5*.pickle files in {pickle_dir}")

    # Each file has ~100K games. Each shard needs 100K games.
    # Shard N loads file N (if files have ~100K each).
    # If files have different sizes, load sequentially and skip.
    games = []
    skip = shard_id * games_per_shard
    loaded = 0

    for f in files:
        with open(f, 'rb') as fh:
            batch = pickle.load(fh)

        if skip >= len(batch):
            skip -= len(batch)
            continue

        start = skip
        skip = 0
        needed = games_per_shard - loaded
        games.extend(batch[start:start + needed])
        loaded = len(games)

        if loaded >= games_per_shard:
            break

    if len(games) < games_per_shard:
        print(f"Warning: shard {shard_id} only loaded {len(games)}/{games_per_shard} games")

    return games[:games_per_shard]
