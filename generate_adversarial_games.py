"""Generate adversarial Othello games using perturbed heuristic MLPs.

Pipeline:
  1. Load trained MLP (move features → 1024 hidden → board state)
  2. Add a policy head (board state → next move) and train it on real games
  3. Perturb the heuristic hidden units by corruption level α
  4. Generate games by: move history → perturbed MLP → board state → policy → next move

Usage:
    # Step 1: Train the policy head (run once)
    python generate_adversarial_games.py --train-policy --max-games 100000

    # Step 2: Generate games at various perturbation levels
    python generate_adversarial_games.py --generate --alpha 0.0 --n-games 100000
    python generate_adversarial_games.py --generate --alpha 0.3 --n-games 100000
    python generate_adversarial_games.py --generate --alpha 1.0 --n-games 100000
"""
import sys, os
sys.path.insert(0, '.')

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _build_mlp, N_MOVES, _VALID_MOVES, _MOVE_TO_IDX, POS_START, POS_END, OPTIONS,
    get_device,
)
from experiments.mathematical_transformation_experiments.probe_variant_boards import (
    load_games, OthelloBoardState, STOI, ITOS,
)

# ---------------------------------------------------------------------------
# Feature computation for a single position (used during game generation)
# ---------------------------------------------------------------------------

def _compute_move_features(move_history, t):
    """Compute 180-d move features for position t given move history.

    move_history: list of raw moves (0-63, excluding center 4)
    t: current position index (0-based, i.e. number of moves played so far - 1)

    Returns: (180,) float32 tensor
    """
    features = np.zeros(180, dtype=np.float32)
    for step, move in enumerate(move_history[:t + 1]):
        idx = _MOVE_TO_IDX[move]
        features[idx] = 1.0                          # played
        features[N_MOVES + idx] = (step + 1) / 60.0  # when
        features[2 * N_MOVES + idx] = float(step % 2 == 0)  # even
    return torch.tensor(features, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Policy head: board state (64*3 logits) → next move (60 logits)
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """Predicts next move from board state logits."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        # Input: 192 (64 squares × 3 classes) — raw logits from board MLP
        # We convert to soft probabilities first, then predict
        self.net = nn.Sequential(
            nn.Linear(64 * OPTIONS, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_MOVES),
        )

    def forward(self, board_logits):
        """board_logits: (B, 192) raw MLP output."""
        return self.net(board_logits)


# ---------------------------------------------------------------------------
# Train policy head
# ---------------------------------------------------------------------------

def _compute_legal_moves_chunk(args):
    """Worker: compute legal moves for a chunk of games."""
    games_chunk, pos_start, pos_end = args
    length = pos_end - pos_start
    n = len(games_chunk)
    legal = np.zeros((n, length, N_MOVES), dtype=np.int8)

    for gi, game in enumerate(games_chunk):
        board = OthelloBoardState()
        for s in range(pos_start):
            board.umpire(game[s])
        for ti, t in enumerate(range(pos_start, pos_end)):
            board.umpire(game[t])
            for m in board.get_valid_moves():
                if m in _MOVE_TO_IDX:
                    legal[gi, ti, _MOVE_TO_IDX[m]] = 1
    return legal


def _compute_legal_moves_parallel(game_list, pos_start, pos_end):
    """Compute legal-move vectors using multiprocessing."""
    from multiprocessing import Pool, cpu_count

    n_games = len(game_list)
    length = pos_end - pos_start
    n_workers = min(cpu_count(), 8)
    chunk_size = (n_games + n_workers - 1) // n_workers

    chunks = []
    for i in range(0, n_games, chunk_size):
        chunks.append((game_list[i:i + chunk_size], pos_start, pos_end))

    print(f"    Using {n_workers} workers for {n_games} games...", flush=True)
    with Pool(n_workers) as pool:
        results = pool.map(_compute_legal_moves_chunk, chunks)

    # Reassemble: each result is (chunk_n, length, 60)
    # Need shape (n_games * length, 60) with games interleaved per position
    legal = np.zeros((n_games * length, N_MOVES), dtype=np.float32)
    gi_offset = 0
    for res in results:
        chunk_n = res.shape[0]
        for ti in range(length):
            idx_start = ti * n_games + gi_offset
            legal[idx_start:idx_start + chunk_n] = res[:, ti]
        gi_offset += chunk_n

    return legal


def _load_precomputed_chunks(chunk_dir, n_games, length, game_offset=0):
    """Load legal moves from precomputed chunk files (from precompute_legal_moves.py).

    Chunk files have shape (chunk_n, length, 60) in game-major order.
    We need position-major order: (length * n_games, 60).

    Args:
        chunk_dir: directory containing legal_moves_chunk_*.npz files
        n_games: number of games to load
        length: number of positions per game
        game_offset: skip this many games from the start (for loading subsets)
    Returns None if chunks not found or insufficient.
    """
    import glob as globmod
    chunk_files = sorted(globmod.glob(os.path.join(chunk_dir, "legal_moves_chunk_*.npz")))
    if not chunk_files:
        return None

    # Load chunks, skipping games before game_offset
    parts = []
    total_games = 0
    games_needed = game_offset + n_games
    for f in chunk_files:
        data = np.load(f)['legal']  # (chunk_n, length, 60)
        parts.append(data)
        total_games += data.shape[0]
        if total_games >= games_needed:
            break

    if total_games < games_needed:
        print(f"  Warning: only {total_games} games in chunks, need {games_needed}")
        return None

    # Concatenate, slice to [game_offset : game_offset + n_games]
    all_legal = np.concatenate(parts, axis=0)[game_offset:game_offset + n_games]
    # Transpose to position-major: (length, n_games, 60) → (length*n_games, 60)
    legal = all_legal.transpose(1, 0, 2).reshape(-1, all_legal.shape[2])
    print(f"  Loaded precomputed legal moves: games [{game_offset}:{game_offset + n_games}]")
    return legal.astype(np.float32)


def _get_legal_moves(game_list, pos_start, pos_end, cache_path=None,
                     chunk_dir=None, game_offset=0, n_games=None):
    """Compute or load cached legal-move vectors.

    Args:
        game_list: list of games (can be None if n_games is provided and chunks exist)
        n_games: number of games (used instead of len(game_list) when game_list is None)

    Returns: (n_samples, 60) float32 tensor
    """
    if n_games is None:
        n_games = len(game_list)

    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached legal moves from {cache_path}")
        legal = np.load(cache_path)['legal']
        return torch.tensor(legal, dtype=torch.float32)

    # Try precomputed chunks
    if chunk_dir:
        length = pos_end - pos_start
        legal = _load_precomputed_chunks(chunk_dir, n_games, length,
                                          game_offset=game_offset)
        if legal is not None:
            return torch.tensor(legal, dtype=torch.float32)

    assert game_list is not None, "game_list required when precomputed chunks unavailable"
    print("  Computing legal moves for each position...")
    legal = _compute_legal_moves_parallel(game_list, pos_start, pos_end)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, legal=legal.astype(np.int8))
        print(f"  Cached legal moves to {cache_path}")

    return torch.tensor(legal, dtype=torch.float32)


def _build_chunk_logits(game_chunk, mlp_even, mlp_odd, device,
                        pos_start, pos_end):
    """Build board logits and position indices for a chunk of games.

    Returns: (board_logits, positions) tensors on CPU.
    """
    from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
        _build_move_features_batch,
    )
    X, _, pos = _build_move_features_batch(game_chunk, pos_start, pos_end,
                                            include_pairwise=False,
                                            skip_labels=True)
    batch_size = 2048
    board_logits = torch.zeros(len(X), 64 * OPTIONS)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = X[i:i + batch_size].to(device)
            p = pos[i:i + batch_size]
            even_mask = (p % 2 == 0)
            odd_mask = ~even_mask
            out = torch.zeros(len(x), 64 * OPTIONS, device=device)
            if even_mask.any():
                out[even_mask] = mlp_even(x[even_mask])
            if odd_mask.any():
                out[odd_mask] = mlp_odd(x[odd_mask])
            board_logits[i:i + batch_size] = out.cpu()
    del X  # free memory
    return board_logits, pos


def _logits_to_board_state(board_logits):
    """Convert 192-dim logits to uint8 board state (argmax over 3 classes per square).

    Input: (N, 192) tensor of raw logits
    Returns: (N, 64) uint8 numpy array with values 0, 1, 2
    """
    # Reshape to (N, 64, 3) and take argmax
    return board_logits.reshape(-1, 64, OPTIONS).argmax(dim=-1).byte().numpy()


def _board_state_to_onehot(board_state):
    """Convert uint8 board state back to one-hot 192-dim tensor.

    Input: (N, 64) uint8 numpy array with values 0, 1, 2
    Returns: (N, 192) float32 tensor
    """
    t = torch.from_numpy(board_state.astype(np.int64))
    return nn.functional.one_hot(t, num_classes=OPTIONS).float().reshape(-1, 64 * OPTIONS)


def _load_board_state_chunk(cache_dir, chunk_id, length):
    """Load a cached board state chunk and convert to one-hot.

    Cache stores game-major (chunk_games, length, 64) uint8.
    Returns position-major (length * chunk_games, 192) float32 tensor, or None.
    """
    path = os.path.join(cache_dir, f"board_state_chunk_{chunk_id:04d}.npz")
    if not os.path.exists(path):
        return None
    bs = np.load(path)['board_state']  # (chunk_games, length, 64)
    # Transpose to position-major: (length, chunk_games, 64) → (length*chunk_games, 64)
    bs = bs.transpose(1, 0, 2).reshape(-1, 64)
    return _board_state_to_onehot(bs)


def precompute_board_states(args):
    """Precompute board states from MLP and save as uint8 chunks.

    Saves ~1.5GB per 500K-game chunk (vs ~8.6GB for float16 logits).
    All sweep jobs can then load these instead of recomputing.
    """
    device = get_device()
    print(f"Device: {device}")

    # Load MLP
    ckpt = torch.load(args.mlp_checkpoint, map_location='cpu')
    input_dim = ckpt['input_dim']
    hidden_dim = ckpt['hidden_dim']
    num_hidden = ckpt.get('num_hidden_layers', 1)

    mlp_even = _build_mlp(input_dim, hidden_dim, 64 * OPTIONS, num_hidden).to(device)
    mlp_odd = _build_mlp(input_dim, hidden_dim, 64 * OPTIONS, num_hidden).to(device)
    mlp_even.load_state_dict(ckpt['even'])
    mlp_odd.load_state_dict(ckpt['odd'])
    mlp_even.eval()
    mlp_odd.eval()
    print(f"Loaded board MLP: input={input_dim}, hidden={hidden_dim}")

    # Load games
    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    print(f"Loaded {len(games)} games")

    # Same train/eval split as train_policy
    n_eval = min(max(int(len(games) * 0.1), 500), 50000)
    train_games = games[:len(games) - n_eval]
    eval_games = games[len(games) - n_eval:]

    game_chunk_size = min(500000, len(train_games))
    n_chunks = (len(train_games) + game_chunk_size - 1) // game_chunk_size
    pos_start, pos_end = POS_START, POS_END - 1
    length = pos_end - pos_start

    cache_dir = os.path.join(args.output_dir, "board_state_cache")
    os.makedirs(cache_dir, exist_ok=True)

    import time as _time

    # Save eval chunk
    eval_path = os.path.join(cache_dir, "board_state_eval.npz")
    if os.path.exists(eval_path):
        print("Eval chunk already exists, skipping", flush=True)
    else:
        t0 = _time.time()
        logits, _ = _build_chunk_logits(
            eval_games, mlp_even, mlp_odd, device, pos_start, pos_end)
        bs = _logits_to_board_state(logits).reshape(length, len(eval_games), 64)
        bs = bs.transpose(1, 0, 2)  # (n_eval, length, 64)
        np.savez_compressed(eval_path, board_state=bs)
        size_mb = os.path.getsize(eval_path) / 1e6
        print(f"Eval: {len(eval_games)} games, {size_mb:.0f}MB, "
              f"{_time.time() - t0:.0f}s", flush=True)
        del logits, bs

    # Save train chunks
    print(f"Precomputing {n_chunks} train chunks of {game_chunk_size} games...")
    for ci in range(n_chunks):
        g_start = ci * game_chunk_size
        g_end = min(g_start + game_chunk_size, len(train_games))
        out_path = os.path.join(cache_dir, f"board_state_chunk_{ci:04d}.npz")
        if os.path.exists(out_path):
            print(f"  Chunk {ci}: already exists, skipping", flush=True)
            continue
        t0 = _time.time()
        chunk_games = train_games[g_start:g_end]
        n_chunk = g_end - g_start
        logits, _ = _build_chunk_logits(
            chunk_games, mlp_even, mlp_odd, device, pos_start, pos_end)
        # logits are position-major: (length * n_chunk, 192)
        board_state = _logits_to_board_state(logits)  # (length * n_chunk, 64)
        # Reshape to position-major (length, n_chunk, 64) then to game-major (n_chunk, length, 64)
        board_state = board_state.reshape(length, n_chunk, 64).transpose(1, 0, 2)
        np.savez_compressed(out_path, board_state=board_state)
        elapsed = _time.time() - t0
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Chunk {ci}: {g_end - g_start} games, "
              f"{size_mb:.0f}MB, {elapsed:.0f}s", flush=True)
        del logits, board_state

    print(f"\nDone. Board state cache saved to {cache_dir}")


def train_policy(args):
    """Train a policy head on real Othello games.

    For each position t, the input is the board MLP's output (192-d)
    and the target is a binary vector of all legal moves (not just the one played).
    Trained with binary cross-entropy.

    For large datasets, processes games in chunks to limit memory usage.
    """
    device = get_device()
    print(f"Device: {device}")

    # Load the trained board-state MLP
    ckpt = torch.load(args.mlp_checkpoint, map_location='cpu')
    input_dim = ckpt['input_dim']
    hidden_dim = ckpt['hidden_dim']
    num_hidden = ckpt.get('num_hidden_layers', 1)

    mlp_even = _build_mlp(input_dim, hidden_dim, 64 * OPTIONS, num_hidden).to(device)
    mlp_odd = _build_mlp(input_dim, hidden_dim, 64 * OPTIONS, num_hidden).to(device)
    mlp_even.load_state_dict(ckpt['even'])
    mlp_odd.load_state_dict(ckpt['odd'])
    mlp_even.eval()
    mlp_odd.eval()
    print(f"Loaded board MLP: input={input_dim}, hidden={hidden_dim}")

    # Load games
    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    print(f"Loaded {len(games)} games")

    n_eval = min(max(int(len(games) * 0.1), 500), 50000)
    train_games = games[:len(games) - n_eval]
    eval_games = games[len(games) - n_eval:]

    # Chunk size for streaming (games per chunk)
    game_chunk_size = min(500000, len(train_games))
    n_train_chunks = (len(train_games) + game_chunk_size - 1) // game_chunk_size
    pos_start, pos_end = POS_START, POS_END - 1
    length = pos_end - pos_start

    print(f"Training: {len(train_games)} games in {n_train_chunks} chunks "
          f"of {game_chunk_size}")

    n_train = len(train_games)  # offset for eval games in original ordering

    # Check for precomputed board state cache
    board_cache_dir = os.path.join(args.output_dir, "board_state_cache")
    use_cache = os.path.exists(os.path.join(board_cache_dir, "board_state_chunk_0000.npz"))
    if use_cache:
        print(f"Using precomputed board states from {board_cache_dir}")
    else:
        print("No board state cache found, will compute logits on the fly")

    def _get_chunk_input(chunk_id, game_chunk):
        """Get policy input (N, 192) for a chunk — from cache or MLP."""
        if use_cache:
            onehot = _load_board_state_chunk(board_cache_dir, chunk_id, length)
            if onehot is not None:
                return onehot
        logits, _ = _build_chunk_logits(
            game_chunk, mlp_even, mlp_odd, device, pos_start, pos_end)
        return logits

    # --- Precompute eval data (small, fits in memory) ---
    print("Building eval data...", flush=True)
    eval_cache_path = os.path.join(board_cache_dir, "board_state_eval.npz")
    if use_cache and os.path.exists(eval_cache_path):
        bs = np.load(eval_cache_path)['board_state']  # (n_eval, length, 64)
        bs = bs.transpose(1, 0, 2).reshape(-1, 64)     # position-major
        ev_input = _board_state_to_onehot(bs)
        del bs
    else:
        ev_input, _ = _build_chunk_logits(
            eval_games, mlp_even, mlp_odd, device, pos_start, pos_end)
    ev_targets = _get_legal_moves(eval_games, pos_start, pos_end,
                                   chunk_dir=args.output_dir,
                                   game_offset=n_train)
    print(f"  Eval: {len(ev_input)} samples")

    # --- Estimate pos_weight from first chunk ---
    print("Estimating class balance from first chunk...", flush=True)
    first_legal = _get_legal_moves(train_games[:game_chunk_size], pos_start, pos_end,
                                    chunk_dir=args.output_dir,
                                    game_offset=0)
    n_pos = first_legal.sum().item()
    n_neg = first_legal.numel() - n_pos
    pw = n_neg / max(n_pos, 1)
    print(f"  Class balance: {n_pos/first_legal.numel():.1%} positive, pos_weight={pw:.1f}")
    pos_weight = torch.tensor([pw], device=device)
    del first_legal

    # --- Train ---
    policy = PolicyHead(hidden_dim=args.policy_hidden).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2)

    best_f1 = 0.0
    best_state = None
    batch_size = 2048

    import time as _time
    for epoch in range(1, args.policy_epochs + 1):
        t0 = _time.time()
        policy.train()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle chunk order each epoch
        chunk_order = torch.randperm(n_train_chunks).tolist()
        for ci in chunk_order:
            g_start = ci * game_chunk_size
            g_end = min(g_start + game_chunk_size, n_train)
            chunk_games = train_games[g_start:g_end]
            ch_input = _get_chunk_input(ci, chunk_games)
            ch_targets = _get_legal_moves(chunk_games, pos_start, pos_end,
                                           chunk_dir=args.output_dir,
                                           game_offset=g_start)

            # Shuffle within chunk
            perm = torch.randperm(len(ch_input))
            for i in range(0, len(ch_input), batch_size):
                idx = perm[i:i + batch_size]
                x = ch_input[idx].to(device)
                y = ch_targets[idx].to(device)
                logits = policy(x)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            del ch_input, ch_targets  # free memory

        # Eval
        policy.eval()
        tp = fp = fn = total_correct = total_moves = 0
        eval_loss = 0.0
        n_eval_batches = 0
        with torch.no_grad():
            for i in range(0, len(ev_input), batch_size):
                x = ev_input[i:i + batch_size].to(device)
                y = ev_targets[i:i + batch_size].to(device)
                logits = policy(x)
                eval_loss += nn.functional.binary_cross_entropy_with_logits(logits, y).item()
                n_eval_batches += 1
                preds = (logits > 0).float()
                total_correct += (preds == y).sum().item()
                total_moves += y.numel()
                tp += ((preds == 1) & (y == 1)).sum().item()
                fp += ((preds == 1) & (y == 0)).sum().item()
                fn += ((preds == 0) & (y == 1)).sum().item()

        acc = total_correct / total_moves
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mean_loss = eval_loss / n_eval_batches
        scheduler.step(mean_loss)
        lr = optimizer.param_groups[0]['lr']
        elapsed = _time.time() - t0
        print(f"  Epoch {epoch}: loss={mean_loss:.4f}  acc={acc:.4%}  "
              f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}  lr={lr:.2e}  "
              f"({elapsed:.0f}s)", flush=True)

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

    # Save
    save_path = os.path.join(args.output_dir, f"policy_head_H{args.policy_hidden}.pt")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({
        'state_dict': best_state,
        'hidden_dim': args.policy_hidden,
        'best_f1': best_f1,
    }, save_path)
    print(f"\nBest policy F1: {best_f1:.4f}")
    print(f"Saved to {save_path}")


# ---------------------------------------------------------------------------
# Perturb heuristic MLP hidden units
# ---------------------------------------------------------------------------

def _perturb_mlp(mlp, alpha, rng):
    """Smoothly interpolate MLP weights toward random weights.

    W_perturbed = (1 - alpha) * W_original + alpha * W_random

    All weights are perturbed proportionally. At alpha=0 the MLP is unchanged,
    at alpha=1 all weights are fully random.

    Args:
        mlp: nn.Sequential (Linear → ReLU → Linear)
        alpha: float in [0, 1], interpolation toward random
        rng: numpy random generator for reproducibility
    Returns:
        new MLP with perturbed weights (original unchanged)
    """
    mlp_new = deepcopy(mlp)
    if alpha == 0:
        return mlp_new

    for param in mlp_new.parameters():
        rand_w = torch.randn_like(param.data) * param.data.std().item()
        param.data = (1 - alpha) * param.data + alpha * rand_w

    return mlp_new


# ---------------------------------------------------------------------------
# Generate games using perturbed MLP + policy
# ---------------------------------------------------------------------------

def generate_games(args):
    """Generate games using perturbed heuristic MLP + trained policy head."""
    device = get_device()
    print(f"Device: {device}")
    print(f"Alpha (perturbation): {args.alpha}")
    print(f"Temperature: {args.temperature}")

    # Load board MLP
    ckpt = torch.load(args.mlp_checkpoint, map_location='cpu')
    input_dim = ckpt['input_dim']
    hidden_dim = ckpt['hidden_dim']
    num_hidden = ckpt.get('num_hidden_layers', 1)

    mlp_even = _build_mlp(input_dim, hidden_dim, 64 * OPTIONS, num_hidden)
    mlp_odd = _build_mlp(input_dim, hidden_dim, 64 * OPTIONS, num_hidden)
    mlp_even.load_state_dict(ckpt['even'])
    mlp_odd.load_state_dict(ckpt['odd'])

    # Perturb
    rng = np.random.default_rng(args.seed)
    if args.alpha > 0:
        print(f"Interpolating weights toward random (alpha={args.alpha:.2f})...")
        mlp_even = _perturb_mlp(mlp_even, args.alpha, rng)
        mlp_odd = _perturb_mlp(mlp_odd, args.alpha, rng)

    mlp_even = mlp_even.to(device).eval()
    mlp_odd = mlp_odd.to(device).eval()

    # Load policy head
    policy_path = os.path.join(args.output_dir, f"policy_head_H{args.policy_hidden}.pt")
    if not os.path.exists(policy_path):
        # Fallback to old naming convention
        policy_path = os.path.join(args.output_dir, "policy_head.pt")
    policy_ckpt = torch.load(policy_path, map_location='cpu')
    policy = PolicyHead(hidden_dim=policy_ckpt['hidden_dim']).to(device)
    policy.load_state_dict(policy_ckpt['state_dict'])
    policy.eval()
    print(f"Loaded policy head (F1={policy_ckpt.get('best_f1', policy_ckpt.get('best_acc', 0)):.4f})")

    # Generate games
    games = []
    n_short = 0  # games that ended early (no predicted legal move available)

    for gi in range(args.n_games):
        move_history = []
        for t in range(60):
            # Compute features for current position
            if t == 0:
                features = torch.zeros(180, dtype=torch.float32)
            else:
                features = _compute_move_features(move_history, t - 1)

            features = features.unsqueeze(0).to(device)  # (1, 180)

            with torch.no_grad():
                # Get board state from heuristic MLP
                if t % 2 == 0:
                    board_logits = mlp_even(features)
                else:
                    board_logits = mlp_odd(features)

                # Get legal move predictions from policy
                move_logits = policy(board_logits)  # (1, 60)

            # Mask already-played moves
            for prev_move in move_history:
                move_logits[0, _MOVE_TO_IDX[prev_move]] = -1e9

            # Predicted legal moves: sigmoid > 0.5 (i.e. logits > 0)
            legal_mask = (move_logits[0] > 0)

            # If no moves predicted legal, fall back to highest-scoring unplayed
            if not legal_mask.any():
                move_idx = move_logits[0].argmax().item()
            else:
                # Sample uniformly from predicted-legal moves
                legal_indices = legal_mask.nonzero(as_tuple=True)[0]
                pick = torch.randint(len(legal_indices), (1,)).item()
                move_idx = legal_indices[pick].item()

            move = _VALID_MOVES[move_idx]
            move_history.append(move)

        if len(move_history) == 60:
            games.append(move_history)
        else:
            n_short += 1

        if (gi + 1) % 10000 == 0:
            print(f"  Generated {gi + 1}/{args.n_games} games", flush=True)

    print(f"\nGenerated {len(games)} complete games ({n_short} short games discarded)")

    # Validate: check what fraction of moves are legal in real Othello
    n_legal = 0
    n_total = 0
    n_games_check = min(1000, len(games))
    for game in games[:n_games_check]:
        board = OthelloBoardState()
        for t, move in enumerate(game):
            valid = board.get_valid_moves()
            if move in valid:
                n_legal += 1
            n_total += 1
            # We can't call umpire on an illegal move, so just track stats
            # For the check, we simulate real Othello to see legality
            if move in valid:
                board.umpire(move)
            else:
                break  # can't continue real simulation after illegal move

    print(f"Legality check (first {n_games_check} games):")
    print(f"  Moves that are also legal in real Othello: {n_legal}/{n_total} "
          f"({n_legal/n_total:.1%})")

    # Save as pickle (same format as training data)
    alpha_str = f"{args.alpha:.2f}".replace('.', '')
    out_path = os.path.join(args.output_dir,
                            f"adversarial_games_alpha{alpha_str}_n{len(games)}.pickle")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(games, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to {out_path}")

    return games


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Generate adversarial Othello games")
parser.add_argument("--train-policy", action="store_true",
                    help="Train the policy head")
parser.add_argument("--generate", action="store_true",
                    help="Generate games with perturbed heuristics")
parser.add_argument("--precompute-logits", action="store_true",
                    help="Precompute board states from MLP (run once, shared by sweep jobs)")

# Shared
parser.add_argument("--mlp-checkpoint",
                    default="experiments/mathematical_transformation_experiments/"
                            "heuristic_probe_results/mlp_checkpoints/"
                            "mlp_all_H1024_streaming.pt",
                    help="Path to trained board-state MLP checkpoint")
parser.add_argument("--output-dir",
                    default="experiments/mathematical_transformation_experiments/"
                            "heuristic_probe_results/adversarial")

# Policy training
parser.add_argument("--max-games", type=int, default=100000)
parser.add_argument("--max-files", type=int, default=None)
parser.add_argument("--policy-epochs", type=int, default=20)
parser.add_argument("--policy-hidden", type=int, default=256,
                    help="Hidden dim of policy head")

# Generation
parser.add_argument("--alpha", type=float, default=0.0,
                    help="Perturbation level [0, 1]")
parser.add_argument("--n-games", type=int, default=100000,
                    help="Number of games to generate")
parser.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature for move selection")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

if args.precompute_logits:
    precompute_board_states(args)
elif args.train_policy:
    train_policy(args)
elif args.generate:
    generate_games(args)
else:
    print("Specify --precompute-logits, --train-policy, or --generate")
