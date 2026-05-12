"""Precompute cumulative rule-firings as a new feature format, alongside
the existing 60/120/180-d move-history features.

For each game, replay through OthelloBoardState and at each turn record:
  - cumulative rule-firings up to and including that turn (960-d uint8)
  - board state at that turn (64-d uint8, 0=empty, 1=white, 2=black)
  - turn number (uint8)

Saves output as `rule_firings_NNNN.npz` files alongside the existing chunks.
Each file holds positions from a contiguous range of pickle files.

Usage:
    python precompute_rule_firings.py --start-file 0 --end-file 5 --chunk-id 0
    # Process pickles 0..4 inclusive, write rule_firings_0000.npz
"""
import sys, os, argparse, time, pickle
sys.path.insert(0, '.')

import numpy as np
from multiprocessing import Pool

from data.othello import OthelloBoardState
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import compute_pattern_labels_batch

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import SYNTHETIC_DIR, GAME_LEN


CENTER_64 = {27, 28, 35, 36}
NON_CENTER_64 = [c for c in range(64) if c not in CENTER_64]
B64_TO_M60 = {c: i for i, c in enumerate(NON_CENTER_64)}


# Pattern targets in 60-cell MOVE_TO_IDX space, built once at module level.
_patterns = enumerate_flanking_patterns()
_pat_target_60 = np.array([MOVE_TO_IDX[p['target']] for p in _patterns],
                           dtype=np.int64)
_pat_arrays = precompute_pattern_arrays(_patterns)
_pat_targets, _pat_terminals, _pat_opp_cells, _pat_opp_mask = _pat_arrays


def encode_state_abs(state_flat):
    """Convert flat 64-cell state (-1/0/1) -> classes (0=empty, 1=white, 2=black)."""
    cls = np.zeros(64, dtype=np.uint8)
    cls[state_flat == 0]  = 0
    cls[state_flat == -1] = 1
    cls[state_flat == 1]  = 2
    return cls


def encode_state_mineview(state_flat, turn):
    """Convert flat state -> player-relative (0=empty, 1=mine, 2=theirs).
    'mine' = whoever plays move `turn` (black if turn even, white if odd)."""
    mine_color = 1 if turn % 2 == 0 else -1
    cls = np.zeros(64, dtype=np.uint8)
    cls[state_flat == 0]  = 0
    cls[state_flat == mine_color] = 1
    cls[state_flat == -mine_color] = 2
    return cls


def process_one_game(game):
    """Replay one game, return (cum_firings (60,960), states_abs (60,64), n_turns)."""
    T = len(game)
    cum_firings = np.zeros(960, dtype=np.uint16)
    cum_seq = np.zeros((T, 960), dtype=np.uint16)
    states_abs_seq = np.zeros((T, 64), dtype=np.uint8)

    board = OthelloBoardState()
    # Compute pre-move board states for each turn in one batch call to
    # compute_pattern_labels_batch (vectorized).
    pre_states = np.zeros((T, 64), dtype=np.int64)
    for t in range(T):
        pre_states[t] = np.asarray(board.state, dtype=np.int64).flatten()
        try:
            board.umpire(game[t])
        except Exception:
            break
        states_abs_seq[t] = encode_state_abs(
            np.asarray(board.state, dtype=np.int64).flatten())

    # Convert pre_states to mineview per turn
    pre_mineview = np.stack([
        encode_state_mineview(pre_states[t], t).astype(np.int64)
        for t in range(T)
    ])
    pos_arr = np.arange(T, dtype=np.int64)
    # compute_pattern_labels_batch wants integer labels
    try:
        labels = compute_pattern_labels_batch(
            pre_mineview, pos_arr,
            _pat_targets, _pat_terminals, _pat_opp_cells, _pat_opp_mask
        ).astype(np.int64)   # (T, 960)
    except Exception as e:
        return None

    # For each turn, the FIRED patterns are those with P.target == move AND label==1
    for t in range(T):
        m64 = game[t]
        m60 = B64_TO_M60.get(m64, -1)
        if m60 < 0:
            cum_seq[t] = cum_firings.copy()
            continue
        fired_mask = (_pat_target_60 == m60) & (labels[t] > 0)
        cum_firings[fired_mask] += 1
        cum_seq[t] = cum_firings

    return cum_seq, states_abs_seq, T


def process_games_worker(games_batch):
    """Worker: process a batch of games, return concatenated arrays."""
    all_cum = []
    all_states = []
    all_turns = []
    for game in games_batch:
        res = process_one_game(game)
        if res is None:
            continue
        cum_seq, states_seq, T = res
        # Keep turns 5..53 (matching existing chunk range)
        keep = list(range(5, min(54, T)))
        all_cum.append(cum_seq[keep].astype(np.uint8))
        all_states.append(states_seq[keep])
        all_turns.append(np.array(keep, dtype=np.uint8))
    if not all_cum:
        return None
    return (np.concatenate(all_cum, axis=0),
            np.concatenate(all_states, axis=0),
            np.concatenate(all_turns, axis=0))


def load_games_range(start_file, end_file):
    """Load games from pickle files [start_file, end_file)."""
    files = sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pickle"))
    files = files[start_file:end_file]
    games = []
    for fname in files:
        with open(os.path.join(SYNTHETIC_DIR, fname), "rb") as f:
            batch = pickle.load(f)
        games.extend(g for g in batch if len(g) == GAME_LEN)
    return games


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start-file", type=int, required=True)
    p.add_argument("--end-file", type=int, required=True)
    p.add_argument("--chunk-id", type=int, required=True,
                   help="Output chunk index, used to name output file.")
    p.add_argument("--output-dir",
                   default="experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2000,
                   help="Games per worker batch.")
    args = p.parse_args()

    t0 = time.time()
    print(f"Loading games from files [{args.start_file}, {args.end_file})...")
    games = load_games_range(args.start_file, args.end_file)
    print(f"  loaded {len(games)} games in {time.time()-t0:.0f}s")

    # Split into worker batches
    batches = [games[i:i + args.batch_size]
               for i in range(0, len(games), args.batch_size)]
    print(f"Processing {len(batches)} batches with {args.workers} workers...")
    t1 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(process_games_worker, batches)
    print(f"  replays done in {time.time()-t1:.0f}s")

    # Concatenate
    results = [r for r in results if r is not None]
    cum_all = np.concatenate([r[0] for r in results], axis=0)
    states_all = np.concatenate([r[1] for r in results], axis=0)
    turns_all = np.concatenate([r[2] for r in results], axis=0)

    print(f"Total positions: {len(cum_all)}")
    print(f"  cum_firings shape: {cum_all.shape}  dtype: {cum_all.dtype}")
    print(f"  states shape:      {states_all.shape}  dtype: {states_all.dtype}")
    print(f"  turns shape:       {turns_all.shape}  dtype: {turns_all.dtype}")

    out_path = os.path.join(args.output_dir,
                            f"rule_firings_{args.chunk_id:04d}.npz")
    print(f"Saving to {out_path} (compressed)...")
    np.savez_compressed(out_path,
                        features=cum_all,
                        labels=states_all,
                        positions=turns_all)
    print(f"Done. Total time: {time.time()-t0:.0f}s")
