"""Precompute per-turn fired patterns for next-rule prediction.

For each (game, turn t), records:
  features (180-d move history at state before turn t)
  fired_mask (960-d binary, patterns that fire on move t)
  is_forfeit (bool, True if turn t has no valid move — center cell etc.)
  position (turn t, 5..53)

Output: fired_patterns_NNNN.npz per chunk-id.

Usage:
    sbatch --array=0-39 run_precompute_fired_patterns.sh
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

_patterns = enumerate_flanking_patterns()
_pat_target_60 = np.array([MOVE_TO_IDX[p['target']] for p in _patterns],
                           dtype=np.int64)
_pat_arrays = precompute_pattern_arrays(_patterns)
_pat_targets, _pat_terminals, _pat_opp_cells, _pat_opp_mask = _pat_arrays


def encode_state_mineview(state_flat, turn):
    mine_color = 1 if turn % 2 == 0 else -1
    cls = np.zeros(64, dtype=np.uint8)
    cls[state_flat == 0]  = 0
    cls[state_flat == mine_color] = 1
    cls[state_flat == -mine_color] = 2
    return cls


def process_one_game(game):
    """Return (features (T,180), fired (T,960), is_forfeit (T,), T)."""
    T = len(game)
    feats = np.zeros((T, 180), dtype=np.float32)
    fired = np.zeros((T, 960), dtype=np.uint8)
    is_forfeit = np.zeros(T, dtype=np.uint8)

    # Build features (move history per turn)
    step_of_move = np.full(60, -1, dtype=np.int32)
    for t in range(T):
        # features at turn t = state through moves 0..t-1
        played = (step_of_move >= 0) & (step_of_move <= t - 1)
        when = np.where(played, (step_of_move + 1) / 60.0, 0.0)
        even = np.where(played, (step_of_move % 2 == 0).astype(np.float32), 0.0)
        feats[t, :60] = played.astype(np.float32)
        feats[t, 60:120] = when
        feats[t, 120:180] = even
        # Update step_of_move for next turn
        m64 = game[t]
        m60 = B64_TO_M60.get(m64, -1)
        if m60 >= 0:
            step_of_move[m60] = t

    # Replay to get pre-move board states
    board = OthelloBoardState()
    pre_states = np.zeros((T, 64), dtype=np.int64)
    for t in range(T):
        pre_states[t] = np.asarray(board.state, dtype=np.int64).flatten()
        try:
            board.umpire(game[t])
        except Exception:
            break

    # Compute pattern availability at each pre-move state
    pre_mineview = np.stack([
        encode_state_mineview(pre_states[t], t).astype(np.int64)
        for t in range(T)
    ])
    pos_arr = np.arange(T, dtype=np.int64)
    try:
        labels = compute_pattern_labels_batch(
            pre_mineview, pos_arr,
            _pat_targets, _pat_terminals, _pat_opp_cells, _pat_opp_mask
        ).astype(np.int64)  # (T, 960)
    except Exception:
        return None

    # Per-turn fired mask: patterns where target == move AND pattern fires
    for t in range(T):
        m64 = game[t]
        m60 = B64_TO_M60.get(m64, -1)
        if m60 < 0:
            is_forfeit[t] = 1
            continue
        mask = (_pat_target_60 == m60) & (labels[t] > 0)
        fired[t] = mask.astype(np.uint8)

    return feats, fired, is_forfeit, T


def process_games_worker(games_batch):
    all_feats, all_fired, all_forfeit, all_turns = [], [], [], []
    for game in games_batch:
        res = process_one_game(game)
        if res is None:
            continue
        feats, fired, is_forfeit, T = res
        keep = list(range(5, min(54, T)))
        all_feats.append(feats[keep])
        all_fired.append(fired[keep])
        all_forfeit.append(is_forfeit[keep])
        all_turns.append(np.array(keep, dtype=np.uint8))
    if not all_feats:
        return None
    return (np.concatenate(all_feats, axis=0),
            np.concatenate(all_fired, axis=0),
            np.concatenate(all_forfeit, axis=0),
            np.concatenate(all_turns, axis=0))


def load_games_range(start_file, end_file):
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
    p.add_argument("--chunk-id", type=int, required=True)
    p.add_argument("--output-dir",
        default="experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2000)
    args = p.parse_args()

    t0 = time.time()
    print(f"Loading games from files [{args.start_file}, {args.end_file})...")
    games = load_games_range(args.start_file, args.end_file)
    print(f"  loaded {len(games)} games in {time.time()-t0:.0f}s")

    batches = [games[i:i + args.batch_size]
               for i in range(0, len(games), args.batch_size)]
    print(f"Processing {len(batches)} batches with {args.workers} workers...")
    t1 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(process_games_worker, batches)
    print(f"  replays done in {time.time()-t1:.0f}s")

    results = [r for r in results if r is not None]
    feats_all = np.concatenate([r[0] for r in results], axis=0)
    fired_all = np.concatenate([r[1] for r in results], axis=0)
    forfeit_all = np.concatenate([r[2] for r in results], axis=0)
    turns_all = np.concatenate([r[3] for r in results], axis=0)

    print(f"Total positions: {len(feats_all)}")
    print(f"  features: {feats_all.shape}  dtype={feats_all.dtype}")
    print(f"  fired:    {fired_all.shape}  dtype={fired_all.dtype}  "
          f"mean fires/move = {fired_all.sum() / len(fired_all):.2f}")
    print(f"  forfeit rate: {forfeit_all.mean()*100:.2f}%")
    print(f"  turns: unique = {sorted(np.unique(turns_all))[:5]}...")

    out_path = os.path.join(args.output_dir,
                            f"fired_patterns_{args.chunk_id:04d}.npz")
    print(f"Saving to {out_path}...")
    np.savez_compressed(out_path,
                        features=feats_all.astype(np.float16),
                        fired=fired_all,
                        is_forfeit=forfeit_all,
                        positions=turns_all)
    print(f"Done. Total time: {time.time()-t0:.0f}s")
