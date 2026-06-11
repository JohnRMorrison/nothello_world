"""
Precompute played+even features and board-state labels for every position
in a set of Othello games, then save to a .npz chunk file.

Each chunk stores:
  features  (float16, 120-d): played[60] || even[60]
  labels    (int8,    64-d):  board state per cell --> 0=empty, 1=white, 2=black
  positions (int8):           turn number (0-58)

Run one chunk locally:
  python precompute_chunks_clean.py --start-file 0 --end-file 6 --chunk-id 0

Run all 40 chunks in parallel on the cluster:
  sbatch --array=0-39%20 run_precompute_chunks_clean.sh
"""
import sys, os, argparse, pickle, time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, '.')
import numpy as np

sys.path.insert(0, 'experiments/mathematical_transformation_experiments')
from probe_state_pred_for_othello import SYNTHETIC_DIR, GAME_LEN
from probe_variant_boards import seq_to_state_normal

CENTER_CELLS = {27, 28, 35, 36}
VALID_MOVES  = [i for i in range(64) if i not in CENTER_CELLS]
MOVE_TO_IDX  = {cell: idx for idx, cell in enumerate(VALID_MOVES)}
N_MOVES = 60


def _label_games(args):
    # computes board state labels for a sub batch of games.
    # seq_to_state_normal returns the 8x8 board after each move as -1 = white 0 = empty 1 = black
    # remap to 0=empty, 1=white, 2=black so all values are not negative.
    
    games_chunk, pos_start, pos_end = args
    n = len(games_chunk)
    length = pos_end - pos_start # turns covered
    out = np.zeros((n, length, 64), dtype=np.int8)

    for gi, game in enumerate(games_chunk):
        states = seq_to_state_normal(game)
        for ti, t in enumerate(range(pos_start, pos_end)):
            flat = states[t].reshape(64)
            out[gi, ti] = np.where(flat == -1, 1, np.where(flat == 1, 2, 0))

    return out


def compute_board_labels(games, pos_start, pos_end):
    # runs _label_games in parallel across CPU cores.
    # returns (n_games * n_turns, 64) int8 and ordered so that all games for turn t
    # comes before all games for turn t+1 and matches the feature layout in build_chunk
    
    n_games = len(games)
    length = pos_end - pos_start
    n_workers = min(cpu_count(), 8)
    chunk_size = (n_games + n_workers - 1) // n_workers

    work = [(games[i:i + chunk_size], pos_start, pos_end)
            for i in range(0, n_games, chunk_size)]

    print(f"  Computing board labels: {n_workers} workers × ~{chunk_size} games", flush=True)
    with Pool(n_workers) as pool:
        parts = pool.map(_label_games, work)

    # each returns (chunk_n, length, 64)
    # row order is --> all games at turn 0, then all games at turn 1...
    labels = np.zeros((n_games * length, 64), dtype=np.int8)
    gi_off = 0
    for part in parts:
        cn = part.shape[0]
        for ti in range(length):
            labels[ti * n_games + gi_off: ti * n_games + gi_off + cn] = part[:, ti]
        gi_off += cn

    return labels


def build_chunk(games, pos_start, pos_end):
    # build features and labels for all games over turns [pos_start, pos_end).
    # features are 120-d: played[60] concat with even[60].
    #   played[i] = 1 if cell i has been played at or before this turn
    #   even[i]   = 1 if cell i was played on an even turn (white's move)

    n_games = len(games)
    length = pos_end - pos_start
    n_samples = n_games * length

    # each game records which turn each cell was played (-1 means not yet played)
    step_of_move = np.full((n_games, N_MOVES), -1, dtype=np.int32)
    for g, game in enumerate(games):
        for step, cell in enumerate(game):
            step_of_move[g, MOVE_TO_IDX[cell]] = step

    features  = np.zeros((n_samples, 2 * N_MOVES), dtype=np.float32)
    positions = np.zeros(n_samples, dtype=np.int32)

    for ti, t in enumerate(range(pos_start, pos_end)):
        rows = slice(ti * n_games, (ti + 1) * n_games)

        played = (step_of_move >= 0) & (step_of_move <= t)
        even   = np.where(played, (step_of_move % 2 == 0).astype(np.float32), 0.0)

        features[rows, :N_MOVES] = played
        features[rows, N_MOVES:] = even
        positions[rows] = t

    labels = compute_board_labels(games, pos_start, pos_end)

    return features.astype(np.float16), labels, positions.astype(np.int8)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start-file", type=int, required=True)
    p.add_argument("--end-file",   type=int, required=True)
    p.add_argument("--chunk-id",   type=int, required=True)
    p.add_argument("--pos-start",  type=int, default=0,
                   help="First turn to include (default 0)")
    p.add_argument("--pos-end",    type=int, default=59,
                   help="Exclusive upper bound on turns (default 59)")
    p.add_argument("--output-dir", default=
        "experiments/mathematical_transformation_experiments"
        "/heuristic_probe_results/feature_chunks")
    args = p.parse_args()

    t0 = time.time()
    all_files = sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pickle"))
    files = all_files[args.start_file:args.end_file]
    print(f"Loading {len(files)} pickle files: {files[0]} … {files[-1]}")

    games = []
    for fname in files:
        with open(os.path.join(SYNTHETIC_DIR, fname), "rb") as f:
            batch = pickle.load(f)
        games.extend(g for g in batch if len(g) == GAME_LEN)
    print(f"Loaded {len(games)} games in {time.time() - t0:.0f}s")

    features, labels, positions = build_chunk(games, args.pos_start, args.pos_end)
    print(f"  features:  {features.shape}  ({features.dtype})")
    print(f"  labels:    {labels.shape}  ({labels.dtype})")
    print(f"  positions: {positions.shape}  turns {positions.min()}–{positions.max()}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"chunk_{args.chunk_id:04d}.npz")
    np.savez_compressed(out_path, features=features, labels=labels, positions=positions)
    print(f"Saved {out_path}  ({time.time() - t0:.0f}s total)")
