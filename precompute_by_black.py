"""Precompute a forfeit-correct 'played-by-black' channel for each feature chunk.

The standard +even feature encodes (played_turn % 2 == 0), which only matches
the actual playing color when the strict B-W-B-W alternation holds. Forfeits
break this -- ~1% of positions in the synthetic training data are after a
forfeit and account for ~35% of MLP top-1 errors.

For each chunk, we:
  1. Identify game boundaries (pos resets to 0)
  2. Per game, reconstruct the move sequence from X's 'when' channel
  3. Replay through OthelloBoardState; at each move, record whether the
     player who actually played that cell was black
  4. For every chunk position belonging to that game, save the snapshot
     of 'played-by-black' as of that turn

Output: alongside each chunk file (chunk_NNNN.npz), writes chunk_NNNN_by_black.npy
with shape (N_chunk, 60), values in {0.0, 1.0}.

Usage:
    python precompute_by_black.py
    python precompute_by_black.py --chunk-glob "chunk_0039.npz"   # one file
"""
import sys, os, argparse, glob, time
sys.path.insert(0, '.')

import numpy as np
from data.othello import OthelloBoardState

N_MOVES = 60
CENTER_64 = {27, 28, 35, 36}
_movable_64 = [c for c in range(64) if c not in CENTER_64]
_b64_to_m60 = {c64: i for i, c64 in enumerate(_movable_64)}
_m60_to_b64 = {i: c64 for c64, i in _b64_to_m60.items()}


def replay_game_and_collect_by_black(when_at_last_pos, last_pos):
    """Given the 'when' channel of the LAST chunk position of a game (which has
    the full move sequence up to last_pos), reconstruct moves and replay.

    Returns: list of length last_pos+1, where entry t is the 60-d
    played-by-black indicator AT END OF TURN t (i.e. after move t has been
    played). Entry 0 is the state after move 0.
    """
    # Recover move sequence: for each cell with when > 0, the move turn is
    # round(when * 60) - 1 (since when[c] = (turn + 1) / 60).
    moves = []
    for c in range(60):
        w = when_at_last_pos[c]
        if w > 0:
            t = int(round(float(w) * N_MOVES)) - 1
            if 0 <= t < N_MOVES:
                moves.append((t, c))
    moves.sort()

    snapshots = []
    by_black = np.zeros(60, dtype=np.float32)
    b = OthelloBoardState()
    for t, c60 in moves:
        c64 = _m60_to_b64[c60]
        try:
            actual_color = b.next_hand_color   # +1 black, -1 white
            b.umpire(c64)
        except Exception:
            break
        if actual_color == 1:
            by_black[c60] = 1.0
        snapshots.append(by_black.copy())
    return snapshots


def process_chunk(chunk_path):
    print(f"Processing {chunk_path}")
    data = np.load(chunk_path)
    X = data['X']      # (N, 180+) features
    pos = data['pos']  # (N,) turn numbers
    N = X.shape[0]
    when_ch = X[:, N_MOVES:2 * N_MOVES]   # columns 60..120

    by_black = np.zeros((N, 60), dtype=np.float32)

    # Identify game boundaries: pos[i] < pos[i-1] means new game starts at i.
    game_starts = [0]
    for i in range(1, N):
        if pos[i] < pos[i - 1]:
            game_starts.append(i)
    game_starts.append(N)
    print(f"  {N} positions, {len(game_starts) - 1} games")

    n_processed = 0
    t0 = time.time()
    for g in range(len(game_starts) - 1):
        start, end = game_starts[g], game_starts[g + 1]
        # The chunk position at the END of the game has the most complete 'when'.
        last_pos = int(pos[end - 1])
        snapshots = replay_game_and_collect_by_black(when_ch[end - 1], last_pos)
        # Assign per chunk position: at position with turn p, by_black is the
        # snapshot AFTER move p-1 (zero if p == 0).
        for i in range(start, end):
            p = int(pos[i])
            if p > 0 and p - 1 < len(snapshots):
                by_black[i] = snapshots[p - 1]
        n_processed += 1
        if n_processed % 5000 == 0:
            dt = time.time() - t0
            rate = n_processed / max(dt, 1e-3)
            print(f"  {n_processed} games done, {rate:.0f} games/sec",
                  flush=True)

    out_path = chunk_path.replace('.npz', '_by_black.npy')
    np.save(out_path, by_black)
    print(f"  Saved {out_path}  shape={by_black.shape}  "
          f"mean(by_black)={by_black.mean():.4f}  (compare to ~0.5)")

    # Diagnostic: how often does by_black DIFFER from the +even feature?
    even_ch = X[:, 2 * N_MOVES:3 * N_MOVES]
    # For played cells, even=1 iff (played_turn even). By default (no forfeit),
    # even=1 iff played by black. So diff = cells where by_black != even.
    played_mask = (when_ch > 0)
    diff_mask = (by_black != even_ch) & played_mask
    per_position_diff = diff_mask.any(axis=1)
    print(f"  Positions with >=1 forfeit-affected cell: "
          f"{per_position_diff.sum()} / {N}  ({per_position_diff.mean():.2%})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-dir",
                   default="experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks")
    p.add_argument("--chunk-glob", default="chunk_*.npz",
                   help="Filename pattern within --chunk-dir.")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.chunk_dir, args.chunk_glob)))
    files = [f for f in files if "_patterns" not in f
                              and "_when60" not in f
                              and "_by_black" not in f]
    print(f"Found {len(files)} chunks to process")
    for f in files:
        process_chunk(f)
