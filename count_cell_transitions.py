"""For each of the 64 cells, count the average number of times the cell
changes category (empty / white / black) across a game.

A "transition" is any time the cell value changes between consecutive
turns, including:
  - placement (empty -> color)
  - flip (white <-> black)

The initial state (turn 0) has center cells filled (d4=W, e4=B, d5=B, e5=W).
Transitions are counted starting from that initial state.

Usage:
    python count_cell_transitions.py --n-games 10000
"""
import sys, os, argparse, time
sys.path.insert(0, '.')

import numpy as np
from data.othello import OthelloBoardState

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import load_games, GAME_LEN


CENTER = {27, 28, 35, 36}


def cell_region(c):
    if c in CENTER: return 'center'
    r, col = c // 8, c % 8
    if r in (0, 7) and col in (0, 7): return 'corner'
    if r in (0, 7) or col in (0, 7):  return 'edge'
    return 'inner'


def state_to_class(state_flat):
    """OthelloBoardState convention: 0=empty, 1=black, -1=white.
    Map to int classes 0=empty, 1=black, 2=white (any consistent mapping
    works for transition counting)."""
    cls = np.zeros(64, dtype=np.int8)
    cls[state_flat == 1]  = 1
    cls[state_flat == -1] = 2
    return cls


def transitions_one_game(game):
    """Return (64,) int array of transition counts for one game."""
    board = OthelloBoardState()
    prev = state_to_class(np.asarray(board.state, dtype=np.int8).flatten())
    counts = np.zeros(64, dtype=np.int32)
    for m in game:
        try:
            board.umpire(m)
        except Exception:
            break
        cur = state_to_class(np.asarray(board.state, dtype=np.int8).flatten())
        counts += (cur != prev).astype(np.int32)
        prev = cur
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=10000)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--output",
        default="experiments/mathematical_transformation_experiments/heuristic_probe_results/cell_transitions.npz")
    args = parser.parse_args()

    print(f"Loading up to {args.n_games} games from {args.max_files} files...")
    t0 = time.time()
    games = load_games(max_files=args.max_files)
    games = [g for g in games if len(g) == GAME_LEN]
    if len(games) > args.n_games:
        games = games[:args.n_games]
    print(f"  Loaded {len(games)} games in {time.time()-t0:.0f}s")

    t1 = time.time()
    total = np.zeros(64, dtype=np.int64)
    for i, g in enumerate(games):
        total += transitions_one_game(g)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(games)} games processed, "
                  f"mean transitions so far = "
                  f"{total.sum() / (64*(i+1)):.2f}", flush=True)
    mean_per_cell = total / len(games)
    print(f"  Replays done in {time.time()-t1:.0f}s")

    # Per-region averages
    print(f"\nPer-region average transitions per cell per game:")
    for label in ('corner', 'edge', 'inner', 'center'):
        cells = [c for c in range(64) if cell_region(c) == label]
        m = np.mean([mean_per_cell[c] for c in cells])
        mn = min(mean_per_cell[c] for c in cells)
        mx = max(mean_per_cell[c] for c in cells)
        print(f"  {label:>6s}: n={len(cells):2d}  "
              f"mean={m:.2f}  min={mn:.2f}  max={mx:.2f}")

    # Highest / lowest cells
    print(f"\nTop 10 most-flipped cells:")
    order = np.argsort(-mean_per_cell)
    for c in order[:10]:
        r, col = c // 8, c % 8
        print(f"  cell {c:2d} (r={r}, c={col}, {cell_region(c):>6s}): "
              f"{mean_per_cell[c]:.2f} transitions/game")
    print(f"\nBottom 10 (least-flipped) cells:")
    for c in order[-10:][::-1]:
        r, col = c // 8, c % 8
        print(f"  cell {c:2d} (r={r}, c={col}, {cell_region(c):>6s}): "
              f"{mean_per_cell[c]:.2f} transitions/game")

    # ASCII 8x8 grid
    print(f"\nMean transitions per game (8x8 grid):")
    grid = mean_per_cell.reshape(8, 8)
    print("     " + " ".join(f"{c:5d}" for c in range(8)))
    for r in range(8):
        row = " ".join(f"{grid[r, c]:5.2f}" for c in range(8))
        print(f"  {r}  {row}")

    np.savez(args.output,
             mean_per_cell=mean_per_cell,
             total_per_cell=total,
             n_games=len(games))
    print(f"\nSaved {args.output}")
