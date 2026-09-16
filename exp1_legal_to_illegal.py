"""Experiment 1 - probability retained on a move that was legal and becomes
illegal (flank-loss).

For each decision point t (mover X, predicting move t+1) and each square c
that is LEGAL at t, we look at how the model treats c once it is no longer a
legal move for X.  Two horizons are recorded per event so the transition
definition is a plot-time choice:

  - "next"        : t -> t+1  (the next number in the input sequence)
  - "same_player" : t -> t+2  (the same player's next turn, assuming strict
                    alternation; forfeits are NOT tokenized so we do not try
                    to parity-correct)

A square "becomes illegal (flank-loss)" at a horizon if, at that later
position, it is still EMPTY but no longer legal.  Squares that became illegal
only by being occupied are still recorded (with the empty_* flags = False) so
they can be filtered out (default) or included.

Recorded per event (one row):
  game, t, square, turn_before(=t+1),
  P_before,
  P_next, empty_next, illegal_next,           # horizon t+1
  P_same, empty_same, illegal_same,           # horizon t+2
  mover                                        # +1 black / -1 white

The eventual bar graph buckets, e.g., retention = P_after / P_before (or the
raw residual P_after) into probability ranges; per-bucket counts + a binomial
CI come straight from these rows.

Usage (pod):
  source $(conda info --base)/etc/profile.d/conda.sh; conda activate othello
  python exp1_legal_to_illegal.py --n-games 40000 --max-files 20 \
      --target-events 100000 --out experiments/exp1_legal_to_illegal.npz
"""
import argparse
import os
import time

import numpy as np

from ogpt_behavioral_common import (
    pick_device, load_ogpt, iter_game_probs, replay_game, M2I,
    load_experiment_games,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    ap.add_argument("--n-games", type=int, default=40000)
    ap.add_argument("--max-files", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--target-events", type=int, default=100000,
                    help="Stop once this many same-player flank-loss events "
                         "have been collected (0 = use all games).")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/exp1_legal_to_illegal.npz")
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}", flush=True)
    model, block_size, cell_tokens = load_ogpt(args.ckpt, device)
    games = load_experiment_games(args.max_files, args.n_games,
                                  seed=args.seed, shuffle=args.shuffle)
    print(f"Loaded {len(games)} games", flush=True)

    # event columns
    col_game, col_t, col_sq, col_turn = [], [], [], []
    col_pbefore = []
    col_pnext, col_empty_next, col_illegal_next = [], [], []
    col_psame, col_empty_same, col_illegal_same = [], [], []
    col_mover = []

    n_flankloss_same = 0        # primary event count (t->t+2, empty & illegal)
    t0 = time.time()
    for gi, game, probs in iter_game_probs(model, games, block_size,
                                           cell_tokens, device, args.batch):
        states, legal, mover = replay_game(game)
        Tp = probs.shape[0]                          # predictable decisions (~59)
        for t in range(Tp):
            legal_now = np.where(legal[t])[0]        # board cells legal at t
            for c in legal_now:
                mi = M2I[c]
                p_before = float(probs[t, mi])
                # horizon t+1 (next input position)
                if t + 1 < Tp:
                    empty_next = bool(states[t + 1, c] == 0)
                    illegal_next = bool(not legal[t + 1, c])
                    p_next = float(probs[t + 1, mi])
                else:
                    empty_next = False; illegal_next = False; p_next = np.nan
                # horizon t+2 (same player's next turn)
                if t + 2 < Tp:
                    empty_same = bool(states[t + 2, c] == 0)
                    illegal_same = bool(not legal[t + 2, c])
                    p_same = float(probs[t + 2, mi])
                else:
                    empty_same = False; illegal_same = False; p_same = np.nan

                fl_next = empty_next and illegal_next
                fl_same = empty_same and illegal_same
                if not (illegal_next or illegal_same):
                    continue                          # never lost legality -> skip
                col_game.append(gi); col_t.append(t); col_sq.append(int(c))
                col_turn.append(t + 1)
                col_pbefore.append(p_before)
                col_pnext.append(p_next)
                col_empty_next.append(empty_next); col_illegal_next.append(illegal_next)
                col_psame.append(p_same)
                col_empty_same.append(empty_same); col_illegal_same.append(illegal_same)
                col_mover.append(int(mover[t]))
                if fl_same:
                    n_flankloss_same += 1

        if (gi + 1) % 2000 == 0:
            print(f"  game {gi+1}/{len(games)}  rows={len(col_game)}  "
                  f"flankloss(t+2)={n_flankloss_same}  "
                  f"({int(time.time()-t0)}s)", flush=True)
        if args.target_events and n_flankloss_same >= args.target_events:
            print(f"reached target {args.target_events} flank-loss (t+2) events "
                  f"at game {gi+1}", flush=True)
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        game=np.array(col_game, dtype=np.int32),
        t=np.array(col_t, dtype=np.int16),
        square=np.array(col_sq, dtype=np.int16),
        turn_before=np.array(col_turn, dtype=np.int16),
        P_before=np.array(col_pbefore, dtype=np.float32),
        P_next=np.array(col_pnext, dtype=np.float32),
        empty_next=np.array(col_empty_next, dtype=bool),
        illegal_next=np.array(col_illegal_next, dtype=bool),
        P_same=np.array(col_psame, dtype=np.float32),
        empty_same=np.array(col_empty_same, dtype=bool),
        illegal_same=np.array(col_illegal_same, dtype=bool),
        mover=np.array(col_mover, dtype=np.int8),
        meta=np.array([f"n_games={len(games)}",
                       f"rows={len(col_game)}",
                       f"flankloss_same={n_flankloss_same}"]),
    )
    # quick summary: flank-loss at t+2
    P_before = np.array(col_pbefore, dtype=np.float64)
    P_same = np.array(col_psame, dtype=np.float64)
    fl_same = np.array(col_empty_same, dtype=bool) & np.array(col_illegal_same, dtype=bool)
    print()
    print(f"rows total: {len(col_game)}   flank-loss(t+2): {int(fl_same.sum())}")
    if fl_same.sum():
        pb = P_before[fl_same]; pa = P_same[fl_same]
        ret = np.where(pb > 0, pa / pb, np.nan)
        print(f"  P_before  mean={np.nanmean(pb):.4f}  median={np.nanmedian(pb):.4f}")
        print(f"  P_after   mean={np.nanmean(pa):.4f}  median={np.nanmedian(pa):.4f}")
        print(f"  retention mean={np.nanmean(ret):.4f}  median={np.nanmedian(ret):.4f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
