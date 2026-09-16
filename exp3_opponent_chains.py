"""Experiment 3 - probability on an ILLEGAL empty square vs the length of the
opponent chain leading up to it.

For each decision point t (mover X, opponent Y=-X, board state S), we take
every EMPTY square that is currently an ILLEGAL move.  For each such square we
measure, in all 8 directions, the length of the contiguous run of opponent (Y)
stones starting adjacent to the square, and take the LONGEST as the square's
chain length (0..7).  We record the model's probability on that (illegal) move.

Confound: a square can be flanked by opponent runs in several directions.  For
now the chain length is the LONGEST run, but we also track, per chain-length
bucket, how many squares had >=2 non-empty runs (multi-chain), and keep a
single-chain-only aggregate, so the confound can be handled at plot time.

Saved (sufficient statistics, so mean + CI are exact):
  For k = 0..MAXL (default 7):
    n[k], sum_p[k], sumsq_p[k]                 # all illegal empty squares
    n_single[k], sum_p_single[k], sumsq_p_single[k]   # exactly 1 non-empty run
    n_multi[k],  sum_p_multi[k],  sumsq_p_multi[k]    # >=2 non-empty runs
  cturn_n[k, turn], cturn_sump[k, turn]        # chain x absolute-turn (confound)
  n_dirs_hist[k, d]                            # how many squares had d chains
Plus a capped raw sample (game, t, square, turn, longest, n_dirs, prob).

Eventual line graph: x = longest opponent chain (0..6), y = mean P(illegal
move) with a 95% CI; overlay single-chain-only to show the multi-chain effect.

Usage (pod):
  source $(conda info --base)/etc/profile.d/conda.sh; conda activate othello
  python exp3_opponent_chains.py --n-games 4000 --max-files 4 \
      --out experiments/exp3_opponent_chains.npz
"""
import argparse
import os
import time

import numpy as np

from ogpt_behavioral_common import (
    pick_device, load_ogpt, iter_game_probs, replay_game, M2I,
    opponent_chain_lengths, load_experiment_games,
)

MAXL = 7          # max possible uncapped opponent run from an empty square


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    ap.add_argument("--n-games", type=int, default=4000)
    ap.add_argument("--max-files", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--raw-cap", type=int, default=1000000,
                    help="max raw per-square rows to also store (0 = none)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/exp3_opponent_chains.npz")
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}", flush=True)
    model, block_size, cell_tokens = load_ogpt(args.ckpt, device)
    games = load_experiment_games(args.max_files, args.n_games,
                                  seed=args.seed, shuffle=args.shuffle)
    print(f"Loaded {len(games)} games", flush=True)

    K = MAXL + 1
    n = np.zeros(K, dtype=np.int64); sump = np.zeros(K); sumsq = np.zeros(K)
    n_s = np.zeros(K, dtype=np.int64); sump_s = np.zeros(K); sumsq_s = np.zeros(K)
    n_m = np.zeros(K, dtype=np.int64); sump_m = np.zeros(K); sumsq_m = np.zeros(K)
    cturn_n = np.zeros((K, 60), dtype=np.int64)
    cturn_sump = np.zeros((K, 60))
    n_dirs_hist = np.zeros((K, 9), dtype=np.int64)      # d = 0..8 chains

    raw_g, raw_t, raw_sq, raw_turn, raw_L, raw_nd, raw_p = [], [], [], [], [], [], []
    n_total = 0
    t0 = time.time()
    for gi, game, probs in iter_game_probs(model, games, block_size,
                                           cell_tokens, device, args.batch):
        states, legal, mover = replay_game(game)
        Tp = probs.shape[0]                             # predictable decisions (~59)
        for t in range(Tp):
            if not legal[t].any():
                continue
            opp = int(-mover[t])
            st = states[t]
            # empty squares that are illegal moves (empty & not legal); center
            # never empty, so all such squares are movable.
            empty = (st == 0)
            illegal_empty = empty & (~legal[t])
            cells = np.where(illegal_empty)[0]
            for c in cells:
                if c not in M2I:                       # safety (center) - skip
                    continue
                runs = opponent_chain_lengths(st, int(c), opp)
                longest = int(runs.max())
                if longest > MAXL:
                    longest = MAXL
                n_dirs = int((runs > 0).sum())
                p = float(probs[t, M2I[c]])
                n[longest] += 1; sump[longest] += p; sumsq[longest] += p * p
                cturn_n[longest, t] += 1; cturn_sump[longest, t] += p
                n_dirs_hist[longest, min(n_dirs, 8)] += 1
                if n_dirs >= 2:
                    n_m[longest] += 1; sump_m[longest] += p; sumsq_m[longest] += p * p
                else:
                    n_s[longest] += 1; sump_s[longest] += p; sumsq_s[longest] += p * p
                n_total += 1
                if args.raw_cap and len(raw_g) < args.raw_cap:
                    raw_g.append(gi); raw_t.append(t); raw_sq.append(int(c))
                    raw_turn.append(t + 1); raw_L.append(longest)
                    raw_nd.append(n_dirs); raw_p.append(p)

        if (gi + 1) % 1000 == 0:
            print(f"  game {gi+1}/{len(games)}  illegal-empty squares={n_total}  "
                  f"({int(time.time()-t0)}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        chain=np.arange(K),
        n=n, sum_p=sump, sumsq_p=sumsq,
        n_single=n_s, sum_p_single=sump_s, sumsq_p_single=sumsq_s,
        n_multi=n_m, sum_p_multi=sump_m, sumsq_p_multi=sumsq_m,
        cturn_n=cturn_n, cturn_sump=cturn_sump,
        n_dirs_hist=n_dirs_hist,
        raw_game=np.array(raw_g, dtype=np.int32),
        raw_t=np.array(raw_t, dtype=np.int16),
        raw_square=np.array(raw_sq, dtype=np.int16),
        raw_turn=np.array(raw_turn, dtype=np.int16),
        raw_longest=np.array(raw_L, dtype=np.int8),
        raw_n_dirs=np.array(raw_nd, dtype=np.int8),
        raw_prob=np.array(raw_p, dtype=np.float32),
        meta=np.array([f"n_games={len(games)}", f"n_illegal_empty={n_total}",
                       f"MAXL={MAXL}"]),
    )
    print()
    print(f"illegal-empty squares scored: {n_total}")
    print("mean P(illegal move) by longest opponent chain:")
    for k in range(K):
        if n[k]:
            mean = sump[k] / n[k]
            var = max(sumsq[k] / n[k] - mean * mean, 0.0)
            se = (var / n[k]) ** 0.5
            multi = n_m[k] / n[k]
            print(f"  chain {k}: mean={mean:.5f} +/-{1.96*se:.5f}  "
                  f"n={int(n[k])}  multi-chain={multi:.2%}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
