"""Experiment 2 - top-1 (argmax) illegal-move rate around a forfeit.

A forfeit (skipped turn) is not tokenized in the input; it shows up as the
SAME colour playing two moves in a row.  We detect each forfeit during replay
(played_colour[t] == played_colour[t-1]) and then record the model's top-1
legality at each decision point in a window of plies around it.

  offset o (relative to the forfeit): the decision point predicting move
  tf + o, where tf is the first move after the skip.  o = 0 is the first
  prediction after the forfeit; negative o are pre-forfeit baseline plies.

Per row:
  forfeit_id, game, forfeit_turn(tf), offset, pred_move,
  mover(+1/-1), skipped_color(+1/-1), is_skipped_player,
  is_error(top-1 argmax illegal), n_legal, top1_square

Also saved: a turn-matched baseline (base_n[turn], base_err[turn]) computed
over ALL decision points, so the forfeit-conditioned error can be compared to
the unconditional top-1 error at the same absolute turn (error rises with turn
regardless of forfeits - this controls for it).

Eventual line graph: x = offset (0..16), y = error rate with a binomial CI,
optionally restricted to is_skipped_player.

Usage (pod):
  source $(conda info --base)/etc/profile.d/conda.sh; conda activate othello
  python exp2_forfeit_top1_error.py --n-games 400000 --max-files 200 \
      --target-forfeits 100000 --out experiments/exp2_forfeit_top1_error.npz
"""
import argparse
import os
import time

import numpy as np

from ogpt_behavioral_common import (
    pick_device, load_ogpt, iter_game_probs, replay_game, MOVABLE, N_MOVES,
    load_experiment_games,
)
from data.othello import OthelloBoardState


def played_colors(game):
    """Colour that PLAYED each move (before umpire advances the hand)."""
    b = OthelloBoardState()
    pc = np.zeros(len(game), dtype=np.int8)
    for t, mv in enumerate(game):
        pc[t] = b.next_hand_color          # side about to play move t
        b.umpire(mv)
    return pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    ap.add_argument("--n-games", type=int, default=400000)
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pre", type=int, default=4, help="baseline plies before forfeit")
    ap.add_argument("--post", type=int, default=16, help="plies after forfeit")
    ap.add_argument("--target-forfeits", type=int, default=100000)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/exp2_forfeit_top1_error.npz")
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}", flush=True)
    model, block_size, cell_tokens = load_ogpt(args.ckpt, device)
    games = load_experiment_games(args.max_files, args.n_games,
                                  seed=args.seed, shuffle=args.shuffle)
    print(f"Loaded {len(games)} games", flush=True)
    movable = np.array(MOVABLE)

    r_fid, r_game, r_tf, r_off, r_pm = [], [], [], [], []
    r_mover, r_skip, r_isskip, r_err, r_nlegal, r_top1 = [], [], [], [], [], []
    base_n = np.zeros(N_MOVES, dtype=np.int64)
    base_err = np.zeros(N_MOVES, dtype=np.int64)

    forfeit_id = 0
    n_games_with_forfeit = 0
    t0 = time.time()
    for gi, game, probs in iter_game_probs(model, games, block_size,
                                           cell_tokens, device, args.batch):
        states, legal, mover = replay_game(game)
        Tp = probs.shape[0]                                 # predictable decisions (~59)
        # top-1 square + error at every predictable decision point
        top1_mi = probs[:Tp].argmax(axis=1)                 # (Tp,) movable idx
        top1_sq = movable[top1_mi]                           # board cell
        nlegal = legal[:Tp].sum(axis=1)                      # (Tp,)
        is_legal_top1 = legal[np.arange(Tp), top1_sq]        # bool
        err = (~is_legal_top1)
        valid = nlegal > 0                                   # has a legal move

        # turn-matched baseline over all valid decision points
        for t in range(Tp):
            if valid[t]:
                base_n[t] += 1
                if err[t]:
                    base_err[t] += 1

        T = len(game)
        pc = played_colors(game)
        forfeits = [t for t in range(1, T) if pc[t] == pc[t - 1]]
        if forfeits:
            n_games_with_forfeit += 1
        for tf in forfeits:
            skipped_color = int(-pc[tf])                    # colour that was skipped
            for o in range(-args.pre, args.post + 1):
                didx = tf - 1 + o                           # decision predicting move tf+o
                if didx < 0 or didx >= Tp or not valid[didx]:
                    continue
                r_fid.append(forfeit_id); r_game.append(gi); r_tf.append(tf)
                r_off.append(o); r_pm.append(didx + 1)
                r_mover.append(int(mover[didx])); r_skip.append(skipped_color)
                r_isskip.append(bool(mover[didx] == skipped_color))
                r_err.append(bool(err[didx])); r_nlegal.append(int(nlegal[didx]))
                r_top1.append(int(top1_sq[didx]))
            forfeit_id += 1

        if (gi + 1) % 5000 == 0:
            print(f"  game {gi+1}/{len(games)}  forfeits={forfeit_id}  "
                  f"rows={len(r_fid)}  ({int(time.time()-t0)}s)", flush=True)
        if args.target_forfeits and forfeit_id >= args.target_forfeits:
            print(f"reached target {args.target_forfeits} forfeits at game {gi+1}",
                  flush=True)
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    off = np.array(r_off, dtype=np.int8)
    err = np.array(r_err, dtype=bool)
    isskip = np.array(r_isskip, dtype=bool)
    np.savez_compressed(
        args.out,
        forfeit_id=np.array(r_fid, dtype=np.int32),
        game=np.array(r_game, dtype=np.int32),
        forfeit_turn=np.array(r_tf, dtype=np.int16),
        offset=off,
        pred_move=np.array(r_pm, dtype=np.int16),
        mover=np.array(r_mover, dtype=np.int8),
        skipped_color=np.array(r_skip, dtype=np.int8),
        is_skipped_player=isskip,
        is_error=err,
        n_legal=np.array(r_nlegal, dtype=np.int16),
        top1_square=np.array(r_top1, dtype=np.int16),
        base_n=base_n, base_err=base_err,
        meta=np.array([f"n_forfeits={forfeit_id}",
                       f"games_with_forfeit={n_games_with_forfeit}",
                       f"n_games={len(games)}", f"pre={args.pre}", f"post={args.post}"]),
    )
    print()
    print(f"forfeits: {forfeit_id}   games_with_forfeit: {n_games_with_forfeit}/{len(games)}")
    print("top-1 error by offset (all movers):")
    for o in range(-args.pre, args.post + 1):
        m = off == o
        if m.sum():
            e = err[m].mean()
            ms = m & isskip
            es = err[ms].mean() if ms.sum() else float("nan")
            print(f"  offset {o:+3d}: err={e:.4f} (n={int(m.sum())})   "
                  f"skipped-player err={es:.4f} (n={int(ms.sum())})")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
