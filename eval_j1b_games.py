"""Per-GAME illegal-mass + top-K legality for a streaming probe (J1B),
evaluated by replaying RAW GAMES instead of reading a feature chunk.

Why not the chunk path: chunk_ext rows carry no game id, so
reeval_argmax_legality.py can only report per-POSITION rates.  The
Othello-GPT column in the paper is the per-GAME maximum over positions, so
the two are not comparable.  Replaying games recovers the grouping.

Metric, matching how the Othello-GPT column was measured:
  * per position -- renormalise the 60 non-centre cell scores to sum to 1
    (the probe's scores are independent sigmoids, not a distribution; OGPT's
    come from a softmax, so they only mean the same thing after this), then
    sum the mass sitting on currently-illegal cells
  * per game -- take the max over positions in the ply window
  * report the fraction of games above 5% and above 10%

Also prints top-1/3/5 legality (FRAC, capped at n_legal) as a CHECK on the
feature construction: these should land near the chunk path's numbers
(prob-OR top-1 ~98.5%).  If the features were built wrongly -- wrong
canonicalize_mover, wrong recent-K block -- top-1 collapses and the illegal
mass is meaningless.  Read that line before trusting the table numbers.

Ply window is INCLUSIVE move counts: move n = the board after n moves have
been played, the same convention as eval_fair_legality.py and
ogpt_topk_legality.py.

Usage:
  python eval_j1b_games.py --probe-ckpts stream_out/J1_B.pt --num-games 8000
"""
import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from opening_tree_mlp import playedeven_features, BOARD_CELLS
from train_streaming_probe import build_hidden_layer_batch, load_leaf_build
from reeval_argmax_legality import load_probe, _probor_and_max, _CENTER_MASK

THRESHOLDS = (0.05, 0.10)


def load_games(data_dir, num_files, num_games, min_length=54):
    """Held-out games: the LAST num_files pickles, matching
    compare_v4_vs_mlp.load_val_games (NOT ogpt_topk_legality.load_games,
    which takes the FIRST files -- those are inside Othello-GPT's training
    set).

    min_length drops games shorter than the ply window (~1% of them; the rest
    are exactly 60 moves).  Every model must score the SAME games for the
    per-game rates to be comparable, and a short game would otherwise take its
    maximum over fewer positions -- and Othello-GPT's batched forward needs a
    rectangular token tensor anyway.
    """
    files = sorted(os.listdir(data_dir))[-num_files:]
    out = []
    for fname in files:
        with open(os.path.join(data_dir, fname), 'rb') as f:
            batch = pickle.load(f)
        out.extend(g for g in batch if len(g) >= min_length)
        if len(out) >= num_games:
            break
    return out[:num_games]


def replay(games, ply_min, ply_max, recent_Ks, canonicalize_mover):
    """(features, legal mask, game id) for every scored position.

    Mirrors train_streaming_probe.process_pickle_chunk's replay exactly --
    including the forfeit handling (no legal move => update([]) and re-check)
    and ply = len(prefix) = moves played -- but keeps the game index, and
    passes canonicalize_mover through, which process_pickle_chunk does not.
    """
    from data.othello import OthelloBoardState
    feats, legals, gids = [], [], []
    for gid, game_moves in enumerate(games):
        board = OthelloBoardState()
        prefix = []
        for move in game_moves:
            valid = board.get_valid_moves()
            if not valid:
                board.update([])                 # forfeit: no pass token exists
                valid = board.get_valid_moves()
                if not valid:
                    break
            ply = len(prefix)
            if ply_min <= ply <= ply_max:
                feats.append(playedeven_features(
                    prefix, recent_Ks=recent_Ks,
                    canonicalize_mover=canonicalize_mover))
                lmask = np.zeros(BOARD_CELLS, dtype=np.uint8)
                for m in valid:
                    lmask[m] = 1
                legals.append(lmask)
                gids.append(gid)
            if move not in valid:
                break
            board.update([move])
            prefix.append(move)
    return (np.stack(feats).astype(np.float32),
            np.stack(legals),
            np.array(gids, dtype=np.int64))


def accumulate(scores, legal, gids, ks, frac, worst):
    """Fold one batch of per-cell scores into the running totals.

    scores (B, 64) float, legal (B, 64) bool, gids (B,) int.
    frac: {k: running sum of per-position FRAC}; worst: (n_games,) per-game
    max illegal mass, updated in place.
    """
    s = scores.copy()
    s[:, _CENTER_MASK] = 0.0
    n_legal = legal[:, ~_CENTER_MASK].sum(axis=1)
    order = s.argsort(axis=1)[:, ::-1]
    for k in ks:
        ke = np.minimum(n_legal, k)
        picked = np.take_along_axis(legal, order[:, :k], axis=1)
        within = np.arange(k)[None, :] < ke[:, None]
        hits = (picked & within).sum(axis=1)
        frac[k] += float(np.where(ke > 0, hits / np.maximum(ke, 1), 0.0).sum())
    sn = s / np.maximum(s.sum(axis=1, keepdims=True), 1e-12)
    mass = np.where((~_CENTER_MASK) & (~legal), sn, 0.0).sum(axis=1)
    for j in range(len(gids)):
        if mass[j] > worst[gids[j]]:
            worst[gids[j]] = mass[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe-ckpts', nargs='+', required=True)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=3)
    ap.add_argument('--num-games', type=int, default=8000)
    ap.add_argument('--ply-min', type=int, default=5)
    ap.add_argument('--ply-max', type=int, default=53)      # INCLUSIVE
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 3, 5])
    ap.add_argument('--canonicalize-mover', action='store_true',
                    help='Override -- otherwise taken from probe saved_args.')
    ap.add_argument('--out-npz', default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)

    paths = []
    for p in args.probe_ckpts:
        paths.extend(sorted(glob.glob(p)) or ([p] if os.path.exists(p) else []))
    print(f'{len(paths)} probe checkpoint(s)\n', flush=True)

    games = load_games(args.data_dir, args.num_data_files, args.num_games,
                       min_length=args.ply_max + 1)
    print(f'{len(games):,} held-out games, >= {args.ply_max + 1} moves '
          f'(last {args.num_data_files} pickles in {args.data_dir})', flush=True)

    cache = {}          # (recent_Ks, canon) -> (X, L, gid)
    for path in paths:
        info = load_probe(path, device)
        sa = info['saved_args']

        # Feature flags must match training exactly, or H is garbage.
        canon = args.canonicalize_mover or sa.get('canonicalize_mover', False)
        if sa.get('flanking_only') or sa.get('no_recent'):
            recent_Ks = None
        else:
            rk = sa.get('recent_Ks')
            recent_Ks = (tuple(int(k) for k in str(rk).split(',') if k.strip())
                         or None) if rk else None
        use_relu = sa.get('use_relu', False)
        no_flanking = info['no_flanking']

        # The leaf-based (J3 ordinal) bank needs a 241-d input rebuilt from a
        # chunk's movesago block; there is no raw-game path for it here.
        if not sa.get('flanking_only') and load_leaf_build(info['tree_path']):
            raise SystemExit(
                f'{os.path.basename(path)}: leaf-based (J3 ordinal) tree bank '
                '-- needs the chunk movesago block, no raw-game path. Use '
                'reeval_argmax_legality.py for this checkpoint.')
        expected = 121 + (60 * len(recent_Ks) if recent_Ks else 0)
        tree_dim = info['mlp'].W.shape[1]
        if tree_dim > expected:
            raise SystemExit(
                f'{os.path.basename(path)}: tree input_dim={tree_dim} > '
                f'featurizer dim={expected} -- recent-Ks mismatch.')

        print(f'--- {os.path.basename(path)}', flush=True)
        print(f'    canonicalize_mover={canon}  recent_Ks={recent_Ks}  '
              f'use_relu={use_relu}  no_flanking={no_flanking}', flush=True)

        key = (recent_Ks, canon)
        if key not in cache:
            t0 = time.time()
            cache[key] = replay(games, args.ply_min, args.ply_max,
                                recent_Ks, canon)
            X, L, gid = cache[key]
            print(f'    replayed {len(X):,} positions '
                  f'(moves {args.ply_min}-{args.ply_max}) in '
                  f'{time.time()-t0:.0f}s', flush=True)
        X, L, gid = cache[key]

        worst = {a: np.zeros(len(games), np.float32) for a in ('probor', 'max')}
        frac = {a: {k: 0.0 for k in args.ks} for a in ('probor', 'max')}
        n = 0
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(X), args.batch_size):
                Xb, Lb, gb = (X[i:i + args.batch_size],
                              L[i:i + args.batch_size],
                              gid[i:i + args.batch_size])
                H = build_hidden_layer_batch(Xb, info['mlp'], info['patterns'],
                                             recent_Ks, use_relu, device,
                                             no_flanking=no_flanking)
                po, mx = _probor_and_max(info['probes'],
                                         H.float() if not use_relu else H)
                legal = (Lb > 0)
                for agg, scores in (('probor', po), ('max', mx)):
                    if scores is None:
                        continue
                    accumulate(scores.cpu().numpy(), legal, gb, args.ks,
                               frac[agg], worst[agg])
                n += len(Xb)
        print(f'    scored {n:,} positions in {time.time()-t0:.0f}s', flush=True)

        ng = len(games)
        for agg in ('probor', 'max'):
            if frac[agg][args.ks[0]] == 0.0 and worst[agg].max() == 0.0:
                continue        # this aggregator is undefined for the head
            tk = '  '.join(f'top{k}={100*frac[agg][k]/n:.2f}%' for k in args.ks)
            print(f'    {agg:6s} FRAC  {tk}', flush=True)
            w = worst[agg]
            cells = '  '.join(
                f'>{int(100*t)}%: {100*float((w > t).mean()):.2f}%'
                for t in THRESHOLDS)
            print(f'    {agg:6s} per-game max illegal mass: '
                  f'median {np.median(w):.4f}   {cells}', flush=True)
        if args.out_npz:
            np.savez(args.out_npz, probor=worst['probor'], max=worst['max'],
                     ply_min=args.ply_min, ply_max=args.ply_max, n_games=ng)
            print(f'    saved {args.out_npz}', flush=True)
        print(flush=True)


if __name__ == '__main__':
    main()
