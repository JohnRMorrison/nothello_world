"""Top-K legal-move rate for Othello-GPT, over a chosen ply window.

Top-K rate = fraction of positions where all of the model's top min(K, n_legal)
moves are legal.  The cap matters: ~5% of positions have fewer than 5 legal
moves, so an uncapped "all top-5 legal" is unachievable there and drags the
number down by several points (95.2% vs 99.9% for Othello-GPT).

Covers the trained model and the --random-init baseline (fresh weights, never
trained -- Li et al.'s convention, model.apply(_init_weights)).

Usage:
    python ogpt_topk_legality.py --n-games 2000 --ply-min 5 --ply-max 54
    python ogpt_topk_legality.py --random-init --n-games 2000
"""
import sys, os, argparse, glob, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState

CENTER = {27, 28, 35, 36}
VALID = [c for c in range(64) if c not in CENTER]
CELL_TO_TOK = {c: i + 1 for i, c in enumerate(VALID)}
TOK_TO_CELL = {t: c for c, t in CELL_TO_TOK.items()}


def load_model(ckpt, random_init):
    mconf = GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)                       # __init__ already random-inits
    if not random_init:
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    return model


def load_games(n_games, max_files=3):
    files = sorted(glob.glob('data/othello_synthetic/*.pickle'))[:max_files]
    out = []
    for f in files:
        with open(f, 'rb') as fh:
            for g in pickle.load(fh):
                if len(g) == 60:
                    out.append(g)
                    if len(out) >= n_games:
                        return out
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='ckpts/gpt_synthetic.ckpt')
    p.add_argument('--random-init', action='store_true')
    p.add_argument('--n-games', type=int, default=2000)
    p.add_argument('--ply-min', type=int, default=5)
    p.add_argument('--ply-max', type=int, default=54)   # half-open, = moves 5-53
    p.add_argument('--ks', type=int, nargs='+', default=[1, 3, 5])
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--out-npz', default=None)
    a = p.parse_args()

    model = load_model(a.ckpt, a.random_init)
    games = load_games(a.n_games)
    label = 'ogpt_random' if a.random_init else 'ogpt'
    print(f'{label}: {len(games)} games, plies [{a.ply_min},{a.ply_max})',
          flush=True)

    hit = {k: 0 for k in a.ks}
    n = 0
    t0 = time.time()
    for b0 in range(0, len(games), a.batch):
        batch = games[b0:b0 + a.batch]
        toks = torch.tensor([[CELL_TO_TOK[c] for c in g[:a.ply_max]]
                             for g in batch], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(toks)               # (B, T, 61)
        for bi, g in enumerate(batch):
            board = OthelloBoardState()
            for ply in range(a.ply_max):
                if ply >= a.ply_min:
                    legal = set(board.get_valid_moves())
                    if legal:
                        # rank the 60 real moves; logit 0 is the pad token
                        sc = logits[bi, ply - 1, 1:]
                        order = torch.argsort(sc, descending=True)
                        for k in a.ks:
                            ke = min(k, len(legal))     # cannot need more
                            top = [TOK_TO_CELL[int(order[j]) + 1]
                                   for j in range(ke)]
                            hit[k] += int(all(c in legal for c in top))
                        n += 1
                board.update([g[ply]])
        if (b0 // a.batch) % 5 == 0:
            print(f'  {b0 + len(batch)}/{len(games)} games  '
                  f'({time.time() - t0:.0f}s)', flush=True)

    print(f'\n=== {label}, plies [{a.ply_min},{a.ply_max}), N={n:,} ===')
    for k in a.ks:
        print(f'  top-{k} legal rate: {100 * hit[k] / max(n, 1):.2f}%')
    if a.out_npz:
        np.savez(a.out_npz, model=label, ks=np.array(a.ks),
                 rates=np.array([hit[k] / max(n, 1) for k in a.ks]),
                 n=n, ply_min=a.ply_min, ply_max=a.ply_max)
        print(f'saved {a.out_npz}')


if __name__ == '__main__':
    main()
