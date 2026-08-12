"""Fixed-threshold recency CCGP for Othello-GPT ONLY (companion to
ccgp_crowd_extreme.py).

Decode "cell C == color X" conditioned on C's age, with a fixed, interpretable
split instead of the per-cell median: train on positions where C was placed
<= --recent-max moves ago, test on positions where C was placed >= --settled-min
moves ago (and reverse). Within = 2-fold within each tail at matched train size.
Gap = Within - CCGP; large Gap => C's decode depends on how long ago it was placed.

  python ccgp_recency_extreme.py --ogpt-ckpt ckpts/gpt_synthetic.ckpt --layer 6 \
      --n 100000 --recent-max 2 --settled-min 15

CAVEAT: "settled" (placed >15 moves ago) can only occur LATE in the game, while
"recent" occurs at any ply -- so this split is confounded with game phase. Use
--match-ply to subsample the two tails to the same ply distribution (recency-only
test). Watch the printed tail sizes: if --match-ply leaves few, the two axes are
nearly the same and you'd need a smaller settled-min.
"""
import argparse
import numpy as np
import compute_ccgp as C


def recency_extreme(h, board, pos, place_step, recent_max=2, settled_min=15,
                    classes=(1, 2), cells=range(64), seed=0, min_per_tail=150,
                    nonlinear=False, match_ply=False):
    rng = np.random.RandomState(seed)
    ccgp_per, within_per, rec_sizes, set_sizes = [], [], [], []

    def _match_ply(a_idx, b_idx):
        out_a, out_b = [], []
        for p in np.unique(np.concatenate([pos[a_idx], pos[b_idx]])):
            aa = a_idx[pos[a_idx] == p]; bb = b_idx[pos[b_idx] == p]
            k = min(len(aa), len(bb))
            if k == 0:
                continue
            out_a.append(rng.choice(aa, k, replace=False))
            out_b.append(rng.choice(bb, k, replace=False))
        if not out_a:
            return a_idx[:0], b_idx[:0]
        return np.concatenate(out_a), np.concatenate(out_b)

    for cell in cells:
        placed = place_step[:, cell] >= 0
        occ = (board[:, cell] != 0) & placed
        rec = pos - place_step[:, cell]                        # moves since placement
        recent_idx = np.where(occ & (rec <= recent_max))[0]
        settled_idx = np.where(occ & (rec >= settled_min))[0]
        if match_ply:
            recent_idx, settled_idx = _match_ply(recent_idx, settled_idx)
        if len(recent_idx) < min_per_tail or len(settled_idx) < min_per_tail:
            continue
        for cls in classes:
            y = (board[:, cell] == cls).astype(np.int32)
            per = []
            for idx in (recent_idx, settled_idx):
                if y[idx].sum() < 30 or (1 - y[idx]).sum() < 30:
                    per.append(None); continue
                per.append(C._balance(idx, y, rng))
            if any(p is None for p in per):
                continue
            ccgp_pool = min(len(per[0]), len(per[1]))
            within_pool = min(len(per[0]) // 2, len(per[1]) // 2)
            t = max(50, min(ccgp_pool, within_pool))

            fa = []
            for held in (0, 1):
                tr = C._subsample(per[1 - held], t, rng); te = per[held]
                a = C._probe_acc(h[tr], y[tr], h[te], y[te], nonlinear=nonlinear)
                if a is not None:
                    fa.append(a)
            if fa:
                ccgp_per.append(np.mean(fa))

            wf = []
            for s in (0, 1):
                idx = per[s].copy(); rng.shuffle(idx); half = len(idx) // 2
                for trp, tep in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
                    tr = C._subsample(trp, t, rng)
                    a = C._probe_acc(h[tr], y[tr], h[tep], y[tep], nonlinear=nonlinear)
                    if a is not None:
                        wf.append(a)
            if wf:
                within_per.append(np.mean(wf))
            rec_sizes.append(len(recent_idx)); set_sizes.append(len(settled_idx))

    return {
        'ccgp':   float(np.mean(ccgp_per))   if ccgp_per else float('nan'),
        'within': float(np.mean(within_per)) if within_per else float('nan'),
        'gap':    float(np.mean(within_per) - np.mean(ccgp_per))
                  if (ccgp_per and within_per) else float('nan'),
        'n_pairs': len(ccgp_per),
        'mean_recent_n': int(np.mean(rec_sizes)) if rec_sizes else 0,
        'mean_settled_n': int(np.mean(set_sizes)) if set_sizes else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ogpt-ckpt", default="ckpts/gpt_synthetic.ckpt")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--recent-max", type=int, default=2, help="placed <= this many moves ago = recent")
    ap.add_argument("--settled-min", type=int, default=15, help="placed >= this many moves ago = settled")
    ap.add_argument("--nonlinear", action="store_true")
    ap.add_argument("--match-ply", action="store_true",
                    help="Subsample the two tails to the same ply distribution "
                         "(removes the late-game confound).")
    args = ap.parse_args()

    print(f"OGPT extreme-recency CCGP: placed <= {args.recent_max} moves ago  vs  "
          f">= {args.settled_min} moves ago  (match_ply={args.match_ply})", flush=True)
    per_parity, aux = C.get_ogpt_activations(args.ogpt_ckpt, args.layer, None, args.n)
    for parity, (h, board, pos) in per_parity.items():
        place_color, place_step = aux[parity]
        r = recency_extreme(h, board, pos, place_step, recent_max=args.recent_max,
                            settled_min=args.settled_min, nonlinear=args.nonlinear,
                            match_ply=args.match_ply)
        print(f"\n--- parity={parity} ---")
        print(f"  CCGP   = {r['ccgp']:.4f}")
        print(f"  Within = {r['within']:.4f}")
        print(f"  Gap    = {r['gap']:.4f}")
        print(f"  ({r['n_pairs']} (cell,class) pairs; mean tail sizes: "
              f"recent~{r['mean_recent_n']}, settled~{r['mean_settled_n']})")


if __name__ == "__main__":
    main()
