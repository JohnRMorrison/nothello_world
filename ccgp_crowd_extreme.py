"""Extreme-crowd CCGP for Othello-GPT ONLY.

Decode "cell C == color X" conditioned on C's local crowding, with a STARK
split instead of the median: train on positions where C has <= --sparse-max
occupied 8-neighbors, test on positions where C has >= --crowd-min occupied
8-neighbors (and reverse). Within = 2-fold within each tail at matched train
size. Gap = Within - CCGP; large Gap => C's decode depends on local crowding.

  python ccgp_crowd_extreme.py --ogpt-ckpt ckpts/gpt_synthetic.ckpt --layer 6 \
      --n 100000 --sparse-max 1 --crowd-min 7

CAVEAT (read before interpreting): <2 occupied neighbors happens EARLY (sparse
board), >6 happens LATE (full board) -- so this extreme split is confounded with
game phase. A large Gap could be crowding OR phase. Use --match-ply to subsample
the two tails to the SAME ply distribution and remove that confound.
"""
import argparse
import numpy as np
import compute_ccgp as C


def crowd_extreme(h, board, pos, sparse_max=1, crowd_min=7, classes=(1, 2),
                  cells=range(64), seed=0, min_per_tail=150, nonlinear=False,
                  match_ply=False, sparse_frac=None, crowd_frac=None):
    rng = np.random.RandomState(seed)
    ccgp_per, within_per = [], []
    sp_sizes, cr_sizes = [], []

    def _match_ply(a_idx, b_idx):
        """Subsample a_idx and b_idx to the same ply histogram (removes the
        early/late confound)."""
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
        nei = C._NEI[cell]
        occ = (board[:, nei] != 0).sum(1)                      # occupied neighbors
        if sparse_frac is not None:                            # FRACTION mode (all cells)
            frac = occ / len(nei)
            sp_idx = np.where(frac <= sparse_frac)[0]
            cr_idx = np.where(frac >= crowd_frac)[0]
        else:                                                  # absolute-count mode (interior only)
            sp_idx = np.where(occ <= sparse_max)[0]
            cr_idx = np.where(occ >= crowd_min)[0]
        if match_ply:
            sp_idx, cr_idx = _match_ply(sp_idx, cr_idx)
        if len(sp_idx) < min_per_tail or len(cr_idx) < min_per_tail:
            continue
        for cls in classes:
            y = (board[:, cell] == cls).astype(np.int32)
            per = []
            for idx in (sp_idx, cr_idx):
                if y[idx].sum() < 30 or (1 - y[idx]).sum() < 30:
                    per.append(None); continue
                per.append(C._balance(idx, y, rng))
            if any(p is None for p in per):
                continue
            ccgp_pool = min(len(per[0]), len(per[1]))
            within_pool = min(len(per[0]) // 2, len(per[1]) // 2)
            t = max(50, min(ccgp_pool, within_pool))

            # CCGP: train one tail, test the other (both directions)
            fa = []
            for held in (0, 1):
                tr = C._subsample(per[1 - held], t, rng); te = per[held]
                a = C._probe_acc(h[tr], y[tr], h[te], y[te], nonlinear=nonlinear)
                if a is not None:
                    fa.append(a)
            if fa:
                ccgp_per.append(np.mean(fa))

            # Within: 2-fold within each tail at the same train size
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
            sp_sizes.append(len(sp_idx)); cr_sizes.append(len(cr_idx))

    return {
        'ccgp':   float(np.mean(ccgp_per))   if ccgp_per else float('nan'),
        'within': float(np.mean(within_per)) if within_per else float('nan'),
        'gap':    float(np.mean(within_per) - np.mean(ccgp_per))
                  if (ccgp_per and within_per) else float('nan'),
        'n_pairs': len(ccgp_per),
        'mean_sparse_n': int(np.mean(sp_sizes)) if sp_sizes else 0,
        'mean_crowd_n':  int(np.mean(cr_sizes)) if cr_sizes else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ogpt-ckpt", default="ckpts/gpt_synthetic.ckpt")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--sparse-max", type=int, default=1, help="<= this many occupied neighbors = sparse (count mode)")
    ap.add_argument("--crowd-min", type=int, default=7, help=">= this many occupied neighbors = crowded (count mode)")
    ap.add_argument("--sparse-frac", type=float, default=None,
                    help="FRACTION mode: occupied/#neighbors <= this = sparse. "
                         "Includes corners/edges (by their own neighbor count) -> "
                         "bigger, cleaner sample. e.g. 0.25")
    ap.add_argument("--crowd-frac", type=float, default=None,
                    help="FRACTION mode: occupied/#neighbors >= this = crowded. e.g. 0.75")
    ap.add_argument("--nonlinear", action="store_true")
    ap.add_argument("--match-ply", action="store_true",
                    help="Subsample the two tails to the same ply distribution "
                         "(removes the early/late confound).")
    args = ap.parse_args()

    if args.sparse_frac is not None:
        desc = (f"FRACTION: occupied/#nei <= {args.sparse_frac} (sparse) vs "
                f">= {args.crowd_frac} (crowded); all cells")
    else:
        desc = (f"COUNT: <= {args.sparse_max} (sparse) vs >= {args.crowd_min} "
                f"(crowded) occupied neighbors; interior cells only")
    print(f"OGPT extreme-crowd CCGP: {desc}  (match_ply={args.match_ply})", flush=True)
    per_parity, _aux = C.get_ogpt_activations(args.ogpt_ckpt, args.layer, None, args.n)
    for parity, (h, board, pos) in per_parity.items():
        r = crowd_extreme(h, board, pos, sparse_max=args.sparse_max, crowd_min=args.crowd_min,
                          sparse_frac=args.sparse_frac, crowd_frac=args.crowd_frac,
                          nonlinear=args.nonlinear, match_ply=args.match_ply)
        print(f"\n--- parity={parity} ---")
        print(f"  CCGP   = {r['ccgp']:.4f}")
        print(f"  Within = {r['within']:.4f}")
        print(f"  Gap    = {r['gap']:.4f}")
        print(f"  ({r['n_pairs']} (cell,class) pairs; mean tail sizes: "
              f"sparse~{r['mean_sparse_n']}, crowd~{r['mean_crowd_n']})")


if __name__ == "__main__":
    main()
