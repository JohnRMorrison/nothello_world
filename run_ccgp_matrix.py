"""Driver: CCGP matrix over {OGPT + N MLPs} x {linear, nonlinear}, all on the
SAME positions, computing the expensive OGPT forward + game replay ONCE.

Samples positions once (sample_shared_positions), reuses that sample for the
OGPT residual and every MLP's hidden, and runs all CCGP modes. Prints a final
Gap table (mode x model) so the MLP-vs-OGPT abstraction comparison is one glance.

Usage (pod):
  python3.13 run_ccgp_matrix.py \
    --mlps <CKDIR>/pattern_simple_direct_H{512,4096}_{playedeven,move_grid}.pt \
    --ogpt-ckpt ckpts/gpt_synthetic.ckpt --layer 6 \
    --n 100000 --probes both
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_ccgp as C

MODES = ["phase", "phase_fwd", "phase_bwd", "context", "crowd", "frontier",
         "spatial", "flip", "recency", "null"]


def infer_mlp(path):
    name = os.path.basename(path)
    hidden = 4096 if "H4096" in name else 8192 if "H8192" in name else 512
    feat = "move_grid" if "move_grid" in name else "playedeven" if "playedeven" in name else "wheneven"
    tag = f"H{hidden}_{feat}"
    return hidden, feat, tag


def _avg(res_by_parity, key):
    vals = [r[key] for r in res_by_parity.values() if r and not np.isnan(r.get(key, np.nan))]
    return float(np.mean(vals)) if vals else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", nargs="+", required=True)
    ap.add_argument("--ogpt-ckpt", default="ckpts/gpt_synthetic.ckpt")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--ply-min", type=int, default=5)
    ap.add_argument("--ply-max", type=int, default=54)
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--ccgp-mode", default="all",
                    help="Which CCGP mode(s) to run (default all). Single mode "
                         "e.g. phase / phase_fwd / phase_bwd / spatial for sweeps.")
    ap.add_argument("--probes", choices=["linear", "nonlinear", "both"], default="both")
    ap.add_argument("--max-cells", type=int, default=None,
                    help="Decode only this many (random) cells per mode instead of "
                         "all 64. Gap is cell-averaged, so ~3x fewer cells barely "
                         "moves the mean but is ~3x faster. e.g. 24.")
    ap.add_argument("--j1b-bank", default=None,
                    help="If set (e.g. banks/J1_perpattern.pt), also run J1B "
                         "(tree-leaf one-hot, SVD-reduced) on the same positions.")
    ap.add_argument("--j1b-flanking", default="hand_crafted_flanking_patterns.pt")
    ap.add_argument("--j1b-svd-k", type=int, default=2048)
    args = ap.parse_args()

    print(f"Sampling {args.n} shared positions (OGPT L{args.layer}) -- computed ONCE ...", flush=True)
    sample = C.sample_shared_positions(args.ogpt_ckpt, args.layer, args.n,
                                       ply_lo=args.ply_min, ply_hi=args.ply_max)

    # Cache each model's (per_parity, aux) on the shared sample -- reused for both probes.
    models = [("OGPT", C.ogpt_from_sample(sample))]
    for path in args.mlps:
        hidden, feat, tag = infer_mlp(path)
        models.append((f"MLP:{tag}", C.mlp_from_sample(sample, path, hidden, feat)))
    if args.j1b_bank:
        models.append((f"J1B:svd{args.j1b_svd_k}",
                       C.j1b_from_sample(sample, args.j1b_bank, args.j1b_flanking,
                                         svd_k=args.j1b_svd_k)))

    passes = (["linear", "nonlinear"] if args.probes == "both" else [args.probes])
    for probe in passes:
        nl = (probe == "nonlinear")
        run_args = argparse.Namespace(ccgp_mode=args.ccgp_mode, nonlinear=nl,
                                      n_bins=args.n_bins, max_cells=args.max_cells)
        print(f"\n############################## PROBE = {probe.upper()} ##############################", flush=True)
        table = {}
        for label, (pp, aux) in models:
            table[label] = C.run_modes(pp, aux, run_args, model_label=label)

        # ---- summary Gap table (avg across parities) ----
        names = [lbl for lbl, _ in models]
        print(f"\n===== {probe.upper()} : CCGP Gap = Within - CCGP  (small = abstract; check null~0) =====")
        print(f"{'mode':10s} " + " ".join(f"{n:>16s}" for n in names))
        for metric in ("gap", "within"):
            print(f"-- {metric} --")
            for mode in MODES:
                row = []
                for n in names:
                    r = table[n].get(mode, {})
                    row.append(f"{_avg(r, metric):>16.4f}" if r else f"{'-':>16s}")
                print(f"{mode:10s} " + " ".join(row))


if __name__ == "__main__":
    main()
