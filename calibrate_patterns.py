"""Fit per-pattern affine calibration on a frozen pattern detector.

For each of 960 patterns, learn (scale_j, bias_j) so that
  cell_score[c] = max_{j: target(j)=c} (s_j * logit_j + b_j)
gives the best argmax-legal accuracy.

Keeps the pattern -> cell mapping intact (no cross-pattern mixing),
so it's strictly a rescaling. If this closes the 80% -> 95%+ gap,
the patterns are well-formed and just miscalibrated.

Usage:
    python calibrate_patterns.py --ckpt pattern_simple_direct_H512.pt \
        --mode direct --hidden 512 --epochs 3
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES,
)
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import (
    DirectMLP, EndToEndMLP, TwoStageMLP, compute_pattern_labels_batch,
    patterns_to_cell_logsumexp, pat_labels_to_cell_labels,
)


def patterns_to_cell_max(pat_logits, pattern_to_cell, n_cells=60):
    """Hard max-per-cell for eval (matches original aggregation)."""
    from train_pattern_simple import _get_cell_pat_index
    idx, mask = _get_cell_pat_index(pattern_to_cell, n_cells)
    gathered = pat_logits[:, idx]
    gathered = gathered.masked_fill(~mask, float('-inf'))
    return gathered.max(dim=-1).values


def run_model(model_even, model_odd, mode, x, pos):
    em = (pos % 2 == 0); om = ~em
    pl = torch.zeros(len(x), 960, device=x.device)
    if mode in ("direct", "randproj"):
        if em.any(): pl[em] = model_even(x[em])
        if om.any(): pl[om] = model_odd(x[om])
    else:
        for mask, m in [(em, model_even), (om, model_odd)]:
            if not mask.any(): continue
            logits, _ = m(x[mask], pos[mask])
            pl[mask] = logits
    return pl


def eval_topk(logits_iter, top_ns=(1, 3, 5, 10)):
    results = {n: {'c': 0, 't': 0} for n in top_ns}
    for cell_scores, legal in logits_iter:
        ps = cell_scores.cpu().numpy()
        gl = (legal > 0.5).cpu().numpy()
        for b in range(ps.shape[0]):
            ls = set(np.where(gl[b])[0].tolist()); K = len(ls)
            if K == 0: continue
            r = np.argsort(-ps[b])
            for n in top_ns:
                k = min(n, K); top = set(r[:k].tolist())
                results[n]['c'] += len(top & ls); results[n]['t'] += k
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["direct", "emergent", "e2e", "two-stage", "randproj"])
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--pos-weight", type=float, default=5.0,
                        help="pos_weight for legal-cell BCE (legal cells ~16%)")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    feature_cols = list(range(N_MOVES, 2 * N_MOVES))

    ckpt = torch.load(args.ckpt, map_location=device)
    n_patterns = ckpt.get('n_patterns', 960)

    if args.mode in ("direct", "randproj"):
        me = DirectMLP(N_MOVES, args.hidden, n_patterns).to(device)
        mo = DirectMLP(N_MOVES, args.hidden, n_patterns).to(device)
    elif args.mode == "two-stage":
        me = TwoStageMLP(N_MOVES, args.hidden, n_patterns).to(device)
        mo = TwoStageMLP(N_MOVES, args.hidden, n_patterns).to(device)
    else:
        me = EndToEndMLP(N_MOVES, args.hidden, n_patterns).to(device)
        mo = EndToEndMLP(N_MOVES, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()
    for p in me.parameters(): p.requires_grad = False
    for p in mo.parameters(): p.requires_grad = False
    print(f"Loaded {args.ckpt} (pat_acc={ckpt.get('best_pat_acc', '?')})")

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device)

    # Calibration parameters: per-pattern scale and bias
    s = nn.Parameter(torch.ones(960, device=device))
    b = nn.Parameter(torch.zeros(960, device=device))
    opt = torch.optim.Adam([s, b], lr=args.lr)
    pw = torch.tensor([args.pos_weight], device=device)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]

    # Evaluate BEFORE calibration on random sample of the eval chunk
    print("\nBefore calibration (raw logits, max-per-cell):")

    def eval_on_chunk(chunk_path, use_calibration):
        X, Y, pos = _load_features(chunk_path)
        X = X[:, feature_cols]
        n = min(len(X), 49 * 10000)
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(len(X), n, replace=False))
        X, Y, pos = X[idx], Y[idx], pos[idx]
        results = {1: {'c': 0, 't': 0}, 3: {'c': 0, 't': 0},
                   5: {'c': 0, 't': 0}, 10: {'c': 0, 't': 0}}
        batch = 1024
        with torch.no_grad():
            for i in range(0, n, batch):
                x = X[i:i + batch].to(device)
                yb = Y[i:i + batch]
                p = pos[i:i + batch]
                pl = run_model(me, mo, args.mode, x, p)
                if use_calibration:
                    pl = s * pl + b
                cell_scores = patterns_to_cell_max(pl, pattern_to_cell)
                gp = torch.from_numpy(compute_pattern_labels_batch(
                    yb.numpy(), p.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                ).to(device)
                legal = pat_labels_to_cell_labels(gp, pattern_to_cell)
                ps = cell_scores.cpu().numpy()
                gl = (legal > 0.5).cpu().numpy()
                for bi in range(len(x)):
                    ls = set(np.where(gl[bi])[0].tolist()); K = len(ls)
                    if K == 0: continue
                    r = np.argsort(-ps[bi])
                    for nn_ in (1, 3, 5, 10):
                        k = min(nn_, K)
                        results[nn_]['c'] += len(set(r[:k].tolist()) & ls)
                        results[nn_]['t'] += k
        return results

    def _print_results(res, label):
        print(f"  {label}")
        for n, d in res.items():
            acc = d['c'] / max(d['t'], 1)
            print(f"    top-{n}: {acc:.4%}  ({d['c']}/{d['t']})")

    _print_results(eval_on_chunk(eval_path, use_calibration=False), "baseline (no calibration)")

    # Train calibration
    print(f"\nTraining calibration: {len(train_paths)} chunks x {args.epochs} epochs, lr={args.lr}, pw={args.pos_weight}")
    for epoch in range(1, args.epochs + 1):
        rng = np.random.RandomState(epoch)
        order = rng.permutation(len(train_paths))
        total_loss = 0.0; total_batches = 0
        for ci in order:
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            tr_X = tr_X[:, feature_cols]
            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), 1024):
                sel = perm[i:i + 1024]
                x = tr_X[sel].to(device); yb = tr_Y[sel]; p = tr_pos[sel]
                with torch.no_grad():
                    pl = run_model(me, mo, args.mode, x, p)
                    gp = torch.from_numpy(compute_pattern_labels_batch(
                        yb.numpy(), p.numpy(),
                        pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                    ).to(device)
                    legal = pat_labels_to_cell_labels(gp, pattern_to_cell)
                calibrated = s * pl + b
                cell_logits = patterns_to_cell_logsumexp(calibrated, pattern_to_cell)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    cell_logits, legal, pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item(); total_batches += 1
            del tr_X, tr_Y, tr_pos
        avg = total_loss / max(total_batches, 1)
        print(f"  Epoch {epoch}: loss={avg:.5f}  "
              f"s mean/std: {s.mean().item():.3f}/{s.std().item():.3f}  "
              f"b mean/std: {b.mean().item():.3f}/{b.std().item():.3f}", flush=True)

    print("\nAfter calibration (s*logit + b, max-per-cell):")
    _print_results(eval_on_chunk(eval_path, use_calibration=True), "calibrated")

    # Save calibration params
    base = os.path.splitext(os.path.basename(args.ckpt))[0]
    save_path = os.path.join(args.output_dir, "pattern_detector_checkpoints",
                             f"calib_{base}.pt")
    torch.save({'scale': s.detach().cpu(), 'bias': b.detach().cpu(),
                'source_ckpt': args.ckpt}, save_path)
    print(f"\nSaved calibration to {save_path}")
