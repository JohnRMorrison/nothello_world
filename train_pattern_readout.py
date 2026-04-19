"""Fresh linear readout from frozen hidden → 960 patterns.

Hypothesis: the model's hidden layer encodes the board state well (probe
~98%), but the existing output layer does a bad job translating that
board state into pattern firings — especially for rare firings.

This script:
  1. Loads a trained checkpoint.
  2. Freezes the first layer (60 → H). The ReLU'd hidden is then
     the representation the probe decodes at ~98%.
  3. Re-initializes and trains a new Linear(H, 960) with pos_weight,
     using BCE on pattern labels.
  4. Saves the new output layer, reports pattern acc/recall/prec and
     top-1/3/5/10 legal under max, logsumexp, prob_or.

Usage:
    python train_pattern_readout.py --ckpt pattern_simple_direct_H512.pt \
        --mode direct --hidden 512 --epochs 3 --pos-weight 10
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
    DirectMLP, compute_pattern_labels_batch,
    pat_labels_to_cell_labels, _get_cell_pat_index,
)


def hidden_activation(model, x):
    """ReLU'd hidden activation from a DirectMLP."""
    return torch.relu(model.net[0](x))


def score_aggregators(pat_logits, idx, mask, legal, agg_totals):
    """Update top-N totals across (max, logsumexp, prob_or) aggregators."""
    gathered = pat_logits[:, idx]
    gathered_ninf = gathered.masked_fill(~mask, float('-inf'))

    scores = {
        'max':       gathered_ninf.max(dim=-1).values,
        'logsumexp': torch.logsumexp(gathered_ninf, dim=-1),
        'prob_or':   -(-nn.functional.softplus(pat_logits))[:, idx].masked_fill(~mask, 0.0).sum(dim=-1),
    }
    gl = (legal > 0.5).cpu().numpy()
    for name, s in scores.items():
        cs = s.cpu().numpy()
        for b in range(cs.shape[0]):
            ls = set(np.where(gl[b])[0].tolist()); K = len(ls)
            if K == 0: continue
            r = np.argsort(-cs[b])
            for nn_ in (1, 3, 5, 10):
                k = min(nn_, K)
                agg_totals[name][nn_]['c'] += len(set(r[:k].tolist()) & ls)
                agg_totals[name][nn_]['t'] += k


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["direct", "randproj"],
                        help="Only direct/randproj supported (uses DirectMLP hidden).")
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--loss", choices=["bce", "mse"], default="bce",
                        help="bce: BCE with logits (unbounded). mse: MSE on sigmoid output (bounded 0/1 targets, uniform scales).")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    feature_cols = list(range(N_MOVES, 2 * N_MOVES))

    ckpt = torch.load(args.ckpt, map_location=device)
    n_patterns = ckpt.get('n_patterns', 960)
    me = DirectMLP(N_MOVES, args.hidden, n_patterns).to(device)
    mo = DirectMLP(N_MOVES, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()
    # Freeze everything in the source model
    for p in me.parameters(): p.requires_grad = False
    for p in mo.parameters(): p.requires_grad = False
    print(f"Loaded {args.ckpt} (pat_acc={ckpt.get('best_pat_acc', '?')})")

    # Fresh output readouts: H -> 960
    readout_even = nn.Linear(args.hidden, n_patterns).to(device)
    readout_odd = nn.Linear(args.hidden, n_patterns).to(device)

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device)
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    pw = torch.tensor([args.pos_weight], device=device)

    params = list(readout_even.parameters()) + list(readout_odd.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    if len(chunk_files) == 1:
        eval_path = chunk_files[0]; train_paths = chunk_files
        print(f"Single-chunk mode: train+eval on {chunk_files[0]}")
    else:
        eval_path = chunk_files[-1]; train_paths = chunk_files[:-1]

    def eval_pass():
        readout_even.eval(); readout_odd.eval()
        totals_pat = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        agg_totals = {'max': {n: {'c': 0, 't': 0} for n in (1, 3, 5, 10)},
                      'logsumexp': {n: {'c': 0, 't': 0} for n in (1, 3, 5, 10)},
                      'prob_or': {n: {'c': 0, 't': 0} for n in (1, 3, 5, 10)}}
        X, Y, pos = _load_features(eval_path)
        X = X[:, feature_cols]
        n = min(len(X), 49 * 10000)
        rng = np.random.RandomState(0)
        si = np.sort(rng.choice(len(X), n, replace=False))
        X, Y, pos = X[si], Y[si], pos[si]
        with torch.no_grad():
            for i in range(0, n, 1024):
                x = X[i:i+1024].to(device); yb = Y[i:i+1024]; p = pos[i:i+1024]
                em = (p % 2 == 0); om = ~em
                pl = torch.zeros(len(x), n_patterns, device=device)
                if em.any():
                    h = hidden_activation(me, x[em])
                    pl[em] = readout_even(h)
                if om.any():
                    h = hidden_activation(mo, x[om])
                    pl[om] = readout_odd(h)
                gp = torch.from_numpy(compute_pattern_labels_batch(
                    yb.numpy(), p.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                ).to(device)
                pred = (pl > 0)
                gt = (gp > 0.5)
                totals_pat['tp'] += (pred & gt).sum().item()
                totals_pat['fp'] += (pred & ~gt).sum().item()
                totals_pat['fn'] += (~pred & gt).sum().item()
                totals_pat['tn'] += (~pred & ~gt).sum().item()
                legal = pat_labels_to_cell_labels(gp, pattern_to_cell)
                score_aggregators(pl, idx, mask, legal, agg_totals)
        return totals_pat, agg_totals

    def _print_pat(t):
        tot = sum(t.values())
        acc = (t['tp'] + t['tn']) / max(tot, 1)
        rec = t['tp'] / max(t['tp'] + t['fn'], 1)
        pre = t['tp'] / max(t['tp'] + t['fp'], 1)
        print(f"  pattern-level: acc={acc:.4%} recall={rec:.4%} prec={pre:.4%}")

    def _print_agg(ag):
        for name in ('max', 'logsumexp', 'prob_or'):
            row = [name]
            for nn_ in (1, 3, 5, 10):
                d = ag[name][nn_]
                row.append(f"{d['c']/max(d['t'],1):.4%}")
            print(f"    {row[0]:12s} top-1={row[1]}  top-3={row[2]}  top-5={row[3]}  top-10={row[4]}")

    # Training
    print(f"\nTraining: {len(train_paths)} chunks x {args.epochs} epochs, lr={args.lr}, pw={args.pos_weight}")
    for epoch in range(1, args.epochs + 1):
        readout_even.train(); readout_odd.train()
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
                    gp = torch.from_numpy(compute_pattern_labels_batch(
                        yb.numpy(), p.numpy(),
                        pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                    ).to(device)
                em = (p % 2 == 0); om = ~em
                loss = torch.tensor(0.0, device=device)
                for msk, r_model in [(em, readout_even), (om, readout_odd)]:
                    if not msk.any(): continue
                    with torch.no_grad():
                        src = me if r_model is readout_even else mo
                        h = hidden_activation(src, x[msk])
                    pl = r_model(h)
                    if args.loss == "bce":
                        loss = loss + nn.functional.binary_cross_entropy_with_logits(
                            pl, gp[msk], pos_weight=pw)
                    else:
                        # MSE on sigmoid output; apply pos_weight to the positives
                        # to counter class imbalance (so rare firings still train).
                        prob = torch.sigmoid(pl)
                        sq = (prob - gp[msk]) ** 2
                        # weight sq by pos_weight on positives
                        w = torch.where(gp[msk] > 0.5, pw, torch.ones_like(sq))
                        loss = loss + (w * sq).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item(); total_batches += 1
            del tr_X, tr_Y, tr_pos
        avg = total_loss / max(total_batches, 1)
        pat_t, agg_t = eval_pass()
        print(f"\nEpoch {epoch}: loss={avg:.5f}", flush=True)
        _print_pat(pat_t)
        _print_agg(agg_t)

    # Save
    base = os.path.splitext(os.path.basename(args.ckpt))[0]
    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,
        f"readout_{base}_{args.loss}_pw{int(args.pos_weight)}.pt")
    torch.save({
        'even': readout_even.state_dict(),
        'odd': readout_odd.state_dict(),
        'source_ckpt': args.ckpt,
        'pos_weight': args.pos_weight,
    }, save_path)
    print(f"\nSaved readout to {save_path}")
