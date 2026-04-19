"""Train 960 independent small networks, one per pattern.

Each pattern gets its own (60 -> H_small -> 1) network. No hidden-unit
sharing across patterns — if hidden-unit competition is the bottleneck,
this should beat the shared-hidden model.

Implementation: a single batched forward pass with per-pattern weights,
equivalent to 960 parallel networks:
    h[b, j, :] = ReLU( x[b, :] @ W1[:, j, :] + b1[j, :] )       (B, 960, H_small)
    y[b, j]    = sum_k h[b, j, k] * W2[j, k] + b2[j]            (B, 960)

Even/odd split (two sets of 960 networks). Pattern BCE + pos_weight.

Usage:
    python train_independent_patterns.py --hidden 16 --epochs 3 --pos-weight 5
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
    compute_pattern_labels_batch,
    pat_labels_to_cell_labels, _get_cell_pat_index,
)


class IndependentPatterns(nn.Module):
    """960 independent (D -> H -> 1) networks, computed in parallel."""
    def __init__(self, input_dim, hidden_small, n_patterns=960):
        super().__init__()
        self.D = input_dim
        self.H = hidden_small
        self.P = n_patterns
        # W1: (D, P, H), b1: (P, H), W2: (P, H), b2: (P,)
        # Kaiming-style init
        self.W1 = nn.Parameter(torch.randn(input_dim, n_patterns, hidden_small)
                                * (2.0 / input_dim) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(n_patterns, hidden_small))
        self.W2 = nn.Parameter(torch.randn(n_patterns, hidden_small)
                                * (2.0 / hidden_small) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(n_patterns))

    def forward(self, x):
        # x: (B, D). Contract over D -> (B, P, H)
        h = torch.einsum('bd,dph->bph', x, self.W1) + self.b1  # (B, P, H)
        h = torch.relu(h)
        # Per-pattern: y[b, j] = sum_k h[b, j, k] * W2[j, k]
        y = (h * self.W2).sum(dim=-1) + self.b2  # (B, P)
        return y


def prob_or_scores(pat_logits, idx, mask):
    log1m = -nn.functional.softplus(pat_logits)
    gathered = log1m[:, idx]
    gathered = gathered.masked_fill(~mask, 0.0)
    return -gathered.sum(dim=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=16,
                        help="Hidden size per small network")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--features", default="when",
                        choices=["when", "played+when", "when+even", "played+even",
                                 "all", "mine_signed", "signed_parity"])
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    _feat_cols = {
        "when":        list(range(N_MOVES, 2 * N_MOVES)),
        "played+when": list(range(0, 2 * N_MOVES)),
        "when+even":   list(range(N_MOVES, 3 * N_MOVES)),
        "played+even": list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)),
        "all":         list(range(0, 3 * N_MOVES)),
    }
    from train_pattern_simple import to_signed_parity_input, to_mine_signed_input
    if args.features in _feat_cols:
        feature_cols = _feat_cols[args.features]; feature_fn = None
        input_dim = len(feature_cols)
    elif args.features == "signed_parity":
        feature_cols = None; feature_fn = lambda X, Y, p: to_signed_parity_input(X)
        input_dim = N_MOVES
    elif args.features == "mine_signed":
        feature_cols = None; feature_fn = lambda X, Y, p: to_mine_signed_input(Y, p)
        input_dim = N_MOVES
    print(f"Features: {args.features} ({input_dim}-d), H_small={args.hidden}, pw={args.pos_weight}")

    model_even = IndependentPatterns(input_dim, args.hidden).to(device)
    model_odd = IndependentPatterns(input_dim, args.hidden).to(device)
    n_params = sum(p.numel() for p in model_even.parameters())
    print(f"  params per model: {n_params:,}  (2x for even+odd)")

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device)
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)
    pw = torch.tensor([args.pos_weight], device=device)

    params = list(model_even.parameters()) + list(model_odd.parameters())
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

    def prep_batch(tr_X, tr_Y, tr_pos):
        if feature_cols is not None:
            return tr_X[:, feature_cols]
        return feature_fn(tr_X, tr_Y, tr_pos)

    def eval_pass():
        model_even.eval(); model_odd.eval()
        totals_pat = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        agg = {a: {n: {'c': 0, 't': 0} for n in (1, 3, 5, 10)}
               for a in ('max', 'logsumexp', 'prob_or')}
        X, Y, pos_ = _load_features(eval_path)
        X = prep_batch(X, Y, pos_)
        n = min(len(X), 49 * 10000)
        rng = np.random.RandomState(0)
        si = np.sort(rng.choice(len(X), n, replace=False))
        X, Y, pos_ = X[si], Y[si], pos_[si]
        with torch.no_grad():
            for i in range(0, n, 1024):
                x = X[i:i+1024].to(device); yb = Y[i:i+1024]; p = pos_[i:i+1024]
                em = (p % 2 == 0); om = ~em
                pl = torch.zeros(len(x), 960, device=device)
                if em.any(): pl[em] = model_even(x[em])
                if om.any(): pl[om] = model_odd(x[om])
                gp = torch.from_numpy(compute_pattern_labels_batch(
                    yb.numpy(), p.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)).to(device)
                pred = (pl > 0); gt = (gp > 0.5)
                totals_pat['tp'] += (pred & gt).sum().item()
                totals_pat['fp'] += (pred & ~gt).sum().item()
                totals_pat['fn'] += (~pred & gt).sum().item()
                totals_pat['tn'] += (~pred & ~gt).sum().item()
                legal = pat_labels_to_cell_labels(gp, pattern_to_cell)
                gathered_ninf = pl[:, idx].masked_fill(~mask, float('-inf'))
                scores = {
                    'max':       gathered_ninf.max(dim=-1).values,
                    'logsumexp': torch.logsumexp(gathered_ninf, dim=-1),
                    'prob_or':   prob_or_scores(pl, idx, mask),
                }
                gl = (legal > 0.5).cpu().numpy()
                for name, s in scores.items():
                    cs = s.cpu().numpy()
                    for b in range(cs.shape[0]):
                        ls = set(np.where(gl[b])[0].tolist()); K = len(ls)
                        if K == 0: continue
                        r = np.argsort(-cs[b])
                        for k in (1, 3, 5, 10):
                            kk = min(k, K)
                            agg[name][k]['c'] += len(set(r[:kk].tolist()) & ls)
                            agg[name][k]['t'] += kk
        return totals_pat, agg

    def _print(pat_t, agg_t, label):
        tot = sum(pat_t.values())
        acc = (pat_t['tp'] + pat_t['tn']) / max(tot, 1)
        rec = pat_t['tp'] / max(pat_t['tp'] + pat_t['fn'], 1)
        pre = pat_t['tp'] / max(pat_t['tp'] + pat_t['fp'], 1)
        print(f"  {label}: acc={acc:.4%} recall={rec:.4%} prec={pre:.4%}")
        for name in ('max', 'logsumexp', 'prob_or'):
            row = [name]
            for k in (1, 3, 5, 10):
                d = agg_t[name][k]
                row.append(f"{d['c']/max(d['t'],1):.4%}")
            print(f"    {row[0]:10s} top-1={row[1]} top-3={row[2]} top-5={row[3]} top-10={row[4]}")

    print(f"\nTraining: {len(train_paths)} chunks x {args.epochs} epochs, lr={args.lr}")
    for epoch in range(1, args.epochs + 1):
        model_even.train(); model_odd.train()
        rng = np.random.RandomState(epoch)
        order = rng.permutation(len(train_paths))
        total_loss = 0.0; total_batches = 0
        for ci in order:
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            tr_X = prep_batch(tr_X, tr_Y, tr_pos)
            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), 1024):
                sel = perm[i:i + 1024]
                x = tr_X[sel].to(device); yb = tr_Y[sel]; p = tr_pos[sel]
                with torch.no_grad():
                    gp = torch.from_numpy(compute_pattern_labels_batch(
                        yb.numpy(), p.numpy(),
                        pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)).to(device)
                em = (p % 2 == 0); om = ~em
                loss = torch.tensor(0.0, device=device)
                for msk, m in [(em, model_even), (om, model_odd)]:
                    if not msk.any(): continue
                    pl = m(x[msk])
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(
                        pl, gp[msk], pos_weight=pw)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item(); total_batches += 1
            del tr_X, tr_Y, tr_pos
        avg = total_loss / max(total_batches, 1)
        pat_t, agg_t = eval_pass()
        print(f"\nEpoch {epoch}: loss={avg:.5f}", flush=True)
        _print(pat_t, agg_t, "independent")

    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    feat_tag = "" if args.features == "when" else f"_{args.features.replace('+', '')}"
    save_path = os.path.join(save_dir,
        f"indep_Hsmall{args.hidden}_pw{int(args.pos_weight)}{feat_tag}.pt")
    torch.save({'even': model_even.state_dict(), 'odd': model_odd.state_dict(),
                'input_dim': input_dim, 'hidden_small': args.hidden}, save_path)
    print(f"\nSaved to {save_path}")
