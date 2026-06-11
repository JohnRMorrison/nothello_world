"""Compare different pattern->cell aggregators on a frozen checkpoint.

Pure post-hoc: loads model once, computes pattern logits, then scores
top-N legal under several aggregation rules. No training needed.

Aggregators:
  max          — current default (argmax over patterns per cell)
  mean         — average over the ~16 patterns targeting a cell
  median       — median over patterns
  logsumexp    — smooth max (temperature 1.0)
  topk_mean-K  — average of top-K patterns (K=2,3,5)
  prob_or      — 1 - prod(1 - sigmoid(logit)) over patterns

Usage:
    python compare_aggregators.py --ckpt pattern_simple_direct_H512.pt \
        --mode direct --hidden 512
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES,
)
from hand_crafted_flanking import (
    enumerate_flanking_patterns,
    enumerate_flanking_patterns_color_specific,
    MOVE_TO_IDX,
)
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import (
    DirectMLP, EndToEndMLP, TwoStageMLP, compute_pattern_labels_batch,
    pat_labels_to_cell_labels, _get_cell_pat_index,
)


def _gather(pat_logits, idx, mask, fill=float('-inf')):
    gathered = pat_logits[:, idx]
    return gathered.masked_fill(~mask, fill)


def agg_max(pat_logits, idx, mask):
    return _gather(pat_logits, idx, mask, float('-inf')).max(dim=-1).values


def agg_mean(pat_logits, idx, mask):
    gathered = _gather(pat_logits, idx, mask, 0.0)
    counts = mask.sum(dim=-1).clamp(min=1).to(gathered.dtype)
    return gathered.sum(dim=-1) / counts


def agg_median(pat_logits, idx, mask):
    # Set invalid to -inf so valid values dominate the median for cells with
    # an odd mix. For even cells this is slightly biased low on invalid-heavy
    # cells — use sort+pick instead.
    gathered = _gather(pat_logits, idx, mask, float('-inf'))
    # For each cell, take median of the valid entries only
    out = torch.zeros(gathered.shape[:2], dtype=gathered.dtype, device=gathered.device)
    counts = mask.sum(dim=-1)
    sorted_, _ = gathered.sort(dim=-1, descending=True)
    # median index per cell
    med_idx = (counts - 1) // 2
    for c in range(gathered.shape[1]):
        out[:, c] = sorted_[:, c, med_idx[c]]
    return out


def agg_logsumexp(pat_logits, idx, mask):
    gathered = _gather(pat_logits, idx, mask, float('-inf'))
    return torch.logsumexp(gathered, dim=-1)


def agg_topk_mean(k):
    def _agg(pat_logits, idx, mask):
        gathered = _gather(pat_logits, idx, mask, float('-inf'))
        # top-K across patterns
        topk_vals, _ = gathered.topk(min(k, gathered.shape[-1]), dim=-1)
        # Average of valid top-K (ignore -inf from padding)
        valid = torch.isfinite(topk_vals)
        safe = torch.where(valid, topk_vals, torch.zeros_like(topk_vals))
        counts = valid.sum(dim=-1).clamp(min=1).to(safe.dtype)
        return safe.sum(dim=-1) / counts
    return _agg


def agg_prob_or(pat_logits, idx, mask):
    # P(cell fires) = 1 - prod_j (1 - sigmoid(logit_j))
    # Log-space: sum_j log(1 - sigmoid(x)) = sum_j -softplus(x) = log(1 - P)
    # Score = -log(1 - P) is monotonic in P, so rank by that.
    log1m = -torch.nn.functional.softplus(pat_logits)
    gathered = log1m[:, idx]
    gathered = gathered.masked_fill(~mask, 0.0)   # log(1 - 0) = 0 for padding
    return -gathered.sum(dim=-1)                   # -log(1-P): higher = more likely legal


def agg_count_above_zero(pat_logits, idx, mask):
    # How many patterns per cell have logit > 0 (hard threshold).
    counts = (pat_logits > 0).to(pat_logits.dtype)
    gathered = counts[:, idx]
    gathered = gathered.masked_fill(~mask, 0.0)
    return gathered.sum(dim=-1)


def agg_sum_sigmoid(pat_logits, idx, mask):
    # Soft count: sum of per-pattern probabilities (0..max_per_cell).
    probs = torch.sigmoid(pat_logits)
    gathered = probs[:, idx]
    gathered = gathered.masked_fill(~mask, 0.0)
    return gathered.sum(dim=-1)


def agg_ensemble_zscore(pat_logits, idx, mask):
    # Combine max + logsumexp + prob_or after per-position z-score.
    # Argmax should favor cells that rank high under multiple aggregators.
    def zscore(x):
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (x - mu) / sd
    s1 = agg_max(pat_logits, idx, mask)
    s2 = agg_logsumexp(pat_logits, idx, mask)
    s3 = agg_prob_or(pat_logits, idx, mask)
    return zscore(s1) + zscore(s2) + zscore(s3)


def score_from_cs(cs, gl, top_ns=(1, 3, 5, 10)):
    """Score pre-computed cell scores cs (B, 60) against legal mask gl (B, 60 bool)."""
    results = {n: {'c': 0, 't': 0} for n in top_ns}
    for b in range(cs.shape[0]):
        ls = set(np.where(gl[b])[0].tolist()); K = len(ls)
        if K == 0: continue
        r = np.argsort(-cs[b])
        for n in top_ns:
            k = min(n, K)
            results[n]['c'] += len(set(r[:k].tolist()) & ls)
            results[n]['t'] += k
    return results


def score(agg, pat_logits, idx, mask, legal, top_ns=(1, 3, 5, 10)):
    cs = agg(pat_logits, idx, mask).cpu().numpy()
    gl = (legal > 0.5).cpu().numpy()
    return score_from_cs(cs, gl, top_ns)


def accumulate(total, new):
    for n in total:
        total[n]['c'] += new[n]['c']
        total[n]['t'] += new[n]['t']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["direct", "emergent", "e2e", "two-stage", "randproj"])
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--features", default=None,
                        choices=[None, "when", "played", "played+when", "when+even",
                                 "played+even", "all", "board_state",
                                 "signed_parity", "mine_signed", "color_split",
                                 "played+halfmask", "played+bit",
                                 "move_grid", "move_grid_onehot"],
                        help="Feature set used during training (inferred from "
                             "checkpoint's input_dim if omitted).")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    parser.add_argument("--save-per-position", default=None,
                        help="If set, write per-position cell scores + legal mask + "
                             "turn to this .npz path for downstream error-distribution "
                             "analysis. Saves scores for the aggregators listed in "
                             "--save-aggregators.")
    parser.add_argument("--save-aggregators", default="max,logsumexp,prob_or",
                        help="Comma-separated aggregator names whose per-cell scores "
                             "to save (only when --save-per-position is set).")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    n_patterns = ckpt.get('n_patterns', 960)
    input_dim = ckpt.get('input_dim', N_MOVES)
    # Color-specific: 1920 outputs (960 black-mover + 960 white-mover), single MLP.
    color_specific = ckpt.get('color_specific', False) or n_patterns == 1920
    if color_specific:
        print(f"  color-specific checkpoint detected: n_patterns={n_patterns}, "
              f"single model, parity slices to 960 relevant patterns per position")

    # Infer features from input_dim if --features not specified
    if args.features is None:
        name = os.path.basename(args.ckpt)
        if "board_state" in name: args.features = "board_state"
        elif "move_grid_onehot" in name: args.features = "move_grid_onehot"
        elif "move_grid" in name: args.features = "move_grid"
        elif "wheneven" in name: args.features = "when+even"
        elif "playedeven" in name: args.features = "played+even"
        elif "color_split" in name: args.features = "color_split"
        elif "halfmask" in name: args.features = "played+halfmask"
        elif "played_bit" in name or "playedbit" in name: args.features = "played+bit"
        elif "signed_parity" in name: args.features = "signed_parity"
        elif "mine_signed" in name: args.features = "mine_signed"
        elif input_dim == 62: args.features = "played+bit"
        elif input_dim == 120: args.features = "when+even"
        elif input_dim == 180: args.features = "all"
        elif input_dim == 192: args.features = "board_state"
        elif input_dim == 3600: args.features = "move_grid"
        elif input_dim == 10800: args.features = "move_grid_onehot"
        else: args.features = "when"
    print(f"Features: {args.features} (input_dim={input_dim})")

    Cls = {"direct": DirectMLP, "randproj": DirectMLP,
           "two-stage": TwoStageMLP,
           "emergent": EndToEndMLP, "e2e": EndToEndMLP}[args.mode]
    me = Cls(input_dim, args.hidden, n_patterns).to(device)
    mo = Cls(input_dim, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()
    print(f"Loaded {args.ckpt} (pat_acc={ckpt.get('best_pat_acc', '?')})")

    if color_specific:
        # Patterns are [first 960 black-mover, next 960 white-mover] in the
        # canonical 960 ordering. For aggregation, we slice the 1920 outputs
        # to the 960 matching the current mover and then use the standard
        # pattern_to_cell / idx / mask mapping on the 960 standard patterns.
        patterns = enumerate_flanking_patterns()
        pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
        pattern_to_cell = torch.tensor(
            [MOVE_TO_IDX[p['target']] for p in patterns],
            dtype=torch.long, device=device)
    else:
        patterns = enumerate_flanking_patterns()
        pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
        pattern_to_cell = torch.tensor(
            [MOVE_TO_IDX[p['target']] for p in patterns],
            dtype=torch.long, device=device)
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.startswith("chunk_0") and f.endswith(".npz")
                         and "_patterns" not in f and "_when60" not in f)
    eval_path = chunk_files[-1]
    print(f"Eval: {os.path.basename(eval_path)} (random sample)")

    from train_pattern_simple import (to_signed_parity_input, to_mine_signed_input,
                                       to_board_state_input, to_color_split_input,
                                       to_played_halfmask_input, to_played_bit_input,
                                       to_move_grid_input, to_move_grid_onehot_input)
    _feat_cols = {
        "when":        list(range(N_MOVES, 2 * N_MOVES)),
        "played":      list(range(0, N_MOVES)),
        "played+when": list(range(0, 2 * N_MOVES)),
        "when+even":   list(range(N_MOVES, 3 * N_MOVES)),
        "played+even": list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)),
        "all":         list(range(0, 3 * N_MOVES)),
    }
    X, Y, pos = _load_features(eval_path)
    # Subsample FIRST so feature expansions (e.g., move_grid 180->3600) don't
    # blow memory when applied to the full chunk.
    n = min(len(X), 49 * 10000)
    rng = np.random.RandomState(0)
    si = np.sort(rng.choice(len(X), n, replace=False))
    X, Y, pos = X[si], Y[si], pos[si]
    if args.features in _feat_cols:
        X = X[:, _feat_cols[args.features]]
    elif args.features == "signed_parity":
        X = to_signed_parity_input(X)
    elif args.features == "mine_signed":
        X = to_mine_signed_input(Y, pos)
    elif args.features == "board_state":
        X = to_board_state_input(Y, pos)
    elif args.features == "color_split":
        X = to_color_split_input(X)
    elif args.features == "played+halfmask":
        X = to_played_halfmask_input(X)
    elif args.features == "played+bit":
        X = to_played_bit_input(X)
    elif args.features == "move_grid":
        X = to_move_grid_input(X)
    elif args.features == "move_grid_onehot":
        X = to_move_grid_onehot_input(X)

    aggregators = {
        'max':         agg_max,
        'mean':        agg_mean,
        'median':      agg_median,
        'logsumexp':   agg_logsumexp,
        'topk_mean-2': agg_topk_mean(2),
        'topk_mean-3': agg_topk_mean(3),
        'topk_mean-5': agg_topk_mean(5),
        'prob_or':     agg_prob_or,
        'count_pos':   agg_count_above_zero,
        'sum_sigmoid': agg_sum_sigmoid,
        'ensemble_z':  agg_ensemble_zscore,
    }
    totals = {name: {n: {'c': 0, 't': 0} for n in (1, 3, 5, 10)}
              for name in aggregators}

    save_per_pos = args.save_per_position is not None
    if save_per_pos:
        save_names = [s.strip() for s in args.save_aggregators.split(',')]
        missing = [s for s in save_names if s not in aggregators]
        if missing:
            raise ValueError(f"Unknown aggregator(s) for --save-aggregators: {missing}")
        saved_scores = {name: [] for name in save_names}
        saved_pos, saved_legal_mask = [], []
        print(f"Will save per-position data for aggregators: {save_names}")

    batch = 1024
    with torch.no_grad():
        for i in range(0, n, batch):
            x = X[i:i + batch].to(device)
            yb = Y[i:i + batch]
            p = pos[i:i + batch]
            em = (p % 2 == 0); om = ~em
            if color_specific:
                # Single model, 1920 outputs. Per-position parity slice:
                # om (pos%2==1, black to move) -> patterns[:960] (mover=+1)
                # em (pos%2==0, white to move) -> patterns[960:] (mover=-1)
                full = me(x)                                 # (B, 1920)
                pl = torch.zeros(len(x), 960, device=device)
                if om.any():
                    pl[om] = full[om, :960]
                if em.any():
                    pl[em] = full[em, 960:]
            elif args.mode in ("direct", "randproj"):
                pl = torch.zeros(len(x), 960, device=device)
                if em.any(): pl[em] = me(x[em])
                if om.any(): pl[om] = mo(x[om])
            else:
                pl = torch.zeros(len(x), 960, device=device)
                for msk, m in [(em, me), (om, mo)]:
                    if not msk.any(): continue
                    logits, _ = m(x[msk], p[msk])
                    pl[msk] = logits
            gp = torch.from_numpy(compute_pattern_labels_batch(
                yb.numpy(), p.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
            ).to(device)
            legal = pat_labels_to_cell_labels(gp, pattern_to_cell)
            gl_np = (legal > 0.5).cpu().numpy()
            if save_per_pos:
                saved_pos.append(p.numpy().astype(np.int8))
                saved_legal_mask.append(gl_np)
            for name, agg in aggregators.items():
                cs = agg(pl, idx, mask).cpu().numpy()
                accumulate(totals[name], score_from_cs(cs, gl_np))
                if save_per_pos and name in save_names:
                    saved_scores[name].append(cs.astype(np.float32))

    print(f"\n{'aggregator':16s} {'top-1':>10} {'top-3':>10} {'top-5':>10} {'top-10':>10}")
    print("-" * 60)
    for name in aggregators:
        row = [name]
        for nn_ in (1, 3, 5, 10):
            d = totals[name][nn_]
            row.append(f"{d['c']/max(d['t'],1):.4%}")
        print(f"{row[0]:16s} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}")

    if save_per_pos:
        out = {
            'pos': np.concatenate(saved_pos),
            'legal_mask': np.concatenate(saved_legal_mask),
        }
        for name in save_names:
            out[f'scores_{name}'] = np.concatenate(saved_scores[name])
        os.makedirs(os.path.dirname(args.save_per_position) or '.', exist_ok=True)
        np.savez_compressed(args.save_per_position, **out)
        print(f"\nSaved per-position data: {args.save_per_position}")
        print(f"  positions: {len(out['pos'])}, aggregators saved: {save_names}")
