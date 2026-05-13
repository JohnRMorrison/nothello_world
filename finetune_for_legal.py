"""Finetune a trained pattern detector with a direct legal-move loss.

Loads a trained MLP (60 -> H -> 960), then retrains with a loss that
aggregates the 960 pattern logits to 60 cell scores via logsumexp and
does BCE against the 60-d legal-move labels. Keeps the same output
dimensionality (960 patterns) and pattern->cell mapping — just adjusts
weights so the aggregation produces better argmax legal predictions.

Two modes:
  --finetune output   Train only the Linear(H, 960) output layer
  --finetune full     Train all layers (60->H and H->960)

Usage:
    python finetune_for_legal.py --ckpt pattern_simple_direct_H512.pt \
        --mode direct --hidden 512 --finetune output --epochs 3
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
    patterns_to_cell_logsumexp, pat_labels_to_cell_labels, _get_cell_pat_index,
    listwise_cell_ce,
)


def prob_or_scores(pat_logits, idx, mask):
    log1m = -nn.functional.softplus(pat_logits)
    gathered = log1m[:, idx]
    gathered = gathered.masked_fill(~mask, 0.0)
    return -gathered.sum(dim=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mode", choices=["direct", "randproj"], default="direct")
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=5.0,
                        help="pos_weight on cell BCE (legal cells ~16%)")
    parser.add_argument("--finetune", choices=["output", "full"], default="output")
    parser.add_argument("--loss", choices=["bce", "listwise"], default="bce",
                        help="bce: cell-aggregated BCE (legacy). listwise: softmax CE over "
                             "60 cells against uniform-over-legal target — directly "
                             "optimizes recall@K.")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    from train_pattern_simple import (
        to_signed_parity_input, to_mine_signed_input, to_board_state_input,
        to_color_split_input, to_played_halfmask_input, to_played_bit_input,
        to_move_grid_input, to_move_grid_onehot_input,
    )
    _feat_cols_map = {
        "when":        list(range(N_MOVES, 2 * N_MOVES)),
        "played":      list(range(0, N_MOVES)),
        "played+when": list(range(0, 2 * N_MOVES)),
        "when+even":   list(range(N_MOVES, 3 * N_MOVES)),
        "played+even": list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)),
        "all":         list(range(0, 3 * N_MOVES)),
    }

    ckpt = torch.load(args.ckpt, map_location=device)
    n_patterns = ckpt.get('n_patterns', 960)
    input_dim = ckpt.get('input_dim', N_MOVES)

    # Infer feature preprocessing from ckpt filename / input_dim
    name = os.path.basename(args.ckpt)
    feature_cols, feature_fn = None, None
    if "wheneven" in name: feature_cols = _feat_cols_map["when+even"]
    elif "playedeven" in name: feature_cols = _feat_cols_map["played+even"]
    elif "color_split" in name: feature_fn = lambda X, Y, p: to_color_split_input(X)
    elif "signed_parity" in name: feature_fn = lambda X, Y, p: to_signed_parity_input(X)
    elif "move_grid_onehot" in name: feature_fn = lambda X, Y, p: to_move_grid_onehot_input(X)
    elif "move_grid" in name: feature_fn = lambda X, Y, p: to_move_grid_input(X)
    elif input_dim == 120: feature_cols = _feat_cols_map["when+even"]
    elif input_dim == 180: feature_cols = _feat_cols_map["all"]
    else: feature_cols = _feat_cols_map["when"]
    print(f"input_dim={input_dim}, feature inferred from {name}")

    me = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    mo = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even'])
    mo.load_state_dict(ckpt['odd'])
    print(f"Loaded {args.ckpt} (pat_acc={ckpt.get('best_pat_acc', '?')})")

    # Decide what to train
    trainable = []
    if args.finetune == "output":
        for m in (me, mo):
            for p in m.parameters(): p.requires_grad = False
            for p in m.net[2].parameters(): p.requires_grad = True
            trainable += list(m.net[2].parameters())
        print("Finetuning only Linear(H, 960) output layer")
    else:
        for m in (me, mo):
            for p in m.parameters(): p.requires_grad = True
            trainable += list(m.parameters())
        print("Finetuning all layers")

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device)
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)
    pw = torch.tensor([args.pos_weight], device=device)

    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    if len(chunk_files) == 1:
        eval_path = chunk_files[0]; train_paths = chunk_files
        print(f"Single-chunk mode: train+eval on {chunk_files[0]}")
    else:
        eval_path = chunk_files[-1]; train_paths = chunk_files[:-1]

    def eval_topk():
        me.eval(); mo.eval()
        results = {agg: {n: {'c': 0, 't': 0} for n in (1, 3, 5, 10)}
                   for agg in ('max', 'logsumexp', 'prob_or')}
        X, Y, pos_ = _load_features(eval_path)
        if feature_cols is not None:
            X = X[:, feature_cols]
        elif feature_fn is not None:
            X = feature_fn(X, Y, pos_)
        n = min(len(X), 49 * 10000)
        rng = np.random.RandomState(0)
        si = np.sort(rng.choice(len(X), n, replace=False))
        X, Y, pos_ = X[si], Y[si], pos_[si]
        with torch.no_grad():
            for i in range(0, n, 1024):
                x = X[i:i+1024].to(device); yb = Y[i:i+1024]; p = pos_[i:i+1024]
                em = (p % 2 == 0); om = ~em
                pl = torch.zeros(len(x), 960, device=device)
                if em.any(): pl[em] = me(x[em])
                if om.any(): pl[om] = mo(x[om])
                gp = torch.from_numpy(compute_pattern_labels_batch(
                    yb.numpy(), p.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                ).to(device)
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
                            results[name][k]['c'] += len(set(r[:kk].tolist()) & ls)
                            results[name][k]['t'] += kk
        return results

    def _print_results(res, label):
        print(f"  {label}")
        for name in ('max', 'logsumexp', 'prob_or'):
            row = [name]
            for k in (1, 3, 5, 10):
                d = res[name][k]
                row.append(f"{d['c']/max(d['t'],1):.4%}")
            print(f"    {row[0]:12s} top-1={row[1]}  top-3={row[2]}  top-5={row[3]}  top-10={row[4]}")

    print("\nBefore finetune:")
    _print_results(eval_topk(), "baseline (loaded ckpt)")

    for epoch in range(1, args.epochs + 1):
        me.train(); mo.train()
        rng = np.random.RandomState(epoch)
        order = rng.permutation(len(train_paths))
        total_loss = 0.0; total_batches = 0
        for ci in order:
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            if feature_cols is not None:
                tr_X = tr_X[:, feature_cols]
            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), 1024):
                sel = perm[i:i + 1024]
                X_raw = tr_X[sel]; yb = tr_Y[sel]; p = tr_pos[sel]
                if feature_fn is not None:
                    x = feature_fn(X_raw, yb, p).to(device)
                else:
                    x = X_raw.to(device)
                with torch.no_grad():
                    gp = torch.from_numpy(compute_pattern_labels_batch(
                        yb.numpy(), p.numpy(),
                        pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                    ).to(device)
                    legal = pat_labels_to_cell_labels(gp, pattern_to_cell)
                em = (p % 2 == 0); om = ~em
                loss = torch.tensor(0.0, device=device)
                for msk, m in [(em, me), (om, mo)]:
                    if not msk.any(): continue
                    pl = m(x[msk])
                    cell_logits = patterns_to_cell_logsumexp(pl, pattern_to_cell)
                    if args.loss == "listwise":
                        loss = loss + listwise_cell_ce(cell_logits, legal[msk])
                    else:
                        loss = loss + nn.functional.binary_cross_entropy_with_logits(
                            cell_logits, legal[msk], pos_weight=pw)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item(); total_batches += 1
            del tr_X, tr_Y, tr_pos
        avg = total_loss / max(total_batches, 1)
        print(f"\nEpoch {epoch}: loss={avg:.5f}", flush=True)
        _print_results(eval_topk(), "finetuned")

        # Save after every epoch so time-outs don't lose progress.
        base = os.path.splitext(os.path.basename(args.ckpt))[0]
        save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        loss_tag = "listw" if args.loss == "listwise" else f"pw{int(args.pos_weight)}"
        save_path = os.path.join(save_dir,
            f"ftlegal_{base}_{args.finetune}_{loss_tag}.pt")
        # Also save input_dim so compare_aggregators / probe can reload.
        torch.save({
            'even': me.state_dict(), 'odd': mo.state_dict(),
            'source_ckpt': args.ckpt, 'finetune': args.finetune,
            'input_dim': input_dim, 'hidden_dim': args.hidden,
            'n_patterns': n_patterns, 'epoch': epoch,
        }, save_path)
        print(f"  Saved {save_path}", flush=True)
