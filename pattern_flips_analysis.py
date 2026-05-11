"""Sanity check on the pattern-length finding: is the model failing on long
patterns intrinsically, or because long patterns are more likely to span cells
that have been flipped (which the move-history feature can't see)?

For each fired pattern instance, we compute the number of "definitely flipped"
cells (opponent + terminal) using the X[+even] channel + Y board state:
  - opponent cell at C is FLIPPED iff played_parity(C) != current_turn_parity
  - terminal cell at C is FLIPPED iff played_parity(C) == current_turn_parity
A "definitely flipped" cell is one whose initial color (from play parity)
differs from its required current color (from the pattern). This counts only
single-flip evidence; even-flips remain invisible. It is therefore a lower
bound on actual flip activity.

Output: per-(length, flip_count) cell of pattern recall (TP / (TP + FN) at
threshold logit > 0). If recall is roughly flat within each length row, the
length-itself story holds. If recall plummets with flip_count within each
row, the model is failing at flip-tracking and length is mostly a proxy.

Usage:
    python pattern_flips_analysis.py \\
        --ckpt experiments/.../pattern_simple_direct_H1024_wheneven.pt \\
        --hidden 1024
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES,
)
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import (
    DirectMLP, EndToEndMLP, TwoStageMLP, compute_pattern_labels_batch,
    to_signed_parity_input, to_mine_signed_input, to_board_state_input,
    to_color_split_input, to_played_halfmask_input, to_played_bit_input,
    to_move_grid_input, to_move_grid_onehot_input,
)


_FEAT_COLS = {
    "when":        list(range(N_MOVES, 2 * N_MOVES)),
    "played":      list(range(0, N_MOVES)),
    "played+when": list(range(0, 2 * N_MOVES)),
    "when+even":   list(range(N_MOVES, 3 * N_MOVES)),
    "played+even": list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)),
    "all":         list(range(0, 3 * N_MOVES)),
}


def infer_features(name, input_dim):
    if "wheneven" in name: return "when+even"
    if input_dim == 120: return "when+even"
    if input_dim == 180: return "all"
    if input_dim == 60: return "when"
    raise ValueError(name)


def select_features(X, Y, pos, features):
    if features in _FEAT_COLS:
        return X[:, _FEAT_COLS[features]]
    if features == "signed_parity": return to_signed_parity_input(X)
    if features == "mine_signed":   return to_mine_signed_input(Y, pos)
    if features == "board_state":   return to_board_state_input(Y, pos)
    if features == "color_split":   return to_color_split_input(X)
    if features == "played+halfmask": return to_played_halfmask_input(X)
    if features == "played+bit":    return to_played_bit_input(X)
    if features == "move_grid":     return to_move_grid_input(X)
    if features == "move_grid_onehot": return to_move_grid_onehot_input(X)
    raise ValueError(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--mode", default="direct",
                        choices=["direct", "emergent", "e2e", "two-stage", "randproj"])
    parser.add_argument("--features", default=None)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = \
        precompute_pattern_arrays(patterns)
    pat_terminals_np = np.asarray(pat_terminals, dtype=np.int64)
    pat_opp_cells_np = np.asarray(pat_opp_cells, dtype=np.int64)
    pat_opp_mask_np  = np.asarray(pat_opp_mask, dtype=bool)
    pattern_length = pat_opp_mask_np.sum(axis=1).astype(np.int64)
    max_len = int(pattern_length.max())
    print(f"Total patterns: {len(patterns)}, max length: {max_len}")
    print(f"pat_opp_cells shape: {pat_opp_cells_np.shape}, "
          f"pat_opp_mask shape: {pat_opp_mask_np.shape}")

    output_dir = "experiments/mathematical_transformation_experiments/heuristic_probe_results"
    chunk_dir = os.path.join(output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz")
                         and "_patterns" not in f
                         and "_when60" not in f)
    eval_path = chunk_files[-1]
    print(f"Loading {eval_path}")

    X_full, Y, pos = _load_features(eval_path)
    N = len(Y)
    print(f"Positions in chunk: {N}")

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    n_patterns = ckpt.get('n_patterns', 960)
    input_dim = ckpt.get('input_dim', N_MOVES)
    if args.features is None:
        args.features = infer_features(os.path.basename(args.ckpt), input_dim)
    print(f"Features: {args.features} (input_dim={input_dim})")
    Cls = {"direct": DirectMLP, "randproj": DirectMLP,
           "two-stage": TwoStageMLP,
           "emergent": EndToEndMLP, "e2e": EndToEndMLP}[args.mode]
    me = Cls(input_dim, args.hidden, n_patterns).to(device)
    mo = Cls(input_dim, args.hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()
    print(f"Loaded {args.ckpt} (pat_acc={ckpt.get('best_pat_acc', '?')})")

    feat_X = select_features(X_full, Y, pos, args.features)
    # X_full columns 120..180 = "even" parity channel (1 if cell played on
    # even turn, 0 otherwise / not played).
    even_full = X_full[:, 2 * N_MOVES:3 * N_MOVES].numpy().astype(np.int8)
    # Sanity diagnostic on encoding: at turn t, # played cells == t. If the
    # even channel is correct, it should sum to ~t/2 on average.
    print("\n(diagnostic) X[+even] sum / position turn for a few rows:")
    for k in (0, 1000, 50000, 200000, 1000000):
        if k < N:
            ev_sum = int(even_full[k].sum())
            print(f"  row {k}: turn={int(pos[k].item())}, "
                  f"sum(even)={ev_sum}  (expect ~ turn/2)")

    # Subsample to 490k positions (same as compare_aggregators)
    n = min(N, 49 * 10000)
    rng = np.random.RandomState(0)
    si = np.sort(rng.choice(N, n, replace=False))
    Y_s, pos_s = Y[si], pos[si]
    feat_X_s = feat_X[si]
    even_s = even_full[si]
    del X_full, feat_X, even_full
    print(f"Sampled to: {n}")

    # Accumulate per (length, flip_count): n_positives, n_correct
    max_flips = max_len + 1
    n_positives = np.zeros((max_len + 1, max_flips + 1), dtype=np.int64)
    n_correct   = np.zeros((max_len + 1, max_flips + 1), dtype=np.int64)

    batch = 4096
    with torch.no_grad():
        for i in range(0, n, batch):
            yb_t = Y_s[i:i + batch]
            pb_t = pos_s[i:i + batch]
            yb = yb_t.numpy()
            pb = pb_t.numpy()
            B = len(yb)
            labels = compute_pattern_labels_batch(
                yb, pb, pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask
            ).astype(np.int64)

            # Model predictions
            xb = feat_X_s[i:i + batch].to(device)
            em = (pb_t % 2 == 0); om = ~em
            preds = torch.zeros(B, len(patterns), device=device)
            if args.mode in ("direct", "randproj"):
                if em.any(): preds[em] = me(xb[em])
                if om.any(): preds[om] = mo(xb[om])
            else:
                for msk, m in [(em, me), (om, mo)]:
                    if not msk.any(): continue
                    logits, _ = m(xb[msk], pb_t[msk])
                    preds[msk] = logits
            pos_pred = (preds > args.threshold).cpu().numpy().astype(np.int64)

            # Compute flip count per (B, 960)
            ev_b = even_s[i:i + batch]   # (B, 60)
            tp_b = (pb % 2).astype(np.int8)   # (B,)
            # Opp cells: flipped iff parity_cell != parity_turn
            opp_parity = ev_b[:, pat_opp_cells_np]                 # (B, 960, max_L)
            flipped_opp = (opp_parity != tp_b[:, None, None]) & pat_opp_mask_np[None]
            n_flip_opp = flipped_opp.sum(-1)                       # (B, 960)
            # Terminal cells: flipped iff parity_cell == parity_turn
            term_parity = ev_b[:, pat_terminals_np]                # (B, 960)
            flipped_term = (term_parity == tp_b[:, None])          # (B, 960)
            flip_count = n_flip_opp + flipped_term.astype(np.int64)  # (B, 960)

            # Accumulate
            for L in range(1, max_len + 1):
                pat_mask = (pattern_length == L)                   # (960,)
                if not pat_mask.any():
                    continue
                fc_L = flip_count[:, pat_mask]                     # (B, n_L)
                lbl_L = labels[:, pat_mask]
                pred_L = pos_pred[:, pat_mask]
                for fc in range(max_flips + 1):
                    m = (fc_L == fc) & (lbl_L == 1)
                    if not m.any():
                        continue
                    n_positives[L, fc] += int(m.sum())
                    n_correct[L, fc]   += int((pred_L * m).sum())

    print()
    print("=" * 78)
    print("Recall by (pattern length, definitely-flipped cell count)")
    print("=" * 78)
    print("(rows = pattern length L; cols = # of L+1 required cells "
          "with parity mismatch")
    print("indicating at least one flip;  '-' = no positives in that bin)")
    print()
    header = ["L"] + [f"f={fc}" for fc in range(max_flips + 1)]
    print("  " + "".join(f"{h:>10s}" for h in header))
    for L in range(1, max_len + 1):
        row = [f"{L:>10d}"]
        for fc in range(max_flips + 1):
            n = n_positives[L, fc]
            c = n_correct[L, fc]
            row.append("        - " if n == 0 else f"{c/n:>9.4f}")
        print("  " + "".join(row))

    print()
    print("Sample sizes (# positive instances per cell):")
    print("  " + "".join(f"{h:>10s}" for h in header))
    for L in range(1, max_len + 1):
        row = [f"{L:>10d}"]
        for fc in range(max_flips + 1):
            n = n_positives[L, fc]
            row.append("        - " if n == 0 else f"{n:>10d}")
        print("  " + "".join(row))

    print()
    print("=" * 78)
    print("Marginal: recall by length (collapsed across flip counts)")
    print("=" * 78)
    for L in range(1, max_len + 1):
        n = n_positives[L].sum()
        c = n_correct[L].sum()
        print(f"  L={L}: recall={c/max(n,1):.4f}  (n_pos={n})")

    print()
    print("=" * 78)
    print("Marginal: recall by flip count (collapsed across lengths)")
    print("=" * 78)
    for fc in range(max_flips + 1):
        n = n_positives[:, fc].sum()
        c = n_correct[:, fc].sum()
        if n == 0:
            continue
        print(f"  flip_count={fc}: recall={c/max(n,1):.4f}  (n_pos={n})")
