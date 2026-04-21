"""Train a 60-cell MLP with a ranking loss directly — no pattern supervision.

Three loss modes:
  next_move : for each sample, strip the most-recent move from the features
              and use it as the softmax target.
  listwise  : softmax cross-entropy against uniform-over-legal distribution.
  pairwise  : for each (legal, illegal) pair, hinge loss with margin.

Architecture: same as our 960-pattern MLP but with 60-d output.
Reports top-1 legal + recall@K (K = num legal) each epoch.

Usage:
  python train_rank_loss.py --features when+even --loss listwise --epochs 3
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
    compute_pattern_labels_batch, pat_labels_to_cell_labels, _get_cell_pat_index,
    to_signed_parity_input, to_mine_signed_input, to_board_state_input,
    to_color_split_input, to_played_halfmask_input, to_played_bit_input,
    to_move_grid_input, to_move_grid_onehot_input,
)


class CellMLP(nn.Module):
    """input_dim -> H -> 60 (no final nonlinearity)."""
    def __init__(self, input_dim, hidden_dim, n_cells=60):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_cells))

    def forward(self, x):
        return self.net(x)


def strip_last_move(X_raw):
    """Given 180-d features, zero out the cell corresponding to the move with
    the largest 'when' value (the most recent move). Return (stripped_X, target).

    If a sample has no played moves (pos=0), target is arbitrary (we'll mask it).
    """
    when = X_raw[:, 60:120]   # (B, 60)
    played = X_raw[:, :60]
    # Cells with played > 0.5 — pick the one with max when among them.
    masked_when = torch.where(played > 0.5, when, torch.full_like(when, -1.0))
    target = masked_when.argmax(dim=-1)   # (B,)

    X_stripped = X_raw.clone()
    rng = torch.arange(X_raw.shape[0])
    X_stripped[rng, target] = 0                      # played[target] = 0
    X_stripped[rng, target + 60] = 0                 # when[target] = 0
    X_stripped[rng, target + 120] = 0                # even[target] = 0
    return X_stripped, target


def compute_losses(logits, legal, target_next, loss_type, margin=1.0):
    """logits: (B, 60), legal: (B, 60) binary, target_next: (B,) cell idx."""
    if loss_type == "next_move":
        return nn.functional.cross_entropy(logits, target_next)
    if loss_type == "listwise":
        K = legal.sum(dim=-1).clamp(min=1).unsqueeze(-1)
        target = legal / K
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(dim=-1).mean()
    if loss_type == "pairwise":
        # For each position, get pairs (legal, illegal). Use max over legal
        # minus min over illegal approximation for efficiency:
        # actually just do all pairs with broadcast (60 x 60 ~ 3600 per pos).
        # loss = mean over (l, i in pairs of position): max(0, margin - s[l] + s[i])
        # Only count valid legal cells and valid illegal cells.
        #
        # Compute via broadcast: (B, 60_legal, 60_illegal)
        # delta[b, l, i] = score[l] - score[i]
        # hinge[b, l, i] = max(0, margin - delta)
        # mask[b, l, i]  = legal[l] * (1 - legal[i])
        # loss = (mask * hinge).sum() / mask.sum()
        s = logits
        delta = s.unsqueeze(-1) - s.unsqueeze(1)                # (B, 60, 60)
        hinge = torch.clamp(margin - delta, min=0.0)
        mask = legal.unsqueeze(-1) * (1.0 - legal).unsqueeze(1)
        total = mask.sum().clamp(min=1.0)
        return (mask * hinge).sum() / total
    raise ValueError(loss_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="when+even",
                        choices=["when", "played", "played+when", "when+even",
                                 "played+even", "all", "signed_parity",
                                 "color_split", "move_grid", "move_grid_onehot"])
    parser.add_argument("--loss", required=True,
                        choices=["next_move", "listwise", "pairwise"])
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Margin for pairwise hinge loss")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()

    _feat_cols = {
        "when":         list(range(N_MOVES, 2 * N_MOVES)),
        "played":       list(range(0, N_MOVES)),
        "played+when":  list(range(0, 2 * N_MOVES)),
        "when+even":    list(range(N_MOVES, 3 * N_MOVES)),
        "played+even":  list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)),
        "all":          list(range(0, 3 * N_MOVES)),
    }
    feature_cols, feature_fn = None, None
    if args.features in _feat_cols:
        feature_cols = _feat_cols[args.features]; input_dim = len(feature_cols)
    elif args.features == "signed_parity":
        feature_fn = lambda X, Y, pos: to_signed_parity_input(X); input_dim = N_MOVES
    elif args.features == "color_split":
        feature_fn = lambda X, Y, pos: to_color_split_input(X); input_dim = 2 * N_MOVES
    elif args.features == "move_grid":
        feature_fn = lambda X, Y, pos: to_move_grid_input(X); input_dim = 60 * 60
    elif args.features == "move_grid_onehot":
        feature_fn = lambda X, Y, pos: to_move_grid_onehot_input(X); input_dim = 60 * 60 * 3
    print(f"Features: {args.features} (input={input_dim})  Loss: {args.loss}  H={args.hidden}")

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    pattern_to_cell = torch.tensor([MOVE_TO_IDX[p['target']] for p in patterns],
                                   dtype=torch.long, device=device)
    idx_pc, mask_pc = _get_cell_pat_index(pattern_to_cell, 60)

    # Two models (even/odd split)
    model_even = CellMLP(input_dim, args.hidden).to(device)
    model_odd = CellMLP(input_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(
        list(model_even.parameters()) + list(model_odd.parameters()), lr=args.lr)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    print(f"Chunks: {len(chunk_files)} ({len(train_paths)} train)")

    def prep_input(X_raw, Y, pos, need_next_move):
        """Apply feature_fn (or column slice) and optionally strip the last move."""
        if args.loss == "next_move":
            X_raw, next_move = strip_last_move(X_raw)
        else:
            next_move = None
        if feature_cols is not None:
            x_in = X_raw[:, feature_cols]
        else:
            x_in = feature_fn(X_raw, Y, pos)
        return x_in, next_move

    def forward_split(x_in, pos):
        em = (pos % 2 == 0).cpu()
        om = ~em
        out = torch.zeros(len(x_in), 60, device=x_in.device)
        if em.any(): out[em] = model_even(x_in[em])
        if om.any(): out[om] = model_odd(x_in[om])
        return out

    def eval_pass():
        model_even.eval(); model_odd.eval()
        X, Y, pos = _load_features(eval_path)
        n = min(len(X), 49 * 10000)
        rng = np.random.RandomState(0)
        si = np.sort(rng.choice(len(X), n, replace=False))
        X, Y, pos = X[si], Y[si], pos[si]
        top1_correct = 0; total = 0
        recallk_sum = 0.0
        with torch.no_grad():
            for i in range(0, n, args.batch_size):
                X_raw = X[i:i+args.batch_size]
                yb = Y[i:i+args.batch_size]
                p = pos[i:i+args.batch_size]
                x_in, _ = prep_input(X_raw, yb, p, need_next_move=False)
                x_in = x_in.to(device)
                logits = forward_split(x_in, p).cpu().numpy()
                gp = compute_pattern_labels_batch(
                    yb.numpy(), p.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                gp_t = torch.from_numpy(gp).to(device)
                legal = pat_labels_to_cell_labels(gp_t, pattern_to_cell).cpu().numpy()
                for b in range(logits.shape[0]):
                    legal_set = set(np.where(legal[b] > 0.5)[0].tolist())
                    K = len(legal_set)
                    if K == 0: continue
                    ranked = np.argsort(-logits[b])
                    top1_correct += int(ranked[0] in legal_set)
                    top_k = set(ranked[:K].tolist())
                    recallk_sum += len(top_k & legal_set) / K
                    total += 1
        return top1_correct / total, recallk_sum / total

    for epoch in range(1, args.epochs + 1):
        model_even.train(); model_odd.train()
        rng_e = np.random.RandomState(epoch)
        order = rng_e.permutation(len(train_paths))
        epoch_loss = 0.0; epoch_batches = 0
        for ci in order:
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), args.batch_size):
                sel = perm[i:i+args.batch_size]
                X_raw = tr_X[sel]
                yb = tr_Y[sel]; p = tr_pos[sel]
                x_in, next_move = prep_input(X_raw, yb, p, need_next_move=True)
                x_in = x_in.to(device)
                gp = compute_pattern_labels_batch(
                    yb.numpy(), p.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                gp_t = torch.from_numpy(gp).to(device)
                legal = pat_labels_to_cell_labels(gp_t, pattern_to_cell)  # (B, 60) float

                logits = forward_split(x_in, p)
                target_next = next_move.to(device) if next_move is not None else None
                loss = compute_losses(logits, legal, target_next, args.loss, args.margin)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item(); epoch_batches += 1
            del tr_X, tr_Y, tr_pos
        avg = epoch_loss / max(epoch_batches, 1)
        top1, recallk = eval_pass()
        print(f"Epoch {epoch}: loss={avg:.5f}  top1={top1:.4%}  recall@K={recallk:.4%}",
              flush=True)

    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    feat_tag = args.features.replace('+', '')
    save_path = os.path.join(save_dir,
        f"rank_{args.loss}_{feat_tag}_H{args.hidden}.pt")
    torch.save({
        'even': model_even.state_dict(), 'odd': model_odd.state_dict(),
        'input_dim': input_dim, 'hidden_dim': args.hidden,
        'loss': args.loss, 'features': args.features,
    }, save_path)
    print(f"Saved to {save_path}")
