"""Train a linear readout from 960 pattern logits → 60-d legal moves.

Freezes the pattern detector model, trains only a Linear(960, 60) layer
on top. Uses BCE with pos_weight to handle legal move imbalance.

Usage:
    python train_legal_readout.py --model-ckpt pattern_simple_direct_H512.pt \
        --mode direct --hidden 512 --epochs 3
"""
import sys, os
sys.path.insert(0, '.')

import argparse
import numpy as np
import torch
import torch.nn as nn
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES, OPTIONS,
)
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX, VALID_MOVES
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import DirectMLP, EndToEndMLP, TwoStageMLP


def compute_legal_labels(board_labels, positions, pat_targets, pat_terminals,
                         pat_opp_cells, pat_opp_mask, patterns):
    """Compute 60-d legal move labels from board state."""
    from train_pattern_simple import compute_pattern_labels_batch
    pat = compute_pattern_labels_batch(
        board_labels, positions, pat_targets, pat_terminals,
        pat_opp_cells, pat_opp_mask)
    # Aggregate: cell is legal if any pattern fires
    legal = np.zeros((len(pat), 60), dtype=np.float32)
    for j, p in enumerate(patterns):
        move_idx = MOVE_TO_IDX[p['target']]
        legal[:, move_idx] = np.maximum(legal[:, move_idx], pat[j] if pat.ndim == 1 else pat[:, j])
    return legal


def train_readout(chunk_dir, device, model_even, model_odd, mode, patterns,
                  pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
                  feature_cols, epochs=3, batch_size=1024, save_path=None):

    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)

    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]

    # Readout: 960 pattern logits → 60 legal move logits
    # Separate even/odd readouts
    readout_even = nn.Linear(960, 60).to(device)
    readout_odd = nn.Linear(960, 60).to(device)

    # Legal moves fire ~15% of cells on average (higher than pattern firing rate)
    pw = torch.tensor([5.0], device=device)

    optimizer = torch.optim.Adam(
        list(readout_even.parameters()) + list(readout_odd.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    # Eval data
    ev_X, ev_Y, ev_pos = _load_features(eval_path)
    if feature_cols is not None:
        ev_X = ev_X[:, feature_cols]
    n_eval = min(len(ev_X), 49 * 10000)
    ev_X = ev_X[:n_eval].clone()
    ev_Y = ev_Y[:n_eval].clone()
    ev_pos = ev_pos[:n_eval].clone()

    print(f"Training legal readout: {len(chunk_files)} chunks, {epochs} epochs")
    print(f"  Eval: {n_eval} samples")

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        readout_even.train(); readout_odd.train()
        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))
        epoch_loss = 0.0
        epoch_batches = 0

        for ci in chunk_order:
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            if feature_cols is not None:
                tr_X = tr_X[:, feature_cols]

            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), batch_size):
                idx = perm[i:i + batch_size]
                x = tr_X[idx].to(device)
                y_board = tr_Y[idx]
                pos = tr_pos[idx]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                # Compute legal move labels
                y_legal_np = compute_legal_labels(
                    y_board.numpy(), pos.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
                    patterns)
                y_legal = torch.from_numpy(y_legal_np).to(device)

                # Get frozen pattern logits
                loss = torch.tensor(0.0, device=device)
                with torch.no_grad():
                    if mode in ("direct", "randproj"):
                        if even_mask.any():
                            pat_logits_e = model_even(x[even_mask])
                        if odd_mask.any():
                            pat_logits_o = model_odd(x[odd_mask])
                    else:
                        if even_mask.any():
                            pat_logits_e, _ = model_even(x[even_mask], pos[even_mask])
                        if odd_mask.any():
                            pat_logits_o, _ = model_odd(x[odd_mask], pos[odd_mask])

                if even_mask.any():
                    legal_logits = readout_even(pat_logits_e)
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(
                        legal_logits, y_legal[even_mask], pos_weight=pw)
                if odd_mask.any():
                    legal_logits = readout_odd(pat_logits_o)
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(
                        legal_logits, y_legal[odd_mask], pos_weight=pw)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1

            del tr_X, tr_Y, tr_pos

        # Eval
        readout_even.eval(); readout_odd.eval()
        top1_correct = 0
        total = 0
        tp = fp = fn = 0
        with torch.no_grad():
            for i in range(0, len(ev_X), batch_size):
                x = ev_X[i:i + batch_size].to(device)
                y_board = ev_Y[i:i + batch_size]
                pos = ev_pos[i:i + batch_size]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                y_legal_np = compute_legal_labels(
                    y_board.numpy(), pos.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
                    patterns)
                y_legal = torch.from_numpy(y_legal_np).to(device)

                legal_logits = torch.zeros(len(x), 60, device=device)
                if mode in ("direct", "randproj"):
                    if even_mask.any():
                        legal_logits[even_mask] = readout_even(model_even(x[even_mask]))
                    if odd_mask.any():
                        legal_logits[odd_mask] = readout_odd(model_odd(x[odd_mask]))
                else:
                    if even_mask.any():
                        pl, _ = model_even(x[even_mask], pos[even_mask])
                        legal_logits[even_mask] = readout_even(pl)
                    if odd_mask.any():
                        pl, _ = model_odd(x[odd_mask], pos[odd_mask])
                        legal_logits[odd_mask] = readout_odd(pl)

                preds = (legal_logits > 0).float()
                tp += ((preds == 1) & (y_legal == 1)).sum().item()
                fp += ((preds == 1) & (y_legal == 0)).sum().item()
                fn += ((preds == 0) & (y_legal == 1)).sum().item()

                # Top-1 legal
                for b in range(len(x)):
                    legal_set = set(torch.where(y_legal[b] > 0.5)[0].tolist())
                    if legal_set:
                        top1 = legal_logits[b].argmax().item()
                        if top1 in legal_set:
                            top1_correct += 1
                        total += 1

        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        top1_acc = top1_correct / max(total, 1)

        if top1_acc > best_acc:
            best_acc = top1_acc
            best_state = {
                'even': {k: v.cpu().clone() for k, v in readout_even.state_dict().items()},
                'odd': {k: v.cpu().clone() for k, v in readout_odd.state_dict().items()},
            }

        scheduler.step(epoch_loss / max(epoch_batches, 1))
        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"  Epoch {epoch}: top1={top1_acc:.4%}  recall={recall:.4%}  "
              f"prec={precision:.4%}  f1={f1:.4f}  loss={avg_loss:.5f}", flush=True)

        if best_state and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'even': best_state['even'],
                'odd': best_state['odd'],
                'best_top1': best_acc,
            }, save_path)
            print(f"  Saved {save_path}", flush=True)

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["direct", "emergent", "e2e", "two-stage", "randproj"])
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    feature_cols = list(range(N_MOVES, 2 * N_MOVES))

    # Load frozen model
    ckpt = torch.load(args.model_ckpt, map_location=device)
    n_patterns = ckpt.get('n_patterns', 960)

    if args.mode in ("direct", "randproj"):
        model_even = DirectMLP(N_MOVES, args.hidden, n_patterns).to(device)
        model_odd = DirectMLP(N_MOVES, args.hidden, n_patterns).to(device)
    elif args.mode == "two-stage":
        model_even = TwoStageMLP(N_MOVES, args.hidden, n_patterns).to(device)
        model_odd = TwoStageMLP(N_MOVES, args.hidden, n_patterns).to(device)
    else:
        model_even = EndToEndMLP(N_MOVES, args.hidden, n_patterns).to(device)
        model_odd = EndToEndMLP(N_MOVES, args.hidden, n_patterns).to(device)

    model_even.load_state_dict(ckpt['even'])
    model_odd.load_state_dict(ckpt['odd'])
    model_even.eval(); model_odd.eval()
    for p in model_even.parameters(): p.requires_grad = False
    for p in model_odd.parameters(): p.requires_grad = False
    print(f"Model: {args.mode} H={args.hidden} (pat_acc={ckpt.get('best_pat_acc', '?')})")

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    save_path = os.path.join(save_dir, f"legal_readout_{args.mode}_H{args.hidden}.pt")

    train_readout(chunk_dir, device, model_even, model_odd, args.mode, patterns,
                  pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
                  feature_cols, epochs=args.epochs, save_path=save_path)
