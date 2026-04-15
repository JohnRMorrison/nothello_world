"""Train 960 flanking-pattern detectors on MLP-predicted board state.

Two modes:
  two-stage: Train MLP on board state (64×3), freeze, then train 960 detectors
  end-to-end: Train MLP + 960 detectors jointly on pattern labels

Input: 60-d "when" features (from precomputed chunks)
MLP: 60 → H → 64×3 (board state predictor)
Detectors: 192-d (P(empty), P(mine), P(opponent) per cell from predicted board) → 960 sigmoid

Usage:
    python train_pattern_detectors.py --mode two-stage --hidden 512 --epochs 3
    python train_pattern_detectors.py --mode end-to-end --hidden 1024 --epochs 3
"""
import sys, os, json, math
sys.path.insert(0, '.')

import argparse
import numpy as np
import torch
import torch.nn as nn

from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES, OPTIONS,
)
from hand_crafted_flanking import (
    enumerate_flanking_patterns,
    VALID_MOVES, MOVE_TO_IDX, CENTER_CELLS,
)
from generate_rule_games import precompute_pattern_arrays, evaluate_rules_vec

# ---------------------------------------------------------------------------
# Pattern label computation
# ---------------------------------------------------------------------------

def evaluate_patterns_vec(flat, is_black_turn, targets, terminals, opp_cells, opp_mask):
    """Like evaluate_rules_vec but returns the full 960-bit firing vector."""
    my_val = 1 if is_black_turn else -1
    opp_val = -my_val

    target_empty = (flat[targets] == 0)
    terminal_friendly = (flat[terminals] == my_val)
    opp_vals = flat[opp_cells]
    opp_match = (opp_vals == opp_val) | ~opp_mask
    opp_all = opp_match.all(axis=1)

    fires = target_empty & opp_all & terminal_friendly
    return fires.astype(np.float32)


def compute_pattern_labels_from_board_labels(board_labels, positions,
                                              targets, terminals, opp_cells, opp_mask):
    """Compute 960-bit pattern labels from board state labels.

    board_labels: (N, 64) int64 with 0=empty, 1=white, 2=black
    positions: (N,) int64 — move index (even=black just moved, odd=white just moved)

    Returns: (N, 960) float32 pattern firing labels
    """
    n = len(board_labels)
    n_patterns = len(targets)
    pattern_labels = np.zeros((n, n_patterns), dtype=np.float32)

    for i in range(n):
        # Convert labels back to -1/0/1 board
        lbl = board_labels[i]  # (64,)
        flat = np.zeros(64, dtype=np.int8)
        flat[lbl == 1] = -1  # white
        flat[lbl == 2] = 1   # black

        # After position t, the next player to move:
        # Black moves at step 0,2,4... so after even step, it's white's turn
        pos = positions[i]
        is_black_turn = (pos % 2 == 1)  # odd position → black's turn next

        pattern_labels[i] = evaluate_patterns_vec(
            flat, is_black_turn, targets, terminals, opp_cells, opp_mask)

    return pattern_labels


def compute_pattern_labels_batch(board_labels, positions,
                                  targets, terminals, opp_cells, opp_mask):
    """Vectorized pattern label computation (much faster than loop)."""
    n = len(board_labels)
    n_patterns = len(targets)

    # Convert labels to flat board: 0=empty, 1=white→-1, 2=black→+1
    flat = np.zeros((n, 64), dtype=np.int8)
    flat[board_labels == 1] = -1  # white
    flat[board_labels == 2] = 1   # black

    # Determine whose turn it is
    is_black = (positions % 2 == 1)  # (N,)

    # Evaluate all patterns vectorized across positions
    # target_empty: (N, n_patterns)
    target_empty = (flat[:, targets] == 0)

    # terminal check depends on whose turn — need to split
    pattern_labels = np.zeros((n, n_patterns), dtype=np.float32)

    for turn_val, mask in [(True, is_black), (False, ~is_black)]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        my_val = 1 if turn_val else -1
        opp_val = -my_val

        sub_flat = flat[idx]  # (M, 64)

        sub_target_empty = target_empty[idx]  # (M, n_patterns)
        sub_terminal_friendly = (sub_flat[:, terminals] == my_val)  # (M, n_patterns)

        # Opponent cells: (M, n_patterns, max_opp)
        sub_opp_vals = sub_flat[:, opp_cells]  # (M, n_patterns, max_opp)
        sub_opp_match = (sub_opp_vals == opp_val) | ~opp_mask[None, :, :]
        sub_opp_all = sub_opp_match.all(axis=2)  # (M, n_patterns)

        fires = sub_target_empty & sub_opp_all & sub_terminal_friendly
        pattern_labels[idx] = fires.astype(np.float32)

    return pattern_labels


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BoardStateMLP(nn.Module):
    """Single hidden layer MLP: 60-d when → 64×3 board state."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64 * OPTIONS),
        )

    def forward(self, x):
        return self.net(x).view(-1, 64, OPTIONS)

    def predict_board(self, x):
        """Return predicted class per cell: (N, 64)."""
        with torch.no_grad():
            logits = self.forward(x)
        return logits.argmax(dim=-1)


def labels_to_192d(board_labels, positions):
    """Convert predicted board labels to 192-d encoding (vectorized).

    Per cell: (P(empty), P(mine), P(opponent)) — one-hot from hard labels.

    board_labels: (N, 64) int — 0=empty, 1=white, 2=black
    positions: (N,) int — move index

    Returns: (N, 192) float32
    """
    n = len(board_labels)
    encoding = torch.zeros(n, 192, dtype=torch.float32)

    is_black_turn = (positions % 2 == 1)  # next player is black

    # Empty cells (class 0) — same regardless of turn
    empty_mask = (board_labels == 0)  # (N, 64)
    encoding[:, 0::3] = empty_mask.float()

    # Black's turn: mine=black(2), opponent=white(1)
    bk = is_black_turn.bool()
    if bk.any():
        encoding[bk, 1::3] = (board_labels[bk] == 2).float()  # mine
        encoding[bk, 2::3] = (board_labels[bk] == 1).float()  # opponent

    # White's turn: mine=white(1), opponent=black(2)
    wk = ~bk
    if wk.any():
        encoding[wk, 1::3] = (board_labels[wk] == 1).float()  # mine
        encoding[wk, 2::3] = (board_labels[wk] == 2).float()  # opponent

    return encoding


def labels_to_192d_soft(logits, positions):
    """Soft version: use softmax probabilities instead of hard argmax.

    Per cell: (P(empty), P(mine), P(opponent)) from MLP softmax.

    logits: (N, 64, 3) — raw logits from MLP (classes: 0=empty, 1=white, 2=black)
    positions: (N,) int

    Returns: (N, 192) float32
    """
    probs = torch.softmax(logits, dim=-1)  # (N, 64, 3)
    n = len(logits)
    encoding = torch.zeros(n, 192, dtype=logits.dtype, device=logits.device)

    is_black_turn = (positions % 2 == 1)
    black_mask = is_black_turn.bool()

    # P(empty) is always class 0 regardless of turn
    encoding[:, 0::3] = probs[:, :, 0]

    # Black's turn: mine=P(black)=class2, opp=P(white)=class1
    if black_mask.any():
        idx = black_mask.nonzero(as_tuple=True)[0]
        encoding[idx, 1::3] = probs[idx, :, 2]  # P(mine) = P(black)
        encoding[idx, 2::3] = probs[idx, :, 1]  # P(opponent) = P(white)

    # White's turn: mine=P(white)=class1, opp=P(black)=class2
    white_mask = ~black_mask
    if white_mask.any():
        idx = white_mask.nonzero(as_tuple=True)[0]
        encoding[idx, 1::3] = probs[idx, :, 1]  # P(mine) = P(white)
        encoding[idx, 2::3] = probs[idx, :, 2]  # P(opponent) = P(black)

    return encoding


class PatternDetectors(nn.Module):
    """960 linear detectors: 192-d → 960 sigmoid outputs."""

    def __init__(self, input_dim=192, n_patterns=960):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_patterns)

    def forward(self, x):
        return self.linear(x)  # raw logits; use BCE with logits


class EndToEndModel(nn.Module):
    """Joint model: 60-d → MLP → soft 192-d → 960 pattern detectors.

    The MLP predicts board state, which is converted to soft
    (P(empty), P(mine), P(opponent)) encoding via differentiable softmax,
    then fed to the pattern detectors.
    """

    def __init__(self, input_dim, hidden_dim, n_patterns=960):
        super().__init__()
        self.mlp = BoardStateMLP(input_dim, hidden_dim)
        self.detectors = PatternDetectors(192, n_patterns)

    def forward(self, x, positions):
        """Returns (pattern_logits, board_logits)."""
        board_logits = self.mlp(x)  # (N, 64, 3)
        encoding = labels_to_192d_soft(board_logits, positions)
        pattern_logits = self.detectors(encoding)  # (N, 960)
        return pattern_logits, board_logits


# ---------------------------------------------------------------------------
# Legal move evaluation from pattern detections
# ---------------------------------------------------------------------------

def patterns_to_legal_moves(pattern_probs, patterns):
    """Aggregate 960 pattern probabilities → 60-d legal move prediction.

    A square is predicted legal if any pattern targeting it fires.
    Use max over pattern probs for each target cell.
    """
    n = len(pattern_probs)
    legal_probs = np.zeros((n, 60), dtype=np.float32)

    for j, pat in enumerate(patterns):
        target = pat['target']
        move_idx = MOVE_TO_IDX[target]
        legal_probs[:, move_idx] = np.maximum(
            legal_probs[:, move_idx], pattern_probs[:, j])

    return legal_probs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_two_stage(chunk_dir, device, input_dim, hidden_dim, patterns,
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
                    feature_cols, epochs=3, batch_size=1024, save_dir=None):
    """Stage 1: train MLP on board state. Stage 2: train detectors on MLP output."""

    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir) if f.endswith(".npz"))
    if not chunk_files:
        raise ValueError(f"No chunk files in {chunk_dir}")

    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    n_patterns = len(patterns)

    print(f"Two-stage training: {len(chunk_files)} chunks, H={hidden_dim}, "
          f"input={input_dim}, {epochs} epochs per stage")

    # --- Stage 1: Train MLP on board state ---
    print(f"\n{'='*60}")
    print("STAGE 1: Train MLP on board state prediction")
    print(f"{'='*60}")

    # Separate even/odd MLPs (consistent with existing codebase)
    mlp_even = BoardStateMLP(input_dim, hidden_dim).to(device)
    mlp_odd = BoardStateMLP(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        list(mlp_even.parameters()) + list(mlp_odd.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    # Load eval data
    ev_X, ev_Y, ev_pos = _load_features(eval_path)
    if feature_cols is not None:
        ev_X = ev_X[:, feature_cols]
    n_eval = min(len(ev_X), 49 * 10000)
    ev_X, ev_Y, ev_pos = ev_X[:n_eval], ev_Y[:n_eval], ev_pos[:n_eval]

    best_acc = 0.0
    best_mlp_state = None

    for epoch in range(1, epochs + 1):
        mlp_even.train(); mlp_odd.train()
        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))
        epoch_loss, epoch_batches = 0.0, 0

        for ci in chunk_order:
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            if feature_cols is not None:
                tr_X = tr_X[:, feature_cols]
            perm = torch.randperm(len(tr_X))

            for i in range(0, len(tr_X), batch_size):
                idx = perm[i:i + batch_size]
                x = tr_X[idx].to(device)
                y = tr_Y[idx].to(device)
                pos = tr_pos[idx]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                loss = torch.tensor(0.0, device=device)
                if even_mask.any():
                    logits = mlp_even(x[even_mask])
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[even_mask].reshape(-1))
                if odd_mask.any():
                    logits = mlp_odd(x[odd_mask])
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[odd_mask].reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1

            del tr_X, tr_Y, tr_pos

        # Eval
        mlp_even.eval(); mlp_odd.eval()
        correct, total = 0, 0
        losses = []
        with torch.no_grad():
            for i in range(0, len(ev_X), batch_size):
                x = ev_X[i:i + batch_size].to(device)
                y = ev_Y[i:i + batch_size].to(device)
                pos = ev_pos[i:i + batch_size]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask
                preds = torch.zeros_like(y)

                if even_mask.any():
                    logits = mlp_even(x[even_mask])
                    preds[even_mask] = logits.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[even_mask].reshape(-1)).item())
                if odd_mask.any():
                    logits = mlp_odd(x[odd_mask])
                    preds[odd_mask] = logits.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[odd_mask].reshape(-1)).item())

                correct += (preds == y).sum().item()
                total += y.numel()

        acc = correct / total
        mean_loss = np.mean(losses)
        if acc > best_acc:
            best_acc = acc
            best_mlp_state = {
                'even': {k: v.cpu().clone() for k, v in mlp_even.state_dict().items()},
                'odd': {k: v.cpu().clone() for k, v in mlp_odd.state_dict().items()},
            }
        scheduler.step(mean_loss)
        cur_lr = optimizer.param_groups[0]['lr']
        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"  Stage 1 Epoch {epoch}: acc={acc:.4%}  loss={avg_loss:.5f}  lr={cur_lr:.2e}",
              flush=True)

    # Restore best MLP
    mlp_even.load_state_dict(best_mlp_state['even'])
    mlp_odd.load_state_dict(best_mlp_state['odd'])
    mlp_even.eval(); mlp_odd.eval()
    for p in mlp_even.parameters(): p.requires_grad = False
    for p in mlp_odd.parameters(): p.requires_grad = False
    print(f"  Stage 1 best acc: {best_acc:.4%}")

    # --- Stage 2: Train pattern detectors on frozen MLP output ---
    print(f"\n{'='*60}")
    print("STAGE 2: Train pattern detectors on frozen MLP output")
    print(f"{'='*60}")

    detectors = PatternDetectors(192, n_patterns).to(device)
    det_optimizer = torch.optim.Adam(detectors.parameters(), lr=1e-3)
    det_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        det_optimizer, mode='min', factor=0.75, patience=1)

    best_det_acc = 0.0
    best_det_state = None

    for epoch in range(1, epochs + 1):
        detectors.train()
        rng = np.random.RandomState(100 + epoch)
        chunk_order = rng.permutation(len(train_paths))
        epoch_loss, epoch_batches = 0.0, 0

        for ci in chunk_order:
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            if feature_cols is not None:
                tr_X = tr_X[:, feature_cols]

            # Compute pattern labels from ground truth board state
            pat_labels = compute_pattern_labels_batch(
                tr_Y.numpy(), tr_pos.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
            pat_labels = torch.tensor(pat_labels, dtype=torch.float32)

            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), batch_size):
                idx = perm[i:i + batch_size]
                x = tr_X[idx].to(device)
                pos = tr_pos[idx]
                y_pat = pat_labels[idx].to(device)

                # Get MLP predictions (frozen)
                with torch.no_grad():
                    even_mask = (pos % 2 == 0)
                    odd_mask = ~even_mask
                    pred_labels = torch.zeros(len(idx), 64, dtype=torch.long)
                    if even_mask.any():
                        pred_labels[even_mask] = mlp_even.predict_board(x[even_mask])
                    if odd_mask.any():
                        pred_labels[odd_mask] = mlp_odd.predict_board(x[odd_mask])

                # Convert to 192-d encoding
                encoding = labels_to_192d(pred_labels, pos).to(device)

                # Train detectors
                logits = detectors(encoding)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y_pat)

                det_optimizer.zero_grad()
                loss.backward()
                det_optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1

            del tr_X, tr_Y, tr_pos, pat_labels

        # Eval
        detectors.eval()
        correct, total = 0, 0
        losses = []
        with torch.no_grad():
            for i in range(0, len(ev_X), batch_size):
                x = ev_X[i:i + batch_size].to(device)
                y_board = ev_Y[i:i + batch_size]
                pos = ev_pos[i:i + batch_size]

                # Pattern ground truth
                y_pat = compute_pattern_labels_batch(
                    y_board.numpy(), pos.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                y_pat_t = torch.tensor(y_pat, dtype=torch.float32).to(device)

                # MLP predictions
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask
                pred_labels = torch.zeros(len(x), 64, dtype=torch.long)
                if even_mask.any():
                    pred_labels[even_mask] = mlp_even.predict_board(x[even_mask])
                if odd_mask.any():
                    pred_labels[odd_mask] = mlp_odd.predict_board(x[odd_mask])

                encoding = labels_to_192d(pred_labels, pos).to(device)
                logits = detectors(encoding)

                preds = (logits > 0).float()
                correct += (preds == y_pat_t).sum().item()
                total += y_pat_t.numel()
                losses.append(nn.functional.binary_cross_entropy_with_logits(
                    logits, y_pat_t).item())

        acc = correct / total
        mean_loss = np.mean(losses)
        if acc > best_det_acc:
            best_det_acc = acc
            best_det_state = {k: v.cpu().clone()
                              for k, v in detectors.state_dict().items()}
        det_scheduler.step(mean_loss)
        cur_lr = det_optimizer.param_groups[0]['lr']
        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"  Stage 2 Epoch {epoch}: pattern_acc={acc:.4%}  "
              f"loss={avg_loss:.5f}  lr={cur_lr:.2e}", flush=True)

    # Restore best detectors
    if best_det_state is not None:
        detectors.load_state_dict(best_det_state)

    # --- Evaluate legal move prediction ---
    _evaluate_legal_moves(mlp_even, mlp_odd, detectors, patterns,
                          ev_X, ev_Y, ev_pos, device, batch_size,
                          pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)

    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir,
            f"pattern_det_two_stage_H{hidden_dim}.pt")
        torch.save({
            'mlp_even': best_mlp_state['even'],
            'mlp_odd': best_mlp_state['odd'],
            'detectors': best_det_state,
            'hidden_dim': hidden_dim,
            'input_dim': input_dim,
            'n_patterns': n_patterns,
            'mlp_best_acc': best_acc,
            'detector_best_acc': best_det_acc,
            'mode': 'two-stage',
        }, save_path)
        print(f"Saved to {save_path}")

    return best_acc, best_det_acc


def train_end_to_end(chunk_dir, device, input_dim, hidden_dim, patterns,
                     pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
                     feature_cols, epochs=3, batch_size=1024, save_dir=None):
    """Train MLP + detectors jointly on pattern labels."""

    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir) if f.endswith(".npz"))
    if not chunk_files:
        raise ValueError(f"No chunk files in {chunk_dir}")

    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    n_patterns = len(patterns)

    print(f"End-to-end training: {len(chunk_files)} chunks, H={hidden_dim}, "
          f"input={input_dim}, {epochs} epochs")

    # Single model for even/odd (pattern detection is perspective-aware via encoding)
    model_even = EndToEndModel(input_dim, hidden_dim, n_patterns).to(device)
    model_odd = EndToEndModel(input_dim, hidden_dim, n_patterns).to(device)

    optimizer = torch.optim.Adam(
        list(model_even.parameters()) + list(model_odd.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    # Load eval data
    ev_X, ev_Y, ev_pos = _load_features(eval_path)
    if feature_cols is not None:
        ev_X = ev_X[:, feature_cols]
    n_eval = min(len(ev_X), 49 * 10000)
    ev_X, ev_Y, ev_pos = ev_X[:n_eval], ev_Y[:n_eval], ev_pos[:n_eval]

    # Precompute eval pattern labels
    ev_pat = compute_pattern_labels_batch(
        ev_Y.numpy(), ev_pos.numpy(),
        pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
    ev_pat = torch.tensor(ev_pat, dtype=torch.float32)

    # Weight for auxiliary board-state loss
    board_loss_weight = 0.5

    best_pat_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model_even.train(); model_odd.train()
        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))
        epoch_loss, epoch_batches = 0.0, 0

        for ci in chunk_order:
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            if feature_cols is not None:
                tr_X = tr_X[:, feature_cols]

            # Compute pattern labels
            tr_pat = compute_pattern_labels_batch(
                tr_Y.numpy(), tr_pos.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
            tr_pat = torch.tensor(tr_pat, dtype=torch.float32)

            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), batch_size):
                idx = perm[i:i + batch_size]
                x = tr_X[idx].to(device)
                y_board = tr_Y[idx].to(device)
                y_pat = tr_pat[idx].to(device)
                pos = tr_pos[idx]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                loss = torch.tensor(0.0, device=device)

                for mask, model in [(even_mask, model_even), (odd_mask, model_odd)]:
                    if not mask.any():
                        continue
                    pat_logits, board_logits = model(x[mask], pos[mask])

                    # Pattern loss (primary)
                    pat_loss = nn.functional.binary_cross_entropy_with_logits(
                        pat_logits, y_pat[mask])

                    # Board state loss (auxiliary — helps MLP learn board state)
                    board_loss = nn.functional.cross_entropy(
                        board_logits.reshape(-1, OPTIONS),
                        y_board[mask].reshape(-1))

                    loss = loss + pat_loss + board_loss_weight * board_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1

            del tr_X, tr_Y, tr_pos, tr_pat

        # Eval
        model_even.eval(); model_odd.eval()
        pat_correct, pat_total = 0, 0
        board_correct, board_total = 0, 0
        losses = []
        with torch.no_grad():
            for i in range(0, len(ev_X), batch_size):
                x = ev_X[i:i + batch_size].to(device)
                y_board = ev_Y[i:i + batch_size].to(device)
                y_pat = ev_pat[i:i + batch_size].to(device)
                pos = ev_pos[i:i + batch_size]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                all_pat_preds = torch.zeros(len(x), n_patterns, device=device)
                all_board_preds = torch.zeros(len(x), 64, dtype=torch.long, device=device)

                for mask, model in [(even_mask, model_even), (odd_mask, model_odd)]:
                    if not mask.any():
                        continue
                    pat_logits, board_logits = model(x[mask], pos[mask])
                    all_pat_preds[mask] = (pat_logits > 0).float()
                    all_board_preds[mask] = board_logits.argmax(-1)

                pat_correct += (all_pat_preds == y_pat).sum().item()
                pat_total += y_pat.numel()
                board_correct += (all_board_preds == y_board).sum().item()
                board_total += y_board.numel()

        pat_acc = pat_correct / pat_total
        board_acc = board_correct / board_total
        if pat_acc > best_pat_acc:
            best_pat_acc = pat_acc
            best_state = {
                'even': {k: v.cpu().clone() for k, v in model_even.state_dict().items()},
                'odd': {k: v.cpu().clone() for k, v in model_odd.state_dict().items()},
            }
        scheduler.step(epoch_loss / max(epoch_batches, 1))
        cur_lr = optimizer.param_groups[0]['lr']
        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"  Epoch {epoch}: pat_acc={pat_acc:.4%}  board_acc={board_acc:.4%}  "
              f"loss={avg_loss:.5f}  lr={cur_lr:.2e}", flush=True)

    # Restore best and evaluate legal moves
    if best_state:
        model_even.load_state_dict(best_state['even'])
        model_odd.load_state_dict(best_state['odd'])
    model_even.eval(); model_odd.eval()

    _evaluate_legal_moves_e2e(model_even, model_odd, patterns,
                              ev_X, ev_Y, ev_pos, device, batch_size,
                              pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)

    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir,
            f"pattern_det_e2e_H{hidden_dim}.pt")
        torch.save({
            'even': best_state['even'] if best_state else None,
            'odd': best_state['odd'] if best_state else None,
            'hidden_dim': hidden_dim,
            'input_dim': input_dim,
            'n_patterns': n_patterns,
            'best_pat_acc': best_pat_acc,
            'board_acc': board_acc,
            'mode': 'end-to-end',
        }, save_path)
        print(f"Saved to {save_path}")

    return best_pat_acc, board_acc


# ---------------------------------------------------------------------------
# Legal move evaluation
# ---------------------------------------------------------------------------

def _evaluate_legal_moves(mlp_even, mlp_odd, detectors, patterns,
                          ev_X, ev_Y, ev_pos, device, batch_size,
                          pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask):
    """Evaluate legal move prediction from pattern detections."""
    print(f"\n{'='*60}")
    print("EVALUATION: Legal move prediction from pattern detectors")
    print(f"{'='*60}")

    detectors.eval()
    all_pat_probs = []
    all_gt_pat = []

    with torch.no_grad():
        for i in range(0, len(ev_X), batch_size):
            x = ev_X[i:i + batch_size].to(device)
            y_board = ev_Y[i:i + batch_size]
            pos = ev_pos[i:i + batch_size]
            even_mask = (pos % 2 == 0)
            odd_mask = ~even_mask

            pred_labels = torch.zeros(len(x), 64, dtype=torch.long)
            if even_mask.any():
                pred_labels[even_mask] = mlp_even.predict_board(x[even_mask])
            if odd_mask.any():
                pred_labels[odd_mask] = mlp_odd.predict_board(x[odd_mask])

            encoding = labels_to_192d(pred_labels, pos).to(device)
            logits = detectors(encoding)
            all_pat_probs.append(torch.sigmoid(logits).cpu().numpy())

            gt_pat = compute_pattern_labels_batch(
                y_board.numpy(), pos.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
            all_gt_pat.append(gt_pat)

    all_pat_probs = np.concatenate(all_pat_probs)
    all_gt_pat = np.concatenate(all_gt_pat)

    _report_legal_move_metrics(all_pat_probs, all_gt_pat, patterns,
                                pat_targets, pat_opp_cells, pat_opp_mask)


def _evaluate_legal_moves_e2e(model_even, model_odd, patterns,
                               ev_X, ev_Y, ev_pos, device, batch_size,
                               pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask):
    """Evaluate legal move prediction for end-to-end model."""
    print(f"\n{'='*60}")
    print("EVALUATION: Legal move prediction (end-to-end)")
    print(f"{'='*60}")

    all_pat_probs = []
    all_gt_pat = []

    with torch.no_grad():
        for i in range(0, len(ev_X), batch_size):
            x = ev_X[i:i + batch_size].to(device)
            y_board = ev_Y[i:i + batch_size]
            pos = ev_pos[i:i + batch_size]
            even_mask = (pos % 2 == 0)
            odd_mask = ~even_mask

            pat_logits = torch.zeros(len(x), len(patterns), device=device)
            for mask, model in [(even_mask, model_even), (odd_mask, model_odd)]:
                if not mask.any():
                    continue
                pl, _ = model(x[mask], pos[mask])
                pat_logits[mask] = pl

            all_pat_probs.append(torch.sigmoid(pat_logits).cpu().numpy())

            gt_pat = compute_pattern_labels_batch(
                y_board.numpy(), pos.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
            all_gt_pat.append(gt_pat)

    all_pat_probs = np.concatenate(all_pat_probs)
    all_gt_pat = np.concatenate(all_gt_pat)

    _report_legal_move_metrics(all_pat_probs, all_gt_pat, patterns,
                                pat_targets, pat_opp_cells, pat_opp_mask)


def _report_legal_move_metrics(pat_probs, gt_pat, patterns,
                                pat_targets, pat_opp_cells, pat_opp_mask):
    """Compute and print legal move metrics from pattern predictions."""
    # Predicted legal moves from patterns
    pred_legal = patterns_to_legal_moves(pat_probs, patterns)

    # Ground truth legal moves from ground truth patterns
    gt_legal = patterns_to_legal_moves(gt_pat, patterns)

    # Binary predictions at threshold 0.5
    pred_binary = (pred_legal > 0.5).astype(np.float32)
    gt_binary = (gt_legal > 0.5).astype(np.float32)

    tp = ((pred_binary == 1) & (gt_binary == 1)).sum()
    fp = ((pred_binary == 1) & (gt_binary == 0)).sum()
    fn = ((pred_binary == 0) & (gt_binary == 1)).sum()
    tn = ((pred_binary == 0) & (gt_binary == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    # Perfect position accuracy (all 60 squares correct)
    perfect = (pred_binary == gt_binary).all(axis=1).mean()

    # Pattern-level metrics
    pat_pred = (pat_probs > 0.5).astype(np.float32)
    pat_acc = (pat_pred == gt_pat).mean()
    pat_firing_rate = gt_pat.mean()
    pred_firing_rate = pat_pred.mean()

    print(f"\nPattern-level metrics:")
    print(f"  Pattern accuracy: {pat_acc:.4%}")
    print(f"  GT firing rate: {pat_firing_rate:.4%}")
    print(f"  Pred firing rate: {pred_firing_rate:.4%}")

    print(f"\nLegal move metrics:")
    print(f"  Per-square accuracy: {accuracy:.4%}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Perfect position rate: {perfect:.4%}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["two-stage", "end-to-end"])
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    # Feature setup: 60-d "when" features
    feature_cols = list(range(N_MOVES, 2 * N_MOVES))
    input_dim = N_MOVES  # 60

    device = get_device()
    print(f"Device: {device}")
    print(f"Mode: {args.mode}, H={args.hidden}, {args.epochs} epochs")

    # Enumerate patterns
    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    print(f"Flanking patterns: {len(patterns)}")

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")

    if args.mode == "two-stage":
        train_two_stage(
            chunk_dir, device, input_dim, args.hidden, patterns,
            pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
            feature_cols, epochs=args.epochs, batch_size=args.batch_size,
            save_dir=save_dir)
    else:
        train_end_to_end(
            chunk_dir, device, input_dim, args.hidden, patterns,
            pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
            feature_cols, epochs=args.epochs, batch_size=args.batch_size,
            save_dir=save_dir)
