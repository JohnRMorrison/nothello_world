"""Train 960 pattern detectors — simple version mirroring train_streaming.py.

Same chunk-streaming pattern that works for board-state MLP training.
The ONLY difference: target is 960 binary pattern labels instead of 64x3.
Pattern labels are computed per-batch from board labels (no precomputed files).

Modes:
  direct:   60-d → H → 960 (no board state intermediate)
  emergent: 60-d → H → 64×3 → softmax → 192-d → 960 (no board aux loss)
  e2e:      same architecture as emergent + 0.5× board state aux loss

Usage:
    python train_pattern_simple.py --mode direct --hidden 512 --epochs 3
    python train_pattern_simple.py --mode emergent --hidden 1024 --epochs 3
    python train_pattern_simple.py --mode e2e --hidden 1024 --epochs 3
"""
import sys, os, json
sys.path.insert(0, '.')

import argparse
import numpy as np
import torch
import torch.nn as nn
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES, OPTIONS,
)
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays


# ---------------------------------------------------------------------------
# Pattern label computation (per-batch, ~6ms for 1024 samples)
# ---------------------------------------------------------------------------

def compute_pattern_labels_batch(board_labels, positions,
                                  targets, terminals, opp_cells, opp_mask):
    """Vectorized: board labels (N,64) + positions (N,) → pattern labels (N,960)."""
    n = len(board_labels)
    # Convert labels (0=empty, 1=white, 2=black) to flat board (-1=white, 0=empty, 1=black)
    flat = np.zeros((n, 64), dtype=np.int8)
    flat[board_labels == 1] = -1
    flat[board_labels == 2] = 1

    is_black = (positions % 2 == 1)
    target_empty = (flat[:, targets] == 0)
    pattern_labels = np.zeros((n, len(targets)), dtype=np.float32)

    for turn_val, mask in [(True, is_black), (False, ~is_black)]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        my_val = 1 if turn_val else -1
        opp_val = -my_val
        sub = flat[idx]
        fires = (target_empty[idx]
                 & (sub[:, terminals] == my_val)
                 & ((sub[:, opp_cells] == opp_val) | ~opp_mask[None, :, :]).all(axis=2))
        pattern_labels[idx] = fires.astype(np.float32)

    return pattern_labels


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DirectMLP(nn.Module):
    """60-d → H → 960."""
    def __init__(self, input_dim, hidden_dim, n_patterns=960):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_patterns))

    def forward(self, x):
        return self.net(x)


class EndToEndMLP(nn.Module):
    """60-d → H → 64×3 → softmax → 192-d → 960."""
    def __init__(self, input_dim, hidden_dim, n_patterns=960):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64 * 3))
        self.detectors = nn.Linear(192, n_patterns)

    def forward(self, x, positions):
        board_logits = self.backbone(x).view(-1, 64, 3)
        probs = torch.softmax(board_logits, dim=-1)

        # Build 192-d encoding: (empty, mine, opponent) per cell
        enc = torch.zeros(len(x), 192, dtype=x.dtype, device=x.device)
        if positions.device != x.device:
            positions = positions.to(x.device)
        is_black = (positions % 2 == 1).bool()

        enc[:, 0::3] = probs[:, :, 0]  # P(empty)
        if is_black.any():
            i = is_black.nonzero(as_tuple=True)[0]
            enc[i, 1::3] = probs[i, :, 2]  # mine = black
            enc[i, 2::3] = probs[i, :, 1]  # opp = white
        wh = (~is_black)
        if wh.any():
            i = wh.nonzero(as_tuple=True)[0]
            enc[i, 1::3] = probs[i, :, 1]
            enc[i, 2::3] = probs[i, :, 2]

        pat_logits = self.detectors(enc)
        return pat_logits, board_logits


class TwoStageMLP(nn.Module):
    """Frozen MLP → hard 192-d encoding → trainable Linear(192, 960).

    The MLP predicts board state (64×3), argmax gives class per cell,
    converted to 192-d one-hot (empty/mine/opponent), then a linear
    detector layer maps to 960 pattern logits.
    """
    def __init__(self, input_dim, hidden_dim, n_patterns=960):
        super().__init__()
        # Board state MLP (will be frozen after loading checkpoint)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64 * 3))
        self.detectors = nn.Linear(192, n_patterns)

    def forward(self, x, positions):
        # Frozen MLP → hard board state → 192-d encoding → detectors
        with torch.no_grad():
            board_logits = self.backbone(x).view(-1, 64, 3)
            pred_classes = board_logits.argmax(dim=-1)  # (N, 64)

        # Build 192-d hard encoding on device
        device = x.device
        if positions.device != device:
            positions = positions.to(device)
        n = len(x)
        enc = torch.zeros(n, 192, dtype=torch.float32, device=device)
        is_black = (positions % 2 == 1).bool()

        enc[:, 0::3] = (pred_classes == 0).float()  # empty
        if is_black.any():
            i = is_black.nonzero(as_tuple=True)[0]
            enc[i, 1::3] = (pred_classes[i] == 2).float()  # mine = black
            enc[i, 2::3] = (pred_classes[i] == 1).float()  # opp = white
        wh = (~is_black)
        if wh.any():
            i = wh.nonzero(as_tuple=True)[0]
            enc[i, 1::3] = (pred_classes[i] == 1).float()
            enc[i, 2::3] = (pred_classes[i] == 2).float()

        pat_logits = self.detectors(enc)
        return pat_logits, board_logits

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False


def _strip_prefix(d, prefix):
    """Strip a key prefix from a state dict, only at the start of each key."""
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in d.items()}


# ---------------------------------------------------------------------------
# Training (mirrors _train_random_proj_streaming exactly)
# ---------------------------------------------------------------------------

def train(chunk_dir, device, input_dim, hidden_dim, mode,
          feature_cols, pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
          board_loss_weight=0.0, lr=1e-3, epochs=3, batch_size=1024,
          save_path=None, mlp_ckpt_dir=None, seed=0, pos_weight=None):


    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    if not chunk_files:
        raise ValueError(f"No chunks in {chunk_dir}")

    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    n_patterns = len(pat_targets)

    # pos_weight for BCE: upweight rare positive class (patterns fire ~1.35%)
    pw_tensor = None
    if pos_weight is not None:
        pw_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        print(f"  pos_weight={pos_weight:.1f}")

    print(f"{mode} training: {len(chunk_files)} chunks, H={hidden_dim}, "
          f"input={input_dim}, {epochs} epochs, board_loss_weight={board_loss_weight}")

    # Build even/odd models
    if mode == "randproj":
        # Frozen random first layer + trained output (same for even/odd)
        import math
        torch.manual_seed(seed)
        proj_W = torch.randn(input_dim, hidden_dim, device=device) * math.sqrt(2.0 / input_dim)
        # Build DirectMLP then replace first layer with frozen random projection
        model_even = DirectMLP(input_dim, hidden_dim, n_patterns).to(device)
        model_odd = DirectMLP(input_dim, hidden_dim, n_patterns).to(device)
        with torch.no_grad():
            model_even.net[0].weight.copy_(proj_W.T)
            model_even.net[0].bias.zero_()
            model_odd.net[0].weight.copy_(proj_W.T)
            model_odd.net[0].bias.zero_()
        # Freeze first layer
        model_even.net[0].weight.requires_grad = False
        model_even.net[0].bias.requires_grad = False
        model_odd.net[0].weight.requires_grad = False
        model_odd.net[0].bias.requires_grad = False
        print(f"  Random projection: seed={seed}, H={hidden_dim}")
    elif mode == "direct":
        model_even = DirectMLP(input_dim, hidden_dim, n_patterns).to(device)
        model_odd = DirectMLP(input_dim, hidden_dim, n_patterns).to(device)
    elif mode == "two-stage":
        model_even = TwoStageMLP(input_dim, hidden_dim, n_patterns).to(device)
        model_odd = TwoStageMLP(input_dim, hidden_dim, n_patterns).to(device)
        # Load frozen MLP backbones from stage-1 checkpoints
        if mlp_ckpt_dir is None:
            mlp_ckpt_dir = os.path.dirname(save_path) if save_path else "."
        ckpt_path = os.path.join(mlp_ckpt_dir, f"mlp_stage1_H{hidden_dim}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"two-stage requires a stage-1 MLP checkpoint at {ckpt_path}. "
                f"Train one first with train_streaming.py.")
        ckpt = torch.load(ckpt_path, map_location=device)
        model_even.backbone.load_state_dict(_strip_prefix(ckpt['even'], "net."))
        model_odd.backbone.load_state_dict(_strip_prefix(ckpt['odd'], "net."))
        print(f"  Loaded MLP backbone from {ckpt_path} (acc={ckpt['best_acc']:.4%})")
        model_even.freeze_backbone()
        model_odd.freeze_backbone()
        # board_loss_weight is ignored for two-stage (backbone is frozen,
        # board_logits computed under no_grad — no gradients flow)
    else:  # e2e or emergent
        model_even = EndToEndMLP(input_dim, hidden_dim, n_patterns).to(device)
        model_odd = EndToEndMLP(input_dim, hidden_dim, n_patterns).to(device)

    # Only optimize trainable parameters
    trainable = [p for p in list(model_even.parameters()) + list(model_odd.parameters())
                 if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    # Load eval data — EXACTLY like train_streaming.py
    ev_X, ev_Y, ev_pos = _load_features(eval_path)
    if feature_cols is not None:
        ev_X = ev_X[:, feature_cols]
    n_eval = min(len(ev_X), 49 * 10000)
    ev_X = ev_X[:n_eval].clone()
    ev_Y = ev_Y[:n_eval].clone()
    ev_pos = ev_pos[:n_eval].clone()
    print(f"  Eval samples: {len(ev_X)}")

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model_even.train(); model_odd.train()
        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))
        epoch_loss = 0.0
        epoch_batches = 0

        for ci in chunk_order:
            # Load chunk — same as train_streaming.py
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            if feature_cols is not None:
                tr_X = tr_X[:, feature_cols]

            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), batch_size):
                idx = perm[i:i + batch_size]
                x = tr_X[idx].to(device)
                y_board = tr_Y[idx]  # keep on CPU for pattern computation
                pos = tr_pos[idx]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                # Compute pattern labels per-batch (~6ms)
                y_pat_np = compute_pattern_labels_batch(
                    y_board.numpy(), pos.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                y_pat = torch.from_numpy(y_pat_np).to(device)

                loss = torch.tensor(0.0, device=device)
                if mode in ("direct", "randproj"):
                    if even_mask.any():
                        logits = model_even(x[even_mask])
                        loss = loss + nn.functional.binary_cross_entropy_with_logits(
                            logits, y_pat[even_mask], pos_weight=pw_tensor)
                    if odd_mask.any():
                        logits = model_odd(x[odd_mask])
                        loss = loss + nn.functional.binary_cross_entropy_with_logits(
                            logits, y_pat[odd_mask], pos_weight=pw_tensor)
                else:
                    y_board_gpu = y_board.to(device)
                    for mask, model in [(even_mask, model_even), (odd_mask, model_odd)]:
                        if not mask.any():
                            continue
                        pat_logits, board_logits = model(x[mask], pos[mask])
                        loss = loss + nn.functional.binary_cross_entropy_with_logits(
                            pat_logits, y_pat[mask], pos_weight=pw_tensor)
                        if board_loss_weight > 0:
                            loss = loss + board_loss_weight * nn.functional.cross_entropy(
                                board_logits.reshape(-1, OPTIONS),
                                y_board_gpu[mask].reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1

            del tr_X, tr_Y, tr_pos

        # Eval
        model_even.eval(); model_odd.eval()
        correct = 0
        total = 0
        tp = 0
        fp = 0
        fn = 0
        board_correct = 0
        board_total = 0
        with torch.no_grad():
            for i in range(0, len(ev_X), batch_size):
                x = ev_X[i:i + batch_size].to(device)
                y_board = ev_Y[i:i + batch_size]
                pos = ev_pos[i:i + batch_size]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                y_pat_np = compute_pattern_labels_batch(
                    y_board.numpy(), pos.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                y_pat = torch.from_numpy(y_pat_np).to(device)

                if mode in ("direct", "randproj"):
                    preds = torch.zeros_like(y_pat)
                    if even_mask.any():
                        preds[even_mask] = (model_even(x[even_mask]) > 0).float()
                    if odd_mask.any():
                        preds[odd_mask] = (model_odd(x[odd_mask]) > 0).float()
                else:
                    preds = torch.zeros_like(y_pat)
                    board_preds = torch.zeros(len(x), 64, dtype=torch.long, device=device)
                    for mask, model in [(even_mask, model_even), (odd_mask, model_odd)]:
                        if not mask.any():
                            continue
                        pl, bl = model(x[mask], pos[mask])
                        preds[mask] = (pl > 0).float()
                        board_preds[mask] = bl.argmax(-1)
                    y_board_gpu = y_board.to(device)
                    board_correct += (board_preds == y_board_gpu).sum().item()
                    board_total += y_board_gpu.numel()

                correct += (preds == y_pat).sum().item()
                total += y_pat.numel()
                # Per-class metrics
                tp += ((preds == 1) & (y_pat == 1)).sum().item()
                fp += ((preds == 1) & (y_pat == 0)).sum().item()
                fn += ((preds == 0) & (y_pat == 1)).sum().item()

        acc = correct / total
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        if acc > best_acc:
            best_acc = acc
            best_state = {
                'even': {k: v.cpu().clone() for k, v in model_even.state_dict().items()},
                'odd': {k: v.cpu().clone() for k, v in model_odd.state_dict().items()},
            }
        scheduler.step(epoch_loss / max(epoch_batches, 1))
        cur_lr = optimizer.param_groups[0]['lr']
        avg_loss = epoch_loss / max(epoch_batches, 1)
        board_str = f"  board_acc={board_correct/max(board_total,1):.4%}" if board_total > 0 else ""
        print(f"  Epoch {epoch}: pat_acc={acc:.4%}  recall={recall:.4%}  prec={precision:.4%}"
              f"{board_str}  loss={avg_loss:.5f}  lr={cur_lr:.2e}", flush=True)

        # Save checkpoint after each epoch
        if best_state and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'even': best_state['even'],
                'odd': best_state['odd'],
                'hidden_dim': hidden_dim,
                'input_dim': input_dim,
                'n_patterns': n_patterns,
                'best_pat_acc': best_acc,
                'mode': mode,
                'epoch': epoch,
            }, save_path)
            print(f"  Saved {save_path}", flush=True)

    return best_acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["direct", "emergent", "e2e", "two-stage", "randproj"])
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for random projection initialization")
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="Upweight positive class in BCE loss (e.g. 50 for ~1.35%% firing rate)")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    feature_cols = list(range(N_MOVES, 2 * N_MOVES))  # 60-d "when"
    input_dim = N_MOVES

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    print(f"Device: {device}, Mode: {args.mode}, H={args.hidden}, {args.epochs} epochs")
    print(f"Patterns: {len(patterns)}")

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    save_path = os.path.join(save_dir, f"pattern_simple_{args.mode}_H{args.hidden}.pt")

    board_loss_weight = 0.5 if args.mode == "e2e" else 0.0
    if args.mode == "randproj":
        save_path = os.path.join(save_dir,
            f"pattern_simple_randproj_s{args.seed}_H{args.hidden}.pt")

    train(chunk_dir, device, input_dim, args.hidden, args.mode,
          feature_cols, pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
          board_loss_weight=board_loss_weight, epochs=args.epochs,
          save_path=save_path, mlp_ckpt_dir=save_dir, seed=args.seed,
          pos_weight=args.pos_weight)

