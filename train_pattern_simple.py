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
from hand_crafted_flanking import (
    enumerate_flanking_patterns,
    enumerate_flanking_patterns_color_specific,
    MOVE_TO_IDX,
)
from generate_rule_games import precompute_pattern_arrays


def precompute_movers_array(patterns):
    """For color-specific patterns, extract the (n_patterns,) `mover` array
    in {+1, -1}. Returns int8 numpy array."""
    return np.array([p['mover'] for p in patterns], dtype=np.int8)


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


def compute_pattern_labels_color_specific_batch(
        board_labels, targets, terminals, opp_cells, opp_mask, movers):
    """Vectorized labels for the 1920 color-specific patterns.

    No parity used — patterns fire purely as a function of the absolute board
    state. `movers` (1920,) ∈ {+1, -1} tags each pattern with its mover color.

    A pattern with mover m fires when:
      - target cell empty
      - terminal cell holds an m piece
      - all opponent cells hold (-m) pieces

    board_labels: (N, 64) with values in {0=empty, 1=white, 2=black}.
    Returns (N, len(targets)) float32.
    """
    n = len(board_labels)
    # Convert labels to signed board: 0=empty, +1=black, -1=white.
    flat = np.zeros((n, 64), dtype=np.int8)
    flat[board_labels == 1] = -1
    flat[board_labels == 2] = 1

    target_empty = (flat[:, targets] == 0)             # (N, P)
    terminal_val = flat[:, terminals]                   # (N, P)
    opp_val = flat[:, opp_cells]                        # (N, P, max_opp)

    # Per-pattern m and -m (broadcast over batch).
    m = movers[None, :].astype(np.int8)                 # (1, P)
    neg_m = (-m).astype(np.int8)                        # (1, P)

    terminal_ok = (terminal_val == m)                   # (N, P)
    # Opponent slots: either the slot is masked out OR holds -m.
    opp_ok = ((opp_val == neg_m[:, :, None]) | ~opp_mask[None, :, :]).all(axis=2)  # (N, P)

    fires = target_empty & terminal_ok & opp_ok
    return fires.astype(np.float32)


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


_cell_pat_cache = {}


def _get_cell_pat_index(pattern_to_cell, n_cells):
    """Build a (n_cells, max_per_cell) index + validity mask, cached by object id.

    Lets aggregation run as a single gather + logsumexp instead of a 60-iter
    Python loop. Caller is expected to reuse the same pattern_to_cell tensor.
    """
    key = (id(pattern_to_cell), n_cells, str(pattern_to_cell.device))
    if key not in _cell_pat_cache:
        device = pattern_to_cell.device
        groups = [torch.where(pattern_to_cell == c)[0] for c in range(n_cells)]
        max_per_cell = max((len(g) for g in groups), default=0)
        idx = torch.zeros((n_cells, max_per_cell), dtype=torch.long, device=device)
        mask = torch.zeros((n_cells, max_per_cell), dtype=torch.bool, device=device)
        for c, g in enumerate(groups):
            idx[c, :len(g)] = g
            mask[c, :len(g)] = True
        _cell_pat_cache[key] = (idx, mask)
    return _cell_pat_cache[key]


def to_move_grid_onehot_input(X_tensor):
    """60x60x3 one-hot, flattened to 10800-d.

    For each (cell, move) pair: one-hot over {black_placed_here,
    white_placed_here, neither}. "Neither" dominates (most slots).

    Same info as move_grid but with independent weights per color
    (not forced to be antisymmetric).
    """
    B = X_tensor.shape[0]
    signed = to_move_grid_input(X_tensor).view(B, 60, 60)
    black = (signed > 0.5).float()
    white = (signed < -0.5).float()
    empty = (signed.abs() < 0.5).float()
    grid = torch.stack([black, white, empty], dim=-1)   # (B, 60, 60, 3)
    return grid.reshape(B, -1)


def to_move_grid_input(X_tensor):
    """3600-d (60 cells x 60 move-numbers), flattened.

    grid[cell, move_num] = +1 if black played `cell` at `move_num`,
                           -1 if white,
                           0 otherwise.

    Convention (from our precompute code): step 0 is white, step 1 black,
    so 'even' step = white. +1 for black = played AND NOT even.
    """
    B = X_tensor.shape[0]
    played = X_tensor[:, :60]
    when = X_tensor[:, 60:120]
    even = X_tensor[:, 120:180]

    # Move number at which each cell was played (0..59); 0 for unplayed
    # (value there will be 0 because 'played' is 0, so harmless).
    move_num = torch.clamp((when * 60.0 - 1.0).round().long(), 0, 59)   # (B, 60)
    # Color: +1 black, -1 white, 0 if not played
    color_pm = (1.0 - 2.0 * even) * played                               # (B, 60)

    grid = torch.zeros(B, 60, 60, device=X_tensor.device, dtype=X_tensor.dtype)
    grid.scatter_(2, move_num.unsqueeze(-1), color_pm.unsqueeze(-1))
    return grid.reshape(B, 60 * 60)


def to_played_halfmask_input(X_tensor):
    """120-d: [0:60]=played, [60:90]=all ones, [90:120]=all zeros.

    Control for "played+even": the last 60 dims are a FIXED vector
    (same for every sample), so they carry zero game-specific info.
    If this matches played+even performance, the win from 'even' is
    from having extra dims, not from color info.
    """
    played = X_tensor[:, :60]
    B = played.shape[0]
    ones = torch.ones(B, 30, device=played.device, dtype=played.dtype)
    zeros = torch.zeros(B, 30, device=played.device, dtype=played.dtype)
    return torch.cat([played, ones, zeros], dim=-1)


def to_played_bit_input(X_tensor):
    """62-d: [0:60]=played, [60]=1, [61]=0.

    Further control: a single '1' and a single '0' appended to played.
    Essentially just a bias trick — zero new info per sample.
    """
    played = X_tensor[:, :60]
    B = played.shape[0]
    ones = torch.ones(B, 1, device=played.device, dtype=played.dtype)
    zeros = torch.zeros(B, 1, device=played.device, dtype=played.dtype)
    return torch.cat([played, ones, zeros], dim=-1)


def to_color_split_input(X_tensor):
    """120-d: [0:60]=played_by_white (played AND even), [60:120]=played_by_black.

    Same info as 'played+even' but split into per-color channels.
    Convention (from precompute code): step 0 = white, so even step = white move.
    """
    played = X_tensor[:, :60]
    even = X_tensor[:, 120:180]
    white_played = played * even
    black_played = played * (1.0 - even)
    return torch.cat([white_played, black_played], dim=-1)


def to_signed_parity_input(X_tensor):
    """60-d: +1 if played-on-even, -1 if played-on-odd, 0 if empty.
    X_tensor is the raw 180-d features. [0:60]=played, [120:180]=even.
    """
    played = X_tensor[:, :60]
    even = X_tensor[:, 120:180]
    return played * (2.0 * even - 1.0)


def to_mine_signed_input(Y_tensor, pos_tensor):
    """60-d: +1 if played by current-turn color, -1 if opp, 0 if empty.
    Collapses the full 192-d board state into one scalar per square.
    """
    is_black = (pos_tensor % 2 == 1).unsqueeze(-1)
    mine_val = torch.where(is_black, torch.full_like(is_black, 2, dtype=Y_tensor.dtype),
                           torch.full_like(is_black, 1, dtype=Y_tensor.dtype))
    mine = (Y_tensor == mine_val).float()
    opp = ((Y_tensor > 0) & (Y_tensor != mine_val)).float()
    return mine - opp


def to_board_state_input(Y_tensor, pos_tensor):
    """Convert label tensor (N,64) + positions (N,) into (N,192) one-hot
    [empty, mine, opp] per cell. Used for --features board_state (upper-bound).

    Y labels use {0=empty, 1=white, 2=black}. Current turn is black when
    position is odd (matches compute_pattern_labels_batch convention).
    """
    N = len(Y_tensor)
    is_black = (pos_tensor % 2 == 1).unsqueeze(-1)  # (N, 1)
    mine_val = torch.where(is_black, torch.full_like(is_black, 2, dtype=Y_tensor.dtype),
                           torch.full_like(is_black, 1, dtype=Y_tensor.dtype))
    opp_val = torch.where(is_black, torch.full_like(is_black, 1, dtype=Y_tensor.dtype),
                          torch.full_like(is_black, 2, dtype=Y_tensor.dtype))
    empty = (Y_tensor == 0).float()
    mine = (Y_tensor == mine_val).float()
    opp = (Y_tensor == opp_val).float()
    return torch.stack([empty, mine, opp], dim=-1).view(N, -1)


def patterns_to_cell_logsumexp(pat_logits, pattern_to_cell, n_cells=60):
    """(B, n_pat) pattern logits → (B, 60) cell logits via per-cell logsumexp.

    Soft-max approximation of patterns_to_move_scores(...max-per-cell) used
    at eval time. Gradient flows through all patterns covering each cell
    (softmax-weighted). Single gather + masked logsumexp — no Python loop.
    """
    idx, mask = _get_cell_pat_index(pattern_to_cell, n_cells)
    gathered = pat_logits[:, idx]                     # (B, n_cells, max_per_cell)
    gathered = gathered.masked_fill(~mask, float('-inf'))
    return torch.logsumexp(gathered, dim=-1)


def listwise_cell_ce(cell_logits, cell_labels):
    """Listwise softmax CE: target distribution is uniform over legal cells.
    cell_logits: (B, 60), cell_labels: (B, 60) binary float (1 if legal cell).
    Returns scalar: -mean over batch of (1/K_b) * sum_{c legal} log_softmax(cell_logits)[c].
    Equivalent to KL(uniform-over-legal || softmax(cell_logits)).
    """
    log_probs = nn.functional.log_softmax(cell_logits, dim=-1)
    legal_mask = (cell_labels > 0.5).float()
    K = legal_mask.sum(dim=-1).clamp(min=1)
    return -(log_probs * legal_mask).sum(dim=-1).div(K).mean()


def pairwise_cell_hinge(cell_logits, cell_labels, margin=1.0):
    """Pairwise hinge: for each (legal, illegal) pair, max(0, margin - s[l] + s[i])."""
    # cell_logits: (B, 60), cell_labels: (B, 60) binary
    s = cell_logits
    delta = s.unsqueeze(-1) - s.unsqueeze(1)                # (B, 60, 60)
    hinge = torch.clamp(margin - delta, min=0.0)
    mask = cell_labels.unsqueeze(-1) * (1.0 - cell_labels).unsqueeze(1)
    total = mask.sum().clamp(min=1.0)
    return (mask * hinge).sum() / total


def pat_labels_to_cell_labels(y_pat, pattern_to_cell, n_cells=60):
    """(B, n_pat) binary labels → (B, 60) legal-cell labels (max over patterns)."""
    idx, mask = _get_cell_pat_index(pattern_to_cell, n_cells)
    gathered = y_pat[:, idx]
    gathered = gathered.masked_fill(~mask, 0)
    return gathered.max(dim=-1).values


# ---------------------------------------------------------------------------
# Training (mirrors _train_random_proj_streaming exactly)
# ---------------------------------------------------------------------------

def train(chunk_dir, device, input_dim, hidden_dim, mode,
          feature_cols, pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
          board_loss_weight=0.0, lr=1e-3, epochs=3, batch_size=1024,
          save_path=None, mlp_ckpt_dir=None, seed=0, pos_weight=None,
          proj_scale=0.183, proj_half_normal=False, single_model=False,
          legal_weight=0.0, pattern_to_cell=None, loss_type="bce",
          feature_fn=None, pairwise_weight=0.0, pairwise_margin=1.0,
          pos_weight_end=None, listwise_weight=0.0,
          chunk_prefix="chunk_",
          movers=None,
          resume_path=None,
          l1_weight=0.0):
    # When `movers` is provided we're in color-specific mode: labels are
    # computed against absolute board state (no parity), patterns number
    # 2 * 960, and the training is single-model regardless of `single_model`.
    color_specific = movers is not None
    if color_specific:
        single_model = True


    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.startswith(chunk_prefix) and f.endswith(".npz")
                         and "_patterns" not in f and "_when60" not in f)
    if not chunk_files:
        raise ValueError(f"No {chunk_prefix}*.npz in {chunk_dir}")

    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    n_patterns = len(pat_targets)

    # pos_weight for BCE: upweight rare positive class (patterns fire ~1.35%).
    # Optionally schedule to an end-value linearly across epochs.
    pw_tensor = None
    pw_start = pos_weight
    pw_end = pos_weight_end if pos_weight_end is not None else pos_weight
    if pos_weight is not None:
        pw_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        if pw_end != pw_start:
            print(f"  pos_weight schedule: {pw_start:.2f} -> {pw_end:.2f} over {epochs} epochs")
        else:
            print(f"  pos_weight={pos_weight:.1f}")
    if pairwise_weight > 0:
        print(f"  pairwise_weight={pairwise_weight:.2f} (margin={pairwise_margin})")

    if legal_weight > 0 or pairwise_weight > 0 or listwise_weight > 0:
        if pattern_to_cell is None:
            raise ValueError("legal/pairwise/listwise loss requires pattern_to_cell tensor")
        pattern_to_cell = pattern_to_cell.to(device)
        if legal_weight > 0:
            print(f"  legal_weight={legal_weight:.2f} (logsumexp aggregator → 60-d legal BCE)")
        if listwise_weight > 0:
            print(f"  listwise_weight={listwise_weight:.2f} "
                  f"(logsumexp aggregator → softmax CE vs uniform-over-legal)")

    if loss_type not in ("bce", "mse"):
        raise ValueError(f"loss_type must be 'bce' or 'mse', got {loss_type}")
    print(f"  loss_type={loss_type}")

    def pattern_loss(logits, targets):
        """Compute per-pattern loss. BCE with logits or MSE on sigmoid."""
        if loss_type == "bce":
            return nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pw_tensor)
        # MSE on sigmoid output; weight positives by pos_weight to offset imbalance
        prob = torch.sigmoid(logits)
        sq = (prob - targets) ** 2
        if pw_tensor is not None:
            w = torch.where(targets > 0.5, pw_tensor, torch.ones_like(sq))
            return (w * sq).mean()
        return sq.mean()

    print(f"{mode} training: {len(chunk_files)} chunks, H={hidden_dim}, "
          f"input={input_dim}, {epochs} epochs, board_loss_weight={board_loss_weight}")

    # Build even/odd models
    if mode == "randproj":
        # Frozen random first layer + trained output (same for even/odd)
        import math
        torch.manual_seed(seed)
        proj_W = torch.randn(input_dim, hidden_dim, device=device) * proj_scale
        if proj_half_normal:
            proj_W = proj_W.abs()  # fold to positive half
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
        dist_str = "half-normal" if proj_half_normal else "normal"
        print(f"  Random projection: seed={seed}, H={hidden_dim}, "
              f"dist={dist_str}, scale={proj_scale:.3f}")
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

    if single_model:
        # Tie: one model for all positions (no even/odd split)
        model_odd = model_even
        print(f"  single_model=True (odd model = even model)")

    # Only optimize trainable parameters — deduplicate by id if single_model
    all_params = list(model_even.parameters())
    if model_odd is not model_even:
        all_params += list(model_odd.parameters())
    trainable = [p for p in all_params if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    # Resume from a prior checkpoint if requested. Loads model weights,
    # optimizer state, scheduler state, and the epoch counter so training
    # continues cleanly from where it left off.
    start_epoch = 1
    if resume_path is not None and os.path.exists(resume_path):
        rckpt = torch.load(resume_path, map_location=device)
        model_even.load_state_dict(rckpt['even'])
        if model_odd is not model_even:
            model_odd.load_state_dict(rckpt['odd'])
        if 'optimizer' in rckpt:
            optimizer.load_state_dict(rckpt['optimizer'])
        else:
            print(f"  WARNING: resume checkpoint has no optimizer state; "
                  f"training will warm-restart Adam moments")
        if 'scheduler' in rckpt:
            scheduler.load_state_dict(rckpt['scheduler'])
        start_epoch = int(rckpt.get('epoch', 0)) + 1
        prev_best = float(rckpt.get('best_pat_acc', 0.0))
        print(f"  RESUMED from {resume_path} at epoch {start_epoch}, "
              f"prior best_pat_acc={prev_best:.4%}")

    # Load eval data. Chunks are position-sorted, so a head slice lands
    # in one narrow position bucket (positions 5-6 in practice). Take a
    # deterministic random sample across all positions so metrics are
    # representative.
    ev_X, ev_Y, ev_pos = _load_features(eval_path)
    if feature_cols is not None:
        ev_X = ev_X[:, feature_cols]
    # For feature_fn (e.g. move_grid), DELAY expansion until sampled indices
    # are chosen so we don't materialize a (N_chunk, 3600-10800) tensor.
    n_eval = min(len(ev_X), 49 * 10000)
    rng = np.random.RandomState(0)
    sample_idx = np.sort(rng.choice(len(ev_X), n_eval, replace=False))
    ev_X_raw = ev_X[sample_idx].clone()
    ev_Y = ev_Y[sample_idx].clone()
    ev_pos = ev_pos[sample_idx].clone()
    if feature_fn is not None:
        # Apply feature_fn on the sampled rows only (fits in memory).
        ev_X = feature_fn(ev_X_raw, ev_Y, ev_pos)
    else:
        ev_X = ev_X_raw
    print(f"  Eval samples: {len(ev_X)} (random across positions)")

    best_acc = 0.0
    if resume_path is not None and os.path.exists(resume_path):
        best_acc = float(torch.load(resume_path, map_location='cpu')
                         .get('best_pat_acc', 0.0))
    best_state = None

    for epoch in range(start_epoch, epochs + 1):
        # Scheduled pos_weight (linear from pw_start to pw_end)
        if pos_weight is not None and pw_end != pw_start:
            frac = (epoch - 1) / max(epochs - 1, 1)
            cur_pw = pw_start + frac * (pw_end - pw_start)
            pw_tensor = torch.tensor([cur_pw], dtype=torch.float32, device=device)
            print(f"  epoch {epoch}: pos_weight={cur_pw:.2f}")
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
            # For feature_fn, defer application to per-batch to avoid materializing
            # a (chunk_size, high_dim) tensor that can OOM (e.g. move_grid=3600-d,
            # move_grid_onehot=10800-d on 14.5M-row chunks).

            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), batch_size):
                idx = perm[i:i + batch_size]
                x_raw = tr_X[idx]
                y_board = tr_Y[idx]  # keep on CPU for pattern computation
                pos = tr_pos[idx]
                if feature_fn is not None:
                    x = feature_fn(x_raw, y_board, pos).to(device)
                else:
                    x = x_raw.to(device)
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                # Compute pattern labels per-batch (~6ms)
                if color_specific:
                    y_pat_np = compute_pattern_labels_color_specific_batch(
                        y_board.numpy(), pat_targets, pat_terminals,
                        pat_opp_cells, pat_opp_mask, movers)
                else:
                    y_pat_np = compute_pattern_labels_batch(
                        y_board.numpy(), pos.numpy(),
                        pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                y_pat = torch.from_numpy(y_pat_np).to(device)

                loss = torch.tensor(0.0, device=device)
                if mode in ("direct", "randproj"):
                    use_cell = (legal_weight > 0 or pairwise_weight > 0
                                or listwise_weight > 0)
                    if even_mask.any():
                        logits = model_even(x[even_mask])
                        loss = loss + pattern_loss(logits, y_pat[even_mask])
                        if use_cell:
                            cell_logits = patterns_to_cell_logsumexp(logits, pattern_to_cell)
                            cell_labels = pat_labels_to_cell_labels(y_pat[even_mask], pattern_to_cell)
                            if legal_weight > 0:
                                loss = loss + legal_weight * nn.functional.binary_cross_entropy_with_logits(
                                    cell_logits, cell_labels)
                            if pairwise_weight > 0:
                                loss = loss + pairwise_weight * pairwise_cell_hinge(
                                    cell_logits, cell_labels, pairwise_margin)
                            if listwise_weight > 0:
                                loss = loss + listwise_weight * listwise_cell_ce(
                                    cell_logits, cell_labels)
                    if odd_mask.any():
                        logits = model_odd(x[odd_mask])
                        loss = loss + pattern_loss(logits, y_pat[odd_mask])
                        if use_cell:
                            cell_logits = patterns_to_cell_logsumexp(logits, pattern_to_cell)
                            cell_labels = pat_labels_to_cell_labels(y_pat[odd_mask], pattern_to_cell)
                            if legal_weight > 0:
                                loss = loss + legal_weight * nn.functional.binary_cross_entropy_with_logits(
                                    cell_logits, cell_labels)
                            if pairwise_weight > 0:
                                loss = loss + pairwise_weight * pairwise_cell_hinge(
                                    cell_logits, cell_labels, pairwise_margin)
                            if listwise_weight > 0:
                                loss = loss + listwise_weight * listwise_cell_ce(
                                    cell_logits, cell_labels)
                else:
                    y_board_gpu = y_board.to(device)
                    for mask, model in [(even_mask, model_even), (odd_mask, model_odd)]:
                        if not mask.any():
                            continue
                        pat_logits, board_logits = model(x[mask], pos[mask])
                        loss = loss + pattern_loss(pat_logits, y_pat[mask])
                        if board_loss_weight > 0:
                            loss = loss + board_loss_weight * nn.functional.cross_entropy(
                                board_logits.reshape(-1, OPTIONS),
                                y_board_gpu[mask].reshape(-1))
                        if legal_weight > 0:
                            cell_logits = patterns_to_cell_logsumexp(pat_logits, pattern_to_cell)
                            cell_labels = pat_labels_to_cell_labels(y_pat[mask], pattern_to_cell)
                            loss = loss + legal_weight * nn.functional.binary_cross_entropy_with_logits(
                                cell_logits, cell_labels)

                # L1 penalty on weights only (not biases) — produces sparse models
                if l1_weight > 0:
                    l1 = 0.0
                    for n_p, p in model_even.named_parameters():
                        if 'weight' in n_p and p.requires_grad:
                            l1 = l1 + p.abs().mean()
                    if model_odd is not model_even:
                        for n_p, p in model_odd.named_parameters():
                            if 'weight' in n_p and p.requires_grad:
                                l1 = l1 + p.abs().mean()
                    loss = loss + l1_weight * l1

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

                if color_specific:
                    y_pat_np = compute_pattern_labels_color_specific_batch(
                        y_board.numpy(), pat_targets, pat_terminals,
                        pat_opp_cells, pat_opp_mask, movers)
                else:
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
                'color_specific': color_specific,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
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
    parser.add_argument("--proj-scale", type=float, default=0.183,
                        help="Std dev for random projection weights (default: Kaiming sqrt(2/60))")
    parser.add_argument("--proj-half-normal", action="store_true",
                        help="Use positive-only (half-normal) weights for random projection")
    parser.add_argument("--single-model", action="store_true",
                        help="Use ONE model for all positions (no even/odd split)")
    parser.add_argument("--color-specific", action="store_true",
                        help="Use 1920 color-specific patterns (absolute color "
                             "encoding) instead of the 960 mine/yours patterns. "
                             "Forces --single-model. Tests whether color-relative "
                             "encoding is actually harder to learn.")
    parser.add_argument("--l1-weight", type=float, default=0.0,
                        help="L1 penalty coefficient on weights (not biases) for "
                             "producing sparse models. Typical 1e-4 .. 1e-2.")
    parser.add_argument("--resume", default=None,
                        help="Path to a prior checkpoint. Loads model weights, "
                             "optimizer state, scheduler state, and resumes from "
                             "epoch+1.")
    parser.add_argument("--legal-weight", type=float, default=0.0,
                        help="Weight on direct legal-cell BCE loss (logsumexp-aggregated)")
    parser.add_argument("--pairwise-weight", type=float, default=0.0,
                        help="Weight on cell-level pairwise hinge loss (legal > illegal + margin)")
    parser.add_argument("--pairwise-margin", type=float, default=1.0)
    parser.add_argument("--listwise-weight", type=float, default=0.0,
                        help="Weight on listwise softmax-CE vs uniform-over-legal target "
                             "(logsumexp-aggregated). Directly minimizes recall@K loss.")
    parser.add_argument("--pos-weight-end", type=float, default=None,
                        help="If set, linearly interpolate pos_weight from --pos-weight to this value across epochs")
    parser.add_argument("--loss", choices=["bce", "mse"], default="bce",
                        help="Pattern-level loss: bce (default) or mse on sigmoid (uniform scales)")
    parser.add_argument("--features", default="when",
                        choices=["when", "played", "played+when", "when+even",
                                 "played+even", "all", "board_state",
                                 "signed_parity", "mine_signed", "color_split",
                                 "played+halfmask", "played+bit", "move_grid",
                                 "move_grid_onehot"],
                        help="Input features. Slices/derivations of the 180-d base. "
                             "signed_parity (60-d): +1/-1 per played color, 0 empty. "
                             "mine_signed (60-d): +1/-1 relative to current turn, 0 empty. "
                             "color_split (120-d): separate channels for white and black placements. "
                             "board_state (192-d): ground-truth board (upper-bound experiment).")
    parser.add_argument("--chunk-prefix", default="chunk_",
                        help="Filename prefix for chunks (e.g. chunk_ext_ for "
                             "extended-range chunks covering turns 5-58).")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    # feature_cols = column-select mode; feature_fn = derived-features mode.
    # Exactly one is used per run.
    _feat_cols = {
        "when":         list(range(N_MOVES, 2 * N_MOVES)),          # 60-d
        "played":       list(range(0, N_MOVES)),                     # 60-d
        "played+when":  list(range(0, 2 * N_MOVES)),                 # 120-d
        "when+even":    list(range(N_MOVES, 3 * N_MOVES)),           # 120-d
        "played+even":  list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)),
        "all":          list(range(0, 3 * N_MOVES)),                  # 180-d
    }
    feature_fn = None
    if args.features in _feat_cols:
        feature_cols = _feat_cols[args.features]
        input_dim = len(feature_cols)
    elif args.features == "signed_parity":
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_signed_parity_input(X)
        input_dim = N_MOVES
    elif args.features == "mine_signed":
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_mine_signed_input(Y, pos)
        input_dim = N_MOVES
    elif args.features == "board_state":
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_board_state_input(Y, pos)
        input_dim = 3 * 64
    elif args.features == "color_split":
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_color_split_input(X)
        input_dim = 2 * N_MOVES
    elif args.features == "played+halfmask":
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_played_halfmask_input(X)
        input_dim = 2 * N_MOVES
    elif args.features == "played+bit":
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_played_bit_input(X)
        input_dim = N_MOVES + 2
    elif args.features == "move_grid":
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_move_grid_input(X)
        input_dim = 60 * 60
    elif args.features == "move_grid_onehot":
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_move_grid_onehot_input(X)
        input_dim = 60 * 60 * 3
    print(f"Features: {args.features} ({input_dim}-d)")

    if args.color_specific:
        patterns = enumerate_flanking_patterns_color_specific()
        movers = precompute_movers_array(patterns)
        print(f"  color-specific mode: 1920 patterns, single model, "
              f"labels computed against absolute board state")
    else:
        patterns = enumerate_flanking_patterns()
        movers = None
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    pattern_to_cell = torch.tensor([MOVE_TO_IDX[p['target']] for p in patterns], dtype=torch.long)
    print(f"Device: {device}, Mode: {args.mode}, H={args.hidden}, {args.epochs} epochs")
    print(f"Patterns: {len(patterns)}")

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    feat_tag = "" if args.features == "when" else f"_{args.features.replace('+', '')}"
    save_path = os.path.join(save_dir,
        f"pattern_simple_{args.mode}_H{args.hidden}{feat_tag}.pt")

    board_loss_weight = 0.5 if args.mode == "e2e" else 0.0
    if args.mode == "randproj":
        dist_tag = "hn" if args.proj_half_normal else "n"
        save_path = os.path.join(save_dir,
            f"pattern_simple_randproj_s{args.seed}_H{args.hidden}_{dist_tag}{args.proj_scale:.2f}.pt")
    if args.pos_weight is not None:
        save_path = save_path.replace('.pt', f'_pw{int(args.pos_weight)}.pt')
    if args.color_specific:
        save_path = save_path.replace('.pt', '_colorspec.pt')
    if args.single_model:
        save_path = save_path.replace('.pt', '_single.pt')
    if args.legal_weight > 0:
        save_path = save_path.replace('.pt', f'_lw{args.legal_weight:g}.pt')
    if args.listwise_weight > 0:
        save_path = save_path.replace('.pt', f'_listw{args.listwise_weight:g}.pt')
    if args.loss != "bce":
        save_path = save_path.replace('.pt', f'_{args.loss}.pt')
    if args.l1_weight > 0:
        save_path = save_path.replace('.pt', f'_l1{args.l1_weight:g}.pt')

    train(chunk_dir, device, input_dim, args.hidden, args.mode,
          feature_cols, pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
          board_loss_weight=board_loss_weight, epochs=args.epochs,
          save_path=save_path, mlp_ckpt_dir=save_dir, seed=args.seed,
          pos_weight=args.pos_weight,
          proj_scale=args.proj_scale, proj_half_normal=args.proj_half_normal,
          single_model=args.single_model,
          legal_weight=args.legal_weight, pattern_to_cell=pattern_to_cell,
          loss_type=args.loss, feature_fn=feature_fn,
          pairwise_weight=args.pairwise_weight, pairwise_margin=args.pairwise_margin,
          pos_weight_end=args.pos_weight_end,
          listwise_weight=args.listwise_weight,
          chunk_prefix=args.chunk_prefix,
          movers=movers,
          resume_path=args.resume,
          l1_weight=args.l1_weight)

