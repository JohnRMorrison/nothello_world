"""Train linear probe on hidden layer of pattern detector models.

Streams through all chunks (same as train_pattern_simple.py) to use
the full 12M games for probe training.

Tests whether board state is encoded in the hidden layer even when
the model wasn't trained to represent it (e.g., direct, emergent modes).

Usage:
    python probe_pattern_models.py --ckpt pattern_simple_direct_H512.pt --mode direct --hidden 512
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


def get_hidden(model, x, mode):
    """Extract H-d hidden activation (after first Linear + ReLU)."""
    with torch.no_grad():
        if mode in ("direct", "randproj"):
            return torch.relu(model.net[0](x))
        else:
            return torch.relu(model.backbone[0](x))


def train_probe(chunk_dir, device, model_even, model_odd, mode,
                feature_cols, hidden_dim, epochs=10, lr=1e-3, batch_size=1024,
                feature_fn=None, chunk_prefix="chunk_"):
    """Train linear probe streaming through all chunks."""

    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.startswith(chunk_prefix) and f.endswith(".npz")
                         and "_patterns" not in f and "_when60" not in f)
    if not chunk_files:
        raise ValueError(f"No chunks matching {chunk_prefix}*.npz in {chunk_dir}")

    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]

    print(f"Probe training: {len(chunk_files)} chunks, H={hidden_dim}, {epochs} epochs")

    # Even/odd probes (separate, matching the model split)
    probe_even = nn.Linear(hidden_dim, 64 * OPTIONS).to(device)
    probe_odd = nn.Linear(hidden_dim, 64 * OPTIONS).to(device)
    optimizer = torch.optim.Adam(
        list(probe_even.parameters()) + list(probe_odd.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    # Load eval data. For column-slice features we slice up front; for
    # derived features (e.g. move_grid expanding 180 -> 3600/10800-d) we
    # apply the transform per-batch later to avoid OOM on the full chunk.
    # Chunks are sorted by turn; stride-slice for uniform turn coverage.
    ev_X, ev_Y, ev_pos = _load_features(eval_path)
    if feature_cols is not None:
        ev_X = ev_X[:, feature_cols]
    n_eval = min(len(ev_X), 49 * 10000)
    stride = max(1, len(ev_X) // n_eval)
    ev_X = ev_X[::stride][:n_eval].clone()
    ev_Y = ev_Y[::stride][:n_eval].clone()
    ev_pos = ev_pos[::stride][:n_eval].clone()
    print(f"  Eval samples: {len(ev_X)} (stride={stride} across turns 5-53)")

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        probe_even.train(); probe_odd.train()
        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))
        epoch_loss = 0.0
        epoch_batches = 0

        for ci in chunk_order:
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            if feature_cols is not None:
                tr_X = tr_X[:, feature_cols]
            # NB: do NOT apply feature_fn here -- derived features like
            # move_grid expand 180-d -> 3600-d which OOMs on full chunks.
            # We apply feature_fn per-batch below.

            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), batch_size):
                idx = perm[i:i + batch_size]
                x_raw = tr_X[idx]
                y = tr_Y[idx].to(device)
                pos = tr_pos[idx]
                if feature_fn is not None:
                    x_raw = feature_fn(x_raw, tr_Y[idx], pos)
                x = x_raw.to(device)
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                loss = torch.tensor(0.0, device=device)
                if even_mask.any():
                    h = get_hidden(model_even, x[even_mask], mode)
                    logits = probe_even(h).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[even_mask].reshape(-1))
                if odd_mask.any():
                    h = get_hidden(model_odd, x[odd_mask], mode)
                    logits = probe_odd(h).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[odd_mask].reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1

            del tr_X, tr_Y, tr_pos

        # Eval
        probe_even.eval(); probe_odd.eval()
        correct = 0
        total = 0
        per_cell_correct = np.zeros(64, dtype=np.int64)
        n_eval_rows = 0
        with torch.no_grad():
            for i in range(0, len(ev_X), batch_size):
                x_raw = ev_X[i:i + batch_size]
                y_raw = ev_Y[i:i + batch_size]
                pos = ev_pos[i:i + batch_size]
                if feature_fn is not None:
                    x_raw = feature_fn(x_raw, y_raw, pos)
                x = x_raw.to(device)
                y = y_raw.to(device)
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask
                preds = torch.zeros_like(y)

                if even_mask.any():
                    h = get_hidden(model_even, x[even_mask], mode)
                    preds[even_mask] = probe_even(h).view(-1, 64, OPTIONS).argmax(-1)
                if odd_mask.any():
                    h = get_hidden(model_odd, x[odd_mask], mode)
                    preds[odd_mask] = probe_odd(h).view(-1, 64, OPTIONS).argmax(-1)

                correct += (preds == y).sum().item()
                total += y.numel()
                per_cell_correct += (preds == y).sum(dim=0).cpu().numpy()
                n_eval_rows += len(y)

        acc = correct / total
        per_cell_acc = per_cell_correct / max(n_eval_rows, 1)
        if acc > best_acc:
            best_acc = acc
            best_probe_state = {
                'even': {k: v.cpu().clone() for k, v in probe_even.state_dict().items()},
                'odd': {k: v.cpu().clone() for k, v in probe_odd.state_dict().items()},
                'per_cell_acc': per_cell_acc,
            }
        scheduler.step(epoch_loss / max(epoch_batches, 1))
        cur_lr = optimizer.param_groups[0]['lr']
        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"  Probe epoch {epoch}: acc={acc:.4%}  loss={avg_loss:.5f}  lr={cur_lr:.2e}",
              flush=True)

    return best_acc, best_probe_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to saved checkpoint")
    parser.add_argument("--mode", required=True,
                        choices=["direct", "emergent", "e2e", "two-stage", "randproj"])
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--chunk-prefix", default="chunk_",
                        help="Filename prefix for chunks to use (e.g. "
                             "'chunk_ext_' for extended-range chunks).")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Probing {args.mode} H={args.hidden} from {args.ckpt}")

    # Load checkpoint and reconstruct models
    ckpt = torch.load(args.ckpt, map_location=device)
    from train_pattern_simple import (
        DirectMLP, EndToEndMLP, TwoStageMLP,
        to_signed_parity_input, to_mine_signed_input, to_board_state_input,
        to_color_split_input, to_played_halfmask_input, to_played_bit_input,
        to_move_grid_input, to_move_grid_onehot_input,
    )
    n_patterns = ckpt.get('n_patterns', 960)
    input_dim = ckpt.get('input_dim', N_MOVES)

    # Infer feature preprocessing from input_dim / ckpt filename
    name = os.path.basename(args.ckpt)
    _feat_cols_map = {
        "when":        list(range(N_MOVES, 2 * N_MOVES)),
        "played":      list(range(0, N_MOVES)),
        "played+when": list(range(0, 2 * N_MOVES)),
        "when+even":   list(range(N_MOVES, 3 * N_MOVES)),
        "played+even": list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)),
        "all":         list(range(0, 3 * N_MOVES)),
    }
    feature_fn = None
    if "board_state" in name:
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_board_state_input(Y, pos)
    elif "wheneven" in name:
        feature_cols = _feat_cols_map["when+even"]
    elif "playedeven" in name:
        feature_cols = _feat_cols_map["played+even"]
    elif "signed_parity" in name:
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_signed_parity_input(X)
    elif "mine_signed" in name:
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_mine_signed_input(Y, pos)
    elif "color_split" in name:
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_color_split_input(X)
    elif "halfmask" in name:
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_played_halfmask_input(X)
    elif "bit" in name and "played" in name:
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_played_bit_input(X)
    elif "move_grid_onehot" in name:
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_move_grid_onehot_input(X)
    elif "move_grid" in name:
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_move_grid_input(X)
    elif input_dim == 62:
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_played_bit_input(X)
    elif input_dim == 3600:
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_move_grid_input(X)
    elif input_dim == 10800:
        feature_cols = None
        feature_fn = lambda X, Y, pos: to_move_grid_onehot_input(X)
    elif input_dim == 180:
        feature_cols = _feat_cols_map["all"]
    elif input_dim == 120:
        feature_cols = _feat_cols_map["when+even"]   # best guess
    elif input_dim == 60 and "played" in name:
        feature_cols = _feat_cols_map["played"]
    else:
        feature_cols = _feat_cols_map["when"]
    print(f"  input_dim={input_dim}")

    if args.mode in ("direct", "randproj"):
        model_even = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
        model_odd = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    elif args.mode == "two-stage":
        model_even = TwoStageMLP(input_dim, args.hidden, n_patterns).to(device)
        model_odd = TwoStageMLP(input_dim, args.hidden, n_patterns).to(device)
    else:
        model_even = EndToEndMLP(input_dim, args.hidden, n_patterns).to(device)
        model_odd = EndToEndMLP(input_dim, args.hidden, n_patterns).to(device)

    model_even.load_state_dict(ckpt['even'])
    model_odd.load_state_dict(ckpt['odd'])
    model_even.eval()
    model_odd.eval()
    # Freeze all model params (we only train the probe)
    for p in model_even.parameters():
        p.requires_grad = False
    for p in model_odd.parameters():
        p.requires_grad = False
    print(f"  Loaded checkpoint (pat_acc={ckpt.get('best_pat_acc', '?')})")

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")

    best_acc, best_probe_state = train_probe(
        chunk_dir, device, model_even, model_odd, args.mode,
        feature_cols, args.hidden, epochs=args.epochs, feature_fn=feature_fn,
        chunk_prefix=args.chunk_prefix)

    # Save probe weights. Derive suffix from source checkpoint filename so
    # probes for variants (_single, _pw50, _lw1, etc.) don't overwrite each other.
    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.basename(args.ckpt)
    if stem.startswith("pattern_simple_"):
        stem = stem[len("pattern_simple_"):]
    if stem.endswith(".pt"):
        stem = stem[:-3]
    probe_path = os.path.join(save_dir, f"probe_{stem}.pt")
    torch.save({
        'even': best_probe_state['even'],
        'odd': best_probe_state['odd'],
        'per_cell_acc': best_probe_state.get('per_cell_acc'),
        'hidden_dim': args.hidden,
        'best_acc': best_acc,
        'mode': args.mode,
    }, probe_path)
    print(f"Saved probe to {probe_path}")

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}, H={args.hidden}")
    print(f"Pattern accuracy: {ckpt.get('best_pat_acc', '?')}")
    print(f"Probe board accuracy: {best_acc:.4%}")
    print(f"{'='*60}")
