"""Train a Nanda-style probe on the hidden of a specified MLP, restricted
to positions at a specific target turn.

Same architecture/loss as probe_pattern_models.py (Linear(H, 64*3), even/odd
split, BCE) but the training data is filtered to one turn so per-turn
specialists and the unified MLP can be probed apples-to-apples on their
turn-specific data distribution.

Usage:
    python train_per_turn_probe.py \\
        --ckpt experiments/.../pattern_simple_direct_H512_wheneven_turn25.pt \\
        --hidden 512 --target-turn 25 --epochs 3
"""
import sys, os, argparse, time
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES, OPTIONS,
)
from train_pattern_simple import DirectMLP


def get_hidden(model, x):
    return torch.relu(model.net[0](x))


def filter_to_turn(feat, Y, pos, target_turn):
    """Return tensors filtered to positions at target_turn."""
    pn = pos.numpy() if hasattr(pos, 'numpy') else np.asarray(pos)
    m = (pn == target_turn)
    if not m.any():
        return None
    idx = np.where(m)[0]
    return feat[idx], Y[idx], pos[idx]


def train(chunk_dir, device, model_even, model_odd, hidden_dim,
          target_turn, epochs, save_path, lr=1e-3, batch_size=1024):
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz")
                         and "_patterns" not in f
                         and "_when60" not in f
                         and not f.endswith("_by_black.npy"))
    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    print(f"Probe for turn={target_turn}: "
          f"{len(train_paths)} train chunks + eval={os.path.basename(eval_path)}")

    probe_even = nn.Linear(hidden_dim, 64 * OPTIONS).to(device)
    probe_odd  = nn.Linear(hidden_dim, 64 * OPTIONS).to(device)
    optimizer = torch.optim.Adam(
        list(probe_even.parameters()) + list(probe_odd.parameters()), lr=lr)

    # Eval data: filter to target turn (random sample of ~20k for speed)
    ev_X, ev_Y, ev_pos = _load_features(eval_path)
    ev_feat = ev_X[:, N_MOVES:3 * N_MOVES]   # when+even (120-d)
    del ev_X
    res = filter_to_turn(ev_feat, ev_Y, ev_pos, target_turn)
    if res is None:
        print(f"  No eval positions at turn {target_turn}")
        return
    ev_feat, ev_Y, ev_pos = res
    if len(ev_feat) > 50000:
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(len(ev_feat), 50000, replace=False))
        ev_feat, ev_Y, ev_pos = ev_feat[idx], ev_Y[idx], ev_pos[idx]
    print(f"  Eval samples: {len(ev_feat)}")

    best_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        probe_even.train(); probe_odd.train()
        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))
        epoch_loss = 0.0; n_batches = 0
        t0 = time.time()
        for ci in chunk_order:
            X, Y, pos = _load_features(train_paths[ci])
            feat = X[:, N_MOVES:3 * N_MOVES]
            del X
            res = filter_to_turn(feat, Y, pos, target_turn)
            if res is None:
                continue
            feat, Y, pos = res
            perm = torch.randperm(len(feat))
            for i in range(0, len(feat), batch_size):
                idx = perm[i:i + batch_size]
                x = feat[idx].to(device)
                y = Y[idx].to(device)
                pb = pos[idx]
                em = (pb % 2 == 0); om = ~em
                loss = torch.tensor(0.0, device=device)
                if em.any():
                    h = get_hidden(model_even, x[em])
                    logits = probe_even(h).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[em].reshape(-1))
                if om.any():
                    h = get_hidden(model_odd, x[om])
                    logits = probe_odd(h).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), y[om].reshape(-1))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += float(loss.item()); n_batches += 1
            del feat, Y, pos

        # Eval
        probe_even.eval(); probe_odd.eval()
        correct = 0; total = 0
        per_cell_correct = np.zeros(64, dtype=np.int64)
        per_cell_total = 0
        with torch.no_grad():
            for i in range(0, len(ev_feat), batch_size):
                x = ev_feat[i:i + batch_size].to(device)
                y = ev_Y[i:i + batch_size].to(device)
                pb = ev_pos[i:i + batch_size]
                em = (pb % 2 == 0); om = ~em
                preds = torch.zeros_like(y)
                if em.any():
                    h = get_hidden(model_even, x[em])
                    preds[em] = probe_even(h).view(-1, 64, OPTIONS).argmax(-1)
                if om.any():
                    h = get_hidden(model_odd, x[om])
                    preds[om] = probe_odd(h).view(-1, 64, OPTIONS).argmax(-1)
                correct += (preds == y).sum().item()
                total += y.numel()
                per_cell_correct += (preds == y).sum(dim=0).cpu().numpy()
                per_cell_total += len(y)
        acc = correct / total
        dt = time.time() - t0
        print(f"  Epoch {epoch}: acc={acc:.4%}  "
              f"loss={epoch_loss/max(n_batches,1):.5f}  time={dt:.0f}s", flush=True)
        if acc > best_acc:
            best_acc = acc
            per_cell_acc = per_cell_correct / max(per_cell_total, 1)
            best_state = {
                'even': {k: v.cpu().clone() for k, v in probe_even.state_dict().items()},
                'odd':  {k: v.cpu().clone() for k, v in probe_odd.state_dict().items()},
                'hidden_dim': hidden_dim,
                'best_acc': best_acc,
                'mode': 'direct',
                'target_turn': target_turn,
                'per_cell_acc': per_cell_acc,
            }
            torch.save(best_state, save_path)
            print(f"  Saved {save_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--target-turn", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    ckpt = torch.load(args.ckpt, map_location=device)
    input_dim = ckpt.get('input_dim', 120)
    me = DirectMLP(input_dim, args.hidden, 960).to(device)
    mo = DirectMLP(input_dim, args.hidden, 960).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()
    for p in me.parameters(): p.requires_grad = False
    for p in mo.parameters(): p.requires_grad = False
    print(f"Loaded {args.ckpt}  input_dim={input_dim}")

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.basename(args.ckpt)
    if stem.startswith("pattern_simple_"):
        stem = stem[len("pattern_simple_"):]
    if stem.endswith(".pt"):
        stem = stem[:-3]
    save_path = os.path.join(save_dir,
        f"probe_{stem}_turnprobe{args.target_turn}.pt")
    print(f"Save path: {save_path}")

    train(chunk_dir, device, me, mo, args.hidden,
          args.target_turn, args.epochs, save_path)
