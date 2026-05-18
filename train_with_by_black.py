"""Train a pattern detector with forfeit-correct by_black feature appended
to when+even (180-d input total). Otherwise identical to direct-mode
training of train_pattern_simple.py: DirectMLP, even/odd parity split,
BCE on 960 pattern logits, streaming chunks.

Prereq: run precompute_by_black.py first to produce chunk_<NNNN>_by_black.npy
files alongside the existing chunks.

Usage:
    python train_with_by_black.py --hidden 512 --epochs 3
"""
import sys, os, argparse, time
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
)


def load_chunk_with_by_black(chunk_path, features="when+even+by_black"):
    """Returns (feat_X, Y, pos).

    features:
      when+even+by_black (default, 180-d)
      when+by_black      (120-d, drops the `even` channel)
      played+by_black    (120-d, minimal: was-played + which-color-placed)
    """
    X, Y, pos = _load_features(chunk_path)
    by_black_path = chunk_path.replace('.npz', '_by_black.npy')
    if not os.path.exists(by_black_path):
        raise FileNotFoundError(
            f"Missing {by_black_path}. Run precompute_by_black.py first.")
    by_black = torch.from_numpy(np.load(by_black_path).astype(np.float32))
    if features == "played+by_black":
        played = X[:, :N_MOVES]                          # 60-d (binary)
        feat = torch.cat([played, by_black], dim=1)      # 120-d
    elif features == "when+by_black":
        when = X[:, N_MOVES:2 * N_MOVES]                # 60-d
        feat = torch.cat([when, by_black], dim=1)        # 120-d
    elif features == "when+even+by_black":
        when_even = X[:, N_MOVES:3 * N_MOVES]            # 120-d
        feat = torch.cat([when_even, by_black], dim=1)   # 180-d
    else:
        raise ValueError(f"Unknown features: {features}")
    return feat, Y, pos


def train(chunk_dir, device, hidden_dim, epochs, save_path, features,
          pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
          lr=1e-3, batch_size=1024):
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.startswith("chunk_") and f.endswith(".npz"))
    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    print(f"Train chunks: {len(train_paths)}, eval chunk: {os.path.basename(eval_path)}")

    input_dim = 120 if features in ("when+by_black", "played+by_black") else 180
    me = DirectMLP(input_dim, hidden_dim, 960).to(device)
    mo = DirectMLP(input_dim, hidden_dim, 960).to(device)
    optimizer = torch.optim.Adam(
        list(me.parameters()) + list(mo.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)
    bce = nn.BCEWithLogitsLoss()

    # Load eval chunk once, stride-slice for uniform turn coverage.
    ev_feat, ev_Y, ev_pos = load_chunk_with_by_black(eval_path, features)
    n_eval = min(len(ev_feat), 49 * 10000)
    stride = max(1, len(ev_feat) // n_eval)
    ev_feat = ev_feat[::stride][:n_eval]
    ev_Y = ev_Y[::stride][:n_eval]
    ev_pos = ev_pos[::stride][:n_eval]
    print(f"  Eval samples: {len(ev_feat)} (stride={stride} across turns 5-53)")

    best_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        me.train(); mo.train()
        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))
        epoch_loss = 0.0
        n_batches = 0
        t_epoch = time.time()

        for ci in chunk_order:
            feat, Y, pos = load_chunk_with_by_black(train_paths[ci], features)
            perm = torch.randperm(len(feat))
            for i in range(0, len(feat), batch_size):
                idx = perm[i:i + batch_size]
                x = feat[idx].to(device)
                yb = Y[idx]
                pb = pos[idx]
                em = (pb % 2 == 0)
                om = ~em

                # Pattern labels
                labels = torch.from_numpy(compute_pattern_labels_batch(
                    yb.numpy(), pb.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask
                )).float().to(device)

                loss = torch.tensor(0.0, device=device)
                if em.any():
                    logits = me(x[em])
                    loss = loss + bce(logits, labels[em])
                if om.any():
                    logits = mo(x[om])
                    loss = loss + bce(logits, labels[om])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            del feat, Y, pos

        # Eval
        me.eval(); mo.eval()
        with torch.no_grad():
            correct = 0; total = 0
            for i in range(0, n_eval, batch_size):
                x = ev_feat[i:i + batch_size].to(device)
                yb = ev_Y[i:i + batch_size]
                pb = ev_pos[i:i + batch_size]
                em = (pb % 2 == 0); om = ~em
                labels = torch.from_numpy(compute_pattern_labels_batch(
                    yb.numpy(), pb.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask
                )).float().to(device)
                preds = torch.zeros(len(x), 960, device=device)
                if em.any(): preds[em] = (me(x[em]) > 0).float()
                if om.any(): preds[om] = (mo(x[om]) > 0).float()
                correct += ((preds == labels).float()).sum().item()
                total += labels.numel()
            pat_acc = correct / total

        avg = epoch_loss / max(n_batches, 1)
        scheduler.step(avg)
        dt = time.time() - t_epoch
        print(f"Epoch {epoch}: pat_acc={pat_acc:.4%}  loss={avg:.5f}  "
              f"time={dt:.0f}s  lr={optimizer.param_groups[0]['lr']:.2e}",
              flush=True)
        if pat_acc > best_acc:
            best_acc = pat_acc
            best_state = {
                'even': {k: v.cpu().clone() for k, v in me.state_dict().items()},
                'odd':  {k: v.cpu().clone() for k, v in mo.state_dict().items()},
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'n_patterns': 960,
                'best_pat_acc': best_acc,
            }
            torch.save(best_state, save_path)
            print(f"  Saved {save_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--features",
                        default="when+even+by_black",
                        choices=["when+even+by_black",
                                 "when+by_black",
                                 "played+by_black"],
                        help="Feature combo: 180-d default; 120-d variants drop "
                             "the even channel; played+by_black uses only the "
                             "played indicator (no timing).")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    input_dim = 120 if args.features in ("when+by_black", "played+by_black") else 180
    print(f"Device: {device}, H={args.hidden}, {args.epochs} epochs")
    print(f"Features: {args.features} ({input_dim}-d)")

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = \
        precompute_pattern_arrays(patterns)
    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    # Match existing naming: pattern_simple_direct_H512_wheneven_byblack.pt
    # "when+even+by_black" -> "wheneven_byblack"
    # "when+by_black"      -> "when_byblack"
    feat_tag = {
        "when+even+by_black": "wheneven_byblack",
        "when+by_black":      "when_byblack",
        "played+by_black":    "played_byblack",
    }[args.features]
    save_path = os.path.join(save_dir,
        f"pattern_simple_direct_H{args.hidden}_{feat_tag}.pt")
    print(f"Save path: {save_path}")

    train(chunk_dir, device, args.hidden, args.epochs, save_path,
          args.features,
          pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
