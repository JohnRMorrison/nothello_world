"""Turn-specialist pattern detector training.

Filters chunk positions to a single target turn N, trains a DirectMLP
(when+even, H=512) just on positions at that turn. Optionally excludes
positions following a forfeit (replays game prefix via the 'when' channel
and checks if any prior turn lacked a legal move for the expected player).

Used to test whether the recall@K gap to OGPT is caused by training-data
dilution across turns (specialists should outperform unified at their turn)
or by a shared architectural ceiling (specialists won't help).

Usage:
    python train_turn_specialist.py --turn 25 --hidden 512 --epochs 3
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
from train_pattern_simple import DirectMLP, compute_pattern_labels_batch
from data.othello import OthelloBoardState


CENTER_64 = {27, 28, 35, 36}
_movable_64 = [c for c in range(64) if c not in CENTER_64]
_m60_to_b64 = np.array(_movable_64, dtype=np.int64)


def position_has_prior_forfeit(when_vec, turn):
    """Return True if any prior turn in this position's game prefix was a
    forfeit (the expected next-player had no legal moves, so the other
    player went). Replays the prefix via OthelloBoardState."""
    moves = []
    for c in range(60):
        w = when_vec[c]
        if w > 0:
            t = int(round(float(w) * N_MOVES)) - 1
            if 0 <= t < turn:
                moves.append((t, c))
    moves.sort()
    if len(moves) != turn:
        return False   # incomplete prefix; can't determine
    b = OthelloBoardState()
    for t, c60 in moves:
        c64 = int(_m60_to_b64[c60])
        expected = 1 if t % 2 == 0 else -1   # black plays even turns
        if b.next_hand_color != expected:
            return True
        try:
            b.umpire(c64)
        except Exception:
            return True
    return False


def filter_chunk(feat_X, Y, pos, target_turn, exclude_forfeit):
    """Return (feat_X, Y, pos) restricted to positions at target_turn,
    optionally excluding forfeit-affected positions."""
    m = (pos == target_turn) if hasattr(pos, 'numpy') else (pos.numpy() == target_turn)
    if hasattr(m, 'numpy'):
        m_np = m.numpy()
    else:
        m_np = m
    idx = np.where(m_np)[0]
    if len(idx) == 0:
        return None
    if exclude_forfeit:
        when = feat_X[idx, N_MOVES:2 * N_MOVES].numpy().astype(np.float32)
        keep = np.ones(len(idx), dtype=bool)
        for i, ii in enumerate(idx):
            if position_has_prior_forfeit(when[i], target_turn):
                keep[i] = False
        idx = idx[keep]
    return feat_X[idx], Y[idx], pos[idx]


def train(chunk_dir, device, hidden_dim, target_turn, epochs, save_path,
          pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
          exclude_forfeit=True, lr=1e-3, batch_size=1024):
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz")
                         and "_patterns" not in f
                         and "_when60" not in f
                         and not f.endswith("_by_black.npy"))
    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    print(f"Turn-{target_turn} specialist:")
    print(f"  Train chunks: {len(train_paths)}, eval: {os.path.basename(eval_path)}")
    print(f"  exclude_forfeit={exclude_forfeit}")

    input_dim = 120
    me = DirectMLP(input_dim, hidden_dim, 960).to(device)
    mo = DirectMLP(input_dim, hidden_dim, 960).to(device)
    optimizer = torch.optim.Adam(
        list(me.parameters()) + list(mo.parameters()), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    # Load eval chunk: filter to target_turn, exclude forfeits
    ev_X, ev_Y, ev_pos = _load_features(eval_path)
    ev_feat = ev_X[:, N_MOVES:3 * N_MOVES]   # when+even
    res = filter_chunk(ev_feat, ev_Y, ev_pos, target_turn, exclude_forfeit)
    if res is None:
        print(f"  No eval positions at turn {target_turn}")
        return
    ev_feat, ev_Y, ev_pos = res
    print(f"  Eval samples: {len(ev_feat)}")

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        me.train(); mo.train()
        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))
        epoch_loss = 0.0; n_batches = 0; t0 = time.time()
        for ci in chunk_order:
            X, Y, pos = _load_features(train_paths[ci])
            feat = X[:, N_MOVES:3 * N_MOVES]
            res = filter_chunk(feat, Y, pos, target_turn, exclude_forfeit)
            if res is None:
                continue
            feat, Y, pos = res
            perm = torch.randperm(len(feat))
            for i in range(0, len(feat), batch_size):
                idx = perm[i:i + batch_size]
                x = feat[idx].to(device)
                yb = Y[idx]; pb = pos[idx]
                em = (pb % 2 == 0); om = ~em
                labels = torch.from_numpy(compute_pattern_labels_batch(
                    yb.numpy(), pb.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask
                )).float().to(device)
                loss = torch.tensor(0.0, device=device)
                if em.any(): loss = loss + bce(me(x[em]), labels[em])
                if om.any(): loss = loss + bce(mo(x[om]), labels[om])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += float(loss.item()); n_batches += 1
            del feat, Y, pos

        # Eval
        me.eval(); mo.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for i in range(0, len(ev_feat), batch_size):
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
        acc = correct / total
        dt = time.time() - t0
        print(f"  Epoch {epoch}: pat_acc={acc:.4%}  "
              f"loss={epoch_loss/max(n_batches,1):.5f}  time={dt:.0f}s",
              flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'even': {k: v.cpu().clone() for k, v in me.state_dict().items()},
                'odd':  {k: v.cpu().clone() for k, v in mo.state_dict().items()},
                'input_dim': input_dim, 'hidden_dim': hidden_dim,
                'n_patterns': 960, 'best_pat_acc': best_acc,
                'target_turn': target_turn,
            }, save_path)
            print(f"  Saved {save_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--no-exclude-forfeit", action="store_true")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}, target_turn={args.turn}, H={args.hidden}, "
          f"epochs={args.epochs}")

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = \
        precompute_pattern_arrays(patterns)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    suffix = "" if args.no_exclude_forfeit else "_noffx"
    save_path = os.path.join(save_dir,
        f"pattern_simple_direct_H{args.hidden}_wheneven_turn{args.turn}{suffix}.pt")
    print(f"Save path: {save_path}")

    train(chunk_dir, device, args.hidden, args.turn, args.epochs, save_path,
          pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
          exclude_forfeit=not args.no_exclude_forfeit)
