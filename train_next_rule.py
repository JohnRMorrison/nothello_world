"""Next-rule prediction: predict which of the 960 patterns fires on the
next move (or a "no-pattern" class for forfeits).

Model: Linear(input, H) -> ReLU -> Linear(H, 961)
Loss: soft-target cross-entropy with target = uniform over fired patterns
      (or 1.0 on the no-pattern class for forfeits).

Equivalent intuition: maximize the average log-probability assigned to the
patterns that actually fired. The model is rewarded for putting mass on
ANY of the fired patterns; ties between them are not penalized.

Usage:
    python train_next_rule.py --hidden 512 --epochs 3 [--features when+even]
"""
import sys, os, argparse, time, glob
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    get_device, N_MOVES,
)


N_PATTERNS = 960
N_CLASSES = N_PATTERNS + 1   # +1 for "no-pattern" class (forfeits)
NO_PATTERN = N_PATTERNS      # index 960


class NextRuleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def soft_target_ce(logits, target):
    """Soft-target cross-entropy: -sum(target * log_softmax(logits)) per row,
    averaged over batch. `target` should be a normalized probability dist."""
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def build_soft_target(fired, is_forfeit, device):
    """fired: (B, 960) uint8, is_forfeit: (B,) uint8. Returns (B, 961) float."""
    B = fired.shape[0]
    tgt = torch.zeros(B, N_CLASSES, dtype=torch.float32, device=device)
    tgt[:, :N_PATTERNS] = fired.float()
    tgt[is_forfeit.bool(), NO_PATTERN] = 1.0
    # Normalize per row (uniform over fired patterns; 1.0 on no-pattern)
    total = tgt.sum(dim=-1, keepdim=True).clamp_min(1.0)
    tgt = tgt / total
    return tgt


def feature_slice(feats, features_arg):
    """Slice 180-d features to the requested subset."""
    if features_arg == "all":
        return feats
    if features_arg == "when":
        return feats[:, N_MOVES:2 * N_MOVES]
    if features_arg == "when+even":
        return feats[:, N_MOVES:3 * N_MOVES]
    raise ValueError(f"Unknown features: {features_arg}")


def load_chunk(path):
    with np.load(path) as d:
        return (d['features'].astype(np.float32),
                d['fired'],
                d['is_forfeit'],
                d['positions'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--features", default="when+even",
                        choices=["when", "when+even", "all"])
    parser.add_argument("--chunk-glob", default="fired_patterns_*.npz")
    parser.add_argument("--output-dir",
        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}, H={args.hidden}, features={args.features}, "
          f"epochs={args.epochs}")

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, args.chunk_glob)))
    if not chunk_files:
        raise FileNotFoundError(f"No chunks matching {args.chunk_glob}")
    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    print(f"Train chunks: {len(train_paths)}, eval: {os.path.basename(eval_path)}")

    # Eval chunk (stride-slice for uniform turn coverage)
    ev_feats, ev_fired, ev_forfeit, ev_pos = load_chunk(eval_path)
    n_eval = min(len(ev_feats), 49 * 10000)
    stride = max(1, len(ev_feats) // n_eval)
    ev_feats = ev_feats[::stride][:n_eval]
    ev_fired = ev_fired[::stride][:n_eval]
    ev_forfeit = ev_forfeit[::stride][:n_eval]
    ev_pos = ev_pos[::stride][:n_eval]
    print(f"  Eval samples: {len(ev_feats)} (stride={stride})")
    print(f"  Eval forfeit rate: {ev_forfeit.mean()*100:.2f}%")

    ev_feats_sub = feature_slice(torch.from_numpy(ev_feats), args.features)
    input_dim = ev_feats_sub.shape[1]
    ev_fired_t = torch.from_numpy(ev_fired)
    ev_forfeit_t = torch.from_numpy(ev_forfeit)

    model = NextRuleMLP(input_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    feat_tag = args.features.replace("+", "")
    save_path = os.path.join(save_dir,
        f"next_rule_H{args.hidden}_{feat_tag}.pt")
    print(f"Save path: {save_path}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))
        epoch_loss = 0.0; n_batches = 0
        t0 = time.time()

        for ci in chunk_order:
            try:
                feats, fired, forfeit, pos = load_chunk(train_paths[ci])
            except Exception as e:
                print(f"  WARN: failed to load {train_paths[ci]}: {e}",
                      flush=True)
                continue
            feats = feature_slice(torch.from_numpy(feats), args.features)
            fired = torch.from_numpy(fired)
            forfeit = torch.from_numpy(forfeit)

            perm = torch.randperm(len(feats))
            for i in range(0, len(perm), args.batch_size):
                idx = perm[i:i + args.batch_size]
                x = feats[idx].to(device)
                f_mask = fired[idx].to(device)
                f_forfeit = forfeit[idx].to(device)
                target = build_soft_target(f_mask, f_forfeit, device)
                logits = model(x)
                loss = soft_target_ce(logits, target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += float(loss.item()); n_batches += 1
            del feats, fired, forfeit

        # Eval: report two metrics
        #  (1) "top-1 in fired": prob the argmax class is a fired pattern (or
        #      no-pattern on forfeits).
        #  (2) "log-prob fired": average log-prob assigned to fired patterns
        #      (= -loss, lower is better).
        model.eval()
        correct = 0; total = 0
        ev_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(ev_feats_sub), args.batch_size):
                x = ev_feats_sub[i:i + args.batch_size].to(device)
                f_mask = ev_fired_t[i:i + args.batch_size].to(device)
                f_forfeit = ev_forfeit_t[i:i + args.batch_size].to(device)
                target = build_soft_target(f_mask, f_forfeit, device)
                logits = model(x)
                ev_loss += soft_target_ce(logits, target).item() * len(x)
                preds = logits.argmax(dim=-1)
                # Correct if predicted index has nonzero target probability
                # (i.e., it's a fired pattern or the no-pattern class on forfeits)
                hit = (target.gather(1, preds.unsqueeze(1)).squeeze(1) > 0)
                correct += hit.sum().item()
                total += len(x)
        acc = correct / total
        ev_loss /= total
        dt = time.time() - t0
        print(f"Epoch {epoch}: train_loss={epoch_loss/max(n_batches,1):.4f}  "
              f"eval_loss={ev_loss:.4f}  top1_in_fired={acc:.4%}  "
              f"time={dt:.0f}s  lr={optimizer.param_groups[0]['lr']:.2e}",
              flush=True)

        scheduler.step(epoch_loss / max(n_batches, 1))
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'state_dict': {k: v.cpu().clone()
                               for k, v in model.state_dict().items()},
                'hidden': args.hidden,
                'input_dim': input_dim,
                'features': args.features,
                'best_acc': best_acc,
            }, save_path)
            print(f"  Saved {save_path}", flush=True)

    print(f"\n=== Final best top1-in-fired: {best_acc:.4%} ===")
