"""Compute the sum_log_prob_or baseline top-K legality on a training chunk.

Used to establish the actual ceiling that the transformer aggregator has
to beat.  The `eval_multi_seed_aggregation.py` numbers are on 500 val
games (~24K positions) from ./data/othello_synthetic, which may or may
not match the distribution of the chunk_ext_*.npz eval set the transformer
trains against.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from train_multi_seed_mlp import (
    slice_played_even,
    _PAT_TARGETS, _PAT_TERMINALS, _PAT_OPP_CELLS, _PAT_OPP_MASK,
)
from train_pattern_simple import compute_pattern_labels_batch, _get_cell_pat_index
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from eval_multi_seed_ensemble import load_vectorized_from_multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi-ckpts', nargs='+', required=True)
    ap.add_argument('--chunk-dir',
                    default='experiments/mathematical_transformation_experiments/'
                            'heuristic_probe_results/feature_chunks')
    ap.add_argument('--chunk-glob', default='chunk_ext_*.npz')
    ap.add_argument('--chunk-idx', type=int, default=39)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--max-positions', type=int, default=1_000_000,
                    help='Cap positions read (default 1M — enough for stable '
                         'estimate, saves wall time on 16M+ chunks).')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading {len(args.multi_ckpts)} checkpoint(s)...")

    all_W1_e, all_b1_e, all_W2_e, all_b2_e = [], [], [], []
    all_W1_o, all_b1_o, all_W2_o, all_b2_o = [], [], [], []
    hidden = None
    for cp in args.multi_ckpts:
        me, mo, N, h, _ = load_vectorized_from_multi(cp, device)
        assert hidden is None or hidden == h
        hidden = h
        all_W1_e.append(me.W1.detach()); all_b1_e.append(me.b1.detach())
        all_W2_e.append(me.W2.detach()); all_b2_e.append(me.b2.detach())
        all_W1_o.append(mo.W1.detach()); all_b1_o.append(mo.b1.detach())
        all_W2_o.append(mo.W2.detach()); all_b2_o.append(mo.b2.detach())
        print(f"  {os.path.basename(cp)}: N={N}, H={h}")
    W1_e = torch.cat(all_W1_e, dim=0); b1_e = torch.cat(all_b1_e, dim=0)
    W2_e = torch.cat(all_W2_e, dim=0); b2_e = torch.cat(all_b2_e, dim=0)
    W1_o = torch.cat(all_W1_o, dim=0); b1_o = torch.cat(all_b1_o, dim=0)
    W2_o = torch.cat(all_W2_o, dim=0); b2_o = torch.cat(all_b2_o, dim=0)
    N_total = W1_e.shape[0]
    print(f"  Total N={N_total}")

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    chunks = sorted(glob.glob(os.path.join(args.chunk_dir, args.chunk_glob)))
    chunk_path = chunks[args.chunk_idx]
    print(f"Eval chunk: {os.path.basename(chunk_path)}")

    t0 = time.time()
    with np.load(chunk_path) as z:
        feats_180 = z['features'].astype(np.float16)
        board_labels = z['labels'].astype(np.int8)
        positions = z['positions'].astype(np.int64)
    feats_120 = slice_played_even(feats_180)
    n_rows = len(feats_120)
    print(f"  loaded n={n_rows:,} in {time.time()-t0:.1f}s")
    # The chunk stores rows in order by turn number (all games at turn 5,
    # then all games at turn 6, ...).  Slicing the first N rows would
    # sample early-game positions only — massively inflating accuracy.
    # Shuffle the row indices before capping so any max_positions cap gives
    # a representative sample.
    n_total = n_rows
    shuffled_idx = np.random.RandomState(0).permutation(n_total)
    if args.max_positions < n_total:
        shuffled_idx = shuffled_idx[:args.max_positions]
    n_rows = len(shuffled_idx)
    print(f"  evaluating {n_rows:,} positions (shuffled sample of {n_total:,})")

    KS = [1, 3, 5, 10]
    AGG_NAMES = ["sum_log_prob_or", "mean_prob_or", "majority_vote"]
    hits = {(a, K): 0 for a in AGG_NAMES for K in KS}
    per_seed_correct = np.zeros(N_total, dtype=np.int64)
    tot = 0

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_rows, args.batch_size):
            end = min(i + args.batch_size, n_rows)
            B = end - i
            batch_idx = shuffled_idx[i:end]
            x = torch.from_numpy(
                feats_120[batch_idx].astype(np.float32)).to(device)
            ks_t = torch.from_numpy(positions[batch_idx]).to(device)
            # Forward through all N MLPs.  Chunk positions use train convention:
            # even parity -> me, odd -> mo (NOT eval_multi_seed_ensemble.py's
            # inverted parity, which encodes k differently).
            use_me = (ks_t % 2 == 0); use_mo = ~use_me
            logits = torch.zeros(N_total, B, 960, device=device)

            def fwd(W1, b1, W2, b2, xs):
                x_nbi = xs.unsqueeze(0).expand(N_total, -1, -1)
                h = F.relu(torch.bmm(x_nbi, W1) + b1)
                return torch.bmm(h, W2) + b2

            if use_me.any():
                logits[:, use_me] = fwd(W1_e, b1_e, W2_e, b2_e, x[use_me])
            if use_mo.any():
                logits[:, use_mo] = fwd(W1_o, b1_o, W2_o, b2_o, x[use_mo])
            log1m = -F.softplus(logits)
            gathered = log1m[:, :, idx]
            gathered = gathered.masked_fill(~mask[None, None], 0.0)
            cell_scores = -gathered.sum(dim=-1)              # (N, B, 60)

            # Derive legal mask from pattern labels
            batch_pat = compute_pattern_labels_batch(
                board_labels[batch_idx].astype(np.int8),
                positions[batch_idx].astype(np.int64),
                _PAT_TARGETS, _PAT_TERMINALS,
                _PAT_OPP_CELLS, _PAT_OPP_MASK,
            )
            bp = torch.from_numpy((batch_pat > 0).astype(np.uint8)).to(device)
            g = bp[:, idx].masked_fill(~mask[None], 0)
            legal = (g.sum(dim=-1) > 0)                       # (B, 60)

            has_legal = legal.any(dim=1)
            if not has_legal.any():
                continue
            valid_scores = cell_scores[:, has_legal]           # (N, B', 60)
            valid_legal  = legal[has_legal]                    # (B', 60)

            # Per-seed top-1 correctness
            per_top1 = valid_scores.argmax(dim=-1)             # (N, B')
            per_correct = valid_legal.gather(
                1, per_top1.t()).t()                            # (N, B')
            per_seed_correct += per_correct.sum(dim=1).cpu().numpy()

            # Aggregators
            agg_sum = valid_scores.sum(dim=0)                  # (B', 60)
            prob_or = 1.0 - torch.exp(-valid_scores.clamp(min=0))
            agg_mean = prob_or.mean(dim=0)                     # (B', 60)
            votes = torch.zeros_like(agg_sum)
            votes.scatter_add_(
                1, per_top1.t(),
                torch.ones_like(per_top1.t(), dtype=torch.float32),
            )
            agg_mv = votes * 1e6 + agg_sum                     # tie-break

            for a_name, agg in [
                ("sum_log_prob_or", agg_sum),
                ("mean_prob_or",    agg_mean),
                ("majority_vote",   agg_mv),
            ]:
                for K in KS:
                    topk = agg.topk(K, dim=1).indices
                    hits[(a_name, K)] += valid_legal.gather(1, topk).sum().item()
            tot += int(has_legal.sum().item())

            if (i // args.batch_size) % 10 == 0:
                print(f"  {end:,}/{n_rows:,}  ({int(time.time()-t0)}s)",
                      flush=True)

    print()
    print(f"=== Baseline top-K on {os.path.basename(chunk_path)} "
          f"({tot:,} positions with ≥1 legal) ===")
    for a in AGG_NAMES:
        row = f"  {a:<20}"
        for K in KS:
            v = hits[(a, K)] / (tot * K)
            row += f"  top-{K}={v:.4f}"
        print(row)
    print()
    ind_acc = per_seed_correct / tot
    print(f"Individual seed top-1 accuracy:")
    print(f"  mean {ind_acc.mean():.4f}  std {ind_acc.std():.4f}  "
          f"min {ind_acc.min():.4f}  max {ind_acc.max():.4f}")


if __name__ == '__main__':
    main()
