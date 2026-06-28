"""Evaluate a next-cell-prediction MLP on legal-move prediction.

The MLP was trained to predict which cell will actually be played next.
Here we test whether it implicitly learned legality: does its output
correctly identify the SET of legal cells, not just the one played?

We use chunk_ext_*.npz chunks which contain pattern-legality labels
(960-d). A cell is legal iff any of its patterns is in the legal set.

Metrics:
  - Top-1 legal:  P(argmax cell is legal)
  - Top-3 legal:  P(top-3 cells are all legal)
  - Recall@K:     fraction of legal cells captured by top-K predictions
  - AUC:          how well logits separate legal from illegal cells
                  (averaged over positions, ignoring positions with 0
                   illegal cells)

Usage:
    python eval_next_cell_legality.py \\
        --ckpt experiments/.../next_cell_mlp_H512_move_grid.pt \\
        --chunks 2
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from train_next_cell_mlp_chunks import (
    NextCellMLP, to_move_grid_input, PATTERN_TO_CELL60,
)


def load_chunk_with_legal_cells(path, row_slice=2_000_000):
    """Load chunk_ext_NNNN.npz and derive 60-d legal-cell mask per row.

    Memory-efficient: processes labels in row slices so the full (N, 960)
    uint8 array (~17 GB for N=18M) is never simultaneously held with
    features and cell_legal.
    """
    with np.load(path) as z:
        feats = z['features'].astype(np.float16)
        N = feats.shape[0]
        # Load labels lazily and derive cell_legal in row slices.
        labels_arr = z['labels']    # NpzFile lazy reference; backed by zip stream
        positions = z['positions'].astype(np.int8) if 'positions' in z.files else None
        cell_legal = np.zeros((N, 60), dtype=np.uint8)
        for start in range(0, N, row_slice):
            end = min(start + row_slice, N)
            chunk_labels = np.asarray(labels_arr[start:end])      # (slice, 960) uint8
            # Vectorized scatter-max via PATTERN_TO_CELL60 grouping.
            for p in range(chunk_labels.shape[1]):
                c = PATTERN_TO_CELL60[p]
                np.maximum(cell_legal[start:end, c], chunk_labels[:, p],
                           out=cell_legal[start:end, c])
            del chunk_labels
    return feats, cell_legal, positions


def evaluate(model, chunks, device, batch_size=4096):
    """Run model over chunks; return aggregated metrics."""
    n_total = 0
    n_top1_legal = 0
    n_top3_all_legal = 0
    n_top5_all_legal = 0
    # Per-K legal counts: top-K predicted cells, sum legal among them
    topK_correct = {k: 0 for k in [1, 2, 3, 5, 10]}
    recall_at_K_sum = {k: 0.0 for k in [1, 2, 3, 5, 10]}
    # AUC: accumulate logits/labels per position to compute per-row ROC, average
    auc_sum = 0.0
    auc_count = 0

    with torch.no_grad():
        for ch_idx, chunk_path in enumerate(chunks):
            feats180, cell_legal, _ = load_chunk_with_legal_cells(chunk_path)
            print(f"Chunk {ch_idx+1}/{len(chunks)}: {os.path.basename(chunk_path)} "
                  f"n={len(feats180):,}")
            # Filter rows that have at least one legal cell
            n_legal_per_row = cell_legal.sum(axis=1)
            valid_idx = np.where(n_legal_per_row > 0)[0]
            for i in range(0, len(valid_idx), batch_size):
                bidx = valid_idx[i:i + batch_size]
                x180 = torch.from_numpy(feats180[bidx].astype(np.float32)).to(device)
                legal = torch.from_numpy(cell_legal[bidx]).to(device).float()  # (B, 60)
                x = to_move_grid_input(x180)
                logits = model(x)                                              # (B, 60)
                # Top-K metrics
                for k in topK_correct:
                    _, topk_idx = logits.topk(k, dim=-1)                       # (B, k)
                    topk_legal = legal.gather(1, topk_idx)                     # 0/1
                    topK_correct[k] += topk_legal.sum().item()
                    # Recall@K = top-K hits / total legal (for rows with legal>0)
                    n_legal_per_batch = legal.sum(dim=1).clamp_min(1.0)
                    recall_at_K_sum[k] += (topk_legal.sum(dim=1)
                                           / n_legal_per_batch).sum().item()
                # Top-1 legal, top-3 all legal, top-5 all legal
                top1_idx = logits.argmax(dim=-1)
                n_top1_legal += legal.gather(1, top1_idx.unsqueeze(1)).sum().item()
                _, top3_idx = logits.topk(3, dim=-1)
                top3_all = (legal.gather(1, top3_idx).sum(dim=1) == 3).sum().item()
                n_top3_all_legal += top3_all
                _, top5_idx = logits.topk(5, dim=-1)
                top5_all = (legal.gather(1, top5_idx).sum(dim=1) == 5).sum().item()
                n_top5_all_legal += top5_all
                # Per-row AUC: skip rows where all-legal or all-illegal
                logits_np = logits.cpu().numpy()
                legal_np = legal.cpu().numpy().astype(bool)
                for r in range(len(bidx)):
                    n_pos = legal_np[r].sum()
                    n_neg = 60 - n_pos
                    if n_pos == 0 or n_neg == 0:
                        continue
                    pos_scores = logits_np[r, legal_np[r]]
                    neg_scores = logits_np[r, ~legal_np[r]]
                    # AUC via tied-rank Mann-Whitney
                    n_concord = (pos_scores[:, None] > neg_scores[None, :]).sum()
                    n_tie = (pos_scores[:, None] == neg_scores[None, :]).sum()
                    auc = (n_concord + 0.5 * n_tie) / (n_pos * n_neg)
                    auc_sum += float(auc)
                    auc_count += 1
                n_total += len(bidx)

    return {
        'n_total':         n_total,
        'top1_legal':      n_top1_legal / n_total,
        'top3_all_legal':  n_top3_all_legal / n_total,
        'top5_all_legal':  n_top5_all_legal / n_total,
        'topK_avg_legal': {k: v / (n_total * k) for k, v in topK_correct.items()},
        'recall_at_K':    {k: v / n_total for k, v in recall_at_K_sum.items()},
        'mean_auc':       auc_sum / max(1, auc_count),
        'auc_count':      auc_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--chunks', type=int, default=2,
                    help='Number of chunk_ext_*.npz chunks to evaluate on')
    ap.add_argument('--chunk-glob', default='chunk_ext_*.npz')
    ap.add_argument('--chunk-dir',
                    default='experiments/mathematical_transformation_experiments/'
                            'heuristic_probe_results/feature_chunks')
    ap.add_argument('--batch-size', type=int, default=4096)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location='cpu')
    hidden = ckpt.get('hidden', 512)
    input_dim = ckpt.get('input_dim', 3600)
    print(f"Loading ckpt: H={hidden}, input_dim={input_dim}")
    model = NextCellMLP(input_dim=input_dim, hidden_dim=hidden, n_cells=60).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    chunks = sorted(glob.glob(os.path.join(args.chunk_dir, args.chunk_glob)))
    print(f"Found {len(chunks)} {args.chunk_glob} chunks; using last {args.chunks}")
    chunks = chunks[-args.chunks:]
    for c in chunks:
        print(f"  {os.path.basename(c)}")

    results = evaluate(model, chunks, device, args.batch_size)

    print()
    print("=" * 60)
    print(f"Total positions evaluated: {results['n_total']:,}")
    print(f"Mean per-row AUC: {results['mean_auc']:.4f}  "
          f"(n_valid_rows={results['auc_count']:,})")
    print()
    print("Top-K legality (model picks K cells, what fraction are legal):")
    for k, v in results['topK_avg_legal'].items():
        print(f"  Top-{k}: {v:.4f}")
    print()
    print("Recall@K (of all legal cells, what fraction captured by top-K):")
    for k, v in results['recall_at_K'].items():
        print(f"  Recall@{k}: {v:.4f}")
    print()
    print(f"Top-1 legal:        {results['top1_legal']:.4f}")
    print(f"Top-3 all legal:    {results['top3_all_legal']:.4f}")
    print(f"Top-5 all legal:    {results['top5_all_legal']:.4f}")


if __name__ == '__main__':
    main()
