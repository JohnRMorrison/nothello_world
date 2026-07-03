"""Train a transformer aggregator over MLP-token predictions.

Loads one or more multi-seed checkpoints and concatenates all seeds into a
single ensemble.  Each training batch:
  1. Reads B positions from a chunk (played+even features + board state).
  2. Forwards through all N MLPs with no_grad -> (B, N, 60) cell scores.
  3. Feeds to the transformer with the 120-d board context concatenated to
     each MLP-token.  Transformer self-attention runs across the N tokens.
  4. Each refined MLP token emits (i) its own 60-d prediction and (ii) a
     softmax weight; final output is the weighted sum.

Only the transformer weights are trained.  The ensemble MLPs are frozen.

Usage:
    python train_mlp_token_transformer.py \\
        --multi-ckpts A.pt B.pt \\
        --train-chunk-start 20 --num-train-chunks 2 \\
        --eval-chunk 39 --epochs 3
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from train_multi_seed_mlp import (
    VectorizedMLP, slice_played_even,
    _PAT_TARGETS, _PAT_TERMINALS, _PAT_OPP_CELLS, _PAT_OPP_MASK,
)
from train_pattern_simple import compute_pattern_labels_batch, _get_cell_pat_index
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from eval_multi_seed_ensemble import load_vectorized_from_multi


N_CELLS = 60
N_PATTERNS = 960
INPUT_DIM = 120


def load_ensemble(ckpt_paths, device):
    """Concat all seeds from multiple checkpoints into one big VectorizedMLP
    per parity.  All checkpoints must share H and input_dim."""
    all_W1_e, all_b1_e, all_W2_e, all_b2_e = [], [], [], []
    all_W1_o, all_b1_o, all_W2_o, all_b2_o = [], [], [], []
    hidden = None
    for cp in ckpt_paths:
        me, mo, N, h, _ = load_vectorized_from_multi(cp, device)
        assert hidden is None or hidden == h, \
            f"Mixed hidden dims across checkpoints not supported"
        hidden = h
        all_W1_e.append(me.W1.detach()); all_b1_e.append(me.b1.detach())
        all_W2_e.append(me.W2.detach()); all_b2_e.append(me.b2.detach())
        all_W1_o.append(mo.W1.detach()); all_b1_o.append(mo.b1.detach())
        all_W2_o.append(mo.W2.detach()); all_b2_o.append(mo.b2.detach())
        print(f"    {os.path.basename(cp)}: N={N}, H={h}", flush=True)
    W1_e = torch.cat(all_W1_e, dim=0); b1_e = torch.cat(all_b1_e, dim=0)
    W2_e = torch.cat(all_W2_e, dim=0); b2_e = torch.cat(all_b2_e, dim=0)
    W1_o = torch.cat(all_W1_o, dim=0); b1_o = torch.cat(all_b1_o, dim=0)
    W2_o = torch.cat(all_W2_o, dim=0); b2_o = torch.cat(all_b2_o, dim=0)
    N_total = W1_e.shape[0]
    print(f"  Total ensemble: N={N_total} seeds, H={hidden}", flush=True)
    return (W1_e, b1_e, W2_e, b2_e, W1_o, b1_o, W2_o, b2_o), N_total, hidden


@torch.no_grad()
def compute_cell_scores(x, ks_t, weights, idx, mask, N, device):
    """Forward through all N MLPs, return (B, N, 60) prob_or cell scores.

    Routes chunk `positions` values to me/mo following train_multi_seed_mlp.py's
    convention: even positions -> me, odd -> mo.  NB: eval_multi_seed_ensemble.py
    uses the OPPOSITE parity because it stores `k` = "moves played before this
    position", not the position's own turn number.  Since chunk `positions`
    matches training semantics directly, we mirror the training convention here."""
    W1_e, b1_e, W2_e, b2_e, W1_o, b1_o, W2_o, b2_o = weights
    use_me = (ks_t % 2 == 0); use_mo = ~use_me
    B = x.shape[0]
    logits = torch.zeros(N, B, N_PATTERNS, device=device)

    def fwd(W1, b1, W2, b2, xs):
        x_nbi = xs.unsqueeze(0).expand(N, -1, -1)
        h = F.relu(torch.bmm(x_nbi, W1) + b1)
        return torch.bmm(h, W2) + b2

    if use_me.any():
        logits[:, use_me] = fwd(W1_e, b1_e, W2_e, b2_e, x[use_me])
    if use_mo.any():
        logits[:, use_mo] = fwd(W1_o, b1_o, W2_o, b2_o, x[use_mo])
    log1m = -F.softplus(logits)
    gathered = log1m[:, :, idx]
    gathered = gathered.masked_fill(~mask[None, None], 0.0)
    cell_scores = -gathered.sum(dim=-1)                # (N, B, 60)
    # Return both:
    #   cell_scores (B, N, 60) — used as the residual baseline
    #   pattern_logits (B, N, 960) — richer per-token input for the transformer
    return cell_scores.permute(1, 0, 2), logits.permute(1, 0, 2)


def derive_legal_mask(batch_pat, idx, mask):
    """(B, 960) uint8 pattern fires -> (B, 60) bool legal mask.
    Cell c is legal iff any pattern targeting c fires."""
    bp = torch.from_numpy(batch_pat).to(idx.device)     # (B, 960)
    gathered = bp[:, idx]                                # (B, 60, K)
    gathered = gathered.masked_fill(~mask[None], 0)
    return (gathered.sum(dim=-1) > 0)                    # (B, 60)


class MLPTokenTransformer(nn.Module):
    """MLPs are tokens.  Each token carries the MLP's RAW 960-d pattern
    logits (not the 60-d aggregated cell scores) plus the board context.
    Self-attention lets tokens see each other; each token emits its own
    60-d "delta"; final output = mean(cell_scores) baseline + weighted
    delta.

    Using pattern logits (960) instead of cell scores (60) as the token
    input preserves per-pattern nuance that the prob_or aggregation would
    have collapsed before the transformer sees anything.

    The residual baseline still uses the aggregated cell scores so we start
    from the sum_log_prob_or output; per_mlp_pred is zero-initialised so
    delta=0 at init.

    Uses seq-first tensor layout (S, B, E) for compatibility with older
    PyTorch that doesn't support batch_first=True."""
    def __init__(self, n_seeds, n_cells=60, n_patterns=960, ctx_dim=120,
                  d_model=64, n_heads=4, n_layers=2, dim_ff=128, dropout=0.0):
        super().__init__()
        self.token_proj = nn.Linear(n_patterns + ctx_dim, d_model)
        self.mlp_id_emb = nn.Embedding(n_seeds, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            activation='relu',
        )
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.per_mlp_pred = nn.Linear(d_model, n_cells)
        self.per_mlp_weight = nn.Linear(d_model, 1)
        # Zero-init the delta head so at init the forward pass is exactly
        # the sum_log_prob_or baseline (delta = 0).
        nn.init.zeros_(self.per_mlp_pred.weight)
        nn.init.zeros_(self.per_mlp_pred.bias)

    def forward(self, pattern_logits, cell_scores, context):
        # pattern_logits: (B, N, 960) — richer per-token input
        # cell_scores:    (B, N, 60)  — used only for the residual baseline
        # context:        (B, 120)
        B, N, _ = pattern_logits.shape
        ctx = context.unsqueeze(1).expand(B, N, -1)              # (B, N, 120)
        x = torch.cat([pattern_logits, ctx], dim=-1)             # (B, N, 960+120)
        x = self.token_proj(x)                                    # (B, N, d)
        ids = torch.arange(N, device=x.device)
        x = x + self.mlp_id_emb(ids)                              # broadcast
        # Seq-first for older PyTorch: (N, B, d)
        x = x.transpose(0, 1)
        x = self.transformer(x)                                   # (N, B, d)
        x = x.transpose(0, 1)                                     # (B, N, d)
        preds = self.per_mlp_pred(x)                              # (B, N, 60)
        weights = F.softmax(
            self.per_mlp_weight(x).squeeze(-1), dim=1)            # (B, N)
        delta = (weights.unsqueeze(-1) * preds).sum(dim=1)         # (B, 60)
        # Residual baseline: MEAN of per-seed CELL SCORES (aggregated
        # prob_or).  Argmax of mean == argmax of sum, so top-K matches
        # sum_log_prob_or at init.
        baseline = cell_scores.mean(dim=1)                         # (B, 60)
        return baseline + delta


def iter_batches(chunk_paths, batch_size):
    """Yield (feats_120_np, positions_np, legal_pat_np) per batch across chunks."""
    for cp in chunk_paths:
        t0 = time.time()
        with np.load(cp) as z:
            feats_180 = z['features'].astype(np.float16)
            board_labels = z['labels'].astype(np.int8)
            positions = z['positions'].astype(np.int64)
        feats_120 = slice_played_even(feats_180)
        n_rows = len(feats_120)
        print(f"  loaded {os.path.basename(cp)} n={n_rows:,} "
              f"in {time.time()-t0:.1f}s", flush=True)
        perm = np.random.permutation(n_rows)
        for i in range(0, n_rows, batch_size):
            idxs = perm[i:i + batch_size]
            batch_pat = compute_pattern_labels_batch(
                board_labels[idxs].astype(np.int8),
                positions[idxs].astype(np.int64),
                _PAT_TARGETS, _PAT_TERMINALS, _PAT_OPP_CELLS, _PAT_OPP_MASK,
            )
            yield (
                feats_120[idxs].astype(np.float32),
                positions[idxs].astype(np.int64),
                (batch_pat > 0).astype(np.uint8),
            )


def eval_topk_legality(model, chunk_path, weights, idx, mask, N, device,
                        batch_size, KS=(1, 3, 5, 10)):
    """Return dict K -> top-K legality on the eval chunk."""
    model.eval()
    hits = {K: 0 for K in KS}
    tot = 0
    with torch.no_grad():
        for feats, pos_np, pat_np in iter_batches([chunk_path], batch_size):
            x = torch.from_numpy(feats).to(device)
            ks_t = torch.from_numpy(pos_np).to(device)
            cell_scores, pattern_logits = compute_cell_scores(
                x, ks_t, weights, idx, mask, N, device)
            logits = model(pattern_logits, cell_scores, x)     # (B, 60)
            legal = derive_legal_mask(pat_np, idx, mask)       # (B, 60)
            # Skip positions with no legal moves.
            has_legal = legal.any(dim=1)
            if not has_legal.any():
                continue
            log_ok = logits[has_legal]
            legal_ok = legal[has_legal]
            for K in KS:
                topk = log_ok.topk(K, dim=1).indices
                hits[K] += legal_ok.gather(1, topk).sum().item()
            tot += int(has_legal.sum().item())
    return {K: hits[K] / (tot * K) for K in KS}, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi-ckpts', nargs='+', required=True,
                    help='One or more multi-seed checkpoint paths.')
    ap.add_argument('--chunk-dir',
                    default='experiments/mathematical_transformation_experiments/'
                            'heuristic_probe_results/feature_chunks')
    ap.add_argument('--chunk-glob', default='chunk_ext_*.npz')
    ap.add_argument('--train-chunk-start', type=int, default=20,
                    help='Index of first training chunk (default 20 to avoid '
                         'the chunks used to train the MLPs).')
    ap.add_argument('--num-train-chunks', type=int, default=2,
                    help='Number of chunks for training (~500K games each).')
    ap.add_argument('--eval-chunk', type=int, default=39,
                    help='Chunk index for held-out top-K eval.')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--n-heads', type=int, default=4)
    ap.add_argument('--n-layers', type=int, default=2)
    ap.add_argument('--dim-ff', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-path', default=None,
                    help='Where to save the trained transformer.')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    print(f"Loading {len(args.multi_ckpts)} checkpoint(s)...", flush=True)
    weights, N_total, hidden = load_ensemble(args.multi_ckpts, device)

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, N_CELLS)

    chunks = sorted(glob.glob(os.path.join(args.chunk_dir, args.chunk_glob)))
    train_chunks = chunks[
        args.train_chunk_start:
        args.train_chunk_start + args.num_train_chunks]
    eval_chunk = chunks[args.eval_chunk]
    print(f"Train chunks: {[os.path.basename(c) for c in train_chunks]}",
          flush=True)
    print(f"Eval chunk:   {os.path.basename(eval_chunk)}", flush=True)

    model = MLPTokenTransformer(
        n_seeds=N_total,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, dim_ff=args.dim_ff,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.75, patience=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer params: {n_params:,}", flush=True)

    save_path = args.save_path
    if save_path is None:
        ckpt_tags = "_".join(os.path.basename(c).replace('.pt', '')
                              for c in args.multi_ckpts)
        save_path = f"mlp_token_transformer_N{N_total}_d{args.d_model}.pt"
    print(f"Save path: {save_path}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        total_loss = 0.0
        total_n = 0
        n_batches = 0
        for feats, pos_np, pat_np in iter_batches(train_chunks, args.batch_size):
            x = torch.from_numpy(feats).to(device)
            ks_t = torch.from_numpy(pos_np).to(device)
            cell_scores, pattern_logits = compute_cell_scores(
                x, ks_t, weights, idx, mask, N_total, device)
            legal = derive_legal_mask(pat_np, idx, mask).float()
            logits = model(pattern_logits, cell_scores, x)
            loss = F.binary_cross_entropy_with_logits(logits, legal)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * x.shape[0]
            total_n += x.shape[0]
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"    batch {n_batches}: loss={total_loss/total_n:.4f}  "
                      f"({int(time.time()-t_epoch)}s)", flush=True)

        # End-of-epoch eval on held-out chunk
        eval_res, n_eval = eval_topk_legality(
            model, eval_chunk, weights, idx, mask, N_total, device,
            args.batch_size)
        scheduler.step(total_loss / max(1, total_n))
        cur_lr = opt.param_groups[0]['lr']
        summary = ", ".join(f"top{K}={v:.4f}" for K, v in eval_res.items())
        print(f"Epoch {epoch}: n={total_n:,}  loss={total_loss/total_n:.4f}  "
              f"lr={cur_lr:.2e}  eval(n={n_eval:,}): {summary}  "
              f"({int(time.time()-t_epoch)}s)", flush=True)

        # Save transformer weights (atomic).
        tmp = save_path + ".tmp"
        torch.save({
            'model_state': model.state_dict(),
            'args':        vars(args),
            'ckpt_paths':  args.multi_ckpts,
            'N_total':     N_total,
            'hidden':      hidden,
            'epoch':       epoch,
            'eval_result': eval_res,
        }, tmp)
        os.replace(tmp, save_path)
        print(f"    SAVED transformer to {save_path}", flush=True)


if __name__ == '__main__':
    main()
