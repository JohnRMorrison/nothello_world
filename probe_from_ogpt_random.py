"""Train a Nanda-style board-state probe on a random projection of
OthelloGPT's INPUTS (token embeddings [+ position embeddings]).

For each game position, reconstruct the sequence of token embeddings
(+ position embeddings if --add-pos-emb) from chunk features. Then
apply one of three aggregations:

  --aggregation sequence : per-token Linear(512, 512) + ReLU; pad to
                            (60, 512), flatten to 30720-d probe input.
  --aggregation mean     : average the per-timestep embeddings across n
                            first, then a single Linear(512, 512)+ReLU.
  --aggregation sum      : like mean but sum.

Train a Nanda-style even/odd probe (Linear(input_dim, 64*3)) on the
result to decode the 64x3 board state. No Othello-GPT weights are
trained; only the random projection and probe are used.

Usage:
    python probe_from_ogpt_random.py \
        --ogpt-ckpt ckpts/gpt_nanda_synthetic.ckpt \
        --aggregation mean --activation relu --epochs 3
"""
import sys, os, argparse, math
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES, OPTIONS,
)


# Nanda's Othello-GPT uses block_size=59 (games are 60 moves but model predicts
# the last; positions 0..58 are inputs). We pad/clip to this length.
MAX_MOVES = 59


def features_to_tokens(played, when, max_n=MAX_MOVES):
    """Reconstruct token sequence from (played, when) features.

    Returns (tokens, n_moves) where tokens is a length-max_n int64 array
    (0 = padding, 1..60 = moves) and n_moves is how many moves were made.
    """
    # played: (N, 60)  bool-ish
    # when:   (N, 60)  float in (0, 1]
    B = len(played)
    tokens = np.zeros((B, max_n), dtype=np.int64)
    n_moves = np.zeros(B, dtype=np.int64)
    for b in range(B):
        played_idx = np.where(played[b] > 0.5)[0]
        if len(played_idx) == 0:
            continue
        steps = (when[b][played_idx] * 60 - 1).round().astype(np.int64)
        order = np.argsort(steps)
        sorted_idx = played_idx[order]
        n = min(len(sorted_idx), max_n)
        tokens[b, :n] = sorted_idx[:n] + 1     # +1 since token 0 = padding
        n_moves[b] = n
    return tokens, n_moves


def load_ogpt_embeddings(ckpt_path, device):
    """Load tok_emb and pos_emb from an OthelloGPT checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    # Checkpoint may be raw state_dict or wrapped dict.
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif isinstance(ckpt, dict) and any(k.startswith('tok_emb') for k in ckpt):
        sd = ckpt
    else:
        sd = ckpt
    tok_key = next(k for k in sd if k.endswith('tok_emb.weight'))
    pos_key = next(k for k in sd if k.endswith('pos_emb'))
    tok_emb = sd[tok_key].to(device)            # (vocab_size, n_embd)
    pos_emb = sd[pos_key].to(device).squeeze(0)  # (block_size, n_embd)
    return tok_emb, pos_emb


def build_random_projection(in_dim, out_dim, seed, device):
    """Kaiming-style random Linear weights (frozen)."""
    torch.manual_seed(seed)
    W = torch.randn(in_dim, out_dim, device=device) * math.sqrt(2.0 / in_dim)
    b = torch.zeros(out_dim, device=device)
    return W, b


def _apply_activation(x, name):
    if name == "relu":
        return torch.relu(x)
    if name == "swish":
        return torch.nn.functional.silu(x)
    raise ValueError(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ogpt-ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    parser.add_argument("--random-embeddings", action="store_true",
                        help="Instead of loading OGPT's learned tok_emb/pos_emb, "
                             "use freshly-initialized random tables (Kaiming). "
                             "Tests whether the sequence FORMAT carries enough info, "
                             "independent of OGPT's specific learned representation.")
    parser.add_argument("--n-embd", type=int, default=512,
                        help="Embedding dim when using --random-embeddings.")
    parser.add_argument("--vocab-size", type=int, default=61,
                        help="Vocab size when using --random-embeddings.")
    parser.add_argument("--aggregation",
                        choices=["sequence", "last_token", "mean", "sum"],
                        required=True,
                        help="sequence: flatten (59, 512) to 30192-d probe input; "
                             "last_token: use only the final valid token's 512-d activation; "
                             "mean/sum: pool over non-pad tokens first, then random project.")
    parser.add_argument("--normalize-input", action="store_true", default=True,
                        help="Divide embeddings by their std before random projection.")
    parser.add_argument("--no-normalize-input", dest="normalize_input", action="store_false")
    parser.add_argument("--activation", choices=["relu", "swish"], default="relu")
    parser.add_argument("--add-pos-emb", action="store_true", default=True,
                        help="Include position embeddings (default True).")
    parser.add_argument("--no-pos-emb", dest="add_pos_emb", action="store_false")
    parser.add_argument("--hidden", type=int, default=512,
                        help="Size of random projection output (not used for sequence mode).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Aggregation: {args.aggregation}  activation: {args.activation}  "
          f"add_pos_emb: {args.add_pos_emb}  seed: {args.seed}")

    # Load OGPT embeddings (frozen) — or use random tables.
    if args.random_embeddings:
        torch.manual_seed(args.seed)
        n_embd = args.n_embd
        # Kaiming-ish random tables, shape matching OGPT conventions.
        tok_emb = torch.randn(args.vocab_size, n_embd, device=device) * (1.0 / math.sqrt(n_embd))
        pos_emb = torch.randn(MAX_MOVES, n_embd, device=device) * (1.0 / math.sqrt(n_embd))
        print(f"Random embeddings: vocab={args.vocab_size}, n_embd={n_embd}, pos={MAX_MOVES}")
    else:
        tok_emb, pos_emb = load_ogpt_embeddings(args.ogpt_ckpt, device)
    n_embd = tok_emb.shape[1]
    assert pos_emb.shape[0] >= MAX_MOVES, \
        f"pos_emb has {pos_emb.shape[0]} positions but MAX_MOVES={MAX_MOVES}"
    pos_emb = pos_emb[:MAX_MOVES]   # trim if longer
    print(f"OGPT: vocab={tok_emb.shape[0]}, n_embd={n_embd}, block_size={pos_emb.shape[0]}")

    # Decide probe input_dim based on aggregation
    proj_in = n_embd
    proj_out = args.hidden
    probe_in = MAX_MOVES * proj_out if args.aggregation == "sequence" else proj_out
    print(f"proj: {proj_in} -> {proj_out}  probe_in: {probe_in}")

    # Input-normalization stats: std of the real move embeddings (excluding
    # token 0) plus pos_emb averaged over positions. Rough but stable.
    input_std = None
    if args.normalize_input:
        # Build a few (token + pos) vectors: one per valid token at one pos.
        tok_sample = tok_emb[1:]                          # (60, 512)
        pos_sample = pos_emb.mean(dim=0, keepdim=True)    # (1, 512)
        sample = tok_sample + pos_sample                   # (60, 512)
        input_std = sample.flatten().std().item()
        print(f"input normalization: dividing by std={input_std:.4f}")

    proj_W, proj_b = build_random_projection(proj_in, proj_out, args.seed, device)

    # Even/odd Nanda-style probes
    probe_even = nn.Linear(probe_in, 64 * OPTIONS).to(device)
    probe_odd = nn.Linear(probe_in, 64 * OPTIONS).to(device)
    optimizer = torch.optim.Adam(
        list(probe_even.parameters()) + list(probe_odd.parameters()), lr=args.lr)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    eval_path = chunk_files[-1]
    train_paths = chunk_files[:-1]
    print(f"Chunks: {len(chunk_files)} total, {len(train_paths)} train")

    played_cols = list(range(0, N_MOVES))
    when_cols = list(range(N_MOVES, 2 * N_MOVES))

    def compute_features(X, n_moves, tokens):
        """tokens: (B, MAX_MOVES) with 0 = padding. Returns (B, probe_in)."""
        B = len(tokens)
        # Embed: (B, MAX_MOVES, n_embd). tok_emb[0] = padding embedding (unused).
        tok_vecs = tok_emb[tokens]
        if args.add_pos_emb:
            tok_vecs = tok_vecs + pos_emb.unsqueeze(0)   # (1, MAX_MOVES, n_embd)
        # Zero out padding positions so they contribute nothing downstream.
        pad_mask = (tokens == 0)  # (B, MAX_MOVES)
        tok_vecs = tok_vecs.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        if input_std is not None:
            tok_vecs = tok_vecs / input_std

        if args.aggregation == "sequence":
            flat = tok_vecs.reshape(-1, n_embd)
            h = _apply_activation(flat @ proj_W + proj_b, args.activation)
            h = h.view(B, MAX_MOVES, proj_out)
            h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            return h.reshape(B, -1)

        if args.aggregation == "last_token":
            # Index of last valid token = n_moves - 1 (clamp to ≥ 0 if sample has no moves).
            last_idx = (torch.from_numpy(n_moves).to(tokens.device) - 1).clamp(min=0)
            # Gather per-sample last-token embedding
            gather_idx = last_idx.view(B, 1, 1).expand(-1, 1, n_embd)
            last_vec = tok_vecs.gather(1, gather_idx).squeeze(1)   # (B, n_embd)
            return _apply_activation(last_vec @ proj_W + proj_b, args.activation)

        if args.aggregation == "mean":
            counts = (~pad_mask).sum(dim=-1, keepdim=True).clamp(min=1).to(tok_vecs.dtype)
            agg = tok_vecs.sum(dim=1) / counts
            return _apply_activation(agg @ proj_W + proj_b, args.activation)

        # sum
        agg = tok_vecs.sum(dim=1)
        return _apply_activation(agg @ proj_W + proj_b, args.activation)

    def eval_pass():
        probe_even.eval(); probe_odd.eval()
        correct = 0; total = 0
        ev_X, ev_Y, ev_pos = _load_features(eval_path)
        n = min(len(ev_X), 49 * 10000)
        rng = np.random.RandomState(0)
        si = np.sort(rng.choice(len(ev_X), n, replace=False))
        ev_X, ev_Y, ev_pos = ev_X[si], ev_Y[si], ev_pos[si]
        with torch.no_grad():
            for i in range(0, n, args.batch_size):
                X = ev_X[i:i+args.batch_size]; Y = ev_Y[i:i+args.batch_size]
                pos = ev_pos[i:i+args.batch_size]
                played = X[:, played_cols].numpy().astype(np.float32)
                when = X[:, when_cols].numpy().astype(np.float32)
                tokens_np, n_moves = features_to_tokens(played, when)
                tokens = torch.from_numpy(tokens_np).to(device)
                h = compute_features(None, n_moves, tokens)
                y = Y.to(device); p = pos
                em = (p % 2 == 0); om = ~em
                preds = torch.zeros_like(y)
                if em.any():
                    logits = probe_even(h[em]).view(-1, 64, OPTIONS)
                    preds[em] = logits.argmax(-1)
                if om.any():
                    logits = probe_odd(h[om]).view(-1, 64, OPTIONS)
                    preds[om] = logits.argmax(-1)
                correct += (preds == y).sum().item()
                total += y.numel()
        return correct / total

    for epoch in range(1, args.epochs + 1):
        probe_even.train(); probe_odd.train()
        rng = np.random.RandomState(epoch)
        order = rng.permutation(len(train_paths))
        epoch_loss = 0.0; epoch_batches = 0
        for ci in order:
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), args.batch_size):
                idx = perm[i:i+args.batch_size]
                X = tr_X[idx]; Y = tr_Y[idx].to(device); pos = tr_pos[idx]
                played = X[:, played_cols].numpy().astype(np.float32)
                when = X[:, when_cols].numpy().astype(np.float32)
                tokens_np, n_moves = features_to_tokens(played, when)
                tokens = torch.from_numpy(tokens_np).to(device)
                h = compute_features(None, n_moves, tokens)
                em = (pos % 2 == 0); om = ~em

                loss = torch.tensor(0.0, device=device)
                if em.any():
                    logits = probe_even(h[em]).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), Y[em].reshape(-1))
                if om.any():
                    logits = probe_odd(h[om]).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, OPTIONS), Y[om].reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item(); epoch_batches += 1
            del tr_X, tr_Y, tr_pos
        avg = epoch_loss / max(epoch_batches, 1)
        acc = eval_pass()
        print(f"Epoch {epoch}: loss={avg:.5f}  board_acc={acc:.4%}", flush=True)
