"""Per-cell Ridge probe on Othello-GPT residual stream at chosen layer(s).

Mirrors probe_per_cell.py but for OGPT instead of the MLP. For each of 64
board cells, train a Linear(d_model, 3) probe from the OGPT residual-stream
activation (post block L) to ground-truth cell state, and report per-cell
accuracy. Three layers by default (2, 4, 6) so we see the representation
grow with depth.

Output should be directly comparable to the MLP's per-cell map:
  - If OGPT shows uniform ~99% across all cells -> qualitatively different
    representation; supports "OGPT does something the MLP can't".
  - If OGPT has the same heavy-flip -> low-acc gradient as the MLP -> OGPT
    is just a scaled-up version of the same pattern matching.
  - If OGPT has a different weak-cell pattern -> reveals what its depth
    is computing.

Usage:
    python probe_per_cell_ogpt.py \\
        --ckpt ckpts/gpt_nanda_synthetic.ckpt \\
        --layers 2,4,6 --n-games 5000
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState

# Reuse helpers / constants from probe_state_pred_for_othello.py
sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, extract_activations, load_games,
    VOCAB_SIZE, GAME_LEN, STOI, ITOS,
)


CENTER_64 = {27, 28, 35, 36}


def cell_class(c):
    if c in CENTER_64: return 'center'
    r, col = c // 8, c % 8
    if r in (0, 7) and col in (0, 7): return 'corner'
    if r in (0, 7) or col in (0, 7):  return 'edge'
    return 'inner'


def cell_alg(c):
    return f"{'abcdefgh'[c % 8]}{c // 8 + 1}"


def compute_board_states(games):
    """Replay games to get (G, T, 64) board states. -1 white, 0 empty, 1 black."""
    states = np.zeros((len(games), GAME_LEN, 64), dtype=np.int8)
    for i, game in enumerate(games):
        board = OthelloBoardState()
        for t, move in enumerate(game):
            board.umpire(move)
            states[i, t] = np.asarray(board.state).flatten().astype(np.int8)
    return states


def state_to_classes(state):
    """Map -1/0/1 to 0/1/2 (empty/white/black -- pick any consistent encoding).

    We use: 0 = empty, 1 = white, 2 = black. Probe accuracy is invariant
    to the label permutation.
    """
    out = np.zeros_like(state, dtype=np.int8)
    out[state == 0] = 0
    out[state == -1] = 1
    out[state == 1] = 2
    return out


def print_grid(acc, label):
    print(f"\nPer-cell board-state probe accuracy ({label}):")
    print("     " + " ".join(f"{c:>6s}" for c in "abcdefgh"))
    for r in range(8):
        row = []
        for c in range(8):
            sq = r * 8 + c
            v = acc[sq]
            tag = "*" if sq in CENTER_64 else " "
            row.append(f"{v:>5.3f}{tag}")
        print(f"  {r+1}  " + " ".join(row))


def print_summary(acc, label):
    print(f"\nSummary ({label}):")
    print(f"  mean: {acc.mean():.4f}  std: {acc.std():.4f}  "
          f"min: {acc.min():.4f} ({cell_alg(int(acc.argmin()))})  "
          f"max: {acc.max():.4f} ({cell_alg(int(acc.argmax()))})")
    print(f"  range: {acc.max() - acc.min():.4f}")
    print(f"  By class:")
    for cls in ('corner', 'edge', 'inner', 'center'):
        m = np.array([cell_class(c) == cls for c in range(64)])
        sub = acc[m]
        print(f"    {cls:>8s} n={int(m.sum()):2d}  mean={sub.mean():.4f}  "
              f"min={sub.min():.4f}  max={sub.max():.4f}")
    order = np.argsort(acc)
    print(f"  Worst 8: " + ", ".join(
        f"{cell_alg(int(c))}({cell_class(int(c))[:3]})={acc[int(c)]:.3f}"
        for c in order[:8]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    parser.add_argument("--layers", default="2,4,6",
                        help="Comma-separated layer indices to probe.")
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-games", type=int, default=5000)
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("--pos-start", type=int, default=5)
    parser.add_argument("--pos-end", type=int, default=55)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = [int(x) for x in args.layers.split(",")]

    state_dict = torch.load(args.ckpt, map_location=device)
    block_size = state_dict["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd)
    model = GPT(config)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f"Loaded OGPT from {args.ckpt} "
          f"(n_layer={args.n_layer}, n_embd={args.n_embd}, "
          f"block_size={block_size})")

    print(f"Loading games (max_files={args.max_files})...")
    games = load_games(max_files=args.max_files)
    if len(games) > args.n_games:
        games = games[:args.n_games]
    print(f"Using {len(games)} games")

    # Compute board states (G, T, 64)
    print("Replaying games to compute board states...")
    states = compute_board_states(games)
    Y = state_to_classes(states)  # (G, T, 64)
    print(f"States shape: {Y.shape}")

    # Tokenize
    print("Tokenizing...")
    tokens = tokenize_games(games, seq_len=block_size).to(device)

    all_results = {}
    for L in layers:
        print(f"\n{'='*68}\nProbing residual stream after block {L}\n{'='*68}")
        # Extract activations
        print("Forwarding through OGPT (batched)...")
        acts = []
        batch = 64
        with torch.no_grad():
            for i in range(0, len(games), batch):
                tk = tokens[i:i + batch]
                h = extract_activations(model, tk, L)  # (B, T, d_model)
                acts.append(h.cpu().numpy())
        H = np.concatenate(acts, axis=0)  # (G, T, d_model)
        print(f"  Activations shape: {H.shape}")

        # Slice to chosen position range and flatten
        Y_s = Y[:, args.pos_start:args.pos_end, :].reshape(-1, 64)
        H_s = H[:, args.pos_start:args.pos_end, :].reshape(
            -1, H.shape[-1])
        print(f"  After slicing/flatten: H_s={H_s.shape}, Y_s={Y_s.shape}")

        # 70/30 split
        N = H_s.shape[0]
        rng = np.random.RandomState(0)
        idx = rng.permutation(N)
        n_train = int(N * 0.7)
        tr_idx, te_idx = idx[:n_train], idx[n_train:]

        scaler = StandardScaler()
        Htr = scaler.fit_transform(H_s[tr_idx])
        Hte = scaler.transform(H_s[te_idx])
        print(f"  Train: {Htr.shape}  Test: {Hte.shape}")

        print("  Training 64 per-cell probes...")
        acc = np.zeros(64)
        for c in range(64):
            ytr = Y_s[tr_idx, c]; yte = Y_s[te_idx, c]
            if len(np.unique(ytr)) < 2:
                acc[c] = float('nan'); continue
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(Htr, ytr)
            acc[c] = clf.score(Hte, yte)

        all_results[L] = acc
        print_grid(acc, f"L={L}")
        print_summary(acc, f"L={L}")

    if args.output:
        np.savez(args.output, **{f"L{L}": v for L, v in all_results.items()})
        print(f"\nSaved per-cell accuracies to {args.output}")
