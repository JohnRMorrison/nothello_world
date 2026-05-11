"""CCGP on OGPT residual stream (the stub in compute_ccgp.py is filled in here).

Loads synthetic games, forwards through OGPT, extracts resid stream after a
chosen layer, replays games to compute board states, and runs the same
CCGP probes (phase and context) used for the MLP analysis. Result is
directly comparable to compute_ccgp.py output on the MLP.

For the no-world-model argument: if OGPT's CCGP gap is comparable to the
MLP's ~5-7pp, then Bernardi-style "abstract representation" analysis does
not distinguish OGPT from a 1-layer MLP.

Usage:
    python compute_ccgp_ogpt.py --layer 4 --n-games 3000 --ccgp-mode both
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch

from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, extract_activations, load_games, VOCAB_SIZE, GAME_LEN,
)
# Reuse the probe machinery from compute_ccgp.py
from compute_ccgp import ccgp_phase, ccgp_context, _print_summary


def compute_states_classes(games):
    """Replay games. Encode board state as 0=empty, 1=white(-1), 2=black(+1)
    to match MLP feature chunks (probes use classes 1 and 2 interchangeably)."""
    out = np.zeros((len(games), GAME_LEN, 64), dtype=np.int8)
    for i, g in enumerate(games):
        b = OthelloBoardState()
        for t, m in enumerate(g):
            b.umpire(m)
            s = np.asarray(b.state).flatten()
            out[i, t][s == 0]  = 0
            out[i, t][s == -1] = 1
            out[i, t][s == 1]  = 2
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ogpt-ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--n-games", type=int, default=3000)
    p.add_argument("--max-files", type=int, default=3)
    p.add_argument("--pos-start", type=int, default=5)
    p.add_argument("--pos-end", type=int, default=55)
    p.add_argument("--ccgp-mode", choices=["phase", "context", "both"],
                   default="both")
    p.add_argument("--n-bins", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(args.ogpt_ckpt, map_location=device)
    block_size = state_dict["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd)
    model = GPT(config)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f"Loaded OGPT from {args.ogpt_ckpt} "
          f"(block_size={block_size}, layer to probe = {args.layer})")

    print(f"Loading games (max_files={args.max_files})...")
    games = load_games(max_files=args.max_files)
    if len(games) > args.n_games:
        games = games[:args.n_games]
    print(f"Using {len(games)} games")

    print("Replaying games to compute board states...")
    Y_full = compute_states_classes(games)   # (G, T, 64) in {0,1,2}
    print(f"  States shape: {Y_full.shape}")

    print("Tokenizing...")
    tokens = tokenize_games(games, seq_len=block_size).to(device)

    print(f"Forwarding through OGPT (batched), extracting layer {args.layer}...")
    acts = []
    batch = 64
    with torch.no_grad():
        for i in range(0, len(games), batch):
            tk = tokens[i:i + batch]
            h = extract_activations(model, tk, args.layer)  # (B, T, d_model)
            acts.append(h.cpu().numpy().astype(np.float32))
    H_full = np.concatenate(acts, axis=0)   # (G, T, d_model)
    print(f"  Activations shape: {H_full.shape}")

    # Slice to chosen position range and flatten across games and turns.
    Y = Y_full[:, args.pos_start:args.pos_end, :].reshape(-1, 64)
    H = H_full[:, args.pos_start:args.pos_end, :].reshape(-1, H_full.shape[-1])
    G_, _, _ = Y_full.shape
    T_ = args.pos_end - args.pos_start
    pos = np.tile(np.arange(args.pos_start, args.pos_end, dtype=np.int32),
                  G_)
    print(f"Flattened: H={H.shape}, Y={Y.shape}, pos={pos.shape}, "
          f"turn_range=[{int(pos.min())}, {int(pos.max())}]")

    if args.ccgp_mode in ("phase", "both"):
        res = ccgp_phase(H, Y, pos, n_bins=args.n_bins)
        _print_summary(f"Option A: cross-game-phase ({args.n_bins} bins) [OGPT L{args.layer}]", res)
    if args.ccgp_mode in ("context", "both"):
        res = ccgp_context(H, Y, pos)
        _print_summary(f"Option B: cross-context-cell [OGPT L{args.layer}]", res)
