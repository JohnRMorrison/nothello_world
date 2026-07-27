"""OGPT top-1 and recall@K legal-move accuracy, stratified by game turn.

Mirrors the per-turn legal-move analysis we did on the MLP, but for OGPT.
For each (game, position) pair, we compute Othello's legal-move set from
the replayed board, then check whether OGPT's top-K predictions match
(K = number of legal moves at that position).

Predicted contrast: OGPT stays at ~99%+ recall@K across all turns,
while the MLP's recall@K dips to ~85% in the mid-game.

Usage:
    python ogpt_recall_by_turn.py --n-games 5000
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F

from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, load_games, VOCAB_SIZE, GAME_LEN, STOI,
)


CENTER_64 = {27, 28, 35, 36}
_movable_64 = [c for c in range(64) if c not in CENTER_64]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    p.add_argument("--n-games", type=int, default=5000)
    p.add_argument("--max-files", type=int, default=2)
    p.add_argument("--random-init", action="store_true",
                   help="Untrained baseline: same GPT architecture, RANDOM "
                        "weights (do not load the ckpt).")
    p.add_argument("--out-npz", default=None,
                   help="Save per-move top-1 legality (turns, top1_by_move, "
                        "n_by_move, overall) for the by-move comparison figure.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    sd = torch.load(args.ckpt, map_location=device)   # for architecture shape
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    if args.random_init:
        torch.manual_seed(0)                          # reproducible random baseline
        print("RANDOM-INIT: untrained random weights (ckpt used only for shape)")
    else:
        model.load_state_dict(sd)
    model = model.to(device).eval()
    cell_tokens = torch.tensor([STOI[c] for c in _movable_64],
                                device=device, dtype=torch.long)

    games = load_games(max_files=args.max_files)
    if len(games) > args.n_games:
        games = games[:args.n_games]
    print(f"Using {len(games)} games")

    tokens = tokenize_games(games, seq_len=block_size).to(device)
    # Forward all games in batches; keep only cell-token logits to save memory.
    all_cell_logits = []
    batch = 32
    with torch.no_grad():
        for i in range(0, len(games), batch):
            out = model(tokens[i:i + batch])
            logits = out[0] if isinstance(out, tuple) else out   # (B, T, vocab)
            cl = logits[:, :, cell_tokens]   # (B, T, 60)
            all_cell_logits.append(cl.cpu())
    all_cell_logits = torch.cat(all_cell_logits, dim=0)   # (G, T, 60)
    print(f"Cell logits shape: {tuple(all_cell_logits.shape)}")

    # Map from movable_64 -> 0..59 index in cell_tokens order
    c64_to_m60 = {c64: i for i, c64 in enumerate(_movable_64)}

    # Per-MOVE stats (move number = the move being predicted = t+1), so this
    # lines up with the MLP script's stream-position convention.
    NMOVES = 60
    per_move_n = np.zeros(NMOVES, dtype=np.int64)
    per_move_top1 = np.zeros(NMOVES, dtype=np.int64)

    for g_idx, g in enumerate(games):
        b = OthelloBoardState()
        for t, move in enumerate(g):
            b.umpire(move)
            if t + 1 >= len(g):
                break
            m = t + 1                        # the move OGPT predicts at position t
            if m >= NMOVES:
                continue
            next_color = 1 if (t + 1) % 2 == 0 else -1
            board_copy = OthelloBoardState()
            board_copy.state = b.state.copy()
            board_copy.next_hand_color = next_color
            legal_set = set(board_copy.get_valid_moves())
            if len(legal_set) == 0:
                continue
            cl = all_cell_logits[g_idx, t]           # (60,)
            top1_64 = _movable_64[int(torch.argmax(cl))]
            per_move_n[m] += 1
            if top1_64 in legal_set:
                per_move_top1[m] += 1
        if (g_idx + 1) % 500 == 0:
            print(f"  game {g_idx+1}/{len(games)}", flush=True)

    top1_by_move = np.where(per_move_n > 0,
                            per_move_top1 / np.maximum(per_move_n, 1), np.nan)
    overall = per_move_top1.sum() / max(per_move_n.sum(), 1)
    label = "ogpt_random" if args.random_init else "ogpt"
    print()
    print(f"=== {label}: top-1 legality ===")
    print(f"OVERALL: {100*overall:.4f}%   (N={int(per_move_n.sum())})")
    for m in range(NMOVES):
        if per_move_n[m] > 0:
            print(f"  move {m:2d}: {100*top1_by_move[m]:7.3f}%  (n={per_move_n[m]})")
    if args.out_npz:
        np.savez(args.out_npz, model=label, turns=np.arange(NMOVES),
                 top1_by_move=top1_by_move, n_by_move=per_move_n,
                 overall=float(overall), n_total=int(per_move_n.sum()))
        print(f"saved {args.out_npz}")
