"""OGPT top-1/3/5/10 legal-move accuracy using the same scoring as
compare_aggregators.py. For each (game, position) sample uniformly across
turns 5-53, count how many of the model's top-min(n,K) predictions are
actually legal, where K is the number of legal moves at that position.

Output mirrors the MLP's compare_aggregators output format so OGPT can be
slotted into the same table.

Usage:
    python ogpt_topk_legal.py --n-games 5000
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
c64_to_m60 = {c: i for i, c in enumerate(_movable_64)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    p.add_argument("--n-games", type=int, default=5000)
    p.add_argument("--max-files", type=int, default=2)
    p.add_argument("--pos-start", type=int, default=5)
    p.add_argument("--pos-end",   type=int, default=54)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config); model.load_state_dict(sd); model = model.to(device).eval()
    cell_tokens = torch.tensor([STOI[c] for c in _movable_64],
                                device=device, dtype=torch.long)

    games = load_games(max_files=args.max_files)
    games = [g for g in games if len(g) == GAME_LEN][:args.n_games]
    print(f"Using {len(games)} games")

    tokens = tokenize_games(games, seq_len=block_size).to(device)

    # Forward all in batches, keep only cell logits
    all_cell_logits = []
    batch = 32
    with torch.no_grad():
        for i in range(0, len(games), batch):
            out = model(tokens[i:i + batch])
            logits = out[0] if isinstance(out, tuple) else out
            cl = logits[:, :, cell_tokens]   # (B, T, 60)
            all_cell_logits.append(cl.cpu())
    all_cell_logits = torch.cat(all_cell_logits, dim=0)
    print(f"Cell logits shape: {tuple(all_cell_logits.shape)}")

    top_ns = (1, 3, 5, 10)
    results = {n: {'c': 0, 't': 0} for n in top_ns}

    for g_idx, game in enumerate(games):
        board = OthelloBoardState()
        for t in range(args.pos_end):
            if t >= args.pos_start:
                # Compute legal moves at this pre-move state
                legal_64 = board.get_valid_moves()
                legal_m60 = [c64_to_m60[c] for c in legal_64 if c in c64_to_m60]
                if not legal_m60:
                    pass
                else:
                    legal_set = set(legal_m60)
                    K = len(legal_set)
                    cl = all_cell_logits[g_idx, t, :].numpy()  # (60,)
                    order = np.argsort(-cl)
                    for n in top_ns:
                        k = min(n, K)
                        results[n]['c'] += len(set(order[:k].tolist()) & legal_set)
                        results[n]['t'] += k
            try:
                board.umpire(game[t])
            except Exception:
                break

    print()
    print(f"{'metric':>12s}  {'top-1':>9s}  {'top-3':>9s}  {'top-5':>9s}  {'top-10':>9s}")
    line = f"{'ogpt':>12s} "
    for n in top_ns:
        if results[n]['t'] == 0:
            line += "        -- "
        else:
            acc = results[n]['c'] / results[n]['t']
            line += f"  {acc*100:8.4f}%"
    print(line)
