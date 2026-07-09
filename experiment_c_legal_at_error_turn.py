"""Fraction of adversarial positions where the illegal top-1 cell C is
LEGAL under the probe-decoded board at the error turn T.

This reproduces the original headline number from the causal analysis
(~90% at N=5000), now on the full 23k position set.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, VOCAB_SIZE, extract_activations,
)
from experiment_probe_causal_analysis import (
    next_hand_color_at_turn, flank_providing_directions,
)


def probe_argmax_state(hidden_512, turn, probe):
    mode = 0 if turn % 2 == 1 else 1
    W = probe[mode]
    h = torch.from_numpy(hidden_512).float()
    logits = torch.einsum('d,drco->rco', h, W)
    cls = logits.argmax(dim=-1).detach().cpu().numpy()
    st = np.zeros((8, 8), dtype=np.int8)
    st[cls == 1] = -1
    st[cls == 2] = 1
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_by_depth')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe',
                    default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    probe = torch.load(args.probe, map_location='cpu')
    assert probe.shape == (3, 512, 8, 8, 3)

    d = np.load(os.path.join(args.adversarial_dir, 'adversarial_records.npz'),
                allow_pickle=True)
    games = d['games']; turns = d['turns']; cells = d['illegal_cells']
    N = len(games) if args.limit == 0 else min(args.limit, len(games))
    print(f"Processing {N} positions")

    n_legal, n_illegal = 0, 0
    t0 = time.time()
    for i in range(N):
        game = list(games[i]); T = int(turns[i]); C = int(cells[i])
        L = min(T + 1, block_size)
        tokens = tokenize_games([game[:L]], seq_len=block_size).to(device)
        with torch.no_grad():
            acts = extract_activations(model, tokens, args.layer)
        acts_np = acts.cpu().numpy()[0]

        # Sanity: replay game to T to confirm C is illegal on actual board
        # (only needed if we want to report that; skipping for speed)

        probe_state = probe_argmax_state(acts_np[T], T, probe)
        color_next = next_hand_color_at_turn(T)
        flank_dirs = flank_providing_directions(probe_state, C, color_next)
        if len(flank_dirs) > 0:
            n_legal += 1
        else:
            n_illegal += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (N - i - 1) / (i + 1)
            print(f"  {i+1}/{N}  ({elapsed:.0f}s, ~{eta:.0f}s remaining)",
                   flush=True)

    total = n_legal + n_illegal
    print()
    print(f"Total positions:            {total}")
    print(f"C legal per probe at T:     {n_legal}  ({n_legal / total * 100:.2f}%)")
    print(f"C illegal per probe at T:   {n_illegal}  ({n_illegal / total * 100:.2f}%)")


if __name__ == '__main__':
    main()
