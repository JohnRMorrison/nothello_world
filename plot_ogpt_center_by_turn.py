"""Center-cell decoding accuracy vs game turn, for Nanda's probe on OGPT.

Uses main_linear_probe.pth (the 95.88% probe) applied per-turn on
OGPT layer-6 activations. The 4 center cells (27, 28, 35, 36 = d4, e4,
d5, e5) are averaged per turn; the curve shows turn-by-turn mean accuracy.

Usage:
    python plot_ogpt_center_by_turn.py --n-games 500
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, load_games, extract_activations, VOCAB_SIZE, GAME_LEN,
)


CENTER_RC = {(3, 3), (3, 4), (4, 3), (4, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    parser.add_argument("--probe", default="mechanistic_interpretability/main_linear_probe.pth")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--n-games", type=int, default=500)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--pos-start", type=int, default=5)
    parser.add_argument("--pos-end",   type=int, default=59,
                        help="Exclusive upper bound. OGPT block_size is 59.")
    parser.add_argument("--output",
                        default="experiments/plots/ogpt_center_by_turn.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    probe = torch.load(args.probe, map_location="cpu")
    print(f"Probe shape: {tuple(probe.shape)}")

    config = GPTConfig(vocab_size=VOCAB_SIZE, block_size=59,
                       n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    state = torch.load(args.ckpt, map_location="cpu")
    if 'model_state_dict' in state: state = state['model_state_dict']
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded OGPT (block_size={config.block_size}), probing layer {args.layer}")

    games = load_games(max_files=args.max_files)
    games = [g for g in games if len(g) == GAME_LEN][:args.n_games]
    print(f"Using {len(games)} games")

    # OGPT block_size=59; drop the last token to match model context.
    tokens = tokenize_games(games).to(device)[:, :-1]
    states = np.zeros((len(games), GAME_LEN, 8, 8), dtype=np.int8)
    for gi, g in enumerate(games):
        board = OthelloBoardState()
        for t, m in enumerate(g):
            try: board.umpire(m)
            except Exception: break
            states[gi, t] = np.asarray(board.state, dtype=np.int8)

    with torch.no_grad():
        acts_chunks = []
        for i in range(0, len(games), 32):
            h = extract_activations(model, tokens[i:i + 32], args.layer)
            acts_chunks.append(h.cpu())
    acts = torch.cat(acts_chunks, dim=0)
    print(f"Activations: {tuple(acts.shape)}")

    # Restrict to a contiguous turn range that OGPT can handle
    pos_end = min(args.pos_end, acts.shape[1])
    acts = acts[:, args.pos_start:pos_end, :]
    states_s = states[:, args.pos_start:pos_end, :, :]
    G, T, D = acts.shape
    turns = np.arange(args.pos_start, args.pos_start + T)

    # Ground truth as 0=empty, 1=white(-1), 2=black(+1)
    gt = np.zeros_like(states_s, dtype=np.int64)
    gt[states_s == 0]  = 0
    gt[states_s == -1] = 1
    gt[states_s == 1]  = 2
    gt_t = torch.from_numpy(gt)   # (G, T, 8, 8)

    # Apply probe per turn (parity mode: ODD pos -> mode 0, EVEN pos -> mode 1).
    # Use absolute position parity so this is robust to arbitrary pos_start.
    per_turn_center = np.zeros(T, dtype=np.float64)
    for ti in range(T):
        pos = ti + args.pos_start
        m = 0 if pos % 2 == 1 else 1
        W = probe[m]                  # (512, 8, 8, 3)
        h = acts[:, ti, :]            # (G, 512)
        logits = torch.einsum('nd,drco->nrco', h, W)
        preds = logits.argmax(dim=-1)  # (G, 8, 8)
        match = (preds == gt_t[:, ti, :, :]).numpy()   # (G, 8, 8)
        # Mean over 4 center cells, then over games
        center_acc = np.mean([match[:, r, c].mean() for (r, c) in CENTER_RC])
        per_turn_center[ti] = center_acc

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(turns, per_turn_center, 'o-', color='C0', linewidth=2, markersize=5)
    ax.set_xlabel("Move number")
    ax.set_ylabel("Center-cell decoding accuracy")
    ax.set_title("OthelloGPT (Nanda probe, L=6): center-cell accuracy by turn")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved {args.output}")

    # Also dump the raw numbers
    print(f"\nturn  center_acc")
    for t, a in zip(turns, per_turn_center):
        print(f"  {t:>3d}    {a:.4%}")


if __name__ == "__main__":
    main()
