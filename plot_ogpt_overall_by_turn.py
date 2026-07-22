"""Overall (all-64-cells) decoding accuracy vs game turn for Nanda's probe
on OGPT layer-6 activations.

Mirrors plot_mlp_overall_by_turn.py in structure, format, and output style
so the two curves can be compared like-for-like.  Averages match-rate
over all 64 cells (not just center) per turn.

Fair comparison note
--------------------
The MLP script (plot_mlp_overall_by_turn.py) evaluates on a precomputed
chunk (chunk_ext_0039.npz).  This script evaluates on val games loaded
via load_games(max_files=...).  Both ultimately trace back to the same
val pickle files under data/othello_synthetic, so if you set --max-files
appropriately to load the same source files, the samples overlap in
distribution.  For a strict apples-to-apples comparison:
  - use --max-files 6 (or however many produce ~300K games matching the
    chunk_ext_0039 size)
  - use --pos-start 5 --pos-end 54 to match the MLP script's default turn range

Usage:
    python plot_ogpt_overall_by_turn.py \\
        --ckpt ckpts/gpt_nanda_synthetic.ckpt \\
        --probe mechanistic_interpretability/main_linear_probe.pth \\
        --n-games 500 --pos-start 5 --pos-end 54
"""
import sys, os, argparse, pickle, random
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
    tokenize_games, extract_activations, VOCAB_SIZE, GAME_LEN,
    SYNTHETIC_DIR,
)


def sample_games_from_pool(n_games, seed, n_files=None):
    """Sample n_games uniformly at random from the full synthetic pool.

    Shuffles the pickle file list with the given seed, then loads pickle
    files until the pool has enough valid (len==GAME_LEN) games, then
    randomly samples n_games from that pool.
    """
    rng = random.Random(seed)
    files = sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pickle"))
    rng.shuffle(files)
    if n_files is not None:
        files = files[:n_files]

    pool = []
    used_files = 0
    for fname in files:
        with open(os.path.join(SYNTHETIC_DIR, fname), "rb") as f:
            batch = pickle.load(f)
        pool.extend(g for g in batch if len(g) == GAME_LEN)
        used_files += 1
        if n_files is None and len(pool) >= n_games * 20:
            # Have >=20x headroom, that's enough to be effectively uniform
            break

    if len(pool) < n_games:
        raise RuntimeError(
            f"Only {len(pool)} valid games in {used_files} files, "
            f"need {n_games}")

    idxs = rng.sample(range(len(pool)), n_games)
    print(f"Sampled {n_games} games uniformly at random from a pool of "
          f"{len(pool):,} (across {used_files} shuffled pickle files, seed={seed})")
    return [pool[i] for i in idxs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    parser.add_argument("--probe",
                        default="mechanistic_interpretability/main_linear_probe.pth")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--n-games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for uniform random game sampling.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap the number of pickle files considered "
                             "(default: enough to give 20x headroom).")
    parser.add_argument("--pos-start", type=int, default=5)
    parser.add_argument("--pos-end",   type=int, default=54,
                        help="Exclusive upper bound. MLP overall default is "
                             "5..54; matching here for a fair comparison.")
    parser.add_argument("--output",
                        default="experiments/plots/ogpt_overall_by_turn.png")
    parser.add_argument("--data-out", default=None,
                        help="Path to save per-turn numbers as .npz "
                             "(default: swap .png -> .npz on --output).")
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

    games = sample_games_from_pool(
        n_games=args.n_games, seed=args.seed, n_files=args.max_files)
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

    pos_end = min(args.pos_end, acts.shape[1])
    acts = acts[:, args.pos_start:pos_end, :]
    states_s = states[:, args.pos_start:pos_end, :, :]
    G, T, D = acts.shape
    turns = np.arange(args.pos_start, args.pos_start + T)

    # GT: 0=empty, 1=white(-1), 2=black(+1)
    gt = np.zeros_like(states_s, dtype=np.int64)
    gt[states_s == 0]  = 0
    gt[states_s == -1] = 1
    gt[states_s == 1]  = 2
    gt_t = torch.from_numpy(gt)                                        # (G, T, 8, 8)

    # Overall accuracy: mean match-rate across ALL 64 cells per turn.
    per_turn_overall = np.zeros(T, dtype=np.float64)
    per_turn_n = np.zeros(T, dtype=np.int64)
    for ti in range(T):
        pos = ti + args.pos_start
        m = 0 if pos % 2 == 1 else 1
        W = probe[m]                                                    # (512, 8, 8, 3)
        h = acts[:, ti, :]                                             # (G, 512)
        logits = torch.einsum('nd,drco->nrco', h, W)
        preds = logits.argmax(dim=-1)                                  # (G, 8, 8)
        match = (preds == gt_t[:, ti, :, :]).numpy()                    # (G, 8, 8)
        # All 64 cells, then averaged across games
        per_turn_overall[ti] = float(match.mean())
        per_turn_n[ti] = G

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(turns, per_turn_overall, 'o-', color='C0', linewidth=2, markersize=5)
    ax.set_xlabel("Move number")
    ax.set_ylabel("Decoding accuracy (all 64 cells)")
    ax.set_title(f"OthelloGPT (Nanda probe, L={args.layer}): "
                 f"overall accuracy by turn")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved {args.output}")

    data_out = args.data_out
    if data_out is None:
        base, _ = os.path.splitext(args.output)
        data_out = base + ".npz"
    np.savez(
        data_out,
        turns=turns,
        per_turn_overall=per_turn_overall,
        per_turn_n=per_turn_n,
        n_games=args.n_games,
        seed=args.seed,
        layer=args.layer,
        pos_start=args.pos_start,
        pos_end=args.pos_start + T,
    )
    print(f"Saved data to {data_out}")

    # Same output format as plot_mlp_overall_by_turn.py:
    print(f"\n{'turn':>4s}  {'n':>8s}  {'overall':>9s}")
    for t, a, n in zip(turns, per_turn_overall, per_turn_n):
        print(f"  {t:>3d}  {n:>8d}  {a:.4%}")


if __name__ == "__main__":
    main()
