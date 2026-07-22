"""Per-cell accuracy of Nanda's pre-trained probe on OGPT layer 6.

Uses the existing main_linear_probe.pth (the 95.88% probe). Reports per-cell
accuracy as an 8x8 grid so we can see whether the "center cells are hard"
geometric pattern from our quick Ridge run survives Nanda's well-trained
probe -- or whether it was an artifact of weak probing methodology.

Probe shape (3, 512, 8, 8, 3) = (modes, d_model, rows, cols, classes).
  modes: 0 = trained on game positions 5,7,9,...  (black to move next)
         1 = trained on game positions 6,8,10,... (white to move next)
         2 = trained on all positions
  classes: 0 = empty, 1 = white (state==-1), 2 = black (state==+1)

Usage:
    python analyze_nanda_probe_per_cell.py --n-games 500 --mode parity
"""
import sys, os, argparse, pickle, random
sys.path.insert(0, '.')

import numpy as np
import torch
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, extract_activations, VOCAB_SIZE, GAME_LEN,
    SYNTHETIC_DIR,
)


CENTER_RC = {(3, 3), (3, 4), (4, 3), (4, 4)}    # d4, e4, d5, e5
CORNER_RC = {(0, 0), (0, 7), (7, 0), (7, 7)}    # a1, h1, a8, h8


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
            break

    if len(pool) < n_games:
        raise RuntimeError(
            f"Only {len(pool)} valid games in {used_files} files, "
            f"need {n_games}")

    idxs = rng.sample(range(len(pool)), n_games)
    print(f"Sampled {n_games} games uniformly at random from a pool of "
          f"{len(pool):,} (across {used_files} shuffled pickle files, seed={seed})")
    return [pool[i] for i in idxs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    parser.add_argument("--probe", default="mechanistic_interpretability/main_linear_probe.pth")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--n-games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for uniform random game sampling.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap the number of pickle files (default: "
                             "enough for 20x headroom).")
    parser.add_argument("--pos-start", type=int, default=5)
    parser.add_argument("--pos-end",   type=int, default=54)
    parser.add_argument("--mode", default="parity",
                        choices=["mode0", "mode1", "mode2", "parity"],
                        help="Which probe mode to use. 'parity' picks mode0 "
                             "on positions 5,7,9,... and mode1 on 6,8,10,...")
    parser.add_argument("--data-out", default=None,
                        help="Path to save per-cell numbers as .npz.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load probe
    probe = torch.load(args.probe, map_location='cpu')
    print(f"Probe shape: {tuple(probe.shape)}")  # (3, 512, 8, 8, 3)
    assert probe.shape == (3, 512, 8, 8, 3), \
        f"Unexpected probe shape: {probe.shape}"

    # Load OGPT
    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    print(f"Loaded OGPT (block_size={block_size}), probing layer {args.layer}")

    # Load games and replay to board states
    games = sample_games_from_pool(
        n_games=args.n_games, seed=args.seed, n_files=args.max_files)
    print(f"Using {len(games)} games")

    states = np.zeros((len(games), GAME_LEN, 8, 8), dtype=np.int8)
    for i, g in enumerate(games):
        b = OthelloBoardState()
        for t, m in enumerate(g):
            b.umpire(m)
            states[i, t] = np.asarray(b.state, dtype=np.int8)

    # Forward to layer L
    tokens = tokenize_games(games, seq_len=block_size).to(device)
    acts = []
    batch = 16
    with torch.no_grad():
        for i in range(0, len(games), batch):
            h = extract_activations(model, tokens[i:i + batch], args.layer)
            acts.append(h.cpu())
    acts = torch.cat(acts, dim=0)   # (G, T, 512)
    print(f"Activations shape: {tuple(acts.shape)}")

    # Slice to the probe's training position range.  Cap pos_end by
    # activation length so pos_end=60 with block_size=59 doesn't
    # desync acts (59 turns) and gt (60 turns).
    pos_end_eff = min(args.pos_end, acts.shape[1])
    acts = acts[:, args.pos_start:pos_end_eff, :]
    states_s = states[:, args.pos_start:pos_end_eff, :, :]
    G, T, D = acts.shape
    print(f"After slicing: G={G}, T={T}, D={D} "
          f"(pos {args.pos_start}..{pos_end_eff - 1})")

    # Ground truth in {0=empty, 1=white(-1), 2=black(+1)}
    gt = np.zeros_like(states_s, dtype=np.int64)
    gt[states_s == 0]  = 0
    gt[states_s == -1] = 1
    gt[states_s == 1]  = 2
    gt_t = torch.from_numpy(gt)

    # Apply probe
    print(f"\nApplying probe with mode = {args.mode} ...")
    if args.mode == "parity":
        # Absolute-position parity. mode 0 trained on positions 5,7,9,...
        # (ODD positions); mode 1 on 6,8,10,... (EVEN positions).
        preds = torch.zeros((G, T, 8, 8), dtype=torch.long)
        for ti in range(T):
            pos = ti + args.pos_start
            m = 0 if pos % 2 == 1 else 1
            W = probe[m]                # (512, 8, 8, 3)
            h = acts[:, ti, :]          # (G, 512)
            logits = torch.einsum('nd,drco->nrco', h, W)
            preds[:, ti, :, :] = logits.argmax(dim=-1)
    else:
        m = {"mode0": 0, "mode1": 1, "mode2": 2}[args.mode]
        W = probe[m]
        H_flat = acts.reshape(-1, D)
        logits = torch.einsum('nd,drco->nrco', H_flat, W)
        preds = logits.argmax(dim=-1).reshape(G, T, 8, 8)

    match = (preds == gt_t).numpy()    # (G, T, 8, 8)
    acc = match.mean(axis=(0, 1))      # (8, 8)

    if args.data_out is not None:
        np.savez(
            args.data_out,
            acc=acc,
            n_games=args.n_games,
            seed=args.seed,
            layer=args.layer,
            mode=args.mode,
            pos_start=args.pos_start,
            pos_end=pos_end_eff,
        )
        print(f"Saved per-cell data to {args.data_out}")

    # Print 8x8 grid
    print()
    print("Per-cell accuracy of Nanda's probe on OGPT layer 6:")
    print("       " + "   ".join(f"{c}" for c in "abcdefgh"))
    for r in range(8):
        row = []
        for c in range(8):
            v = acc[r, c]
            tag = "*" if (r, c) in CENTER_RC else " "
            row.append(f"{v:.3f}{tag}")
        print(f"  {r+1}  " + " ".join(row))

    print(f"\nMean: {acc.mean():.4f}  Std: {acc.std():.4f}  "
          f"Min: {acc.min():.4f}  Max: {acc.max():.4f}")

    # Per-class summary
    edge_rc = [(r, c) for r in range(8) for c in range(8)
               if (r in (0, 7) or c in (0, 7)) and (r, c) not in CORNER_RC]
    inner_rc = [(r, c) for r in range(8) for c in range(8)
                if r not in (0, 7) and c not in (0, 7)
                and (r, c) not in CENTER_RC]

    print(f"\nBy region:")
    for label, rc_list in [("corner", list(CORNER_RC)),
                           ("edge",   edge_rc),
                           ("inner",  inner_rc),
                           ("center", list(CENTER_RC))]:
        vals = [acc[r, c] for r, c in rc_list]
        print(f"  {label:>6s} n={len(vals):>2d}  "
              f"mean={np.mean(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    # Worst 8 cells
    flat_idx = np.argsort(acc.flatten())
    print(f"\nWorst 8 cells:")
    for idx in flat_idx[:8]:
        r, c = divmod(int(idx), 8)
        name = f"{'abcdefgh'[c]}{r+1}"
        is_center = (r, c) in CENTER_RC
        is_corner = (r, c) in CORNER_RC
        cls = "center" if is_center else ("corner" if is_corner
              else "edge" if (r in (0, 7) or c in (0, 7)) else "inner")
        print(f"  {name} ({cls:>6s})  acc={acc[r, c]:.4f}")

    print(f"\nBest 8 cells:")
    for idx in flat_idx[-8:][::-1]:
        r, c = divmod(int(idx), 8)
        name = f"{'abcdefgh'[c]}{r+1}"
        cls = "center" if (r, c) in CENTER_RC else (
              "corner" if (r, c) in CORNER_RC
              else "edge" if (r in (0, 7) or c in (0, 7)) else "inner")
        print(f"  {name} ({cls:>6s})  acc={acc[r, c]:.4f}")

    # ----------------------- Turn stratification ---------------------------
    print("\n" + "=" * 64)
    print("OGPT per-cell accuracy stratified by game turn (Nanda probe)")
    print("=" * 64)
    # Need per-position turn info. With pos_start..pos_end slicing,
    # turn for slice-index ti is args.pos_start + ti.
    bins = [(5, 10), (10, 15), (15, 20), (20, 25),
            (25, 30), (30, 35), (35, 40), (40, 45), (45, args.pos_end)]
    # Re-apply probe per turn bin
    print(f"\n{'turn':>10s} {'n':>8s} {'overall':>9s} "
          f"{'corner':>9s} {'edge':>9s} {'inner':>9s} {'center':>9s}")
    for lo, hi in bins:
        # Slice indices that map to turns in [lo, hi)
        ti_lo = max(0, lo - args.pos_start)
        ti_hi = min(T, hi - args.pos_start)
        if ti_lo >= ti_hi:
            continue
        if args.mode == "parity":
            preds_bin = torch.zeros((G, ti_hi - ti_lo, 8, 8), dtype=torch.long)
            for ti in range(ti_lo, ti_hi):
                pos = ti + args.pos_start
                m = 0 if pos % 2 == 1 else 1
                W = probe[m]
                h = acts[:, ti, :]
                logits = torch.einsum('nd,drco->nrco', h, W)
                preds_bin[:, ti - ti_lo, :, :] = logits.argmax(dim=-1)
        else:
            m = {"mode0": 0, "mode1": 1, "mode2": 2}[args.mode]
            W = probe[m]
            slice_acts = acts[:, ti_lo:ti_hi, :].reshape(-1, D)
            logits = torch.einsum('nd,drco->nrco', slice_acts, W)
            preds_bin = logits.argmax(dim=-1).reshape(G, ti_hi - ti_lo, 8, 8)
        gt_bin = gt_t[:, ti_lo:ti_hi, :, :]
        match_bin = (preds_bin == gt_bin).numpy()
        n_pos = G * (ti_hi - ti_lo)
        acc_bin = match_bin.mean(axis=(0, 1))   # (8, 8)

        def _mean_region(rc_list):
            return np.mean([acc_bin[r, c] for r, c in rc_list])

        print(f"  {lo:>3d}-{hi-1:<3d} {n_pos:>8d} {acc_bin.mean():>9.4f} "
              f"{_mean_region(list(CORNER_RC)):>9.4f} "
              f"{_mean_region(edge_rc):>9.4f} "
              f"{_mean_region(inner_rc):>9.4f} "
              f"{_mean_region(list(CENTER_RC)):>9.4f}")
