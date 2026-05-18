"""Sweep intervention scale alpha; measure board-state decoding crosstalk.

Experimental question: as the size (alpha) of a probe-directed intervention on
a single board square grows, do the probe's predictions for *other* squares
start flipping? That would indicate the intervention is no longer a clean,
local edit and has started corrupting the rest of the board representation.

Design (held fixed):
  - Intervention direction = Nanda probe "empty direction" for the target
    square (mode 2, all positions), normalized to unit norm.
  - Apply intervention at the probe's layer; decode immediately with the same
    probe so we measure crosstalk directly in probe space.
  - For each alpha and each target square, count the number of squares whose
    argmax prediction flips relative to the alpha=0 baseline (board-state
    reconstruction error). All 64 squares are scored, including the
    intervened one.
  - Positions filtered to those where the target square is currently
    occupied (so "push toward empty" is a meaningful perturbation).

Usage:
    python sweep_intervention_alpha.py \\
        --ckpt ckpts/gpt_nanda_synthetic.ckpt \\
        --probe mechanistic_interpretability/main_linear_probe.pth \\
        --layer 6 \\
        --alphas 0,1,2,4,6,8,12,16,24 \\
        --squares 3,3 0,3 2,3 \\
        --n-games 500 \\
        --output experiments/intervention_alpha_sweep/results.json
"""
import sys, os, argparse, json
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F

from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, load_games, extract_activations, VOCAB_SIZE, GAME_LEN,
)


def parse_squares(specs):
    """Parse ['3,3', '0,3', ...] -> [(3,3), (0,3), ...]."""
    out = []
    for s in specs:
        r, c = s.split(',')
        out.append((int(r), int(c)))
    return out


def parse_alphas(spec):
    return [float(x) for x in spec.split(',')]


def compute_ground_truth_states(games):
    """Replay each game and record the 8x8 board state at every move."""
    states = np.zeros((len(games), GAME_LEN, 8, 8), dtype=np.int8)
    for i, g in enumerate(games):
        b = OthelloBoardState()
        for t, m in enumerate(g):
            b.umpire(m)
            states[i, t] = np.asarray(b.state, dtype=np.int8)
    return states


def build_empty_direction(probe_mode2, r, c):
    """Nanda 'empty direction' for cell (r,c): empty - mean(white, black),
    normalized to unit L2 norm.

    probe_mode2: (d_model, 8, 8, 3) -- the mode-2 slice of the full probe.
    """
    w = probe_mode2[:, r, c, :]                      # (d_model, 3)
    d = w[:, 0] - 0.5 * (w[:, 1] + w[:, 2])
    n = float(np.linalg.norm(d))
    return (d / n) if n > 0 else d


def decode_all_cells(h_flat, probe_mode2):
    """Decode every cell from activations.

    h_flat: (N, d_model) torch tensor.
    probe_mode2: (d_model, 8, 8, 3) torch tensor.

    Returns (N, 8, 8) int64 of argmax class.
    """
    logits = torch.einsum('nd,drco->nrco', h_flat, probe_mode2)
    return logits.argmax(dim=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    parser.add_argument("--probe",
                        default="mechanistic_interpretability/main_linear_probe.pth")
    parser.add_argument("--layer", type=int, default=6,
                        help="Layer at which to intervene AND decode.")
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head",  type=int, default=8)
    parser.add_argument("--n-embd",  type=int, default=512)
    parser.add_argument("--n-games", type=int, default=500)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--pos-start", type=int, default=5)
    parser.add_argument("--pos-end",   type=int, default=54)
    parser.add_argument("--alphas", default="0,1,2,4,6,8,12,16,24",
                        help="Comma-separated alpha values to sweep.")
    parser.add_argument("--squares", nargs='+',
                        default=["3,3", "0,3", "2,3"],
                        help="Target squares as 'row,col' pairs. Default: "
                             "(3,3) center, (0,3) edge, (2,3) intermediate.")
    parser.add_argument("--output",
                        default="experiments/intervention_alpha_sweep/results.json")
    parser.add_argument("--raw-npz", default=None,
                        help="Optional path to also dump raw per-position "
                             "flip counts as an .npz file.")
    args = parser.parse_args()

    alphas = parse_alphas(args.alphas)
    squares = parse_squares(args.squares)
    assert 0.0 in alphas, "alphas must include 0 (baseline)."

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Alphas: {alphas}")
    print(f"Squares: {squares}")

    # --- Load probe ---
    probe_full = torch.load(args.probe, map_location='cpu')
    print(f"Probe shape: {tuple(probe_full.shape)}")
    assert probe_full.shape == (3, args.n_embd, 8, 8, 3), \
        f"Unexpected probe shape: {probe_full.shape}"
    probe_mode2_np = probe_full[2].detach().cpu().numpy().astype(np.float32)
    probe_mode2 = torch.from_numpy(probe_mode2_np).to(device)
    print("Using probe mode 2 (all positions).")

    # --- Load OGPT ---
    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    print(f"OGPT loaded; intervening + decoding at layer {args.layer}.")

    # --- Load games and ground-truth states ---
    games = load_games(max_files=args.max_files)[:args.n_games]
    print(f"Using {len(games)} games.")
    states_gt = compute_ground_truth_states(games)   # (G, T, 8, 8) in {-1,0,1}

    # Pos-sliced ground truth in {0=empty, 1=white(-1), 2=black(+1)} for filtering.
    gt_classes = np.zeros_like(states_gt, dtype=np.int8)
    gt_classes[states_gt == 0] = 0
    gt_classes[states_gt == -1] = 1
    gt_classes[states_gt == 1] = 2

    # --- Extract activations at the probe layer (once) ---
    tokens = tokenize_games(games, seq_len=block_size).to(device)
    acts = []
    batch = 16
    with torch.no_grad():
        for i in range(0, len(games), batch):
            h = extract_activations(model, tokens[i:i + batch], args.layer)
            acts.append(h.cpu())
    acts = torch.cat(acts, dim=0)                      # (G, T, d_model)
    print(f"Activations: {tuple(acts.shape)}")

    # Restrict to the probe's training position range.
    acts = acts[:, args.pos_start:args.pos_end, :]     # (G, T', D)
    gt_slice = gt_classes[:, args.pos_start:args.pos_end, :, :]   # (G, T', 8, 8)
    G, T, D = acts.shape
    N_total = G * T
    acts_flat = acts.reshape(N_total, D).to(device)

    # Baseline (alpha=0) decode for all positions.
    with torch.no_grad():
        baseline_preds = decode_all_cells(acts_flat, probe_mode2)   # (N, 8, 8) int64
    baseline_preds_np = baseline_preds.cpu().numpy()

    # --- Sweep ---
    results = {
        "config": {
            "ckpt": args.ckpt, "probe": args.probe,
            "layer": args.layer, "n_games": len(games),
            "pos_start": args.pos_start, "pos_end": args.pos_end,
            "alphas": alphas, "squares": [list(s) for s in squares],
            "probe_mode": 2,
            "n_positions_total": int(N_total),
        },
        "squares": {},
    }

    raw_per_square = {}   # for optional npz dump

    for (r, c) in squares:
        key = f"{r},{c}"
        print(f"\n=== Target square ({r},{c}) ===")

        # Build intervention direction for this square.
        d_np = build_empty_direction(probe_mode2_np, r, c)
        direction = torch.from_numpy(d_np).to(device)   # (D,)
        print(f"  ||direction|| = {float(torch.linalg.norm(direction)):.4f}")

        # Filter positions to those where this square is currently occupied.
        occupied_mask = (gt_slice[:, :, r, c] != 0).reshape(N_total)   # (N,)
        idx = np.flatnonzero(occupied_mask)
        n_keep = int(idx.size)
        print(f"  Positions with square occupied: {n_keep} / {N_total}")
        if n_keep == 0:
            print(f"  Skipping ({r},{c}): no occupied positions.")
            continue

        idx_t = torch.from_numpy(idx).to(device)
        h_subset = acts_flat[idx_t]                              # (n_keep, D)
        baseline_subset = baseline_preds[idx_t]                  # (n_keep, 8, 8)

        # Per-alpha flips.
        per_alpha = {"alpha": [], "flips_mean": [], "flips_std": [],
                     "flips_median": [], "flips_q25": [], "flips_q75": []}
        raw_flip_matrix = np.zeros((len(alphas), n_keep), dtype=np.int32)

        with torch.no_grad():
            for a_i, alpha in enumerate(alphas):
                h_intv = h_subset + alpha * direction              # (n_keep, D)
                preds = decode_all_cells(h_intv, probe_mode2)      # (n_keep, 8, 8)
                # Flip count per position (any of 64 cells different from baseline).
                flips = (preds != baseline_subset).sum(dim=(1, 2)).cpu().numpy()
                raw_flip_matrix[a_i] = flips
                per_alpha["alpha"].append(float(alpha))
                per_alpha["flips_mean"].append(float(np.mean(flips)))
                per_alpha["flips_std"].append(float(np.std(flips)))
                per_alpha["flips_median"].append(float(np.median(flips)))
                per_alpha["flips_q25"].append(float(np.percentile(flips, 25)))
                per_alpha["flips_q75"].append(float(np.percentile(flips, 75)))
                print(f"  alpha={alpha:>5.1f}  mean_flips={per_alpha['flips_mean'][-1]:.3f}  "
                      f"median={per_alpha['flips_median'][-1]:.1f}  "
                      f"q25-q75=[{per_alpha['flips_q25'][-1]:.1f}, {per_alpha['flips_q75'][-1]:.1f}]")

        per_alpha["n_positions"] = n_keep
        results["squares"][key] = per_alpha
        raw_per_square[key] = raw_flip_matrix

    # --- Save ---
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {args.output}")

    if args.raw_npz:
        os.makedirs(os.path.dirname(args.raw_npz), exist_ok=True)
        np.savez(args.raw_npz, alphas=np.array(alphas), **raw_per_square)
        print(f"Saved raw per-position flips to {args.raw_npz}")
