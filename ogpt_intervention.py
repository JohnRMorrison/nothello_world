"""Probe-directed intervention on OGPT residual stream, mirroring mlp_intervention.py.

Matched protocol so the resulting top-K success rate is directly comparable
to the MLP number:
  - Same probe: Ridge classifier on hidden -> per-cell board state (3-class).
  - Same "empty direction": coef_[empty] - mean(coef_[other]), normalized.
  - Same target selection: top-5 illegal cells per position by baseline
    next-move probability.
  - Same scale: K=3 only (where the MLP peaks).
  - Same metrics: eps>0, delta>0, joint (eps & delta), Li-style top-K.

Usage:
    python ogpt_intervention.py --ckpt ckpts/gpt_nanda_synthetic.ckpt \\
        --layer 4 --scale 3 --n-games 1000
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
from hand_crafted_flanking import MOVE_TO_IDX

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, load_games, VOCAB_SIZE, GAME_LEN, STOI, ITOS,
)


CENTER_64 = {27, 28, 35, 36}


def forward_to_layer(model, x, layer):
    """Forward through blocks 0..layer (inclusive), return resid stream at that point."""
    b, t = x.size()
    tok = model.tok_emb(x)
    pos = model.pos_emb[:, :t, :]
    h = model.drop(tok + pos)
    for block in model.blocks[:layer + 1]:
        h = block(h)
    return h


def forward_from_layer(model, h, layer):
    """Continue forward from a resid stream at layer through blocks (layer+1)..end,
    apply ln_f and head, return logits."""
    for block in model.blocks[layer + 1:]:
        h = block(h)
    h = model.ln_f(h)
    return model.head(h)


def compute_states_classes(games):
    out = np.zeros((len(games), GAME_LEN, 64), dtype=np.int8)
    for i, g in enumerate(games):
        b = OthelloBoardState()
        for t, m in enumerate(g):
            b.umpire(m)
            s = np.asarray(b.state).flatten()
            cur = np.zeros(64, dtype=np.int8)
            cur[s == 0]  = 0
            cur[s == -1] = 1
            cur[s == 1]  = 2
            out[i, t] = cur
    return out


def board_to_move_index(board_state_flat):
    """Map a 64-cell board state to the 60 movable cells (skip center cells)."""
    keep = [c for c in range(64) if c not in CENTER_64]
    return board_state_flat[..., keep]   # shape (..., 60)


def legal_moves_from_board(board_64, next_player_color):
    """Compute legal moves given current board and next-player color.

    board_64: (64,) int8 with 0=empty, 1=white, 2=black.
    next_player_color: 1 (white) or 2 (black).

    Returns a 60-bool array (legal for the 60 movable cells).
    """
    # Use OthelloBoardState's get_valid_moves logic by setting up a board.
    board = OthelloBoardState()
    state = np.array([
        [{0: 0, 1: -1, 2: 1}[int(board_64[r * 8 + c])] for c in range(8)]
        for r in range(8)
    ], dtype=np.int8)
    board.state = state
    board.next_hand_color = 1 if next_player_color == 2 else -1
    valid = board.get_valid_moves()   # list of 0..63 indices
    legal_60 = np.zeros(60, dtype=bool)
    skip = sorted(CENTER_64)
    for v in valid:
        # Map 64-index -> 60-index
        idx60 = v - sum(1 for s in skip if s < v)
        legal_60[idx60] = True
    return legal_60


def get_cell_token_indices():
    """Return a list of length 60 where index i is the OGPT token id for the
    i-th movable cell (in 60-cell MOVE_TO_IDX order)."""
    skip = sorted(CENTER_64)
    out = []
    for c64 in range(64):
        if c64 in skip:
            continue
        # cell name 'a1'..'h8'
        name = c64
        # STOI is keyed on integer 64-cell index in this codebase.
        # (Tokens 1..60 correspond to the 60 movable cells in some order.)
        out.append(STOI[name])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head",  type=int, default=8)
    parser.add_argument("--n-embd",  type=int, default=512)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--n-games", type=int, default=1000)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--probe-pos-start", type=int, default=5)
    parser.add_argument("--probe-pos-end",   type=int, default=55)
    parser.add_argument("--test-positions-per-game", type=int, default=5)
    parser.add_argument("--max-cells-per-pos", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--nanda-probe", default=None,
                        help="If set, load Nanda's pre-trained probe from this "
                             "path (e.g. mechanistic_interpretability/main_linear_probe.pth). "
                             "Probe shape (3, 512, 8, 8, 3); we extract the "
                             "'empty' direction at each cell from the mode-2 "
                             "(all positions) head and skip Ridge fitting.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(args.ckpt, map_location=device)
    block_size = state_dict["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd)
    model = GPT(config)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    d_model = args.n_embd
    print(f"Loaded OGPT (n_layer={args.n_layer}, d_model={d_model}); "
          f"intervening at layer {args.layer}, scale {args.scale}")

    games = load_games(max_files=args.max_files)
    if len(games) > args.n_games:
        games = games[:args.n_games]
    print(f"Using {len(games)} games")

    # Split games: first 70% for probe training, last 30% for test.
    n_probe = int(len(games) * 0.7)
    probe_games = games[:n_probe]
    test_games = games[n_probe:]
    print(f"  probe_games={len(probe_games)}, test_games={len(test_games)}")

    states_probe = compute_states_classes(probe_games)
    states_test  = compute_states_classes(test_games)
    print(f"States: probe {states_probe.shape}, test {states_test.shape}")

    # Forward all probe games to chosen layer to harvest activations.
    print("Forwarding probe games to layer", args.layer, "...")
    tokens_probe = tokenize_games(probe_games, seq_len=block_size).to(device)
    acts_probe = []
    batch = 32
    with torch.no_grad():
        for i in range(0, len(probe_games), batch):
            h = forward_to_layer(model, tokens_probe[i:i + batch], args.layer)
            acts_probe.append(h.cpu().numpy().astype(np.float32))
    H_probe = np.concatenate(acts_probe, axis=0)
    print(f"  H_probe: {H_probe.shape}")

    skip = sorted(CENTER_64)
    movable_64 = [c for c in range(64) if c not in CENTER_64]

    directions = np.zeros((60, d_model), dtype=np.float32)
    probe_accs = np.zeros(60, dtype=np.float32)

    if args.nanda_probe is not None:
        # Load Nanda's probe: shape (3, 512, 8, 8, 3) = (modes, d, rows, cols, classes)
        # class 0 = empty, class 1 = white, class 2 = black.
        probe = torch.load(args.nanda_probe, map_location='cpu')
        print(f"Loaded Nanda probe from {args.nanda_probe}, shape {tuple(probe.shape)}")
        # Use mode 2 ("all positions") so we have a single direction per cell
        # that doesn't depend on turn parity.
        W_all = probe[2].numpy()   # (512, 8, 8, 3)
        for m, c64 in enumerate(movable_64):
            r, c = c64 // 8, c64 % 8
            # W[r, c, :] gives (512, 3) per cell; we want the empty direction.
            empty_row = W_all[:, r, c, 0]
            other = 0.5 * (W_all[:, r, c, 1] + W_all[:, r, c, 2])
            d = empty_row - other
            n = float(np.linalg.norm(d))
            if n > 0:
                directions[m] = (d / n).astype(np.float32)
        probe_accs[:] = float('nan')   # we don't have a per-cell accuracy here
    else:
        # Fall back: train inline Ridge probes (conservative methodology).
        Y_probe = states_probe[:, args.probe_pos_start:args.probe_pos_end, :].reshape(-1, 64)
        H_flat  = H_probe[:, args.probe_pos_start:args.probe_pos_end, :].reshape(
            -1, d_model)
        print(f"  Probe-train flattened: H {H_flat.shape}, Y {Y_probe.shape}")
        scaler = StandardScaler().fit(H_flat)
        H_scaled = scaler.transform(H_flat)

        for m, c64 in enumerate(movable_64):
            y = Y_probe[:, c64]
            if len(np.unique(y)) < 2: continue
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(H_scaled, y)
            probe_accs[m] = clf.score(H_scaled, y)  # training accuracy (proxy)
            classes = clf.classes_
            coef = clf.coef_
            if 0 in classes:
                empty_row = coef[list(classes).index(0)]
                other = (np.mean(np.delete(coef, list(classes).index(0), axis=0),
                                 axis=0)
                         if coef.shape[0] > 1 else np.zeros_like(empty_row))
                d = empty_row - other
                d_raw = d / np.maximum(scaler.scale_, 1e-8)
                n = np.linalg.norm(d_raw)
                if n > 0:
                    directions[m] = d_raw / n
        print(f"  Mean (training) probe acc: {probe_accs.mean():.4f}")

    # Cell -> OGPT token mapping (token id for each movable cell).
    cell_tokens = get_cell_token_indices()   # list of 60
    cell_tokens_t = torch.tensor(cell_tokens, device=device, dtype=torch.long)
    print(f"  Cell token ids: first 5 = {cell_tokens[:5]}")

    # Interventions on test games.
    print(f"\nRunning interventions on test games...")
    tokens_test = tokenize_games(test_games, seq_len=block_size).to(device)
    rng = np.random.RandomState(0)

    eps_all, delta_all, joint_all, li_topk_all, li_topkp1_all = [], [], [], [], []
    pos_recall_all, pos_turn_all = [], []
    cross_all = []
    n_test = 0

    with torch.no_grad():
        # Pre-forward each game to layer L (avoids redundant compute per target).
        h_test_cache = []
        for i in range(0, len(test_games), 16):
            h = forward_to_layer(model, tokens_test[i:i + 16], args.layer)
            h_test_cache.append(h)
        H_test_at_L = torch.cat(h_test_cache, dim=0)  # (G_test, T, d_model)
        print(f"  H_test_at_L: {H_test_at_L.shape}")

        for g_idx, game in enumerate(test_games):
            h_game = H_test_at_L[g_idx:g_idx + 1]   # (1, T, d_model)
            # Sample which positions to probe in this game (avoid edges).
            positions = rng.choice(
                range(args.probe_pos_start, args.probe_pos_end),
                size=args.test_positions_per_game, replace=False)

            for t in positions:
                # Baseline forward (no intervention) at this position.
                logits_base = forward_from_layer(model, h_game, args.layer)
                cell_logits_base = logits_base[0, t, cell_tokens_t]
                base_cell_probs = F.softmax(cell_logits_base, dim=-1)

                # Ground truth legal mask at this position.
                board_t = states_test[g_idx, t]   # (64,)
                # Next player: white if turn is odd (Othello black plays first
                # at turn 0). turn t = #moves played so far when predicting
                # move t+1. After replaying through move t, next is white if
                # t+1 odd => t even => black; consistent with even=black plays.
                next_color = 1 if (t + 1) % 2 == 1 else 2
                # Conversion: OthelloBoardState expects -1 white / 1 black.
                # legal_moves_from_board interprets 1 white / 2 black -- match.
                legal = legal_moves_from_board(board_t, next_color)
                if legal.sum() == 0:
                    continue

                # Pick targets: top illegal cells by baseline prob.
                illegal = ~legal
                ill_idx = np.where(illegal)[0]
                if len(ill_idx) == 0: continue
                base_probs_np = base_cell_probs.cpu().numpy()
                ill_scores = base_probs_np[ill_idx]
                order = np.argsort(-ill_scores)
                targets = ill_idx[order[:args.max_cells_per_pos]]

                # Baseline recall@K for stratification.
                K = int(legal.sum())
                base_ranks = np.argsort(-base_probs_np)
                base_topK = set(base_ranks[:K].tolist())
                pos_recall = sum(1 for c in np.where(legal)[0] if c in base_topK) / K

                for c in targets:
                    d = torch.from_numpy(directions[int(c)]).to(device)
                    if d.norm() == 0: continue
                    # Apply intervention: perturb resid stream at position t.
                    h_pert = h_game.clone()
                    h_pert[0, t, :] = h_pert[0, t, :] + args.scale * d
                    logits_pert = forward_from_layer(model, h_pert, args.layer)
                    cell_logits_pert = logits_pert[0, t, cell_tokens_t]
                    new_probs = F.softmax(cell_logits_pert, dim=-1).cpu().numpy()

                    p_target = float(new_probs[c])
                    other_ill = ill_idx[ill_idx != c]
                    max_other_ill = float(new_probs[other_ill].max()) if len(other_ill) > 0 else 0.0
                    legal_cells = np.where(legal)[0]
                    min_legal = float(new_probs[legal_cells].min())
                    eps = p_target - max_other_ill
                    delta = p_target - min_legal
                    joint = (eps > 0) and (delta > 0)
                    ranks = np.argsort(-new_probs)
                    topK = set(ranks[:K].tolist())
                    topKp1 = set(ranks[:K + 1].tolist())
                    li_topk = 1 if int(c) in topK else 0
                    li_topkp1 = 1 if int(c) in topKp1 else 0
                    non_target = np.ones(60, dtype=bool); non_target[c] = False
                    cross = float(np.abs(new_probs[non_target] - base_probs_np[non_target]).mean())

                    eps_all.append(eps)
                    delta_all.append(delta)
                    joint_all.append(int(joint))
                    li_topk_all.append(li_topk)
                    li_topkp1_all.append(li_topkp1)
                    pos_recall_all.append(pos_recall)
                    pos_turn_all.append(int(t))
                    cross_all.append(cross)
                    n_test += 1

    print(f"\nTotal interventions: {n_test}")
    if n_test == 0:
        sys.exit(1)

    eps_arr = np.array(eps_all)
    delta_arr = np.array(delta_all)
    joint_arr = np.array(joint_all)
    tk = np.array(li_topk_all)
    tkp = np.array(li_topkp1_all)
    pos_r = np.array(pos_recall_all)
    cross_arr = np.array(cross_all)

    print()
    print("=" * 78)
    print(f"OGPT intervention at L={args.layer}, scale K={args.scale}")
    print("=" * 78)
    print(f"  n interventions:  {n_test}")
    print(f"  eps > 0:          {(eps_arr > 0).mean():.4%}")
    print(f"  delta > 0:        {(delta_arr > 0).mean():.4%}")
    print(f"  joint (Li proxy): {joint_arr.mean():.4%}")
    print(f"  Li top-K direct:  {tk.mean():.4%}")
    print(f"  Li top-(K+1):     {tkp.mean():.4%}")
    print(f"  mean eps:         {eps_arr.mean():.4f}")
    print(f"  mean delta:       {delta_arr.mean():.4f}")
    print(f"  mean crosstalk:   {cross_arr.mean():.4f}")
    print(f"  mean pos recall:  {pos_r.mean():.4f}")

    print()
    print("Stratified by baseline position recall@K (quintiles):")
    edges = np.quantile(pos_r, np.linspace(0, 1, 6))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    print(f"  {'recall bin':>14s} {'n':>6s} {'top-K%':>8s} {'eps>0%':>8s} "
          f"{'joint%':>8s}")
    for q in range(5):
        lo, hi = edges[q], edges[q + 1]
        m = (pos_r >= lo) & (pos_r < hi)
        n = int(m.sum())
        if n == 0: continue
        print(f"  [{lo:>5.2f},{hi:>5.2f}) {n:>6d} {tk[m].mean():>7.2%} "
              f"{(eps_arr[m] > 0).mean():>7.2%} {joint_arr[m].mean():>7.2%}")

    if args.output:
        np.savez(args.output,
                 layer=args.layer, scale=args.scale,
                 eps=eps_arr, delta=delta_arr, joint=joint_arr,
                 li_topk=tk, li_topkp1=tkp,
                 pos_recall=pos_r, pos_turn=np.array(pos_turn_all),
                 cross=cross_arr, probe_accs=probe_accs)
        print(f"\nSaved raw samples to {args.output}")
