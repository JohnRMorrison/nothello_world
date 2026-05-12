"""Side-by-side intervention comparison: MLP vs OGPT on identical positions.

For each (game, position, intervention type, target cell), apply the same
probe-directed perturbation to both models, then measure dP on cells whose
legality should change according to Othello rules:

  - "mine":   make currently-empty cell C the current player's color.
              Predicts newly_legal cells (cells flanking through C with C
              as a mine-terminator).
  - "theirs": make currently-empty cell C the opponent's color.
              Predicts newly_legal cells (cells flanking through C with C
              as an opp_cell).
  - "empty":  make currently-occupied cell C empty.
              Predicts newly_illegal cells (cells whose flanks required
              the piece at C).

For each pair (model, intervention), records P_target / P_others before
and after, so we can compute dP for both newly_legal and newly_illegal
sets and compare distributions / find divergence cases.

Probes (both Nanda-style, 99%+):
  - MLP:   probe_direct_H512_wheneven.pt  (BCE-trained, 99.25%, classes
           are 0=empty, 1=mine, 2=theirs in player-relative encoding)
  - OGPT:  main_linear_probe.pth          (Nanda's, 95.88%, mode 2 / all-
           positions; classes are 0=empty, 1=white(-1), 2=black(+1))

For OGPT, we map mine/theirs -> white/black per-position based on whose
turn it is. For MLP, mine/theirs are direct.

Usage:
    python compare_interventions.py --n-games 100000 --positions-per-game 10
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import (
    DirectMLP, compute_pattern_labels_batch, pat_labels_to_cell_labels,
    _get_cell_pat_index,
)
sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, load_games, VOCAB_SIZE, GAME_LEN, STOI, ITOS,
)

N_MOVES = 60
CENTER_64 = {27, 28, 35, 36}
# 64->60 mapping
_movable_64 = [c for c in range(64) if c not in CENTER_64]
_b64_to_m60 = {c64: i for i, c64 in enumerate(_movable_64)}


# ============================== Othello helpers ===============================

def board_from_state_array(state):
    """Build OthelloBoardState given a (8,8) array with -1/0/+1."""
    b = OthelloBoardState()
    b.state = np.array(state, dtype=np.int8)
    return b


def legal_moves_for_board(state8, next_color_int):
    """state8: (8,8) -1/0/1. next_color_int: 1=black, -1=white."""
    b = OthelloBoardState()
    b.state = np.array(state8, dtype=np.int8)
    b.next_hand_color = next_color_int
    valid = b.get_valid_moves()
    out = np.zeros(60, dtype=bool)
    for v in valid:
        if v in _b64_to_m60:
            out[_b64_to_m60[v]] = True
    return out


def compute_newly_legal_illegal(state8, c64, new_state, next_color_int):
    """Given a current board state and a target cell C in {0..63}, set
    state[C] = new_state (in {-1, 0, +1}) and return (newly_legal,
    newly_illegal) as 60-bool arrays."""
    orig = legal_moves_for_board(state8, next_color_int)
    modified = np.array(state8, dtype=np.int8)
    r, c = c64 // 8, c64 % 8
    modified[r, c] = new_state
    new_l = legal_moves_for_board(modified, next_color_int)
    newly_legal = new_l & ~orig
    newly_illegal = orig & ~new_l
    return orig, newly_legal, newly_illegal


# ============================ Feature reconstruction =========================

def game_to_features_at_t(game_cells, t, n_moves=60):
    """Return when+even features (120-d) for a position at turn t in a game
    represented as a list of 60-cell move indices (in MOVE_TO_IDX space).
    """
    when = np.zeros(n_moves, dtype=np.float32)
    even = np.zeros(n_moves, dtype=np.float32)
    for s in range(t):
        c60 = game_cells[s]
        when[c60] = (s + 1) / n_moves
        even[c60] = 1.0 if s % 2 == 0 else 0.0
    return np.concatenate([when, even]).astype(np.float32)


# ============================ OGPT helpers ====================================

def forward_to_layer(model, x, layer):
    b, t = x.size()
    tok = model.tok_emb(x)
    pos = model.pos_emb[:, :t, :]
    h = model.drop(tok + pos)
    for block in model.blocks[:layer + 1]:
        h = block(h)
    return h


def forward_from_layer(model, h, layer):
    for block in model.blocks[layer + 1:]:
        h = block(h)
    h = model.ln_f(h)
    return model.head(h)


# ============================ Probe direction extraction ======================

def mlp_directions(probe_ck, hidden):
    """Return (60, 3, hidden) directions from saved probe with classes
    0=empty, 1=mine, 2=theirs. Direction for class k =
    W[k] - 0.5*(W[a] + W[b]) where a,b are the other classes, normalized.
    We average even/odd probe heads."""
    w_even = probe_ck['even']['weight'].numpy().reshape(64, 3, hidden)
    w_odd  = probe_ck['odd']['weight'].numpy().reshape(64, 3, hidden)
    w_mean = 0.5 * (w_even + w_odd)
    dirs = np.zeros((60, 3, hidden), dtype=np.float32)
    for m, c64 in enumerate(_movable_64):
        for k in range(3):
            others = [j for j in range(3) if j != k]
            d = w_mean[c64, k, :] - 0.5 * (w_mean[c64, others[0], :]
                                            + w_mean[c64, others[1], :])
            n = float(np.linalg.norm(d))
            if n > 0:
                dirs[m, k, :] = (d / n).astype(np.float32)
    return dirs


def ogpt_directions(probe, d_model):
    """Nanda probe (3, 512, 8, 8, 3). Use mode 2 (all positions). Return
    (60, 3, d_model) directions for class indices 0=empty, 1=white, 2=black."""
    W = probe[2].detach().cpu().numpy()   # (512, 8, 8, 3)
    dirs = np.zeros((60, 3, d_model), dtype=np.float32)
    for m, c64 in enumerate(_movable_64):
        r, c = c64 // 8, c64 % 8
        for k in range(3):
            others = [j for j in range(3) if j != k]
            d = W[:, r, c, k] - 0.5 * (W[:, r, c, others[0]]
                                        + W[:, r, c, others[1]])
            n = float(np.linalg.norm(d))
            if n > 0:
                dirs[m, k, :] = (d / n).astype(np.float32)
    return dirs


# ============================ Cell prob aggregators ===========================

def mlp_cell_probs(pat_logits, idx_t, mask_t):
    """prob_or aggregation over the 960 pattern logits -> (B, 60) cell probs."""
    log1m = -F.softplus(pat_logits)
    g = log1m[:, idx_t]
    g = g.masked_fill(~mask_t, 0.0)
    return 1.0 - torch.exp(g.sum(dim=-1))


def ogpt_cell_probs(logits_at_pos, cell_tokens_t):
    """logits_at_pos: (B, 61). Return (B, 60) softmax cell probs."""
    probs = F.softmax(logits_at_pos, dim=-1)
    return probs[:, cell_tokens_t]


# ============================ MAIN ============================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mlp-ckpt", default="experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_direct_H512_wheneven.pt")
    p.add_argument("--mlp-probe", default="experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/probe_direct_H512_wheneven.pt")
    p.add_argument("--mlp-hidden", type=int, default=512)
    p.add_argument("--ogpt-ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    p.add_argument("--ogpt-probe", default="mechanistic_interpretability/main_linear_probe.pth")
    p.add_argument("--ogpt-layer", type=int, default=4)
    p.add_argument("--scale", type=float, default=3.0)
    p.add_argument("--n-games", type=int, default=100000)
    p.add_argument("--positions-per-game", type=int, default=10)
    p.add_argument("--max-files", type=int, default=20)
    p.add_argument("--pos-start", type=int, default=10)
    p.add_argument("--pos-end",   type=int, default=50)
    p.add_argument("--max-targets-per-direction", type=int, default=2)
    p.add_argument("--output", default="logs/compare_interventions.npz")
    p.add_argument("--checkpoint-every", type=int, default=5000,
                   help="Save partial npz every N games processed.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------- Load MLP + probe ---------------------------------
    mlp_ck = torch.load(args.mlp_ckpt, map_location=device)
    n_patterns = mlp_ck.get('n_patterns', 960)
    input_dim = mlp_ck.get('input_dim', 120)
    me = DirectMLP(input_dim, args.mlp_hidden, n_patterns).to(device)
    mo = DirectMLP(input_dim, args.mlp_hidden, n_patterns).to(device)
    me.load_state_dict(mlp_ck['even']); me.eval()
    mo.load_state_dict(mlp_ck['odd']); mo.eval()
    mlp_probe_ck = torch.load(args.mlp_probe, map_location='cpu')
    assert mlp_probe_ck['hidden_dim'] == args.mlp_hidden
    print(f"MLP probe accuracy: {mlp_probe_ck.get('best_acc'):.4f}")
    mlp_dirs = mlp_directions(mlp_probe_ck, args.mlp_hidden)
    mlp_dirs_t = torch.from_numpy(mlp_dirs).to(device)
    print(f"MLP directions: {mlp_dirs_t.shape}")

    # ---------------------- Load OGPT + probe --------------------------------
    sd = torch.load(args.ogpt_ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    ogpt = GPT(config); ogpt.load_state_dict(sd); ogpt = ogpt.to(device).eval()
    ogpt_probe = torch.load(args.ogpt_probe, map_location='cpu')
    print(f"OGPT probe shape: {tuple(ogpt_probe.shape)}")
    ogpt_dirs = ogpt_directions(ogpt_probe, 512)
    ogpt_dirs_t = torch.from_numpy(ogpt_dirs).to(device)
    print(f"OGPT directions: {ogpt_dirs_t.shape}")

    # Cell tokens
    cell_tokens = [STOI[c64] for c64 in _movable_64]
    cell_tokens_t = torch.tensor(cell_tokens, device=device, dtype=torch.long)

    # Pattern indexing for MLP cell probs
    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    pattern_to_cell = torch.tensor([MOVE_TO_IDX[pat['target']] for pat in patterns],
                                    dtype=torch.long, device=device)
    idx_t, mask_t = _get_cell_pat_index(pattern_to_cell, 60)

    # ---------------------- Load games --------------------------------------
    games_raw = load_games(max_files=args.max_files)
    if len(games_raw) > args.n_games:
        games_raw = games_raw[:args.n_games]
    print(f"Loaded {len(games_raw)} games")

    # Convert each game's moves to MOVE_TO_IDX (60-cell index) -- they should
    # already be in 0..63 board space; map skipping centers.
    games_cells = []
    for g in games_raw:
        cells = [_b64_to_m60[m] for m in g]
        games_cells.append(cells)

    # Choose fixed positions for batching efficiency
    rng = np.random.RandomState(0)
    positions = sorted(rng.choice(
        range(args.pos_start, args.pos_end), args.positions_per_game,
        replace=False).tolist())
    print(f"Positions: {positions}")

    # ---------------------- Forward all games through OGPT -------------------
    tokens = tokenize_games(games_raw, seq_len=block_size).to(device)
    print(f"Tokens: {tokens.shape}, forwarding through OGPT to layer {args.ogpt_layer}...")
    H_test_at_L = []
    with torch.no_grad():
        bs = 64
        for i in range(0, len(games_raw), bs):
            H_test_at_L.append(
                forward_to_layer(ogpt, tokens[i:i+bs], args.ogpt_layer).cpu()
            )
    H_test_at_L = torch.cat(H_test_at_L, dim=0)   # (G, T, d_model)
    print(f"OGPT cached resid stream shape: {H_test_at_L.shape}")

    # ---------------------- Iterate, compute interventions -------------------
    # Storage: per-intervention rows.
    records = []   # each row dict-like
    # Pre-compute board states once per game (we need them for legality checks).
    print("Replaying games for board states...")
    boards_per_game = np.zeros((len(games_raw), GAME_LEN, 8, 8), dtype=np.int8)
    for gi, g in enumerate(games_raw):
        b = OthelloBoardState()
        for t, m in enumerate(g):
            b.umpire(m)
            boards_per_game[gi, t] = np.asarray(b.state, dtype=np.int8)

    interventions_done = 0
    print(f"\nStarting interventions for {len(games_raw)} games x "
          f"{len(positions)} positions...")
    print(f"({interventions_done} so far)", flush=True)

    with torch.no_grad():
        for gi, game_cells in enumerate(games_cells):
            for pos in positions:
                if pos < 1: continue
                state8 = boards_per_game[gi, pos - 1]   # board AFTER move (pos-1)
                # Next-player color: at OGPT position 'pos', we predict move
                # 'pos+1'; player is black if (pos+1) even (since move 0=black).
                # In Othello convention: black=+1, white=-1.
                next_move_idx = pos + 1
                next_color_int = 1 if next_move_idx % 2 == 1 else -1
                # ^ check: move 0 is by black, so next_move_idx % 2 == 1 means
                #   move 1, by white. Hmm wait. Move 0: black. Move 1: white.
                # So move m is by black iff m even. next_color_int = +1 if
                # (pos+1) even == ... let me re-derive.
                next_color_int = 1 if (pos + 1) % 2 == 0 else -1

                # OGPT class indices for mine/theirs:
                #   class 1 = white(-1), class 2 = black(+1)
                ogpt_mine_class = 2 if next_color_int == 1 else 1
                ogpt_theirs_class = 1 if next_color_int == 1 else 2

                # MLP class indices (player-relative):
                #   class 0=empty, 1=mine, 2=theirs
                mlp_mine_class = 1
                mlp_theirs_class = 2

                # ---- Compute baseline cell probs for both models -----------
                # MLP forward
                feats = game_to_features_at_t(game_cells, pos)
                feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
                even_pos = (pos % 2 == 0)
                mlp_net = me if even_pos else mo
                # Hidden
                h_mlp = torch.relu(mlp_net.net[0](feats_t))   # (1, H)
                out_mlp = mlp_net.net[2](h_mlp)               # (1, 960) -- skip ReLU? No, net is Sequential(Linear, ReLU, Linear)
                # Wait, DirectMLP uses net = nn.Sequential(Linear, ReLU, Linear).
                # h_mlp after Linear+ReLU. out is the next Linear.
                base_p_mlp = mlp_cell_probs(out_mlp, idx_t, mask_t).cpu().numpy()[0]   # (60,)

                # OGPT: get logits at this position from the cached residual stream
                h_ogpt = H_test_at_L[gi:gi+1].to(device)
                logits_ogpt = forward_from_layer(ogpt, h_ogpt, args.ogpt_layer)
                base_p_ogpt = ogpt_cell_probs(logits_ogpt[0, pos, :].unsqueeze(0), cell_tokens_t)[0].cpu().numpy()

                # ---- Iterate intervention types ----------------------------
                # For each direction type, sample target cells and apply intervention.
                # Three types: 'mine', 'theirs', 'empty'.
                for intv_type in ("mine", "theirs", "empty"):
                    if intv_type in ("mine", "theirs"):
                        # Target = empty cell (currently 0 on board).
                        candidates_64 = []
                        for c64 in _movable_64:
                            r, c = c64 // 8, c64 % 8
                            if state8[r, c] == 0:
                                candidates_64.append(c64)
                        new_state = next_color_int if intv_type == "mine" else -next_color_int
                        mlp_k = mlp_mine_class if intv_type == "mine" else mlp_theirs_class
                        ogpt_k = ogpt_mine_class if intv_type == "mine" else ogpt_theirs_class
                    else:   # empty
                        # Target = currently occupied cell.
                        candidates_64 = []
                        for c64 in _movable_64:
                            r, c = c64 // 8, c64 % 8
                            if state8[r, c] != 0:
                                candidates_64.append(c64)
                        new_state = 0
                        mlp_k = 0
                        ogpt_k = 0

                    # Filter candidates: only those where the modification has
                    # a non-empty newly_legal/newly_illegal set.
                    useful = []
                    for c64 in candidates_64:
                        orig_l, nl, ni = compute_newly_legal_illegal(
                            state8, c64, new_state, next_color_int)
                        if nl.sum() == 0 and ni.sum() == 0:
                            continue
                        useful.append((c64, nl, ni))
                    if not useful:
                        continue
                    # Cap at max_targets
                    if len(useful) > args.max_targets_per_direction:
                        useful = useful[:args.max_targets_per_direction]

                    # Apply intervention for each target
                    for c64, nl, ni in useful:
                        m60 = _b64_to_m60[c64]
                        # MLP intervention
                        d_mlp = mlp_dirs_t[m60, mlp_k]
                        h_mlp_new = h_mlp + args.scale * d_mlp.unsqueeze(0)
                        out_mlp_new = mlp_net.net[2](h_mlp_new)
                        new_p_mlp = mlp_cell_probs(out_mlp_new, idx_t, mask_t).cpu().numpy()[0]

                        # OGPT intervention
                        d_ogpt = ogpt_dirs_t[m60, ogpt_k]
                        h_ogpt_new = h_ogpt.clone()
                        h_ogpt_new[0, pos, :] = h_ogpt_new[0, pos, :] + args.scale * d_ogpt
                        logits_new = forward_from_layer(ogpt, h_ogpt_new, args.ogpt_layer)
                        new_p_ogpt = ogpt_cell_probs(logits_new[0, pos, :].unsqueeze(0), cell_tokens_t)[0].cpu().numpy()

                        # Record per-cell dP, plus targeted sets
                        dP_mlp = new_p_mlp - base_p_mlp
                        dP_ogpt = new_p_ogpt - base_p_ogpt
                        records.append({
                            'gi': gi, 'pos': pos, 'target_c64': int(c64),
                            'intv_type': intv_type,
                            'n_newly_legal': int(nl.sum()),
                            'n_newly_illegal': int(ni.sum()),
                            # Aggregates
                            'mean_dP_nl_mlp':   float(dP_mlp[nl].mean()) if nl.any() else float('nan'),
                            'mean_dP_nl_ogpt':  float(dP_ogpt[nl].mean()) if nl.any() else float('nan'),
                            'mean_dP_ni_mlp':   float(dP_mlp[ni].mean()) if ni.any() else float('nan'),
                            'mean_dP_ni_ogpt':  float(dP_ogpt[ni].mean()) if ni.any() else float('nan'),
                            'dP_target_mlp':    float(dP_mlp[m60]),
                            'dP_target_ogpt':   float(dP_ogpt[m60]),
                        })
                        interventions_done += 1

            if (gi + 1) % 1000 == 0:
                print(f"  game {gi+1}/{len(games_raw)}: "
                      f"{interventions_done} interventions so far",
                      flush=True)
            if (gi + 1) % args.checkpoint_every == 0 and records:
                # Partial save (overwrites prior partial). Survives timeouts.
                keys = list(records[0].keys())
                out_partial = {}
                for k in keys:
                    if k == 'intv_type':
                        out_partial[k] = np.array([r[k] for r in records], dtype='<U10')
                    else:
                        out_partial[k] = np.array([r[k] for r in records])
                out_partial['games_processed'] = np.array([gi + 1])
                np.savez_compressed(args.output, **out_partial)
                print(f"  [checkpoint] saved {len(records)} rows "
                      f"after game {gi+1}", flush=True)

    print(f"\nTotal interventions: {interventions_done}")
    if interventions_done == 0:
        sys.exit(1)

    # Convert records to arrays for npz save
    keys = list(records[0].keys())
    out = {}
    for k in keys:
        if k == 'intv_type':
            out[k] = np.array([r[k] for r in records], dtype='<U10')
        else:
            out[k] = np.array([r[k] for r in records])
    np.savez_compressed(args.output, **out)
    print(f"Saved {len(records)} rows to {args.output}")

    # Quick summary
    print()
    print("Per-intervention-type summary (mean dP on affected cells):")
    print(f"{'type':>10s} {'n':>8s} {'<dP_nl MLP>':>14s} {'<dP_nl OGPT>':>14s} "
          f"{'<dP_ni MLP>':>14s} {'<dP_ni OGPT>':>14s}")
    for t in ("mine", "theirs", "empty"):
        mask = out['intv_type'] == t
        n = int(mask.sum())
        if n == 0: continue
        nl_mlp  = float(np.nanmean(out['mean_dP_nl_mlp'][mask]))
        nl_ogpt = float(np.nanmean(out['mean_dP_nl_ogpt'][mask]))
        ni_mlp  = float(np.nanmean(out['mean_dP_ni_mlp'][mask]))
        ni_ogpt = float(np.nanmean(out['mean_dP_ni_ogpt'][mask]))
        print(f"{t:>10s} {n:>8d} {nl_mlp:>+14.4f} {nl_ogpt:>+14.4f} "
              f"{ni_mlp:>+14.4f} {ni_ogpt:>+14.4f}")

    print()
    print("Sign agreement (fraction of cases where both models have dP in the "
          "predicted direction):")
    for t in ("mine", "theirs"):
        mask = out['intv_type'] == t
        good = ~np.isnan(out['mean_dP_nl_mlp']) & ~np.isnan(out['mean_dP_nl_ogpt'])
        m = mask & good
        if m.sum() == 0: continue
        both_pos = (out['mean_dP_nl_mlp'][m] > 0) & (out['mean_dP_nl_ogpt'][m] > 0)
        print(f"  {t:>10s} newly_legal:   both dP > 0 in {both_pos.mean():.2%} of {m.sum()} cases")
    for t in ("empty",):
        mask = out['intv_type'] == t
        good = ~np.isnan(out['mean_dP_ni_mlp']) & ~np.isnan(out['mean_dP_ni_ogpt'])
        m = mask & good
        if m.sum() == 0: continue
        both_neg = (out['mean_dP_ni_mlp'][m] < 0) & (out['mean_dP_ni_ogpt'][m] < 0)
        print(f"  {t:>10s} newly_illegal: both dP < 0 in {both_neg.mean():.2%} of {m.sum()} cases")
