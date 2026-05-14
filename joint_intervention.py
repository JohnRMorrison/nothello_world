"""Joint intervention experiment: OGPT vs MLP (H=4096 movegrid) on the
EXACT SAME positions with matched cell-flip modifications.

For each (game, position, modification) sample:
  - Apply a counterfactual modification (flip one board cell's color).
  - OGPT: intervene in hidden state using Nanda's probe direction for the
    flipped cell. Forward through remaining layers to softmax (60-d).
  - MLP: same flip spec, intervene in the MLP H=4096 movegrid hidden using
    the MLP probe's direction for the flipped cell. Forward through the
    pattern head, aggregate to per-cell prob_or, NORMALIZE so the 60-cell
    output is a proper distribution.
  - Save orig_probs and intv_probs for both models, plus original_legal /
    counterfactual_legal sets.

Then computes 5 metrics for both models on the saved samples.

Usage:
    python joint_intervention.py --n-games 200 --scale 3.0
"""
import sys, os, json, argparse, time
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F

from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
from train_pattern_simple import DirectMLP, to_move_grid_input

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, load_games, VOCAB_SIZE, GAME_LEN,
)
sys.path.insert(0, "mechanistic_interpretability")


CENTER_64 = {27, 28, 35, 36}
NON_CENTER_64 = [c for c in range(64) if c not in CENTER_64]
B64_TO_M60 = {c: i for i, c in enumerate(NON_CENTER_64)}


# --------- modification picker (single-cell flip, non-trivial position) ---------
def pick_flip_modification(board_2d, rng):
    """Pick one cell to flip from its current non-empty color to the opposite.
    Returns (row, col, src_class, tgt_class) or None.
    Classes: 0=empty, 1=white, 2=black (matching probe class convention)."""
    # Find non-empty cells
    occupied = [(r, c) for r in range(8) for c in range(8)
                if board_2d[r, c] != 0]
    if not occupied:
        return None
    rng.shuffle(occupied)
    r, c = occupied[0]
    src = 1 if board_2d[r, c] == -1 else 2   # white(-1)->1, black(+1)->2
    tgt = 3 - src                            # 1<->2
    return (r, c, src, tgt)


def compute_legal_moves(board_state, is_black_turn):
    """Returns set of cell-index 0..63 where the given player can legally play."""
    b = OthelloBoardState()
    b.state = np.copy(board_state)
    b.next_hand_color = 1 if is_black_turn else -1
    return set(b.get_valid_moves())


# --------- OGPT intervention ---------
def ogpt_forward_and_intervene(model, tokens_b, pos, probe_ogpt, mod, scale,
                                layer_intervene, device):
    """Run OGPT once with no intervention and once with intervention at
    layer_intervene, return (orig_probs_60, intv_probs_60)."""
    # Run baseline forward with cache
    def run_forward(tokens, intervene_at_pos=None, h_delta=None):
        # Manual forward through mingpt
        b, t = tokens.size()
        tok_emb = model.tok_emb(tokens)
        pos_emb = model.pos_emb[:, :t, :]
        x = model.drop(tok_emb + pos_emb)
        for li, block in enumerate(model.blocks):
            x = block(x)
            if intervene_at_pos is not None and li == layer_intervene - 1:
                # Apply h_delta at the target position
                x[0, intervene_at_pos] = x[0, intervene_at_pos] + h_delta
        x = model.ln_f(x)
        logits = model.head(x)
        return logits[0, intervene_at_pos if intervene_at_pos is not None else pos]

    with torch.no_grad():
        # Baseline (no intervention)
        b, t = tokens_b.size()
        tok_emb = model.tok_emb(tokens_b)
        pos_emb = model.pos_emb[:, :t, :]
        x = model.drop(tok_emb + pos_emb)
        for li, block in enumerate(model.blocks):
            x = block(x)
            if li == layer_intervene - 1:
                h_pre = x[0, pos].clone()
        x_baseline = model.ln_f(x)
        logits_baseline = model.head(x_baseline)[0, pos]

        # Build probe direction for the flipped cell
        r, c, src, tgt = mod
        parity_mode = 0 if pos % 2 == 0 else 1
        w_src = probe_ogpt[parity_mode, :, r, c, src]   # (512,)
        w_tgt = probe_ogpt[parity_mode, :, r, c, tgt]   # (512,)
        direction = w_tgt - w_src
        direction = direction / direction.norm().clamp_min(1e-8)
        h_delta = scale * direction

        # Intervened forward
        tok_emb = model.tok_emb(tokens_b)
        x = model.drop(tok_emb + pos_emb)
        for li, block in enumerate(model.blocks):
            x = block(x)
            if li == layer_intervene - 1:
                x[0, pos] = x[0, pos] + h_delta
        x_intv = model.ln_f(x)
        logits_intv = model.head(x_intv)[0, pos]

    # Convert to 60-cell prob distribution
    # OGPT vocab: token i maps to board cell NON_CENTER_64[i]. Take softmax over
    # the 60 non-special tokens. Vocab includes special tokens (eg 0=pass).
    # Standard convention: tokens 1..60 = cell moves.
    def logits_to_60(logits):
        # mingpt OGPT vocab size 61: token 0 = special (pass), tokens 1..60 = cells
        probs = F.softmax(logits, dim=-1)
        return probs[1:61].cpu().numpy()  # (60,)
    return logits_to_60(logits_baseline), logits_to_60(logits_intv)


# --------- MLP intervention ---------
def hidden_of(model, x):
    return torch.relu(model.net[0](x))


def aggregate_cell_probs(pat_logits, idx_t, mask_t):
    """prob_or aggregation per cell -> 60-d probabilities in [0,1]."""
    log1m = -F.softplus(pat_logits)
    g = log1m[:, idx_t]
    g = g.masked_fill(~mask_t, 0.0)
    return 1.0 - torch.exp(g.sum(dim=-1))   # (B, 60)


def mlp_forward_and_intervene(features_180, board_state, pos, mod, scale,
                               model_e, model_o, probe_e, probe_o,
                               idx_t, mask_t, device, negate=False):
    """Run MLP intervention, return (orig_probs_60, intv_probs_60) NORMALIZED."""
    # Build 3600-d move_grid input
    x180 = features_180.unsqueeze(0).to(device)   # (1, 180)
    x = to_move_grid_input(x180)                   # (1, 3600)

    # Parity-correct model + probe
    if pos % 2 == 0:
        model, probe = model_e, probe_e
    else:
        model, probe = model_o, probe_o

    with torch.no_grad():
        # Baseline forward
        h = hidden_of(model, x)                    # (1, H)
        out_layer = model.net[2]                   # Linear(H, 960)
        pat_logits_pre = out_layer(h)              # (1, 960)
        cell_probs_pre = aggregate_cell_probs(pat_logits_pre, idx_t, mask_t)
        cell_probs_pre = cell_probs_pre / cell_probs_pre.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Compute probe direction for the flipped cell (in 64-cell space)
        r, c, src, tgt = mod
        cell = r * 8 + c   # 0..63
        # probe.weight is (192, H) where 192 = 64*3 (cell-class flattened)
        # Row [cell*3 + class] is the direction in hidden for that (cell, class).
        w_src = probe.weight[cell * 3 + src]       # (H,)
        w_tgt = probe.weight[cell * 3 + tgt]       # (H,)
        direction = w_tgt - w_src
        direction = direction / direction.norm().clamp_min(1e-8)
        sign = -1.0 if negate else 1.0
        h_intv = h + sign * scale * direction.unsqueeze(0)

        pat_logits_post = out_layer(h_intv)
        cell_probs_post = aggregate_cell_probs(pat_logits_post, idx_t, mask_t)
        cell_probs_post = cell_probs_post / cell_probs_post.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    return cell_probs_pre.squeeze().cpu().numpy(), cell_probs_post.squeeze().cpu().numpy()


# --------- Driver ---------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ogpt-ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    parser.add_argument("--ogpt-probe", default="mechanistic_interpretability/main_linear_probe.pth")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--mlp-ckpt", required=True)
    parser.add_argument("--mlp-probe-ckpt", required=True)
    parser.add_argument("--mlp-hidden", type=int, default=4096)
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--turn-min", type=int, default=20)
    parser.add_argument("--turn-max", type=int, default=50)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--mlp-negate-direction", action="store_true",
                        help="Negate the MLP intervention direction "
                             "(sign-flipped sanity check).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="logs/joint_intervention.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(args.seed)
    print(f"Device: {device}, seed={args.seed}, scale={args.scale}")

    # OGPT + probe
    config = GPTConfig(vocab_size=VOCAB_SIZE, block_size=59,
                       n_layer=8, n_head=8, n_embd=512)
    ogpt = GPT(config); ogpt.to(device).eval()
    state = torch.load(args.ogpt_ckpt, map_location=device)
    if 'model_state_dict' in state: state = state['model_state_dict']
    ogpt.load_state_dict(state)
    probe_ogpt = torch.load(args.ogpt_probe, map_location=device)
    print(f"OGPT probe: {tuple(probe_ogpt.shape)}")

    # MLP + probe (parity-split)
    pat = torch.load(args.mlp_ckpt, map_location=device)
    mlp_input_dim = pat.get('input_dim', 3600)
    model_e = DirectMLP(mlp_input_dim, args.mlp_hidden,
                        pat.get('n_patterns', 960)).to(device)
    model_o = DirectMLP(mlp_input_dim, args.mlp_hidden,
                        pat.get('n_patterns', 960)).to(device)
    model_e.load_state_dict(pat['even']); model_o.load_state_dict(pat['odd'])
    model_e.eval(); model_o.eval()
    probe_state = torch.load(args.mlp_probe_ckpt, map_location=device)
    import torch.nn as nn
    probe_e = nn.Linear(args.mlp_hidden, 64 * 3).to(device)
    probe_o = nn.Linear(args.mlp_hidden, 64 * 3).to(device)
    probe_e.load_state_dict(probe_state['even']); probe_o.load_state_dict(probe_state['odd'])
    probe_e.eval(); probe_o.eval()

    # Pattern index for prob_or aggregation
    from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
    from train_pattern_simple import _get_cell_pat_index
    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor([MOVE_TO_IDX[p['target']] for p in patterns],
                                    dtype=torch.long, device=device)
    idx_t, mask_t = _get_cell_pat_index(pattern_to_cell, 60)

    # Load games (shared)
    games = load_games(max_files=args.max_files)
    games = [g for g in games if len(g) == GAME_LEN][:args.n_games]
    print(f"Using {len(games)} games")
    # OGPT block_size=59, so drop the last token (same as Nanda's training).
    tokens_all = tokenize_games(games).to(device)[:, :-1]   # (G, 59)

    # For each game, pick one (turn, mod) and run both interventions
    samples = []
    t0 = time.time()
    for gi, game in enumerate(games):
        turn = int(rng.randint(args.turn_min, args.turn_max + 1))
        # Replay game to get pre-move board state at this turn
        board = OthelloBoardState()
        for s in range(turn):
            try: board.umpire(game[s])
            except Exception: break
        board_2d = np.asarray(board.state, dtype=np.int8)

        mod = pick_flip_modification(board_2d, rng)
        if mod is None: continue

        # Compute counterfactual board (flip the chosen cell)
        cf_board_2d = np.copy(board_2d)
        r, c, src, tgt = mod
        cf_board_2d[r, c] = 1 if tgt == 2 else -1   # 2->black(+1), 1->white(-1)

        is_black_turn = (turn % 2 == 0)
        original_legal = compute_legal_moves(board_2d, is_black_turn)
        counterfactual_legal = compute_legal_moves(cf_board_2d, is_black_turn)
        if not (counterfactual_legal - original_legal) and not (original_legal - counterfactual_legal):
            continue   # no meaningful change

        # Build MLP features (180-d move history at this position)
        played = np.zeros(60, dtype=np.float32)
        when = np.zeros(60, dtype=np.float32)
        even = np.zeros(60, dtype=np.float32)
        for s in range(turn):
            m60 = B64_TO_M60.get(game[s], -1)
            if m60 < 0: continue
            played[m60] = 1.0
            when[m60] = (s + 1) / 60.0
            if s % 2 == 0: even[m60] = 1.0
        features_180 = torch.from_numpy(
            np.concatenate([played, when, even]).astype(np.float32))

        # OGPT intervention
        try:
            ogpt_orig, ogpt_intv = ogpt_forward_and_intervene(
                ogpt, tokens_all[gi:gi+1], turn, probe_ogpt, mod,
                args.scale, args.layer, device)
        except Exception as e:
            print(f"  game {gi}: OGPT failed: {e}")
            continue

        # MLP intervention
        try:
            mlp_orig, mlp_intv = mlp_forward_and_intervene(
                features_180, board_2d, turn, mod, args.scale,
                model_e, model_o, probe_e, probe_o,
                idx_t, mask_t, device,
                negate=args.mlp_negate_direction)
        except Exception as e:
            print(f"  game {gi}: MLP failed: {e}")
            continue

        samples.append({
            "game_id": gi, "turn": turn, "mod": mod,
            "original_legal": sorted(original_legal),
            "counterfactual_legal": sorted(counterfactual_legal),
            "ogpt_orig_probs": ogpt_orig.tolist(),
            "ogpt_intv_probs": ogpt_intv.tolist(),
            "mlp_orig_probs": mlp_orig.tolist(),
            "mlp_intv_probs": mlp_intv.tolist(),
        })
        if (gi + 1) % 50 == 0:
            print(f"  {gi+1}/{len(games)} games, "
                  f"{len(samples)} samples kept, "
                  f"{time.time()-t0:.0f}s", flush=True)

    print(f"\nKept {len(samples)} samples ({time.time()-t0:.0f}s total)")

    # ---------------- Compute the 5 metrics ----------------
    def compute_metrics(samples, model_key):
        """Return dict of the 5 metrics for a given model."""
        d_leg_mass = []         # (1) ΔP_legal_mass
        d_illeg_mass = []       # (= -ΔP_legal_mass; sanity check)
        ratio_nl_pre = []       # (3a) newly_legal / cf_legal mass pre
        ratio_nl_post = []      # (3b) newly_legal / cf_legal mass post
        nl_mean_dp = []         # (4) mean ΔP over newly_legal cells
        ni_mean_dp = []         # (5) mean ΔP over newly_illegal cells

        for s in samples:
            orig = np.array(s[f"{model_key}_orig_probs"])
            intv = np.array(s[f"{model_key}_intv_probs"])
            origL = set(s["original_legal"])
            cfL   = set(s["counterfactual_legal"])
            newly_legal   = list(cfL - origL)
            newly_illegal = list(origL - cfL)
            cf_list = list(cfL)

            # b64 -> m60 (probs are over 60 non-center cells)
            def to_m60(cells):
                return [B64_TO_M60[c] for c in cells if c in B64_TO_M60]
            nl_m60 = to_m60(newly_legal)
            ni_m60 = to_m60(newly_illegal)
            cf_m60 = to_m60(cf_list)

            if not cf_m60: continue
            leg_pre = orig[cf_m60].sum()
            leg_post = intv[cf_m60].sum()
            d_leg_mass.append(leg_post - leg_pre)
            d_illeg_mass.append((1 - leg_post) - (1 - leg_pre))

            if nl_m60:
                ratio_nl_pre.append(orig[nl_m60].sum() / max(leg_pre, 1e-9))
                ratio_nl_post.append(intv[nl_m60].sum() / max(leg_post, 1e-9))
                nl_mean_dp.append((intv[nl_m60] - orig[nl_m60]).mean())
            if ni_m60:
                ni_mean_dp.append((intv[ni_m60] - orig[ni_m60]).mean())

        return {
            "n": len(d_leg_mass),
            "1_ΔP_legal_mass":          float(np.mean(d_leg_mass)),
            "2_ΔP_illegal_mass":        float(np.mean(d_illeg_mass)),
            "3a_ratio_NL_pre":          float(np.mean(ratio_nl_pre)) if ratio_nl_pre else None,
            "3b_ratio_NL_post":         float(np.mean(ratio_nl_post)) if ratio_nl_post else None,
            "4_mean_ΔP_newly_legal":    float(np.mean(nl_mean_dp)) if nl_mean_dp else None,
            "5_mean_ΔP_newly_illegal":  float(np.mean(ni_mean_dp)) if ni_mean_dp else None,
        }

    ogpt_metrics = compute_metrics(samples, "ogpt")
    mlp_metrics  = compute_metrics(samples, "mlp")
    print("\n========== METRICS ==========")
    print(f"{'metric':<28s} {'OGPT':>10s} {'MLP':>10s}")
    for k in ogpt_metrics:
        if k == "n":
            print(f"{k:<28s} {ogpt_metrics[k]:>10d} {mlp_metrics[k]:>10d}")
        else:
            ov = ogpt_metrics[k]; mv = mlp_metrics[k]
            ov_s = f"{ov:>10.5f}" if ov is not None else f"{'-':>10s}"
            mv_s = f"{mv:>10.5f}" if mv is not None else f"{'-':>10s}"
            print(f"{k:<28s} {ov_s} {mv_s}")

    # Save samples + metrics
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output,
        samples_json=json.dumps(samples),
        ogpt_metrics=json.dumps(ogpt_metrics),
        mlp_metrics=json.dumps(mlp_metrics),
        scale=args.scale, layer=args.layer)
    print(f"\nSaved {args.output}")
