"""
OGPT causal-intervention probability-mass shift measurement.

For 100 (game, position) pairs (one position per game), apply a single
probe-directed intervention at the residual stream and measure how
probability mass on counterfactual-legal moves shifts.

Setup:
  - Model: OthelloGPT (synthetic-trained, mingpt format)
  - Probe: Nanda's linear probe (mode 2 = "all positions"), located at
    mingpt resid_post of block 6
  - Intervention point: resid_post of block 4 (mingpt notation)
  - Calibration: cd=2 — per-cell binary search for the minimal scale that
    flips the Nanda probe's argmax at resid_post of block 6 (i.e. after
    running through blocks 5 and 6 post-intervention). Probe direction is
    Nanda probe at the cell, target class.
  - Sampling: 100 distinct games, one random position per game in
    [POS_LO, POS_HI). For each position, sample one random (square, type)
    consistent with current board state; resample if intervention does
    not change legality of any move (capped retries).

Intervention type (orig_val → target_val on the board cell):
  B->W  (+1 → -1)
  W->B  (-1 → +1)
  E->B  ( 0 → +1)
  E->W  ( 0 → -1)

Metrics (per intervention), using softmax over the 60 valid cell logits:
  legal_cf      = legal moves under modified board (next player to move)
  newly_legal   = legal_cf - legal_orig

  P_before(S)   = sum of pre-intervention probs over cells in S
  P_after(S)    = sum of post-intervention probs over cells in S

  (1) abs_dP_legal       = P_after(legal_cf) - P_before(legal_cf)
  (2) pct_dP_legal       = abs_dP_legal / P_after(legal_cf)
  (3) abs_dP_newly_legal = P_after(newly_legal) - P_before(newly_legal)
  (4) pct_dP_newly_legal = abs_dP_newly_legal / P_after(legal_cf)

Usage:
  python ogpt_legal_mass_shift.py --n-games 100 --seed 42 \
      --output logs/ogpt_legal_mass_shift.csv
"""

import argparse
import os
import random
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.othello import OthelloBoardState
from mingpt.model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STOI_INDICES = list(range(27)) + list(range(29, 35)) + list(range(37, 64))  # 60 valid cells
STOI_INDEX_OF = {c: i for i, c in enumerate(STOI_INDICES)}
CENTER_CELLS = {27, 28, 35, 36}
ALPHA = "ABCDEFGH"
# Empirical: only mode 0 (and mode 1) of main_linear_probe.pth reaches ~99%
# board-state accuracy at layer 6. Both use a parity-aware "mine/yours/empty"
# label convention, NOT absolute colors. Mode 2 is degraded (~79%) and should
# not be used (multi_intervention.py / sweep_intervention_alpha.py /
# ogpt_intervention.py / compare_interventions.py defaults all use mode 2 +
# absolute and are incorrect on this point).
# Mode 0 convention: empty=0, yours=1, mine=2  (mine = next-to-play color).
PROBE_LAYER = 6      # Nanda probe lives at resid_post of mingpt block 6
# INTERVENE_LAYER is now determined by --cal-depth at runtime:
#   intervene_layer = PROBE_LAYER - cal_depth
# cd=0: intervene at block 6 (same layer as probe).
# cd=2: intervene at block 4 (two blocks upstream of probe).


def board_val_to_probe_class(val, next_color, probe_kind):
    """Return the probe class index for `val` under the chosen convention.

    probe_kind = "mode0_mineyours" (correct, ~99% probe acc):
        empty=0, yours=1, mine=2  (mine = next-to-play color).
    probe_kind = "mode2_abs" (buggy reference, ~77% probe acc):
        empty=0, white(-1)=1, black(+1)=2.
    """
    if probe_kind == "mode0_mineyours":
        if val == 0:
            return 0
        return 2 if val == next_color else 1
    if probe_kind == "mode2_abs":
        if val == 0:
            return 0
        return 1 if val == -1 else 2
    raise ValueError(probe_kind)


PROBE_KIND_TO_MODE = {"mode0_mineyours": 0, "mode2_abs": 2}


def intervene_layer_for(cal_depth, probe_layer=6):
    """Layer at which we apply the intervention, given calibration depth.

    cd=0: intervene at the probe layer; same-layer round-trip.
    cd=k: intervene k blocks upstream of the probe layer.
    """
    return probe_layer - cal_depth


def cell_label(cell):
    return f"{ALPHA[cell // 8]}{cell % 8}"


def intervention_label(orig_val, target_val):
    name = {-1: "W", 0: "E", 1: "B"}
    return f"{name[orig_val]}->{name[target_val]}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_model(ckpt_path, device):
    config = GPTConfig(vocab_size=61, block_size=59, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_probe(probe_path, device):
    """Nanda linear probe, shape (3, 512, 8, 8, 3) = (modes, d_model, r, c, class)."""
    probe = torch.load(probe_path, map_location=device)
    return probe


def load_games(int_path, string_path):
    board_seqs_int = np.load(int_path)
    board_seqs_string = np.load(string_path)
    return board_seqs_int, board_seqs_string


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------
def replay(game_string, pos):
    """Replay game (list of cell indices 0-63) up to and including `pos`.
    Returns (8x8 board state, next_hand_color)."""
    board = OthelloBoardState()
    for i in range(pos + 1):
        board.umpire(int(game_string[i]))
    return board.state.copy(), board.next_hand_color


def legal_moves_for(board_state, color):
    """Return set of legal move cells (0-63) given a board state + side to move."""
    b = OthelloBoardState()
    b.state = board_state.copy().astype(np.float64)
    b.next_hand_color = int(color)
    return set(b.get_valid_moves())


# ---------------------------------------------------------------------------
# Intervention sampling
# ---------------------------------------------------------------------------
CATEGORY_TARGETS = {
    # category -> dict[orig_val] -> list of valid target_vals
    "remove": {1: [0], -1: [0]},                # B->E or W->E (occupied -> empty)
    "add":    {0: [1, -1]},                     # E->B or E->W
    "flip":   {1: [-1], -1: [1]},               # B->W or W->B
    "any":    {0: [1, -1], 1: [-1, 0], -1: [1, 0]},   # superset (legacy)
}


def sample_intervention(board_state, color, rng, category="any", max_tries=200):
    """Pick a random non-center cell + valid intervention of the given
    category that changes legality of at least one move.

    category: 'remove' | 'add' | 'flip' | 'any' (see CATEGORY_TARGETS).

    Returns (cell, orig_val, target_val, legal_orig, legal_cf) or None.
    """
    legal_orig = legal_moves_for(board_state, color)
    cells = [c for c in range(64) if c not in CENTER_CELLS]
    targets_by_orig = CATEGORY_TARGETS[category]
    tried = set()
    for _ in range(max_tries):
        cell = rng.choice(cells)
        r, c = cell // 8, cell % 8
        orig_val = int(board_state[r, c])
        if orig_val not in targets_by_orig:
            continue  # this cell's state doesn't match the category
        target_val = rng.choice(targets_by_orig[orig_val])
        key = (cell, target_val)
        if key in tried:
            continue
        tried.add(key)
        modified = board_state.copy()
        modified[r, c] = target_val
        legal_cf = legal_moves_for(modified, color)
        if legal_cf == legal_orig:
            continue  # no legality change — try another
        if not legal_cf:
            continue  # degenerate: no legal moves at all (forfeit edge case)
        return cell, orig_val, target_val, legal_orig, legal_cf
    return None


# ---------------------------------------------------------------------------
# cd=2 calibration with the Nanda probe
# ---------------------------------------------------------------------------
def compute_prefix(model, input_tokens, layer_intervene):
    """Forward through embedding + blocks[:layer_intervene]; returns the
    residual stream at the intervention point."""
    b, t = input_tokens.size()
    tok = model.tok_emb(input_tokens)
    pos = model.pos_emb[:, :t, :]
    x = model.drop(tok + pos)
    for block in model.blocks[:layer_intervene]:
        x = block(x)
    return x


def forward_from_prefix(model, x, layer_intervene):
    """Continue forward from intervention point through remaining blocks
    + final layernorm + head. Returns logits at last position."""
    for block in model.blocks[layer_intervene:]:
        x = block(x)
    x = model.ln_f(x)
    return model.head(x)


def forward_from_prefix_capturing(model, x, layer_intervene, capture_layer):
    """Like forward_from_prefix but also captures resid_post of mingpt block
    `capture_layer` along the way. Returns (logits, resid_at_capture_layer).

    Edge case: if capture_layer == layer_intervene - 1, the input x already
    IS resid_post of the capture layer (since prefix_acts was built by
    running blocks[:layer_intervene]) — so we capture x itself before any
    further blocks run. This is what cd=0 needs."""
    resid_at_capture = None
    if capture_layer == layer_intervene - 1:
        resid_at_capture = x.clone()
    for i, block in enumerate(model.blocks[layer_intervene:],
                              start=layer_intervene):
        x = block(x)
        if i == capture_layer:
            resid_at_capture = x.clone()
    x = model.ln_f(x)
    logits = model.head(x)
    return logits, resid_at_capture


def cdN_calibrate(model, prefix_acts, pos, probe_cell_W, current_class,
                  target_class, intervene_layer, probe_layer):
    """Binary search for minimal scale that flips the probe's argmax at the
    probe layer after intervening at the intervene layer.

    Works for any cal_depth = probe_layer - intervene_layer:
      cd=0: same layer, no forward propagation between intervention and probe.
      cd>0: forward through (probe_layer - intervene_layer) blocks before
            probing.

    intervene_layer = mingpt block index whose resid_post we modify
    probe_layer    = mingpt block index whose resid_post the probe reads

    `prefix_acts` is already shaped (1, T, d_model) at resid_post of
    block `intervene_layer` (i.e. obtained via
    compute_prefix(model, tokens, intervene_layer + 1)).

    probe_cell_W: (d_model, 3) — Nanda probe weights for this cell.
    Returns scale s in [0, 10].
    """
    # In mingpt: resid_post of block L = output of blocks[:L+1].
    # We have prefix at resid_post of block `intervene_layer` and want to
    # run through blocks (intervene_layer+1)..probe_layer to reach resid_post
    # of block `probe_layer`.
    h = prefix_acts[0, pos].detach()
    flip_dir = probe_cell_W[:, target_class] - probe_cell_W[:, current_class]
    d_hat = flip_dir / flip_dir.norm()
    coeff = (h @ d_hat).item()

    blocks_to_run = model.blocks[intervene_layer + 1: probe_layer + 1]

    def probe_argmax_at_scale(s):
        h_mod = h - s * coeff * d_hat
        x = prefix_acts.clone()
        x[0, pos] = h_mod
        with torch.no_grad():
            for block in blocks_to_run:
                x = block(x)
            act = x[0, pos]
            logits = probe_cell_W.T @ act  # (3,)
        return int(logits.argmax().item())

    # First check feasibility / triviality.
    if probe_argmax_at_scale(0.0) == target_class:
        return 0.5  # already at target with no nudge — use small scale
    if probe_argmax_at_scale(10.0) != target_class:
        return 10.0  # cap; cannot flip within budget

    lo, hi = 0.0, 10.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if probe_argmax_at_scale(mid) == target_class:
            hi = mid
        else:
            lo = mid
    return min(hi * 1.1, 10.0)  # 10% safety margin


def run_with_intervention(model, prefix_acts, pos, flip_dir_unit, coeff, scale,
                          intervene_layer, capture_layer=None):
    """Apply intervention to prefix_acts (at pos) and forward to logits.

    intervene_layer here is the mingpt block whose resid_post we modify.
    prefix_acts was built via compute_prefix(..., intervene_layer + 1).
    If capture_layer is given, also returns resid_post of that block."""
    x = prefix_acts.clone()
    x[0, pos] = x[0, pos] - scale * coeff * flip_dir_unit
    if capture_layer is None:
        return forward_from_prefix(model, x, intervene_layer + 1)
    return forward_from_prefix_capturing(model, x, intervene_layer + 1,
                                         capture_layer)


# ---------------------------------------------------------------------------
# Probability-mass metrics
# ---------------------------------------------------------------------------
def mass_on(probs60, cell_set):
    """Sum probs over the cells in cell_set that are valid (i.e. in STOI)."""
    total = 0.0
    for c in cell_set:
        idx = STOI_INDEX_OF.get(c)
        if idx is not None:
            total += probs60[idx].item()
    return total


def li_topn_accuracy(cell_logits60, legal_set):
    """Li et al. top-N metric.

    Take the model's top-N predicted cells, where N = |legal_set ∩ STOI|.
    Count false positives (in top-N but not legal) + false negatives (in legal
    but not top-N). Return 1 - errors / (2 * N).

    Returns None if N == 0 (degenerate).
    """
    legal_stoi = set(legal_set) & set(STOI_INDICES)
    n = len(legal_stoi)
    if n == 0:
        return None
    topn_idx = cell_logits60.argsort(descending=True)[:n]
    topn_cells = {STOI_INDICES[int(i)] for i in topn_idx}
    fp = topn_cells - legal_stoi
    fn = legal_stoi - topn_cells
    errors = len(fp) + len(fn)
    return 1.0 - errors / (2.0 * n)


def measure_crosstalk_count(resid_clean, resid_intv, probe_full_mode, pos,
                            target_cell):
    """Number of non-target, non-center cells whose probe argmax flipped.

    resid_clean, resid_intv: (1, T, 512) — captured at the probe layer.
    probe_full_mode: (512, 8, 8, 3) — single-mode slice of the Nanda probe.
    target_cell: 0-63 board index of the intervened cell (excluded from count).
    """
    h_clean = resid_clean[0, pos]
    h_intv = resid_intv[0, pos]
    logits_clean = torch.einsum("d,drco->rco", h_clean, probe_full_mode)
    logits_intv = torch.einsum("d,drco->rco", h_intv, probe_full_mode)
    pred_clean = logits_clean.argmax(-1)  # (8, 8)
    pred_intv = logits_intv.argmax(-1)
    changed = (pred_clean != pred_intv)  # (8, 8) bool
    tr, tc = target_cell // 8, target_cell % 8
    count = 0
    for r in range(8):
        for c in range(8):
            if (r, c) == (tr, tc):
                continue
            cell = r * 8 + c
            if cell in CENTER_CELLS:
                continue
            if bool(changed[r, c]):
                count += 1
    return count


def compute_metrics(clean_logits_last, intv_logits_last, legal_orig, legal_cf):
    """Return the four mass-shift metrics + Li top-N for one intervention.

    legal_orig, legal_cf: sets of board-cell indices (0-63).
    """
    newly_legal = legal_cf - legal_orig
    cell_logits_before = clean_logits_last[1:61]  # 60 cell logits
    cell_logits_after = intv_logits_last[1:61]
    probs_before = torch.softmax(cell_logits_before, dim=0)
    probs_after = torch.softmax(cell_logits_after, dim=0)

    P_before_legal_cf = mass_on(probs_before, legal_cf)
    P_after_legal_cf = mass_on(probs_after, legal_cf)
    P_before_newly = mass_on(probs_before, newly_legal)
    P_after_newly = mass_on(probs_after, newly_legal)

    abs_dP_legal = P_after_legal_cf - P_before_legal_cf
    abs_dP_newly = P_after_newly - P_before_newly

    if P_after_legal_cf > 0:
        pct_dP_legal = abs_dP_legal / P_after_legal_cf
        pct_dP_newly = abs_dP_newly / P_after_legal_cf
    else:
        pct_dP_legal = float("nan")
        pct_dP_newly = float("nan")

    # Li et al. top-N: model's top-|cf_legal| predictions vs counterfactual.
    li_topn_before = li_topn_accuracy(cell_logits_before, legal_cf)
    li_topn_after = li_topn_accuracy(cell_logits_after, legal_cf)
    li_topn_shift = (
        li_topn_after - li_topn_before
        if li_topn_before is not None and li_topn_after is not None
        else None
    )

    return {
        "P_before_legal_cf": P_before_legal_cf,
        "P_after_legal_cf": P_after_legal_cf,
        "P_before_newly_legal": P_before_newly,
        "P_after_newly_legal": P_after_newly,
        "abs_dP_legal": abs_dP_legal,
        "pct_dP_legal": pct_dP_legal,
        "abs_dP_newly_legal": abs_dP_newly,
        "pct_dP_newly_legal": pct_dP_newly,
        "li_topn_before": li_topn_before,
        "li_topn_after": li_topn_after,
        "li_topn_shift": li_topn_shift,
        "n_legal_orig": None,  # filled in by caller
        "n_legal_cf": None,
        "n_newly_legal": len(newly_legal),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-games", type=int, default=100)
    p.add_argument("--pos-lo", type=int, default=10)
    p.add_argument("--pos-hi", type=int, default=50, help="exclusive")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    p.add_argument("--probe-path",
                   default="mechanistic_interpretability/main_linear_probe.pth")
    p.add_argument("--int-data",
                   default="mechanistic_interpretability/board_seqs_int_small.npy")
    p.add_argument("--string-data",
                   default="mechanistic_interpretability/board_seqs_string_small.npy")
    p.add_argument("--device", default=None,
                   help="cuda / mps / cpu; auto-detect if not set")
    p.add_argument("--output", default="logs/ogpt_legal_mass_shift.csv")
    p.add_argument("--max-tries-per-pos", type=int, default=200)
    p.add_argument("--max-extra-games", type=int, default=400,
                   help="Per category, the max games to try beyond --n-games "
                        "when a position fails to yield a valid intervention.")
    p.add_argument("--categories", default="remove,add,flip",
                   help="Comma-separated subset of: remove, add, flip, any.")
    p.add_argument("--cal-depth", type=int, default=0,
                   help="Calibration depth. 0 = intervene+decode at probe "
                        "layer (no propagation). N>0 = intervene N blocks "
                        "upstream and propagate through N blocks to the "
                        "probe layer.")
    p.add_argument("--scale-multipliers", default="1",
                   help="Comma-separated list of K values. Each intervention "
                        "is run at alpha = K * alpha_cd2 for every K. Output "
                        "has one row per (intervention, K). Default '1'.")
    p.add_argument("--probe-kind", default="mode0_mineyours",
                   choices=list(PROBE_KIND_TO_MODE.keys()),
                   help="Probe convention. mode0_mineyours = correct (~99%% "
                        "probe accuracy). mode2_abs = the convention used by "
                        "sweep_intervention_alpha.py / ogpt_intervention.py / "
                        "compare_interventions.py (~77%% probe accuracy).")
    args = p.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    print("Loading model + probe + data...")
    model = load_model(args.ckpt, device)
    probe = load_probe(args.probe_path, device)  # (3, 512, 8, 8, 3)
    board_seqs_int, board_seqs_string = load_games(args.int_data, args.string_data)
    n_total = len(board_seqs_int)
    print(f"Loaded {n_total} games.")

    rng = random.Random(args.seed)
    scale_multipliers = [float(s) for s in args.scale_multipliers.split(",")]
    print(f"Scale multipliers K: {scale_multipliers}")
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    print(f"Categories: {categories}  (target {args.n_games} interventions each)")
    INTERVENE_LAYER = intervene_layer_for(args.cal_depth, PROBE_LAYER)
    if INTERVENE_LAYER < 0:
        raise SystemExit(
            f"cal_depth {args.cal_depth} > probe_layer {PROBE_LAYER}; invalid."
        )
    print(f"Calibration: cd={args.cal_depth}  "
          f"(intervene at mingpt block {INTERVENE_LAYER}, "
          f"probe at block {PROBE_LAYER})")

    rows = []
    cat_skipped = {cat: 0 for cat in categories}

    # Sample a per-category pool of candidate game indices (with extras to
    # absorb skips). Disjoint across categories.
    all_pool = rng.sample(range(n_total),
                          min(n_total,
                              len(categories) * (args.n_games + args.max_extra_games)))
    pool_per_cat = {}
    cursor = 0
    for cat in categories:
        size = args.n_games + args.max_extra_games
        pool_per_cat[cat] = all_pool[cursor:cursor + size]
        cursor += size

    for cat in categories:
        kept = 0
        pbar = tqdm(pool_per_cat[cat], desc=f"{cat:>6}")
        for gi in pbar:
            if kept >= args.n_games:
                break
            game_int = torch.from_numpy(
                board_seqs_int[gi].astype(np.int64)
            ).to(device)
            game_string = board_seqs_string[gi]
            pos = rng.randint(args.pos_lo, args.pos_hi - 1)
            try:
                board_state, color = replay(game_string, pos)
            except Exception:
                cat_skipped[cat] += 1
                continue

            intv = sample_intervention(board_state, color, rng,
                                       category=cat,
                                       max_tries=args.max_tries_per_pos)
            if intv is None:
                cat_skipped[cat] += 1
                continue
            cell, orig_val, target_val, legal_orig, legal_cf = intv
            r, c = cell // 8, cell % 8

            input_tokens = game_int[: pos + 1].unsqueeze(0)

            with torch.no_grad():
                prefix_acts = compute_prefix(model, input_tokens,
                                              INTERVENE_LAYER + 1)
                clean_x = prefix_acts.clone()
                clean_logits, clean_resid_at_probe = \
                    forward_from_prefix_capturing(
                        model, clean_x, INTERVENE_LAYER + 1, PROBE_LAYER,
                    )
                clean_last = clean_logits[0, -1]

                probe_mode = PROBE_KIND_TO_MODE[args.probe_kind]
                probe_cell_W = probe[probe_mode, :, r, c, :].detach()
                probe_full_mode = probe[probe_mode].detach()

                current_class = board_val_to_probe_class(
                    orig_val, color, args.probe_kind)
                target_class = board_val_to_probe_class(
                    target_val, color, args.probe_kind)

                scale_min = cdN_calibrate(
                    model, prefix_acts, pos, probe_cell_W,
                    current_class, target_class,
                    intervene_layer=INTERVENE_LAYER,
                    probe_layer=PROBE_LAYER,
                )
                flip_dir = (probe_cell_W[:, target_class]
                            - probe_cell_W[:, current_class])
                d_hat = flip_dir / flip_dir.norm()
                coeff = (prefix_acts[0, pos] @ d_hat).item()

                for K in scale_multipliers:
                    alpha = K * scale_min
                    intv_logits, intv_resid_at_probe = run_with_intervention(
                        model, prefix_acts, pos, d_hat, coeff, alpha,
                        INTERVENE_LAYER, capture_layer=PROBE_LAYER,
                    )
                    intv_last = intv_logits[0, -1]
                    m = compute_metrics(clean_last.cpu(), intv_last.cpu(),
                                        legal_orig, legal_cf)
                    m["n_legal_orig"] = len(legal_orig)
                    m["n_legal_cf"] = len(legal_cf)
                    crosstalk = measure_crosstalk_count(
                        clean_resid_at_probe, intv_resid_at_probe,
                        probe_full_mode, pos, cell,
                    )
                    rows.append({
                        "category": cat,
                        "cal_depth": int(args.cal_depth),
                        "game_idx": int(gi),
                        "position": int(pos),
                        "cell": int(cell),
                        "square": cell_label(cell),
                        "intervention": intervention_label(orig_val, target_val),
                        "K": float(K),
                        "scale": float(scale_min),
                        "alpha": float(alpha),
                        "n_legal_orig": m["n_legal_orig"],
                        "n_legal_cf": m["n_legal_cf"],
                        "n_newly_legal": m["n_newly_legal"],
                        "P_before_legal_cf": m["P_before_legal_cf"],
                        "P_after_legal_cf": m["P_after_legal_cf"],
                        "P_before_newly_legal": m["P_before_newly_legal"],
                        "P_after_newly_legal": m["P_after_newly_legal"],
                        "abs_dP_legal": m["abs_dP_legal"],
                        "pct_dP_legal": m["pct_dP_legal"],
                        "abs_dP_newly_legal": m["abs_dP_newly_legal"],
                        "pct_dP_newly_legal": m["pct_dP_newly_legal"],
                        "li_topn_before": m["li_topn_before"],
                        "li_topn_after": m["li_topn_after"],
                        "li_topn_shift": m["li_topn_shift"],
                        "crosstalk_cells": int(crosstalk),
                    })
            kept += 1

    print(f"\nDone. {len(rows)} interventions kept; skipped per category: "
          f"{cat_skipped}")

    # Sort rows: category in the order requested, then game_idx, position.
    cat_order = {c: i for i, c in enumerate(categories)}
    rows.sort(key=lambda r: (cat_order[r["category"]], r["K"],
                              r["game_idx"], r["position"]))

    # Write CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fieldnames = [
        "category", "cal_depth",
        "game_idx", "position", "cell", "square", "intervention",
        "K", "scale", "alpha",
        "n_legal_orig", "n_legal_cf", "n_newly_legal",
        "P_before_legal_cf", "P_after_legal_cf",
        "P_before_newly_legal", "P_after_newly_legal",
        "abs_dP_legal", "pct_dP_legal",
        "abs_dP_newly_legal", "pct_dP_newly_legal",
        "li_topn_before", "li_topn_after", "li_topn_shift",
        "crosstalk_cells",
    ]
    float_cols = {
        "K", "scale", "alpha",
        "P_before_legal_cf", "P_after_legal_cf",
        "P_before_newly_legal", "P_after_newly_legal",
        "abs_dP_legal", "pct_dP_legal",
        "abs_dP_newly_legal", "pct_dP_newly_legal",
        "li_topn_before", "li_topn_after", "li_topn_shift",
    }
    import csv
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            formatted = {
                k: (f"{v:.2f}" if k in float_cols and v is not None else v)
                for k, v in row.items()
            }
            w.writerow(formatted)
    print(f"Wrote {args.output}")

    # Pretty stdout summary — aggregate by category (and by K if >1 K)
    if rows:
        print()
        print(f"{'cat':>6} {'K':>4} {'n':>4} "
              f"{'mean_dPL':>10} {'mean_dPNL':>11} "
              f"{'liN_before':>11} {'liN_after':>10} {'liN_shift':>10} "
              f"{'xtalk':>7} {'scale':>10}")
        print("-" * 90)
        for cat in categories:
            for K in scale_multipliers:
                sub = [r for r in rows if r["category"] == cat and r["K"] == K]
                if not sub:
                    continue
                def arr(k):
                    return np.array([
                        r[k] for r in sub
                        if r[k] is not None and not (
                            isinstance(r[k], float) and np.isnan(r[k])
                        )
                    ])
                adL = arr("abs_dP_legal")
                adNL = arr("abs_dP_newly_legal")
                lib = arr("li_topn_before")
                lia = arr("li_topn_after")
                lis = arr("li_topn_shift")
                xtalk = arr("crosstalk_cells")
                scale = arr("scale")
                print(
                    f"{cat:>6} {K:>4.1f} {len(sub):>4} "
                    f"{adL.mean():>+10.4f} {adNL.mean():>+11.4f} "
                    f"{lib.mean():>11.4f} {lia.mean():>10.4f} "
                    f"{lis.mean():>+10.4f} {xtalk.mean():>7.2f} "
                    f"{scale.mean():>10.2f}"
                )


if __name__ == "__main__":
    main()
