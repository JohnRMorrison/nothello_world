"""Nanda-style interventions on pattern detector MLPs.

Modifies the H-dimensional hidden layer along probe directions to
"add" or "flip" pieces, then measures whether the model's pattern
predictions (and thus legal move predictions) change correctly.

Uses the same metrics as multi_intervention.py for direct comparison
with OthelloGPT interventions.

Usage:
    python intervene_pattern_models.py \
        --model-ckpt pattern_simple_direct_H512.pt \
        --probe-ckpt probe_direct_H512.pt \
        --mode direct --hidden 512 --n-games 200
"""
import sys, os, json, random
sys.path.insert(0, '.')

import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    get_device, N_MOVES, OPTIONS,
)
from hand_crafted_flanking import (
    enumerate_flanking_patterns, MOVE_TO_IDX, VALID_MOVES, CENTER_CELLS,
)
from generate_rule_games import precompute_pattern_arrays
from data.othello import OthelloBoardState
from train_pattern_simple import DirectMLP, EndToEndMLP, TwoStageMLP


# ---------------------------------------------------------------------------
# Board utilities
# ---------------------------------------------------------------------------

def replay_game(game_moves, pos):
    """Replay game to position pos, return (board_8x8, next_color)."""
    board = OthelloBoardState()
    for i in range(pos + 1):
        board.update([game_moves[i]])
    return np.copy(board.state), board.next_hand_color


def compute_legal_moves(board_2d, color):
    """Compute legal moves from 8×8 board state."""
    board = OthelloBoardState()
    board.state = board_2d.copy()
    board.next_hand_color = color
    valid = board.get_valid_moves()
    return set(valid) if valid else set()


def compute_counterfactual_legal(board_2d, modifications, color):
    """Apply modifications to board, compute legal moves."""
    modified = board_2d.copy()
    for (r, c, new_val) in modifications:
        modified[r, c] = new_val
    return compute_legal_moves(modified, color)


def board_val_to_probe_class(val):
    """Map board value (-1, 0, 1) to probe class (0=empty, 1=white, 2=black)."""
    if val == 0: return 0
    if val == -1: return 1
    return 2  # val == 1


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

# STOI mapping for features
_VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})
_MOVE_TO_IDX = {m: i for i, m in enumerate(_VALID_MOVES)}


def build_when_features(game_moves, pos):
    """Build 60-d 'when' feature vector at position pos."""
    when = np.zeros(N_MOVES, dtype=np.float32)
    for s in range(pos + 1):
        move = game_moves[s]
        if move in _MOVE_TO_IDX:
            when[_MOVE_TO_IDX[move]] = (s + 1) / 60.0
    return when


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------

def compute_flip_direction(probe_weights, cell_idx, current_class, target_class):
    """Compute the probe direction to flip a cell's class.

    probe_weights: (64*3, H) from nn.Linear(H, 64*3).weight
    cell_idx: 0-63 board position
    current_class, target_class: 0=empty, 1=white, 2=black

    Returns: (H,) direction vector
    """
    # probe maps H-d → 64*3, so weight is (64*3, H)
    # For cell c, class k: weight row is c*3 + k
    target_row = cell_idx * OPTIONS + target_class
    current_row = cell_idx * OPTIONS + current_class
    flip_dir = probe_weights[target_row] - probe_weights[current_row]
    return flip_dir


def apply_intervention(h, flip_dir, scale):
    """Modify hidden activation h along flip_dir.

    h: (H,) hidden activation
    flip_dir: (H,) direction
    scale: intervention strength

    Returns: modified h (new tensor, doesn't modify in-place)
    """
    d_hat = flip_dir / flip_dir.norm()
    coeff = h @ d_hat
    return h - scale * coeff * d_hat


def calibrate_scale(h, probe_weights, cell_idx, current_class, target_class):
    """Binary search for minimal scale that flips probe's prediction."""
    flip_dir = compute_flip_direction(probe_weights, cell_idx, current_class, target_class)
    d_hat = flip_dir / flip_dir.norm()
    coeff = h @ d_hat

    W = probe_weights.view(64, OPTIONS, -1)  # (64, 3, H)

    def probe_pred_at_scale(s):
        h_mod = h - s * coeff * d_hat
        logits = W[cell_idx] @ h_mod  # (3,)
        return logits.argmax().item()

    lo, hi = 0.0, 10.0
    if probe_pred_at_scale(hi) != target_class:
        return hi
    if probe_pred_at_scale(0.0) == target_class:
        return 0.5

    for _ in range(30):
        mid = (lo + hi) / 2
        if probe_pred_at_scale(mid) == target_class:
            hi = mid
        else:
            lo = mid

    return min(hi * 1.1, 10.0)


# ---------------------------------------------------------------------------
# Metrics (mirrors multi_intervention.py)
# ---------------------------------------------------------------------------

def patterns_to_legal_set(pattern_probs, patterns, threshold=0.5):
    """Convert 960 pattern probabilities to set of legal cell positions."""
    legal = set()
    for j, pat in enumerate(patterns):
        if pattern_probs[j] > threshold:
            legal.add(pat['target'])
    return legal


def measure_metrics(orig_pat_probs, intv_pat_probs, original_legal,
                    counterfactual_legal, patterns):
    """Compute intervention metrics from pattern probabilities."""
    newly_legal = counterfactual_legal - original_legal
    newly_illegal = original_legal - counterfactual_legal

    # Convert to per-cell logits (max pattern prob per cell)
    orig_cell = np.full(64, -10.0)
    intv_cell = np.full(64, -10.0)
    for j, pat in enumerate(patterns):
        t = pat['target']
        orig_cell[t] = max(orig_cell[t], orig_pat_probs[j])
        intv_cell[t] = max(intv_cell[t], intv_pat_probs[j])

    result = {}

    # Legal probability mass (on counterfactual-legal cells)
    # Use softmax over valid cells for probability
    valid_orig = np.array([orig_cell[c] for c in VALID_MOVES])
    valid_intv = np.array([intv_cell[c] for c in VALID_MOVES])
    intv_probs = np.exp(valid_intv) / np.exp(valid_intv).sum()
    legal_mask = np.array([1.0 if VALID_MOVES[i] in counterfactual_legal else 0.0
                           for i in range(len(VALID_MOVES))])
    result["legal_prob_mass"] = float((intv_probs * legal_mask).sum())

    # Boundary margin: mean logit difference (legal - illegal) after intervention
    cf_legal_logits = [intv_cell[c] for c in counterfactual_legal if c in set(VALID_MOVES)]
    cf_illegal_logits = [intv_cell[c] for c in VALID_MOVES if c not in counterfactual_legal]
    if cf_legal_logits and cf_illegal_logits:
        result["boundary_margin"] = float(np.mean(cf_legal_logits) - np.mean(cf_illegal_logits))
    else:
        result["boundary_margin"] = None

    # Logprob shifts for newly legal/illegal
    orig_lp = np.log(np.exp(valid_orig) / np.exp(valid_orig).sum() + 1e-10)
    intv_lp = np.log(intv_probs + 1e-10)

    shifts_legal = []
    shifts_illegal = []
    for i, c in enumerate(VALID_MOVES):
        if c in newly_legal:
            shifts_legal.append(intv_lp[i] - orig_lp[i])
        if c in newly_illegal:
            shifts_illegal.append(intv_lp[i] - orig_lp[i])

    result["mean_logprob_shift_legal"] = float(np.mean(shifts_legal)) if shifts_legal else None
    result["mean_logprob_shift_illegal"] = float(np.mean(shifts_illegal)) if shifts_illegal else None

    # Pattern-level: how many patterns changed correctly?
    orig_legal_set = patterns_to_legal_set(orig_pat_probs, patterns)
    intv_legal_set = patterns_to_legal_set(intv_pat_probs, patterns)
    result["n_original_legal"] = len(original_legal)
    result["n_cf_legal"] = len(counterfactual_legal)
    result["n_pred_legal_orig"] = len(orig_legal_set)
    result["n_pred_legal_intv"] = len(intv_legal_set)

    # Li et al. top-N accuracy (original): ranking-based
    N = len(counterfactual_legal)
    if N > 0:
        top_n_cells = set()
        cell_scores = [(intv_cell[c], c) for c in VALID_MOVES]
        cell_scores.sort(reverse=True)
        for _, c in cell_scores[:N]:
            top_n_cells.add(c)
        li_correct = len(top_n_cells & counterfactual_legal)
        result["li_topn_accuracy"] = li_correct / N
    else:
        result["li_topn_accuracy"] = None

    # Adapted legal-set accuracy: threshold patterns at 0.5, aggregate to
    # legal moves (cell is legal if any pattern fires), compare predicted
    # legal set to counterfactual legal set.
    pred_legal_set = patterns_to_legal_set(intv_pat_probs, patterns, threshold=0.5)
    pred_legal_valid = pred_legal_set & set(VALID_MOVES)
    cf_legal_valid = counterfactual_legal & set(VALID_MOVES)
    if cf_legal_valid:
        # Precision: of predicted legal, how many are actually legal?
        legal_precision = (len(pred_legal_valid & cf_legal_valid) /
                           len(pred_legal_valid)) if pred_legal_valid else 0.0
        # Recall: of actually legal, how many did we predict?
        legal_recall = len(pred_legal_valid & cf_legal_valid) / len(cf_legal_valid)
        # Exact match: predicted set == actual set?
        legal_exact = 1.0 if pred_legal_valid == cf_legal_valid else 0.0
        # Error count (Li-style): FP + FN
        fp = len(pred_legal_valid - cf_legal_valid)
        fn = len(cf_legal_valid - pred_legal_valid)
        legal_errors = fp + fn
        legal_adapted_acc = 1.0 - legal_errors / (2 * len(cf_legal_valid))
    else:
        legal_precision = legal_recall = legal_exact = legal_adapted_acc = None
        legal_errors = None

    result["legal_precision"] = legal_precision
    result["legal_recall"] = legal_recall
    result["legal_exact_match"] = legal_exact
    result["legal_errors"] = legal_errors
    result["legal_adapted_acc"] = legal_adapted_acc

    return result


def measure_probe_accuracy(h_modified, probe_weights, modifications):
    """Check if probe reads the intended state for modified cells."""
    W = probe_weights.view(64, OPTIONS, -1)
    correct = 0
    for (r, c, orig_val, target_val) in modifications:
        cell_idx = r * 8 + c
        target_class = board_val_to_probe_class(target_val)
        logits = W[cell_idx] @ h_modified
        if logits.argmax().item() == target_class:
            correct += 1
    return correct / len(modifications) if modifications else 0.0


def measure_probe_crosstalk(h_orig, h_modified, probe_weights, modifications):
    """Mean absolute probe logit change for non-modified cells."""
    W = probe_weights.view(64, OPTIONS, -1)  # (64, 3, H)
    modified_cells = {r * 8 + c for (r, c, _, _) in modifications}
    changes = []
    for cell_idx in range(64):
        if cell_idx in modified_cells:
            continue
        orig_logits = W[cell_idx] @ h_orig
        mod_logits = W[cell_idx] @ h_modified
        changes.append((mod_logits - orig_logits).abs().mean().item())
    return np.mean(changes) if changes else 0.0


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def select_add_modifications(board_state, n, rng):
    """Select n empty cells to add pieces (non-interacting)."""
    empty = [(r, c) for r in range(8) for c in range(8)
             if board_state[r, c] == 0 and (r * 8 + c) not in CENTER_CELLS]
    if len(empty) < n:
        return None
    rng.shuffle(empty)
    mods = []
    for r, c in empty[:n]:
        color = rng.choice([1, -1])
        mods.append((r, c, 0, color))
    return mods


def select_flip_modifications(board_state, n, rng):
    """Select n occupied cells to flip color."""
    occupied = [(r, c) for r in range(8) for c in range(8)
                if board_state[r, c] != 0 and (r * 8 + c) not in CENTER_CELLS]
    if len(occupied) < n:
        return None
    rng.shuffle(occupied)
    mods = []
    for r, c in occupied[:n]:
        orig = int(board_state[r, c])
        mods.append((r, c, orig, -orig))
    return mods


def run_experiment(model_even, model_odd, probe_even, probe_odd,
                   mode, patterns, pat_targets,
                   games, device, n_games=200, seed=42):
    """Run intervention experiment across games."""
    rng = random.Random(seed)
    POS_RANGE = (10, 50)

    conditions = [
        ("add_1", select_add_modifications, 1),
        ("add_2", select_add_modifications, 2),
        ("flip_1", select_flip_modifications, 1),
        ("flip_2", select_flip_modifications, 2),
    ]

    results = {cond_name: [] for cond_name, _, _ in conditions}

    for gi in tqdm(range(n_games), desc="Games"):
        game_moves = games[gi]
        pos = rng.randint(POS_RANGE[0], min(POS_RANGE[1], len(game_moves) - 2))

        board_state, color = replay_game(game_moves, pos)
        original_legal = compute_legal_moves(board_state, color)
        is_even = (pos % 2 == 0)

        # Build features
        features = build_when_features(game_moves, pos)
        x = torch.tensor(features, dtype=torch.float32).to(device)

        # Select model and probe based on parity
        model = model_even if is_even else model_odd
        probe_w = probe_even if is_even else probe_odd  # (64*3, H)

        # Original forward pass: get hidden and output
        with torch.no_grad():
            if mode == "direct":
                h_orig = torch.relu(model.net[0](x))
                orig_logits = model.net[2](h_orig)
            else:
                h_orig = torch.relu(model.backbone[0](x))
                # For non-direct, we'd need to go through the full forward
                # For now, just use backbone's second layer as logits proxy
                # This is approximate for e2e/emergent but exact for direct
                orig_logits = model.backbone[2](h_orig)

        orig_pat_probs = torch.sigmoid(orig_logits).cpu().numpy()

        for cond_name, select_fn, n_mods in conditions:
            mods = select_fn(board_state, n_mods, rng)
            if mods is None:
                continue

            cf_mods = [(r, c, tgt) for (r, c, _, tgt) in mods]
            cf_legal = compute_counterfactual_legal(board_state, cf_mods, color)

            # Apply interventions to hidden layer
            h_modified = h_orig.clone()
            for (r, c, orig_val, target_val) in mods:
                cell_idx = r * 8 + c
                current_class = board_val_to_probe_class(orig_val)
                target_class = board_val_to_probe_class(target_val)

                scale = calibrate_scale(
                    h_modified, probe_w, cell_idx, current_class, target_class)
                flip_dir = compute_flip_direction(
                    probe_w, cell_idx, current_class, target_class)
                h_modified = apply_intervention(h_modified, flip_dir, scale)

            # Forward modified hidden through rest of model
            with torch.no_grad():
                if mode == "direct":
                    intv_logits = model.net[2](h_modified)
                else:
                    intv_logits = model.backbone[2](h_modified)

            intv_pat_probs = torch.sigmoid(intv_logits).cpu().numpy()

            # Metrics
            metrics = measure_metrics(
                orig_pat_probs, intv_pat_probs,
                original_legal, cf_legal, patterns)
            probe_acc = measure_probe_accuracy(h_modified, probe_w, mods)
            crosstalk = measure_probe_crosstalk(h_orig, h_modified, probe_w, mods)

            sample = {
                "game_idx": gi,
                "pos": pos,
                "color": color,
                "modifications": [(r, c, int(o), int(t)) for r, c, o, t in mods],
                **metrics,
                "probe_acc": probe_acc,
                "crosstalk": crosstalk,
            }
            results[cond_name].append(sample)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_results(results):
    """Print summary table."""
    print(f"\n{'='*80}")
    print("INTERVENTION RESULTS")
    print(f"{'='*80}")

    header = (f"{'Condition':<12} {'N':>4} {'AdaptAcc':>8} {'Exact':>7} "
              f"{'Prec':>7} {'Recall':>7} {'ProbeAcc':>8} {'Xtalk':>7} "
              f"{'Li TopN':>7}")
    print(header)
    print("-" * len(header))

    for cond_name, samples in sorted(results.items()):
        if not samples:
            continue

        def mean_or(key):
            vals = [s[key] for s in samples if s.get(key) is not None]
            return np.mean(vals) if vals else None

        aa = mean_or("legal_adapted_acc")
        ex = mean_or("legal_exact_match")
        pr = mean_or("legal_precision")
        rc = mean_or("legal_recall")
        pa = mean_or("probe_acc")
        xt = mean_or("crosstalk")
        li = mean_or("li_topn_accuracy")

        def fmt(v):
            return f"{v:.4f}" if v is not None else "   N/A"

        print(f"{cond_name:<12} {len(samples):>4} {fmt(aa):>8} {fmt(ex):>7} "
              f"{fmt(pr):>7} {fmt(rc):>7} {fmt(pa):>8} {fmt(xt):>7} "
              f"{fmt(li):>7}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--probe-ckpt", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["direct", "emergent", "e2e", "two-stage"])
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiments/pattern_interventions")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Mode: {args.mode}, H={args.hidden}, {args.n_games} games")

    # Load model
    model_ckpt = torch.load(args.model_ckpt, map_location=device)
    n_patterns = model_ckpt.get('n_patterns', 960)
    input_dim = N_MOVES

    if args.mode == "direct":
        model_even = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
        model_odd = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    elif args.mode == "two-stage":
        model_even = TwoStageMLP(input_dim, args.hidden, n_patterns).to(device)
        model_odd = TwoStageMLP(input_dim, args.hidden, n_patterns).to(device)
    else:
        model_even = EndToEndMLP(input_dim, args.hidden, n_patterns).to(device)
        model_odd = EndToEndMLP(input_dim, args.hidden, n_patterns).to(device)

    model_even.load_state_dict(model_ckpt['even'])
    model_odd.load_state_dict(model_ckpt['odd'])
    model_even.eval(); model_odd.eval()
    print(f"  Model loaded (pat_acc={model_ckpt.get('best_pat_acc', '?')})")

    # Load probe
    probe_ckpt = torch.load(args.probe_ckpt, map_location=device)
    # Probe is nn.Linear(H, 64*3) — we need the weight matrix (64*3, H)
    probe_even = probe_ckpt['even']['weight'].to(device)  # (64*3, H)
    probe_odd = probe_ckpt['odd']['weight'].to(device)
    print(f"  Probe loaded (acc={probe_ckpt.get('best_acc', '?')})")

    # Load games
    import pickle
    game_dir = "data/othello_synthetic"
    game_files = sorted(f for f in os.listdir(game_dir) if f.endswith(".pickle"))
    print(f"Loading games from {game_files[0]}...")
    with open(os.path.join(game_dir, game_files[0]), "rb") as f:
        games_raw = pickle.load(f)
    games = [g for g in games_raw if len(g) == 60]
    print(f"  {len(games)} games loaded")

    # Patterns
    patterns = enumerate_flanking_patterns()
    pat_targets, _, _, _ = precompute_pattern_arrays(patterns)

    # Run experiment
    results = run_experiment(
        model_even, model_odd, probe_even, probe_odd,
        args.mode, patterns, pat_targets,
        games, device, n_games=args.n_games, seed=args.seed)

    # Report
    report_results(results)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir,
                            f"intervention_{args.mode}_H{args.hidden}.json")
    serializable = {}
    for cond, samples in results.items():
        serializable[cond] = [
            {k: (float(v) if isinstance(v, (np.floating, float)) else v)
             for k, v in s.items()}
            for s in samples
        ]
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {out_path}")
