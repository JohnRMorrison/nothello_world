"""Compare v4's top-1 legal-move predictions against the MLP pattern detector
(H=8192) and simple rule-based heuristics on the same positions.

If v4 and MLP agree on >95% of positions, they're likely using similar
features (flanking-pattern detection from played+even).  If they disagree
often, v4 is using a different mechanism.

Heuristics tested:
  - random: random unplayed cell
  - adjacent: random cell adjacent to any played cell
  - parity_line: for each candidate, count directions where parity sequence
                 looks like a flanking pattern (opposite parities then same).
                 Approximates Othello's flanking rule using parity as a
                 proxy for piece color.
  - mlp: the user's MLP pattern detector (gold-standard 99% baseline)
  - v4: the trained transformer

For each position we record what each predictor would pick, and whether
that pick is actually legal in the true board state.

Usage:
    python compare_v4_vs_mlp.py \\
        --v4-ckpt ckpts/gpt_shuffled_v4_<ts>.ckpt \\
        --mlp-ckpt experiments/.../pattern_simple_direct_H8192.pt \\
        --num-games 200
"""
import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, '.')
from data.othello import OthelloBoardState


# 60-cell index <-> 64-cell index mappings (excluding 4 center cells)
CENTER_64 = {27, 28, 35, 36}
MOVABLE_64 = [c for c in range(64) if c not in CENTER_64]
C64_TO_C60 = {c: i for i, c in enumerate(MOVABLE_64)}
C60_TO_C64 = {i: c for i, c in enumerate(MOVABLE_64)}


def load_val_games(data_dir='./data/othello_synthetic', num_files=1):
    files = sorted(os.listdir(data_dir))
    games = []
    for fname in files[-num_files:]:
        with open(os.path.join(data_dir, fname), 'rb') as f:
            batch = pickle.load(f)
        if len(batch) >= 9e4:
            games.extend(batch)
    return games


# ---------- Heuristic predictors ----------

def heuristic_random_unplayed(played_set_64):
    """Pick a random unplayed cell."""
    avail = [c for c in MOVABLE_64 if c not in played_set_64]
    return random.choice(avail) if avail else None


def heuristic_adjacent_to_played(played_set_64):
    """Pick a random unplayed cell adjacent to any played cell."""
    DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    candidates = []
    for c in MOVABLE_64:
        if c in played_set_64:
            continue
        r, col = c // 8, c % 8
        for dr, dc in DIRS:
            nr, ncol = r + dr, col + dc
            if 0 <= nr < 8 and 0 <= ncol < 8:
                n = nr * 8 + ncol
                if n in played_set_64:
                    candidates.append(c)
                    break
    return random.choice(candidates) if candidates else None


def heuristic_parity_line(played_parity_map):
    """For each candidate cell, score = number of directions where the
    parity sequence resembles a flanking pattern (one or more cells of one
    parity, then a terminator of the OTHER parity).  This treats parity
    as a proxy for current piece color — wrong when captures flipped pieces,
    but a reasonable rule-based approximation.

    played_parity_map: dict cell_64 -> parity (0 or 1)
    Returns the cell with the highest score.
    """
    DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    scores = {}
    for c in MOVABLE_64:
        if c in played_parity_map:
            continue
        # Predict for the next move's parity (we don't know it, so try
        # both and take the max).
        c_score = 0
        for next_parity in (0, 1):
            for dr, dc in DIRS:
                r, col = c // 8, c % 8
                r += dr; col += dc
                seen_opposite = 0
                while 0 <= r < 8 and 0 <= col < 8:
                    n = r * 8 + col
                    if n not in played_parity_map:
                        break
                    parity = played_parity_map[n]
                    if parity != next_parity:
                        seen_opposite += 1
                    else:
                        if seen_opposite >= 1:
                            c_score += 1
                        break
                    r += dr; col += dc
        scores[c] = c_score
    if not scores:
        return None
    return max(scores, key=scores.get)


# ---------- Inference adapters ----------

def predict_v4(model, game, k, cell_stoi, device):
    """Return predicted cell (0..59 cell-stoi space) for next move at depth k."""
    cell_scores = v4_cell_scores(model, game, k, cell_stoi, device)
    return int(np.argmax(cell_scores))


def v4_cell_scores(model, game, k, cell_stoi, device):
    """Return per-cell scores from v4 (60-d numpy).  Aggregates the 2-parity
    token vocab to per-cell scores via max over the two parities."""
    from train_gpt_shuffled_v4 import CellIndexedMaskedDataset
    ds = CellIndexedMaskedDataset([list(game)], cell_stoi=cell_stoi)
    Lc = ds.context_len
    x, _, mask = ds[0]
    x = x.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x, attn_mask=mask)
    qpos = Lc + k
    vec = logits[0, qpos].cpu().numpy()   # shape (vocab,)
    # tokens 1..120 are (cell, parity); per cell, take max over its 2 parity copies
    scores = np.full(60, -np.inf)
    for tok in range(1, 121):
        cell_idx = (tok - 1) % 60
        if vec[tok] > scores[cell_idx]:
            scores[cell_idx] = vec[tok]
    return scores


def load_mlp(ckpt_path, hidden, device):
    """Load a played+even pattern-detector MLP (returns (me, mo, idx, mask))."""
    sys.path.insert(0, '.')
    from train_pattern_simple import DirectMLP, _get_cell_pat_index
    from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX

    ckpt = torch.load(ckpt_path, map_location=device)
    n_patterns = ckpt.get('n_patterns', 960)
    input_dim = ckpt.get('input_dim', 120)
    me = DirectMLP(input_dim, hidden, n_patterns).to(device)
    mo = DirectMLP(input_dim, hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even']); me.eval()
    mo.load_state_dict(ckpt['odd']); mo.eval()

    # Build pattern->cell mapping (each pattern targets a specific cell)
    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device)
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)
    return me, mo, idx, mask


def played_even_features(game_prefix):
    """Build 120-d binary feature vector: 60 played + 60 played-at-parity-0."""
    feat = torch.zeros(120)
    for i, c in enumerate(game_prefix):
        if c not in C64_TO_C60:
            continue
        c60 = C64_TO_C60[c]
        feat[c60] = 1.0
        if i % 2 == 0:  # parity 0 (move index 0 = move M_1, etc.)
            feat[60 + c60] = 1.0
    return feat


def predict_mlp(mlp_bundle, game, k, device):
    """Top-1 cell prediction from a played+even MLP (returns 64-cell idx)."""
    scores = mlp_cell_scores(mlp_bundle, game, k, device)
    return C60_TO_C64.get(int(np.argmax(scores)))


def mlp_cell_scores(mlp_bundle, game, k, device):
    """Per-cell scores from MLP via prob_or aggregator.  Returns 60-d numpy."""
    me, mo, idx, mask = mlp_bundle
    features = played_even_features(game[:k]).unsqueeze(0).to(device)
    use_even = (k % 2 == 1)
    model = me if use_even else mo
    with torch.no_grad():
        logits = model(features)
    log1m = -torch.nn.functional.softplus(logits)
    gathered = log1m[:, idx]
    gathered = gathered.masked_fill(~mask, 0.0)
    cell_scores = -gathered.sum(dim=-1)[0]
    return cell_scores.cpu().numpy()


# ---------- Main eval loop ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v4-ckpt', required=True)
    ap.add_argument('--mlp-ckpt-512', default=None,
                    help='Path to pattern_simple_direct_H512_playedeven.pt')
    ap.add_argument('--mlp-ckpt-8192', default=None,
                    help='Path to pattern_simple_direct_H8192_playedeven.pt')
    ap.add_argument('--num-games', type=int, default=200)
    ap.add_argument('--pos-start', type=int, default=10)
    ap.add_argument('--pos-end', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading val games...")
    games = load_val_games()[:5000]
    games = [g for g in games if len(g) >= args.pos_end]
    games = random.sample(games, min(args.num_games, len(games)))
    print(f"  using {len(games)} games\n")

    # Build cell_stoi from games
    cells = set()
    for g in games:
        cells.update(g)
    cell_stoi = {c: i for i, c in enumerate(sorted(cells))}

    # Load v4
    print(f"Loading v4 from {args.v4_ckpt}...")
    from train_gpt_shuffled_v4 import CellIndexedMaskedDataset
    from mingpt.model import GPT, GPTConfig
    config = GPTConfig(CellIndexedMaskedDataset.VOCAB_SIZE,
                       CellIndexedMaskedDataset(games[:1]).block_size,
                       n_layer=8, n_head=8, n_embd=512)
    v4 = GPT(config)
    v4.load_state_dict(torch.load(args.v4_ckpt, map_location='cpu'))
    v4 = v4.to(device).eval()

    # Load MLPs (optional)
    mlp_512 = mlp_8192 = None
    if args.mlp_ckpt_512:
        print(f"Loading MLP H=512 from {args.mlp_ckpt_512}...")
        mlp_512 = load_mlp(args.mlp_ckpt_512, hidden=512, device=device)
    if args.mlp_ckpt_8192:
        print(f"Loading MLP H=8192 from {args.mlp_ckpt_8192}...")
        mlp_8192 = load_mlp(args.mlp_ckpt_8192, hidden=8192, device=device)

    # Stats per heuristic
    predictor_names = ['random_unplayed', 'adjacent', 'parity_line']
    if mlp_512 is not None: predictor_names.append('mlp_h512')
    if mlp_8192 is not None: predictor_names.append('mlp_h8192')
    predictor_names.append('v4')
    stats = {name: {'n': 0, 'legal': 0, 'agree_v4': 0,
                    'v4_top1_in_top3': 0, 'v4_top1_in_top5': 0,
                    'jaccard_top5_sum': 0.0,
                    'spearman_sum': 0.0,
                    'n_correlated': 0}
             for name in predictor_names}

    # Joint mistake analysis: 2x2 confusion matrix for each predictor vs v4
    # Cells: (both correct, v4 correct only, predictor correct only, both wrong)
    # "correct" = predicted cell is legal at this position.
    joint_stats = {name: {'both_right': 0, 'v4_only_right': 0,
                          'other_only_right': 0, 'both_wrong': 0,
                          'both_wrong_same_pick': 0}
                   for name in predictor_names if name != 'v4'}

    for game in tqdm(games, desc="evaluating"):
        board = OthelloBoardState()
        for k in range(args.pos_end):
            if k >= len(game):
                break
            if not (args.pos_start <= k < args.pos_end):
                board.update([game[k]])
                continue

            legal = set(board.get_valid_moves())  # 64-cell ids
            if not legal:
                board.update([game[k]])
                continue

            played_set = set(game[:k])
            played_parity = {c: i % 2 for i, c in enumerate(game[:k])}

            # Get v4 scores once (used for top-k & rank computations)
            v4_scores = v4_cell_scores(v4, game, k, cell_stoi, device)
            pred_v4_60 = int(np.argmax(v4_scores))
            pred_v4 = C60_TO_C64.get(pred_v4_60)
            v4_top3 = set(np.argsort(-v4_scores)[:3].tolist())
            v4_top5 = set(np.argsort(-v4_scores)[:5].tolist())
            v4_rank = (-v4_scores).argsort().argsort()  # rank 0 = best

            # Predictions
            pred_random = heuristic_random_unplayed(played_set)
            pred_adjacent = heuristic_adjacent_to_played(played_set)
            pred_parity = heuristic_parity_line(played_parity)
            mlp512_scores = mlp_cell_scores(mlp_512, game, k, device) if mlp_512 else None
            mlp8192_scores = mlp_cell_scores(mlp_8192, game, k, device) if mlp_8192 else None
            pred_mlp512 = C60_TO_C64.get(int(np.argmax(mlp512_scores))) if mlp_512 else None
            pred_mlp8192 = C60_TO_C64.get(int(np.argmax(mlp8192_scores))) if mlp_8192 else None

            picks = [('random_unplayed', pred_random, None),
                     ('adjacent', pred_adjacent, None),
                     ('parity_line', pred_parity, None)]
            if mlp_512 is not None: picks.append(('mlp_h512', pred_mlp512, mlp512_scores))
            if mlp_8192 is not None: picks.append(('mlp_h8192', pred_mlp8192, mlp8192_scores))
            picks.append(('v4', pred_v4, v4_scores))

            v4_correct = pred_v4 is not None and pred_v4 in legal

            for name, pred, scores in picks:
                stats[name]['n'] += 1
                if pred is not None and pred in legal:
                    stats[name]['legal'] += 1
                if pred == pred_v4:
                    stats[name]['agree_v4'] += 1
                # Predictor's top-1 → is it in v4's top-3 / top-5?
                if pred is not None and pred in C64_TO_C60:
                    p60 = C64_TO_C60[pred]
                    if p60 in v4_top3:
                        stats[name]['v4_top1_in_top3'] += 1
                    if p60 in v4_top5:
                        stats[name]['v4_top1_in_top5'] += 1
                # Score-based measures (only available for v4/MLP)
                if scores is not None:
                    top5 = set(np.argsort(-scores)[:5].tolist())
                    jacc = len(top5 & v4_top5) / len(top5 | v4_top5)
                    stats[name]['jaccard_top5_sum'] += jacc
                    # Spearman: rank correlation
                    other_rank = (-scores).argsort().argsort()
                    # Pearson on ranks = Spearman
                    sp = np.corrcoef(v4_rank, other_rank)[0, 1]
                    stats[name]['spearman_sum'] += sp
                    stats[name]['n_correlated'] += 1

                # Joint mistake analysis (predictor vs v4)
                if name != 'v4':
                    other_correct = pred is not None and pred in legal
                    js = joint_stats[name]
                    if v4_correct and other_correct:
                        js['both_right'] += 1
                    elif v4_correct and not other_correct:
                        js['v4_only_right'] += 1
                    elif other_correct and not v4_correct:
                        js['other_only_right'] += 1
                    else:
                        js['both_wrong'] += 1
                        if pred == pred_v4:
                            js['both_wrong_same_pick'] += 1

            board.update([game[k]])

    n_total = stats['v4']['n']
    print(f"\n=== Comparison (positions {args.pos_start}..{args.pos_end-1}, "
          f"n={n_total}) ===")
    hdr = (f"  {'Predictor':<20s}  {'top1 legal':>10s}  "
           f"{'top1=v4top1':>11s}  {'in v4 top3':>10s}  {'in v4 top5':>10s}  "
           f"{'Jacc top5':>10s}  {'Spearman':>9s}")
    print(hdr)
    for name in predictor_names:
        s = stats[name]
        n = max(1, s['n'])
        legal_pct = 100 * s['legal'] / n
        agree_pct = 100 * s['agree_v4'] / n
        t3_pct = 100 * s['v4_top1_in_top3'] / n
        t5_pct = 100 * s['v4_top1_in_top5'] / n
        nc = max(1, s['n_correlated'])
        if s['n_correlated'] > 0:
            jacc_avg = s['jaccard_top5_sum'] / nc
            sp_avg = s['spearman_sum'] / nc
            jacc_str = f"{jacc_avg:.3f}"
            sp_str = f"{sp_avg:.3f}"
        else:
            jacc_str = sp_str = "—"
        print(f"  {name:<20s}  {legal_pct:>9.2f}%  "
              f"{agree_pct:>10.2f}%  {t3_pct:>9.2f}%  {t5_pct:>9.2f}%  "
              f"{jacc_str:>10s}  {sp_str:>9s}")

    # Joint mistake analysis
    print(f"\n=== Mistake-overlap with v4 (n={n_total}) ===")
    print(f"  {'Predictor':<20s}  {'both right':>10s}  {'v4 only':>9s}  "
          f"{'other only':>11s}  {'both wrong':>10s}  "
          f"{'P(both|either)':>15s}  {'P(other|v4)':>13s}  {'P(v4|other)':>13s}")
    for name in predictor_names:
        if name == 'v4' or name not in joint_stats:
            continue
        js = joint_stats[name]
        n = max(1, js['both_right'] + js['v4_only_right']
                + js['other_only_right'] + js['both_wrong'])
        # Conditional mistake probabilities
        v4_wrong = js['v4_only_right'] == 0 and False  # placeholder
        v4_wrong_n = js['other_only_right'] + js['both_wrong']  # v4 was wrong
        other_wrong_n = js['v4_only_right'] + js['both_wrong']  # other was wrong
        either_wrong_n = (js['v4_only_right'] + js['other_only_right']
                          + js['both_wrong'])
        both_given_either = (js['both_wrong'] / max(1, either_wrong_n) * 100)
        # P(other wrong | v4 wrong)
        other_g_v4 = (js['both_wrong'] / max(1, v4_wrong_n) * 100)
        # P(v4 wrong | other wrong)
        v4_g_other = (js['both_wrong'] / max(1, other_wrong_n) * 100)
        print(f"  {name:<20s}  {js['both_right']:>10d}  "
              f"{js['v4_only_right']:>9d}  {js['other_only_right']:>11d}  "
              f"{js['both_wrong']:>10d}  "
              f"{both_given_either:>13.2f}%  "
              f"{other_g_v4:>11.2f}%  {v4_g_other:>11.2f}%")

    print("\n  P(both wrong | either wrong): high → same heuristic, fail together")
    print("  P(other wrong | v4 wrong):    high → other shares v4's blindspots")
    print("  P(v4 wrong | other wrong):    high → v4 shares other's blindspots")


if __name__ == '__main__':
    main()
