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
    """Return predicted cell (0..59 cell-stoi space) for next move at depth k.

    For v4 (cell-indexed), context positions hold the played-cell+parity tags.
    """
    from train_gpt_shuffled_v4 import CellIndexedMaskedDataset
    ds = CellIndexedMaskedDataset([list(game)], cell_stoi=cell_stoi)
    Lc = ds.context_len
    x, _, mask = ds[0]
    x = x.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x, attn_mask=mask)
    # Query position Lc+k predicts game[k] from prefix game[:k].
    # (At training time, target y[Lc+m] = game[m].)
    qpos = Lc + k
    vec = logits[0, qpos]
    # Token id = 1 + cell + 60 * parity. Argmax over UNIQUE cells.
    sorted_tokens = vec.argsort(descending=True).cpu().tolist()
    for tok in sorted_tokens:
        if 1 <= tok <= 120:
            return (tok - 1) % 60   # returns 0..59 cell-stoi index
    return None


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
    """Run a played+even pattern-detector MLP on game[:k], aggregate via
    prob_or, return predicted cell (64-cell index) for move at depth k.

    prob_or aggregator:  P(cell c is legal) = 1 - prod_p (1 - sigmoid(logit_p))
    over patterns p targeting c; we rank by -log(1 - P) = sum_p softplus(logit_p),
    which is monotone in P.
    """
    me, mo, idx, mask = mlp_bundle
    features = played_even_features(game[:k]).unsqueeze(0).to(device)
    # MLP convention: at training "position" t, features include cells played
    # at steps 0..t (inclusive) and the label is legality at the NEXT step t+1.
    # `me` (model_even) was trained where pos % 2 == 0.
    # In our eval we predict game[k] using features for steps 0..k-1.
    # That maps to MLP position t = k - 1.  So use `me` iff (k-1) % 2 == 0,
    # i.e. when k is odd.
    use_even = (k % 2 == 1)
    model = me if use_even else mo
    with torch.no_grad():
        logits = model(features)  # (1, n_patterns)
    # prob_or aggregation: per-cell score = sum_p softplus(logit_p) over patterns
    # p targeting cell c.  (Higher score = higher P(cell c legal).)
    log1m = -torch.nn.functional.softplus(logits)        # (1, n_patterns)
    gathered = log1m[:, idx]                              # (1, 60, max_p_per_cell)
    gathered = gathered.masked_fill(~mask, 0.0)
    cell_scores = -gathered.sum(dim=-1)[0]                # (60,)
    pred_c60 = cell_scores.argmax().item()
    return C60_TO_C64.get(pred_c60)


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
    stats = {name: {'n': 0, 'legal': 0, 'agree_v4': 0}
             for name in predictor_names}

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

            # Predictions
            pred_random = heuristic_random_unplayed(played_set)
            pred_adjacent = heuristic_adjacent_to_played(played_set)
            pred_parity = heuristic_parity_line(played_parity)
            pred_v4_60 = predict_v4(v4, game, k, cell_stoi, device)
            pred_v4 = C60_TO_C64.get(pred_v4_60) if pred_v4_60 is not None else None
            pred_mlp512 = predict_mlp(mlp_512, game, k, device) if mlp_512 else None
            pred_mlp8192 = predict_mlp(mlp_8192, game, k, device) if mlp_8192 else None

            picks = [('random_unplayed', pred_random),
                     ('adjacent', pred_adjacent),
                     ('parity_line', pred_parity)]
            if mlp_512 is not None: picks.append(('mlp_h512', pred_mlp512))
            if mlp_8192 is not None: picks.append(('mlp_h8192', pred_mlp8192))
            picks.append(('v4', pred_v4))

            for name, pred in picks:
                stats[name]['n'] += 1
                if pred is not None and pred in legal:
                    stats[name]['legal'] += 1
                if pred == pred_v4:
                    stats[name]['agree_v4'] += 1

            board.update([game[k]])

    print(f"\n=== Comparison (positions {args.pos_start}..{args.pos_end-1}, "
          f"n={stats['v4']['n']}) ===")
    print(f"  {'Predictor':<20s}  {'top-1 legal':>12s}  {'agree w/ v4':>12s}")
    for name in predictor_names:
        s = stats[name]
        legal_pct = 100 * s['legal'] / max(1, s['n'])
        agree_pct = 100 * s['agree_v4'] / max(1, s['n'])
        print(f"  {name:<20s}  {legal_pct:>11.2f}%  {agree_pct:>11.2f}%")


if __name__ == '__main__':
    main()
