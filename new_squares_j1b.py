"""New Squares Transfer Experiment — J1B (tree-bank Othello-MLP) runner.

The J1B analog of new_squares_transfer.py (the GPT runner).  Instead of
fine-tuning a transformer, J1B learns the 8 new squares (cells 64-71, a 9th
row) by (a) FITTING one decision tree per new-square flanking pattern
(feature discovery from a played/placed feature vector) and (b) SGD-training a
prob-OR readout on the concatenated tree-leaf one-hots against new-square
legality.

The experiment measures data efficiency: IL_prob (probability mass the model
puts on the legal new-square target(s)) as a function of the number of
TRAINING GAMES, for coherent_all (condition 2) vs incoherent_all (condition 3).
Tree hyperparameters are held FIXED at the base J1 bank's values
(banks/J1_perpattern.pt args); only the number of games moves along the curve.

Data + eval are model-agnostic and shared with the GPT runner via
new_squares_data.py (rules, game engine, test manifest, score_manifest), so the
two models are scored by identical code.  J1B-specific pieces here:
  - features_72(): base 121-d played/placed features (opening_tree_mlp) EXTENDED
    with 16 new-square bits (played + placed-as-mover for cells 64-71), since
    the base feature extractor only encodes the 60 movable 8x8 cells.
  - per-pattern firing targets recomputed from the board state at every
    training position (OthelloBoardState for the 8x8 part + a new-square array).
  - NewSquareProbOr: Linear(H -> K patterns) -> sigmoid -> prob-OR grouped by
    target new square (the LinearPatternProbOr recipe, but over the 8 new
    squares 64-71 instead of the 64 base cells).

Usage:
    # smoke
    python new_squares_j1b.py --condition-id 2 --schedule 250,1000 \
        --num-test-games 1500
    # full curve
    python new_squares_j1b.py --condition-id 2 --schedule 250,750,2000,6000,20000
    python new_squares_j1b.py --condition-id 3 --schedule 250,750,2000,6000,20000
"""

import argparse
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.othello import OthelloBoardState
from opening_tree_mlp import (
    playedeven_features, train_pattern_trees, extract_paths, path_to_weight,
    OpeningTreeMLP,
)
from new_squares_data import (
    CONDITIONS, NEW_SQUARE_IDS, N_CELLS, N_NEW, make_rules,
    generate_game_with_new_squares, build_test_manifest, score_manifest,
)

# ----------------------------------------------------------------------------
# Feature extension: base 121-d played/placed features (8x8 movable cells only)
# ⊕ 16 new-square bits (played + placed-as-mover for cells 64-71).
# ----------------------------------------------------------------------------
BASE_DIM = 121
FEAT_DIM = BASE_DIM + 2 * N_NEW      # 121 + 16 = 137


def features_72(prefix):
    """Base J1 features (canonicalize_mover) extended with the new squares.

    Base 121-d encodes only the 60 movable 8x8 cells.  We append, per new
    square k in 0..7:
      ext[k]        = played bit    (1 iff cell 64+k appears in the prefix)
      ext[N_NEW+k]  = placed-as-mover bit (1 iff that move was made by the side
                      currently to move — parity(t) == parity(len(prefix)),
                      the SAME mover-relative convention playedeven_features
                      uses for the 8x8 cells).
    New-square moves never flip anything, so the played/placed bit is an exact
    encoding of the new-square state for the mover-relative colour.
    """
    base = playedeven_features(prefix, canonicalize_mover=True)   # (121,)
    ext = np.zeros(2 * N_NEW, dtype=np.float32)
    mp = len(prefix) % 2
    for t, c in enumerate(prefix):
        if c >= 64:
            k = c - 64
            ext[k] = 1.0
            if (t % 2) == mp:
                ext[N_NEW + k] = 1.0
    return np.concatenate([base, ext]).astype(np.float32)


# ----------------------------------------------------------------------------
# Per-pattern firing + per-new-square legality, recomputed from board state.
# Mirrors generate_game_with_new_squares' move execution + firing logic exactly
# so the replayed board states match the states the games were generated under.
# ----------------------------------------------------------------------------

def replay_positions(moves, rules):
    """Replay a generated game; at each ply (BEFORE the move) return the
    per-pattern firing vector and the per-new-square legality vector.

    A pattern (rule with target>=64, opponents, terminal) FIRES iff:
      - the target new square is empty,
      - every opponent cell holds the OPPONENT's colour, and
      - the terminal cell holds the MOVER's colour.
    This is the identical condition generate_game_with_new_squares uses.

    Returns list of (prefix_len, fire (K,) int8, legal (N_NEW,) int8).
    """
    K = len(rules)
    board = OthelloBoardState()
    new_state = np.zeros(N_NEW, dtype=np.int32)   # 0 empty, +1 black, -1 white
    out = []
    for i, mv in enumerate(moves):
        flat = board.state.flatten()
        is_black = (board.next_hand_color == 1)
        my_val = 1 if is_black else -1
        opp_val = -my_val

        fire = np.zeros(K, dtype=np.int8)
        legal = np.zeros(N_NEW, dtype=np.int8)
        for j, rule in enumerate(rules):
            target = rule['target']
            if target >= 64:
                if new_state[target - 64] != 0:
                    continue
            elif flat[target] != 0:
                continue
            ok = True
            for opp_cell in rule['opponents']:
                val = new_state[opp_cell - 64] if opp_cell >= 64 else flat[opp_cell]
                if val != opp_val:
                    ok = False
                    break
            if not ok:
                continue
            term_cell = rule['terminal']
            term_val = (new_state[term_cell - 64] if term_cell >= 64
                        else flat[term_cell])
            if term_val != my_val:
                continue
            fire[j] = 1
            if target >= 64:
                legal[target - 64] = 1
        out.append((i, fire, legal))

        # Execute the move EXACTLY as the generator does.
        if mv >= 64:
            new_state[mv - 64] = my_val
            board.next_hand_color *= -1
        else:
            if board.tentative_move(mv) != 0:
                board.update([mv])
            else:
                r, c = mv // 8, mv % 8
                board.state[r, c] = my_val
                board.next_hand_color *= -1
    return out


def build_training_data(games, rules, max_positions=None, seed=0, min_prefix=1):
    """Enumerate all (prefix) positions in `games`, returning
    X (N, FEAT_DIM), FIRE (N, K) per-pattern firing, LEGAL (N, N_NEW).

    Optionally subsample to `max_positions` (cap for speed at large D)."""
    Xs, FIREs, LEGALs = [], [], []
    for moves in games:
        for (plen, fire, legal) in replay_positions(moves, rules):
            if plen < min_prefix:
                continue
            Xs.append(features_72(moves[:plen]))
            FIREs.append(fire)
            LEGALs.append(legal)
    X = np.stack(Xs).astype(np.float32)
    FIRE = np.stack(FIREs).astype(np.int8)
    LEGAL = np.stack(LEGALs).astype(np.float32)
    if max_positions is not None and X.shape[0] > max_positions:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], max_positions, replace=False)
        X, FIRE, LEGAL = X[idx], FIRE[idx], LEGAL[idx]
    return X, FIRE, LEGAL


# ----------------------------------------------------------------------------
# Hidden layer (tree-leaf one-hots) — ±1 path linearization is EXACT here
# because every feature is binary (played/placed bits).
# ----------------------------------------------------------------------------

def build_hidden_layer(trees, input_dim, device):
    """One hidden unit per tree leaf across all pattern trees; returns an
    OpeningTreeMLP.  step(W x + b) reproduces the exact leaf indicator for
    binary features (see path_to_weight)."""
    all_w, all_b, meta = [], [], []
    for j, tree_list in enumerate(trees):
        tree = tree_list[0]                       # n_trees_per_pattern == 1
        for path_idx, (conds, leaf_class, leaf_counts) in enumerate(
                extract_paths(tree)):
            w, b = path_to_weight(conds, input_dim=input_dim)
            all_w.append(w)
            all_b.append(b)
            meta.append({'pattern': j, 'path_idx': path_idx})
    W = np.stack(all_w)
    B = np.array(all_b, dtype=np.float32)
    return OpeningTreeMLP(W, B, meta, device)


def hidden_activations(mlp, X_np, device, batch=4096):
    """Float 0/1 hidden activations (N, H) on `device`."""
    x = torch.from_numpy(X_np).to(device)
    H = mlp(x, batch=batch, out_device=device, out_dtype=torch.float32)
    del x
    return H


# ----------------------------------------------------------------------------
# Prob-OR readout over the 8 new squares (the LinearPatternProbOr recipe, but
# the target space is the new squares 64-71, not the 64 base cells).
# ----------------------------------------------------------------------------

class NewSquareProbOr(nn.Module):
    def __init__(self, hidden_dim, rules):
        super().__init__()
        K = len(rules)
        self.linear = nn.Linear(hidden_dim, K)
        by_tgt = defaultdict(list)
        for j, rule in enumerate(rules):
            by_tgt[rule['target'] - 64].append(j)      # 0..7
        max_per = max(len(v) for v in by_tgt.values())
        idx = torch.zeros(N_NEW, max_per, dtype=torch.long)
        msk = torch.zeros(N_NEW, max_per, dtype=torch.bool)
        for cell in range(N_NEW):
            for k, pos in enumerate(by_tgt.get(cell, [])):
                idx[cell, k] = pos
                msk[cell, k] = True
        self.register_buffer('cell_pat_indices', idx)
        self.register_buffer('cell_pat_mask', msk)

    def forward(self, h):
        pat_probs = torch.sigmoid(self.linear(h))          # (N, K)
        gathered = pat_probs[:, self.cell_pat_indices]     # (N, N_NEW, max)
        one_minus = torch.where(self.cell_pat_mask.unsqueeze(0),
                                1.0 - gathered, torch.ones_like(gathered))
        return 1.0 - one_minus.prod(dim=2)                 # (N, N_NEW)


def train_readout(H_tr, LEGAL_tr, rules, epochs=60, lr=0.05, batch=2048,
                  weight_decay=1e-4, device='cpu', seed=0, bias_init=None):
    """SGD-train the prob-OR readout to predict new-square LEGALITY -- the base
    default recipe (binary_cross_entropy on the prob-OR *output* vs legal
    targets).  This is the meaningful readout-learning step: it learns to
    COMBINE the pattern detectors into per-new-square legality (the trees
    already detect firing; supervising firing would just re-read them).

    Class weighting is applied per new square (pos_weight = neg/pos).  The base
    64-cell readout used UNWEIGHTED BCE, but it had ~11 legal cells per position;
    with only 8 sparse new squares (each legal ~10% of positions) unweighted BCE
    collapses the readout to all-zeros (it did, for incoherent).  We add balanced
    weighting explicitly to counter that -- documented, not silent."""
    torch.manual_seed(seed)
    probe = NewSquareProbOr(H_tr.shape[1], rules).to(device)
    if bias_init is not None:
        # Negative bias -> initial prob-OR near the base legality rate instead
        # of ~1.  Without it, a wide/uninformative hidden layer (e.g. the FROZEN
        # bank) lets the ~90% negatives crush all logits into the saturated
        # sigmoid=0 region on step 1, where gradients vanish and the readout is
        # stuck at 0.  Informative (refit) leaves escape this, so the default
        # (None) leaves the original init untouched.
        torch.nn.init.constant_(probe.linear.bias, float(bias_init))
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    L = torch.from_numpy(LEGAL_tr.astype(np.float32)).to(device)    # (N, N_NEW) legality
    pos = L.sum(0); neg = L.shape[0] - pos
    pos_weight = (neg / pos.clamp(min=1)).clamp(max=1000)           # balanced per new square
    N = H_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            h = H_tr[idx].to(device=device, dtype=torch.float32)
            p = probe(h).clamp(1e-6, 1 - 1e-6)                      # prob-OR legality output
            l = L[idx]
            loss = -(pos_weight * l * torch.log(p)
                     + (1.0 - l) * torch.log(1.0 - p)).mean()       # class-weighted BCE
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def make_score_fn(mlp, probe, device):
    """prefix -> per-cell (72,) score.  New squares 64-71 get the readout's
    legality prob; base cells 0-63 are left at 0 (J1B's base bank scores those,
    not this new-squares readout)."""
    probe.eval()

    def score_fn(prefix):
        scores = np.zeros(N_CELLS, dtype=np.float32)
        x = torch.from_numpy(features_72(prefix)[None, :]).to(device)
        h = mlp(x, out_device=device, out_dtype=torch.float32)
        with torch.no_grad():
            p = probe(h)[0].cpu().numpy()          # (N_NEW,)
        scores[64:64 + N_NEW] = p
        return scores
    return score_fn


# ----------------------------------------------------------------------------
# Data pool (generate once per condition; each D uses the first D games)
# ----------------------------------------------------------------------------

def get_data_pool(condition_id, n_train_games, n_test_games, seed,
                  n_test_positions, cache_dir, overwrite=False):
    """Generate (or load) the training-game pool + fixed test manifest.

    A single RandomState drives both training and test games (test games are
    generated AFTER the training pool, so they are disjoint draws from the same
    uniform-random policy)."""
    cond_name = CONDITIONS[condition_id]
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f'pool_cond{condition_id:03d}_tr{n_train_games}'
                       f'_te{n_test_games}_seed{seed}.pkl')
        if os.path.exists(cache_path) and not overwrite:
            print(f'[data] loading cached pool {cache_path}', flush=True)
            with open(cache_path, 'rb') as f:
                d = pickle.load(f)
            return d['rules'], d['games'], d['manifest']

    rng = np.random.RandomState(seed)
    rules = make_rules(cond_name, np.random.RandomState(seed))
    print(f'[data] cond {condition_id} ({cond_name}): {len(rules)} rules; '
          f'generating {n_train_games} train + {n_test_games} test games...',
          flush=True)
    t0 = time.time()
    games = []
    for _ in range(n_train_games):
        moves, _ = generate_game_with_new_squares(rules, rng, save_legal=True)
        games.append(moves)
    test_games, test_records = [], []
    for _ in range(n_test_games):
        moves, recs = generate_game_with_new_squares(rules, rng, save_legal=True)
        test_games.append(moves)
        test_records.append(recs)
    manifest = build_test_manifest(test_games, test_records,
                                   n_positions=n_test_positions, min_turn=4)
    print(f'[data]   done ({time.time()-t0:.0f}s); '
          f'IL={len(manifest["IL"])} LL={len(manifest["LL"])}', flush=True)
    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump({'rules': rules, 'games': games, 'manifest': manifest}, f)
        print(f'[data]   cached -> {cache_path}', flush=True)
    return rules, games, manifest


# ----------------------------------------------------------------------------
# One point on the curve: train J1B on the first D games, score the manifest.
# ----------------------------------------------------------------------------

def fit_and_score(D, rules, games, manifest, args, device):
    t0 = time.time()
    X, FIRE, LEGAL = build_training_data(
        games[:D], rules, max_positions=args.max_positions, seed=args.seed)
    n_fire = int((FIRE.sum(axis=1) > 0).sum())
    print(f'    D={D}: {X.shape[0]} positions '
          f'({n_fire} with >=1 pattern firing); fitting {FIRE.shape[1]} trees...',
          flush=True)

    trees = train_pattern_trees(
        X, FIRE, n_trees_per_pattern=args.pattern_n_trees,
        max_depth=args.tree_max_depth,
        min_samples_leaf=args.tree_min_samples_leaf,
        max_leaf_nodes=args.max_leaf_nodes,
        class_weight=args.class_weight,
        n_jobs=args.tree_n_jobs)

    mlp = build_hidden_layer(trees, input_dim=X.shape[1], device=device)
    H_tr = hidden_activations(mlp, X, device)
    print(f'    D={D}: hidden dim H={H_tr.shape[1]}; training readout '
          f'({args.readout_epochs} epochs)...', flush=True)
    probe = train_readout(H_tr, LEGAL, rules, epochs=args.readout_epochs,
                          lr=args.readout_lr, device=device, seed=args.seed)

    score_fn = make_score_fn(mlp, probe, device)
    m = score_manifest(score_fn, manifest, per_bucket=True)
    print(f'    D={D}: IL_prob={m["IL_prob"]:.4f} '
          f'IL_frac={m["IL_prob_frac"]:.4f} '
          f'IL_per_tgt={m["IL_prob_per_target"]:.4f} '
          f'IL_acc={m["IL_acc"]:.4f}  ({time.time()-t0:.0f}s)', flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition-id', type=int, required=True, choices=[2, 3],
                    help='2=coherent_all, 3=incoherent_all')
    ap.add_argument('--output-dir', default='experiments/new_squares')
    ap.add_argument('--schedule', default='250,750,2000,6000,20000',
                    help='Comma-separated training-game counts (x-axis).')
    ap.add_argument('--num-test-games', type=int, default=5000)
    ap.add_argument('--n-test-positions', type=int, default=5000)
    ap.add_argument('--max-positions', type=int, default=200000,
                    help='Cap on training positions (subsampled) for speed.')
    # Tree hyperparameters — base J1 bank defaults (banks/J1_perpattern.pt args).
    ap.add_argument('--tree-max-depth', type=int, default=15)
    ap.add_argument('--tree-min-samples-leaf', type=int, default=50)
    ap.add_argument('--max-leaf-nodes', type=int, default=50)
    ap.add_argument('--pattern-n-trees', type=int, default=1)
    ap.add_argument('--class-weight', default='balanced')
    ap.add_argument('--tree-n-jobs', type=int, default=-1)
    # Readout SGD.
    ap.add_argument('--readout-epochs', type=int, default=60)
    ap.add_argument('--readout-lr', type=float, default=0.05)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cache-dir', default='experiments/new_squares/j1b_cache')
    ap.add_argument('--overwrite-pool', action='store_true')
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != 'cuda'
                          or torch.cuda.is_available()) else 'cpu')
    schedule = [int(s) for s in args.schedule.split(',') if s.strip()]
    D_max = max(schedule)
    cond_name = CONDITIONS[args.condition_id]
    print(f'Condition {args.condition_id}: {cond_name}', flush=True)
    print(f'Schedule (n_train_games): {schedule}', flush=True)
    print(f'Started at: {time.strftime("%c")}  device={device}', flush=True)

    rules, games, manifest = get_data_pool(
        args.condition_id, D_max, args.num_test_games, args.seed,
        args.n_test_positions, args.cache_dir, overwrite=args.overwrite_pool)

    results = {
        'condition_id': args.condition_id, 'condition_name': cond_name,
        'model': 'J1B',
        'n_train_games': schedule,
        'eval_steps': schedule,          # so plot_new_squares_il can reuse x-axis
        'num_test_games': args.num_test_games,
        'n_test_positions': args.n_test_positions,
        'max_positions': args.max_positions,
        'tree_max_depth': args.tree_max_depth,
        'tree_min_samples_leaf': args.tree_min_samples_leaf,
        'max_leaf_nodes': args.max_leaf_nodes,
        'pattern_n_trees': args.pattern_n_trees,
        'class_weight': args.class_weight,
        'readout_epochs': args.readout_epochs, 'readout_lr': args.readout_lr,
        'seed': args.seed,
        'IL_prob': [], 'IL_prob_frac': [], 'IL_prob_per_target': [],
        'IL_acc': [], 'LL_prob': [], 'LL_acc': [],
        'IL_n': [], 'IL_buckets': [],
    }
    t0 = time.time()
    for D in schedule:
        m = fit_and_score(D, rules, games, manifest, args, device)
        results['IL_prob'].append(m['IL_prob'])
        results['IL_prob_frac'].append(m['IL_prob_frac'])
        results['IL_prob_per_target'].append(m['IL_prob_per_target'])
        results['IL_acc'].append(m['IL_acc'])
        results['LL_prob'].append(m['LL_prob'])
        results['LL_acc'].append(m['LL_acc'])
        results['IL_n'].append(m['IL_n'])
        results['IL_buckets'].append(m.get('IL_buckets', {}))
    results['elapsed_seconds'] = time.time() - t0

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir,
                            f'j1b_cond_{args.condition_id:03d}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nD-schedule vs IL_prob:', flush=True)
    for D, ilp, ila in zip(schedule, results['IL_prob'], results['IL_acc']):
        print(f'  D={D:6d}   IL_prob={ilp:.4f}   IL_acc={ila:.4f}', flush=True)
    print(f'Saved {out_path}  ({results["elapsed_seconds"]:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
