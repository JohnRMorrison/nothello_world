"""Stage 4: Validate extracted heuristics.

Three validation checks for each of three fitting methods:
  1. Heuristic policy games (10K): measure legal move rate
  2. Distribution comparison (10K positions): KL divergence, rank correlation
  3. Coverage analysis (100K positions): fraction of legal cells with firing heuristic

Usage:
    python behavioral_validate.py --data-dir behavioral_data
    python behavioral_validate.py --data-dir behavioral_data --policy-games 100  # test
"""

import argparse
import json
import os
import sys
import time
import random
import numpy as np
from scipy.stats import spearmanr

from behavioral_utils import (
    build_120d_features, N_MOVES, VALID_MOVES, MOVE_TO_IDX, IDX_TO_MOVE
)
from data.othello import OthelloBoardState


# =============================================================================
# Load heuristics
# =============================================================================

def load_heuristics(data_dir, method):
    """Load all per-cell heuristic files for a method.

    Returns: dict mapping cell_idx -> list of heuristic dicts
    """
    heur_dir = os.path.join(data_dir, f"heuristics_{method}")
    all_heuristics = {}

    for cell in range(N_MOVES):
        path = os.path.join(heur_dir, f"cell_{cell:02d}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            cell_data = json.load(f)
        all_heuristics[cell] = cell_data.get('heuristics', [])

    return all_heuristics


def evaluate_conjunction(features_120d, conjunction):
    """Check if a conjunction of conditions is satisfied.

    Args:
        features_120d: (120,) numpy array
        conjunction: list of {feature_idx, direction, threshold} dicts

    Returns: bool
    """
    for cond in conjunction:
        feat_idx = cond['feature_idx']
        val = features_120d[feat_idx]
        if cond['direction'] == '<=':
            if not (val <= cond['threshold']):
                return False
        elif cond['direction'] == '>':
            if not (val > cond['threshold']):
                return False
    return True


def compute_heuristic_scores(features_120d, all_heuristics):
    """Compute heuristic scores for all 60 cells at a single position.

    For each cell, finds the maximum avg_model_prob among firing promoting
    heuristics.

    Args:
        features_120d: (120,) numpy array
        all_heuristics: dict mapping cell_idx -> list of heuristics

    Returns:
        scores: (60,) numpy array of heuristic scores
    """
    scores = np.zeros(N_MOVES, dtype=np.float32)

    for cell in range(N_MOVES):
        if cell not in all_heuristics:
            continue
        for h in all_heuristics[cell]:
            if h.get('type') != 'promoting':
                continue
            if evaluate_conjunction(features_120d, h['conjunction']):
                scores[cell] = max(scores[cell], h['avg_model_prob'])

    return scores


# =============================================================================
# Validation 1: Heuristic policy games
# =============================================================================

def play_heuristic_game(all_heuristics, temperature=1.0, max_retries=10):
    """Play a game using heuristics as policy.

    Returns:
        game: list of moves played
        total_attempts: total move attempts (including illegal)
        illegal_attempts: number of illegal attempts
    """
    board = OthelloBoardState()
    game = []
    total_attempts = 0
    illegal_attempts = 0

    while True:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            break

        # Compute features
        features, _ = build_120d_features([game + [0]], len(game), len(game) + 1)
        # The features are for the current position (before making a move)
        # We need features for positions 0..len(game)-1 being played
        if len(game) >= 4:
            f, _ = build_120d_features([game], len(game) - 1, len(game))
            feat = f[0]
        else:
            feat = np.zeros(120, dtype=np.float32)
            for s, m in enumerate(game):
                idx = MOVE_TO_IDX[m]
                feat[idx] = 1.0
                feat[N_MOVES + idx] = 1.0 if s % 2 == 0 else 0.0

        scores = compute_heuristic_scores(feat, all_heuristics)

        # Softmax
        if scores.max() > 0:
            exp_scores = np.exp((scores - scores.max()) / max(temperature, 0.01))
            probs = exp_scores / exp_scores.sum()
        else:
            # No heuristic fires — uniform over all cells
            probs = np.ones(N_MOVES) / N_MOVES

        # Sample and check legality
        valid_set = set(valid_moves)
        for attempt in range(max_retries):
            total_attempts += 1
            move_idx = np.random.choice(N_MOVES, p=probs)
            move = IDX_TO_MOVE[move_idx]
            if move in valid_set:
                game.append(move)
                board.umpire(move)
                break
            else:
                illegal_attempts += 1
        else:
            # All retries failed — fall back to random legal move
            total_attempts += 1
            move = random.choice(valid_moves)
            game.append(move)
            board.umpire(move)

    return game, total_attempts, illegal_attempts


def validate_policy_games(all_heuristics, num_games=10000):
    """Play games using heuristics, measure legal move rate."""
    total_moves = 0
    total_illegal = 0
    game_lengths = []

    for i in range(num_games):
        game, attempts, illegal = play_heuristic_game(all_heuristics)
        total_moves += attempts
        total_illegal += illegal
        game_lengths.append(len(game))

        if (i + 1) % 1000 == 0:
            rate = 1 - total_illegal / max(total_moves, 1)
            print(f"  {i+1}/{num_games} games, legal rate: {rate:.4f}", flush=True)

    legal_rate = 1 - total_illegal / max(total_moves, 1)
    return {
        'num_games': num_games,
        'total_attempts': total_moves,
        'illegal_attempts': total_illegal,
        'legal_move_rate': float(legal_rate),
        'avg_game_length': float(np.mean(game_lengths)),
    }


# =============================================================================
# Validation 2: Distribution comparison
# =============================================================================

def validate_distribution(all_heuristics, data_dir, num_positions=10000):
    """Compare heuristic distribution to model's actual distribution."""
    # Load random positions from first shard
    shard_path = os.path.join(data_dir, "shard_00.npz")
    if not os.path.exists(shard_path):
        return {'error': 'no shard data available'}

    data = np.load(shard_path)
    features = data['features'].astype(np.float32)
    probs = data['probs'].astype(np.float32)

    n = min(num_positions, len(features))
    idx = np.random.choice(len(features), n, replace=False)

    kl_divs = []
    rank_corrs = []

    for i in idx:
        feat = features[i]
        model_probs = probs[i]

        # Compute heuristic scores
        h_scores = compute_heuristic_scores(feat, all_heuristics)

        # Softmax to get heuristic distribution
        if h_scores.max() > 0:
            exp_s = np.exp(h_scores - h_scores.max())
            h_probs = exp_s / exp_s.sum()
        else:
            h_probs = np.ones(N_MOVES) / N_MOVES

        # KL divergence: KL(model || heuristic)
        # Clip to avoid log(0)
        m = np.clip(model_probs, 1e-10, 1.0)
        h = np.clip(h_probs, 1e-10, 1.0)
        kl = float(np.sum(m * np.log(m / h)))
        kl_divs.append(kl)

        # Rank correlation
        corr, _ = spearmanr(model_probs, h_probs)
        if not np.isnan(corr):
            rank_corrs.append(corr)

    return {
        'num_positions': n,
        'kl_divergence_mean': float(np.mean(kl_divs)),
        'kl_divergence_std': float(np.std(kl_divs)),
        'rank_correlation_mean': float(np.mean(rank_corrs)),
        'rank_correlation_std': float(np.std(rank_corrs)),
    }


# =============================================================================
# Validation 3: Coverage analysis
# =============================================================================

def validate_coverage(all_heuristics, data_dir, num_positions=100000):
    """Check what fraction of legal cells have a firing heuristic."""
    shard_files = sorted([f for f in os.listdir(data_dir)
                          if f.startswith("shard_") and f.endswith(".npz")])
    if not shard_files:
        return {'error': 'no shard data'}

    total_legal_cells = 0
    covered_legal_cells = 0
    total_false_positives = 0
    model_agrees_on_fp = 0
    positions_checked = 0

    for shard_file in shard_files:
        data = np.load(os.path.join(data_dir, shard_file))
        features = data['features'].astype(np.float32)
        probs = data['probs'].astype(np.float32)
        legal = data['legal']

        # Sample from this shard
        n_from_shard = min(num_positions - positions_checked,
                          len(features))
        if n_from_shard <= 0:
            break

        idx = np.random.choice(len(features), n_from_shard, replace=False)

        for i in idx:
            feat = features[i]
            model_p = probs[i]
            legal_mask = legal[i]

            h_scores = compute_heuristic_scores(feat, all_heuristics)

            for cell in range(N_MOVES):
                fires = h_scores[cell] > 0
                is_legal = legal_mask[cell] > 0

                if is_legal:
                    total_legal_cells += 1
                    if fires:
                        covered_legal_cells += 1

                if fires and not is_legal:
                    total_false_positives += 1
                    if model_p[cell] > 0.02:
                        model_agrees_on_fp += 1

        positions_checked += n_from_shard
        del data

        if positions_checked >= num_positions:
            break

    coverage = covered_legal_cells / max(total_legal_cells, 1)
    fp_model_agreement = model_agrees_on_fp / max(total_false_positives, 1)

    return {
        'num_positions': positions_checked,
        'total_legal_cells': total_legal_cells,
        'covered_legal_cells': covered_legal_cells,
        'coverage': float(coverage),
        'total_false_positives': total_false_positives,
        'model_agrees_on_false_positives': model_agrees_on_fp,
        'fp_model_agreement_rate': float(fp_model_agreement),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Validate heuristics")
    parser.add_argument("--data-dir", type=str, default="behavioral_data")
    parser.add_argument("--policy-games", type=int, default=10000)
    parser.add_argument("--dist-positions", type=int, default=10000)
    parser.add_argument("--coverage-positions", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    methods = ['two_level', 'beam_weighted', 'natural_weighted']
    results = {}

    for method in methods:
        print(f"\n{'='*60}", flush=True)
        print(f"Validating method: {method}", flush=True)
        print(f"{'='*60}", flush=True)

        heuristics = load_heuristics(args.data_dir, method)
        n_cells = len(heuristics)
        n_promoting = sum(
            1 for cell_heurs in heuristics.values()
            for h in cell_heurs if h.get('type') == 'promoting'
        )
        print(f"  Loaded heuristics for {n_cells} cells, "
              f"{n_promoting} promoting total", flush=True)

        if n_promoting == 0:
            print(f"  No promoting heuristics — skipping validation", flush=True)
            results[method] = {'error': 'no promoting heuristics'}
            continue

        # Validation 1: Policy games
        print(f"\n  --- Policy games ({args.policy_games}) ---", flush=True)
        policy_results = validate_policy_games(heuristics, args.policy_games)
        print(f"  Legal move rate: {policy_results['legal_move_rate']:.4f}", flush=True)

        # Validation 2: Distribution comparison
        print(f"\n  --- Distribution comparison ({args.dist_positions}) ---", flush=True)
        dist_results = validate_distribution(heuristics, args.data_dir,
                                             args.dist_positions)
        if 'error' not in dist_results:
            print(f"  KL divergence: {dist_results['kl_divergence_mean']:.4f} "
                  f"± {dist_results['kl_divergence_std']:.4f}", flush=True)
            print(f"  Rank correlation: {dist_results['rank_correlation_mean']:.4f} "
                  f"± {dist_results['rank_correlation_std']:.4f}", flush=True)

        # Validation 3: Coverage
        print(f"\n  --- Coverage ({args.coverage_positions}) ---", flush=True)
        cov_results = validate_coverage(heuristics, args.data_dir,
                                        args.coverage_positions)
        if 'error' not in cov_results:
            print(f"  Coverage: {cov_results['coverage']:.4f}", flush=True)
            print(f"  False positives: {cov_results['total_false_positives']}", flush=True)
            print(f"  FP model agreement: {cov_results['fp_model_agreement_rate']:.4f}",
                  flush=True)

        results[method] = {
            'n_cells': n_cells,
            'n_promoting': n_promoting,
            'policy_games': policy_results,
            'distribution': dist_results,
            'coverage': cov_results,
        }

    # Save
    out_path = os.path.join(args.data_dir, "validation_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. Results saved to {out_path} ({elapsed:.0f}s)", flush=True)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for method in methods:
        r = results.get(method, {})
        if 'error' in r:
            print(f"  {method}: {r['error']}")
            continue
        pg = r.get('policy_games', {})
        cov = r.get('coverage', {})
        dist = r.get('distribution', {})
        print(f"  {method}:")
        print(f"    Legal rate: {pg.get('legal_move_rate', 'N/A'):.4f}")
        print(f"    Coverage:   {cov.get('coverage', 'N/A'):.4f}")
        print(f"    KL div:     {dist.get('kl_divergence_mean', 'N/A'):.4f}")
        print(f"    Rank corr:  {dist.get('rank_correlation_mean', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
