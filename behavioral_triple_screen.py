"""Screen triple features for heuristic signal on adversarial positions.

For each target cell, compare triple firing rates between:
  A) Adversarial positions: cell is illegal but model assigns >0.5% prob
  B) Control positions: cell is illegal and model assigns <0.1% prob

Triples with high A/B firing rate ratio are patterns that trick the model.

Usage:
    python behavioral_triple_screen.py --cells 10,20,30,40,50 --data-dir behavioral_data
"""

import argparse
import os
import time
import numpy as np
from itertools import combinations
from behavioral_utils import N_MOVES, VALID_MOVES, MOVE_TO_IDX


def get_move_number(features):
    """Infer move number from when features (count nonzero when values)."""
    return (features[:, :N_MOVES] > 0).sum(axis=1)


def load_adversarial_and_control(data_dir, cell, max_control=50000,
                                  phase_matched=False):
    """Load adversarial positions (A) and control positions (B) for a cell.

    A: cell is illegal, model prob > 0.5%  (from Stage 2 beam search)
    B: cell is illegal, model prob < 0.1%  (from Stage 1 random games)

    If phase_matched=True, control set is sampled to match the adversarial
    set's move-number distribution.
    """
    # Adversarial positions
    adv_path = os.path.join(data_dir, "adversarial", f"cell_{cell:02d}.npz")
    adv_data = np.load(adv_path)
    adv_features = adv_data['features'].astype(np.float32)
    adv_probs = adv_data['target_prob']
    print(f"  Adversarial (A): {len(adv_features)} positions, "
          f"mean prob={adv_probs.mean():.4f}", flush=True)

    adv_moves = get_move_number(adv_features)
    print(f"  Adversarial move range: {adv_moves.min()}-{adv_moves.max()}, "
          f"mean={adv_moves.mean():.1f}", flush=True)

    # Control positions from Stage 1 shards
    ctrl_features_list = []
    ctrl_positions_list = []
    shard_files = sorted([f for f in os.listdir(data_dir)
                          if f.startswith("shard_") and f.endswith(".npz")])

    needed = max_control * 3 if phase_matched else max_control
    for shard_file in shard_files:
        data = np.load(os.path.join(data_dir, shard_file))
        features = data['features'].astype(np.float32)
        probs = data['probs'].astype(np.float32)[:, cell]
        legal = data['legal'][:, cell]
        positions = data['positions']

        # Control: illegal AND model assigns < 0.1%
        ctrl_mask = (legal == 0) & (probs < 0.001)
        if ctrl_mask.any():
            ctrl_features_list.append(features[ctrl_mask])
            ctrl_positions_list.append(positions[ctrl_mask])

        del data, features, probs, legal, positions

        if sum(len(c) for c in ctrl_features_list) >= needed:
            break

    ctrl_features = np.concatenate(ctrl_features_list)
    ctrl_positions = np.concatenate(ctrl_positions_list)

    if phase_matched:
        # Match control set to adversarial move-number distribution
        ctrl_moves = get_move_number(ctrl_features)
        adv_hist, bins = np.histogram(adv_moves, bins=range(0, 62))

        selected_idx = []
        for bin_idx in range(len(adv_hist)):
            if adv_hist[bin_idx] == 0:
                continue
            move_num = bin_idx
            # Find control positions at this move number
            candidates = np.where(ctrl_moves == move_num)[0]
            if len(candidates) == 0:
                continue
            # Sample proportionally (scale up to fill max_control)
            n_sample = min(len(candidates),
                           int(adv_hist[bin_idx] / len(adv_moves) * max_control))
            n_sample = max(n_sample, 1)
            chosen = np.random.choice(candidates, min(n_sample, len(candidates)),
                                      replace=False)
            selected_idx.extend(chosen.tolist())

        if selected_idx:
            ctrl_features = ctrl_features[selected_idx]
        else:
            ctrl_features = ctrl_features[:max_control]

        ctrl_moves_after = get_move_number(ctrl_features)
        print(f"  Control (B, phase-matched): {len(ctrl_features)} positions, "
              f"move range={ctrl_moves_after.min()}-{ctrl_moves_after.max()}, "
              f"mean={ctrl_moves_after.mean():.1f}", flush=True)
    else:
        if len(ctrl_features) > max_control:
            idx = np.random.choice(len(ctrl_features), max_control, replace=False)
            ctrl_features = ctrl_features[idx]
        ctrl_moves = get_move_number(ctrl_features)
        print(f"  Control (B): {len(ctrl_features)} positions, "
              f"move range={ctrl_moves.min()}-{ctrl_moves.max()}, "
              f"mean={ctrl_moves.mean():.1f}", flush=True)

    return adv_features, ctrl_features


def compute_occupied(features):
    """Extract occupancy from when features. when > 0 means occupied."""
    # First 60 features are when[i], > 0 if occupied
    return features[:, :N_MOVES] > 0


def screen_triples(adv_features, ctrl_features, target_cell, top_k=50):
    """Screen all triples by adversarial/control firing rate ratio.

    Returns top_k triples sorted by A/B ratio.
    """
    adv_occ = compute_occupied(adv_features)  # (n_adv, 60) bool
    ctrl_occ = compute_occupied(ctrl_features)  # (n_ctrl, 60) bool
    n_adv = len(adv_occ)
    n_ctrl = len(ctrl_occ)

    # All cells except target
    other_cells = [i for i in range(N_MOVES) if i != target_cell]

    results = []
    n_triples = 0

    # Test all triples of other cells
    for a, b, c in combinations(other_cells, 3):
        # Triple fires when all three cells are occupied
        adv_fire = adv_occ[:, a] & adv_occ[:, b] & adv_occ[:, c]
        ctrl_fire = ctrl_occ[:, a] & ctrl_occ[:, b] & ctrl_occ[:, c]

        adv_rate = adv_fire.mean()
        ctrl_rate = ctrl_fire.mean()

        # Only consider triples that fire at least 5% in adversarial set
        if adv_rate < 0.05:
            continue

        ratio = adv_rate / max(ctrl_rate, 1e-6)

        if ratio > 1.5:  # At least 1.5x more common in adversarial
            results.append({
                'cells': (a, b, c),
                'cell_names': (
                    f"{'ABCDEFGH'[VALID_MOVES[a]//8]}{VALID_MOVES[a]%8+1}",
                    f"{'ABCDEFGH'[VALID_MOVES[b]//8]}{VALID_MOVES[b]%8+1}",
                    f"{'ABCDEFGH'[VALID_MOVES[c]//8]}{VALID_MOVES[c]%8+1}",
                ),
                'adv_rate': float(adv_rate),
                'ctrl_rate': float(ctrl_rate),
                'ratio': float(ratio),
            })

        n_triples += 1

    results.sort(key=lambda x: x['ratio'], reverse=True)
    print(f"  Screened {n_triples} triples with adv_rate >= 5%", flush=True)
    print(f"  Found {len(results)} with ratio > 1.5", flush=True)

    return results[:top_k]


def screen_pairs(adv_features, ctrl_features, target_cell, top_k=50):
    """Also screen pairs for simpler patterns."""
    adv_occ = compute_occupied(adv_features)
    ctrl_occ = compute_occupied(ctrl_features)

    other_cells = [i for i in range(N_MOVES) if i != target_cell]
    results = []

    for a, b in combinations(other_cells, 2):
        adv_fire = adv_occ[:, a] & adv_occ[:, b]
        ctrl_fire = ctrl_occ[:, a] & ctrl_occ[:, b]

        adv_rate = adv_fire.mean()
        ctrl_rate = ctrl_fire.mean()

        if adv_rate < 0.05:
            continue

        ratio = adv_rate / max(ctrl_rate, 1e-6)
        if ratio > 1.5:
            results.append({
                'cells': (a, b),
                'cell_names': (
                    f"{'ABCDEFGH'[VALID_MOVES[a]//8]}{VALID_MOVES[a]%8+1}",
                    f"{'ABCDEFGH'[VALID_MOVES[b]//8]}{VALID_MOVES[b]%8+1}",
                ),
                'adv_rate': float(adv_rate),
                'ctrl_rate': float(ctrl_rate),
                'ratio': float(ratio),
            })

    results.sort(key=lambda x: x['ratio'], reverse=True)
    print(f"  Pairs with ratio > 1.5: {len(results)}", flush=True)
    return results[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=str, default="10,20,30,40,50",
                        help="Comma-separated cell indices to analyze")
    parser.add_argument("--data-dir", type=str, default="behavioral_data")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--phase-matched", action="store_true",
                        help="Match control set move-number distribution to adversarial")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    cells = [int(c) for c in args.cells.split(",")]

    for cell in cells:
        pos = VALID_MOVES[cell]
        r, c = pos // 8, pos % 8
        cell_name = f"{'ABCDEFGH'[r]}{c+1}"
        print(f"\n{'='*60}", flush=True)
        print(f"Cell {cell} ({cell_name}, board pos {pos})", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        adv_feat, ctrl_feat = load_adversarial_and_control(
            args.data_dir, cell, phase_matched=args.phase_matched)

        # Screen pairs
        print("\n  --- Pair screening ---", flush=True)
        top_pairs = screen_pairs(adv_feat, ctrl_feat, cell, args.top_k)
        if top_pairs:
            print(f"\n  Top {min(10, len(top_pairs))} pairs:", flush=True)
            for p in top_pairs[:10]:
                print(f"    {p['cell_names']}: "
                      f"adv={p['adv_rate']:.3f} ctrl={p['ctrl_rate']:.3f} "
                      f"ratio={p['ratio']:.1f}x", flush=True)

        # Screen triples
        print("\n  --- Triple screening ---", flush=True)
        top_triples = screen_triples(adv_feat, ctrl_feat, cell, args.top_k)
        if top_triples:
            print(f"\n  Top {min(10, len(top_triples))} triples:", flush=True)
            for t in top_triples[:10]:
                print(f"    {t['cell_names']}: "
                      f"adv={t['adv_rate']:.3f} ctrl={t['ctrl_rate']:.3f} "
                      f"ratio={t['ratio']:.1f}x", flush=True)

        elapsed = time.time() - t0
        print(f"\n  Elapsed: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
