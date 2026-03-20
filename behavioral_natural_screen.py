"""Screen triples using NATURAL adversarial positions only (no beam search).

Streams shards to avoid OOM. Phase-matched control.

Usage:
    python behavioral_natural_screen.py --cells 10,20,30,40,50 --data-dir behavioral_data
"""

import argparse
import os
import time
import numpy as np
from itertools import combinations
from behavioral_utils import N_MOVES, VALID_MOVES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=str, default="10,20,30,40,50")
    parser.add_argument("--data-dir", type=str, default="behavioral_data")
    parser.add_argument("--n-shards", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    cells = [int(c) for c in args.cells.split(",")]

    shard_files = sorted([f for f in os.listdir(args.data_dir)
                          if f.startswith("shard_") and f.endswith(".npz")])
    shard_files = shard_files[:args.n_shards]

    for cell in cells:
        pos = VALID_MOVES[cell]
        name = chr(65 + pos // 8) + str(pos % 8 + 1)
        print(f"\n{'=' * 60}", flush=True)
        print(f"Cell {cell} ({name}, board pos {pos})", flush=True)
        print(f"{'=' * 60}", flush=True)

        t0 = time.time()

        # Stream shards, keep only occupancy (bool) to save memory
        adv_occ_list, adv_moves_list = [], []
        ctrl_occ_list, ctrl_moves_list = [], []

        for shard_file in shard_files:
            data = np.load(os.path.join(args.data_dir, shard_file))
            X = data['features'].astype(np.float32)
            p = data['probs'].astype(np.float32)[:, cell]
            l = data['legal'][:, cell]

            occ = X[:, :N_MOVES] > 0  # (n, 60) bool
            moves = occ.sum(axis=1).astype(np.int8)

            adv_mask = (l == 0) & (p > 0.005)
            ctrl_mask = (l == 0) & (p < 0.001)

            if adv_mask.any():
                adv_occ_list.append(occ[adv_mask])
                adv_moves_list.append(moves[adv_mask])
            if ctrl_mask.any():
                ctrl_occ_list.append(occ[ctrl_mask])
                ctrl_moves_list.append(moves[ctrl_mask])

            del data, X, p, l, occ, moves

        adv_occ = np.concatenate(adv_occ_list)
        adv_moves = np.concatenate(adv_moves_list)
        ctrl_occ_all = np.concatenate(ctrl_occ_list)
        ctrl_moves_all = np.concatenate(ctrl_moves_list)

        del adv_occ_list, adv_moves_list, ctrl_occ_list, ctrl_moves_list

        print(f"  Natural adv: {len(adv_occ)}, Control pool: {len(ctrl_occ_all)}",
              flush=True)
        print(f"  Adv move range: {adv_moves.min()}-{adv_moves.max()}, "
              f"mean={adv_moves.mean():.1f}", flush=True)

        # Phase match: sample control to match adv move distribution
        selected = []
        for mv in range(60):
            adv_at = int((adv_moves == mv).sum())
            if adv_at == 0:
                continue
            ctrl_at = np.where(ctrl_moves_all == mv)[0]
            if len(ctrl_at) == 0:
                continue
            n = min(len(ctrl_at), adv_at * 15)
            chosen = np.random.choice(ctrl_at, n, replace=False)
            selected.extend(chosen.tolist())

        ctrl_occ = ctrl_occ_all[selected]
        ctrl_moves = ctrl_moves_all[selected]
        del ctrl_occ_all, ctrl_moves_all

        print(f"  Phase-matched control: {len(ctrl_occ)}, "
              f"move mean={ctrl_moves.mean():.1f}", flush=True)

        # Screen pairs
        other = [i for i in range(N_MOVES) if i != cell]
        pair_results = []
        for a, b in combinations(other, 2):
            af = float((adv_occ[:, a] & adv_occ[:, b]).mean())
            if af < 0.05:
                continue
            cf = float((ctrl_occ[:, a] & ctrl_occ[:, b]).mean())
            ratio = af / max(cf, 1e-6)
            if ratio > 1.3:
                pair_results.append((a, b, af, cf, ratio))

        pair_results.sort(key=lambda x: x[4], reverse=True)
        print(f"\n  Pairs with ratio > 1.3: {len(pair_results)}", flush=True)
        for a, b, af, cf, ratio in pair_results[:10]:
            na = chr(65 + VALID_MOVES[a] // 8) + str(VALID_MOVES[a] % 8 + 1)
            nb = chr(65 + VALID_MOVES[b] // 8) + str(VALID_MOVES[b] % 8 + 1)
            print(f"    ({na}, {nb}): adv={af:.3f} ctrl={cf:.3f} ratio={ratio:.2f}x",
                  flush=True)

        # Screen triples
        triple_results = []
        for a, b, c in combinations(other, 3):
            af = float((adv_occ[:, a] & adv_occ[:, b] & adv_occ[:, c]).mean())
            if af < 0.05:
                continue
            cf = float((ctrl_occ[:, a] & ctrl_occ[:, b] & ctrl_occ[:, c]).mean())
            ratio = af / max(cf, 1e-6)
            if ratio > 1.3:
                triple_results.append((a, b, c, af, cf, ratio))

        triple_results.sort(key=lambda x: x[5], reverse=True)
        print(f"\n  Triples with ratio > 1.3: {len(triple_results)}", flush=True)
        for a, b, c, af, cf, ratio in triple_results[:10]:
            na = chr(65 + VALID_MOVES[a] // 8) + str(VALID_MOVES[a] % 8 + 1)
            nb = chr(65 + VALID_MOVES[b] // 8) + str(VALID_MOVES[b] % 8 + 1)
            nc = chr(65 + VALID_MOVES[c] // 8) + str(VALID_MOVES[c] % 8 + 1)
            print(f"    ({na}, {nb}, {nc}): adv={af:.3f} ctrl={cf:.3f} "
                  f"ratio={ratio:.2f}x", flush=True)

        elapsed = time.time() - t0
        print(f"\n  Elapsed: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
