"""For late-k multisets that the database method labels as "1 board observed"
(because only 1 game in 20M has that multiset), determine whether the
multiset is truly consistent with only 1 board, or whether it COULD admit
multiple boards — we just never saw the alternative orderings.

Approach: biased Monte Carlo.  For each sampled multiset:
  1. Start with empty Othello board
  2. At each step k, current parity = k % 2
  3. Available cells = (parity-k%2 cells in M) ∩ (legal moves at current pos)
  4. Sample uniformly from available, play
  5. Continue until all multiset cells played -> record final board hash
By construction every completed trial is a valid Othello ordering.

Repeat N=1000 trials per multiset, count distinct boards reached.  If the
count is 1, the multiset is truly order-redundant.  If >1, the database's
"1 board" label was an artifact of having only one game in the dataset.

Usage:
    python analyze_late_ambiguity.py --positions 15 20 30 40 50 \\
        --num-multisets 200 --samples-per-multiset 1000
"""
import argparse
import csv
import multiprocessing as mp
import os
import pickle
import random
import sys
from collections import Counter

import numpy as np
from tqdm import tqdm

sys.path.insert(0, '.')
from data.othello import OthelloBoardState


def load_games(data_dir='./data/othello_synthetic', num_files=1):
    files = sorted(os.listdir(data_dir))
    out = []
    for fname in files[-num_files:]:
        with open(os.path.join(data_dir, fname), 'rb') as f:
            batch = pickle.load(f)
        if len(batch) >= 9e4:
            out.extend(batch)
    return out


def sample_one_valid_ordering(p0_cells, p1_cells):
    """Sample a valid Othello ordering of the (p0_cells, p1_cells) multiset
    via uniform-random choice at each step among legal x remaining.
    Returns (valid, board_state_bytes).  valid=False on dead-end."""
    remaining_p0 = list(p0_cells)
    remaining_p1 = list(p1_cells)
    board = OthelloBoardState()
    step = 0
    while remaining_p0 or remaining_p1:
        parity = step % 2
        cands = remaining_p0 if parity == 0 else remaining_p1
        legal = set(board.get_valid_moves())
        available = [c for c in cands if c in legal]
        if not available:
            return False, None
        chosen = random.choice(available)
        board.update([chosen])
        if parity == 0:
            remaining_p0.remove(chosen)
        else:
            remaining_p1.remove(chosen)
        step += 1
    return True, board.state.tobytes()


def measure_multiset(multiset, n_samples, max_dead_end_retries=5):
    """For a given (p0_cells, p1_cells) multiset, return:
        n_valid_trials, n_dead_ends, board_counts
    where board_counts is a Counter {board-state bytes: n_times_sampled}.
    The sample frequency approximates P(board | moveset) because biased-MC
    uniform-random-legal sampling matches the synthetic data's generation."""
    p0_cells, p1_cells = multiset
    board_counts = Counter()
    n_valid = 0
    n_dead = 0
    for _ in range(n_samples):
        for _ in range(max_dead_end_retries + 1):
            valid, board = sample_one_valid_ordering(p0_cells, p1_cells)
            if valid:
                board_counts[board] += 1
                n_valid += 1
                break
            else:
                n_dead += 1
    return n_valid, n_dead, board_counts


def max_decoder_accuracy(board_counts, state_dtype):
    """Best per-cell board-state accuracy an ORDER-BLIND decoder can reach on
    this moveset.  A moveset fixes which cells are occupied; only colours are
    ambiguous.  For each of the 64 cells the optimal decoder predicts the
    frequency-weighted MODE value across the consistent boards, so its accuracy
    on that cell is the mode's probability.  Return the mean over 64 cells.
    board_counts: Counter {board bytes: n_times_sampled}."""
    if not board_counts:
        return float('nan')
    boards = np.stack([np.frombuffer(b, dtype=state_dtype).reshape(64)
                       for b in board_counts])            # (D, 64)
    weights = np.array([board_counts[b] for b in board_counts], dtype=float)
    total = weights.sum()
    acc = 0.0
    for c in range(64):
        vals = boards[:, c]
        best = 0.0
        for v in np.unique(vals):
            w = weights[vals == v].sum()
            if w > best:
                best = w
        acc += best / total
    return acc / 64.0


def _measure_task(task):
    """Worker: seed deterministically, then measure one moveset.
    task = (task_seed, p0, p1, n_samples, max_retries)."""
    task_seed, p0, p1, n_samples, max_retries = task
    random.seed(task_seed)
    return measure_multiset((p0, p1), n_samples, max_dead_end_retries=max_retries)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-multisets', type=int, default=200,
                   help='How many random multisets to test per k')
    p.add_argument('--samples-per-multiset', type=int, default=1000)
    p.add_argument('--positions', type=int, nargs='+',
                   default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    p.add_argument('--num-pickle-files', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-csv', default=None,
                   help='Per-k summary CSV: mean/median/p90/max distinct boards.')
    p.add_argument('--examples-pkl', default=None,
                   help='Save example movesets + their consistent board states '
                        '(for diversity plots).')
    p.add_argument('--n-examples', type=int, default=3,
                   help='Example movesets to save per k (the MOST diverse).')
    p.add_argument('--max-example-boards', type=int, default=25,
                   help='Cap on distinct boards saved per example moveset.')
    p.add_argument('--per-moveset-csv', default=None,
                   help='Record EVERY moveset: k, n_valid, n_distinct, and '
                        'max_decoder_acc (the per-moveset decoder ceiling).')
    p.add_argument('--max-dead-end-retries', type=int, default=5)
    p.add_argument('--workers', type=int, default=0,
                   help='Parallel worker processes over movesets. '
                        '0 = all cores (CPU-only job; no GPU used); 1 = serial.')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    state_dtype = OthelloBoardState().state.dtype
    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    pool = mp.Pool(n_workers) if n_workers > 1 else None
    print(f"workers: {n_workers}  (CPU-only; the GPU is not used)")

    def to_board(b):
        return np.frombuffer(b, dtype=state_dtype).reshape(8, 8).astype(np.int8)

    print(f"Loading games...")
    games = load_games(num_files=args.num_pickle_files)
    print(f"  {len(games)} games loaded\n")

    summary_rows = []
    moveset_rows = []          # one row per moveset (all k)
    examples = {}
    for k in args.positions:
        eligible = [g for g in games if len(g) > k]
        sample_games = random.sample(eligible,
                                      min(args.num_multisets, len(eligible)))
        print(f"k={k}: sampling {args.samples_per_multiset} valid orderings of "
              f"each of {len(sample_games)} multisets...")

        distinct_counts = []
        valid_trials = []
        dead_end_rates = []
        dec_accs = []            # max_decoder_acc per moveset
        per_k = []   # (n_distinct, prefix, p0, p1, board_counts, mdacc)

        # Build one task per moveset, each with a deterministic seed so results
        # are reproducible regardless of --workers.
        prefixes, tasks = [], []
        for idx, game in enumerate(sample_games):
            prefix = list(game[:k])
            p0 = tuple(prefix[0::2])
            p1 = tuple(prefix[1::2])
            seed_i = (args.seed * 1_000_003 + k * 10_007 + idx) % (2 ** 32)
            prefixes.append((prefix, p0, p1))
            tasks.append((seed_i, p0, p1, args.samples_per_multiset,
                          args.max_dead_end_retries))

        if pool is not None:
            results = list(tqdm(pool.imap(_measure_task, tasks, chunksize=1),
                                total=len(tasks), desc=f"k={k}", leave=False))
        else:
            results = [_measure_task(t)
                       for t in tqdm(tasks, desc=f"k={k}", leave=False)]

        for idx, ((n_valid, n_dead, board_counts), (prefix, p0, p1)) in \
                enumerate(zip(results, prefixes)):
            n_distinct = len(board_counts)
            mdacc = max_decoder_accuracy(board_counts, state_dtype)
            distinct_counts.append(n_distinct)
            valid_trials.append(n_valid)
            dead_end_rates.append(n_dead / max(1, n_dead + n_valid))
            dec_accs.append(mdacc)
            moveset_rows.append({
                'k': k, 'moveset_idx': idx, 'n_valid': n_valid,
                'n_distinct': n_distinct,
                'max_decoder_acc': round(float(mdacc), 6),
            })
            per_k.append((n_distinct, prefix, list(p0), list(p1),
                          board_counts, mdacc))

        arr = np.array(distinct_counts)
        valid_arr = np.array(valid_trials)
        dead_arr = np.array(dead_end_rates)
        dacc_arr = np.array(dec_accs)

        print(f"\nk={k} summary (n={len(arr)} multisets, "
              f"{args.samples_per_multiset} sample attempts each):")
        print(f"  trials per multiset that completed validly: "
              f"mean {valid_arr.mean():.0f}  median {np.median(valid_arr):.0f}")
        print(f"  dead-end rate: mean {dead_arr.mean()*100:.1f}%  "
              f"max {dead_arr.max()*100:.1f}%")
        print(f"  distinct boards per multiset:")
        print(f"    mean {arr.mean():.2f}  median {np.median(arr):.0f}  "
              f"p90 {np.percentile(arr, 90):.0f}  max {arr.max()}")
        print(f"  max decoder accuracy per multiset:")
        print(f"    mean {dacc_arr.mean():.4f}  median {np.median(dacc_arr):.4f}  "
              f"min {dacc_arr.min():.4f}")
        print(f"  distinct-boards distribution:")
        cnt = Counter(arr.tolist())
        for nb in sorted(cnt):
            n_sets = cnt[nb]
            pct = 100 * n_sets / len(arr)
            print(f"    {nb:>3} boards: {n_sets:>4} multisets ({pct:5.1f}%)")
        print()

        summary_rows.append({
            'k': k, 'n_multisets': len(arr),
            'samples_per_multiset': args.samples_per_multiset,
            'mean_distinct': round(float(arr.mean()), 4),
            'median_distinct': float(np.median(arr)),
            'p90_distinct': float(np.percentile(arr, 90)),
            'max_distinct': int(arr.max()),
            'mean_max_decoder_acc': round(float(dacc_arr.mean()), 6),
            'median_max_decoder_acc': round(float(np.median(dacc_arr)), 6),
            'min_max_decoder_acc': round(float(dacc_arr.min()), 6),
            'mean_valid_trials': round(float(valid_arr.mean()), 1),
            'mean_dead_end_rate': round(float(dead_arr.mean()), 4),
        })

        # Keep the most-diverse example movesets (+ their board arrays) for k.
        per_k.sort(key=lambda t: t[0], reverse=True)
        ex_list = []
        for n_distinct, prefix, p0, p1, board_counts, mdacc in \
                per_k[:args.n_examples]:
            items = board_counts.most_common(args.max_example_boards)
            boards = [to_board(b) for b, _ in items]
            counts = [c for _, c in items]
            ex_list.append({
                'k': k, 'n_distinct': n_distinct,
                'max_decoder_acc': float(mdacc),
                'moveset_prefix': prefix,   # one valid ordering (the played game)
                'p0_cells': p0, 'p1_cells': p1,
                'boards': boards,           # list of (8, 8) int8 arrays
                'board_counts': counts,     # sample frequency of each board
            })
        examples[k] = ex_list

    if pool is not None:
        pool.close()
        pool.join()

    # Overall aggregate across ALL sampled movesets (e.g. a 5-54 estimate when
    # --positions spans that range).  Each moveset weighted equally.
    all_acc = np.array([r['max_decoder_acc'] for r in moveset_rows], dtype=float)
    all_dist = np.array([r['n_distinct'] for r in moveset_rows], dtype=float)
    all_valid = np.array([r['n_valid'] for r in moveset_rows], dtype=float)
    kmin, kmax = min(args.positions), max(args.positions)
    print(f"=== OVERALL (positions {kmin}-{kmax}, N={len(all_acc)} movesets) ===")
    print(f"  mean max_decoder_acc = {all_acc.mean():.4f}  "
          f"(median {np.median(all_acc):.4f}, min {all_acc.min():.4f})")
    print(f"  mean distinct boards = {all_dist.mean():.2f}\n")
    summary_rows.append({
        'k': 'overall', 'n_multisets': len(all_acc),
        'samples_per_multiset': args.samples_per_multiset,
        'mean_distinct': round(float(all_dist.mean()), 4),
        'median_distinct': float(np.median(all_dist)),
        'p90_distinct': float(np.percentile(all_dist, 90)),
        'max_distinct': int(all_dist.max()),
        'mean_max_decoder_acc': round(float(all_acc.mean()), 6),
        'median_max_decoder_acc': round(float(np.median(all_acc)), 6),
        'min_max_decoder_acc': round(float(all_acc.min()), 6),
        'mean_valid_trials': round(float(all_valid.mean()), 1),
        'mean_dead_end_rate': '',
    })

    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"saved summary CSV -> {args.out_csv}")

    if args.per_moveset_csv:
        with open(args.per_moveset_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(moveset_rows[0].keys()))
            w.writeheader()
            w.writerows(moveset_rows)
        print(f"saved per-moveset CSV -> {args.per_moveset_csv}  "
              f"({len(moveset_rows)} movesets)")

    if args.examples_pkl:
        with open(args.examples_pkl, 'wb') as f:
            pickle.dump(examples, f)
        n_ex = sum(len(v) for v in examples.values())
        print(f"saved examples -> {args.examples_pkl}  ({n_ex} example movesets, "
              f"each with its distinct board states)")


if __name__ == '__main__':
    main()
