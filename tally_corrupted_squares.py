"""Corruption tallies from a records .npz saved by generate_paper_numbers.py
(--corruption --save-records; needs the `corrupted_squares` array).

Writes two CSVs next to the .npz:
  <stem>_by_square.csv : for every square (flat = 8*r + c), how many
                         interventions corrupted it, at each layer (Li 4..8).
                         "Corrupted" = that square's probe reading differs from
                         the clean run at that layer. The intervened square
                         itself is never counted.
  <stem>_count_dist.csv: how many interventions corrupted 0, 1, 2, ... other
                         squares, at each layer.
and prints the last-layer summary (incl. how many have more than 6).

Usage:
    python tally_corrupted_squares.py "Intervention Results/corruption/with_dist/records_add_fixed_2.5x.npz"
"""
import csv
import os
import sys

import numpy as np

path = sys.argv[1]
d = np.load(path)
cs = d['corrupted_squares']                 # (N, n_layers, 64) bool
li_layers = [b + 1 for b in d['layer_blocks']]
n, n_layers, _ = cs.shape
assert (cs.sum(-1) == d['flips_other']).all(), 'corrupted_squares does not match flips_other'
stem = os.path.splitext(path)[0]

by_square = cs.sum(0)                       # (n_layers, 64)
with open(stem + '_by_square.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['flat', 'r', 'c'] + [f'n_corrupted_L{L}' for L in li_layers]
               + [f'pct_of_interventions_L{L}' for L in li_layers])
    for sq in range(64):
        w.writerow([sq, sq // 8, sq % 8] + [int(x) for x in by_square[:, sq]]
                   + [f'{100 * x / n:.4f}' for x in by_square[:, sq]])

counts = cs.sum(-1)                         # (N, n_layers)
mx = int(counts.max())
with open(stem + '_count_dist.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['n_other_squares_corrupted'] + [f'n_interventions_L{L}' for L in li_layers])
    for k in range(mx + 1):
        w.writerow([k] + [int((counts[:, j] == k).sum()) for j in range(n_layers)])

last = counts[:, -1]
print(f'{n} interventions; last layer = L{li_layers[-1]}')
print(f'  mean {last.mean():.3f}  median {np.median(last):g}  max {last.max()}')
print(f'  more than 6: {(last > 6).sum()} ({100 * (last > 6).mean():.2f}%)')
print('  most-corrupted squares at the last layer (flat: times, % of interventions):')
for sq in np.argsort(-by_square[-1])[:10]:
    print(f'    {sq:2d} (r{sq // 8} c{sq % 8}): {by_square[-1, sq]:7d}  {100 * by_square[-1, sq] / n:.2f}%')
print(f'Saved -> {stem}_by_square.csv and {stem}_count_dist.csv')
