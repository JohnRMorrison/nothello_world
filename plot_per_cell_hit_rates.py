"""Plot per-cell probe hit rates from the NPZ produced by
consistent_board_analysis.py --per-cell-npz.

Produces two figures:
  - Histogram of the 64 per-cell hit rates: shows whether failures are
    concentrated at a few cells (bimodal / long left tail) or spread
    across many cells (unimodal near the aggregate).
  - 8x8 heatmap of per-cell hit rates: shows which specific cells are
    hardest for the probe.

Usage:
    python plot_per_cell_hit_rates.py --npz per_cell_hits_k25_H512.npz \\
        --out-prefix plots/per_cell_H512
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


CENTER_CELLS = {27, 28, 35, 36}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--weighted', action='store_true',
                    help='Use MC-count-weighted hits instead of unweighted.')
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    if args.weighted:
        hits = d['hits_w']
        n = int(d['n_w'])
        label = 'MC-count-weighted'
    else:
        hits = d['hits_uw']
        n = int(d['n_uw'])
        label = 'unweighted'
    k = int(d['k']); H = int(d['hidden_dim']); N = int(d['N'])
    rate = hits.astype(np.float64) / max(n, 1)   # (64,)

    print(f'Loaded {args.npz}: k={k}, H={H}, N={N}, n_rows={n}, mode={label}')
    print(f'Per-cell hit rate summary:')
    print(f'  mean={rate.mean():.4f}  median={np.median(rate):.4f}')
    print(f'  min={rate.min():.4f}  max={rate.max():.4f}')
    # Show the 5 worst cells
    order = rate.argsort()
    print(f'  5 worst cells (cell_idx: rate):')
    for c in order[:5]:
        r_, c_ = c // 8, c % 8
        cell_name = "abcdefgh"[c_] + str(r_ + 1)
        marker = " (center)" if c in CENTER_CELLS else ""
        print(f'    {int(c):>2} ({cell_name}): {rate[c]:.4f}{marker}')

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    # ---- Figure 1: histogram of per-cell hit rates ----
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(rate, bins=bins, color='#1f77b4', edgecolor='white')
    ax.axvline(rate.mean(), color='#d1341a', linestyle='--',
                linewidth=1.5, label=f'mean {rate.mean():.3f}')
    ax.axvline(np.median(rate), color='#333333', linestyle=':',
                linewidth=1.5, label=f'median {np.median(rate):.3f}')
    ax.set_xlabel(f'per-cell hit rate ({label})')
    ax.set_ylabel('number of cells (out of 64)')
    ax.set_title(f'Distribution of per-cell probe hit rates '
                  f'(k={k}, H={H}, N={N}, n={n})')
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    p1 = f'{args.out_prefix}_histogram.png'
    fig.savefig(p1, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {p1}')

    # ---- Figure 2: 8x8 heatmap of hit rates ----
    fig, ax = plt.subplots(figsize=(6, 5.5))
    grid = rate.reshape(8, 8)
    # Mask the 4 center cells (they're never targets)
    for c in CENTER_CELLS:
        grid[c // 8, c % 8] = np.nan
    im = ax.imshow(grid, vmin=0, vmax=1, cmap='RdYlGn', origin='upper')
    for r in range(8):
        for c in range(8):
            cell_i = r * 8 + c
            if cell_i in CENTER_CELLS:
                ax.text(c, r, '·', ha='center', va='center',
                         fontsize=14, color='#333333')
            else:
                v = grid[r, c]
                text_color = 'black' if 0.35 < v < 0.85 else 'white'
                ax.text(c, r, f'{v:.2f}', ha='center', va='center',
                         fontsize=8, color=text_color)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(list("abcdefgh"))
    ax.set_yticklabels([str(i + 1) for i in range(8)])
    ax.set_title(f'Per-cell probe hit rate ({label})\n'
                  f'k={k}, H={H}, n={n}')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('hit rate')
    fig.tight_layout()
    p2 = f'{args.out_prefix}_heatmap.png'
    fig.savefig(p2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {p2}')


if __name__ == '__main__':
    main()
