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

    # ---- Figure X: correlation between hit rate and cross-board diversity ----
    # Diversity = 1 - avg_plurality.  If a cell has high diversity, consistent
    # boards genuinely disagree there, so the probe cannot be correct on all
    # of them.  We expect a negative correlation (or equivalently: hit_rate
    # positively correlates with plurality).
    if 'avg_plurality_uw' in d.files:
        pl_key = 'avg_plurality_w' if args.weighted else 'avg_plurality_uw'
        plurality = d[pl_key]                       # (64,)
        diversity = 1.0 - plurality
        n_pos = int(d['n_positions_plurality'])
        # Include all 64 cells.
        mask = np.ones(64, dtype=bool)
        x = diversity[mask]
        y = rate[mask]
        # Pearson + Spearman
        pear = float(np.corrcoef(x, y)[0, 1])
        rx = x.argsort().argsort()
        ry = y.argsort().argsort()
        spear = float(np.corrcoef(rx, ry)[0, 1])

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(x, y, s=40, c='#1f77b4', edgecolor='white')
        # Annotate the worst cells
        for i, on_off in enumerate(mask):
            if not on_off:
                continue
            if rate[i] < 0.75 or diversity[i] > 0.4:
                r_, c_ = i // 8, i % 8
                nm = "abcdefgh"[c_] + str(r_ + 1)
                ax.annotate(nm, (diversity[i], rate[i]),
                             xytext=(3, 3), textcoords='offset points',
                             fontsize=8, color='#333333')
        # y=1-x reference (the "optimal predictor" upper bound: probe would
        # achieve at most `plurality` if it always picked the plurality class)
        ref_x = np.linspace(0, max(x.max(), 0.7), 50)
        ax.plot(ref_x, 1.0 - ref_x, ls='--', color='#666666',
                 lw=1.0, label='hit rate = plurality  (optimal upper bound)')
        ax.set_xlabel('per-cell cross-board diversity  (= 1 - avg plurality)')
        ax.set_ylabel('per-cell probe hit rate')
        ax.set_title(f'Probe accuracy vs. cross-board disagreement\n'
                      f'k={k}, H={H}, n_positions={n_pos}, mode={label}')
        ax.text(0.02, 0.02,
                 f'Pearson r = {pear:+.3f}\nSpearman ρ = {spear:+.3f}\n'
                 f'{int(mask.sum())} cells',
                 transform=ax.transAxes, va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white',
                           ec='#333333', alpha=0.9), fontsize=9)
        ax.legend(loc='lower left', fontsize=8, bbox_to_anchor=(0, 0.15))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p3 = f'{args.out_prefix}_corr.png'
        fig.savefig(p3, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {p3}')
        print(f'  Pearson r  (diversity vs hit rate) = {pear:+.4f}')
        print(f'  Spearman rho                       = {spear:+.4f}')
    else:
        print('(no plurality data in NPZ - skip correlation figure)')

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
