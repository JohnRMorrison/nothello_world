"""Plot analyses from the NPZ produced by consistent_board_analysis.py
--per-cell-npz.

Produces figures:
  - histogram of per-moveset mean pairwise Jaccard of probe error sets
    across the consistent boards (the "how overlapping are errors" plot)
  - per-cell hit rate histogram (auxiliary, distribution across cells)
  - per-cell hit rate 8x8 heatmap (auxiliary, spatial layout)

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
                    help='For the per-cell HIT-rate plots, use MC-count-weighted '
                         'hits instead of unweighted.  (Jaccard is unweighted '
                         'over board pairs by construction.)')
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    k = int(d['k']); H = int(d['hidden_dim']); N = int(d['N'])
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    # ---- Figure 1: Jaccard histogram + mean-vs-stdev scatter + multi-set overlay ----
    if 'moveset_jaccard' in d.files:
        jac_mean = d['moveset_jaccard']
        jac_std = d.get('moveset_jac_std', np.zeros_like(jac_mean))
        jac_min = d.get('moveset_jac_min', jac_mean)
        jac_max = d.get('moveset_jac_max', jac_mean)
        jac_multi = d.get('moveset_jac_multi', np.array([]))
        n_moves = len(jac_mean)
        print(f'Per-moveset stats (n={n_moves}):')
        print(f'  mean pairwise Jaccard:  mean={jac_mean.mean():.4f}  median={np.median(jac_mean):.4f}')
        print(f'  stdev pairwise Jaccard: mean={jac_std.mean():.4f}  median={np.median(jac_std):.4f}')
        print(f'  min pairwise Jaccard:   mean={jac_min.mean():.4f}  median={np.median(jac_min):.4f}')
        print(f'  max pairwise Jaccard:   mean={jac_max.mean():.4f}  median={np.median(jac_max):.4f}')
        if len(jac_multi) > 0:
            print(f'  multi-set Jaccard:      mean={jac_multi.mean():.4f}  median={np.median(jac_multi):.4f}')

        # ---- 1x3 figure: mean / min / max pairwise Jaccard, shared y-axis ----
        bins = np.linspace(0, 1, 41)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        panels = [
            (axes[0], jac_mean, 'mean pairwise Jaccard', '#1f77b4'),
            (axes[1], jac_min, 'min pairwise Jaccard (worst pair)', '#c65500'),
            (axes[2], jac_max, 'max pairwise Jaccard (best pair)', '#177245'),
        ]
        for ax, arr, lbl, col in panels:
            ax.hist(arr, bins=bins, color=col, edgecolor='white')
            ax.axvline(arr.mean(), color='#333333', linestyle=':',
                        linewidth=1.2, label=f'mean {arr.mean():.3f}')
            ax.axvline(np.median(arr), color='#111111', linestyle='--',
                        linewidth=1.2, label=f'median {np.median(arr):.3f}')
            ax.set_xlabel(lbl)
            ax.set_xlim(0, 1)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        axes[0].set_ylabel('number of movesets')
        fig.suptitle(f'Jaccard of probe error sets across consistent boards\n'
                      f'k={k}, H={H}, movesets={n_moves}',
                      fontsize=11, y=1.02)
        fig.tight_layout()
        p_jac = f'{args.out_prefix}_jaccard.png'
        fig.savefig(p_jac, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {p_jac}')

        # ---- Supplementary: stdev + mean-vs-stdev scatter + multi-set overlay ----
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ax = axes[0]
        ax.hist(jac_mean, bins=bins, color='#1f77b4', edgecolor='white',
                label='mean pairwise')
        if len(jac_multi) > 0:
            ax.hist(jac_multi, bins=bins, color='#d1341a', edgecolor='white',
                    alpha=0.55, label='multi-set (∩/∪)')
        ax.set_xlabel('Jaccard')
        ax.set_ylabel('number of movesets')
        ax.set_title('Mean pairwise + multi-set overlay')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[1]
        ax.hist(jac_std, bins=np.linspace(0, jac_std.max() * 1.1 + 0.01, 41),
                color='#2ca02c', edgecolor='white')
        ax.set_xlabel('stdev of pairwise Jaccard within moveset')
        ax.set_ylabel('number of movesets')
        ax.set_title(f'Within-moveset variance\n(median stdev = {np.median(jac_std):.3f})')
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[2]
        ax.scatter(jac_mean, jac_std, s=14, c='#1f77b4', alpha=0.5,
                    edgecolor='none')
        ax.set_xlabel('mean pairwise Jaccard')
        ax.set_ylabel('stdev pairwise Jaccard')
        ax.set_title('Mean vs stdev per moveset')
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'Supplementary: variance decomposition  '
                      f'(k={k}, H={H}, movesets={n_moves})', fontsize=11, y=1.02)
        fig.tight_layout()
        p_supp = f'{args.out_prefix}_jaccard_supp.png'
        fig.savefig(p_supp, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {p_supp}')
    else:
        print('(no moveset_jaccard in NPZ)')

    # ---- Per-moveset "how many cells are wrong on X+ boards" histograms ----
    if 'per_cell_wrong_counts' in d.files:
        wc = d['per_cell_wrong_counts']       # (n_movesets, 64) int
        n_moves = wc.shape[0]
        union_count = (wc > 0).sum(axis=1)    # cells wrong on >= 1 board
        thr3_count  = (wc >= 3).sum(axis=1)   # cells wrong on >= 3 boards
        panels = [
            ('cells wrong on ≥ 1 board (union)', union_count, '#1f77b4'),
            ('cells wrong on ≥ 3 boards', thr3_count, '#c65500'),
        ]
        # Shared x-range for comparability; use max seen so bins align
        xmax = max(union_count.max(), thr3_count.max()) if n_moves > 0 else 1
        bins = np.arange(0, xmax + 2)         # integer bins
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, (label, arr, col) in zip(axes, panels):
            ax.hist(arr, bins=bins, color=col, edgecolor='white', align='left')
            ax.axvline(arr.mean(), color='#333333', linestyle=':',
                        linewidth=1.2, label=f'mean {arr.mean():.2f}')
            ax.axvline(np.median(arr), color='#111111', linestyle='--',
                        linewidth=1.2, label=f'median {np.median(arr):.1f}')
            ax.set_xlabel(f'number of squares — {label}')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        axes[0].set_ylabel('number of movesets')
        fig.suptitle(f'Wrong squares per moveset  '
                      f'(k={k}, H={H}, movesets={n_moves})', fontsize=11, y=1.02)
        fig.tight_layout()
        p_wc = f'{args.out_prefix}_wrong_counts.png'
        fig.savefig(p_wc, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {p_wc}')
        print(f'  cells wrong on >=1 board:  mean={union_count.mean():.2f}  '
              f'median={np.median(union_count):.1f}')
        print(f'  cells wrong on >=3 boards: mean={thr3_count.mean():.2f}  '
              f'median={np.median(thr3_count):.1f}')

    # ---- Aux Figure: per-cell hit rate histogram (across 64 cells) ----
    if args.weighted:
        hits = d['hits_w']
        n = int(d['n_w'])
        label = 'MC-count-weighted'
    else:
        hits = d['hits_uw']
        n = int(d['n_uw'])
        label = 'unweighted'
    rate = hits.astype(np.float64) / max(n, 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rate, bins=np.linspace(0, 1, 41), color='#1f77b4', edgecolor='white')
    ax.axvline(rate.mean(), color='#d1341a', linestyle='--', linewidth=1.5,
                label=f'mean {rate.mean():.3f}')
    ax.axvline(np.median(rate), color='#333333', linestyle=':', linewidth=1.5,
                label=f'median {np.median(rate):.3f}')
    ax.set_xlabel(f'per-cell hit rate ({label})')
    ax.set_ylabel('number of cells (out of 64)')
    ax.set_title(f'Per-cell probe hit rate distribution\n'
                  f'k={k}, H={H}, n={n}')
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    p_hist = f'{args.out_prefix}_percell_histogram.png'
    fig.savefig(p_hist, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {p_hist}')

    # ---- Aux Figure: 8x8 heatmap of per-cell hit rates ----
    fig, ax = plt.subplots(figsize=(6, 5.5))
    grid = rate.reshape(8, 8).copy()
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
    ax.set_xticklabels(list('abcdefgh'))
    ax.set_yticklabels([str(i + 1) for i in range(8)])
    ax.set_title(f'Per-cell probe hit rate ({label})\n'
                  f'k={k}, H={H}, n={n}')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('hit rate')
    fig.tight_layout()
    p_hm = f'{args.out_prefix}_percell_heatmap.png'
    fig.savefig(p_hm, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {p_hm}')


if __name__ == '__main__':
    main()
