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

    # ---- Figure 1: Jaccard histogram (the main analysis) ----
    if 'moveset_jaccard' in d.files:
        jac = d['moveset_jaccard']
        n_moves = len(jac)
        print(f'Per-moveset Jaccard: n={n_moves}, '
              f'mean={jac.mean():.4f}, median={np.median(jac):.4f}')

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(jac, bins=np.linspace(0, 1, 41), color='#1f77b4',
                edgecolor='white')
        ax.axvline(jac.mean(), color='#d1341a', linestyle='--', linewidth=1.5,
                    label=f'mean {jac.mean():.3f}')
        ax.axvline(np.median(jac), color='#333333', linestyle=':', linewidth=1.5,
                    label=f'median {np.median(jac):.3f}')
        ax.set_xlabel('mean pairwise Jaccard of probe error sets\n'
                       '(across consistent boards, one value per moveset)')
        ax.set_ylabel('number of movesets')
        ax.set_title(f'How overlapping are probe errors across consistent boards?\n'
                      f'k={k}, H={H}, N={N}, movesets={n_moves}')
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        p_jac = f'{args.out_prefix}_jaccard.png'
        fig.savefig(p_jac, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {p_jac}')
    else:
        print('(no moveset_jaccard in NPZ)')

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
