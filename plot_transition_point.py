"""Two-panel paired distribution plot for the L->I transition:

  Panel A: P(C) at last-legal vs first-illegal turn
  Panel B: probe CE loss around C at last-legal vs first-illegal (log-y)

Reads transition_point.csv produced by experiment_transition_point.py.

Usage:
    python plot_transition_point.py \\
        --csv transition.csv --out plots/transition.png
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    P_L, P_I, Bm_L, Bm_I, Bl_L, Bl_I = [], [], [], [], [], []
    with open(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip().split(',')
            if len(parts) < 11:
                continue
            try:
                P_L.append(float(parts[5]))
                P_I.append(float(parts[6]))
                Bm_L.append(float(parts[7]))
                Bm_I.append(float(parts[8]))
                Bl_L.append(float(parts[9]))
                Bl_I.append(float(parts[10]))
            except ValueError:
                continue
    return (np.array(P_L), np.array(P_I),
            np.array(Bm_L), np.array(Bm_I),
            np.array(Bl_L), np.array(Bl_I))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='transition.csv')
    ap.add_argument('--out', default='plots/transition.png')
    ap.add_argument('--use-loss', action='store_true', default=True,
                    help='Use CE loss (default).  If disabled, use -margin.')
    args = ap.parse_args()

    P_L, P_I, Bm_L, Bm_I, Bl_L, Bl_I = load_csv(args.csv)
    n = len(P_L)
    print(f"Loaded {n} rows from {args.csv}")
    if n == 0:
        print("No data.")
        return

    B_L = Bl_L if args.use_loss else Bm_L
    B_I = Bl_I if args.use_loss else Bm_I
    B_label = ("mean CE loss on C's ray cells" if args.use_loss
               else "mean −logit-margin on C's ray cells")

    ratio = P_I / np.maximum(P_L, 1e-12)
    sticky_frac = float((ratio > 0.5).mean())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ---- Panel A: P(C) before / after ----
    bp_A = axA.boxplot(
        [P_L, P_I],
        tick_labels=['last legal (tₘ)', 'first illegal (tᵢ)'],
        widths=0.4,
        whis=(10, 90),           # whiskers to p10/p90
        showfliers=False,
        patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 1.5},
    )
    for patch, c in zip(bp_A['boxes'], ['#a6cee3', '#fb9a99']):
        patch.set_facecolor(c)
        patch.set_edgecolor('#333333')
    m_PL, m_PI = float(np.median(P_L)), float(np.median(P_I))
    axA.plot([1, 2], [m_PL, m_PI], 'o-', color='#d1341a',
             lw=2.5, markersize=8, zorder=5)
    axA.set_ylabel('P(C)')
    axA.set_title('OGPT probability on C\nat the L→I transition')
    axA.grid(True, alpha=0.3, axis='y')
    axA.annotate(
        f'median {m_PL*100:.1f}% → {m_PI*100:.1f}%\n'
        f'{sticky_frac*100:.1f}% of positions preserve >50%',
        xy=(1.5, m_PL), xytext=(1.5, m_PL + max(0.02, m_PL * 0.4)),
        ha='center', fontsize=10, color='#333333',
        bbox=dict(boxstyle='round,pad=0.4', fc='white',
                  ec='#333333', alpha=0.9),
    )

    # ---- Panel B: B(C) before / after (log-y) ----
    bp_B = axB.boxplot(
        [B_L, B_I],
        tick_labels=['last legal (tₘ)', 'first illegal (tᵢ)'],
        widths=0.4,
        whis=(10, 90),
        showfliers=False,
        patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 1.5},
    )
    for patch, c in zip(bp_B['boxes'], ['#a6cee3', '#fb9a99']):
        patch.set_facecolor(c)
        patch.set_edgecolor('#333333')
    if args.use_loss:
        axB.set_yscale('log')
    m_BL, m_BI = float(np.median(B_L)), float(np.median(B_I))
    axB.plot([1, 2], [m_BL, m_BI], 'o-', color='#d1341a',
             lw=2.5, markersize=8, zorder=5)
    axB.set_ylabel(B_label)
    axB.set_title('Probe corruption around C\nat the L→I transition')
    axB.grid(True, alpha=0.3, axis='y', which='both')
    # Annotation placed inside the plot area, near the median line
    y_ann = m_BL * (m_BI / m_BL) ** 0.5 if args.use_loss else 0.5 * (m_BL + m_BI)
    axB.annotate(
        f'median {m_BL:.4f} → {m_BI:.4f}',
        xy=(1.5, y_ann), xytext=(1.5, y_ann),
        ha='center', fontsize=10, color='#333333',
        bbox=dict(boxstyle='round,pad=0.4', fc='white',
                  ec='#333333', alpha=0.9),
    )

    fig.suptitle(f'Transition-point analysis (n = {n:,} adversarial positions)',
                  fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
