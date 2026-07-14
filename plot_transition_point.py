"""Plots for the L->I transition analysis.  Reads transition.csv produced
by experiment_transition_point.py and produces up to three figures:

  1. persistence  -- histogram of P_I / P_L (fraction of P retained)
  2. absolute     -- 2-panel boxplot of P and B_loss before/after
  3. change       -- 2-panel boxplot of (P_I - P_L) and (B_loss_I - B_loss_L)

Pass --kind {persistence,absolute,change,all} (default: all).  With --kind
all, files land at <out_prefix>_persistence.png, <out_prefix>_absolute.png,
<out_prefix>_change.png.

Usage:
    python plot_transition_point.py --csv transition.csv --out-prefix plots/transition
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    P_L, P_I, Bm_L, Bm_I, Bl_L, Bl_I = [], [], [], [], [], []
    with open(path) as f:
        f.readline()
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


COLOR_L = '#a6cee3'   # last-legal (before)
COLOR_I = '#fb9a99'   # first-illegal (after)
COLOR_MED = '#d1341a'


def plot_persistence(P_L, P_I, out_path):
    ratio = P_I / np.maximum(P_L, 1e-12)
    ratio = np.clip(ratio, 0, 1.5)   # cap tail; annotate the overflow

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1.5, 31)
    counts, edges, _ = ax.hist(
        ratio, bins=bins, color='#1f77b4', edgecolor='white',
        weights=100 * np.ones_like(ratio) / len(ratio),
    )
    ax.axvline(1.0, color='#333333', linestyle='--', linewidth=1,
               label='no drop (P_I = P_L)')
    ax.axvline(0.5, color=COLOR_MED, linestyle=':', linewidth=1.5,
               label='>½ retained')
    m = float(np.median(ratio))
    ax.axvline(m, color='black', linewidth=1.5)
    frac_half = float((P_I / np.maximum(P_L, 1e-12) > 0.5).mean())
    ax.set_xlabel('persistence: P(C, first illegal) / P(C, last legal)')
    ax.set_ylabel('% of adversarial positions')
    ax.set_title('How much probability does OGPT retain\n'
                  'when C flips from legal to illegal?')
    ax.text(0.02, 0.95,
            f'n = {len(ratio):,}\n'
            f'median = {m:.2f}\n'
            f'>½ retained: {frac_half*100:.1f}%',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white',
                      ec='#333333', alpha=0.9))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_path}")


def _paired_box(ax, before, after, before_label, after_label,
                 y_label, title, log_y=False, med_fmt='{:.3f}'):
    bp = ax.boxplot(
        [before, after],
        tick_labels=[before_label, after_label],
        widths=0.4,
        whis=(10, 90),
        showfliers=False,
        patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 1.5},
    )
    for patch, c in zip(bp['boxes'], [COLOR_L, COLOR_I]):
        patch.set_facecolor(c)
        patch.set_edgecolor('#333333')
    if log_y:
        ax.set_yscale('log')
    m_b = float(np.median(before))
    m_a = float(np.median(after))
    ax.plot([1, 2], [m_b, m_a], 'o-', color=COLOR_MED, lw=2.5,
             markersize=8, zorder=5)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y', which='both' if log_y else 'major')
    y_ann = (m_b * (m_a / max(m_b, 1e-12)) ** 0.5) if log_y else 0.5 * (m_b + m_a)
    ax.annotate(
        f'median {med_fmt.format(m_b)} → {med_fmt.format(m_a)}',
        xy=(1.5, y_ann), xytext=(1.5, y_ann),
        ha='center', fontsize=10, color='#333333',
        bbox=dict(boxstyle='round,pad=0.4', fc='white',
                  ec='#333333', alpha=0.9),
    )


def plot_absolute(P_L, P_I, B_L, B_I, B_label, out_path):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.5))
    _paired_box(axA, P_L, P_I, 'last legal (tₘ)', 'first illegal (tᵢ)',
                'P(C)', 'OGPT probability on C\nat the L→I transition',
                log_y=False, med_fmt='{:.3f}')
    _paired_box(axB, B_L, B_I, 'last legal (tₘ)', 'first illegal (tᵢ)',
                B_label, 'Probe corruption around C\nat the L→I transition',
                log_y=True, med_fmt='{:.4f}')
    fig.suptitle(f'Before/after values at the transition (n = {len(P_L):,})',
                  fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_change(P_L, P_I, B_L, B_I, B_label, out_path):
    dP = P_I - P_L
    dB = B_I - B_L
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.5))

    bpA = axA.boxplot([dP], widths=0.4, whis=(10, 90), showfliers=False,
                       patch_artist=True,
                       medianprops={'color': 'black', 'linewidth': 1.5})
    bpA['boxes'][0].set_facecolor('#c6dbef')
    bpA['boxes'][0].set_edgecolor('#333333')
    axA.axhline(0, color='#333333', linewidth=1, linestyle='--')
    axA.set_xticks([1])
    axA.set_xticklabels(['ΔP(C) = P_I − P_L'])
    axA.set_ylabel('change in P(C)')
    axA.set_title('Change in OGPT probability on C')
    axA.grid(True, alpha=0.3, axis='y')
    axA.annotate(f'median = {np.median(dP):+.4f}\n'
                  f'p25..p75 = {np.percentile(dP, 25):+.4f} .. '
                  f'{np.percentile(dP, 75):+.4f}',
                  xy=(1.15, np.median(dP)),
                  ha='left', va='center', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.4', fc='white',
                            ec='#333333', alpha=0.9))

    bpB = axB.boxplot([dB], widths=0.4, whis=(10, 90), showfliers=False,
                       patch_artist=True,
                       medianprops={'color': 'black', 'linewidth': 1.5})
    bpB['boxes'][0].set_facecolor('#fdd0a2')
    bpB['boxes'][0].set_edgecolor('#333333')
    axB.axhline(0, color='#333333', linewidth=1, linestyle='--')
    axB.set_xticks([1])
    axB.set_xticklabels(['ΔB(C) = B_I − B_L'])
    axB.set_ylabel(f'change in {B_label}')
    axB.set_title('Change in probe corruption around C')
    axB.grid(True, alpha=0.3, axis='y')
    axB.annotate(f'median = {np.median(dB):+.4f}\n'
                  f'p25..p75 = {np.percentile(dB, 25):+.4f} .. '
                  f'{np.percentile(dB, 75):+.4f}',
                  xy=(1.15, np.median(dB)),
                  ha='left', va='center', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.4', fc='white',
                            ec='#333333', alpha=0.9))

    fig.suptitle(f'Per-position change at the transition (n = {len(P_L):,})',
                  fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='transition.csv')
    ap.add_argument('--out-prefix', default='plots/transition')
    ap.add_argument('--kind', choices=['persistence', 'absolute',
                                         'change', 'all'],
                    default='all')
    ap.add_argument('--metric', choices=['loss', 'margin'], default='loss',
                    help='B metric: CE loss (default) or -logit margin.')
    args = ap.parse_args()

    P_L, P_I, Bm_L, Bm_I, Bl_L, Bl_I = load_csv(args.csv)
    print(f"Loaded {len(P_L)} rows from {args.csv}")
    if len(P_L) == 0:
        print("No data.")
        return

    if args.metric == 'loss':
        B_L, B_I = Bl_L, Bl_I
        B_label = "mean CE loss on C's ray cells"
    else:
        B_L, B_I = Bm_L, Bm_I
        B_label = "mean −logit margin on C's ray cells"

    kinds = ['persistence', 'absolute', 'change'] if args.kind == 'all' else [args.kind]
    for k in kinds:
        out = f"{args.out_prefix}_{k}.png"
        if k == 'persistence':
            plot_persistence(P_L, P_I, out)
        elif k == 'absolute':
            plot_absolute(P_L, P_I, B_L, B_I, B_label, out)
        elif k == 'change':
            plot_change(P_L, P_I, B_L, B_I, B_label, out)


if __name__ == '__main__':
    main()
