"""IL learning curves on the appendix's axis: P(newly legal) against the number
of TRAINING POSITIONS IN WHICH A NEW SQUARE IS LEGAL.

That axis, not games, because coherent rules sit in a new bottom row whose
neighbours fill late, while incoherent rules draw uniformly from the board and
so fire earlier and more often per game.  Counting new-square-legal positions
controls for it.

For the conv runs the count is MEASURED (newsq_legal_seen, accumulated during
training).  For Othello-GPT's fine-tune it is derived exactly: each epoch covers
the training set once, and that set was grown to hold `new_square_legal_budget`
new-square-legal positions, so cumulative = budget * step / steps_per_epoch.

Putting both on this axis is the point -- the two protocols differ hugely in
batch size and step count, and this is the unit that makes them comparable.
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'monospace'
COH, INC = '#1f77b4', '#d62728'


def conv_band(pat, key='IL_prob'):
    """mean/sd over seeds, on a common x grid of new-square-legal positions."""
    runs = [json.load(open(f)) for f in sorted(glob.glob(pat))]
    if not runs:
        return None
    x = np.array(runs[0]['newsq_legal_seen'], dtype=float)
    Y = np.array([r[key] for r in runs], dtype=float)
    X = np.array([r['newsq_legal_seen'] for r in runs], dtype=float)
    return x if X.shape[0] == 1 else X.mean(0), Y.mean(0), Y.std(0), len(runs)


def gpt_curve(path, key='IL_prob'):
    r = json.load(open(path))
    spe = r['steps_per_epoch']
    budget = r['new_square_legal_budget']
    x = np.array([s / spe * budget for s in r['eval_steps']], dtype=float)
    return x, np.array(r[key], dtype=float), r['condition_name']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conv-dir', default='experiments/new_squares/conv_pre')
    ap.add_argument('--gpt-dir', default='experiments/new_squares')
    ap.add_argument('--arm', default='board', choices=['board', 'ogpt'])
    ap.add_argument('--mlp', type=int, default=0)
    ap.add_argument('--key', default='IL_prob')
    ap.add_argument('--out', default='experiments/new_squares/il_conv_vs_gpt.png')
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    # Othello-GPT's own fine-tune -- dashed, and n=1 is stated, because this
    # experiment flips sign between runs of an identical config.
    for cid, c in ((2, COH), (3, INC)):
        p = os.path.join(a.gpt_dir, f'gpt_cond_{cid:03d}.json')
        if not os.path.exists(p):
            continue
        x, y, nm = gpt_curve(p, a.key)
        ax.plot(x, y, ls='--', marker='s', ms=3, lw=1.2, color=c, alpha=0.75,
                label=f'Othello-GPT, {nm.replace("_all","")} (n=1)')

    # conv readout -- solid, mean +/- sd over seeds
    for cid, c, nm in ((2, COH, 'coherent'), (3, INC, 'incoherent')):
        r = conv_band(os.path.join(
            a.conv_dir, f'{a.arm}_s*_cond{cid}_mlp{a.mlp}.json'), a.key)
        if r is None:
            continue
        x, m, s, n = r
        ax.plot(x, m, marker='o', ms=4, lw=1.8, color=c,
                label=f'conv readout, {nm} (n={n})')
        if n > 1:
            ax.fill_between(x, m - s, m + s, color=c, alpha=0.18, lw=0)
        ax.annotate(f'{m[-1]:.3f}', (x[-1], m[-1]), color=c, fontsize=9,
                    xytext=(5, 0), textcoords='offset points', va='center')

    ax.set_xscale('log')
    ax.set_xlabel('training positions where a new square is legal')
    ax.set_ylabel('P(newly legal)' if a.key == 'IL_prob' else a.key)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor='white')
    fig.savefig(a.out.replace('.png', '.pdf'), facecolor='white')
    print(f'saved {a.out} (+ .pdf)')


if __name__ == '__main__':
    main()
