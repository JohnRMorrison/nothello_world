"""Plot IL (new-square legality) learning curves for the new-squares transfer
experiment: coherent vs incoherent.  Reads gpt_cond_000.json (coherent) and
gpt_cond_001.json (incoherent) from --dir."""
import argparse, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='experiments/new_squares')
    ap.add_argument('--out', default='experiments/new_squares/il_curves.png')
    ap.add_argument('--xscale', default='log', choices=['log', 'linear'],
                    help='x-axis scale for the vs-step panel.')
    args = ap.parse_args()

    coh = json.load(open(os.path.join(args.dir, 'gpt_cond_000.json')))
    inc = json.load(open(os.path.join(args.dir, 'gpt_cond_001.json')))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    styles = [(coh, '#1f77b4', 'coherent (9th row)'),
              (inc, '#d62728', 'incoherent (random)')]

    # Panel A: vs training step (log x) -- the raw learning curves
    ax = axes[0]
    for r, c, lab in styles:
        steps = ([max(s, 1) for s in r['eval_steps']]
                 if args.xscale == 'log' else r['eval_steps'])
        ax.plot(steps, r['IL_prob'], marker='o', ms=4, color=c, label=lab)
        ax.annotate(f"{r['IL_prob'][-1]:.3f}", (steps[-1], r['IL_prob'][-1]),
                    color=c, fontsize=9, xytext=(4, 0),
                    textcoords='offset points', va='center')
    ax.set_xscale(args.xscale)
    ax.set_xlabel(f'training step ({args.xscale})')
    ax.set_ylabel('IL: prob. on legal new-square move')
    ax.set_title('IL learning curve vs step')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which='both', alpha=0.25)

    # Panel B: vs epoch-fraction (linear) -- matched-exposure view (Control 1)
    ax = axes[1]
    for r, c, lab in styles:
        frac = [s / r['total_steps'] for s in r['eval_steps']]
        ax.plot(frac, r['IL_prob'], marker='o', ms=4, color=c, label=lab)
    ax.set_xlabel('fraction of training (matched new-square exposure)')
    ax.set_ylabel('IL prob.')
    ax.set_title('IL vs matched exposure')
    ax.grid(True, alpha=0.25)

    fig.suptitle('New-square legality: coherent vs incoherent  '
                 f"(retention std_lpm: coherent {coh['std_lpm'][-1]:.3f}, "
                 f"incoherent {inc['std_lpm'][-1]:.3f})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
