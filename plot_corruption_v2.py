"""Plot corruption v2 (rule-based) fine-tuning loss curves."""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

LOSS_DIR = "experiments/corruption_v2/losses_100k"
OUT_DIR = "experiments/corruption_v2/figures"

ALPHAS = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

def load_all():
    results = {}
    for f in sorted(glob.glob(os.path.join(LOSS_DIR, "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        results[d['label']] = d
    return results

def smooth(losses, window=50):
    kernel = np.ones(window) / window
    return np.convolve(losses, kernel, mode='valid')

def plot_loss_curves(results):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.viridis

    for alpha in ALPHAS:
        alpha_str = f"{int(alpha * 100):03d}"
        label = f"alpha{alpha_str}"
        if label not in results:
            continue
        losses = results[label]['losses']
        smoothed = smooth(losses)
        color = cmap(norm(alpha))
        ax.plot(smoothed, color=color, linewidth=1.2, alpha=0.9,
                label=f"α={alpha}")

    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss (smoothed, window=50)")
    ax.set_title("OthelloGPT fine-tuning on rule-corrupted games (100K)")
    ax.set_ylim(1.5, 3.5)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02, label="Alpha (corruption level)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "v2_loss_curves.png"), dpi=150, bbox_inches='tight')
    print(f"Saved {OUT_DIR}/v2_loss_curves.png")

def plot_initial_vs_final(results):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    alphas_plot = []
    initials = []
    finals = []
    for alpha in ALPHAS:
        alpha_str = f"{int(alpha * 100):03d}"
        label = f"alpha{alpha_str}"
        if label not in results:
            continue
        losses = results[label]['losses']
        alphas_plot.append(alpha)
        initials.append(np.mean(losses[:10]))
        finals.append(np.mean(losses[-100:]))

    ax.plot(alphas_plot, initials, 'o--', label='Initial (batch 0-9)', color='red',
            linewidth=1.5, markersize=6)
    ax.plot(alphas_plot, finals, 's-', label='Final (last 100 batches)', color='blue',
            linewidth=2, markersize=6)
    ax.fill_between(alphas_plot, finals, initials, alpha=0.15, color='gray',
                    label='Learning gap')

    ax.set_xlabel("Alpha (fraction of rules corrupted)")
    ax.set_ylabel("Loss")
    ax.set_title("Rule corruption: Initial vs Final loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "v2_initial_vs_final.png"), dpi=150, bbox_inches='tight')
    print(f"Saved {OUT_DIR}/v2_initial_vs_final.png")

if __name__ == '__main__':
    results = load_all()
    print(f"Loaded {len(results)} conditions")
    plot_loss_curves(results)
    plot_initial_vs_final(results)
