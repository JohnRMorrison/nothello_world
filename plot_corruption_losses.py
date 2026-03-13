"""Plot corruption fine-tuning loss curves batch-by-batch."""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

LOSS_DIR = "experiments/corruption/losses"
OUT_DIR = "experiments/corruption/figures"

ALPHAS = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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

def plot_by_type(results):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    type_names = {1: "Interpolate toward random", 2: "Randomize individual weights", 3: "Redirect outputs"}

    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.viridis

    for tidx, ctype in enumerate([1, 2, 3]):
        ax = axes[tidx]
        for alpha in ALPHAS:
            alpha_str = f"{int(alpha * 100):03d}"
            label = f"type{ctype}_alpha{alpha_str}"
            if label not in results:
                continue
            losses = results[label]['losses']
            smoothed = smooth(losses)
            color = cmap(norm(alpha))
            ax.plot(smoothed, color=color, linewidth=0.8, alpha=0.8)

        ax.set_title(f"Type {ctype}: {type_names[ctype]}", fontsize=11)
        ax.set_xlabel("Batch")
        if tidx == 0:
            ax.set_ylabel("Loss (smoothed, window=50)")
        ax.set_ylim(1.5, 4.5)
        ax.grid(True, alpha=0.3)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Alpha (corruption level)")

    fig.suptitle("OthelloGPT fine-tuning on corrupted heuristic games", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "corruption_loss_curves.png"), dpi=150, bbox_inches='tight')
    print(f"Saved {OUT_DIR}/corruption_loss_curves.png")

def plot_final_loss(results):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for ctype, marker, label in [(1, 'o', 'Type 1: Interpolate'), (2, 's', 'Type 2: Rand weights'), (3, '^', 'Type 3: Redirect')]:
        alphas_plot = []
        finals = []
        initials = []
        for alpha in ALPHAS:
            alpha_str = f"{int(alpha * 100):03d}"
            key = f"type{ctype}_alpha{alpha_str}"
            if key not in results:
                continue
            losses = results[key]['losses']
            alphas_plot.append(alpha)
            finals.append(np.mean(losses[-100:]))
            initials.append(np.mean(losses[:10]))

        ax.plot(alphas_plot, finals, marker=marker, label=f"{label} (final)", linewidth=2, markersize=6)
        ax.plot(alphas_plot, initials, marker=marker, label=f"{label} (initial)", linewidth=1, linestyle='--', alpha=0.5, markersize=4)

    ax.set_xlabel("Alpha (corruption level)")
    ax.set_ylabel("Loss")
    ax.set_title("Initial vs Final loss by corruption type")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "corruption_final_vs_initial.png"), dpi=150, bbox_inches='tight')
    print(f"Saved {OUT_DIR}/corruption_final_vs_initial.png")

if __name__ == '__main__':
    results = load_all()
    print(f"Loaded {len(results)} conditions")
    plot_by_type(results)
    plot_final_loss(results)
