"""Simple two-panel survival plots for newly-legal cells under intervention.

  ε panel: ε = P(newly-legal) - max(P(always-illegal))   — distance above
           the most probable always-illegal cell. Near 0 for "just barely
           crossing" interventions, positive for decisive promotion.

  δ panel: δ = P(newly-legal) - min(P(always-legal))     — distance above
           the weakest still-legal cell. Typically negative (target hasn't
           reached the actual legal-probability range).

For each, plots a survival curve: fraction of newly-legal cells achieving
≥ threshold, with the value at threshold=0 annotated.

Usage:
    python plot_eps_delta_simple.py \\
        --results-dir experiments/multi_intervention_probs_cd0_K3 \\
        --label OthelloGPT_cd0_K3 --n 1
"""
import sys, os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_samples(results_dir):
    path = os.path.join(results_dir, "raw_samples.json")
    with open(path) as f:
        data = json.load(f)
    samples = []
    for cond in data:
        for n_str in data[cond]:
            for s in data[cond][n_str]:
                if "intv_probs" not in s:
                    continue
                if "original_legal" not in s or "counterfactual_legal" not in s:
                    continue
                samples.append({
                    "n": int(n_str),
                    "intv_probs": np.array(s["intv_probs"]),
                    "original_legal": set(s["original_legal"]),
                    "counterfactual_legal": set(s["counterfactual_legal"]),
                })
    return samples


def compute_eps_delta_newly_legal(samples):
    """For each newly-legal cell across all interventions, return:
       eps = P(newly_legal) - max(P(always_illegal))
       delta = P(newly_legal) - min(P(always_legal))
    """
    eps_list, delta_list = [], []
    for s in samples:
        orig = s["original_legal"]
        cf = s["counterfactual_legal"]
        newly_legal = cf - orig
        always_legal = orig & cf
        always_illegal = set(range(64)) - (orig | cf)
        ip = s["intv_probs"]

        ai_probs = [ip[c] for c in always_illegal if ip[c] >= 0]
        al_probs = [ip[c] for c in always_legal if ip[c] >= 0]
        if not ai_probs or not al_probs:
            continue
        max_ai = max(ai_probs)
        min_al = min(al_probs)
        for c in newly_legal:
            if ip[c] < 0:
                continue
            eps_list.append(ip[c] - max_ai)
            delta_list.append(ip[c] - min_al)
    return np.array(eps_list), np.array(delta_list)


def plot_panel(ax, values, label, color="#cf5044"):
    """Survival curve: fraction with metric >= threshold."""
    n = len(values)
    if n == 0:
        ax.text(0.5, 0.5, "no samples", ha="center", va="center")
        return
    vmin = float(np.percentile(values, 1))
    vmax = float(np.percentile(values, 99))
    pad = 0.1 * (vmax - vmin if vmax > vmin else max(abs(vmax), abs(vmin), 1e-6))
    xs = np.linspace(vmin - pad, vmax + pad, 400)
    ys = np.array([np.mean(values >= x) for x in xs])

    ax.plot(xs, ys, color=color, linewidth=2)
    # Annotation at threshold = 0
    frac_at_zero = float(np.mean(values >= 0))
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)
    ax.plot(0, frac_at_zero, "o", color=color, markersize=7)
    ax.annotate(f"  {frac_at_zero*100:.1f}%", xy=(0, frac_at_zero),
                color=color, fontsize=11,
                ha="left", va="center")
    ax.text(0.97, 0.95, f"n = {n}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

    ax.set_ylabel("fraction", fontsize=12)
    ax.set_xlabel(label, fontsize=14)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--label", required=True,
                        help="Label used in output filenames")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of simultaneous interventions to analyze")
    parser.add_argument("--output-dir",
                        default="experiments/intervention_threshold_analysis")
    args = parser.parse_args()

    samples = load_samples(args.results_dir)
    samples_n = [s for s in samples if s["n"] == args.n]
    print(f"Loaded {len(samples)} total samples, {len(samples_n)} with N={args.n}")

    eps, delta = compute_eps_delta_newly_legal(samples_n)
    print(f"Newly-legal cells: {len(eps)}")
    print(f"  ε > 0: {np.mean(eps > 0)*100:.1f}%   median ε = {np.median(eps):.4f}")
    print(f"  δ > 0: {np.mean(delta > 0)*100:.1f}%  median δ = {np.median(delta):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_panel(axes[0], eps, "ε")
    plot_panel(axes[1], delta, "δ")
    suffix = f"_N{args.n}" if args.n != 1 else ""
    out = os.path.join(args.output_dir, f"eps_delta_simple_{args.label}{suffix}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
