"""Plot ε-threshold and δ-gap curves for intervention experiments.

Produces two figures showing how well interventions work across all
possible thresholds, without committing to any single metric.

Figure 1: ε-threshold curves
  - Newly legal: fraction of interventions where P(newly legal) > ε
  - Newly illegal: fraction where P(newly illegal) < ε

Figure 2: δ-gap curves
  - Newly legal: δ = P(newly legal) - max(P(always-illegal))
  - Newly illegal: δ = min(P(always-legal)) - P(newly illegal)
  Survival curves; key question is fraction crossing δ=0.

Usage:
    python plot_intervention_thresholds.py
    python plot_intervention_thresholds.py --results-dir experiments/multi_intervention_probs
"""
import sys, os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_intervention_data(results_dir):
    """Load raw_samples.json and extract per-intervention probability data."""
    path = os.path.join(results_dir, "raw_samples.json")
    with open(path) as f:
        data = json.load(f)

    samples = []
    for cond in data:
        for n_str in data[cond]:
            for s in data[cond][n_str]:
                if "orig_probs" not in s or "intv_probs" not in s:
                    continue
                if "original_legal" not in s or "counterfactual_legal" not in s:
                    continue
                samples.append({
                    "condition": cond,
                    "n": int(n_str),
                    "orig_probs": np.array(s["orig_probs"]),
                    "intv_probs": np.array(s["intv_probs"]),
                    "original_legal": set(s["original_legal"]),
                    "counterfactual_legal": set(s["counterfactual_legal"]),
                })

    print(f"Loaded {len(samples)} interventions from {path}")
    return samples


def classify_cells(s):
    """Classify each cell into newly_legal, newly_illegal, always_legal, always_illegal."""
    orig = s["original_legal"]
    cf = s["counterfactual_legal"]
    newly_legal = cf - orig
    newly_illegal = orig - cf
    always_legal = orig & cf
    always_illegal = set(range(64)) - orig - cf - newly_legal - newly_illegal
    # Actually: always_illegal = not in orig AND not in cf
    always_illegal = set(range(64)) - (orig | cf)
    return newly_legal, newly_illegal, always_legal, always_illegal


def compute_epsilon_data(samples):
    """For each intervention, collect P(newly legal) and P(newly illegal) post-intervention."""
    promo_probs = []  # P(newly legal cell) after intervention
    suppress_probs = []  # P(newly illegal cell) after intervention
    always_illegal_probs = []
    always_legal_probs = []

    for s in samples:
        newly_legal, newly_illegal, always_legal, always_illegal = classify_cells(s)
        ip = s["intv_probs"]

        for c in newly_legal:
            if ip[c] >= 0:  # -1 means invalid cell
                promo_probs.append(ip[c])
        for c in newly_illegal:
            if ip[c] >= 0:
                suppress_probs.append(ip[c])
        for c in always_illegal:
            if ip[c] >= 0:
                always_illegal_probs.append(ip[c])
        for c in always_legal:
            if ip[c] >= 0:
                always_legal_probs.append(ip[c])

    return (np.array(promo_probs), np.array(suppress_probs),
            np.array(always_illegal_probs), np.array(always_legal_probs))


def compute_delta_data(samples):
    """For each intervention, compute δ for newly legal and newly illegal cells."""
    delta_legal = []   # δ = P(newly legal) - max(P(always illegal))
    delta_illegal = []  # δ = min(P(always legal)) - P(newly illegal)

    for s in samples:
        newly_legal, newly_illegal, always_legal, always_illegal = classify_cells(s)
        ip = s["intv_probs"]

        # max P(always illegal) — the "smoothing floor"
        ai_probs = [ip[c] for c in always_illegal if ip[c] >= 0]
        max_ai = max(ai_probs) if ai_probs else 0.0

        # min P(always legal) — the "weakest legal"
        al_probs = [ip[c] for c in always_legal if ip[c] >= 0]
        min_al = min(al_probs) if al_probs else 0.0

        for c in newly_legal:
            if ip[c] >= 0:
                delta_legal.append(ip[c] - max_ai)
        for c in newly_illegal:
            if ip[c] >= 0:
                delta_illegal.append(min_al - ip[c])

    return np.array(delta_legal), np.array(delta_illegal)


def plot_epsilon_curves(promo_probs, suppress_probs, always_illegal_probs,
                        always_legal_probs, output_path, label="OthelloGPT"):
    """Figure 1: ε-threshold curves."""
    eps_range = np.logspace(-4, np.log10(0.20), 200)

    # Promotion: fraction where P(newly legal) > ε
    promo_curve = np.array([np.mean(promo_probs > e) for e in eps_range])

    # Suppression: fraction where P(newly illegal) < ε
    suppress_curve = np.array([np.mean(suppress_probs < e) for e in eps_range])

    # Reference values
    smoothing_floor = np.mean(always_illegal_probs) if len(always_illegal_probs) > 0 else 0
    legal_threshold = np.mean(always_legal_probs) if len(always_legal_probs) > 0 else 0
    # Actually use the weakest always-legal mean
    uniform = 1.0 / 60

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(eps_range, promo_curve, 'b-', linewidth=2,
            label=f'Newly legal: P > ε ({label})')
    ax.plot(eps_range, suppress_curve, 'r-', linewidth=2,
            label=f'Newly illegal: P < ε ({label})')

    # Reference lines
    ax.axvline(smoothing_floor, color='gray', linestyle=':', alpha=0.7,
               label=f'Mean P(always-illegal) = {smoothing_floor:.4f}')
    ax.axvline(uniform, color='green', linestyle='--', alpha=0.7,
               label=f'Uniform baseline (1/60) = {uniform:.4f}')
    if legal_threshold > 0:
        ax.axvline(legal_threshold, color='orange', linestyle=':', alpha=0.7,
                   label=f'Mean P(always-legal) = {legal_threshold:.4f}')

    ax.set_xscale('log')
    ax.set_xlabel('ε (probability threshold)', fontsize=12)
    ax.set_ylabel('Fraction of interventions', fontsize=12)
    ax.set_title('ε-Threshold Curves: Intervention Effectiveness', fontsize=14)
    ax.legend(fontsize=9, loc='center left')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_delta_curves(delta_legal, delta_illegal, output_path, label="OthelloGPT"):
    """Figure 2: δ-gap survival curves."""
    delta_range = np.linspace(-0.15, 0.15, 300)

    # Survival: fraction achieving at least δ
    legal_survival = np.array([np.mean(delta_legal >= d) for d in delta_range])
    illegal_survival = np.array([np.mean(delta_illegal >= d) for d in delta_range])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(delta_range, legal_survival, 'b-', linewidth=2,
            label=f'Newly legal: δ ≥ threshold ({label})')
    ax.plot(delta_range, illegal_survival, 'r-', linewidth=2,
            label=f'Newly illegal: δ ≥ threshold ({label})')

    ax.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax.annotate('δ = 0\n(boundary)', xy=(0, 0.5), fontsize=9, alpha=0.7,
                ha='right', va='center')

    # Mark the fraction crossing δ=0
    frac_legal_cross = np.mean(delta_legal >= 0) if len(delta_legal) > 0 else 0
    frac_illegal_cross = np.mean(delta_illegal >= 0) if len(delta_illegal) > 0 else 0
    ax.plot(0, frac_legal_cross, 'bo', markersize=8)
    ax.plot(0, frac_illegal_cross, 'ro', markersize=8)
    ax.annotate(f'  {frac_legal_cross:.1%}', xy=(0.002, frac_legal_cross),
                fontsize=10, color='blue')
    ax.annotate(f'  {frac_illegal_cross:.1%}', xy=(0.002, frac_illegal_cross),
                fontsize=10, color='red')

    ax.set_xlabel('δ (probability gap)', fontsize=12)
    ax.set_ylabel('Fraction of interventions achieving ≥ δ', fontsize=12)
    ax.set_title('δ-Gap Survival Curves: Crossing the Legal/Illegal Boundary', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")

    return frac_legal_cross, frac_illegal_cross


def analyze_one_dataset(results_dir, output_dir, label="OthelloGPT", n=1):
    """Run full analysis on one results directory.

    n: number of simultaneous interventions to analyze (1/2/3).
    """
    samples = load_intervention_data(results_dir)

    samples_n = [s for s in samples if s["n"] == n]
    print(f"  N={n} interventions: {len(samples_n)}")

    if not samples_n:
        print(f"  No N={n} samples found, using all")
        samples_n = samples

    promo, suppress, ai, al = compute_epsilon_data(samples_n)
    print(f"  Newly legal cells: {len(promo)}")
    print(f"  Newly illegal cells: {len(suppress)}")
    print(f"  Always illegal: mean P = {np.mean(ai):.6f}" if len(ai) > 0 else "")
    print(f"  Always legal: mean P = {np.mean(al):.4f}" if len(al) > 0 else "")

    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_N{n}" if n != 1 else ""
    plot_epsilon_curves(promo, suppress, ai, al,
                        os.path.join(output_dir,
                                     f"epsilon_curves_{label}{suffix}.png"),
                        label=f"{label} N={n}")

    delta_legal, delta_illegal = compute_delta_data(samples_n)
    frac_l, frac_i = plot_delta_curves(
        delta_legal, delta_illegal,
        os.path.join(output_dir, f"delta_curves_{label}{suffix}.png"),
        label=f"{label} N={n}")

    print(f"\n  δ=0 crossing rates:")
    print(f"    Newly legal moved above max(always-illegal): {frac_l:.1%}")
    print(f"    Newly illegal moved below min(always-legal): {frac_i:.1%}")

    return {
        "label": label,
        "n_interventions": len(samples_n),
        "n_newly_legal": len(promo),
        "n_newly_illegal": len(suppress),
        "mean_promo_prob": float(np.mean(promo)) if len(promo) > 0 else None,
        "mean_suppress_prob": float(np.mean(suppress)) if len(suppress) > 0 else None,
        "frac_legal_cross_delta0": frac_l,
        "frac_illegal_cross_delta0": frac_i,
        "mean_always_illegal_prob": float(np.mean(ai)) if len(ai) > 0 else None,
        "mean_always_legal_prob": float(np.mean(al)) if len(al) > 0 else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",
                        default="experiments/multi_intervention_probs",
                        help="Directory with raw_samples.json")
    parser.add_argument("--output-dir",
                        default="experiments/intervention_threshold_analysis")
    parser.add_argument("--label", default="OthelloGPT")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of simultaneous interventions to analyze")
    # Support multiple datasets for overlay comparison
    parser.add_argument("--compare", nargs="*", default=[],
                        help="Additional results dirs to overlay (format: dir:label)")
    args = parser.parse_args()

    all_results = []

    # Primary dataset
    print(f"\n=== Analyzing: {args.label} (N={args.n}) ===")
    result = analyze_one_dataset(args.results_dir, args.output_dir,
                                  args.label, n=args.n)
    all_results.append(result)

    # Additional datasets for comparison
    for comp in args.compare:
        if ":" in comp:
            comp_dir, comp_label = comp.split(":", 1)
        else:
            comp_dir = comp
            comp_label = os.path.basename(comp)
        print(f"\n=== Analyzing: {comp_label} (N={args.n}) ===")
        r = analyze_one_dataset(comp_dir, args.output_dir, comp_label, n=args.n)
        all_results.append(r)

    # Save summary
    suffix = f"_N{args.n}" if args.n != 1 else ""
    summary_path = os.path.join(args.output_dir,
                                f"threshold_analysis_summary{suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # If multiple datasets, make overlay plots
    if len(all_results) > 1:
        print("\nOverlay plots not yet implemented — individual plots saved per dataset")
