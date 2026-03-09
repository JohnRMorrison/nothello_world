"""Extract human-readable heuristics from a trained MLP checkpoint.

Only needs the checkpoint file — no data loading required.

Usage:
    python extract_heuristics.py [--hidden 1024] [--output-dir ...]
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

# Valid moves: 0-63 excluding center squares 27,28,35,36
_VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})
N_MOVES = 60
OPTIONS = 3  # empty, white, black


def _feature_name(idx):
    if idx < N_MOVES:
        pos = _VALID_MOVES[idx]
        row, col = pos // 8, pos % 8
        return f"played[{chr(65+row)}{col+1}]"
    elif idx < 2 * N_MOVES:
        i = idx - N_MOVES
        pos = _VALID_MOVES[i]
        row, col = pos // 8, pos % 8
        return f"when[{chr(65+row)}{col+1}]"
    else:
        i = idx - 2 * N_MOVES
        pos = _VALID_MOVES[i]
        row, col = pos // 8, pos % 8
        return f"even[{chr(65+row)}{col+1}]"


def _cell_name(cell_idx):
    row, col = cell_idx // 8, cell_idx % 8
    return f"{chr(65+row)}{col+1}"


def _class_name(cls_idx):
    return ["empty", "white", "black"][cls_idx]


def _build_mlp(input_dim, hidden_dim, output_dim, num_hidden_layers=1):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_hidden_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def analyze_mlp(mlp, hidden_dim, parity_name):
    """Analyze weight structure of a trained MLP."""
    W1 = mlp[0].weight.detach().cpu().numpy()  # (H, 180)
    b1 = mlp[0].bias.detach().cpu().numpy()
    W2 = mlp[-1].weight.detach().cpu().numpy()  # (192, H)

    units = []
    for j in range(hidden_dim):
        w_in = W1[j]
        bias = float(b1[j])

        # Top 10 input features by absolute weight
        top_in_idx = np.argsort(np.abs(w_in))[::-1][:10]
        top_inputs = []
        for k in top_in_idx:
            top_inputs.append({
                "feature": _feature_name(int(k)),
                "index": int(k),
                "weight": float(w_in[k]),
            })

        # Output weights for unit j
        w_out = W2[:, j]
        w_out_reshaped = w_out.reshape(64, OPTIONS)

        # Top 5 output contributions
        flat_idx = np.argsort(np.abs(w_out))[::-1][:5]
        top_outputs = []
        for k in flat_idx:
            cell = int(k) // OPTIONS
            cls = int(k) % OPTIONS
            top_outputs.append({
                "cell": _cell_name(cell),
                "cell_idx": cell,
                "class": _class_name(cls),
                "class_idx": cls,
                "weight": float(w_out[k]),
            })

        out_l2 = float(np.linalg.norm(w_out))

        # Description
        desc_parts = []
        for inp in top_inputs[:3]:
            sign = "+" if inp["weight"] > 0 else "-"
            desc_parts.append(f"{sign}{inp['feature']}")
        input_desc = ", ".join(desc_parts)

        output_parts = []
        for out in top_outputs[:2]:
            sign = "+" if out["weight"] > 0 else "-"
            output_parts.append(f"{sign}{out['cell']}={out['class']}")
        output_desc = ", ".join(output_parts)

        description = f"[{parity_name}] Unit {j}: IF {input_desc} THEN {output_desc}"

        # Categorize
        n_played = sum(1 for inp in top_inputs if inp["feature"].startswith("played"))
        n_when = sum(1 for inp in top_inputs if inp["feature"].startswith("when"))
        n_even = sum(1 for inp in top_inputs if inp["feature"].startswith("even"))
        max_count = max(n_played, n_when, n_even)
        if n_played == max_count and n_when < max_count and n_even < max_count:
            category = "placement"
        elif n_when == max_count and n_played < max_count and n_even < max_count:
            category = "temporal"
        elif n_even == max_count and n_played < max_count and n_when < max_count:
            category = "parity"
        else:
            category = "interaction"

        # Output scope
        cell_contributions = np.abs(w_out_reshaped).max(axis=1)
        threshold = cell_contributions.max() * 0.1
        n_significant_cells = int((cell_contributions > threshold).sum())
        output_scope = "global" if n_significant_cells > 5 else "local"

        # Primary class
        class_totals = np.abs(w_out_reshaped).sum(axis=0)
        primary_class = _class_name(int(np.argmax(class_totals)))

        units.append({
            "unit": j,
            "parity": parity_name,
            "bias": bias,
            "top_inputs": top_inputs,
            "top_outputs": top_outputs,
            "out_l2": out_l2,
            "description": description,
            "category": category,
            "output_scope": output_scope,
            "primary_class": primary_class,
        })

    return units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--output-dir", default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    H = args.hidden
    ckpt_dir = os.path.join(args.output_dir, "mlp_checkpoints")
    ckpt_path = os.path.join(ckpt_dir, f"mlp_180_H{H}.pt")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt["hidden_dim"]
    n_layers = ckpt.get("num_hidden_layers", 1)
    print(f"  Input: {input_dim}, Hidden: {hidden_dim}, Layers: {n_layers}")

    mlp_even = _build_mlp(input_dim, hidden_dim, 64 * OPTIONS, n_layers)
    mlp_odd = _build_mlp(input_dim, hidden_dim, 64 * OPTIONS, n_layers)
    mlp_even.load_state_dict(ckpt["even"])
    mlp_odd.load_state_dict(ckpt["odd"])

    print("Analyzing even MLP...")
    even_units = analyze_mlp(mlp_even, hidden_dim, "even")
    print("Analyzing odd MLP...")
    odd_units = analyze_mlp(mlp_odd, hidden_dim, "odd")

    all_units = even_units + odd_units
    all_units_sorted = sorted(all_units, key=lambda u: -u["out_l2"])

    # Category stats
    categories = {}
    for u in all_units:
        cat = u["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print(f"\nCategory distribution ({len(all_units)} total units):")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Write descriptions
    out_path = os.path.join(args.output_dir, f"mlp_H{H}_heuristics.txt")
    with open(out_path, "w") as f:
        f.write(f"MLP Heuristics (H={H})\n")
        f.write(f"{'='*80}\n\n")
        for u in all_units_sorted:
            f.write(f"{u['description']}\n")
            f.write(f"  Category: {u['category']}, Scope: {u['output_scope']}, "
                    f"Primary class: {u['primary_class']}, L2: {u['out_l2']:.4f}\n")
            inp_strs = [f"{inp['feature']}({inp['weight']:.4f})" for inp in u['top_inputs'][:5]]
            f.write(f"  Top inputs: {', '.join(inp_strs)}\n")
            out_strs = [f"{out['cell']}={out['class']}({out['weight']:.4f})" for out in u['top_outputs'][:3]]
            f.write(f"  Top outputs: {', '.join(out_strs)}\n\n")

    print(f"\nSaved to {out_path}")

    # Also save JSON
    json_path = os.path.join(args.output_dir, f"mlp_H{H}_heuristics.json")
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(json_path, "w") as f:
        json.dump([{k: convert(v) for k, v in u.items()} for u in all_units_sorted], f, indent=2)
    print(f"Saved JSON to {json_path}")


if __name__ == "__main__":
    main()
