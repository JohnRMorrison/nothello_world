"""
Extract human-readable IF-THEN rules from pre-trained decision trees for Othello-GPT neurons.

Usage:
    python extract_rules.py                          # all layers, top 20 neurons per layer
    python extract_rules.py --layer 5                # single layer
    python extract_rules.py --layer 5 --neuron 100   # single neuron
    python extract_rules.py --top_n 50               # top 50 neurons per layer
    python extract_rules.py --min_score 0.99          # all neurons with F1 > 0.99
    python extract_rules.py --tree_type regression    # use regression trees instead
    python extract_rules.py --save rules.json        # save to JSON

    # Influence-based ranking (pick neurons that matter most to the model):
    python extract_rules.py --min_score 0.95 --top_n_influential 5 --layer 5
    python extract_rules.py --min_score 0.95 --top_n_influential 5 --precise_influence --layer 5
"""

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch as t

# decision_trees/dtypes.py must be importable for pickle to load the dataclasses
sys.path.insert(0, str(Path(__file__).resolve().parent / "OthelloReverseEngineering" / "decision_trees"))

from sklearn.tree import _tree
from skimage.filters import threshold_otsu

from OthelloReverseEngineering.utils.feature_extraction_utils import (
    create_feature_names,
    get_feature_names_cont_dt,
    rule_infer,
)


BASE_DIR = Path(__file__).resolve().parent / "OthelloReverseEngineering" / "decision_trees"
N_LAYERS = 8
N_NEURONS = 2048


# ---------------------------------------------------------------------------
# Influence scoring (DLA + ablation)
# ---------------------------------------------------------------------------

def _load_model_and_data(device=None):
    """Lazily load the Othello-GPT model and a batch of game data for scoring."""
    from OthelloReverseEngineering.utils.circuits_utils import get_model, construct_othello_dataset

    if device is None:
        device = "cuda:1" if t.cuda.is_available() else "cpu"

    model = get_model("Baidicoot/Othello-GPT-Transformer-Lens", t.device(device))
    dataset = construct_othello_dataset(
        custom_functions=[],
        n_inputs=500,
        split="train",
        device=device,
        precompute_dataset=False,
    )
    game_data_BL = t.tensor(dataset["encoded_inputs"], device=device)
    return model, game_data_BL


def compute_dla_scores(model, layer, neuron_indices, game_data_BL):
    """Compute Direct Logit Attribution scores for neurons in a given layer.

    Uses the static weight path (W_out @ W_U) multiplied by actual activations
    to estimate how much each neuron contributes to the model's output logits.

    Returns {neuron_idx: float_score} for each neuron in neuron_indices.
    """
    import einops

    # Static weight attribution: W_out[layer] @ W_U[:, 1:]
    w_out = model.W_out[layer].detach().clone()        # [neuron, d_model]
    W_U = model.W_U[:, 1:].detach().clone()             # [d_model, 60]
    write_attr = einops.einsum(
        w_out, W_U,
        "neuron d_model, d_model square -> neuron square",
    )  # [neuron, 60]

    # Forward pass to get actual activations
    with t.no_grad():
        with model.trace(game_data_BL):
            acts_BLN = model.blocks[layer].mlp.hook_post.output.save()

    # acts_BLN: [batch, seq, neuron]
    # For each neuron, score = sum_over(batch,seq) of |activation * write_attr|
    neuron_list = list(neuron_indices)
    idx_tensor = t.tensor(neuron_list, device=acts_BLN.device)
    selected_acts = acts_BLN[:, :, idx_tensor]           # [batch, seq, len(neuron_list)]
    selected_attr = write_attr[idx_tensor, :]             # [len(neuron_list), 60]

    # Multiply activation by attribution magnitude, sum over squares
    attr_magnitude = selected_attr.abs().sum(dim=-1)      # [len(neuron_list)]
    act_magnitude = selected_acts.abs().mean(dim=(0, 1))  # [len(neuron_list)]
    scores = (act_magnitude * attr_magnitude)              # [len(neuron_list)]

    return {idx: scores[i].item() for i, idx in enumerate(neuron_list)}


def compute_ablation_scores(model, layer, neuron_indices, game_data_BL):
    """Compute mean-ablation KL divergence for each neuron individually.

    For each neuron, replaces its activation with its dataset mean and measures
    the resulting KL divergence from the clean distribution.

    Returns {neuron_idx: mean_kl_div}.
    """
    from OthelloReverseEngineering.utils.helper_fns import neuron_intervention, compute_kl_divergence

    scores = {}
    for idx in neuron_indices:
        logits_clean_BLV, logits_patch_BLV = neuron_intervention(
            model,
            layers_neurons={layer: [idx]},
            game_batch_BL=game_data_BL,
            ablation_method="mean",
        )
        kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_patch_BLV)
        scores[idx] = kl_div_BL.mean().item()

    return scores


# ---------------------------------------------------------------------------
# Rule extraction (self-contained, no dependency on the batch wrappers)
# ---------------------------------------------------------------------------

def extract_rules_from_classification_tree(tree, feature_names, target_class=1, value_threshold=0.7):
    """Walk a fitted BinaryDecisionTreeClassifier and return every root-to-leaf
    path whose majority class equals *target_class*.

    Returns list of dicts, one per leaf:
        rule        – human-readable AND-rule string
        precision   – fraction of leaf samples belonging to target_class
        samples     – number of training samples that reached this leaf
        features    – set of feature names used in the path
    """
    tree_ = tree.tree_
    fname = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    results = []

    def recurse(node, conditions, feats):
        values = tree_.value[node][0]
        is_leaf = tree_.feature[node] == _tree.TREE_UNDEFINED
        # Also treat as leaf if value_threshold is met (early stop)
        high_confidence = values[target_class] / values.sum() >= value_threshold if values.sum() > 0 else False
        if not is_leaf and not high_confidence:
            name = fname[node]
            recurse(tree_.children_left[node],  conditions + [f"(NOT {name})"], feats | {name})
            recurse(tree_.children_right[node], conditions + [f"({name})"],     feats | {name})
        else:
            pred_class = values.argmax()
            if pred_class == target_class:
                precision = values[target_class] / values.sum() if values.sum() > 0 else 0.0
                results.append({
                    "rule": rule_infer(" AND ".join(conditions)),
                    "precision": round(float(precision), 4),
                    "samples": int(tree_.n_node_samples[node]),
                    "features": feats,
                })

    recurse(0, [], set())
    return results


def extract_rules_from_regression_tree(tree, feature_names):
    """Walk a fitted DecisionTreeRegressor. Use Otsu thresholding on leaf values
    to separate 'active' vs 'inactive' leaves, then return rules for active leaves.

    Returns list of dicts:
        rule             – human-readable AND-rule string
        predicted_value  – mean predicted activation at this leaf
        samples          – number of training samples that reached this leaf
        features         – set of feature names used in the path
    """
    tree_ = tree.tree_
    fname = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    # Compute activation threshold via Otsu on leaf values
    leaf_mask = tree_.children_right == -1
    leaf_values = tree_.value[leaf_mask].flatten()
    if len(leaf_values) < 2 or leaf_values.max() == leaf_values.min():
        on_off_threshold = float(leaf_values.mean())
    else:
        on_off_threshold = float(threshold_otsu(leaf_values))

    results = []

    def recurse(node, conditions, feats):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = fname[node]
            recurse(tree_.children_left[node],  conditions + [f"(NOT {name})"], feats | {name})
            recurse(tree_.children_right[node], conditions + [f"({name})"],     feats | {name})
        else:
            value = float(tree_.value[node][0][0])
            if value > on_off_threshold:
                results.append({
                    "rule": rule_infer(" AND ".join(conditions)),
                    "predicted_value": round(value, 4),
                    "samples": int(tree_.n_node_samples[node]),
                    "features": feats,
                })

    recurse(0, [], set())
    return results


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_classification_trees(layer):
    path = BASE_DIR / "ground_truth_features" / "classification" / "results" / f"layer_{layer}_trees.pkl.gz"
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_regression_trees(layer):
    path = BASE_DIR / "ground_truth_features" / "regression" / "results" / f"layer_{layer}_trees.pkl.gz"
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_continuous_trees(layer):
    path = BASE_DIR / "continuous_features" / "results" / f"layer_{layer}_trees.pkl.gz"
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def get_feature_names_for_tree(tree_obj):
    """Infer the right feature names from a fitted sklearn tree."""
    n = tree_obj.n_features_in_
    # Ground truth trees use 320 features (192 board + 64 flipped + 64 played)
    # Continuous probe trees use 248 features
    if n == 320:
        return create_feature_names(n, "games_batch_to_board_state_flipped_played_BLC")
    elif n == 248:
        return get_feature_names_cont_dt()
    else:
        # Fallback: try common sizes, else generic
        try:
            return create_feature_names(n, "games_batch_to_board_state_flipped_played_BLC")
        except Exception:
            return [f"feature_{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_all_rules(layers, tree_type="classification", top_n=20, neuron_idx=None,
                      min_score=None, min_rules=1,
                      top_n_influential=None, precise_influence=False,
                      model=None, game_data_BL=None):
    """Extract rules across layers/neurons. Returns a nested dict:
        result[layer][neuron] = { neuron_score, total_samples, rules: [...] }

    If top_n_influential is set, neurons are ranked by influence (DLA, optionally
    refined via mean-ablation KL divergence) rather than decision tree accuracy.
    """
    all_results = {}

    for layer in layers:
        if tree_type == "classification":
            trees = load_classification_trees(layer)
        elif tree_type == "regression":
            trees = load_regression_trees(layer)
        elif tree_type == "continuous":
            trees = load_continuous_trees(layer)
        else:
            raise ValueError(f"Unknown tree_type: {tree_type}")

        # Build (neuron_idx, quality_score) pairs for ranking
        scored = []
        for i, tree_result in enumerate(trees):
            if not hasattr(tree_result.tree, "tree_"):
                continue
            if tree_type == "classification":
                score = tree_result.test_F1
            else:
                score = tree_result.test_R2
            scored.append((i, score))

        # Pick neurons
        if neuron_idx is not None:
            scored = [(i, s) for i, s in scored if i == neuron_idx]
        elif min_score is not None:
            scored = [(i, s) for i, s in scored if s > min_score]
            scored.sort(key=lambda x: -x[1])
        else:
            scored.sort(key=lambda x: -x[1])
            scored = scored[:top_n]

        # --- Influence-based ranking ---
        influence_scores = {}
        dla_for_display = {}
        if top_n_influential is not None and scored:
            candidate_indices = [i for i, _ in scored]
            print(f"  Layer {layer}: computing DLA scores for {len(candidate_indices)} candidates...")
            dla_scores = compute_dla_scores(model, layer, candidate_indices, game_data_BL)

            if precise_influence:
                # Tier 1: narrow to 3*N by DLA
                n_dla_candidates = min(3 * top_n_influential, len(candidate_indices))
                top_by_dla = sorted(dla_scores, key=dla_scores.get, reverse=True)[:n_dla_candidates]
                print(f"  Layer {layer}: computing ablation scores for top {len(top_by_dla)} DLA candidates...")
                abl_scores = compute_ablation_scores(model, layer, top_by_dla, game_data_BL)
                # Tier 2: pick top N by KL div
                top_by_kl = sorted(abl_scores, key=abl_scores.get, reverse=True)[:top_n_influential]
                influence_scores = {idx: abl_scores[idx] for idx in top_by_kl}
                # Also store DLA for display
                dla_for_display = {idx: dla_scores[idx] for idx in top_by_kl}
            else:
                top_by_dla = sorted(dla_scores, key=dla_scores.get, reverse=True)[:top_n_influential]
                influence_scores = {idx: dla_scores[idx] for idx in top_by_dla}
                dla_for_display = influence_scores

            # Filter scored to only the selected neurons, preserving influence order
            score_map = dict(scored)
            scored = [(idx, score_map[idx]) for idx in influence_scores]

        layer_results = {}
        for idx, score in scored:
            tree_obj = trees[idx].tree
            feature_names = get_feature_names_for_tree(tree_obj)

            if tree_type == "classification":
                rules = extract_rules_from_classification_tree(tree_obj, feature_names)
            else:
                rules = extract_rules_from_regression_tree(tree_obj, feature_names)

            total_samples = int(tree_obj.tree_.n_node_samples[0])
            min_samples = total_samples / 59 * 0.05  # ~5% of per-position samples

            # Filter small leaves and sort by samples desc, then strength desc
            rules = [r for r in rules if r["samples"] >= min_samples]
            sort_key = "precision" if tree_type == "classification" else "predicted_value"
            rules.sort(key=lambda r: (-r["samples"], -r[sort_key]))

            # Strip 'features' set (not JSON-serializable) — keep as display-only
            for r in rules:
                r["features"] = sorted(r["features"])

            score_name = "test_F1" if tree_type == "classification" else "test_R2"
            if len(rules) >= min_rules:
                entry = {
                    score_name: round(score, 4),
                    "total_training_samples": total_samples,
                    "num_rules": len(rules),
                    "rules": rules,
                }
                if idx in influence_scores:
                    entry["influence_score"] = round(influence_scores[idx], 6)
                    if precise_influence and idx in dla_for_display:
                        entry["dla_score"] = round(dla_for_display[idx], 6)
                layer_results[idx] = entry

        all_results[layer] = layer_results

    return all_results


def print_results(results, tree_type):
    score_name = "test_F1" if tree_type == "classification" else "test_R2"
    strength_name = "precision" if tree_type == "classification" else "predicted_value"

    for layer in sorted(results):
        neurons = results[layer]
        if not neurons:
            continue
        print(f"\n{'='*80}")
        print(f"  LAYER {layer}  ({len(neurons)} neurons)")
        print(f"{'='*80}")
        for neuron_idx in sorted(neurons, key=lambda k: neurons[k].get("influence_score", 0), reverse=True):
            info = neurons[neuron_idx]
            header = f"\n  L{layer}N{neuron_idx}  {score_name}={info[score_name]:.4f}"
            if "influence_score" in info:
                header += f"  influence={info['influence_score']:.6f}"
                if "dla_score" in info:
                    header += f"  dla={info['dla_score']:.6f}"
            header += f"  ({info['num_rules']} rules, {info['total_training_samples']} training samples)"
            print(header)
            print(f"  {'-'*60}")
            for i, rule in enumerate(info["rules"]):
                strength = rule[strength_name]
                print(f"    Rule {i+1}: {rule['rule']}")
                print(f"             {strength_name}={strength:.4f}  "
                      f"samples={rule['samples']}  "
                      f"({rule['samples']/info['total_training_samples']*100:.1f}% of data)")


def main():
    parser = argparse.ArgumentParser(description="Extract IF-THEN rules from Othello-GPT neuron decision trees")
    parser.add_argument("--layer", type=int, default=None, help="Single layer to extract (0-7). Default: all layers.")
    parser.add_argument("--neuron", type=int, default=None, help="Single neuron index (0-2047). Requires --layer.")
    parser.add_argument("--top_n", type=int, default=None, help="Top N neurons per layer by quality score")
    parser.add_argument("--min_score", type=float, default=None,
                        help="Return all neurons with score strictly above this threshold (e.g. 0.99). Mutually exclusive with --top_n.")
    parser.add_argument("--top_n_influential", type=int, default=None,
                        help="Top N most influential neurons per layer (by DLA, optionally refined via ablation). Requires --min_score.")
    parser.add_argument("--precise_influence", action="store_true",
                        help="Refine influence ranking: use DLA to get 3*N candidates, then mean-ablation KL divergence to pick top N.")
    parser.add_argument("--tree_type", choices=["classification", "regression", "continuous"], default="classification",
                        help="Which decision trees to use (default: classification)")
    parser.add_argument("--min_rules", type=int, default=1, help="Exclude neurons with fewer than this many rules (default: 1)")
    parser.add_argument("--count_only", action="store_true", help="Only print the number of matching neurons per layer")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    if args.top_n is not None and args.min_score is not None:
        parser.error("--top_n and --min_score are mutually exclusive")
    if args.top_n is not None and args.top_n_influential is not None:
        parser.error("--top_n and --top_n_influential are mutually exclusive")
    if args.top_n_influential is not None and args.min_score is None:
        parser.error("--top_n_influential requires --min_score")
    if args.neuron is not None and args.layer is None:
        parser.error("--neuron requires --layer")

    layers = [args.layer] if args.layer is not None else list(range(N_LAYERS))
    if args.tree_type == "continuous":
        layers = [l for l in layers if l >= 1]  # continuous trees start at layer 1

    # Lazy model/data loading — only when influence ranking is requested
    model, game_data_BL = None, None
    if args.top_n_influential is not None:
        print("Loading model and game data for influence scoring...")
        model, game_data_BL = _load_model_and_data()

    top_n = args.top_n if args.top_n is not None else (None if args.min_score is not None else 20)
    results = extract_all_rules(
        layers, tree_type=args.tree_type, top_n=top_n, neuron_idx=args.neuron,
        min_score=args.min_score, min_rules=args.min_rules,
        top_n_influential=args.top_n_influential, precise_influence=args.precise_influence,
        model=model, game_data_BL=game_data_BL,
    )
    if args.count_only:
        total = 0
        for layer in sorted(results):
            neurons = results[layer]
            n = len(neurons)
            total += n
            # Count conditions per rule (number of ANDs + 1) for each neuron's first/largest rule
            condition_counts = []
            for info in neurons.values():
                for rule in info["rules"]:
                    condition_counts.append(rule["rule"].count(" AND ") + 1)
            if condition_counts:
                arr = np.array(condition_counts)
                print(f"  Layer {layer}: {n} neurons, {len(arr)} rules  "
                      f"(conditions per rule: mean={arr.mean():.1f}, median={np.median(arr):.1f}, std={arr.std():.1f}, min={arr.min()}, max={arr.max()})")
            else:
                print(f"  Layer {layer}: {n} neurons")
        print(f"  Total: {total} neurons")
    else:
        print_results(results, args.tree_type)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
