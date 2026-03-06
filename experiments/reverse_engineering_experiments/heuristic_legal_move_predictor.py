"""
Test whether heuristic IF-THEN rules extracted from Othello-GPT neurons carry
enough information to predict legal moves — an interpretability proof-of-concept.

Usage:
    # First generate rules:
    python extract_rules.py --layers 5 --min_score 0.9 --save rules.json

    # Then run this predictor:
    python heuristic_legal_move_predictor.py --rules rules.json
    python heuristic_legal_move_predictor.py --rules rules.json --n_games 500 --epochs 20
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from OthelloReverseEngineering.utils.circuits_utils import construct_othello_dataset
from OthelloReverseEngineering.utils.othello_utils import (
    games_batch_to_board_state_flipped_played_BLC,
    games_batch_to_valid_moves_BLRRC,
)
from OthelloReverseEngineering.utils.feature_extraction_utils import create_feature_names


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

def load_and_parse_rules(rules_path: str):
    """Load JSON rules file and parse rule strings into structured conditions.

    Returns:
        parsed_rules: dict mapping (layer, neuron) -> list of rules,
            where each rule is a list of (feature_index, polarity_bool) tuples.
            polarity_bool=True means feature must be 1, False means feature must be 0.
        neuron_keys: sorted list of (layer, neuron) keys
    """
    feature_names = create_feature_names(320, "games_batch_to_board_state_flipped_played_BLC")
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    with open(rules_path) as f:
        raw = json.load(f)

    parsed_rules = {}

    for layer_str, neurons in raw.items():
        layer = int(layer_str)
        for neuron_str, info in neurons.items():
            neuron = int(neuron_str)
            neuron_rules = []
            for rule_entry in info["rules"]:
                rule_str = rule_entry["rule"]
                if not rule_str.strip():
                    continue
                conditions = _parse_rule_string(rule_str, name_to_idx)
                if conditions is not None:
                    neuron_rules.append(conditions)
            if neuron_rules:
                parsed_rules[(layer, neuron)] = neuron_rules

    neuron_keys = sorted(parsed_rules.keys())
    print(f"Loaded {len(neuron_keys)} neurons with parseable rules "
          f"(total {sum(len(v) for v in parsed_rules.values())} rules)")
    return parsed_rules, neuron_keys


def _parse_rule_string(rule_str: str, name_to_idx: dict):
    """Parse e.g. '(A5_mine) AND (NOT B3_theirs)' into [(idx, True/False), ...]."""
    parts = rule_str.split(" AND ")
    conditions = []
    for part in parts:
        part = part.strip()
        # Match (NOT feature_name) or (feature_name)
        m = re.match(r"^\((?:NOT\s+)?(.+?)\)$", part)
        if not m:
            return None  # unparseable
        feature_name = m.group(1).strip()
        polarity = "NOT " not in part
        if feature_name not in name_to_idx:
            return None  # unknown feature
        conditions.append((name_to_idx[feature_name], polarity))
    return conditions


# ---------------------------------------------------------------------------
# Heuristic evaluation (vectorized)
# ---------------------------------------------------------------------------

def evaluate_heuristics(features: torch.Tensor, parsed_rules: dict,
                        neuron_keys: list, chunk_size: int = 50000) -> torch.Tensor:
    """Evaluate all heuristic rules on feature vectors, processing in chunks.

    Args:
        features: (N, 320) binary feature tensor (CPU)
        parsed_rules: from load_and_parse_rules
        neuron_keys: sorted list of (layer, neuron) keys
        chunk_size: number of positions to process at once

    Returns:
        (N, num_neurons) float tensor of boolean heuristic activations (CPU)
    """
    N = features.shape[0]
    num_neurons = len(neuron_keys)
    result = torch.zeros(N, num_neurons)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = features[start:end]
        C = chunk.shape[0]

        for ni, key in enumerate(neuron_keys):
            rules = parsed_rules[key]
            neuron_activation = torch.zeros(C, dtype=torch.bool)

            for conditions in rules:
                rule_match = torch.ones(C, dtype=torch.bool)
                for feat_idx, polarity in conditions:
                    feat_vals = chunk[:, feat_idx] > 0.5
                    if polarity:
                        rule_match &= feat_vals
                    else:
                        rule_match &= ~feat_vals
                neuron_activation |= rule_match

            result[start:end, ni] = neuron_activation.float()

        print(f"  Heuristic eval: {end}/{N} positions done")

    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_linear_model(X_train, y_train, X_test, y_test,
                       epochs=20, batch_size=256, lr=1e-3, label="Model",
                       device="cpu"):
    """Train a linear layer with BCE loss and return test metrics.

    Data is kept on CPU; only batches are moved to device during training/eval.
    """
    input_dim = X_train.shape[1]

    model = nn.Linear(input_dim, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = total_loss / n_batches
            print(f"  [{label}] Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

    # Evaluate in batches to avoid OOM
    model.eval()
    all_preds = []
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu()
            all_preds.append(preds)

    all_preds = torch.cat(all_preds, dim=0)
    return compute_metrics(all_preds, y_test, label)


def compute_metrics(preds, targets, label="Model"):
    """Compute and return classification metrics for legal move prediction."""
    # Per-square accuracy
    correct = (preds == targets).float()
    per_square_acc = correct.mean().item()

    # Precision / Recall / F1 on "legal" class
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Exact match: all 64 squares correct
    exact_match = (preds == targets).all(dim=1).float().mean().item()

    metrics = {
        "per_square_accuracy": per_square_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
    }

    print(f"\n  [{label}] Results:")
    print(f"    Per-square accuracy : {per_square_acc:.4f}")
    print(f"    Precision (legal)   : {precision:.4f}")
    print(f"    Recall (legal)      : {recall:.4f}")
    print(f"    F1 (legal)          : {f1:.4f}")
    print(f"    Exact-match accuracy: {exact_match:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test whether heuristic rules predict legal moves")
    parser.add_argument("--rules", type=str, required=True,
                        help="Path to JSON rules file from extract_rules.py --save")
    parser.add_argument("--n_games", type=int, default=1000,
                        help="Number of games to load (default: 1000)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto)")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # --- Load rules ---
    parsed_rules, neuron_keys = load_and_parse_rules(args.rules)
    num_neurons = len(neuron_keys)

    # --- Load game data ---
    print(f"\nLoading {args.n_games} games...")
    dataset = construct_othello_dataset(
        custom_functions=[],
        n_inputs=args.n_games,
        split="train",
        precompute_dataset=False,
    )
    decoded_inputs = dataset["decoded_inputs"]

    print("Computing features (320-dim)...")
    features_BLC = games_batch_to_board_state_flipped_played_BLC(decoded_inputs)
    print("Computing legal moves...")
    valid_moves_BLRRC = games_batch_to_valid_moves_BLRRC(decoded_inputs)

    # Reshape: (B, L, 8, 8, 1) -> (B, L, 64)
    B, L = features_BLC.shape[:2]
    valid_moves_BL64 = valid_moves_BLRRC.reshape(B, L, 64)

    # Flatten to (B*L, ...)
    features_flat = features_BLC.reshape(-1, 320).float()
    labels_flat = valid_moves_BL64.reshape(-1, 64).float()

    print(f"Dataset: {features_flat.shape[0]} positions, "
          f"{labels_flat.sum().item():.0f} total legal squares "
          f"({labels_flat.mean().item()*100:.1f}% positive rate)")

    # --- Train/test split (all data stays on CPU) ---
    N = features_flat.shape[0]
    perm = torch.randperm(N)
    split = int(0.8 * N)
    train_idx, test_idx = perm[:split], perm[split:]

    X_train_raw = features_flat[train_idx]
    X_test_raw = features_flat[test_idx]
    y_train = labels_flat[train_idx]
    y_test = labels_flat[test_idx]

    # --- Evaluate heuristics (on CPU, in chunks) ---
    print(f"\nEvaluating heuristic rules on {N} positions ({num_neurons} neurons)...")
    heuristic_all = evaluate_heuristics(features_flat, parsed_rules, neuron_keys)
    X_train_heur = heuristic_all[train_idx]
    X_test_heur = heuristic_all[test_idx]

    active_frac = heuristic_all.mean().item()
    print(f"Heuristic activation rate: {active_frac*100:.1f}%")

    del features_flat, labels_flat, heuristic_all  # free memory

    # --- Train heuristic model ---
    print(f"\n{'='*60}")
    print("Training HEURISTIC model (rules -> legal moves)")
    print(f"{'='*60}")
    heur_metrics = train_linear_model(
        X_train_heur, y_train, X_test_heur, y_test,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        label="Heuristic", device=device,
    )

    # --- Train baseline model ---
    print(f"\n{'='*60}")
    print("Training BASELINE model (raw 320 features -> legal moves)")
    print(f"{'='*60}")
    base_metrics = train_linear_model(
        X_train_raw, y_train, X_test_raw, y_test,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        label="Baseline", device=device,
    )

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Heuristic':>12} {'Baseline':>12} {'Ratio':>8}")
    print(f"{'-'*57}")
    for key in ["per_square_accuracy", "precision", "recall", "f1", "exact_match"]:
        h = heur_metrics[key]
        b = base_metrics[key]
        ratio = h / b if b > 0 else float("inf")
        print(f"{key:<25} {h:>12.4f} {b:>12.4f} {ratio:>7.2f}x")

    print(f"\nHeuristic input dim: {num_neurons}, Baseline input dim: 320")


if __name__ == "__main__":
    main()
