"""Train MLP on move features using chunk-streaming (constant memory).

Loads one chunk at a time — scales to any number of games.

Usage:
    python train_streaming.py --hidden 1024 --epochs 10 [--features when+even]
"""
import sys, os, json
sys.path.insert(0, '.')

import argparse
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _train_mlp_streaming, get_device, N_MOVES,
)

parser = argparse.ArgumentParser()
parser.add_argument("--hidden", type=int, default=1024)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--features", type=str, default="all",
                    choices=["all", "when+even", "played+when", "played+even",
                             "when", "played", "even"],
                    help="Which feature subset to use")
parser.add_argument("--output-dir",
                    default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
args = parser.parse_args()

# Feature column selections
FEATURE_SUBSETS = {
    "all":          (list(range(3 * N_MOVES)), 3 * N_MOVES),
    "when+even":    (list(range(N_MOVES, 3 * N_MOVES)), 2 * N_MOVES),
    "played+when":  (list(range(0, 2 * N_MOVES)), 2 * N_MOVES),
    "played+even":  (list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)), 2 * N_MOVES),
    "when":         (list(range(N_MOVES, 2 * N_MOVES)), N_MOVES),
    "played":       (list(range(0, N_MOVES)), N_MOVES),
    "even":         (list(range(2 * N_MOVES, 3 * N_MOVES)), N_MOVES),
}

feature_cols, input_dim = FEATURE_SUBSETS[args.features]
if args.features == "all":
    feature_cols = None  # no slicing needed
    input_dim = 3 * N_MOVES

device = get_device()
print(f"Device: {device}")
print(f"Features: {args.features} ({input_dim}-d), H={args.hidden}, {args.epochs} epochs")

chunk_dir = os.path.join(args.output_dir, "feature_chunks")

save_dir = os.path.join(args.output_dir, "mlp_checkpoints")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"mlp_{args.features}_H{args.hidden}_streaming.pt")

acc = _train_mlp_streaming(
    chunk_dir, device, input_dim, args.hidden,
    feature_cols=feature_cols,
    epochs=args.epochs,
    save_path=save_path,
)

print(f"\nFinal: {args.features} H={args.hidden}: {acc:.4%}")
