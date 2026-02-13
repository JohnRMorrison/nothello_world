"""
Generate boolean labels for existing othello_synthetic datasets.

Usage:
    python experiments/mathematical_transformation_experiments/generate_labels.py \
        --transform dot_product --seed 42

This will:
1. Generate a fixed transform vector (determined by seed + transform name)
2. Compute the median scalar over the first pickle file (~100k games)
3. Apply the transform to ALL pickle files, producing boolean labels
4. Save:
   - config (vector, median, metadata) to output_dir/config_<transform>_seed<seed>.pkl
   - per-file labels to output_dir/labels_<original_filename>.pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.mathematical_transformation_experiments.transforms import TRANSFORMS

SYNTHETIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "othello_synthetic")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_labels")
GAME_LEN = 60
NORMALIZE_MAX = 63.0


def normalize_and_pad(games: list[list[int]]) -> np.ndarray:
    """Normalize game move values to [0,1] and zero-pad short games to length 60."""
    out = np.zeros((len(games), GAME_LEN))
    for i, game in enumerate(games):
        n = min(len(game), GAME_LEN)
        out[i, :n] = game[:n]
    out /= NORMALIZE_MAX
    return out


def load_pickle(path: str) -> list[list[int]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def list_synthetic_files() -> list[str]:
    """Return sorted list of pickle filenames in othello_synthetic."""
    return sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pickle"))


def compute_config(transform_name: str, seed: int) -> dict:
    """
    Generate the transform vector and compute the median over the first
    pickle file (~100k games). Returns a config dict.
    """
    transform_fn = TRANSFORMS[transform_name]
    rng = np.random.default_rng(seed)

    # Load first file to compute vector + median
    files = list_synthetic_files()
    if not files:
        raise FileNotFoundError(f"No pickle files in {SYNTHETIC_DIR}")

    games_raw = load_pickle(os.path.join(SYNTHETIC_DIR, files[0]))
    games = normalize_and_pad(games_raw)

    vec, scalars = transform_fn(games, rng)
    median = float(np.median(scalars))

    print(f"Transform: {transform_name}, seed: {seed}")
    print(f"Computed median from {len(games_raw)} games (file: {files[0]}): {median:.6f}")
    print(f"Scalar stats — min: {scalars.min():.4f}, max: {scalars.max():.4f}, "
          f"mean: {scalars.mean():.4f}, std: {scalars.std():.4f}")

    return {
        "transform_name": transform_name,
        "seed": seed,
        "vector": vec,
        "median": median,
        "normalize_max": NORMALIZE_MAX,
        "game_len": GAME_LEN,
    }


def apply_transform(games_raw: list[list[int]], config: dict) -> np.ndarray:
    """Apply the saved transform vector + median to produce boolean labels."""
    games = normalize_and_pad(games_raw)
    scalars = games @ config["vector"]
    labels = (scalars > config["median"]).astype(np.int8)
    return labels


def config_filename(transform_name: str, seed: int) -> str:
    return f"config_{transform_name}_seed{seed}.pkl"


def labels_filename(original_pickle_name: str, transform_name: str, seed: int) -> str:
    """Deterministic mapping from source data file to labels file."""
    base = original_pickle_name.replace(".pickle", "")
    return f"labels_{base}_{transform_name}_seed{seed}.pkl"


def main():
    parser = argparse.ArgumentParser(description="Generate boolean labels for othello_synthetic data")
    parser.add_argument("--transform", type=str, default="dot_product", choices=list(TRANSFORMS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--recompute-config", action="store_true",
                        help="Force recompute even if config file exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: load or compute config (vector + median)
    cfg_path = os.path.join(args.output_dir, config_filename(args.transform, args.seed))
    if os.path.exists(cfg_path) and not args.recompute_config:
        print(f"Loading existing config from {cfg_path}")
        with open(cfg_path, "rb") as f:
            config = pickle.load(f)
        print(f"Transform: {config['transform_name']}, median: {config['median']:.6f}")
    else:
        config = compute_config(args.transform, args.seed)
        with open(cfg_path, "wb") as f:
            pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved config to {cfg_path}")

    # Step 2: generate labels for every synthetic pickle file
    files = list_synthetic_files()
    print(f"\nProcessing {len(files)} pickle files...")

    total_games = 0
    total_positive = 0
    for fname in tqdm(files, desc="Generating labels"):
        out_path = os.path.join(args.output_dir, labels_filename(fname, args.transform, args.seed))
        if os.path.exists(out_path):
            continue

        games_raw = load_pickle(os.path.join(SYNTHETIC_DIR, fname))
        labels = apply_transform(games_raw, config)

        with open(out_path, "wb") as f:
            pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)

        total_games += len(labels)
        total_positive += labels.sum()

    if total_games > 0:
        print(f"\nGenerated labels for {total_games} games")
        print(f"Label balance: {total_positive/total_games:.1%} positive, "
              f"{1 - total_positive/total_games:.1%} negative")
    else:
        print("\nAll label files already exist. Use --recompute-config to regenerate.")


if __name__ == "__main__":
    main()
