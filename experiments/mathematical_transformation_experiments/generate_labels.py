"""

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
from __future__ import annotations

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


def compute_config(transform_name: str, seed: int, n_transforms: int = 1) -> dict:
    """
    Generate the transform params and compute the median(s) over the first
    pickle file (~100k games). Returns a config dict.
    """
    transform = TRANSFORMS[transform_name]
    rng = np.random.default_rng(seed)

    # Load first file to compute params + median
    files = list_synthetic_files()
    if not files:
        raise FileNotFoundError(f"No pickle files in {SYNTHETIC_DIR}")

    games_raw = load_pickle(os.path.join(SYNTHETIC_DIR, files[0]))
    games = normalize_and_pad(games_raw)

    params, scalars = transform.generate(games, rng, n=n_transforms)
    median = np.median(scalars, axis=0)  # (n_transforms,)

    print(f"Transform: {transform_name}, seed: {seed}, n_transforms: {n_transforms}")
    print(f"Computed medians from {len(games_raw)} games (file: {files[0]})")
    if n_transforms <= 4:
        print(f"  medians: {median}")
    else:
        print(f"  median range: [{median.min():.4f}, {median.max():.4f}], "
              f"mean: {median.mean():.4f}")

    return {
        "transform_name": transform_name,
        "seed": seed,
        "n_transforms": n_transforms,
        "params": params,      # dict of numpy arrays (transform-specific)
        "median": median,      # (n_transforms,)
        "normalize_max": NORMALIZE_MAX,
        "game_len": GAME_LEN,
    }


def apply_transform(games_raw: list[list[int]], config: dict) -> np.ndarray:
    """Apply the saved transform params + median(s) to produce boolean labels.

    Returns shape (N, n_transforms) int8 array.
    """
    games = normalize_and_pad(games_raw)

    # Backward compat: old configs store "vector" instead of "params"
    if "params" not in config:
        params = {"vector": config["vector"]}
    else:
        params = config["params"]

    transform = TRANSFORMS[config["transform_name"]]
    scalars = transform.apply(games, params)   # (N, n_transforms)
    labels = (scalars > config["median"]).astype(np.int8)  # (N, n_transforms)
    return labels


def config_filename(transform_name: str, seed: int, n_transforms: int = 1) -> str:
    base = f"config_{transform_name}_seed{seed}"
    if n_transforms > 1:
        base += f"_n{n_transforms}"
    return base + ".pkl"


def labels_filename(original_pickle_name: str, transform_name: str, seed: int,
                    n_transforms: int = 1) -> str:
    """Deterministic mapping from source data file to labels file."""
    base = original_pickle_name.replace(".pickle", "")
    name = f"labels_{base}_{transform_name}_seed{seed}"
    if n_transforms > 1:
        name += f"_n{n_transforms}"
    return name + ".pkl"


def main():
    parser = argparse.ArgumentParser(description="Generate boolean labels for othello_synthetic data")
    parser.add_argument("--transform", type=str, default="dot_product", choices=list(TRANSFORMS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-transforms", type=int, default=1,
                        help="Number of independent random transforms (default: 1)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--recompute-config", action="store_true",
                        help="Force recompute even if config file exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    n = args.n_transforms

    # Step 1: load or compute config (vector(s) + median(s))
    cfg_path = os.path.join(args.output_dir, config_filename(args.transform, args.seed, n))
    if os.path.exists(cfg_path) and not args.recompute_config:
        print(f"Loading existing config from {cfg_path}")
        with open(cfg_path, "rb") as f:
            config = pickle.load(f)
        print(f"Transform: {config['transform_name']}, "
              f"n_transforms: {config.get('n_transforms', 1)}")
    else:
        config = compute_config(args.transform, args.seed, n_transforms=n)
        with open(cfg_path, "wb") as f:
            pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved config to {cfg_path}")

    # Step 2: generate labels for every synthetic pickle file
    files = list_synthetic_files()
    print(f"\nProcessing {len(files)} pickle files...")

    total_games = 0
    total_positive = np.zeros(n)
    for fname in tqdm(files, desc="Generating labels"):
        out_path = os.path.join(
            args.output_dir, labels_filename(fname, args.transform, args.seed, n)
        )
        if os.path.exists(out_path):
            continue

        games_raw = load_pickle(os.path.join(SYNTHETIC_DIR, fname))
        labels = apply_transform(games_raw, config)  # (N, n_transforms)

        with open(out_path, "wb") as f:
            pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)

        total_games += len(labels)
        total_positive += labels.sum(axis=0)

    if total_games > 0:
        frac = total_positive / total_games
        print(f"\nGenerated labels for {total_games} games ({n} transforms)")
        if n <= 4:
            for i in range(n):
                print(f"  transform {i}: {frac[i]:.1%} positive")
        else:
            print(f"  positive fraction: min={frac.min():.1%}, max={frac.max():.1%}, "
                  f"mean={frac.mean():.1%}")
    else:
        print("\nAll label files already exist. Use --recompute-config to regenerate.")


if __name__ == "__main__":
    main()
