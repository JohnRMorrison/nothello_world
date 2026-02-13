"""
PyTorch Dataset that pairs existing othello_synthetic games with generated boolean labels.

Usage:
    from experiments.mathematical_transformation_experiments.dataset import BooleanLabelDataset

    dataset = BooleanLabelDataset(transform_name="dot_product", seed=42, max_files=5)
    train_ds, test_ds = dataset.split(train_frac=0.8)
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from experiments.mathematical_transformation_experiments.generate_labels import (
    SYNTHETIC_DIR,
    OUTPUT_DIR,
    GAME_LEN,
    NORMALIZE_MAX,
    config_filename,
    labels_filename,
    list_synthetic_files,
)


# Vocabulary matching CharDataset: 60 valid moves + 1 padding token = 61.
# Squares 27, 28, 35, 36 (center, pre-occupied) are excluded.
# Sorted order: -100 (pad) -> idx 0, then board squares in ascending order -> idx 1-60.
_VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})
STOI = {-100: 0}
STOI.update({move: i + 1 for i, move in enumerate(_VALID_MOVES)})
ITOS = {v: k for k, v in STOI.items()}
PAD_IDX = 0  # index for padding token
VOCAB_SIZE = len(STOI)  # 61


def _load_games_and_labels(transform_name, seed, labels_dir, max_files):
    """Shared loading logic: returns (games list, labels np.ndarray, config dict)."""
    games = []
    label_chunks = []

    files = list_synthetic_files()
    if max_files is not None:
        files = files[:max_files]

    for fname in files:
        lbl_path = os.path.join(labels_dir, labels_filename(fname, transform_name, seed))
        if not os.path.exists(lbl_path):
            raise FileNotFoundError(
                f"Labels not found: {lbl_path}\n"
                f"Run generate_labels.py first."
            )

        with open(os.path.join(SYNTHETIC_DIR, fname), "rb") as f:
            games_raw = pickle.load(f)
        with open(lbl_path, "rb") as f:
            labels = pickle.load(f)

        assert len(games_raw) == len(labels), (
            f"Mismatch: {fname} has {len(games_raw)} games but {len(labels)} labels"
        )

        games.extend(games_raw)
        label_chunks.append(labels)

    labels = np.concatenate(label_chunks)

    cfg_path = os.path.join(labels_dir, config_filename(transform_name, seed))
    with open(cfg_path, "rb") as f:
        config = pickle.load(f)

    return games, labels, config


class BooleanLabelDataset(Dataset):
    """Returns (x, y) where x is a float32 vector (normalized [0,1]) and y is 0/1."""

    def __init__(self, transform_name: str = "dot_product", seed: int = 42,
                 labels_dir: str = OUTPUT_DIR, max_files: int = None):
        self.games, self.labels, self.config = _load_games_and_labels(
            transform_name, seed, labels_dir, max_files
        )

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        padded = np.zeros(GAME_LEN)
        n = min(len(game), GAME_LEN)
        padded[:n] = game[:n]
        padded /= NORMALIZE_MAX

        x = torch.tensor(padded, dtype=torch.float32)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y

    def split(self, train_frac: float = 0.8, seed: int = 42):
        """Split into train/test datasets."""
        n_train = int(train_frac * len(self))
        n_test = len(self) - n_train
        return random_split(self, [n_train, n_test],
                            generator=torch.Generator().manual_seed(seed))


class BooleanLabelTokenDataset(Dataset):
    """Returns (x, y) where x uses the same vocab mapping as CharDataset (61 tokens, pad=0)."""

    def __init__(self, transform_name: str = "dot_product", seed: int = 42,
                 labels_dir: str = OUTPUT_DIR, max_files: int = None):
        self.games, self.labels, self.config = _load_games_and_labels(
            transform_name, seed, labels_dir, max_files
        )
        self.stoi = STOI
        self.itos = ITOS

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        tokens = np.full(GAME_LEN, PAD_IDX, dtype=np.int64)
        n = min(len(game), GAME_LEN)
        for i in range(n):
            tokens[i] = STOI[game[i]]

        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y

    def split(self, train_frac: float = 0.8, seed: int = 42):
        n_train = int(train_frac * len(self))
        n_test = len(self) - n_train
        return random_split(self, [n_train, n_test],
                            generator=torch.Generator().manual_seed(seed))
