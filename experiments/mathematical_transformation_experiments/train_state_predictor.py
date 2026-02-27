#!/usr/bin/env python
"""
Train a causal transformer to predict the SUM of the per-position state vector
derived from a mathematical transformation of Othello move sequences.

The model architecture matches Othello-GPT (causal transformer), but the
training target is a single scalar at each position: the sum of a
60-dimensional binary state vector (i.e. count of 1s, ranging from 0 to 60).
The state is computed via a deterministic flip rule using a fixed random lookup
table (V_table). The V_table has no spatial or strategic relationship to Othello.

State update rule (1-indexed steps):
  - Initialize s = [+1, +1, ..., +1] (length 60)
  - t=1: for each cell c, if V_table[m_1][c] == 1, flip s[c]
  - t=2: for each cell c, if V_table[m_2][c] == 1, flip s[c]
  - t>=3: for each cell c, if at least 2 of {V_table[m_t][c],
           V_table[m_{t-1}][c], V_table[m_{t-2}][c]} are 1, flip s[c]
  - State at each step: s_t mapped to {0,1} (+1 -> 1, -1 -> 0)
  - Target at each step: sum(s_t), an integer in [0, 60]

The hypothesis is that predicting this scalar sum (via MSE regression) provides
cleaner gradients than predicting the full 60-dim binary state vector.

Run from the project root:

    python experiments/mathematical_transformation_experiments/train_state_predictor.py [OPTIONS]

Options:
    --layers 8           Number of transformer layers (default: 8)
    --n-embd 512         Embedding dimension (default: 512)
    --n-head 8           Number of attention heads (default: 8)
    --epochs 20          Training epochs (default: 20)
    --batch-size 256     Batch size (default: 256)
    --lr 5e-4            Learning rate (default: 5e-4)
    --max-files 5        Number of data pickle files to load (default: 5, ~100K games each)
    --vtable-seed 42     Seed for V_table generation (default: 42)
    --seed 42            Training random seed (default: 42)
    --ckpt-dir ...       Checkpoint directory (default: auto)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.mathematical_transformation_experiments.dataset import (
    GAME_LEN,
    PAD_IDX,
    STOI,
    VOCAB_SIZE,
)
from experiments.mathematical_transformation_experiments.generate_labels import (
    SYNTHETIC_DIR,
    list_synthetic_files,
    load_pickle,
)
from mingpt.model import GPT, GPTConfig

STATE_DIM = 60
VTABLE_DIR = os.path.join(SCRIPT_DIR, "vtables")


# ============================= V_table ========================================

def generate_vtable(seed=42):
    """Generate V_table of shape (VOCAB_SIZE, STATE_DIM) with binary entries.

    Row 0 (padding token) is all zeros. Rows 1-60 (valid moves) are random 0/1.
    """
    rng = np.random.default_rng(seed)
    vtable = np.zeros((VOCAB_SIZE, STATE_DIM), dtype=np.int8)
    vtable[1:] = rng.integers(0, 2, size=(VOCAB_SIZE - 1, STATE_DIM)).astype(np.int8)
    return vtable


def get_vtable(seed=42):
    """Load V_table from disk if it exists, otherwise generate and save it."""
    os.makedirs(VTABLE_DIR, exist_ok=True)
    path = os.path.join(VTABLE_DIR, f"vtable_seed{seed}.npy")
    if os.path.exists(path):
        vtable = np.load(path)
        print(f"Loaded V_table from {path}")
    else:
        vtable = generate_vtable(seed)
        np.save(path, vtable)
        print(f"Generated and saved V_table to {path}")
    return vtable


# ============================= State computation ==============================

def compute_states(token_seqs, vtable):
    """Compute per-position state vectors for a batch of tokenized games.

    Uses a sliding window for V_table lookups to keep memory usage low.
    Vectorized across games (N) and state dimensions (STATE_DIM), sequential
    across time steps (unavoidable due to state dependency).

    Args:
        token_seqs: (N, T) int array of token indices
        vtable: (VOCAB_SIZE, STATE_DIM) int8 array

    Returns:
        states: (N, T, STATE_DIM) uint8 array, values in {0, 1}
    """
    N, T = token_seqs.shape
    non_pad = (token_seqs != PAD_IDX)  # (N, T)

    s = np.ones((N, STATE_DIM), dtype=np.float32)
    states = np.zeros((N, T, STATE_DIM), dtype=np.uint8)

    v_prev2 = None
    v_prev1 = None

    for t in range(T):
        v_t = vtable[token_seqs[:, t]]  # (N, STATE_DIM)
        active = non_pad[:, t : t + 1]  # (N, 1)

        if t <= 1:
            flip = (v_t == 1) & active
        else:
            vote = (
                v_t.astype(np.int32)
                + v_prev1.astype(np.int32)
                + v_prev2.astype(np.int32)
            )
            flip = (vote >= 2) & active

        s = np.where(flip, -s, s)
        states[:, t, :] = ((s + 1) / 2).astype(np.uint8)

        v_prev2 = v_prev1
        v_prev1 = v_t

    return states


# ============================= Dataset ========================================

def tokenize_game(game_raw):
    """Convert a raw game (list of move IDs) to token indices."""
    tokens = np.full(GAME_LEN, PAD_IDX, dtype=np.int64)
    n = min(len(game_raw), GAME_LEN)
    for j in range(n):
        tokens[j] = STOI[game_raw[j]]
    return tokens


class StateDataset(Dataset):
    """Othello token sequences paired with per-position state-sum targets."""

    def __init__(self, vtable, max_files=None):
        all_tokens = []
        all_sums = []

        files = list_synthetic_files()
        if max_files is not None:
            files = files[:max_files]

        for fname in tqdm(files, desc="Loading games"):
            games_raw = load_pickle(os.path.join(SYNTHETIC_DIR, fname))
            tokens = np.array(
                [tokenize_game(g) for g in games_raw], dtype=np.int64
            )
            states = compute_states(tokens, vtable)  # (N, T, STATE_DIM)
            state_sums = states.sum(axis=-1).astype(np.float32)  # (N, T)

            all_tokens.append(tokens)
            all_sums.append(state_sums)

        self.tokens = np.concatenate(all_tokens, axis=0)  # (N, GAME_LEN)
        self.state_sums = np.concatenate(all_sums, axis=0)  # (N, GAME_LEN)

        print(
            f"Dataset: {len(self)} games, "
            f"tokens {self.tokens.shape}, state_sums {self.state_sums.shape}, "
            f"sum range [{self.state_sums.min():.0f}, {self.state_sums.max():.0f}]"
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.tokens[idx].copy())
        y = torch.from_numpy(self.state_sums[idx].copy())
        return x, y  # (GAME_LEN,) long, (GAME_LEN,) float32

    def split(self, train_frac=0.8, seed=42):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self))
        n_train = int(len(self) * train_frac)
        return (
            Subset(self, perm[:n_train].tolist()),
            Subset(self, perm[n_train:].tolist()),
        )


# ============================= Model ==========================================

class GPTStateSumPredictor(GPT):
    """Causal transformer predicting the sum of the state vector at each position.

    Same architecture as Othello-GPT, but the output head produces a single
    scalar per position (the predicted sum, in [0, 60]) instead of vocab_size
    logits for next-token prediction. Trained with MSE loss.
    """

    def __init__(self, config):
        super().__init__(config)
        self.head = nn.Linear(config.n_embd, 1, bias=True)
        self.head.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.constant_(self.head.bias, STATE_DIM / 2.0)  # init near mean

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)  # (B, T, n_embd)

        pred = self.head(x).squeeze(-1)  # (B, T)
        return pred


# ============================= Device =========================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================= Training =======================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_elements = 0

    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)  # x: (B, T), y: (B, T)
        pred = model(x)  # (B, T)

        non_pad = (x != PAD_IDX)  # (B, T)

        loss = nn.functional.mse_loss(pred[non_pad], y[non_pad])

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n = non_pad.sum().item()
        total_loss += loss.item() * n
        total_mae += (pred[non_pad] - y[non_pad]).abs().sum().item()
        total_elements += n

    return total_loss / total_elements, total_mae / total_elements


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_elements = 0

    for x, y in tqdm(loader, desc="  eval", leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x)

        non_pad = (x != PAD_IDX)

        loss = nn.functional.mse_loss(pred[non_pad], y[non_pad])

        n = non_pad.sum().item()
        total_loss += loss.item() * n
        total_mae += (pred[non_pad] - y[non_pad]).abs().sum().item()
        total_elements += n

    return total_loss / total_elements, total_mae / total_elements


# ============================= Main ===========================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train causal transformer on mathematical state prediction task"
    )
    p.add_argument("--layers", type=int, default=8, help="Number of transformer layers")
    p.add_argument("--n-embd", type=int, default=512, help="Embedding dimension")
    p.add_argument("--n-head", type=int, default=8, help="Number of attention heads")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--max-files", type=int, default=5,
        help="Data files to load (None=all). ~100K games per file.",
    )
    p.add_argument(
        "--vtable-seed", type=int, default=42,
        help="Seed for V_table generation (default: 42)",
    )
    p.add_argument("--seed", type=int, default=42, help="Training random seed")
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument(
        "--save-random-init", action="store_true",
        help="Save a randomly initialized (untrained) model checkpoint and exit",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # ---- V_table ----
    vtable = get_vtable(seed=args.vtable_seed)
    print(f"V_table shape: {vtable.shape} (density of valid rows: {vtable[1:].mean():.3f})")

    # ---- Checkpoint dir ----
    if args.ckpt_dir is None:
        name = f"state_sum_vseed{args.vtable_seed}_{args.layers}L_{args.n_embd}d"
        args.ckpt_dir = os.path.join(SCRIPT_DIR, "ckpts", name)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- Save random init and exit if requested ----
    if args.save_random_init:
        config = GPTConfig(
            VOCAB_SIZE, GAME_LEN,
            n_layer=args.layers, n_head=args.n_head, n_embd=args.n_embd,
        )
        model = GPTStateSumPredictor(config)
        init_path = os.path.join(args.ckpt_dir, "random_init.pt")
        torch.save(model.state_dict(), init_path)
        with open(os.path.join(args.ckpt_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Saved random init to {init_path}")
        print(
            f"Model: {args.layers}L / {args.n_embd}d / {args.n_head}h — {n_params:,} params"
        )
        return

    # ---- Data ----
    print("Loading dataset...")
    t0 = time.time()
    dataset = StateDataset(vtable=vtable, max_files=args.max_files)
    train_ds, test_ds = dataset.split(train_frac=args.train_frac, seed=args.seed)
    print(
        f"Loaded {len(dataset)} games in {time.time() - t0:.1f}s "
        f"(train={len(train_ds)}, test={len(test_ds)})"
    )

    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, shuffle=False, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    config = GPTConfig(
        VOCAB_SIZE, GAME_LEN,
        n_layer=args.layers, n_head=args.n_head, n_embd=args.n_embd,
    )
    model = GPTStateSumPredictor(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: {args.layers}L / {args.n_embd}d / {args.n_head}h — {n_params:,} params"
    )

    class OptimConfig:
        learning_rate = args.lr
        weight_decay = 0.1
        betas = (0.9, 0.95)

    optimizer = model.configure_optimizers(OptimConfig())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Save args + vtable path ----
    args_dict = vars(args)
    args_dict["vtable_path"] = os.path.join(VTABLE_DIR, f"vtable_seed{args.vtable_seed}.npy")
    with open(os.path.join(args.ckpt_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    # ---- Train ----
    history = []
    best_test_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.2e})")

        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_mae = evaluate(model, test_loader, device)
        scheduler.step()

        print(f"  train mse={train_loss:.4f}  mae={train_mae:.4f}")
        print(f"  test  mse={test_loss:.4f}  mae={test_mae:.4f}")

        record = {
            "epoch": epoch,
            "train_loss": train_loss, "train_mae": train_mae,
            "test_loss": test_loss, "test_mae": test_mae,
        }
        history.append(record)

        ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_loss": test_loss,
            "test_mae": test_mae,
        }, ckpt_path)

        if test_mae < best_test_mae:
            best_test_mae = test_mae
            best_path = os.path.join(args.ckpt_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  ** new best test MAE: {best_test_mae:.4f} — saved to {best_path}")

    # ---- Save history ----
    hist_path = os.path.join(args.ckpt_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best test MAE: {best_test_mae:.4f}")
    print(f"Checkpoints: {args.ckpt_dir}")


if __name__ == "__main__":
    main()
