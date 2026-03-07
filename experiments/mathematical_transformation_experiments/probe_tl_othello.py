#!/usr/bin/env python
"""
Probe Othello-GPT for board state using TransformerLens.

This replicates Nanda/Teo's approach exactly:
- Load model from HuggingFace via TransformerLens
- Extract resid_post activations
- Train Nanda-style 3-mode linear probe

Usage:
    python -m experiments.mathematical_transformation_experiments.probe_tl_othello \
        --layer 6 --max-games 100000 --probe-epochs 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
import importlib.util

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import OthelloBoardState without triggering data/__init__.py
sys.modules["pgn"] = types.ModuleType("pgn")
_spec = importlib.util.spec_from_file_location(
    "othello_module", os.path.join(PROJECT_ROOT, "data", "othello.py"),
    submodule_search_locations=[],
)
_othello_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_othello_mod)
OthelloBoardState = _othello_mod.OthelloBoardState

from transformer_lens import HookedTransformer, HookedTransformerConfig
import transformer_lens.utils as tl_utils

GAME_LEN = 60
ROWS, COLS = 8, 8
OPTIONS = 3
MODES = 3

SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "othello_synthetic")

# Valid moves: 0-63 excluding center squares {27, 28, 35, 36}
_VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_games(max_files=None):
    import pickle
    files = sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pickle"))
    if max_files is not None:
        files = files[:max_files]
    games = []
    for fname in files:
        with open(os.path.join(SYNTHETIC_DIR, fname), "rb") as f:
            batch = pickle.load(f)
        games.extend(g for g in batch if len(g) == GAME_LEN)
    return games


def games_to_int_tensor(game_list):
    """Convert raw board-position games to TransformerLens int encoding."""
    batch = torch.zeros((len(game_list), GAME_LEN), dtype=torch.long)
    for i, game in enumerate(game_list):
        for j, move in enumerate(game):
            if move < 27:
                batch[i, j] = move + 1
            elif move < 35:
                # moves 27,28 are invalid, so only 29-34 appear
                batch[i, j] = move - 1
            else:
                # moves 35,36 are invalid, so only 37-63 appear
                batch[i, j] = move - 3
    return batch


def seq_to_state_stack(raw_moves):
    """Compute board state after each move."""
    board = OthelloBoardState()
    states = []
    for move in raw_moves:
        board.umpire(move)
        states.append(np.copy(board.state).astype(np.int8))
    return np.stack(states, axis=0)  # (T, 8, 8)


def train_board_probe(model, games, device, layer,
                      pos_start=5, pos_end=54, batch_size=100,
                      lr=1e-4, epochs=2):
    """Train Nanda-style 3-mode linear probe using TransformerLens activations."""
    d_model = model.cfg.d_model
    length = pos_end - pos_start

    linear_probe = torch.randn(
        MODES, d_model, ROWS, COLS, OPTIONS, device=device,
    ) / np.sqrt(d_model)
    linear_probe.requires_grad = True

    optimizer = torch.optim.AdamW(
        [linear_probe], lr=lr, betas=(0.9, 0.99), weight_decay=0.01,
    )

    num_games = len(games)
    n_eval = max(int(num_games * 0.1), batch_size)
    n_train = num_games - n_eval
    train_games = games[:n_train]
    eval_games = games[n_train:]

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_train)
        for i in tqdm(range(0, n_train, batch_size), desc=f"  epoch {epoch}", leave=False):
            idx = perm[i:i + batch_size]
            batch_games = [train_games[j] for j in idx]

            # Get int-encoded input (drop last token for autoregressive)
            input_tensor = games_to_int_tensor(batch_games).to(device)
            input_tensor = input_tensor[:, :-1]  # (B, 59) — matches Nanda exactly

            # Get ground truth states
            state_stack = torch.stack([
                torch.tensor(seq_to_state_stack(g)) for g in batch_games
            ])
            state_stack = state_stack[:, pos_start:pos_end, :, :]

            # One-hot encode: 0=empty, 1=white, 2=black
            one_hot = torch.zeros(
                MODES, len(batch_games), pos_end - pos_start,
                ROWS, COLS, OPTIONS, device=device, dtype=torch.int,
            )
            one_hot[:, ..., 0] = state_stack == 0
            one_hot[:, ..., 1] = state_stack == -1
            one_hot[:, ..., 2] = state_stack == 1

            # Extract activations using TransformerLens
            with torch.no_grad():
                _, cache = model.run_with_cache(input_tensor, return_type=None)
                acts = cache["resid_post", layer][:, pos_start:pos_end]

            probe_out = torch.einsum("bpd,mdrco->mbprco", acts, linear_probe)
            probe_log_probs = probe_out.log_softmax(-1)
            probe_correct_log_probs = (
                (probe_log_probs * one_hot).mean(dim=1).sum(dim=-1) * OPTIONS
            )
            loss_even = -probe_correct_log_probs[0, 0::2].mean(0).sum()
            loss_odd = -probe_correct_log_probs[1, 1::2].mean(0).sum()
            loss_all = -probe_correct_log_probs[2, :].mean(0).sum()
            loss = loss_even + loss_odd + loss_all

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Eval
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for i in range(0, n_eval, batch_size):
                batch_games = eval_games[i:i + batch_size]
                input_tensor = games_to_int_tensor(batch_games).to(device)
                input_tensor = input_tensor[:, :-1]

                state_stack = torch.stack([
                    torch.tensor(seq_to_state_stack(g)) for g in batch_games
                ])
                state_stack = state_stack[:, pos_start:pos_end, :, :]

                _, cache = model.run_with_cache(input_tensor, return_type=None)
                acts = cache["resid_post", layer][:, pos_start:pos_end]

                probe_out = torch.einsum("bpd,mdrco->mbprco", acts, linear_probe)
                preds = probe_out[2].argmax(dim=-1)

                t_state = state_stack.to(device)
                targets = torch.zeros_like(t_state, dtype=torch.long)
                targets[t_state == 0] = 0
                targets[t_state == -1] = 1
                targets[t_state == 1] = 2
                eval_correct += (preds == targets).sum().item()
                eval_total += targets.numel()

        acc = eval_correct / eval_total
        best_acc = max(best_acc, acc)
        print(f"  Epoch {epoch}: eval acc={acc:.4%}", flush=True)

    return best_acc


def parse_args():
    p = argparse.ArgumentParser(description="Probe Othello-GPT via TransformerLens")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--model", type=str, default="synthetic",
                    choices=["synthetic", "championship"])
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--max-games", type=int, default=100000)
    p.add_argument("--probe-epochs", type=int, default=2)
    p.add_argument("--probe-batch-size", type=int, default=100)
    p.add_argument("--probe-lr", type=float, default=1e-4)
    p.add_argument("--pos-start", type=int, default=5)
    p.add_argument("--pos-end", type=int, default=54)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # Load TransformerLens model
    cfg = HookedTransformerConfig(
        n_layers=8,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=61,
        n_ctx=59,
        act_fn="gelu",
        normalization_type="LNPre",
    )
    model = HookedTransformer(cfg)
    model_file = f"{args.model}_model.pth"
    print(f"Downloading {model_file} from HuggingFace...")
    sd = tl_utils.download_file_from_hf(
        "NeelNanda/Othello-GPT-Transformer-Lens", model_file
    )
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    print(f"Loaded TransformerLens model: {model.cfg.n_layers} layers, {model.cfg.d_model} dims")

    # Load games
    print("Loading games...")
    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    print(f"Using {len(games)} games, probing layer {args.layer}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Probing layer {args.layer} (resid_post) for Othello board state", flush=True)
    print(f"{'='*60}", flush=True)
    acc = train_board_probe(
        model, games, device, args.layer,
        pos_start=args.pos_start, pos_end=args.pos_end,
        batch_size=args.probe_batch_size, lr=args.probe_lr,
        epochs=args.probe_epochs,
    )

    above = acc - 1/3
    print(f"\nLayer {args.layer}: acc={acc:.4%} (chance=33.3%, +{above:.4%})", flush=True)

    # Save
    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "tl_probe_results")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"results_layer{args.layer}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "backend": "transformer_lens",
            "layer": args.layer,
            "num_games": len(games),
            "accuracy": acc,
            "chance": 1/3,
        }, f, indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
