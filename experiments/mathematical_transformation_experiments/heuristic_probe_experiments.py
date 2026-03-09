#!/usr/bin/env python
"""
Heuristic & baseline probing experiments for Othello-GPT.

Experiments:
  standard_probe  (D) Standard linear probe without Nanda's even/odd split
  resid_pre       (B) Probe on raw embeddings before any transformer blocks
  alt_boards      (C) Decode alternative board states from layer 0
  by_move         (A) Per-move-number accuracy for Normal vs No Flip
  heuristic       (1) Heuristic features → board state

Usage (from project root):
    python -m experiments.mathematical_transformation_experiments.heuristic_probe_experiments \
        --experiment standard_probe --max-games 10000
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.mathematical_transformation_experiments.probe_variant_boards import (
    load_games, tokenize_games, extract_activations, seq_to_state_normal,
    seq_to_state_no_flip, get_device, STOI, ITOS, GAME_LEN, VOCAB_SIZE,
    PAD_IDX, ROWS, COLS, OPTIONS, SYNTHETIC_DIR,
)
from mingpt.model import GPT, GPTConfig

POS_START = 5
POS_END = 54
LENGTH = POS_END - POS_START  # 49


# ============================= Shared Utilities ==============================

def load_model(ckpt_path, device, n_layer=8, n_head=8, n_embd=512):
    """Load a mingpt model from checkpoint."""
    state_dict = torch.load(ckpt_path, map_location=device)
    block_size = state_dict["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    model = GPT(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, block_size


def create_random_model(device, n_layer=8, n_head=8, n_embd=512, block_size=59):
    """Create a randomly initialized model (no training)."""
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    model = GPT(config)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_activations_pre(model, x):
    """Extract raw embeddings (token + positional) before any transformer blocks."""
    b, t = x.size()
    tok = model.tok_emb(x)
    pos = model.pos_emb[:, :t, :]
    h = model.drop(tok + pos)
    return h


def get_board_states(games, simulator, pos_start, pos_end):
    """Compute board states for games. Returns (N_games, length, 8, 8) int8 array."""
    stacks = []
    for game in games:
        states = simulator(game)  # (T, 8, 8)
        stacks.append(states[pos_start:pos_end])
    return np.stack(stacks, axis=0)


def states_to_labels(state_stack):
    """Convert state_stack (-1/0/1) to class labels (0=empty, 1=white, 2=black)."""
    labels = np.zeros_like(state_stack, dtype=np.int64)
    labels[state_stack == -1] = 1  # white
    labels[state_stack == 0] = 0   # empty
    labels[state_stack == 1] = 2   # black
    return labels


# ============================= Standard Linear Probe ========================

def train_standard_probe(train_X, train_Y, eval_X, eval_Y, device,
                         input_dim, lr=1e-3, epochs=16, batch_size=1024):
    """Train nn.Linear(input_dim, 64*3) with cross-entropy. No even/odd split.

    train_X: (N, input_dim) float tensor
    train_Y: (N, 64) long tensor (class labels 0/1/2 for each cell)
    Returns best accuracy.
    """
    probe = nn.Linear(input_dim, 64 * OPTIONS).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(len(train_X))
        for i in range(0, len(train_X), batch_size):
            idx = perm[i:i + batch_size]
            x = train_X[idx].to(device)
            y = train_Y[idx].to(device)  # (B, 64)
            logits = probe(x).view(-1, 64, OPTIONS)  # (B, 64, 3)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, OPTIONS), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval
        probe.eval()
        correct = 0
        total = 0
        losses = []
        with torch.no_grad():
            for i in range(0, len(eval_X), batch_size):
                x = eval_X[i:i + batch_size].to(device)
                y = eval_Y[i:i + batch_size].to(device)
                logits = probe(x).view(-1, 64, OPTIONS)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, OPTIONS), y.reshape(-1))
                losses.append(loss.item())
                preds = logits.argmax(-1)
                correct += (preds == y).sum().item()
                total += y.numel()

        acc = correct / total
        best_acc = max(best_acc, acc)
        scheduler.step(np.mean(losses))
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: acc={acc:.4%}  loss={np.mean(losses):.5f}  lr={cur_lr:.2e}",
              flush=True)

    return best_acc


def train_nanda_probe(train_X, train_Y, train_positions, eval_X, eval_Y,
                      eval_positions, device, input_dim, lr=1e-3, epochs=16,
                      batch_size=1024, expand_fn=None):
    """Train Nanda-style even/odd probe on arbitrary features.

    Uses separate linear probes for even and odd positions, mimicking
    Nanda's 3-mode probe but applied to non-activation feature vectors.

    train_positions: (N,) int tensor — the position index (5-53) for each sample.
        Even/odd is determined by position parity.
    expand_fn: optional callable(X, idx) -> expanded_X. If provided, applies to
        each batch before probing. Useful for on-the-fly pairwise expansion.
    Returns best accuracy.
    """
    probe_even = nn.Linear(input_dim, 64 * OPTIONS).to(device)
    probe_odd = nn.Linear(input_dim, 64 * OPTIONS).to(device)
    optimizer = torch.optim.Adam(
        list(probe_even.parameters()) + list(probe_odd.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        probe_even.train()
        probe_odd.train()
        perm = torch.randperm(len(train_X))
        for i in range(0, len(train_X), batch_size):
            idx = perm[i:i + batch_size]
            if expand_fn is not None:
                x = expand_fn(train_X, idx).to(device)
            else:
                x = train_X[idx].to(device)
            y = train_Y[idx].to(device)
            pos = train_positions[idx]
            even_mask = (pos % 2 == 0)
            odd_mask = ~even_mask

            loss = torch.tensor(0.0, device=device)
            if even_mask.any():
                logits_e = probe_even(x[even_mask]).view(-1, 64, OPTIONS)
                loss = loss + nn.functional.cross_entropy(
                    logits_e.reshape(-1, OPTIONS), y[even_mask].reshape(-1))
            if odd_mask.any():
                logits_o = probe_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                loss = loss + nn.functional.cross_entropy(
                    logits_o.reshape(-1, OPTIONS), y[odd_mask].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval
        probe_even.eval()
        probe_odd.eval()
        correct = 0
        total = 0
        losses = []
        with torch.no_grad():
            for i in range(0, len(eval_X), batch_size):
                idx_ev = list(range(i, min(i + batch_size, len(eval_X))))
                if expand_fn is not None:
                    x = expand_fn(eval_X, idx_ev).to(device)
                else:
                    x = eval_X[i:i + batch_size].to(device)
                y = eval_Y[i:i + batch_size].to(device)
                pos = eval_positions[i:i + batch_size]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                preds = torch.zeros_like(y)
                if even_mask.any():
                    logits_e = probe_even(x[even_mask]).view(-1, 64, OPTIONS)
                    preds[even_mask] = logits_e.argmax(-1)
                    loss_e = nn.functional.cross_entropy(
                        logits_e.reshape(-1, OPTIONS), y[even_mask].reshape(-1))
                    losses.append(loss_e.item())
                if odd_mask.any():
                    logits_o = probe_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                    preds[odd_mask] = logits_o.argmax(-1)
                    loss_o = nn.functional.cross_entropy(
                        logits_o.reshape(-1, OPTIONS), y[odd_mask].reshape(-1))
                    losses.append(loss_o.item())
                correct += (preds == y).sum().item()
                total += y.numel()

        acc = correct / total
        best_acc = max(best_acc, acc)
        scheduler.step(np.mean(losses))
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: acc={acc:.4%}  loss={np.mean(losses):.5f}  lr={cur_lr:.2e}",
              flush=True)

    return best_acc


def collect_activations_and_labels(model, games, device, layer, block_size,
                                   simulator=seq_to_state_normal, use_pre=False):
    """Collect flattened (position-level) activations and labels.

    Returns (acts_tensor, labels_tensor) where:
      acts: (N_total, d_model) float
      labels: (N_total, 64) long
    """
    acts_list = []
    labels_list = []
    game_batch = 64
    for start in tqdm(range(0, len(games), game_batch), desc="  collecting", leave=False):
        batch_games = games[start:start + game_batch]
        tokens = tokenize_games(batch_games, seq_len=block_size).to(device)

        with torch.no_grad():
            if use_pre:
                h = extract_activations_pre(model, tokens)
            else:
                h = extract_activations(model, tokens, layer)
        h = h[:, POS_START:POS_END]  # (B, length, d_model)

        states = get_board_states(batch_games, simulator, POS_START, POS_END)
        labels = states_to_labels(states)  # (B, length, 8, 8)
        labels = labels.reshape(len(batch_games), LENGTH, 64)  # (B, length, 64)

        for gi in range(len(batch_games)):
            for t in range(LENGTH):
                acts_list.append(h[gi, t].cpu())
                labels_list.append(torch.tensor(labels[gi, t], dtype=torch.long))

    return torch.stack(acts_list), torch.stack(labels_list)


# ============================= Experiment D: Standard Probe ==================

def experiment_standard_probe(args):
    """Standard probe without even/odd split on Othello-GPT and random init."""
    device = get_device()
    print(f"Device: {device}")

    model, block_size = load_model(args.ckpt_path, device)
    print(f"Loaded Othello-GPT from {args.ckpt_path}")

    random_model = create_random_model(device, block_size=block_size)
    print("Created random init model")

    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    n_eval = max(int(len(games) * 0.1), 100)
    n_train = len(games) - n_eval
    train_games = games[:n_train]
    eval_games = games[n_train:]
    print(f"Using {len(games)} games ({n_train} train, {n_eval} eval)")

    results = {}
    for layer in range(9):
        print(f"\n--- Layer {layer} ---")

        # Othello-GPT
        print("  Othello-GPT:")
        tr_X, tr_Y = collect_activations_and_labels(
            model, train_games, device, layer, block_size)
        ev_X, ev_Y = collect_activations_and_labels(
            model, eval_games, device, layer, block_size)
        acc_gpt = train_standard_probe(tr_X, tr_Y, ev_X, ev_Y, device, tr_X.shape[1])

        # Random init
        print("  Random init:")
        tr_X_r, tr_Y_r = collect_activations_and_labels(
            random_model, train_games, device, layer, block_size)
        ev_X_r, ev_Y_r = collect_activations_and_labels(
            random_model, eval_games, device, layer, block_size)
        acc_rnd = train_standard_probe(tr_X_r, tr_Y_r, ev_X_r, ev_Y_r, device, tr_X_r.shape[1])

        results[layer] = {"othello_gpt": acc_gpt, "random_init": acc_rnd}
        print(f"  Layer {layer}: GPT={acc_gpt:.4%}  Random={acc_rnd:.4%}")

    # Summary
    print(f"\n{'='*60}")
    print("STANDARD PROBE RESULTS (no even/odd split)")
    print(f"{'='*60}")
    print(f"Layer | Othello-GPT | Random Init")
    print(f"------|-------------|------------")
    for layer in range(9):
        r = results[layer]
        print(f"  {layer}   |   {r['othello_gpt']:.2%}    |   {r['random_init']:.2%}")

    _save_results(args, "standard_probe", results)
    return results


# ============================= Experiment B: resid_pre =======================

def experiment_resid_pre(args):
    """Probe on raw embeddings (before any transformer blocks)."""
    device = get_device()
    model, block_size = load_model(args.ckpt_path, device)
    print(f"Loaded Othello-GPT, device={device}")

    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    n_eval = max(int(len(games) * 0.1), 100)
    train_games = games[:len(games) - n_eval]
    eval_games = games[len(games) - n_eval:]
    print(f"Using {len(games)} games")

    results = {}

    # resid_pre (no blocks)
    print("\nresid_pre (raw embeddings, no blocks):")
    tr_X, tr_Y = collect_activations_and_labels(
        model, train_games, device, 0, block_size, use_pre=True)
    ev_X, ev_Y = collect_activations_and_labels(
        model, eval_games, device, 0, block_size, use_pre=True)
    acc_pre = train_standard_probe(tr_X, tr_Y, ev_X, ev_Y, device, tr_X.shape[1])
    results["resid_pre"] = acc_pre

    # resid_post layer 0 for comparison
    print("\nresid_post layer 0 (after first block):")
    tr_X0, tr_Y0 = collect_activations_and_labels(
        model, train_games, device, 0, block_size)
    ev_X0, ev_Y0 = collect_activations_and_labels(
        model, eval_games, device, 0, block_size)
    acc_post0 = train_standard_probe(tr_X0, tr_Y0, ev_X0, ev_Y0, device, tr_X0.shape[1])
    results["resid_post_layer0"] = acc_post0

    print(f"\n{'='*60}")
    print("RESID_PRE vs RESID_POST RESULTS")
    print(f"{'='*60}")
    print(f"  resid_pre (no blocks):    {acc_pre:.4%}")
    print(f"  resid_post layer 0:       {acc_post0:.4%}")
    print(f"  Difference:               {acc_post0 - acc_pre:+.4%}")

    _save_results(args, "resid_pre", results)
    return results


# ============================= Experiment C: Alt Board States ================

def experiment_alt_boards(args):
    """Decode alternative board states from Othello-GPT layer 0."""
    device = get_device()
    model, block_size = load_model(args.ckpt_path, device)
    print(f"Loaded Othello-GPT, device={device}")

    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    n_eval = max(int(len(games) * 0.1), 100)
    train_games = games[:len(games) - n_eval]
    eval_games = games[len(games) - n_eval:]
    print(f"Using {len(games)} games")

    # Collect layer 0 activations once
    print("Collecting layer 0 activations...")
    tr_X, _ = collect_activations_and_labels(
        model, train_games, device, 0, block_size)
    ev_X, _ = collect_activations_and_labels(
        model, eval_games, device, 0, block_size)

    # Define board state variants
    rng = np.random.RandomState(42)
    cell_perm = rng.permutation(64)

    def make_shifted_col(states):
        """Shift columns: col -> (col+1) % 8."""
        return np.roll(states, shift=1, axis=-1)

    def make_shifted_row(states):
        """Shift rows: row -> (row+1) % 8."""
        return np.roll(states, shift=1, axis=-2)

    def make_transposed(states):
        """Transpose: swap rows and cols."""
        return np.swapaxes(states, -2, -1)

    def make_color_inverted(states):
        """Swap black and white."""
        return -states

    def make_cell_permuted(states):
        """Apply fixed random permutation to the 64 cells."""
        flat = states.reshape(*states.shape[:-2], 64)
        permuted = flat[..., cell_perm]
        return permuted.reshape(*states.shape[:-2], 8, 8)

    variants = [
        ("Real Othello", seq_to_state_normal, None),
        ("No Flip", seq_to_state_no_flip, None),
        ("Column-shifted", seq_to_state_normal, make_shifted_col),
        ("Row-shifted", seq_to_state_normal, make_shifted_row),
        ("Transposed", seq_to_state_normal, make_transposed),
        ("Color-inverted", seq_to_state_normal, make_color_inverted),
        ("Cell-permuted", seq_to_state_normal, make_cell_permuted),
    ]

    results = {}
    for name, simulator, transform in variants:
        print(f"\n--- {name} ---")

        # Compute labels for train
        tr_states = get_board_states(train_games, simulator, POS_START, POS_END)
        if transform is not None:
            tr_states = transform(tr_states)
        tr_labels = states_to_labels(tr_states).reshape(-1, 64)
        tr_Y = torch.tensor(tr_labels, dtype=torch.long)

        # Compute labels for eval
        ev_states = get_board_states(eval_games, simulator, POS_START, POS_END)
        if transform is not None:
            ev_states = transform(ev_states)
        ev_labels = states_to_labels(ev_states).reshape(-1, 64)
        ev_Y = torch.tensor(ev_labels, dtype=torch.long)

        # Flatten: (N_games, length, ...) -> (N_games * length, ...)
        n_tr = len(train_games) * LENGTH
        n_ev = len(eval_games) * LENGTH
        assert tr_X.shape[0] == n_tr, f"Mismatch: {tr_X.shape[0]} vs {n_tr}"

        acc = train_standard_probe(tr_X, tr_Y, ev_X, ev_Y, device, tr_X.shape[1])
        results[name] = acc
        print(f"  {name}: {acc:.4%}")

    print(f"\n{'='*60}")
    print("ALT BOARD STATES FROM LAYER 0 (standard probe)")
    print(f"{'='*60}")
    for name, acc in results.items():
        print(f"  {name:25s}  {acc:.4%}")

    _save_results(args, "alt_boards", results)
    return results


# ============================= Experiment A: Per-Move Accuracy ===============

def experiment_by_move(args):
    """Per-move-number accuracy for Normal vs No Flip using Nanda's 3-mode probe."""
    device = get_device()
    model, block_size = load_model(args.ckpt_path, device)
    print(f"Loaded Othello-GPT, device={device}")

    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    n_eval = max(int(len(games) * 0.1), 100)
    n_train = len(games) - n_eval
    train_games = games[:n_train]
    eval_games = games[n_train:]
    print(f"Using {len(games)} games, layer {args.layer}")

    d_model = model.pos_emb.shape[-1]
    MODES = 3

    results = {}
    for variant_name, simulator in [("normal", seq_to_state_normal),
                                     ("no_flip", seq_to_state_no_flip)]:
        print(f"\n--- {variant_name} ---")

        # Train Nanda's 3-mode probe
        linear_probe = torch.randn(
            MODES, d_model, ROWS, COLS, OPTIONS,
            requires_grad=False, device=device,
        ) / np.sqrt(d_model)
        linear_probe.requires_grad = True
        optimizer = torch.optim.AdamW(
            [linear_probe], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

        batch_size = 100
        for epoch in range(1, 3):
            perm = torch.randperm(n_train)
            for i in tqdm(range(0, n_train, batch_size),
                          desc=f"  Epoch {epoch}", leave=False):
                idx = perm[i:i + batch_size]
                batch_games = [train_games[j] for j in idx]
                tokens = tokenize_games(batch_games, seq_len=block_size).to(device)

                states = get_board_states(batch_games, simulator, POS_START, POS_END)
                state_stack = torch.tensor(states)
                one_hot = torch.zeros(
                    MODES, len(batch_games), LENGTH, ROWS, COLS, OPTIONS,
                    device=device, dtype=torch.int)
                one_hot[:, ..., 0] = state_stack == 0
                one_hot[:, ..., 1] = state_stack == -1
                one_hot[:, ..., 2] = state_stack == 1

                with torch.no_grad():
                    resid = extract_activations(
                        model, tokens, args.layer)[:, POS_START:POS_END]

                probe_out = torch.einsum(
                    "bpd,mdrco->mbprco", resid, linear_probe)
                probe_log_probs = probe_out.log_softmax(-1)
                probe_correct = (
                    (probe_log_probs * one_hot).mean(dim=(1, -1))
                ) * OPTIONS
                loss_even = -probe_correct[0, 0::2].mean(0).sum()
                loss_odd = -probe_correct[1, 1::2].mean(0).sum()
                loss_all = -probe_correct[2, :].mean(0).sum()
                loss = loss_even + loss_odd + loss_all
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Per-position eval
        per_pos_correct = np.zeros(LENGTH)
        per_pos_total = np.zeros(LENGTH)

        with torch.no_grad():
            for i in range(0, n_eval, batch_size):
                batch_games = eval_games[i:i + batch_size]
                tokens = tokenize_games(batch_games, seq_len=block_size).to(device)
                resid = extract_activations(
                    model, tokens, args.layer)[:, POS_START:POS_END]

                states = get_board_states(batch_games, simulator, POS_START, POS_END)
                gt = states_to_labels(states)
                gt_tensor = torch.tensor(gt, device=device, dtype=torch.long)

                probe_out = torch.einsum(
                    "bpd,mdrco->mbprco", resid, linear_probe)
                B, L = resid.shape[0], resid.shape[1]
                preds = torch.zeros(B, L, 8, 8, device=device, dtype=torch.long)
                preds[:, 0::2] = probe_out[0, :, 0::2].argmax(-1)
                preds[:, 1::2] = probe_out[1, :, 1::2].argmax(-1)

                for t in range(LENGTH):
                    correct = (preds[:, t] == gt_tensor[:, t]).sum().item()
                    per_pos_correct[t] += correct
                    per_pos_total[t] += gt_tensor[:, t].numel()

        per_pos_acc = per_pos_correct / per_pos_total
        results[variant_name] = {
            f"move_{POS_START + t}": float(per_pos_acc[t]) for t in range(LENGTH)
        }

        print(f"  {variant_name} per-move accuracy:")
        for t in range(0, LENGTH, 5):
            move = POS_START + t
            print(f"    Move {move:2d}: {per_pos_acc[t]:.4%}")

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"PER-MOVE ACCURACY: Normal vs No Flip (layer {args.layer})")
    print(f"{'='*60}")
    print(f"Move | Normal   | No Flip  | Gap")
    print(f"-----|----------|----------|--------")
    for t in range(LENGTH):
        move = POS_START + t
        n_acc = results["normal"][f"move_{move}"]
        nf_acc = results["no_flip"][f"move_{move}"]
        gap = n_acc - nf_acc
        if t % 3 == 0:  # print every 3rd move
            print(f"  {move:2d} | {n_acc:.4%} | {nf_acc:.4%} | {gap:+.4%}")

    _save_results(args, "by_move", results)
    return results


# ============================= Experiment 1: Heuristic Features ==============

def _parse_square(sq_str):
    """Convert 'A0' -> board position (row*8 + col)."""
    row = ord(sq_str[0]) - ord('A')
    col = int(sq_str[1])
    return row * 8 + col


def _load_rules_json():
    """Load rules from Teo's branch via git show."""
    rules_path = os.path.join(
        SCRIPT_DIR, "..", "reverse_engineering_experiments", "rules_060_2000_2-6.json")
    if os.path.exists(rules_path):
        with open(rules_path) as f:
            return json.load(f)
    # Try git show
    try:
        result = subprocess.run(
            ["git", "show",
             "mine/Teo:experiments/reverse_engineering_experiments/rules_060_2000_2-6.json"],
            capture_output=True, text=True, cwd=PROJECT_ROOT)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    raise FileNotFoundError(
        "Cannot find rules_060_2000_2-6.json. Either place it in "
        "experiments/reverse_engineering_experiments/ or ensure git remote 'mine' "
        "has the Teo branch.")


def _parse_rule(rule_str):
    """Parse rule string like '(A5_mine) AND (NOT B3_theirs)' into conditions.

    Returns list of (square_pos, feature_type, negated) tuples.
    feature_type is one of: 'mine', 'theirs', 'empty', 'just_played', 'flipped'.
    """
    conditions = []
    # Split on AND
    parts = re.split(r'\s+AND\s+', rule_str)
    for part in parts:
        part = part.strip()
        negated = False
        if part.startswith("(NOT "):
            negated = True
            part = part[5:]  # remove "(NOT "
        part = part.strip("()")
        # Parse feature name like "A5_mine"
        match = re.match(r'^([A-H]\d)_(.+)$', part)
        if not match:
            return None  # unparseable
        sq_str, feat_type = match.groups()
        pos = _parse_square(sq_str)
        conditions.append((pos, feat_type, negated))
    return conditions


def _eval_condition_move_only(pos, feat_type, negated, move_history, current_step):
    """Evaluate a single condition using only move history.

    move_history: dict mapping board_position -> step_index (when it was played)
    current_step: current move number (0-indexed)
    """
    current_parity = current_step % 2  # 0 = black's turn, 1 = white's turn

    if feat_type == "empty":
        val = pos not in move_history
    elif feat_type == "just_played":
        val = (pos in move_history and move_history[pos] == current_step)
    elif feat_type == "mine":
        # "mine" = same parity as current player
        if pos not in move_history:
            val = False
        else:
            played_parity = move_history[pos] % 2
            val = (played_parity == current_parity)
    elif feat_type == "theirs":
        # "theirs" = opposite parity
        if pos not in move_history:
            val = False
        else:
            played_parity = move_history[pos] % 2
            val = (played_parity != current_parity)
    elif feat_type == "flipped":
        return None  # cannot evaluate from moves alone
    else:
        return None

    if negated:
        val = not val
    return val


def _build_heuristic_features(rules_data, mode="convert"):
    """Build list of parsed rules, filtering based on mode.

    mode="convert": drop rules with _flipped, convert mine/theirs to parity
    mode="strict": drop rules with _flipped, _mine, or _theirs
    """
    parsed_rules = []
    for layer, neurons in rules_data.items():
        for neuron, info in neurons.items():
            for rule in info.get("rules", []):
                conditions = _parse_rule(rule["rule"])
                if conditions is None:
                    continue
                feat_types = [c[1] for c in conditions]
                if any(ft == "flipped" for ft in feat_types):
                    continue
                if mode == "strict" and any(ft in ("mine", "theirs") for ft in feat_types):
                    continue
                parsed_rules.append(conditions)
    return parsed_rules


def _compute_heuristic_vector(parsed_rules, game, step):
    """Compute binary feature vector for a game at a given step.

    game: list of raw moves (board positions 0-63)
    step: 0-indexed move number
    Returns numpy array of shape (len(parsed_rules),) with 0/1 values.
    """
    # Build move history: position -> step index
    move_history = {}
    for s in range(step + 1):
        move_history[game[s]] = s

    features = np.zeros(len(parsed_rules), dtype=np.float32)
    for ri, conditions in enumerate(parsed_rules):
        all_true = True
        for pos, feat_type, negated in conditions:
            val = _eval_condition_move_only(pos, feat_type, negated, move_history, step)
            if val is None or val is False:
                all_true = False
                break
        features[ri] = 1.0 if all_true else 0.0
    return features


# Condition type IDs for vectorized evaluation
_CTYPE_EMPTY = 0
_CTYPE_JUST_PLAYED = 1
_CTYPE_MINE = 2
_CTYPE_THEIRS = 3
_CTYPE_MAP = {"empty": _CTYPE_EMPTY, "just_played": _CTYPE_JUST_PLAYED,
              "mine": _CTYPE_MINE, "theirs": _CTYPE_THEIRS}


def _compile_rules(parsed_rules):
    """Precompile parsed rules into arrays for vectorized evaluation.

    Returns:
      cond_pos: (total_conditions,) int array of board positions
      cond_type: (total_conditions,) int array of condition type IDs
      cond_neg: (total_conditions,) bool array of negation flags
      rule_starts: (n_rules,) int array — index into cond arrays where each rule starts
      rule_lengths: (n_rules,) int array — number of conditions per rule
    """
    all_pos = []
    all_type = []
    all_neg = []
    starts = []
    lengths = []
    offset = 0
    for conditions in parsed_rules:
        starts.append(offset)
        lengths.append(len(conditions))
        for pos, feat_type, negated in conditions:
            all_pos.append(pos)
            all_type.append(_CTYPE_MAP[feat_type])
            all_neg.append(negated)
        offset += len(conditions)
    return (np.array(all_pos, dtype=np.int32),
            np.array(all_type, dtype=np.int32),
            np.array(all_neg, dtype=np.bool_),
            np.array(starts, dtype=np.int32),
            np.array(lengths, dtype=np.int32))


def _compute_heuristic_batch(compiled_rules, games, pos_start, pos_end):
    """Vectorized heuristic feature computation for many games.

    Returns features array of shape (n_samples, n_rules).
    """
    cond_pos, cond_type, cond_neg, rule_starts, rule_lengths = compiled_rules
    n_rules = len(rule_starts)
    n_conds = len(cond_pos)
    max_len = int(rule_lengths.max())

    # Build padded condition-to-rule mapping: (n_rules, max_len) index array
    # cond_idx[r, c] = index into cond arrays for rule r, condition c
    # Use 0 as padding (will be masked out)
    cond_idx = np.zeros((n_rules, max_len), dtype=np.int32)
    cond_mask = np.zeros((n_rules, max_len), dtype=np.bool_)
    for ri in range(n_rules):
        s = rule_starts[ri]
        l = rule_lengths[ri]
        cond_idx[ri, :l] = np.arange(s, s + l)
        cond_mask[ri, :l] = True

    n_per_game = pos_end - pos_start
    n_samples = len(games) * n_per_game
    features = np.zeros((n_samples, n_rules), dtype=np.float32)

    for gi, game in enumerate(games):
        for ti, t in enumerate(range(pos_start, pos_end)):
            si = gi * n_per_game + ti

            # Build per-position state
            is_played = np.zeros(64, dtype=np.bool_)
            played_step = np.full(64, -1, dtype=np.int32)
            played_parity = np.zeros(64, dtype=np.int32)
            for s in range(t + 1):
                p = game[s]
                is_played[p] = True
                played_step[p] = s
                played_parity[p] = s % 2
            current_parity = t % 2

            # Evaluate all conditions at once
            played_at_pos = is_played[cond_pos]
            step_at_pos = played_step[cond_pos]
            parity_at_pos = played_parity[cond_pos]

            val = np.zeros(n_conds, dtype=np.bool_)
            m = cond_type == _CTYPE_EMPTY
            val[m] = ~played_at_pos[m]
            m = cond_type == _CTYPE_JUST_PLAYED
            val[m] = played_at_pos[m] & (step_at_pos[m] == t)
            m = cond_type == _CTYPE_MINE
            val[m] = played_at_pos[m] & (parity_at_pos[m] == current_parity)
            m = cond_type == _CTYPE_THEIRS
            val[m] = played_at_pos[m] & (parity_at_pos[m] != current_parity)
            val[cond_neg] = ~val[cond_neg]

            # AND reduction: gather condition values per rule, AND across conditions
            # val_per_rule[r, c] = val[cond_idx[r, c]], but masked
            val_gathered = val[cond_idx.ravel()].reshape(n_rules, max_len)
            # Unmasked positions set to True (neutral for AND)
            val_gathered[~cond_mask] = True
            features[si] = val_gathered.all(axis=1).astype(np.float32)

    return features


def experiment_heuristic(args):
    """Heuristic features → board state."""
    device = get_device()
    print(f"Device: {device}")

    # Load rules
    print("Loading heuristic rules...")
    rules_data = _load_rules_json()

    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    n_eval = max(int(len(games) * 0.1), 100)
    train_games = games[:len(games) - n_eval]
    eval_games = games[len(games) - n_eval:]
    print(f"Using {len(games)} games")

    # Also load random model for variation 1b/1c and Othello-GPT for 1d/1e
    random_model = create_random_model(device, block_size=59)
    othello_model, othello_block_size = load_model(args.ckpt_path, device)

    results = {}

    for mode in ["convert", "strict"]:
        print(f"\n{'='*60}")
        print(f"Heuristic mode: {mode}")
        print(f"{'='*60}")

        parsed_rules = _build_heuristic_features(rules_data, mode=mode)
        n_rules = len(parsed_rules)
        print(f"  {n_rules} rules after filtering")

        # Compute heuristic features for all games/positions
        print("  Computing heuristic features for train...")
        tr_heur = []
        tr_labels = []
        tr_positions = []
        for game in tqdm(train_games, desc="  train games", leave=False):
            states = seq_to_state_normal(game)
            for t in range(POS_START, POS_END):
                tr_heur.append(_compute_heuristic_vector(parsed_rules, game, t))
                lbl = states_to_labels(states[t:t+1].reshape(1, 8, 8))
                tr_labels.append(lbl.reshape(64))
                tr_positions.append(t)
        tr_heur = torch.tensor(np.stack(tr_heur), dtype=torch.float32)
        tr_labels = torch.tensor(np.stack(tr_labels), dtype=torch.long)
        tr_positions = torch.tensor(tr_positions, dtype=torch.long)

        print("  Computing heuristic features for eval...")
        ev_heur = []
        ev_labels = []
        ev_positions = []
        for game in tqdm(eval_games, desc="  eval games", leave=False):
            states = seq_to_state_normal(game)
            for t in range(POS_START, POS_END):
                ev_heur.append(_compute_heuristic_vector(parsed_rules, game, t))
                lbl = states_to_labels(states[t:t+1].reshape(1, 8, 8))
                ev_labels.append(lbl.reshape(64))
                ev_positions.append(t)
        ev_heur = torch.tensor(np.stack(ev_heur), dtype=torch.float32)
        ev_positions = torch.tensor(ev_positions, dtype=torch.long)
        ev_labels = torch.tensor(np.stack(ev_labels), dtype=torch.long)

        # Variation 1a: heuristic features only
        print(f"\n  Variation 1a: Heuristic only ({n_rules} features)")
        acc_1a = train_standard_probe(
            tr_heur, tr_labels, ev_heur, ev_labels, device, n_rules)

        # Variation 1c: random network layer 0 only
        print("\n  Variation 1c: Random network layer 0 only")
        tr_X_r, _ = collect_activations_and_labels(
            random_model, train_games, device, 0, 59)
        ev_X_r, _ = collect_activations_and_labels(
            random_model, eval_games, device, 0, 59)
        acc_1c = train_standard_probe(
            tr_X_r, tr_labels, ev_X_r, ev_labels, device, tr_X_r.shape[1])

        # Variation 1b: heuristic + random network layer 0
        print(f"\n  Variation 1b: Heuristic + random layer 0 ({n_rules + 512} features)")
        tr_combined = torch.cat([tr_heur, tr_X_r], dim=1)
        ev_combined = torch.cat([ev_heur, ev_X_r], dim=1)
        acc_1b = train_standard_probe(
            tr_combined, tr_labels, ev_combined, ev_labels,
            device, tr_combined.shape[1])

        # Variation 1d: Othello-GPT layer 0 only
        print("\n  Variation 1d: Othello-GPT layer 0 only")
        tr_X_o, _ = collect_activations_and_labels(
            othello_model, train_games, device, 0, othello_block_size)
        ev_X_o, _ = collect_activations_and_labels(
            othello_model, eval_games, device, 0, othello_block_size)
        acc_1d = train_standard_probe(
            tr_X_o, tr_labels, ev_X_o, ev_labels, device, tr_X_o.shape[1])

        # Variation 1e: heuristic + Othello-GPT layer 0
        print(f"\n  Variation 1e: Heuristic + Othello-GPT layer 0 ({n_rules + 512} features)")
        tr_combined_o = torch.cat([tr_heur, tr_X_o], dim=1)
        ev_combined_o = torch.cat([ev_heur, ev_X_o], dim=1)
        acc_1e = train_standard_probe(
            tr_combined_o, tr_labels, ev_combined_o, ev_labels,
            device, tr_combined_o.shape[1])

        # Variation 1f: Nanda even/odd probe on heuristic features only
        print(f"\n  Variation 1f: Nanda even/odd probe on heuristic ({n_rules} features)")
        acc_1f = train_nanda_probe(
            tr_heur, tr_labels, tr_positions,
            ev_heur, ev_labels, ev_positions,
            device, n_rules)

        # Variation 1g: Nanda even/odd probe on heuristic + Othello-GPT L0
        print(f"\n  Variation 1g: Nanda even/odd probe on heuristic + Othello L0 ({n_rules + 512} features)")
        acc_1g = train_nanda_probe(
            tr_combined_o, tr_labels, tr_positions,
            ev_combined_o, ev_labels, ev_positions,
            device, tr_combined_o.shape[1])

        results[mode] = {
            "n_rules": n_rules,
            "heuristic_only": acc_1a,
            "heuristic_plus_random": acc_1b,
            "random_only": acc_1c,
            "othello_gpt_only": acc_1d,
            "heuristic_plus_othello_gpt": acc_1e,
            "nanda_heuristic_only": acc_1f,
            "nanda_heuristic_plus_othello": acc_1g,
        }

        print(f"\n  Summary ({mode}):")
        print(f"    1a Heuristic only:           {acc_1a:.4%}")
        print(f"    1b Heuristic + random L0:    {acc_1b:.4%}")
        print(f"    1c Random L0 only:           {acc_1c:.4%}")
        print(f"    1d Othello-GPT L0 only:      {acc_1d:.4%}")
        print(f"    1e Heuristic + Othello L0:   {acc_1e:.4%}")
        print(f"    1f Nanda heuristic only:      {acc_1f:.4%}")
        print(f"    1g Nanda heur + Othello L0:   {acc_1g:.4%}")

    print(f"\n{'='*60}")
    print("HEURISTIC EXPERIMENT RESULTS")
    print(f"{'='*60}")
    for mode, r in results.items():
        print(f"\n  {mode} ({r['n_rules']} rules):")
        print(f"    Heuristic only:          {r['heuristic_only']:.4%}")
        print(f"    Heuristic + random L0:   {r['heuristic_plus_random']:.4%}")
        print(f"    Random L0 only:          {r['random_only']:.4%}")
        print(f"    Othello-GPT L0 only:     {r['othello_gpt_only']:.4%}")
        print(f"    Heuristic + Othello L0:  {r['heuristic_plus_othello_gpt']:.4%}")
        print(f"    Nanda heuristic only:    {r['nanda_heuristic_only']:.4%}")
        print(f"    Nanda heur + Othello:    {r['nanda_heuristic_plus_othello']:.4%}")

    _save_results(args, "heuristic", results)
    return results


# ==================== Move-History Feature Construction =====================

_VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})
_MOVE_TO_IDX = {m: i for i, m in enumerate(_VALID_MOVES)}
N_MOVES = 60  # number of valid move IDs


def _build_move_history_features(game, step, include_pairwise=True):
    """Build move-history feature vector at position `step`.

    Base features (180-d):
      played[i]: 1 if move i has been played by step
      when[i]:   step_played / 60 (0 if not played)
      even[i]:   1 if move i was played on an even step

    Pairwise interactions (3600-d):
      played[i] * played[j] for i<j (1770)
      played[i] * even[i] (60)
      even[i] * even[j] for i<j (1770)

    Returns numpy array of shape (180,) or (3780,).
    """
    played = np.zeros(N_MOVES, dtype=np.float32)
    when = np.zeros(N_MOVES, dtype=np.float32)
    even = np.zeros(N_MOVES, dtype=np.float32)

    for s in range(step + 1):
        idx = _MOVE_TO_IDX[game[s]]
        played[idx] = 1.0
        when[idx] = (s + 1) / 60.0
        even[idx] = 1.0 if (s % 2 == 0) else 0.0

    base = np.concatenate([played, when, even])  # (180,)

    if not include_pairwise:
        return base

    # Pairwise interactions using upper triangle indices
    idx_i, idx_j = np.triu_indices(N_MOVES, k=1)  # i < j pairs

    pp = played[idx_i] * played[idx_j]  # (1770,)
    pe = played * even  # (60,)
    ee = even[idx_i] * even[idx_j]  # (1770,)

    return np.concatenate([base, pp, pe, ee])  # (3780,)


def _compute_labels_for_games(args):
    """Worker function for parallel label computation."""
    games_chunk, pos_start, pos_end = args
    length = pos_end - pos_start
    n = len(games_chunk)
    labels = np.zeros((n, length, 64), dtype=np.int64)
    for gi, game in enumerate(games_chunk):
        states = seq_to_state_normal(game)  # (60, 8, 8)
        for ti, t in enumerate(range(pos_start, pos_end)):
            lbl = states_to_labels(states[t:t+1].reshape(1, 8, 8))
            labels[gi, ti] = lbl.reshape(64)
    return labels


def _compute_labels_parallel(games, pos_start, pos_end, n_games):
    """Compute labels using multiprocessing. Returns (n_samples, 64) array."""
    from multiprocessing import Pool, cpu_count
    length = pos_end - pos_start
    n_samples = n_games * length

    n_workers = min(cpu_count(), 8)
    chunk_size = (n_games + n_workers - 1) // n_workers
    chunks = []
    for i in range(0, n_games, chunk_size):
        chunks.append((games[i:i+chunk_size], pos_start, pos_end))

    print(f"  Using {n_workers} workers for {n_games} games...", flush=True)
    with Pool(n_workers) as pool:
        results = pool.map(_compute_labels_for_games, chunks)

    # Reassemble: each result is (chunk_n, length, 64), need to interleave
    # into (length * n_games, 64) where index = ti * n_games + gi
    labels = np.zeros((n_samples, 64), dtype=np.int64)
    gi_offset = 0
    for res in results:
        chunk_n = res.shape[0]
        for ti in range(length):
            idx_start = ti * n_games + gi_offset
            labels[idx_start:idx_start + chunk_n] = res[:, ti]
        gi_offset += chunk_n

    return labels


def _build_move_features_batch(games, pos_start, pos_end, include_pairwise=True):
    """Build move-history features and labels for a list of games.

    Vectorized: converts all games to a (N, 60) move-index array, then
    computes played/when/even features for all positions using broadcasting.

    Returns (features, labels, positions) tensors.
    """
    n_games = len(games)
    length = pos_end - pos_start
    n_samples = n_games * length

    # Convert games to (N, 60) array of move indices
    game_arr = np.zeros((n_games, 60), dtype=np.int32)
    for i, game in enumerate(games):
        for s, move in enumerate(game):
            game_arr[i, s] = _MOVE_TO_IDX[move]

    # For each position t in [pos_start, pos_end), compute features
    # played[i] = 1 if move_idx i appears in steps 0..t
    # when[i] = (step+1)/60 for the step where move_idx i was played
    # even[i] = 1 if move_idx i was played on an even step

    # Build per-game lookup: for each move index, which step was it played?
    # step_of_move[g, m] = step when move m was played in game g (-1 if never)
    step_of_move = np.full((n_games, N_MOVES), -1, dtype=np.int32)
    for s in range(60):
        move_indices = game_arr[:, s]  # (N,)
        step_of_move[np.arange(n_games), move_indices] = s

    # Pre-allocate output
    features = np.zeros((n_samples, 180), dtype=np.float32)
    positions = np.zeros(n_samples, dtype=np.int64)

    for ti, t in enumerate(range(pos_start, pos_end)):
        start = ti * n_games
        end = start + n_games

        # played: move was played at or before step t
        played = (step_of_move >= 0) & (step_of_move <= t)  # (N, 60)
        # when: normalized step
        when = np.where(played, (step_of_move + 1) / 60.0, 0.0)  # (N, 60)
        # even: played on even step
        even = np.where(played, (step_of_move % 2 == 0).astype(np.float32), 0.0)  # (N, 60)

        features[start:end, :N_MOVES] = played.astype(np.float32)
        features[start:end, N_MOVES:2*N_MOVES] = when
        features[start:end, 2*N_MOVES:] = even
        positions[start:end] = t

    if include_pairwise:
        idx_i, idx_j = np.triu_indices(N_MOVES, k=1)
        played_all = features[:, :N_MOVES]
        even_all = features[:, 2*N_MOVES:]
        pp = played_all[:, idx_i] * played_all[:, idx_j]
        pe = played_all * even_all
        ee = even_all[:, idx_i] * even_all[:, idx_j]
        features = np.concatenate([features, pp, pe, ee], axis=1)

    # Compute labels (board simulation — parallelized across CPU cores)
    print("  computing board states for labels...", flush=True)
    labels = _compute_labels_parallel(games, pos_start, pos_end, n_games)

    return (torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long))


# ==================== Approach 4: Brute-Force Features =======================

def experiment_brute_force(args):
    """Approach 4: Brute-force move-history features with Nanda's probe."""
    device = get_device()
    print(f"Device: {device}")

    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    n_eval = max(int(len(games) * 0.1), 100)
    train_games = games[:len(games) - n_eval]
    eval_games = games[len(games) - n_eval:]
    print(f"Using {len(games)} games ({len(train_games)} train, {len(eval_games)} eval)")

    results = {}

    # First: base features only (180-d)
    print("\n--- Base features only (180-d) ---")
    print("Building train features...")
    tr_X, tr_Y, tr_pos = _build_move_features_batch(
        train_games, POS_START, POS_END, include_pairwise=False)
    print("Building eval features...")
    ev_X, ev_Y, ev_pos = _build_move_features_batch(
        eval_games, POS_START, POS_END, include_pairwise=False)
    print(f"  Feature shape: {tr_X.shape}")

    print("\n  Nanda even/odd probe on 180-d base features:")
    acc_base_nanda = train_nanda_probe(
        tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos, device, 180)
    results["base_180_nanda"] = acc_base_nanda

    print("\n  Standard probe on 180-d base features:")
    acc_base_std = train_standard_probe(tr_X, tr_Y, ev_X, ev_Y, device, 180)
    results["base_180_standard"] = acc_base_std

    # Second: base + pairwise features (3780-d) — expanded on-the-fly
    print("\n--- Base + pairwise features (3780-d, on-the-fly expansion) ---")

    print("\n  Nanda even/odd probe on 3780-d features:")
    acc_pw_nanda = train_nanda_probe(
        tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos,
        device, 3780, expand_fn=_expand_pairwise_batch_cached)
    results["pairwise_3780_nanda"] = acc_pw_nanda

    print(f"\n{'='*60}")
    print("BRUTE-FORCE FEATURE RESULTS")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v:.4%}")

    _save_results(args, "brute_force", results)
    return results


# ==================== Approach 3: Learned Heuristic Combos ===================

def _train_learned_heuristic(tr_heur, tr_Y, tr_pos, ev_heur, ev_Y, ev_pos,
                              device, n_features, K, use_original=True,
                              lr=1e-3, epochs=16, batch_size=1024,
                              shuffle_chunk=200000):
    """Train learnable linear layer + ReLU on heuristic features, jointly with
    Nanda's even/odd probe.

    Architecture:
      heuristic (n_features) -> Linear(n_features, K) -> ReLU -> learned (K)
      if use_original: concat [heuristic, learned] -> (n_features + K)
      else: just learned -> (K)
      -> even/odd probe -> (64*3)

    For large datasets (memmap-backed), shuffles within sequential chunks
    of `shuffle_chunk` samples to avoid random page faults.
    """
    input_dim = (n_features + K) if use_original else K

    feature_layer = nn.Linear(n_features, K).to(device)
    probe_even = nn.Linear(input_dim, 64 * OPTIONS).to(device)
    probe_odd = nn.Linear(input_dim, 64 * OPTIONS).to(device)

    all_params = (list(feature_layer.parameters()) +
                  list(probe_even.parameters()) +
                  list(probe_odd.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    n_train = len(tr_heur)
    # Use chunked iteration for large datasets to preserve sequential access
    use_chunks = n_train > shuffle_chunk * 2

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        feature_layer.train()
        probe_even.train()
        probe_odd.train()

        if use_chunks:
            # Shuffle chunk order, then shuffle within each chunk
            chunk_starts = list(range(0, n_train, shuffle_chunk))
            np.random.shuffle(chunk_starts)
            for cs in chunk_starts:
                ce = min(cs + shuffle_chunk, n_train)
                # Load chunk into memory
                chunk_X = torch.tensor(
                    np.array(tr_heur[cs:ce]), dtype=torch.float32)
                chunk_Y = tr_Y[cs:ce]
                chunk_pos = tr_pos[cs:ce]
                perm = torch.randperm(len(chunk_X))
                for i in range(0, len(chunk_X), batch_size):
                    idx = perm[i:i + batch_size]
                    x_raw = chunk_X[idx].to(device)
                    y = chunk_Y[idx].to(device)
                    pos = chunk_pos[idx]

                    even_mask = (pos % 2 == 0)
                    odd_mask = ~even_mask

                    learned = torch.relu(feature_layer(x_raw))
                    if use_original:
                        x = torch.cat([x_raw, learned], dim=1)
                    else:
                        x = learned

                    loss = torch.tensor(0.0, device=device)
                    if even_mask.any():
                        logits_e = probe_even(x[even_mask]).view(-1, 64, OPTIONS)
                        loss = loss + nn.functional.cross_entropy(
                            logits_e.reshape(-1, OPTIONS), y[even_mask].reshape(-1))
                    if odd_mask.any():
                        logits_o = probe_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                        loss = loss + nn.functional.cross_entropy(
                            logits_o.reshape(-1, OPTIONS), y[odd_mask].reshape(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                del chunk_X
        else:
            perm = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                idx = perm[i:i + batch_size]
                x_raw = tr_heur[idx].to(device)
                y = tr_Y[idx].to(device)
                pos = tr_pos[idx]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                learned = torch.relu(feature_layer(x_raw))
                if use_original:
                    x = torch.cat([x_raw, learned], dim=1)
                else:
                    x = learned

                loss = torch.tensor(0.0, device=device)
                if even_mask.any():
                    logits_e = probe_even(x[even_mask]).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits_e.reshape(-1, OPTIONS), y[even_mask].reshape(-1))
                if odd_mask.any():
                    logits_o = probe_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                    loss = loss + nn.functional.cross_entropy(
                        logits_o.reshape(-1, OPTIONS), y[odd_mask].reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Eval
        feature_layer.eval()
        probe_even.eval()
        probe_odd.eval()
        correct = 0
        total = 0
        losses = []
        with torch.no_grad():
            for i in range(0, len(ev_heur), batch_size):
                x_raw = ev_heur[i:i + batch_size].to(device)
                y = ev_Y[i:i + batch_size].to(device)
                pos = ev_pos[i:i + batch_size]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                learned = torch.relu(feature_layer(x_raw))
                if use_original:
                    x = torch.cat([x_raw, learned], dim=1)
                else:
                    x = learned

                preds = torch.zeros_like(y)
                if even_mask.any():
                    logits_e = probe_even(x[even_mask]).view(-1, 64, OPTIONS)
                    preds[even_mask] = logits_e.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits_e.reshape(-1, OPTIONS),
                        y[even_mask].reshape(-1)).item())
                if odd_mask.any():
                    logits_o = probe_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                    preds[odd_mask] = logits_o.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits_o.reshape(-1, OPTIONS),
                        y[odd_mask].reshape(-1)).item())
                correct += (preds == y).sum().item()
                total += y.numel()

        acc = correct / total
        best_acc = max(best_acc, acc)
        scheduler.step(np.mean(losses))
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: acc={acc:.4%}  loss={np.mean(losses):.5f}  lr={cur_lr:.2e}",
              flush=True)

    return best_acc


def _compute_labels_batch(games, pos_start, pos_end):
    """Compute board state labels for a list of games.

    Returns (labels, positions) tensors.
    """
    all_labels = []
    all_positions = []
    for game in games:
        states = seq_to_state_normal(game)
        for t in range(pos_start, pos_end):
            lbl = states_to_labels(states[t:t+1].reshape(1, 8, 8))
            all_labels.append(lbl.reshape(64))
            all_positions.append(t)
    return (torch.tensor(np.stack(all_labels), dtype=torch.long),
            torch.tensor(all_positions, dtype=torch.long))


def _precompute_heuristic_to_memmap(compiled_rules, games, pos_start, pos_end,
                                     n_rules, out_path, chunk_size=5000):
    """Compute heuristic features in chunks, writing to a memory-mapped file.

    Returns the memmap array of shape (n_samples, n_rules).
    """
    n_per_game = pos_end - pos_start
    n_samples = len(games) * n_per_game

    # Create memmap file
    fp = np.memmap(out_path, dtype=np.float32, mode='w+',
                   shape=(n_samples, n_rules))

    for start in range(0, len(games), chunk_size):
        end = min(start + chunk_size, len(games))
        chunk_games = games[start:end]
        chunk_features = _compute_heuristic_batch(
            compiled_rules, chunk_games, pos_start, pos_end)
        sample_start = start * n_per_game
        sample_end = end * n_per_game
        fp[sample_start:sample_end] = chunk_features
        fp.flush()
        print(f"    Games {start}-{end} / {len(games)}", flush=True)

    return fp


def experiment_learned_heuristic(args):
    """Approach 3: Heuristic features + learnable linear combos."""
    device = get_device()
    print(f"Device: {device}")

    # Load heuristic features (convert mode)
    print("Loading heuristic rules...")
    rules_data = _load_rules_json()

    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    n_eval = max(int(len(games) * 0.1), 100)
    train_games = games[:len(games) - n_eval]
    eval_games = games[len(games) - n_eval:]
    print(f"Using {len(games)} games ({len(train_games)} train, {len(eval_games)} eval)")

    parsed_rules = _build_heuristic_features(rules_data, mode="convert")
    n_rules = len(parsed_rules)
    print(f"  {n_rules} rules")

    # Compile rules for vectorized evaluation
    compiled = _compile_rules(parsed_rules)

    # Use memmap for large datasets, in-memory for small ones
    use_memmap = len(games) > 20000
    tmp_dir = os.path.join(args.output_dir, "_tmp")

    if use_memmap:
        os.makedirs(tmp_dir, exist_ok=True)
        print("Computing heuristic features for train (vectorized, memmap)...")
        t0 = time.time()
        tr_path = os.path.join(tmp_dir, "tr_heur.dat")
        tr_heur_mm = _precompute_heuristic_to_memmap(
            compiled, train_games, POS_START, POS_END, n_rules, tr_path)
        n_tr = len(train_games) * LENGTH
        print(f"  {n_tr} samples in {time.time()-t0:.1f}s")

        print("Computing heuristic features for eval (vectorized, memmap)...")
        t0 = time.time()
        ev_path = os.path.join(tmp_dir, "ev_heur.dat")
        ev_heur_mm = _precompute_heuristic_to_memmap(
            compiled, eval_games, POS_START, POS_END, n_rules, ev_path)
        n_ev = len(eval_games) * LENGTH
        print(f"  {n_ev} samples in {time.time()-t0:.1f}s")
    else:
        print("Computing heuristic features for train (vectorized)...")
        t0 = time.time()
        tr_heur_np = _compute_heuristic_batch(compiled, train_games, POS_START, POS_END)
        print(f"  {len(tr_heur_np)} samples in {time.time()-t0:.1f}s")

        print("Computing heuristic features for eval (vectorized)...")
        t0 = time.time()
        ev_heur_np = _compute_heuristic_batch(compiled, eval_games, POS_START, POS_END)
        print(f"  {len(ev_heur_np)} samples in {time.time()-t0:.1f}s")

    print("Computing labels...")
    tr_labels, tr_positions = _compute_labels_batch(train_games, POS_START, POS_END)
    ev_labels, ev_positions = _compute_labels_batch(eval_games, POS_START, POS_END)

    # Convert to tensors — torch can wrap numpy memmap arrays without copying
    if use_memmap:
        tr_heur = torch.from_numpy(tr_heur_mm)  # backed by memmap, no copy
        ev_heur = torch.tensor(np.array(ev_heur_mm), dtype=torch.float32)
        del ev_heur_mm
    else:
        tr_heur = torch.tensor(tr_heur_np, dtype=torch.float32)
        del tr_heur_np
        ev_heur = torch.tensor(ev_heur_np, dtype=torch.float32)
        del ev_heur_np

    results = {}

    for K in [200, 500, 1000]:
        # With original features
        print(f"\n--- K={K}, heuristic + learned (concat) ---")
        acc_concat = _train_learned_heuristic(
            tr_heur, tr_labels, tr_positions,
            ev_heur, ev_labels, ev_positions,
            device, n_rules, K, use_original=True)
        results[f"K{K}_concat"] = acc_concat

        # Learned only (no original)
        print(f"\n--- K={K}, learned only (no original heuristics) ---")
        acc_learned = _train_learned_heuristic(
            tr_heur, tr_labels, tr_positions,
            ev_heur, ev_labels, ev_positions,
            device, n_rules, K, use_original=False)
        results[f"K{K}_learned_only"] = acc_learned

    print(f"\n{'='*60}")
    print("LEARNED HEURISTIC RESULTS")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v:.4%}")

    _save_results(args, "learned_heuristic", results)

    # Clean up memmap temp files
    if use_memmap:
        del tr_heur
        import gc; gc.collect()
        for f in [tr_path, ev_path]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)

    return results


# ==================== Approach 2: MLP on Move-History =======================

def _build_mlp(input_dim, hidden_dim, output_dim, num_hidden_layers=1):
    """Build an MLP with the given number of hidden layers."""
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_hidden_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def _eval_mlp_nanda(mlp_even, mlp_odd, ev_X, ev_Y, ev_pos, device, batch_size=1024):
    """Evaluate an even/odd MLP pair. Returns (accuracy, mean_loss)."""
    mlp_even.eval()
    mlp_odd.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for i in range(0, len(ev_X), batch_size):
            x = ev_X[i:i + batch_size].to(device)
            y = ev_Y[i:i + batch_size].to(device)
            pos = ev_pos[i:i + batch_size]
            even_mask = (pos % 2 == 0)
            odd_mask = ~even_mask

            preds = torch.zeros_like(y)
            if even_mask.any():
                logits_e = mlp_even(x[even_mask]).view(-1, 64, OPTIONS)
                preds[even_mask] = logits_e.argmax(-1)
                losses.append(nn.functional.cross_entropy(
                    logits_e.reshape(-1, OPTIONS),
                    y[even_mask].reshape(-1)).item())
            if odd_mask.any():
                logits_o = mlp_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                preds[odd_mask] = logits_o.argmax(-1)
                losses.append(nn.functional.cross_entropy(
                    logits_o.reshape(-1, OPTIONS),
                    y[odd_mask].reshape(-1)).item())
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / total, np.mean(losses)


def _train_mlp_nanda(tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos,
                     device, input_dim, hidden_dim,
                     lr=1e-3, epochs=16, batch_size=1024,
                     num_hidden_layers=1, save_path=None):
    """Train an MLP with Nanda's even/odd split.

    Returns (best_acc, mlp_even, mlp_odd) if save_path is given, else best_acc.
    If save_path is given, saves best checkpoint to save_path.
    """
    mlp_even = _build_mlp(input_dim, hidden_dim, 64 * OPTIONS, num_hidden_layers).to(device)
    mlp_odd = _build_mlp(input_dim, hidden_dim, 64 * OPTIONS, num_hidden_layers).to(device)

    all_params = list(mlp_even.parameters()) + list(mlp_odd.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    best_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        mlp_even.train()
        mlp_odd.train()
        perm = torch.randperm(len(tr_X))
        for i in range(0, len(tr_X), batch_size):
            idx = perm[i:i + batch_size]
            x = tr_X[idx].to(device)
            y = tr_Y[idx].to(device)
            pos = tr_pos[idx]
            even_mask = (pos % 2 == 0)
            odd_mask = ~even_mask

            loss = torch.tensor(0.0, device=device)
            if even_mask.any():
                logits_e = mlp_even(x[even_mask]).view(-1, 64, OPTIONS)
                loss = loss + nn.functional.cross_entropy(
                    logits_e.reshape(-1, OPTIONS), y[even_mask].reshape(-1))
            if odd_mask.any():
                logits_o = mlp_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                loss = loss + nn.functional.cross_entropy(
                    logits_o.reshape(-1, OPTIONS), y[odd_mask].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval
        acc, mean_loss = _eval_mlp_nanda(mlp_even, mlp_odd, ev_X, ev_Y, ev_pos,
                                         device, batch_size)
        if acc > best_acc:
            best_acc = acc
            best_state = {
                'even': {k: v.cpu().clone() for k, v in mlp_even.state_dict().items()},
                'odd': {k: v.cpu().clone() for k, v in mlp_odd.state_dict().items()},
            }
        scheduler.step(mean_loss)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: acc={acc:.4%}  loss={mean_loss:.5f}  lr={cur_lr:.2e}",
              flush=True)

    # Restore best and optionally save
    if best_state is not None:
        mlp_even.load_state_dict(best_state['even'])
        mlp_odd.load_state_dict(best_state['odd'])
        if save_path is not None:
            torch.save({
                'even': best_state['even'],
                'odd': best_state['odd'],
                'hidden_dim': hidden_dim,
                'input_dim': input_dim,
                'num_hidden_layers': num_hidden_layers,
                'best_acc': best_acc,
            }, save_path)
            print(f"  Saved checkpoint to {save_path}")

    if save_path is not None:
        return best_acc, mlp_even, mlp_odd
    return best_acc


def _expand_pairwise_batch(base_X, batch_idx):
    """Expand base features (180-d) to pairwise features (3780-d) for a batch.

    base_X: (N, 180) tensor of [played(60), when(60), even(60)]
    batch_idx: indices into base_X
    Returns (batch_size, 3780) tensor.
    """
    x = base_X[batch_idx]  # (B, 180)
    played = x[:, :N_MOVES]     # (B, 60)
    # when = x[:, N_MOVES:2*N_MOVES]  # not used in pairwise
    even = x[:, 2*N_MOVES:]     # (B, 60)

    idx_i, idx_j = np.triu_indices(N_MOVES, k=1)
    idx_i_t = torch.tensor(idx_i, dtype=torch.long)
    idx_j_t = torch.tensor(idx_j, dtype=torch.long)

    pp = played[:, idx_i_t] * played[:, idx_j_t]  # (B, 1770)
    pe = played * even                              # (B, 60)
    ee = even[:, idx_i_t] * even[:, idx_j_t]       # (B, 1770)

    return torch.cat([x, pp, pe, ee], dim=1)  # (B, 3780)


# Cache triu indices for efficiency
_TRIU_CACHE = {}

def _expand_pairwise_batch_cached(base_X, batch_idx, device=None):
    """Like _expand_pairwise_batch but caches triu indices."""
    global _TRIU_CACHE
    if 'idx' not in _TRIU_CACHE:
        idx_i, idx_j = np.triu_indices(N_MOVES, k=1)
        _TRIU_CACHE['idx_i'] = torch.tensor(idx_i, dtype=torch.long)
        _TRIU_CACHE['idx_j'] = torch.tensor(idx_j, dtype=torch.long)

    x = base_X[batch_idx]  # (B, 180)
    played = x[:, :N_MOVES]
    even = x[:, 2*N_MOVES:]
    idx_i = _TRIU_CACHE['idx_i']
    idx_j = _TRIU_CACHE['idx_j']

    pp = played[:, idx_i] * played[:, idx_j]
    pe = played * even
    ee = even[:, idx_i] * even[:, idx_j]

    return torch.cat([x, pp, pe, ee], dim=1)


def _train_mlp_nanda_onthefly(tr_base, tr_Y, tr_pos, ev_base, ev_Y, ev_pos,
                               device, hidden_dim, lr=1e-3, epochs=16,
                               batch_size=1024):
    """Train MLP on 3780-d pairwise features computed on-the-fly from 180-d base.

    This avoids storing all 3780-d features in memory.
    """
    input_dim = 3780
    mlp_even = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 64 * OPTIONS),
    ).to(device)
    mlp_odd = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 64 * OPTIONS),
    ).to(device)

    all_params = list(mlp_even.parameters()) + list(mlp_odd.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    n_train = len(tr_base)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        mlp_even.train()
        mlp_odd.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            x = _expand_pairwise_batch_cached(tr_base, idx).to(device)
            y = tr_Y[idx].to(device)
            pos = tr_pos[idx]
            even_mask = (pos % 2 == 0)
            odd_mask = ~even_mask

            loss = torch.tensor(0.0, device=device)
            if even_mask.any():
                logits_e = mlp_even(x[even_mask]).view(-1, 64, OPTIONS)
                loss = loss + nn.functional.cross_entropy(
                    logits_e.reshape(-1, OPTIONS), y[even_mask].reshape(-1))
            if odd_mask.any():
                logits_o = mlp_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                loss = loss + nn.functional.cross_entropy(
                    logits_o.reshape(-1, OPTIONS), y[odd_mask].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval
        mlp_even.eval()
        mlp_odd.eval()
        correct = 0
        total = 0
        losses = []
        with torch.no_grad():
            for i in range(0, len(ev_base), batch_size):
                idx_ev = list(range(i, min(i + batch_size, len(ev_base))))
                x = _expand_pairwise_batch_cached(ev_base, idx_ev).to(device)
                y = ev_Y[i:i + batch_size].to(device)
                pos = ev_pos[i:i + batch_size]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                preds = torch.zeros_like(y)
                if even_mask.any():
                    logits_e = mlp_even(x[even_mask]).view(-1, 64, OPTIONS)
                    preds[even_mask] = logits_e.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits_e.reshape(-1, OPTIONS),
                        y[even_mask].reshape(-1)).item())
                if odd_mask.any():
                    logits_o = mlp_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                    preds[odd_mask] = logits_o.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits_o.reshape(-1, OPTIONS),
                        y[odd_mask].reshape(-1)).item())
                correct += (preds == y).sum().item()
                total += y.numel()

        acc = correct / total
        best_acc = max(best_acc, acc)
        scheduler.step(np.mean(losses))
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: acc={acc:.4%}  loss={np.mean(losses):.5f}  lr={cur_lr:.2e}",
              flush=True)

    return best_acc


def _chunk_features_path(output_dir, chunk_id):
    """Return path for a chunk's cached features file."""
    return os.path.join(output_dir, "feature_chunks", f"chunk_{chunk_id:04d}.npz")


def _save_features(path, X, Y, pos):
    """Save precomputed features to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    feat = X.numpy() if hasattr(X, 'numpy') else X
    lab = Y.numpy() if hasattr(Y, 'numpy') else Y
    pos_arr = pos.numpy() if hasattr(pos, 'numpy') else pos
    np.savez(path,
             features=feat.astype(np.float16),
             labels=lab.astype(np.int8),
             positions=pos_arr.astype(np.int8))
    print(f"  Saved to {path} ({os.path.getsize(path) / 1e9:.2f} GB)")


def _load_features(path):
    """Load precomputed features from disk."""
    data = np.load(path)
    X = torch.tensor(data['features'].astype(np.float32))
    Y = torch.tensor(data['labels'].astype(np.int64))
    pos = torch.tensor(data['positions'].astype(np.int64))
    return X, Y, pos


def experiment_precompute(args):
    """Precompute and save 180-d move features + labels to disk.

    Supports chunked computation for parallel SLURM jobs:
      --file-start 0 --file-end 10 --chunk-id 0
    Each chunk processes files [file_start, file_end) and saves independently.

    Or compute all at once:
      --experiment precompute --max-files 30
    """
    import pickle

    if args.chunk_id is not None:
        # Chunked mode: process only files [file_start, file_end)
        files = sorted(f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".pickle"))
        chunk_files = files[args.file_start:args.file_end]
        print(f"Chunk {args.chunk_id}: files {args.file_start}-{args.file_end-1} "
              f"({len(chunk_files)} files)")

        games = []
        for fname in chunk_files:
            with open(os.path.join(SYNTHETIC_DIR, fname), "rb") as f:
                batch = pickle.load(f)
            games.extend(g for g in batch if len(g) == GAME_LEN)
        print(f"  Loaded {len(games)} games")

        print("  Building features (180-d)...")
        X, Y, pos = _build_move_features_batch(
            games, POS_START, POS_END, include_pairwise=False)

        path = _chunk_features_path(args.output_dir, args.chunk_id)
        _save_features(path, X, Y, pos)
        print(f"  Chunk {args.chunk_id}: {X.shape} -> {path}")
    else:
        # Single-job mode: process all files
        games = load_games(max_files=args.max_files)
        if args.max_games and len(games) > args.max_games:
            games = games[:args.max_games]
        n_games = len(games)
        print(f"Using {n_games} games")

        print("Building features (180-d)...")
        X, Y, pos = _build_move_features_batch(
            games, POS_START, POS_END, include_pairwise=False)

        path = _chunk_features_path(args.output_dir, 0)
        _save_features(path, X, Y, pos)
        print(f"\nSaved {X.shape} -> {path}")

    print("Done.")


def _load_all_chunks(output_dir, eval_frac=0.1):
    """Load all chunk files and split into train/eval.

    Returns (tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos).
    """
    chunk_dir = os.path.join(output_dir, "feature_chunks")
    chunk_files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".npz"))
    print(f"Loading {len(chunk_files)} chunks from {chunk_dir}...")

    all_X, all_Y, all_pos = [], [], []
    for fname in chunk_files:
        path = os.path.join(chunk_dir, fname)
        X, Y, pos = _load_features(path)
        all_X.append(X)
        all_Y.append(Y)
        all_pos.append(pos)
        print(f"  {fname}: {X.shape[0]} samples")

    X = torch.cat(all_X)
    Y = torch.cat(all_Y)
    pos = torch.cat(all_pos)
    print(f"Total: {X.shape[0]} samples, {X.shape}")

    # Split into train/eval (last eval_frac of samples)
    n_total = X.shape[0]
    n_eval = max(int(n_total * eval_frac), 49 * 100)  # at least 100 games worth
    n_train = n_total - n_eval

    return X[:n_train], Y[:n_train], pos[:n_train], X[n_train:], Y[n_train:], pos[n_train:]


def _try_load_precomputed(args):
    """Try to load precomputed features. Returns (tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos) or None."""
    if not args.precomputed:
        return None
    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    if not os.path.exists(chunk_dir):
        print(f"  No feature_chunks directory in {args.output_dir}")
        return None
    chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith(".npz")]
    if not chunk_files:
        print(f"  No chunk files found in {chunk_dir}")
        return None
    return _load_all_chunks(args.output_dir)


def experiment_mlp(args):
    """Approach 2: MLP on move-history features (180-d and 3780-d)."""
    device = get_device()
    print(f"Device: {device}")

    cached = _try_load_precomputed(args)
    if cached is not None:
        tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos = cached
    else:
        games = load_games(max_files=args.max_files)
        if args.max_games and len(games) > args.max_games:
            games = games[:args.max_games]
        n_eval = max(int(len(games) * 0.1), 100)
        train_games = games[:len(games) - n_eval]
        eval_games = games[len(games) - n_eval:]
        print(f"Using {len(games)} games ({len(train_games)} train, {len(eval_games)} eval)")

        # Build 180-d base features (kept in memory for all experiments)
        print("Building train features (180-d base)...")
        tr_X, tr_Y, tr_pos = _build_move_features_batch(
            train_games, POS_START, POS_END, include_pairwise=False)
        print("Building eval features...")
        ev_X, ev_Y, ev_pos = _build_move_features_batch(
            eval_games, POS_START, POS_END, include_pairwise=False)

    # Cap at max_games worth of samples if requested
    if args.max_games:
        max_samples = args.max_games * LENGTH
        if len(tr_X) + len(ev_X) > max_samples:
            n_eval = max(int(max_samples * 0.1), 49 * 100)
            n_train = max_samples - n_eval
            if n_train < len(tr_X):
                tr_X, tr_Y, tr_pos = tr_X[:n_train], tr_Y[:n_train], tr_pos[:n_train]
            if n_eval < len(ev_X):
                ev_X, ev_Y, ev_pos = ev_X[:n_eval], ev_Y[:n_eval], ev_pos[:n_eval]
            print(f"  Capped to ~{args.max_games} games: {len(tr_X)} train, {len(ev_X)} eval samples")

    print(f"  Feature shape: {tr_X.shape}")

    # Determine which hidden dims to run
    hidden_dims = args.mlp_hidden if args.mlp_hidden else [256, 512, 1024]

    results = {}

    if not args.mlp_only:
        # Linear baseline (no hidden layer) with even/odd split
        print("\n--- Linear (no hidden layer) with even/odd split on 180-d ---")
        acc_linear = train_nanda_probe(
            tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos, device, 180)
        results["linear_180_nanda"] = acc_linear

    # MLP on 180-d with various hidden dimensions
    n_layers = getattr(args, 'mlp_layers', 1)
    n_epochs = getattr(args, 'epochs', None) or 16
    ckpt_dir = os.path.join(args.output_dir, "mlp_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for H in hidden_dims:
        layer_str = f"_{n_layers}L" if n_layers > 1 else ""
        print(f"\n--- MLP 180-d hidden={H} x{n_layers} layers, {n_epochs} epochs ---")
        save_path = os.path.join(ckpt_dir, f"mlp_180_H{H}{layer_str}.pt")
        acc_mlp = _train_mlp_nanda(
            tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos,
            device, 180, H, num_hidden_layers=n_layers,
            epochs=n_epochs, save_path=save_path)
        if isinstance(acc_mlp, tuple):
            acc_mlp = acc_mlp[0]  # _train_mlp_nanda returns (acc, even, odd) when save_path set
        results[f"mlp_180_H{H}{layer_str}"] = acc_mlp

    # MLP on 3780-d pairwise features (computed on-the-fly from 180-d base)
    if not args.mlp_only:
        for H in hidden_dims:
            print(f"\n--- MLP 3780-d pairwise hidden={H} with even/odd split ---")
            acc_mlp_pw = _train_mlp_nanda_onthefly(
                tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos,
                device, H)
            results[f"mlp_3780_H{H}"] = acc_mlp_pw

    print(f"\n{'='*60}")
    print("MLP ON MOVE-HISTORY RESULTS")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v:.4%}")

    _save_results(args, "mlp", results)

    # Plot accuracy vs hidden dim
    mlp_results = {k: v for k, v in results.items() if k.startswith("mlp_180_H")}
    if len(mlp_results) > 1:
        _plot_mlp_width_sweep(mlp_results, args.output_dir)

    return results


def _plot_mlp_width_sweep(mlp_results, output_dir):
    """Plot accuracy vs hidden dimension."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Parse H values and accuracies
    items = []
    for k, v in mlp_results.items():
        # k is like "mlp_180_H64" or "mlp_180_H1024_2L"
        h_str = k.split("_H")[1].split("_")[0]
        items.append((int(h_str), v * 100))
    items.sort()
    hs, accs = zip(*items)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hs, accs, 'o-', markersize=6, linewidth=2)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Hidden dimension (H)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('MLP on 180-d Move Features: Accuracy vs Width')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(hs)
    ax.set_xticklabels([str(h) for h in hs], rotation=45)

    out_path = os.path.join(output_dir, 'mlp_width_sweep.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {out_path}")
    plt.close()


def experiment_mlp_dropout(args):
    """Evaluate trained MLP with hidden unit dropout (zeroing out units)."""
    device = get_device()
    print(f"Device: {device}")

    cached = _try_load_precomputed(args)
    if cached is not None:
        tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos = cached
    else:
        games = load_games(max_files=args.max_files)
        if args.max_games and len(games) > args.max_games:
            games = games[:args.max_games]
        n_eval = max(int(len(games) * 0.1), 100)
        train_games = games[:len(games) - n_eval]
        eval_games = games[len(games) - n_eval:]
        print(f"Using {len(games)} games ({len(train_games)} train, {len(eval_games)} eval)")
        print("Building eval features (180-d base)...")
        _, _, _ = None, None, None  # don't need train features
        ev_X, ev_Y, ev_pos = _build_move_features_batch(
            eval_games, POS_START, POS_END, include_pairwise=False)

    print(f"  Eval shape: {ev_X.shape}")

    # Load the H=1024 checkpoint
    ckpt_dir = os.path.join(args.output_dir, "mlp_checkpoints")
    H = args.mlp_hidden[0] if args.mlp_hidden else 1024
    n_layers = getattr(args, 'mlp_layers', 1)
    layer_str = f"_{n_layers}L" if n_layers > 1 else ""
    ckpt_path = os.path.join(ckpt_dir, f"mlp_180_H{H}{layer_str}.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    mlp_even = _build_mlp(ckpt['input_dim'], ckpt['hidden_dim'],
                          64 * OPTIONS, ckpt.get('num_hidden_layers', 1)).to(device)
    mlp_odd = _build_mlp(ckpt['input_dim'], ckpt['hidden_dim'],
                         64 * OPTIONS, ckpt.get('num_hidden_layers', 1)).to(device)
    mlp_even.load_state_dict(ckpt['even'])
    mlp_odd.load_state_dict(ckpt['odd'])

    # Baseline accuracy (no dropout)
    base_acc, _ = _eval_mlp_nanda(mlp_even, mlp_odd, ev_X, ev_Y, ev_pos, device)
    print(f"Baseline (0 dropped): {base_acc:.4%}")

    # Dropout schedule: 0, 4, 8, 16, 32, 64, 128, 256, 512
    drop_counts = [0, 4, 8, 16, 32, 64, 128, 256, 512]
    drop_counts = [d for d in drop_counts if d <= H]

    # Rank units by L2 norm of output weights (importance)
    # For a Sequential: [Linear, ReLU, Linear], output layer is index 2
    # Weight shape: (64*3, H) — each column is one hidden unit's contribution
    out_weight_even = mlp_even[-1].weight.data  # (192, H)
    out_weight_odd = mlp_odd[-1].weight.data    # (192, H)
    importance_even = out_weight_even.norm(dim=0)  # (H,)
    importance_odd = out_weight_odd.norm(dim=0)

    # Sort by importance (least important first for progressive dropout)
    order_even = importance_even.argsort()  # ascending
    order_odd = importance_odd.argsort()

    results = {}
    n_trials = 5  # average over random orderings too

    for n_drop in drop_counts:
        if n_drop == 0:
            results[0] = base_acc
            continue

        # Method 1: Drop least important units
        accs_least = []
        for mlp, order, state_key in [(mlp_even, order_even, 'even'),
                                       (mlp_odd, order_odd, 'odd')]:
            mlp.load_state_dict(ckpt[state_key])

        # Zero out least important units in the first hidden layer output
        mask_even = torch.ones(H, device=device)
        mask_odd = torch.ones(H, device=device)
        mask_even[order_even[:n_drop]] = 0
        mask_odd[order_odd[:n_drop]] = 0

        # Apply mask by modifying the bias and zeroing weights
        # Hook approach: modify forward pass
        with torch.no_grad():
            # Save original weights
            orig_even_w = mlp_even[0].weight.data.clone()
            orig_even_b = mlp_even[0].bias.data.clone()
            orig_odd_w = mlp_odd[0].weight.data.clone()
            orig_odd_b = mlp_odd[0].bias.data.clone()

            # Zero out dropped units' rows in first layer
            mlp_even[0].weight.data[order_even[:n_drop]] = 0
            mlp_even[0].bias.data[order_even[:n_drop]] = 0
            mlp_odd[0].weight.data[order_odd[:n_drop]] = 0
            mlp_odd[0].bias.data[order_odd[:n_drop]] = 0

        acc_least, _ = _eval_mlp_nanda(mlp_even, mlp_odd, ev_X, ev_Y, ev_pos, device)

        # Method 2: Drop random units (average over trials)
        accs_random = []
        for trial in range(n_trials):
            mlp_even.load_state_dict(ckpt['even'])
            mlp_odd.load_state_dict(ckpt['odd'])
            mlp_even.to(device)
            mlp_odd.to(device)

            rng = np.random.RandomState(trial)
            rand_even = rng.permutation(H)[:n_drop]
            rand_odd = rng.permutation(H)[:n_drop]

            with torch.no_grad():
                mlp_even[0].weight.data[rand_even] = 0
                mlp_even[0].bias.data[rand_even] = 0
                mlp_odd[0].weight.data[rand_odd] = 0
                mlp_odd[0].bias.data[rand_odd] = 0

            acc_rand, _ = _eval_mlp_nanda(mlp_even, mlp_odd, ev_X, ev_Y, ev_pos, device)
            accs_random.append(acc_rand)

        acc_random = np.mean(accs_random)
        results[n_drop] = {
            'least_important': acc_least,
            'random': acc_random,
            'random_std': np.std(accs_random),
        }
        print(f"  Drop {n_drop:>4}: least_important={acc_least:.4%}  "
              f"random={acc_random:.4%} ± {np.std(accs_random):.4%}")

        # Restore
        mlp_even.load_state_dict(ckpt['even'])
        mlp_odd.load_state_dict(ckpt['odd'])
        mlp_even.to(device)
        mlp_odd.to(device)

    # Plot
    _plot_mlp_dropout(results, H, args.output_dir)
    _save_results(args, "mlp_dropout", results)
    return results


def _plot_mlp_dropout(results, H, output_dir):
    """Plot accuracy vs number of dropped hidden units."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    drop_counts = sorted(results.keys())
    acc_least = []
    acc_random = []
    acc_random_std = []
    for d in drop_counts:
        if d == 0:
            acc_least.append(results[0] * 100)
            acc_random.append(results[0] * 100)
            acc_random_std.append(0)
        else:
            acc_least.append(results[d]['least_important'] * 100)
            acc_random.append(results[d]['random'] * 100)
            acc_random_std.append(results[d]['random_std'] * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(drop_counts, acc_least, 'o-', label='Drop least important', markersize=6)
    ax.errorbar(drop_counts, acc_random, yerr=acc_random_std,
                fmt='s-', label='Drop random', markersize=5, capsize=3)
    ax.set_xlabel(f'Number of hidden units dropped (out of {H})')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'MLP H={H} on 180-d Features: Effect of Dropping Hidden Units')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, 'mlp_dropout.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {out_path}")
    plt.close()


# ================== Part A: Fix Heuristic + Activations Combination =========

def _train_nanda_probe_returning_probes(train_X, train_Y, train_positions,
                                         eval_X, eval_Y, eval_positions,
                                         device, input_dim, lr=1e-3, epochs=16,
                                         batch_size=1024):
    """Like train_nanda_probe but also returns the trained probes."""
    probe_even = nn.Linear(input_dim, 64 * OPTIONS).to(device)
    probe_odd = nn.Linear(input_dim, 64 * OPTIONS).to(device)
    optimizer = torch.optim.Adam(
        list(probe_even.parameters()) + list(probe_odd.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    best_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        probe_even.train()
        probe_odd.train()
        perm = torch.randperm(len(train_X))
        for i in range(0, len(train_X), batch_size):
            idx = perm[i:i + batch_size]
            x = train_X[idx].to(device)
            y = train_Y[idx].to(device)
            pos = train_positions[idx]
            even_mask = (pos % 2 == 0)
            odd_mask = ~even_mask
            loss = torch.tensor(0.0, device=device)
            if even_mask.any():
                logits_e = probe_even(x[even_mask]).view(-1, 64, OPTIONS)
                loss = loss + nn.functional.cross_entropy(
                    logits_e.reshape(-1, OPTIONS), y[even_mask].reshape(-1))
            if odd_mask.any():
                logits_o = probe_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                loss = loss + nn.functional.cross_entropy(
                    logits_o.reshape(-1, OPTIONS), y[odd_mask].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe_even.eval()
        probe_odd.eval()
        correct = 0
        total = 0
        losses = []
        with torch.no_grad():
            for i in range(0, len(eval_X), batch_size):
                x = eval_X[i:i + batch_size].to(device)
                y = eval_Y[i:i + batch_size].to(device)
                pos = eval_positions[i:i + batch_size]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask
                preds = torch.zeros_like(y)
                if even_mask.any():
                    logits_e = probe_even(x[even_mask]).view(-1, 64, OPTIONS)
                    preds[even_mask] = logits_e.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits_e.reshape(-1, OPTIONS), y[even_mask].reshape(-1)).item())
                if odd_mask.any():
                    logits_o = probe_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                    preds[odd_mask] = logits_o.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits_o.reshape(-1, OPTIONS), y[odd_mask].reshape(-1)).item())
                correct += (preds == y).sum().item()
                total += y.numel()
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = {
                'even': {k: v.clone() for k, v in probe_even.state_dict().items()},
                'odd': {k: v.clone() for k, v in probe_odd.state_dict().items()},
            }
        scheduler.step(np.mean(losses))
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: acc={acc:.4%}  loss={np.mean(losses):.5f}  lr={cur_lr:.2e}",
              flush=True)

    # Restore best
    probe_even.load_state_dict(best_state['even'])
    probe_odd.load_state_dict(best_state['odd'])
    return best_acc, probe_even, probe_odd


def _get_nanda_logits(probe_even, probe_odd, X, positions, device,
                      batch_size=1024):
    """Get per-sample logits from trained even/odd probes.

    Returns (N, 64, 3) tensor of logits.
    """
    probe_even.eval()
    probe_odd.eval()
    all_logits = torch.zeros(len(X), 64, OPTIONS)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = X[i:i + batch_size].to(device)
            pos = positions[i:i + batch_size]
            even_mask = (pos % 2 == 0)
            odd_mask = ~even_mask
            if even_mask.any():
                logits_e = probe_even(x[even_mask]).view(-1, 64, OPTIONS)
                all_logits[i:i + batch_size][even_mask] = logits_e.cpu()
            if odd_mask.any():
                logits_o = probe_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                all_logits[i:i + batch_size][odd_mask] = logits_o.cpu()
    return all_logits


def _eval_combined_logits(logits_A, logits_B, Y, alpha, beta):
    """Combine logits and compute accuracy."""
    combined = alpha * logits_A + beta * logits_B  # (N, 64, 3)
    preds = combined.argmax(-1)  # (N, 64)
    correct = (preds == Y).sum().item()
    total = Y.numel()
    return correct / total


def experiment_combine_heuristic(args):
    """Part A: Fix heuristic + activations combination."""
    device = get_device()
    print(f"Device: {device}")

    # Load rules
    print("Loading heuristic rules...")
    rules_data = _load_rules_json()

    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    n_eval = max(int(len(games) * 0.1), 100)
    train_games = games[:len(games) - n_eval]
    eval_games = games[len(games) - n_eval:]
    print(f"Using {len(games)} games ({len(train_games)} train, {len(eval_games)} eval)")

    # Build heuristic features (convert mode)
    parsed_rules = _build_heuristic_features(rules_data, mode="convert")
    n_rules = len(parsed_rules)
    print(f"  {n_rules} rules (convert mode)")

    compiled_rules = _compile_rules(parsed_rules)

    print("  Computing heuristic features for train...")
    tr_heur = torch.tensor(
        _compute_heuristic_batch(compiled_rules, train_games, POS_START, POS_END),
        dtype=torch.float32)

    print("  Computing heuristic features for eval...")
    ev_heur = torch.tensor(
        _compute_heuristic_batch(compiled_rules, eval_games, POS_START, POS_END),
        dtype=torch.float32)

    # Build labels and positions
    print("  Computing labels...")
    tr_labels_list = []
    tr_positions = []
    for game in train_games:
        states = seq_to_state_normal(game)
        for t in range(POS_START, POS_END):
            lbl = states_to_labels(states[t:t+1].reshape(1, 8, 8))
            tr_labels_list.append(lbl.reshape(64))
            tr_positions.append(t)
    tr_labels = torch.tensor(np.stack(tr_labels_list), dtype=torch.long)
    tr_positions = torch.tensor(tr_positions, dtype=torch.long)

    ev_labels_list = []
    ev_positions = []
    for game in eval_games:
        states = seq_to_state_normal(game)
        for t in range(POS_START, POS_END):
            lbl = states_to_labels(states[t:t+1].reshape(1, 8, 8))
            ev_labels_list.append(lbl.reshape(64))
            ev_positions.append(t)
    ev_labels = torch.tensor(np.stack(ev_labels_list), dtype=torch.long)
    ev_positions = torch.tensor(ev_positions, dtype=torch.long)

    # Get activations from random and Othello-GPT models
    random_model = create_random_model(device, block_size=59)
    othello_model, othello_block_size = load_model(args.ckpt_path, device)

    print("  Collecting random L0 activations...")
    tr_X_r, _ = collect_activations_and_labels(
        random_model, train_games, device, 0, 59)
    ev_X_r, _ = collect_activations_and_labels(
        random_model, eval_games, device, 0, 59)

    print("  Collecting Othello-GPT L0 activations...")
    tr_X_o, _ = collect_activations_and_labels(
        othello_model, train_games, device, 0, othello_block_size)
    ev_X_o, _ = collect_activations_and_labels(
        othello_model, eval_games, device, 0, othello_block_size)

    results = {}

    # ---- Method 1: Train separately, combine predictions ----
    print(f"\n{'='*60}")
    print("Method 1: Separate probes, combined predictions")
    print(f"{'='*60}")

    # Train probe A on heuristic features
    print("\n  Training probe A on heuristic features...")
    acc_heur, probe_heur_even, probe_heur_odd = _train_nanda_probe_returning_probes(
        tr_heur, tr_labels, tr_positions,
        ev_heur, ev_labels, ev_positions,
        device, n_rules)
    results["heuristic_alone"] = acc_heur
    print(f"  Heuristic alone: {acc_heur:.4%}")

    # Train probe B on random L0
    print("\n  Training probe B on random L0...")
    acc_rand, probe_rand_even, probe_rand_odd = _train_nanda_probe_returning_probes(
        tr_X_r, tr_labels, tr_positions,
        ev_X_r, ev_labels, ev_positions,
        device, 512)
    results["random_L0_alone"] = acc_rand
    print(f"  Random L0 alone: {acc_rand:.4%}")

    # Train probe C on Othello-GPT L0
    print("\n  Training probe C on Othello-GPT L0...")
    acc_ogpt, probe_ogpt_even, probe_ogpt_odd = _train_nanda_probe_returning_probes(
        tr_X_o, tr_labels, tr_positions,
        ev_X_o, ev_labels, ev_positions,
        device, 512)
    results["othello_L0_alone"] = acc_ogpt
    print(f"  Othello-GPT L0 alone: {acc_ogpt:.4%}")

    # Get logits from each
    logits_heur = _get_nanda_logits(probe_heur_even, probe_heur_odd,
                                     ev_heur, ev_positions, device)
    logits_rand = _get_nanda_logits(probe_rand_even, probe_rand_odd,
                                     ev_X_r, ev_positions, device)
    logits_ogpt = _get_nanda_logits(probe_ogpt_even, probe_ogpt_odd,
                                     ev_X_o, ev_positions, device)

    # Grid search alpha, beta for heuristic + random
    print("\n  Grid search: heuristic + random L0")
    best_acc_hr = 0.0
    best_ab_hr = (0, 0)
    for a_int in range(11):
        for b_int in range(11):
            alpha = a_int / 10.0
            beta = b_int / 10.0
            acc = _eval_combined_logits(logits_heur, logits_rand, ev_labels, alpha, beta)
            if acc > best_acc_hr:
                best_acc_hr = acc
                best_ab_hr = (alpha, beta)
    results["method1_heur_random"] = best_acc_hr
    results["method1_heur_random_alpha_beta"] = best_ab_hr
    print(f"  Best: {best_acc_hr:.4%} (alpha={best_ab_hr[0]:.1f}, beta={best_ab_hr[1]:.1f})")

    # Grid search alpha, beta for heuristic + Othello-GPT L0
    print("\n  Grid search: heuristic + Othello-GPT L0")
    best_acc_ho = 0.0
    best_ab_ho = (0, 0)
    for a_int in range(11):
        for b_int in range(11):
            alpha = a_int / 10.0
            beta = b_int / 10.0
            acc = _eval_combined_logits(logits_heur, logits_ogpt, ev_labels, alpha, beta)
            if acc > best_acc_ho:
                best_acc_ho = acc
                best_ab_ho = (alpha, beta)
    results["method1_heur_othello"] = best_acc_ho
    results["method1_heur_othello_alpha_beta"] = best_ab_ho
    print(f"  Best: {best_acc_ho:.4%} (alpha={best_ab_ho[0]:.1f}, beta={best_ab_ho[1]:.1f})")

    # ---- Method 2: Stacking / Two-Stage ----
    print(f"\n{'='*60}")
    print("Method 2: Stacking (soft predictions + activations)")
    print(f"{'='*60}")

    # Get soft predictions from heuristic probe on train and eval sets
    tr_logits_heur = _get_nanda_logits(probe_heur_even, probe_heur_odd,
                                        tr_heur, tr_positions, device)
    tr_soft_heur = torch.softmax(tr_logits_heur, dim=-1).reshape(len(tr_heur), -1)  # (N, 192)
    ev_soft_heur = torch.softmax(logits_heur, dim=-1).reshape(len(ev_heur), -1)  # (N, 192)

    # Stack with random L0
    print("\n  Stacking: heuristic soft preds + random L0")
    tr_stack_r = torch.cat([tr_soft_heur, tr_X_r], dim=1)  # (N, 704)
    ev_stack_r = torch.cat([ev_soft_heur, ev_X_r], dim=1)
    acc_stack_r = train_nanda_probe(
        tr_stack_r, tr_labels, tr_positions,
        ev_stack_r, ev_labels, ev_positions,
        device, tr_stack_r.shape[1])
    results["method2_stack_random"] = acc_stack_r
    print(f"  Stacking heur + random L0: {acc_stack_r:.4%}")

    # Stack with Othello-GPT L0
    print("\n  Stacking: heuristic soft preds + Othello-GPT L0")
    tr_stack_o = torch.cat([tr_soft_heur, tr_X_o], dim=1)  # (N, 704)
    ev_stack_o = torch.cat([ev_soft_heur, ev_X_o], dim=1)
    acc_stack_o = train_nanda_probe(
        tr_stack_o, tr_labels, tr_positions,
        ev_stack_o, ev_labels, ev_positions,
        device, tr_stack_o.shape[1])
    results["method2_stack_othello"] = acc_stack_o
    print(f"  Stacking heur + Othello L0: {acc_stack_o:.4%}")

    # ---- Method 3: Feature Normalization ----
    print(f"\n{'='*60}")
    print("Method 3: Normalized concatenation")
    print(f"{'='*60}")

    # Normalize heuristic features
    heur_mean = tr_heur.mean(dim=0, keepdim=True)
    heur_std = tr_heur.std(dim=0, keepdim=True).clamp(min=1e-8)
    tr_heur_norm = (tr_heur - heur_mean) / heur_std
    ev_heur_norm = (ev_heur - heur_mean) / heur_std

    # Normalize random L0
    rand_mean = tr_X_r.mean(dim=0, keepdim=True)
    rand_std = tr_X_r.std(dim=0, keepdim=True).clamp(min=1e-8)
    tr_X_r_norm = (tr_X_r - rand_mean) / rand_std
    ev_X_r_norm = (ev_X_r - rand_mean) / rand_std

    # Normalize Othello-GPT L0
    ogpt_mean = tr_X_o.mean(dim=0, keepdim=True)
    ogpt_std = tr_X_o.std(dim=0, keepdim=True).clamp(min=1e-8)
    tr_X_o_norm = (tr_X_o - ogpt_mean) / ogpt_std
    ev_X_o_norm = (ev_X_o - ogpt_mean) / ogpt_std

    # Normalized concat with random L0
    print("\n  Normalized concat: heuristic + random L0")
    tr_norm_r = torch.cat([tr_heur_norm, tr_X_r_norm], dim=1)
    ev_norm_r = torch.cat([ev_heur_norm, ev_X_r_norm], dim=1)
    acc_norm_r = train_nanda_probe(
        tr_norm_r, tr_labels, tr_positions,
        ev_norm_r, ev_labels, ev_positions,
        device, tr_norm_r.shape[1])
    results["method3_norm_random"] = acc_norm_r
    print(f"  Normalized heur + random L0: {acc_norm_r:.4%}")

    # Normalized concat with Othello-GPT L0
    print("\n  Normalized concat: heuristic + Othello-GPT L0")
    tr_norm_o = torch.cat([tr_heur_norm, tr_X_o_norm], dim=1)
    ev_norm_o = torch.cat([ev_heur_norm, ev_X_o_norm], dim=1)
    acc_norm_o = train_nanda_probe(
        tr_norm_o, tr_labels, tr_positions,
        ev_norm_o, ev_labels, ev_positions,
        device, tr_norm_o.shape[1])
    results["method3_norm_othello"] = acc_norm_o
    print(f"  Normalized heur + Othello L0: {acc_norm_o:.4%}")

    # ---- Method 4: 180-d move features + heuristics ----
    print(f"\n{'='*60}")
    print("Method 4: 180-d move features + heuristic features")
    print(f"{'='*60}")

    # Build 180-d move features
    print("\n  Building 180-d move features for train...")
    tr_move, _, _ = _build_move_features_batch(
        train_games, POS_START, POS_END, include_pairwise=False)
    print("  Building 180-d move features for eval...")
    ev_move, _, _ = _build_move_features_batch(
        eval_games, POS_START, POS_END, include_pairwise=False)

    # 180-d alone (linear probe with even/odd)
    print("\n  180-d move features alone (Nanda probe)")
    acc_move = train_nanda_probe(
        tr_move, tr_labels, tr_positions,
        ev_move, ev_labels, ev_positions,
        device, 180)
    results["move_180_alone"] = acc_move
    print(f"  180-d alone: {acc_move:.4%}")

    # Concat 180-d + heuristic (4946-d)
    print("\n  Concat: 180-d + heuristic features")
    tr_mh = torch.cat([tr_move, tr_heur], dim=1)
    ev_mh = torch.cat([ev_move, ev_heur], dim=1)
    acc_mh = train_nanda_probe(
        tr_mh, tr_labels, tr_positions,
        ev_mh, ev_labels, ev_positions,
        device, tr_mh.shape[1])
    results["move_180_plus_heuristic"] = acc_mh
    print(f"  180-d + heuristic: {acc_mh:.4%}")

    # MLP on 180-d + heuristic (4946-d)
    print("\n  MLP on 180-d + heuristic features (H=1024)")
    acc_mh_mlp = _train_mlp_nanda(
        tr_mh, tr_labels, tr_positions,
        ev_mh, ev_labels, ev_positions,
        device, tr_mh.shape[1], 1024)
    results["mlp_move_plus_heuristic_H1024"] = acc_mh_mlp
    print(f"  MLP 180-d + heuristic H=1024: {acc_mh_mlp:.4%}")

    # Summary
    print(f"\n{'='*60}")
    print("COMBINATION EXPERIMENT RESULTS")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4%}")
        else:
            print(f"  {k}: {v}")

    _save_results(args, "combine_heuristic", results)
    return results


# ================== Part B: Recover Heuristics from the MLP ================

def _train_mlp_nanda_returning_model(tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos,
                                      device, input_dim, hidden_dim,
                                      lr=1e-3, epochs=16, batch_size=1024):
    """Like _train_mlp_nanda but returns the trained models."""
    mlp_even = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 64 * OPTIONS),
    ).to(device)
    mlp_odd = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 64 * OPTIONS),
    ).to(device)

    all_params = list(mlp_even.parameters()) + list(mlp_odd.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    best_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        mlp_even.train()
        mlp_odd.train()
        perm = torch.randperm(len(tr_X))
        for i in range(0, len(tr_X), batch_size):
            idx = perm[i:i + batch_size]
            x = tr_X[idx].to(device)
            y = tr_Y[idx].to(device)
            pos = tr_pos[idx]
            even_mask = (pos % 2 == 0)
            odd_mask = ~even_mask
            loss = torch.tensor(0.0, device=device)
            if even_mask.any():
                logits_e = mlp_even(x[even_mask]).view(-1, 64, OPTIONS)
                loss = loss + nn.functional.cross_entropy(
                    logits_e.reshape(-1, OPTIONS), y[even_mask].reshape(-1))
            if odd_mask.any():
                logits_o = mlp_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                loss = loss + nn.functional.cross_entropy(
                    logits_o.reshape(-1, OPTIONS), y[odd_mask].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mlp_even.eval()
        mlp_odd.eval()
        correct = 0
        total = 0
        losses = []
        with torch.no_grad():
            for i in range(0, len(ev_X), batch_size):
                x = ev_X[i:i + batch_size].to(device)
                y = ev_Y[i:i + batch_size].to(device)
                pos = ev_pos[i:i + batch_size]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask
                preds = torch.zeros_like(y)
                if even_mask.any():
                    logits_e = mlp_even(x[even_mask]).view(-1, 64, OPTIONS)
                    preds[even_mask] = logits_e.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits_e.reshape(-1, OPTIONS),
                        y[even_mask].reshape(-1)).item())
                if odd_mask.any():
                    logits_o = mlp_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                    preds[odd_mask] = logits_o.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits_o.reshape(-1, OPTIONS),
                        y[odd_mask].reshape(-1)).item())
                correct += (preds == y).sum().item()
                total += y.numel()
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = {
                'even': {k: v.clone() for k, v in mlp_even.state_dict().items()},
                'odd': {k: v.clone() for k, v in mlp_odd.state_dict().items()},
            }
        scheduler.step(np.mean(losses))
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: acc={acc:.4%}  loss={np.mean(losses):.5f}  lr={cur_lr:.2e}",
              flush=True)

    mlp_even.load_state_dict(best_state['even'])
    mlp_odd.load_state_dict(best_state['odd'])
    return best_acc, mlp_even, mlp_odd


def _feature_name(idx):
    """Convert feature index (0-179) to human-readable name."""
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
    """Convert cell index (0-63) to board position name."""
    row, col = cell_idx // 8, cell_idx % 8
    return f"{chr(65+row)}{col+1}"


def _class_name(cls_idx):
    """Convert class index to name."""
    return ["empty", "white", "black"][cls_idx]


def _analyze_mlp_weights(mlp, hidden_dim, parity_name):
    """Analyze weight structure of a trained MLP.

    Steps 1-2 and 4: Extract weight structure, top features, and descriptions.
    """
    # Layer 0: Linear(180, hidden_dim) — weights shape (hidden_dim, 180)
    W1 = mlp[0].weight.detach().cpu().numpy()  # (H, 180)
    b1 = mlp[0].bias.detach().cpu().numpy()     # (H,)

    # Layer 2: Linear(hidden_dim, 64*3) — weights shape (192, H)
    W2 = mlp[2].weight.detach().cpu().numpy()  # (192, H)
    b2 = mlp[2].bias.detach().cpu().numpy()     # (192,)

    units = []
    for j in range(hidden_dim):
        # Input weights for unit j
        w_in = W1[j]  # (180,)
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

        # Output weights for unit j — column j of W2
        w_out = W2[:, j]  # (192,)
        # Reshape to (64, 3) to identify cell and class
        w_out_reshaped = w_out.reshape(64, OPTIONS)

        # Top 5 output contributions by absolute weight
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

        # Output L2 norm (for progressive ablation sorting)
        out_l2 = float(np.linalg.norm(w_out))

        # Generate description (Step 4)
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

        # Categorize (Step 5)
        # Count feature types in top 10
        n_played = sum(1 for inp in top_inputs if inp["feature"].startswith("played"))
        n_when = sum(1 for inp in top_inputs if inp["feature"].startswith("when"))
        n_even = sum(1 for inp in top_inputs if inp["feature"].startswith("even"))

        # Determine primary category
        max_count = max(n_played, n_when, n_even)
        if n_played == max_count and n_when < max_count and n_even < max_count:
            category = "placement"
        elif n_when == max_count and n_played < max_count and n_even < max_count:
            category = "temporal"
        elif n_even == max_count and n_played < max_count and n_when < max_count:
            category = "parity"
        else:
            category = "interaction"

        # Output scope: how many cells does it contribute to significantly?
        cell_contributions = np.abs(w_out_reshaped).max(axis=1)  # (64,)
        threshold = cell_contributions.max() * 0.1
        n_significant_cells = int((cell_contributions > threshold).sum())
        output_scope = "global" if n_significant_cells > 5 else "local"

        # Primary prediction class
        class_totals = np.abs(w_out_reshaped).sum(axis=0)  # (3,)
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
            "n_played": n_played,
            "n_when": n_when,
            "n_even": n_even,
        })

    return units


def _find_max_activating(mlp, X, positions, hidden_dim, n_top=20,
                          batch_size=1024):
    """Step 3: Find max-activating examples for each hidden unit.

    Returns dict mapping unit index to list of (activation, sample_idx) tuples.
    """
    mlp.eval()
    # Extract hidden activations (after ReLU)
    # MLP: layer 0 = Linear, layer 1 = ReLU
    W1 = mlp[0].weight.detach()  # (H, 180)
    b1 = mlp[0].bias.detach()    # (H,)

    # Collect activations for all samples
    n_samples = len(X)
    # Store top-n per unit using a running buffer
    top_acts = {j: [] for j in range(hidden_dim)}  # list of (act_value, sample_idx)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            x = X[i:i + batch_size]  # (B, 180)
            # Hidden activations after ReLU
            h = torch.relu(x @ W1.cpu().T + b1.cpu())  # (B, H)
            for j in range(hidden_dim):
                acts_j = h[:, j].numpy()
                for bi, act_val in enumerate(acts_j):
                    sample_idx = i + bi
                    if len(top_acts[j]) < n_top:
                        top_acts[j].append((float(act_val), sample_idx))
                        if len(top_acts[j]) == n_top:
                            top_acts[j].sort(key=lambda x: -x[0])
                    elif act_val > top_acts[j][-1][0]:
                        top_acts[j][-1] = (float(act_val), sample_idx)
                        top_acts[j].sort(key=lambda x: -x[0])

    return top_acts


def _progressive_ablation(mlp_even, mlp_odd, ev_X, ev_Y, ev_pos, device,
                           hidden_dim, ns_to_test=None):
    """Step 7: Test accuracy using only top-N hidden units by output L2 norm."""
    if ns_to_test is None:
        ns_to_test = [50, 100, 200, 500, 1000, hidden_dim]

    # Get output weight L2 norms for each unit
    W2_even = mlp_even[2].weight.detach().cpu().numpy()  # (192, H)
    W2_odd = mlp_odd[2].weight.detach().cpu().numpy()

    l2_even = np.linalg.norm(W2_even, axis=0)  # (H,)
    l2_odd = np.linalg.norm(W2_odd, axis=0)
    l2_combined = l2_even + l2_odd  # proxy for overall importance

    # Sort by combined L2 (.copy() needed for torch compatibility with negative strides)
    unit_order = np.argsort(l2_combined)[::-1].copy()

    results = {}
    for N in ns_to_test:
        if N > hidden_dim:
            continue
        # Create ablated copies
        mlp_even_abl = nn.Sequential(
            nn.Linear(180, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64 * OPTIONS),
        ).to(device)
        mlp_odd_abl = nn.Sequential(
            nn.Linear(180, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64 * OPTIONS),
        ).to(device)
        mlp_even_abl.load_state_dict(mlp_even.state_dict())
        mlp_odd_abl.load_state_dict(mlp_odd.state_dict())

        # Zero out units not in top N
        mask = torch.zeros(hidden_dim)
        mask[unit_order[:N]] = 1.0
        mask = mask.to(device)

        # Hook to apply mask after ReLU (layer 1)
        def make_hook(m):
            def hook_fn(module, input, output):
                return output * m
            return hook_fn

        h_even = mlp_even_abl[1].register_forward_hook(make_hook(mask))
        h_odd = mlp_odd_abl[1].register_forward_hook(make_hook(mask))

        # Evaluate
        mlp_even_abl.eval()
        mlp_odd_abl.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(ev_X), 1024):
                x = ev_X[i:i + 1024].to(device)
                y = ev_Y[i:i + 1024].to(device)
                pos = ev_pos[i:i + 1024]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask
                preds = torch.zeros_like(y)
                if even_mask.any():
                    preds[even_mask] = mlp_even_abl(x[even_mask]).view(-1, 64, OPTIONS).argmax(-1)
                if odd_mask.any():
                    preds[odd_mask] = mlp_odd_abl(x[odd_mask]).view(-1, 64, OPTIONS).argmax(-1)
                correct += (preds == y).sum().item()
                total += y.numel()

        h_even.remove()
        h_odd.remove()

        acc = correct / total
        results[N] = acc
        print(f"  Top {N:5d} units: {acc:.4%}")

    return results


def experiment_mlp_analysis(args):
    """Part B: Recover heuristics from the trained MLP."""
    device = get_device()
    print(f"Device: {device}")

    hidden_dim = args.mlp_hidden[0] if args.mlp_hidden else 2048
    print(f"Hidden dim: {hidden_dim}")

    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    n_eval = max(int(len(games) * 0.1), 100)
    train_games = games[:len(games) - n_eval]
    eval_games = games[len(games) - n_eval:]
    print(f"Using {len(games)} games ({len(train_games)} train, {len(eval_games)} eval)")

    # Build 180-d features
    print("Building train features (180-d)...")
    tr_X, tr_Y, tr_pos = _build_move_features_batch(
        train_games, POS_START, POS_END, include_pairwise=False)
    print("Building eval features (180-d)...")
    ev_X, ev_Y, ev_pos = _build_move_features_batch(
        eval_games, POS_START, POS_END, include_pairwise=False)
    print(f"  Train: {tr_X.shape}, Eval: {ev_X.shape}")

    # Train MLP
    print(f"\n--- Training MLP H={hidden_dim} ---")
    best_acc, mlp_even, mlp_odd = _train_mlp_nanda_returning_model(
        tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos,
        device, 180, hidden_dim)
    print(f"  Best accuracy: {best_acc:.4%}")

    results = {"accuracy": best_acc, "hidden_dim": hidden_dim}

    # Step 1-2, 4-5: Analyze weight structure
    print("\n--- Analyzing weight structure ---")
    even_units = _analyze_mlp_weights(mlp_even, hidden_dim, "even")
    odd_units = _analyze_mlp_weights(mlp_odd, hidden_dim, "odd")
    all_units = even_units + odd_units

    # Step 5: Category distribution
    categories = {}
    scopes = {}
    primary_classes = {}
    for u in all_units:
        cat = u["category"]
        categories[cat] = categories.get(cat, 0) + 1
        sc = u["output_scope"]
        scopes[sc] = scopes.get(sc, 0) + 1
        pc = u["primary_class"]
        primary_classes[pc] = primary_classes.get(pc, 0) + 1

    results["category_distribution"] = categories
    results["scope_distribution"] = scopes
    results["primary_class_distribution"] = primary_classes

    print("\n  Category distribution:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")
    print("  Output scope distribution:")
    for sc, count in sorted(scopes.items()):
        print(f"    {sc}: {count}")
    print("  Primary class distribution:")
    for pc, count in sorted(primary_classes.items()):
        print(f"    {pc}: {count}")

    # Step 3: Max-activating examples
    print("\n--- Finding max-activating examples ---")
    print("  Even MLP...")
    top_acts_even = _find_max_activating(mlp_even, ev_X, ev_pos, hidden_dim, n_top=20)
    print("  Odd MLP...")
    top_acts_odd = _find_max_activating(mlp_odd, ev_X, ev_pos, hidden_dim, n_top=20)

    # Build game index for eval samples
    # Each game has LENGTH=49 positions, so sample_idx -> (game_idx, move_number)
    def sample_to_game_info(sample_idx):
        game_idx = sample_idx // LENGTH
        move_within = sample_idx % LENGTH
        move_number = POS_START + move_within
        if game_idx < len(eval_games):
            game = eval_games[game_idx]
            moves_so_far = game[:move_number + 1]
            # Board state
            states = seq_to_state_normal(game)
            board = states[move_number]  # (8, 8)
            return {
                "game_idx": int(game_idx),
                "move_number": int(move_number),
                "moves": [int(m) for m in moves_so_far],
                "board": board.tolist(),
            }
        return {"game_idx": int(game_idx), "move_number": int(move_number)}

    # Save max-activating for top 50 most important units (by output L2)
    all_units_sorted = sorted(all_units, key=lambda u: -u["out_l2"])
    top_unit_indices = []
    max_act_data = {}
    for u in all_units_sorted[:50]:
        j = u["unit"]
        parity = u["parity"]
        key = f"{parity}_unit_{j}"
        top_acts = top_acts_even[j] if parity == "even" else top_acts_odd[j]
        examples = []
        for act_val, sample_idx in top_acts[:10]:
            info = sample_to_game_info(sample_idx)
            info["activation"] = act_val
            examples.append(info)
        max_act_data[key] = {
            "description": u["description"],
            "category": u["category"],
            "examples": examples,
        }
        top_unit_indices.append((parity, j))

    # Step 6: Validate individual heuristics
    print("\n--- Validating individual heuristics ---")
    validation = []
    for parity, j in top_unit_indices[:20]:
        mlp = mlp_even if parity == "even" else mlp_odd
        u = [u for u in all_units if u["unit"] == j and u["parity"] == parity][0]

        # Get the top output cell and class for this unit
        top_out = u["top_outputs"][0]
        cell_idx = top_out["cell_idx"]
        cls_idx = top_out["class_idx"]

        # Compute standalone accuracy for this one cell
        W1 = mlp[0].weight.detach().cpu()
        b1 = mlp[0].bias.detach().cpu()
        W2 = mlp[2].weight.detach().cpu()

        # Get hidden unit j's contribution to cell_idx predictions
        # For each eval sample of matching parity:
        pos_mask = (ev_pos % 2 == 0) if parity == "even" else (ev_pos % 2 == 1)
        x_sub = ev_X[pos_mask]
        y_sub = ev_Y[pos_mask]

        # Unit j activation
        h_j = torch.relu(x_sub @ W1[j] + b1[j])  # (N,)

        # Unit j's output for this cell: W2[cell_idx*3:(cell_idx+1)*3, j] * h_j
        w2_cell = W2[cell_idx * OPTIONS:(cell_idx + 1) * OPTIONS, j]  # (3,)
        logits_j = h_j.unsqueeze(1) * w2_cell.unsqueeze(0)  # (N, 3)

        preds_j = logits_j.argmax(-1)
        gt = y_sub[:, cell_idx]
        cell_acc = (preds_j == gt).float().mean().item()

        validation.append({
            "parity": parity,
            "unit": j,
            "description": u["description"],
            "category": u["category"],
            "target_cell": top_out["cell"],
            "target_class": top_out["class"],
            "standalone_cell_accuracy": cell_acc,
            "out_l2": u["out_l2"],
        })
        print(f"  [{parity}] Unit {j}: cell {top_out['cell']} acc={cell_acc:.4%}  "
              f"({u['category']}, L2={u['out_l2']:.3f})")

    results["validation"] = validation

    # Step 7: Progressive ablation
    print("\n--- Progressive ablation ---")
    ns = [50, 100, 200, 500, 1000, hidden_dim]
    ablation_results = _progressive_ablation(
        mlp_even, mlp_odd, ev_X, ev_Y, ev_pos, device, hidden_dim, ns)
    results["ablation"] = {str(k): v for k, v in ablation_results.items()}

    # Save everything
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save unit analysis JSON
    unit_data = {
        "even_units": even_units,
        "odd_units": odd_units,
    }
    with open(os.path.join(out_dir, "mlp_unit_analysis.json"), "w") as f:
        json.dump(unit_data, f, indent=2)
    print(f"  Saved unit analysis to mlp_unit_analysis.json")

    # Save max-activating examples
    with open(os.path.join(out_dir, "mlp_max_activating.json"), "w") as f:
        json.dump(max_act_data, f, indent=2)
    print(f"  Saved max-activating examples to mlp_max_activating.json")

    # Save descriptions text file
    desc_path = os.path.join(out_dir, "mlp_unit_descriptions.txt")
    with open(desc_path, "w") as f:
        f.write(f"MLP Unit Descriptions (H={hidden_dim})\n")
        f.write(f"Best accuracy: {best_acc:.4%}\n")
        f.write(f"{'='*80}\n\n")
        for u in all_units_sorted:
            f.write(f"{u['description']}\n")
            f.write(f"  Category: {u['category']}, Scope: {u['output_scope']}, "
                    f"Primary class: {u['primary_class']}, L2: {u['out_l2']:.4f}\n")
            inp_strs = [inp['feature'] + '({:.4f})'.format(inp['weight']) for inp in u['top_inputs'][:5]]
            f.write("  Top inputs: {}\n".format(', '.join(inp_strs)))
            out_strs = [out['cell'] + '=' + out['class'] + '({:.4f})'.format(out['weight']) for out in u['top_outputs'][:3]]
            f.write("  Top outputs: {}\n\n".format(', '.join(out_strs)))
    print(f"  Saved descriptions to mlp_unit_descriptions.txt")

    # Summary
    print(f"\n{'='*60}")
    print("MLP ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy: {best_acc:.4%}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Categories: {categories}")
    print(f"  Ablation:")
    for n, acc in sorted(ablation_results.items()):
        print(f"    Top {n}: {acc:.4%}")

    _save_results(args, "mlp_analysis", results)
    return results


# ============================= Utilities =====================================

def _save_results(args, exp_name, results):
    """Save results to JSON."""
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{exp_name}_results.json")

    # Convert numpy/tensor types
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        return obj

    with open(out_path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"Saved to {out_path}")


def experiment_probe_directions(args):
    """Compare probe directions and subspaces across board state variants.

    Trains 4 probes per layer (Othello, Othello2, No Flip, Random Perm) at
    layers 0-8. Computes pairwise cosine similarity, subspace overlap,
    and cross-layer direction stability. Saves all probe checkpoints.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = get_device()
    print(f"Device: {device}")

    model, block_size = load_model(args.ckpt_path, device)
    print(f"Loaded Othello-GPT from {args.ckpt_path}")

    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    n_eval = max(int(len(games) * 0.1), 100)
    n_train = len(games) - n_eval
    train_games = games[:n_train]
    eval_games = games[n_train:]
    print(f"Using {len(games)} games ({n_train} train, {n_eval} eval)")

    # Output directory
    out_dir = os.path.join(args.output_dir, "probe_directions")
    os.makedirs(out_dir, exist_ok=True)
    probe_dir = os.path.join(out_dir, "probe_checkpoints")
    os.makedirs(probe_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Probe names: othello, othello2 (independent seed), no_flip, shuffled
    # "shuffled" = real Othello labels with sample order randomized (breaks
    # the activation-label correspondence while preserving label statistics)
    probe_names = ["othello", "othello2", "no_flip", "shuffled"]
    simulators = {
        "othello": seq_to_state_normal,
        "othello2": seq_to_state_normal,
        "no_flip": seq_to_state_no_flip,
        "shuffled": seq_to_state_normal,  # labels will be shuffled below
    }

    # Helpers
    def get_directions(probe_even):
        """Extract black-white direction for each cell from even probe."""
        w = probe_even.weight.data.cpu()  # (64*3, d_model)
        w = w.view(64, OPTIONS, -1)  # (64, 3, d_model)
        return w[:, 2, :] - w[:, 1, :]  # (64, d_model)

    def cosine_sim_per_cell(d1, d2):
        """Absolute cosine similarity for each of 64 cells."""
        d1_n = d1 / (d1.norm(dim=1, keepdim=True) + 1e-8)
        d2_n = d2 / (d2.norm(dim=1, keepdim=True) + 1e-8)
        return (d1_n * d2_n).sum(dim=1).abs()  # (64,)

    def _probe_svd(probe_even):
        """SVD of probe weight matrix. Returns singular values and right vectors."""
        w = probe_even.weight.data.cpu().float()  # (192, d_model)
        _, s, vt = torch.linalg.svd(w, full_matrices=False)
        return s, vt  # s: (192,), vt: (192, d_model)

    def _effective_rank(s, threshold=0.9):
        """Number of singular values capturing `threshold` fraction of variance."""
        cumvar = (s ** 2).cumsum(0) / (s ** 2).sum()
        return int((cumvar < threshold).sum().item()) + 1

    def subspace_overlap(probe1_even, probe2_even, k=None):
        """Compute subspace overlap between two probes.

        Returns dict with:
          unweighted: ||V1_k^T V2_k||_F^2 / k  (treats all top-k dims equally)
          weighted:   sum_i sum_j w1_i * w2_j * (v1_i . v2_j)^2 / (sum w1 * sum w2)
                      where w_i = s_i^2 / sum(s^2) (variance fraction)
          k1, k2:     effective rank of each probe (90% variance)
          k_used:     max(k1, k2), used for unweighted overlap
        """
        s1, v1t = _probe_svd(probe1_even)
        s2, v2t = _probe_svd(probe2_even)
        k1 = _effective_rank(s1)
        k2 = _effective_rank(s2)
        if k is None:
            k = max(k1, k2)
        v1 = v1t[:k]  # (k, d_model)
        v2 = v2t[:k]  # (k, d_model)
        gram = v1 @ v2.T  # (k, k)
        unweighted = (gram ** 2).sum().item() / k

        # Variance-weighted overlap: for each of probe1's top-k directions,
        # measure how much of it lies in probe2's top-k subspace, weighted
        # by probe1's singular value. Then normalize.
        # projection_i = ||V2_k^T @ v1_i||^2 (fraction of v1_i in V2's subspace)
        # weighted = sum_i (s1_i^2 / sum(s1^2)) * projection_i
        var1 = s1[:k] ** 2
        w1 = var1 / var1.sum()  # (k,)
        # gram[i,j] = v1_i . v2_j, so projection_i = sum_j gram[i,j]^2
        projections = (gram ** 2).sum(dim=1)  # (k,)
        weighted = (w1 * projections).sum().item()

        return {
            "unweighted": unweighted,
            "weighted": weighted,
            "k1": k1,
            "k2": k2,
            "k_used": k,
        }

    # Pre-compute labels for each simulator (same across layers)
    print("\nPre-computing board state labels...")
    n_tr_games = len(train_games)
    n_ev_games = len(eval_games)
    tr_pos = torch.arange(POS_START, POS_END).repeat(n_tr_games)
    ev_pos = torch.arange(POS_START, POS_END).repeat(n_ev_games)

    all_tr_labels = {}
    all_ev_labels = {}
    # othello and othello2 share the same labels
    for name in ["othello", "no_flip"]:
        sim = simulators[name]
        tr_states = get_board_states(train_games, sim, POS_START, POS_END)
        all_tr_labels[name] = torch.tensor(
            states_to_labels(tr_states).reshape(n_tr_games * LENGTH, 64),
            dtype=torch.long)
        ev_states = get_board_states(eval_games, sim, POS_START, POS_END)
        all_ev_labels[name] = torch.tensor(
            states_to_labels(ev_states).reshape(n_ev_games * LENGTH, 64),
            dtype=torch.long)
        print(f"  {name}: done")
    all_tr_labels["othello2"] = all_tr_labels["othello"]
    all_ev_labels["othello2"] = all_ev_labels["othello"]

    # Shuffled: randomly permute the sample order of Othello labels
    # This breaks the activation-label correspondence while preserving
    # the marginal distribution of labels
    shuffle_rng = np.random.RandomState(12345)
    tr_shuffle_idx = shuffle_rng.permutation(len(all_tr_labels["othello"]))
    ev_shuffle_idx = shuffle_rng.permutation(len(all_ev_labels["othello"]))
    all_tr_labels["shuffled"] = all_tr_labels["othello"][tr_shuffle_idx]
    all_ev_labels["shuffled"] = all_ev_labels["othello"][ev_shuffle_idx]
    print(f"  shuffled: done (permuted {len(tr_shuffle_idx)} train, {len(ev_shuffle_idx)} eval samples)")

    # Pairwise comparisons
    pairs = [
        ("othello", "othello2"),
        ("othello", "no_flip"),
        ("othello", "shuffled"),
        ("othello2", "no_flip"),
        ("othello2", "shuffled"),
        ("no_flip", "shuffled"),
    ]
    pair_names = [f"{a}_vs_{b}" for a, b in pairs]

    # Storage across layers
    results_table = []
    dirs_by_layer = {name: {} for name in probe_names}  # name -> {layer -> (64, d)}
    pairwise_cossim = {pn: {} for pn in pair_names}  # pair -> {layer -> (64,)}
    pairwise_subspace = {pn: {} for pn in pair_names}  # pair -> {layer -> dict}
    probe_eff_rank = {name: {} for name in probe_names}  # name -> {layer -> k}

    for layer in range(9):
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        # Collect activations
        print("  Collecting activations...")
        tr_acts, _ = collect_activations_and_labels(
            model, train_games, device, layer, block_size,
            simulator=seq_to_state_normal)
        ev_acts, _ = collect_activations_and_labels(
            model, eval_games, device, layer, block_size,
            simulator=seq_to_state_normal)

        # Train all 4 probes
        probes = {}
        accs = {}
        for name in probe_names:
            # Use different random seed for othello2
            if name == "othello2":
                torch.manual_seed(99999)
                np.random.seed(99999)
            print(f"\n  --- {name} probe ---")
            acc, pe, po = _train_nanda_probe_returning_probes(
                tr_acts, all_tr_labels[name], tr_pos,
                ev_acts, all_ev_labels[name], ev_pos,
                device, input_dim=tr_acts.shape[1])
            # Restore default seed
            if name == "othello2":
                torch.manual_seed(42)
                np.random.seed(42)
            accs[name] = acc
            probes[name] = {"even": pe, "odd": po}

            # Save probe checkpoint
            ckpt_path = os.path.join(probe_dir, f"{name}_layer{layer}.pt")
            torch.save({
                "even": pe.state_dict(),
                "odd": po.state_dict(),
                "accuracy": acc,
            }, ckpt_path)

        # Extract directions
        for name in probe_names:
            dirs_by_layer[name][layer] = get_directions(probes[name]["even"])

        # Compute effective rank for each probe
        for name in probe_names:
            s, _ = _probe_svd(probes[name]["even"])
            probe_eff_rank[name][layer] = _effective_rank(s)

        # Pairwise cosine similarity and subspace overlap
        for (a, b), pn in zip(pairs, pair_names):
            cs = cosine_sim_per_cell(dirs_by_layer[a][layer], dirs_by_layer[b][layer])
            pairwise_cossim[pn][layer] = cs.numpy()
            pairwise_subspace[pn][layer] = subspace_overlap(
                probes[a]["even"], probes[b]["even"])

        # Print layer summary
        print(f"\n  Layer {layer} Summary:")
        print(f"  {'Probe':<15} {'Accuracy':>10} {'Eff. Rank':>10}")
        for name in probe_names:
            print(f"  {name:<15} {accs[name]:>10.4%} {probe_eff_rank[name][layer]:>10}")
        print(f"\n  {'Pair':<30} {'Mean |cos|':>10} {'Unwt Sub':>10} {'Wt Sub':>10} {'k':>5}")
        for (a, b), pn in zip(pairs, pair_names):
            mc = pairwise_cossim[pn][layer].mean()
            sd = pairwise_subspace[pn][layer]
            print(f"  {pn:<30} {mc:>10.4f} {sd['unweighted']:>10.4f} {sd['weighted']:>10.4f} {sd['k_used']:>5}")

        # Store for table
        row = {"layer": layer}
        for name in probe_names:
            row[f"{name}_acc"] = accs[name]
            row[f"{name}_eff_rank"] = probe_eff_rank[name][layer]
        for pn in pair_names:
            row[f"cossim_{pn}"] = float(pairwise_cossim[pn][layer].mean())
            row[f"subspace_unweighted_{pn}"] = pairwise_subspace[pn][layer]["unweighted"]
            row[f"subspace_weighted_{pn}"] = pairwise_subspace[pn][layer]["weighted"]
            row[f"subspace_k_{pn}"] = pairwise_subspace[pn][layer]["k_used"]
        results_table.append(row)

    # ==================== Summary Tables ====================
    print(f"\n{'='*80}")
    print("PROBE ACCURACY BY LAYER")
    print(f"{'='*80}")
    print(f"{'Layer':>5}", end="")
    for name in probe_names:
        print(f"  {name:>12}", end="")
    print()
    for row in results_table:
        print(f"{row['layer']:>5}", end="")
        for name in probe_names:
            print(f"  {row[f'{name}_acc']:>12.4%}", end="")
        print()

    print(f"\n{'='*80}")
    print("PAIRWISE MEAN |COSINE SIMILARITY| BY LAYER")
    print(f"{'='*80}")
    print(f"{'Layer':>5}", end="")
    for pn in pair_names:
        print(f"  {pn:>14}", end="")
    print()
    for row in results_table:
        print(f"{row['layer']:>5}", end="")
        for pn in pair_names:
            print(f"  {row[f'cossim_{pn}']:>14.4f}", end="")
        print()

    print(f"\n{'='*80}")
    print("PROBE EFFECTIVE RANK (90% variance) BY LAYER")
    print(f"{'='*80}")
    print(f"{'Layer':>5}", end="")
    for name in probe_names:
        print(f"  {name:>12}", end="")
    print()
    for row in results_table:
        print(f"{row['layer']:>5}", end="")
        for name in probe_names:
            print(f"  {row[f'{name}_eff_rank']:>12}", end="")
        print()

    print(f"\n{'='*80}")
    print("PAIRWISE SUBSPACE OVERLAP BY LAYER (unweighted / weighted)")
    print(f"{'='*80}")
    for pn in pair_names:
        print(f"\n  {pn}:")
        print(f"  {'Layer':>5} {'Unweighted':>12} {'Weighted':>12} {'k':>5}")
        for row in results_table:
            print(f"  {row['layer']:>5} {row[f'subspace_unweighted_{pn}']:>12.4f} "
                  f"{row[f'subspace_weighted_{pn}']:>12.4f} {row[f'subspace_k_{pn}']:>5}")

    # ==================== Figures ====================

    # Fig 1: Pairwise cosine similarity heatmaps (8x8) per layer, one row per pair
    fig1, axes1 = plt.subplots(len(pairs), 9, figsize=(27, len(pairs) * 3))
    for pi, (pn, (a, b)) in enumerate(zip(pair_names, pairs)):
        for layer in range(9):
            ax = axes1[pi, layer]
            cs = pairwise_cossim[pn][layer].reshape(8, 8)
            im = ax.imshow(cs, vmin=0, vmax=1, cmap="RdYlBu_r")
            ax.set_xticks(range(8))
            ax.set_xticklabels([str(i+1) for i in range(8)], fontsize=6)
            ax.set_yticks(range(8))
            ax.set_yticklabels([chr(65+i) for i in range(8)], fontsize=6)
            if pi == 0:
                ax.set_title(f"L{layer}", fontsize=9)
            if layer == 0:
                ax.set_ylabel(f"{a}\nvs {b}", fontsize=8)
    fig1.colorbar(im, ax=axes1, shrink=0.6, label="|cosine similarity|")
    fig1.suptitle("Per-Cell |Cosine Similarity| Between Probe Directions", fontsize=14)
    fig1.savefig(os.path.join(fig_dir, "pairwise_cossim_heatmaps.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved pairwise cosine similarity heatmaps")

    # Fig 2: Summary line plots — cosine sim, unweighted subspace, weighted subspace
    fig2, (ax_cos, ax_sub_u, ax_sub_w) = plt.subplots(1, 3, figsize=(21, 6))
    for pn in pair_names:
        cos_vals = [float(pairwise_cossim[pn][l].mean()) for l in range(9)]
        sub_u_vals = [pairwise_subspace[pn][l]["unweighted"] for l in range(9)]
        sub_w_vals = [pairwise_subspace[pn][l]["weighted"] for l in range(9)]
        ax_cos.plot(range(9), cos_vals, "o-", label=pn, markersize=5)
        ax_sub_u.plot(range(9), sub_u_vals, "o-", label=pn, markersize=5)
        ax_sub_w.plot(range(9), sub_w_vals, "o-", label=pn, markersize=5)
    for ax, title, ylabel in [
        (ax_cos, "Direction Similarity", "Mean |Cosine Similarity|"),
        (ax_sub_u, "Subspace Overlap (unweighted)", "Overlap"),
        (ax_sub_w, "Subspace Overlap (variance-weighted)", "Overlap"),
    ]:
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.set_xticks(range(9))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(os.path.join(fig_dir, "pairwise_summary_lines.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved pairwise summary line plots")

    # Fig 3: Cross-layer direction similarity for each probe type
    n_layers = 9

    def compute_cross_layer_mat(dirs_dict):
        mat = np.zeros((n_layers, n_layers))
        for li in range(n_layers):
            for lj in range(n_layers):
                cs = cosine_sim_per_cell(dirs_dict[li], dirs_dict[lj])
                mat[li, lj] = cs.mean().item()
        return mat

    cross_layer_mats = {}
    for name in probe_names:
        cross_layer_mats[name] = compute_cross_layer_mat(dirs_by_layer[name])

    fig3, axes3 = plt.subplots(1, 4, figsize=(24, 6))
    for ax, name in zip(axes3, probe_names):
        mat = cross_layer_mats[name]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlBu_r")
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_title(f"{name}")
        for i in range(n_layers):
            for j in range(n_layers):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if mat[i, j] > 0.5 else "black")
    fig3.colorbar(im, ax=axes3, shrink=0.8, label="|cosine similarity|")
    fig3.suptitle("Cross-Layer Probe Direction Stability (mean |cos| over 64 cells)", fontsize=14)
    plt.tight_layout()
    fig3.savefig(os.path.join(fig_dir, "cross_layer_direction_stability.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Saved cross-layer direction stability heatmaps")

    # Fig 4: Cross-layer subspace overlap for each probe type
    def compute_cross_layer_subspace(name):
        """Load saved probes and compute pairwise subspace overlap across layers."""
        mat_u = np.zeros((n_layers, n_layers))
        mat_w = np.zeros((n_layers, n_layers))
        probes_loaded = {}
        for l in range(n_layers):
            ckpt = torch.load(
                os.path.join(probe_dir, f"{name}_layer{l}.pt"),
                map_location="cpu")
            pe = nn.Linear(512, 64 * OPTIONS)
            pe.load_state_dict(ckpt["even"])
            probes_loaded[l] = pe
        for li in range(n_layers):
            for lj in range(n_layers):
                result = subspace_overlap(probes_loaded[li], probes_loaded[lj])
                mat_u[li, lj] = result["unweighted"]
                mat_w[li, lj] = result["weighted"]
        return mat_u, mat_w

    cross_layer_subspace_u = {}
    cross_layer_subspace_w = {}
    for name in probe_names:
        print(f"  Computing cross-layer subspace overlap for {name}...")
        mat_u, mat_w = compute_cross_layer_subspace(name)
        cross_layer_subspace_u[name] = mat_u
        cross_layer_subspace_w[name] = mat_w

    # Unweighted cross-layer subspace
    fig4, axes4 = plt.subplots(1, 4, figsize=(24, 6))
    for ax, name in zip(axes4, probe_names):
        mat = cross_layer_subspace_u[name]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlBu_r")
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_title(f"{name}")
        for i in range(n_layers):
            for j in range(n_layers):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if mat[i, j] > 0.5 else "black")
    fig4.colorbar(im, ax=axes4, shrink=0.8, label="subspace overlap")
    fig4.suptitle("Cross-Layer Subspace Overlap (unweighted)", fontsize=14)
    plt.tight_layout()
    fig4.savefig(os.path.join(fig_dir, "cross_layer_subspace_unweighted.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # Weighted cross-layer subspace
    fig5, axes5 = plt.subplots(1, 4, figsize=(24, 6))
    for ax, name in zip(axes5, probe_names):
        mat = cross_layer_subspace_w[name]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlBu_r")
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_title(f"{name}")
        for i in range(n_layers):
            for j in range(n_layers):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if mat[i, j] > 0.5 else "black")
    fig5.colorbar(im, ax=axes5, shrink=0.8, label="subspace overlap (weighted)")
    fig5.suptitle("Cross-Layer Subspace Overlap (variance-weighted)", fontsize=14)
    plt.tight_layout()
    fig5.savefig(os.path.join(fig_dir, "cross_layer_subspace_weighted.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print("Saved cross-layer subspace overlap heatmaps (unweighted + weighted)")

    # ==================== Save Results JSON ====================
    out_path = os.path.join(out_dir, "probe_directions_results.json")
    json_results = {
        "config": {
            "n_games": len(games),
            "n_train": n_train,
            "n_eval": n_eval,
            "shuffle_seed": 12345,
        },
        "table": results_table,
        "pairwise_cossim_per_cell": {
            pn: {str(l): pairwise_cossim[pn][l].tolist() for l in range(9)}
            for pn in pair_names
        },
        "pairwise_subspace": {
            pn: {str(l): pairwise_subspace[pn][l] for l in range(9)}
            for pn in pair_names
        },
        "cross_layer_direction": {
            name: cross_layer_mats[name].tolist() for name in probe_names
        },
        "cross_layer_subspace_unweighted": {
            name: cross_layer_subspace_u[name].tolist() for name in probe_names
        },
        "cross_layer_subspace_weighted": {
            name: cross_layer_subspace_w[name].tolist() for name in probe_names
        },
        "probe_effective_rank": {
            name: {str(l): probe_eff_rank[name][l] for l in range(9)}
            for name in probe_names
        },
    }
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n{'='*80}")
    print("ALL RESULTS SAVED")
    print(f"{'='*80}")
    print(f"  Results JSON: {out_path}")
    print(f"  Probe checkpoints: {probe_dir}/")
    print(f"  Figures: {fig_dir}/")
    print(f"  Probes saved: {len(probe_names) * 9} total ({len(probe_names)} types x 9 layers)")


def parse_args():
    p = argparse.ArgumentParser(description="Heuristic & baseline probing experiments")
    p.add_argument("--experiment", type=str, required=True,
                   choices=["standard_probe", "resid_pre", "alt_boards",
                            "by_move", "heuristic", "brute_force",
                            "learned_heuristic", "mlp", "precompute",
                            "combine_heuristic", "mlp_analysis",
                            "probe_directions", "mlp_dropout"],
                   help="Which experiment to run")
    p.add_argument("--ckpt-path", type=str, default="ckpts/gpt_synthetic.ckpt")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--max-games", type=int, default=10000)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--output-dir", type=str,
                   default=os.path.join(SCRIPT_DIR, "heuristic_probe_results"))
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=None,
                   help="Hidden dims for MLP experiment (default: 256 512 1024)")
    p.add_argument("--mlp-only", action="store_true",
                   help="For mlp experiment: skip linear baseline and pairwise MLPs")
    p.add_argument("--mlp-layers", type=int, default=1,
                   help="Number of hidden layers in MLP (default: 1)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Training epochs (default: 16 for mlp, 10 for others)")
    p.add_argument("--precomputed", action="store_true",
                   help="Load precomputed features from feature_chunks/ (skip feature building)")
    p.add_argument("--chunk-id", type=int, default=None,
                   help="Chunk ID for parallel precompute (used with --file-start/--file-end)")
    p.add_argument("--file-start", type=int, default=0,
                   help="First file index for chunked precompute")
    p.add_argument("--file-end", type=int, default=None,
                   help="Last file index (exclusive) for chunked precompute")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    experiments = {
        "standard_probe": experiment_standard_probe,
        "resid_pre": experiment_resid_pre,
        "alt_boards": experiment_alt_boards,
        "by_move": experiment_by_move,
        "heuristic": experiment_heuristic,
        "brute_force": experiment_brute_force,
        "learned_heuristic": experiment_learned_heuristic,
        "mlp": experiment_mlp,
        "precompute": experiment_precompute,
        "combine_heuristic": experiment_combine_heuristic,
        "mlp_analysis": experiment_mlp_analysis,
        "probe_directions": experiment_probe_directions,
        "mlp_dropout": experiment_mlp_dropout,
    }

    experiments[args.experiment](args)


if __name__ == "__main__":
    main()
