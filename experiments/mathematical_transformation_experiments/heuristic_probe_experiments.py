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
    PAD_IDX, ROWS, COLS, OPTIONS,
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


def _build_move_features_batch(games, pos_start, pos_end, include_pairwise=True):
    """Build move-history features and labels for a list of games.

    Returns (features, labels, positions) tensors.
    """
    features = []
    labels = []
    positions = []
    for game in tqdm(games, desc="  building features", leave=False):
        states = seq_to_state_normal(game)
        for t in range(pos_start, pos_end):
            features.append(_build_move_history_features(
                game, t, include_pairwise=include_pairwise))
            lbl = states_to_labels(states[t:t+1].reshape(1, 8, 8))
            labels.append(lbl.reshape(64))
            positions.append(t)
    return (torch.tensor(np.stack(features), dtype=torch.float32),
            torch.tensor(np.stack(labels), dtype=torch.long),
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

def _train_mlp_nanda(tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos,
                     device, input_dim, hidden_dim,
                     lr=1e-3, epochs=16, batch_size=1024):
    """Train a 2-layer MLP with Nanda's even/odd split.

    Even MLP: Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 64*3)
    Odd MLP:  same architecture, separate weights
    """
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
        best_acc = max(best_acc, acc)
        scheduler.step(np.mean(losses))
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: acc={acc:.4%}  loss={np.mean(losses):.5f}  lr={cur_lr:.2e}",
              flush=True)

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


def experiment_mlp(args):
    """Approach 2: MLP on move-history features (180-d and 3780-d)."""
    device = get_device()
    print(f"Device: {device}")

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
    for H in hidden_dims:
        print(f"\n--- MLP 180-d hidden={H} with even/odd split ---")
        acc_mlp = _train_mlp_nanda(
            tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos,
            device, 180, H)
        results[f"mlp_180_H{H}"] = acc_mlp

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
    return results


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


def parse_args():
    p = argparse.ArgumentParser(description="Heuristic & baseline probing experiments")
    p.add_argument("--experiment", type=str, required=True,
                   choices=["standard_probe", "resid_pre", "alt_boards",
                            "by_move", "heuristic", "brute_force",
                            "learned_heuristic", "mlp",
                            "combine_heuristic", "mlp_analysis"],
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
        "combine_heuristic": experiment_combine_heuristic,
        "mlp_analysis": experiment_mlp_analysis,
    }

    experiments[args.experiment](args)


if __name__ == "__main__":
    main()
