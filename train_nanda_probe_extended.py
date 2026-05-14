"""Re-train Nanda-style linear probe on OthelloGPT, but covering the
FULL position range (5-58) instead of Nanda's original (5-53).

Faithful to mechanistic_interpretability/tl_probing_v1.py but uses the
mingpt forward pass (so it works with our gpt_nanda_synthetic.ckpt) and
extends pos_end from 54 to 59 to include the very-late game positions.

Probe tensor shape: (3, 512, 8, 8, 3)
  modes: 0 = even-parity, 1 = odd-parity, 2 = all positions
  classes: 0 = empty, 1 = white (state==-1), 2 = black (state==+1)

Output: main_linear_probe_ext_5_58.pth

Usage:
    python train_nanda_probe_extended.py --epochs 2 --n-games 100000
"""
import sys, os, argparse, time
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F

from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, load_games, extract_activations, VOCAB_SIZE, GAME_LEN,
)


def state_stack_to_one_hot(state_stack, modes=3):
    """state_stack: (G, T, 8, 8) int in {-1, 0, 1}.
    Returns (modes, G, T, 8, 8, 3) one-hot in {0, 1}."""
    G, T, R, C = state_stack.shape
    oh = torch.zeros(modes, G, T, R, C, 3, device=state_stack.device,
                     dtype=torch.float32)
    oh[:, ..., 0] = (state_stack == 0).float()
    oh[:, ..., 1] = (state_stack == -1).float()
    oh[:, ..., 2] = (state_stack == 1).float()
    return oh


def compute_states(games):
    """Replay games through OthelloBoardState. Returns (G, GAME_LEN, 8, 8) int8."""
    G = len(games)
    states = np.zeros((G, GAME_LEN, 8, 8), dtype=np.int8)
    for gi, g in enumerate(games):
        board = OthelloBoardState()
        for t, m in enumerate(g):
            try: board.umpire(m)
            except Exception: break
            states[gi, t] = np.asarray(board.state, dtype=np.int8)
    return states


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ckpts/gpt_nanda_synthetic.ckpt")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--n-games", type=int, default=100000)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--pos-start", type=int, default=5)
    parser.add_argument("--pos-end", type=int, default=59,
                        help="Exclusive upper bound. OGPT block_size=59, "
                             "so max useful pos_end is 59.")
    parser.add_argument("--output",
                        default="mechanistic_interpretability/main_linear_probe_ext_5_58.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = GPTConfig(vocab_size=VOCAB_SIZE, block_size=59,
                       n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    state = torch.load(args.ckpt, map_location="cpu")
    if 'model_state_dict' in state: state = state['model_state_dict']
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded OGPT (block_size=59), probing layer {args.layer}")

    games = load_games(max_files=args.max_files)
    games = [g for g in games if len(g) == GAME_LEN]
    if len(games) > args.n_games:
        games = games[:args.n_games]
    print(f"Using {len(games)} games")

    # Tokenize once
    tokens = tokenize_games(games).to(device)   # (G, GAME_LEN)
    # Compute board states once (CPU)
    print("Replaying games for ground truth...")
    t0 = time.time()
    states_np = compute_states(games)   # (G, GAME_LEN, 8, 8) int8
    print(f"  done in {time.time()-t0:.0f}s")

    states = torch.from_numpy(states_np)   # CPU
    # Slice to probe training range
    pos_end = min(args.pos_end, tokens.shape[1])
    pos_start = args.pos_start
    print(f"Training on positions [{pos_start}, {pos_end})  =  "
          f"{pos_end-pos_start} turns")

    # Initialize probe tensor (3, 512, 8, 8, 3)
    modes = 3
    d_model = 512
    probe = (torch.randn(modes, d_model, 8, 8, 3, device=device)
             / np.sqrt(d_model))
    probe.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [probe], lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)

    G = len(games)
    n_train = G
    batch = args.batch_size

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = torch.randperm(n_train)
        total_loss = 0.0; n_batches = 0
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            tok_b = tokens[idx]   # (B, GAME_LEN)
            state_b = states[idx, pos_start:pos_end].to(device)  # (B, T, 8, 8)

            with torch.no_grad():
                # extract_activations returns (B, T_full, 512)
                h = extract_activations(model, tok_b, args.layer)
            h_sliced = h[:, pos_start:pos_end, :]   # (B, T, 512)

            # Build one-hot target: (modes, B, T, 8, 8, 3)
            oh = state_stack_to_one_hot(state_b, modes=modes)

            # probe_out: (modes, B, T, 8, 8, 3)
            #   logits[m, b, t, r, c, o] = sum_d h[b, t, d] * probe[m, d, r, c, o]
            logits = torch.einsum('btd,mdrco->mbtrco', h_sliced, probe)
            log_probs = F.log_softmax(logits, dim=-1)
            # per-mode loss = -mean over (batch, options) of log_p[correct]
            #   * options to undo the mean over the 3-class dimension
            correct_log_probs = (log_probs * oh).mean(dim=(1, 5)) * 3  # (modes, T, 8, 8)
            # Mode 0: only even-parity slice positions
            T_actual = correct_log_probs.shape[1]
            even_mask = torch.arange(T_actual, device=device) % 2 == 0
            odd_mask  = ~even_mask
            loss_even = -correct_log_probs[0, even_mask].mean(0).sum()
            loss_odd  = -correct_log_probs[1, odd_mask].mean(0).sum()
            loss_all  = -correct_log_probs[2, :].mean(0).sum()
            loss = loss_even + loss_odd + loss_all

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()); n_batches += 1

        dt = time.time() - t0
        print(f"Epoch {epoch}: avg_loss={total_loss/max(n_batches,1):.4f}  "
              f"time={dt:.0f}s", flush=True)

    # Save in Nanda's tensor format
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(probe.detach().cpu(), args.output)
    print(f"\nSaved probe to {args.output}")
    print(f"Shape: {tuple(probe.shape)}")
