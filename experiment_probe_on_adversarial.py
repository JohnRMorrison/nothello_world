"""Compare Nanda's probe accuracy on adversarial vs. turn-matched control positions.

Logic: if OGPT's world model is intact, probe accuracy on adversarial
positions should be no worse than on non-adversarial ones matched at
the same turn.  A gap indicates the world model itself is broken there;
no gap indicates the failure is downstream (heuristics/shortcuts in the
prediction head that override the internal state).

Turn matching: for each adversarial (game, turn), we bootstrap a control
sample of non-adversarial positions at the same turn — this avoids the
confound "adversarial positions cluster at late turns where probe acc is
naturally lower."

Usage:
    python experiment_probe_on_adversarial.py \\
        --adversarial-dir experiment1_data \\
        --ckpt ckpts/gpt_nanda_synthetic.ckpt \\
        --probe mechanistic_interpretability/main_linear_probe.pth \\
        --n-control-games 5000
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, load_games, extract_activations, VOCAB_SIZE, GAME_LEN,
)


def state_to_gt(state_8x8):
    """OthelloBoardState.state -> Nanda probe GT: 0=empty, 1=white, 2=black."""
    gt = np.zeros((8, 8), dtype=np.int64)
    gt[state_8x8 == -1] = 1
    gt[state_8x8 == 1]  = 2
    return gt


def find_adversarial_positions(model, dataset, device, games, block_size,
                                 max_positions=None):
    """For each game, find positions where OGPT's top-1 is illegal.

    Returns list of (game_tuple, turn, top1_board_pos) tuples.
    """
    stoi = dataset.stoi
    positions = []
    with torch.no_grad():
        for g_idx, game in enumerate(games):
            board = OthelloBoardState()
            L = min(len(game), block_size)
            # Batch: score every position from turn 1..L-1
            tokens = torch.tensor(
                [stoi[s] for s in game[:L]], dtype=torch.long, device=device
            ).unsqueeze(0)  # (1, L)
            logits, _ = model(tokens)                   # (1, L, V)
            top1 = logits.argmax(dim=-1)[0].cpu().numpy()  # (L,)
            for t in range(L):
                valid = board.get_valid_moves()
                if valid:
                    # OGPT predicts move at position t+1 given tokens[:t+1]
                    top1_token = int(top1[t])
                    top1_board_pos = None
                    for c, i in stoi.items():
                        if i == top1_token and isinstance(c, int):
                            top1_board_pos = c
                            break
                    if top1_board_pos is not None and top1_board_pos not in valid:
                        positions.append((tuple(game[:L]), t, top1_board_pos))
                        if max_positions and len(positions) >= max_positions:
                            return positions
                if t < len(game):
                    board.umpire(game[t])
    return positions


def get_hidden_and_state(model, dataset, device, game_tuple, turn, layer,
                          block_size):
    """Return (hidden_512, true_state_8x8) at the position after 'turn' moves."""
    L = min(len(game_tuple), block_size)
    game = list(game_tuple)[:L]
    tokens = torch.tensor(
        [dataset.stoi[s] for s in game], dtype=torch.long, device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        h = extract_activations(model, tokens, layer)   # (1, L, 512)
    hidden = h[0, turn, :].cpu().numpy()               # (512,)

    # Ground truth state after `turn` moves
    board = OthelloBoardState()
    for m in game[:turn + 1]:
        board.umpire(m)
    return hidden, np.asarray(board.state, dtype=np.int8)


def probe_predict(hidden, turn, probe):
    """Run Nanda's probe (mode selected by parity) on hidden state.

    Returns (8, 8) prediction in {0, 1, 2}.
    """
    # position parity: mode 0 for odd positions, mode 1 for even (see
    # analyze_nanda_probe_per_cell.py)
    mode = 0 if turn % 2 == 1 else 1
    W = probe[mode]                        # (512, 8, 8, 3)
    h = torch.from_numpy(hidden).float()
    logits = torch.einsum('d,drco->rco', h, W)
    return logits.argmax(dim=-1).numpy()   # (8, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adversarial-dir', default='experiment1_data',
                    help='Directory of experiment1 output .npz files.')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe',
                    default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--n-control-games', type=int, default=5000)
    ap.add_argument('--max-files', type=int, default=2)
    ap.add_argument('--max-adversarial', type=int, default=2000)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load probe
    probe = torch.load(args.probe, map_location='cpu')
    print(f"Probe shape: {tuple(probe.shape)}")
    assert probe.shape == (3, 512, 8, 8, 3)

    # Load OGPT
    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    print(f"Loaded OGPT (block_size={block_size}), probing layer {args.layer}")

    # ---- Build the vocab/dataset shim (analogous to eval_legal_move.py) ----
    print(f"Loading val games (up to {args.max_files} files)...")
    val_games = load_games(max_files=args.max_files)
    print(f"  {len(val_games)} val games loaded")
    stoi = {}; itos = {}
    for game in val_games:
        for m in game:
            if m not in stoi:
                i = len(stoi)
                stoi[m] = i
                itos[i] = m
    class _Shim: pass
    dataset = _Shim(); dataset.stoi = stoi; dataset.itos = itos
    dataset.block_size = block_size

    # ---- Collect adversarial games from experiment1 output ----
    print(f"\nCollecting adversarial games from {args.adversarial_dir}...")
    adv_games = []
    for c in range(60):
        p = os.path.join(args.adversarial_dir, f'cell_{c:02d}.npz')
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        for g in d['top_games']:
            adv_games.append(list(g))
    print(f"  {len(adv_games)} adversarial candidate games")

    # ---- Find adversarial POSITIONS (top-1 illegal) in those games ----
    print("Finding positions where OGPT top-1 is illegal...")
    adv_positions = find_adversarial_positions(
        model, dataset, device, adv_games, block_size,
        max_positions=args.max_adversarial,
    )
    print(f"  {len(adv_positions)} adversarial positions")

    # Bucket by turn
    adv_by_turn = {}
    for game, t, top1 in adv_positions:
        adv_by_turn.setdefault(t, []).append((game, top1))
    print(f"  turn range: {min(adv_by_turn)}..{max(adv_by_turn)}")
    for t in sorted(adv_by_turn):
        print(f"    turn {t:>2}: {len(adv_by_turn[t]):>4} adversarial positions")

    # ---- Build control set (natural non-adversarial positions matched by turn) ----
    print(f"\nBuilding turn-matched control from {args.n_control_games} val games...")
    control_by_turn = {t: [] for t in adv_by_turn}
    needed_by_turn = {t: len(v) for t, v in adv_by_turn.items()}
    stoi_arr = np.zeros(64, dtype=np.int64)
    for pos, tok in stoi.items():
        if isinstance(pos, int):
            stoi_arr[pos] = tok

    with torch.no_grad():
        for g_idx, game in enumerate(val_games[:args.n_control_games]):
            if all(len(control_by_turn[t]) >= needed_by_turn[t] * 3
                   for t in needed_by_turn):
                break
            board = OthelloBoardState()
            L = min(len(game), block_size)
            tokens = torch.tensor(
                [stoi[s] for s in game[:L]], dtype=torch.long, device=device
            ).unsqueeze(0)
            logits, _ = model(tokens)
            top1 = logits.argmax(dim=-1)[0].cpu().numpy()
            for t in range(L):
                valid = board.get_valid_moves()
                if valid and t in control_by_turn and \
                   len(control_by_turn[t]) < needed_by_turn[t] * 3:
                    top1_token = int(top1[t])
                    top1_board_pos = None
                    for c, i in stoi.items():
                        if i == top1_token and isinstance(c, int):
                            top1_board_pos = c
                            break
                    if top1_board_pos is not None and top1_board_pos in valid:
                        control_by_turn[t].append(tuple(game[:L]))
                if t < len(game):
                    board.umpire(game[t])
    total_ctrl = sum(len(v) for v in control_by_turn.values())
    print(f"  {total_ctrl} control positions found across matched turns")

    # ---- Evaluate probe on both sets ----
    def eval_probe(positions_by_turn):
        """positions_by_turn: {turn: [(game_tuple, ...)] or [game_tuple]}."""
        per_turn_acc = {}
        for t, items in positions_by_turn.items():
            correct, total = 0, 0
            for item in items:
                game = item[0] if isinstance(item, tuple) else item
                if not isinstance(game, (list, tuple)):
                    game = item
                hidden, state = get_hidden_and_state(
                    model, dataset, device, tuple(game), t, args.layer,
                    block_size)
                pred = probe_predict(hidden, t, probe)
                gt = state_to_gt(state)
                correct += int((pred == gt).sum())
                total += 64
            per_turn_acc[t] = (correct / total) if total > 0 else None
        return per_turn_acc

    print("\nEvaluating probe on adversarial positions...")
    t0 = time.time()
    adv_acc = eval_probe(adv_by_turn)
    print(f"  done in {int(time.time()-t0)}s")

    print("Evaluating probe on turn-matched control positions...")
    t0 = time.time()
    ctrl_acc = eval_probe(control_by_turn)
    print(f"  done in {int(time.time()-t0)}s")

    # ---- Report ----
    print()
    print("=== Probe accuracy: adversarial vs. turn-matched control ===")
    print(f"  {'turn':>4}  {'adv_acc':>8}  {'ctrl_acc':>8}  {'delta':>8}  "
          f"{'n_adv':>6}  {'n_ctrl':>7}")
    print("  " + "-" * 55)
    adv_sum_c = adv_sum_t = ctrl_sum_c = ctrl_sum_t = 0
    for t in sorted(adv_acc):
        n_adv = len(adv_by_turn[t])
        n_ctrl = len(control_by_turn[t])
        a = adv_acc[t]
        c = ctrl_acc.get(t)
        delta = ((a - c) if (a is not None and c is not None) else None)
        a_str = f"{a:.4f}" if a is not None else "n/a"
        c_str = f"{c:.4f}" if c is not None else "n/a"
        d_str = f"{delta:+.4f}" if delta is not None else "n/a"
        print(f"  {t:>4}  {a_str:>8}  {c_str:>8}  {d_str:>8}  "
              f"{n_adv:>6}  {n_ctrl:>7}")
        if a is not None:
            adv_sum_c += a * n_adv * 64
            adv_sum_t += n_adv * 64
        if c is not None:
            ctrl_sum_c += c * n_ctrl * 64
            ctrl_sum_t += n_ctrl * 64
    print("  " + "-" * 55)
    if adv_sum_t > 0 and ctrl_sum_t > 0:
        adv_overall = adv_sum_c / adv_sum_t
        ctrl_overall = ctrl_sum_c / ctrl_sum_t
        print(f"  {'ALL':>4}  {adv_overall:>8.4f}  {ctrl_overall:>8.4f}  "
              f"{adv_overall - ctrl_overall:+.4f}")


if __name__ == '__main__':
    main()
