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


def _tokenize_one(game, block_size, device):
    """Tokenize one game using Nanda's canonical vocab (tokenize_games handles
    the mapping)."""
    toks = tokenize_games([game], seq_len=block_size).to(device)
    return toks  # (1, block_size)


def state_to_gt(state_8x8):
    """OthelloBoardState.state -> Nanda probe GT: 0=empty, 1=white, 2=black."""
    gt = np.zeros((8, 8), dtype=np.int64)
    gt[state_8x8 == -1] = 1
    gt[state_8x8 == 1]  = 2
    return gt


def _build_token_to_board_pos(block_size, device):
    """Return array of shape (VOCAB_SIZE,) mapping token index -> board pos.

    Uses tokenize_games to derive Nanda's canonical vocab.  Returns -1 for
    tokens that don't correspond to a board cell (e.g. padding token 0).
    """
    valid_moves = [i for i in range(64) if i not in {27, 28, 35, 36}]
    dummy_game = list(valid_moves)  # play each valid cell once (length 60)
    toks = tokenize_games([dummy_game], seq_len=block_size)  # (1, block_size)
    toks = toks[0].tolist()
    token_to_pos = np.full(VOCAB_SIZE, -1, dtype=np.int64)
    for i, m in enumerate(dummy_game):
        if i < len(toks):
            token_to_pos[toks[i]] = m
    return token_to_pos


def find_adversarial_positions(model, device, games, block_size,
                                 token_to_pos, max_positions=None):
    """For each game, find positions where OGPT's top-1 is illegal.

    Returns list of (game_tuple, turn, top1_board_pos) tuples.

    Convention: 'turn' T means state after T+1 moves played (matches
    analyze_nanda_probe_per_cell.py's states[i, t] = state after t+1 moves,
    where t is the 0-indexed absolute position).  Model's prediction for
    move at index T+1 given tokens[0..T] = logits[0, T].  We check whether
    that prediction is legal against the board state AFTER move T is played
    (i.e., the state the model has "seen" via tokens[0..T]).
    """
    positions = []
    with torch.no_grad():
        for g_idx, game in enumerate(games):
            L = min(len(game), block_size)
            tokens = tokenize_games([list(game)], seq_len=block_size).to(device)
            logits, _ = model(tokens)                        # (1, block_size, V)
            top1 = logits.argmax(dim=-1)[0].cpu().numpy()    # (block_size,)

            # Replay to get board state at each position; state[T] = state
            # after moves 0..T played (T+1 moves total).
            board = OthelloBoardState()
            for T in range(L):
                try:
                    board.umpire(game[T])
                except Exception:
                    break
                # After playing game[T], model's prediction for the NEXT
                # move (index T+1) uses logits[0, T].  Board.get_valid_moves()
                # now reflects the state after T+1 moves — the correct
                # comparison population for logits[0, T].
                valid = board.get_valid_moves()
                if not valid:
                    continue
                top1_token = int(top1[T])
                top1_board_pos = int(token_to_pos[top1_token])
                if top1_board_pos < 0:
                    continue
                if top1_board_pos not in valid:
                    positions.append((tuple(game[:L]), T, top1_board_pos))
                    if max_positions and len(positions) >= max_positions:
                        return positions
    return positions


def get_hidden_and_state(model, device, game_tuple, turn, layer, block_size):
    """Return (hidden_512, true_state_8x8) at position `turn` (state after
    turn+1 moves — matches probe training convention)."""
    L = min(len(game_tuple), block_size)
    game = list(game_tuple)[:L]
    tokens = tokenize_games([game], seq_len=block_size).to(device)
    with torch.no_grad():
        h = extract_activations(model, tokens, layer)  # (1, block_size, 512)
    hidden = h[0, turn, :].cpu().numpy()               # (512,)

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

    # Build the canonical token->board-pos map from Nanda's tokenize_games.
    token_to_pos = _build_token_to_board_pos(block_size, device)
    print(f"Built Nanda token->board-pos map: "
          f"{int((token_to_pos >= 0).sum())} valid entries / {VOCAB_SIZE}")

    print(f"Loading val games (up to {args.max_files} files)...")
    val_games = load_games(max_files=args.max_files)
    print(f"  {len(val_games)} val games loaded")

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
        model, device, adv_games, block_size, token_to_pos,
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

    with torch.no_grad():
        for g_idx, game in enumerate(val_games[:args.n_control_games]):
            if all(len(control_by_turn[t]) >= needed_by_turn[t] * 3
                   for t in needed_by_turn):
                break
            L = min(len(game), block_size)
            tokens = tokenize_games([list(game)], seq_len=block_size).to(device)
            logits, _ = model(tokens)
            top1 = logits.argmax(dim=-1)[0].cpu().numpy()

            board = OthelloBoardState()
            for T in range(L):
                try:
                    board.umpire(game[T])
                except Exception:
                    break
                valid = board.get_valid_moves()
                if not valid:
                    continue
                if T not in control_by_turn:
                    continue
                if len(control_by_turn[T]) >= needed_by_turn[T] * 3:
                    continue
                top1_token = int(top1[T])
                top1_board_pos = int(token_to_pos[top1_token])
                if top1_board_pos >= 0 and top1_board_pos in valid:
                    control_by_turn[T].append(tuple(game[:L]))
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
                    model, device, tuple(game), t, args.layer, block_size)
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

    # ---- Causal test: are probe errors ON the illegal-predicted cell? ----
    # For each adversarial (game, turn, top1_illegal_cell) triple, check
    # whether the probe's per-cell prediction is WRONG on that specific cell.
    # Compare to two baselines:
    #   - Probe error rate on a random OTHER cell in the same position
    #   - Probe error rate on any played (non-empty) cell
    # If error on the top1 cell isn't higher than the baselines, the world-
    # model degradation is INCIDENTAL, not causal.
    print("\nCausal test — is the probe's decoded state responsible for the illegal pick?")

    # Helper: reconstruct nanda state (1=black, -1=white, 0=empty) from probe pred
    def probe_to_nanda_state(pred_flat):
        """pred_flat: (64,) with 0=empty, 1=white, 2=black -> (8,8) nanda state."""
        st = np.zeros((8, 8), dtype=np.int8)
        for cell in range(64):
            r_, c_ = cell // 8, cell % 8
            if pred_flat[cell] == 1:
                st[r_, c_] = -1
            elif pred_flat[cell] == 2:
                st[r_, c_] = 1
        return st

    def next_hand_color_at_turn(t):
        """Turn t = t+1 moves played; next player = BLACK if (t+1) even else WHITE."""
        k = t + 1
        return 1 if (k % 2 == 0) else -1  # 1=black, -1=white

    def flanking_ray_cells(cell_C):
        """Cells along the 8 directions from C, up to the board edge — the set
        whose contents matter for whether any flanking pattern makes C legal."""
        rr, cc = cell_C // 8, cell_C % 8
        rays = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            r_, c_ = rr + dr, cc + dc
            while 0 <= r_ < 8 and 0 <= c_ < 8:
                rays.append(r_ * 8 + c_)
                r_ += dr; c_ += dc
        return rays

    n_pos_analyzed = 0
    n_wrong_on_top1 = 0
    n_wrong_on_random = 0
    n_wrong_on_random_played = 0
    n_wrong_on_any_ray_cell = 0
    n_legal_under_probe = 0     # KEY: does probe's board make C legal?
    n_had_probe_state = 0
    for t, items in adv_by_turn.items():
        color_next = next_hand_color_at_turn(t)
        for game, top1_board_pos in items:
            hidden, state = get_hidden_and_state(
                model, device, tuple(game), t, args.layer, block_size)
            pred = probe_predict(hidden, t, probe)                # (8, 8)
            gt = state_to_gt(state)                                # (8, 8)
            pred_flat = pred.flatten()
            gt_flat = gt.flatten()
            per_cell_wrong = (pred_flat != gt_flat)               # (64,)
            n_pos_analyzed += 1

            # (a) probe wrong on top1 cell itself
            if per_cell_wrong[top1_board_pos]:
                n_wrong_on_top1 += 1

            # (b) probe wrong on ANY cell along the 8 rays from C
            ray_cells = flanking_ray_cells(top1_board_pos)
            if any(per_cell_wrong[i] for i in ray_cells):
                n_wrong_on_any_ray_cell += 1

            # (c) DIRECT causal question: is C legal under the probe's decoded board?
            probe_state = probe_to_nanda_state(pred_flat)
            try:
                b = OthelloBoardState()
                b.state = probe_state.copy()
                b.next_hand_color = color_next
                valid_under_probe = b.get_valid_moves()
                n_had_probe_state += 1
                if top1_board_pos in valid_under_probe:
                    n_legal_under_probe += 1
            except Exception:
                pass

            # Baseline: random OTHER cell + random played cell
            other_cells = [i for i in range(64) if i != top1_board_pos]
            rnd = other_cells[np.random.randint(len(other_cells))]
            if per_cell_wrong[rnd]:
                n_wrong_on_random += 1
            played = [i for i in range(64)
                       if state.flatten()[i] != 0 and i != top1_board_pos]
            if played:
                r2 = played[np.random.randint(len(played))]
                if per_cell_wrong[r2]:
                    n_wrong_on_random_played += 1

    print(f"  Positions analyzed: {n_pos_analyzed}")
    if n_pos_analyzed > 0:
        p_top1 = n_wrong_on_top1 / n_pos_analyzed
        p_ray = n_wrong_on_any_ray_cell / n_pos_analyzed
        p_legal_under_probe = (n_legal_under_probe / max(n_had_probe_state, 1))
        p_rnd = n_wrong_on_random / n_pos_analyzed
        p_rnd_played = n_wrong_on_random_played / n_pos_analyzed
        print()
        print(f"  === Category (a): probe error ON the illegal cell ===")
        print(f"    P(probe wrong on top1 illegal cell): {p_top1:.4f}")
        print(f"    P(probe wrong on any other cell):    {p_rnd:.4f}")
        print(f"    P(probe wrong on random played cell):{p_rnd_played:.4f}")
        print()
        print(f"  === Category (b): probe error on any ray/flanking-relevant cell ===")
        print(f"    P(probe wrong on ANY cell in 8-ray from illegal cell): {p_ray:.4f}")
        print()
        print(f"  === Category (c) — direct causal test ===")
        print(f"    P(illegal cell is LEGAL under probe's decoded board): "
              f"{p_legal_under_probe:.4f}")
        print(f"    (Positions analyzed for (c): {n_had_probe_state})")
        print()
        print(f"  Interpretation of (c):")
        print(f"    High P -> World model DOES rationalize the illegal pick (causal)")
        print(f"    Low P  -> World model does NOT rationalize the illegal pick")
        print(f"             (the model would agree C is illegal from its own state,")
        print(f"              but still picks it -> downstream heuristic override)")

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
