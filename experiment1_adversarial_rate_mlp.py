"""Experiment 1 — MLP baseline: per-cell adversarial success rate.

Same setup as experiment1_adversarial_rate.py (OGPT version), but the
model is a DirectMLP (played+even features -> 960 pattern logits ->
prob_or aggregation -> 60 cell scores).

Enumerates all 1,396 unique 5-move openings.  For each opening, runs
beam-width-10 search that maximizes the MLP's cell-score for a target
cell when that cell is illegal.  At every beam step, checks whether the
MLP's top-1 cell IS the target AND the target is currently illegal.

Reports per-cell success rate (# adversarial openings / 1,396).

Usage:
    python experiment1_adversarial_rate_mlp.py --cell 0 \\
        --mlp-ckpt $BASE/pattern_simple_direct_H512_playedeven.pt \\
        --hidden 512 --beam-width 10 --max-depth 40 --prefix-len 5
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from train_pattern_simple import DirectMLP, _get_cell_pat_index
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from compare_v4_vs_mlp import played_even_features, C64_TO_C60
from data.othello import OthelloBoardState

# Mirror the same VALID_MOVES / IDX_TO_MOVE conventions used elsewhere
CENTER_CELLS = {27, 28, 35, 36}
VALID_MOVES = [i for i in range(64) if i not in CENTER_CELLS]  # board pos 0..63
IDX_TO_MOVE = {idx: cell for idx, cell in enumerate(VALID_MOVES)}


def fast_copy_board(board):
    new = OthelloBoardState.__new__(OthelloBoardState)
    new.board_size = board.board_size
    new.state = board.state.copy()
    new.next_hand_color = board.next_hand_color
    new.history = list(board.history)
    new.age = board.age.copy()
    return new


def enumerate_prefixes(prefix_len):
    prefixes = []
    def rec(board, depth, seq):
        if depth == 0:
            prefixes.append(list(seq))
            return
        for m in board.get_valid_moves():
            nb = fast_copy_board(board)
            nb.umpire(m)
            seq.append(m)
            rec(nb, depth - 1, seq)
            seq.pop()
    rec(OthelloBoardState(), prefix_len, [])
    return prefixes


def load_mlp(mlp_ckpt_path, hidden, device):
    ckpt = torch.load(mlp_ckpt_path, map_location=device)
    input_dim = ckpt.get('input_dim', 120)
    n_patterns = ckpt.get('n_patterns', 960)
    me = DirectMLP(input_dim, hidden, n_patterns).to(device)
    mo = DirectMLP(input_dim, hidden, n_patterns).to(device)
    me.load_state_dict(ckpt['even'])
    mo.load_state_dict(ckpt['odd'])
    me.eval(); mo.eval()
    return me, mo, input_dim, n_patterns


@torch.no_grad()
def batch_cell_scores(me, mo, seqs, idx, mask, device):
    """Given a list of game sequences, compute 60-d cell scores per sequence.

    Returns:
        cell_scores: (N, 60) float32 tensor on device
    """
    feats = torch.stack([played_even_features(s) for s in seqs]).to(device)
    # Parity routing: at k = len(seq) moves played,
    # use_me (model_even) when (k % 2 == 1), matching val-game convention.
    ks = torch.tensor([len(s) for s in seqs], device=device)
    use_me = (ks % 2 == 1)
    use_mo = ~use_me

    B = len(seqs)
    logits = torch.zeros(B, 960, device=device)
    if use_me.any():
        logits[use_me] = me(feats[use_me])
    if use_mo.any():
        logits[use_mo] = mo(feats[use_mo])

    log1m = -F.softplus(logits)
    gathered = log1m[:, idx]
    gathered = gathered.masked_fill(~mask[None], 0.0)
    cell_scores = -gathered.sum(dim=-1)                # (B, 60), higher = more legal-likely
    return cell_scores


def run_start(me, mo, idx, mask, device, target_cell_60, prefix,
               beam_width, max_depth):
    """Beam search from one prefix.  Return (adversarial_found, best_seq, best_score)."""
    board = OthelloBoardState()
    for m in prefix:
        board.umpire(m)
    if not board.get_valid_moves():
        return False, list(prefix), 0.0

    beams = [(list(prefix), fast_copy_board(board), 0.0)]
    adversarial_found = False
    best_seq, best_score = list(prefix), 0.0

    for depth in range(len(prefix), max_depth):
        candidates = []
        for game_seq, brd, cum in beams:
            valid = brd.get_valid_moves()
            if not valid:
                continue
            for m in valid:
                new_seq = game_seq + [m]
                nb = fast_copy_board(brd)
                nb.umpire(m)
                candidates.append((new_seq, nb, cum))
        if not candidates:
            break

        # Batched MLP forward pass
        seqs = [c[0] for c in candidates]
        cell_scores = batch_cell_scores(me, mo, seqs, idx, mask, device)  # (N, 60)
        # Convert to probability under prob_or: p = 1 - exp(-cell_score)
        cell_probs = 1.0 - torch.exp(-cell_scores.clamp(min=0))  # (N, 60)
        top1 = cell_scores.argmax(dim=1).cpu().numpy()            # (N,)
        target_probs = cell_probs[:, target_cell_60].cpu().numpy()

        # Score cumulative + target-cell probability when illegal
        new_scored = []
        for i, (seq, brd, cum) in enumerate(candidates):
            valid_60 = {C64_TO_C60[m] for m in brd.get_valid_moves()
                         if m in C64_TO_C60}
            target_illegal = target_cell_60 not in valid_60
            if target_illegal and int(top1[i]) == target_cell_60:
                adversarial_found = True
            score = cum + (target_probs[i] if target_illegal else 0.0)
            new_scored.append((seq, brd, float(score)))
            if score > best_score:
                best_score = float(score)
                best_seq = list(seq)

        new_scored.sort(key=lambda x: x[2], reverse=True)
        beams = new_scored[:beam_width]

        if adversarial_found and depth >= len(prefix) + 8:
            break

    return adversarial_found, best_seq, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', type=int, required=True,
                    help='Target cell index 0..59 (indexes VALID_MOVES).')
    ap.add_argument('--mlp-ckpt', required=True)
    ap.add_argument('--hidden', type=int, required=True)
    ap.add_argument('--output-dir', default='experiment1_data_mlp')
    ap.add_argument('--beam-width', type=int, default=10)
    ap.add_argument('--prefix-len', type=int, default=5)
    ap.add_argument('--max-depth', type=int, default=40)
    ap.add_argument('--top-save', type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cell {args.cell} (board pos {VALID_MOVES[args.cell]})", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Loading MLP {args.mlp_ckpt}...", flush=True)
    me, mo, input_dim, n_patterns = load_mlp(args.mlp_ckpt, args.hidden, device)
    print(f"  H={args.hidden}, input_dim={input_dim}, n_patterns={n_patterns}",
          flush=True)

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    print(f"Enumerating {args.prefix_len}-move prefixes...", flush=True)
    prefixes = enumerate_prefixes(args.prefix_len)
    print(f"  {len(prefixes)} unique prefixes", flush=True)

    n_success = 0
    successes = []
    for pi, prefix in enumerate(prefixes):
        adv, seq, score = run_start(
            me, mo, idx, mask, device, args.cell, prefix,
            args.beam_width, args.max_depth,
        )
        if adv:
            n_success += 1
            successes.append((score, seq))
        if (pi + 1) % 100 == 0:
            print(f"  {pi+1}/{len(prefixes)}  "
                  f"success: {n_success}  ({int(time.time()-t0)}s)",
                  flush=True)

    rate = n_success / len(prefixes) if prefixes else 0.0
    print()
    print(f"=== Cell {args.cell} results ===")
    print(f"  Prefixes:              {len(prefixes)}")
    print(f"  Adversarial successes: {n_success}")
    print(f"  Success rate:          {rate:.4f}")
    print(f"  Elapsed:               {int(time.time()-t0)}s")

    os.makedirs(args.output_dir, exist_ok=True)
    successes.sort(key=lambda x: x[0], reverse=True)
    top_games = [s for _, s in successes[:args.top_save]]
    np.savez_compressed(
        os.path.join(args.output_dir, f"cell_{args.cell:02d}.npz"),
        cell=args.cell,
        hidden=args.hidden,
        n_prefixes=len(prefixes),
        n_success=n_success,
        rate=rate,
        top_scores=np.array([sc for sc, _ in successes[:args.top_save]],
                             dtype=np.float32),
        top_games=np.array(top_games, dtype=object),
    )


if __name__ == '__main__':
    main()
