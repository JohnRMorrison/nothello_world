"""Handcrafted pattern readout on top of a trained board-state MLP.

Zero training. For each of 960 patterns, we know exactly which cells'
probabilities should be high for the pattern to fire:
  - target: P(empty) = 1
  - each opp_cell: P(opp_color) = 1
  - terminal: P(my_color) = 1

We build a Linear(192, 960) where each row has weight 1/N on its N
required cells' correct channels (N = length + 2) and zero elsewhere.
Pattern score = average of required probabilities. Saturates at 1.0
when all conditions are met. Separate even/odd readouts because
"my color" flips with current turn.

If this gives >95% top-1 legal, the entire 85% ceiling on trained
pattern-detectors is "imperfect pattern computation", not "missing
information in the hidden." If it caps at 88%, the board-state MLP's
192-d output itself is the bottleneck.

Usage:
    python handcrafted_readout.py \
        --backbone experiments/.../mlp_checkpoints/mlp_when_H512_streaming.pt \
        --hidden 512
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, get_device, N_MOVES,
)
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from generate_rule_games import precompute_pattern_arrays
from train_pattern_simple import (
    compute_pattern_labels_batch, pat_labels_to_cell_labels, _get_cell_pat_index,
)


def build_handcrafted_readout(patterns, for_even_turn):
    """Return a (960, 192) weight matrix.

    Board-state output layout: each cell has 3 channels
        ch 0 = empty, ch 1 = white, ch 2 = black.
    Pattern j requires: target empty, opp_cells opp_color, terminal my_color.
    "My color" is white (ch 1) on even positions, black (ch 2) on odd.

    Weight per required cell channel = 1/N (with N = num required cells),
    so the raw sum is the fraction of conditions met (1.0 when all hold).
    """
    mine_ch = 1 if for_even_turn else 2
    opp_ch = 2 if for_even_turn else 1
    W = torch.zeros(len(patterns), 64 * 3)
    for j, p in enumerate(patterns):
        required = [(p['target'], 0)]                        # empty at target
        for c in p['opponents']:
            required.append((c, opp_ch))                     # opp color at opp cell
        required.append((p['terminal'], mine_ch))             # my color at terminal
        N = len(required)
        for cell, ch in required:
            W[j, cell * 3 + ch] = 1.0 / N
    return W


def load_board_mlp(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    H, D = ckpt['hidden_dim'], ckpt['input_dim']
    # _build_mlp wraps Linear + ReLU + Linear in nn.Sequential.
    mlp_even = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, 64 * 3)).to(device)
    mlp_odd = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, 64 * 3)).to(device)
    mlp_even.load_state_dict(ckpt['even'])
    mlp_odd.load_state_dict(ckpt['odd'])
    mlp_even.eval(); mlp_odd.eval()
    return mlp_even, mlp_odd, H, D, ckpt.get('best_acc', None)


def prob_or_scores(pat_logits, idx, mask):
    log1m = -nn.functional.softplus(pat_logits)
    gathered = log1m[:, idx]
    gathered = gathered.masked_fill(~mask, 0.0)
    return -gathered.sum(dim=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True,
                        help="Path to mlp_when_H{H}_streaming.pt (board-state MLP)")
    parser.add_argument("--hidden", type=int, required=True, help="H_frozen for sanity")
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    feature_cols = list(range(N_MOVES, 2 * N_MOVES))

    mlp_even, mlp_odd, H, D, backbone_acc = load_board_mlp(args.backbone, device)
    assert H == args.hidden
    print(f"Loaded backbone {args.backbone} (H={H}, board_acc={backbone_acc})")

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device)
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    # Handcrafted readouts for even/odd turns
    W_even = build_handcrafted_readout(patterns, for_even_turn=True).to(device)
    W_odd = build_handcrafted_readout(patterns, for_even_turn=False).to(device)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    eval_path = chunk_files[-1]
    print(f"Eval: {os.path.basename(eval_path)} (random sample)")

    X, Y, pos = _load_features(eval_path)
    X = X[:, feature_cols]
    n = min(len(X), 49 * 10000)
    rng = np.random.RandomState(0)
    si = np.sort(rng.choice(len(X), n, replace=False))
    X, Y, pos = X[si], Y[si], pos[si]

    # Also record the sanity-check version: handcrafted readout on GROUND-TRUTH
    # probabilities (built from Y labels). This must give near-100% top-1 legal.
    aggregators_to_report = ('max', 'logsumexp', 'prob_or')
    mlp_res = {a: {k: {'c': 0, 't': 0} for k in (1, 3, 5, 10)} for a in aggregators_to_report}
    gt_res = {a: {k: {'c': 0, 't': 0} for k in (1, 3, 5, 10)} for a in aggregators_to_report}

    batch = 1024
    with torch.no_grad():
        for i in range(0, n, batch):
            x = X[i:i+batch].to(device); yb = Y[i:i+batch]; p = pos[i:i+batch]
            em = (p % 2 == 0); om = ~em

            # 1. Board-state probabilities from the MLP
            probs = torch.zeros(len(x), 64 * 3, device=device)
            if em.any():
                logits = mlp_even(x[em]).view(-1, 64, 3)
                probs[em] = torch.softmax(logits, dim=-1).view(-1, 192)
            if om.any():
                logits = mlp_odd(x[om]).view(-1, 64, 3)
                probs[om] = torch.softmax(logits, dim=-1).view(-1, 192)

            # 2. Pattern scores using handcrafted readouts
            pl_mlp = torch.zeros(len(x), len(patterns), device=device)
            if em.any(): pl_mlp[em] = probs[em] @ W_even.T
            if om.any(): pl_mlp[om] = probs[om] @ W_odd.T

            # 3. Ground-truth one-hot board (upper bound)
            Y_gpu = yb.to(device)
            gt_probs = torch.zeros(len(x), 64 * 3, device=device)
            gt_probs[:, 0::3] = (Y_gpu == 0).float()
            gt_probs[:, 1::3] = (Y_gpu == 1).float()
            gt_probs[:, 2::3] = (Y_gpu == 2).float()
            pl_gt = torch.zeros(len(x), len(patterns), device=device)
            if em.any(): pl_gt[em] = gt_probs[em] @ W_even.T
            if om.any(): pl_gt[om] = gt_probs[om] @ W_odd.T

            # 4. Legal labels from ground truth
            gp = torch.from_numpy(compute_pattern_labels_batch(
                yb.numpy(), p.numpy(),
                pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)).to(device)
            legal = pat_labels_to_cell_labels(gp, pattern_to_cell)
            gl = (legal > 0.5).cpu().numpy()

            for results_dict, pl in [(mlp_res, pl_mlp), (gt_res, pl_gt)]:
                gathered_ninf = pl[:, idx].masked_fill(~mask, float('-inf'))
                scores_map = {
                    'max':       gathered_ninf.max(dim=-1).values,
                    'logsumexp': torch.logsumexp(gathered_ninf, dim=-1),
                    'prob_or':   prob_or_scores(pl, idx, mask),
                }
                for name, s in scores_map.items():
                    cs = s.cpu().numpy()
                    for b in range(cs.shape[0]):
                        ls = set(np.where(gl[b])[0].tolist()); K = len(ls)
                        if K == 0: continue
                        r = np.argsort(-cs[b])
                        for k in (1, 3, 5, 10):
                            kk = min(k, K)
                            results_dict[name][k]['c'] += len(set(r[:kk].tolist()) & ls)
                            results_dict[name][k]['t'] += kk

    def _print(res, label):
        print(f"  {label}")
        for name in aggregators_to_report:
            row = [name]
            for k in (1, 3, 5, 10):
                d = res[name][k]
                row.append(f"{d['c']/max(d['t'],1):.4%}")
            print(f"    {row[0]:10s} top-1={row[1]} top-3={row[2]} top-5={row[3]} top-10={row[4]}")

    print("\nHandcrafted readout on GROUND-TRUTH board (sanity; expect ~100%):")
    _print(gt_res, "ground-truth board")
    print("\nHandcrafted readout on MLP board-state output:")
    _print(mlp_res, "MLP board-state")
