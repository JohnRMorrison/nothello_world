"""Two-layer readout on top of a frozen board-state MLP.

Motivation: the board-state MLP's hidden decodes cells at ~98.7%. But a
single Linear(H, 960) output can't compute AND functions, which is what
pattern firings require. Adding a second hidden layer + ReLU inside the
readout gives the nonlinearity needed to compute ANDs of board facts.

Architecture:
    x (60)
      -> Linear_frozen + ReLU   (loaded from trained board-state MLP)
           -> H_frozen hidden
      -> Linear_new + ReLU      (trainable)
           -> H_new hidden
      -> Linear(H_new, 960)     (trainable)

Only the last two layers are trained. Pattern BCE + pos_weight.

Usage:
    python train_deep_readout.py \
        --backbone experiments/.../mlp_checkpoints/mlp_when_H512_streaming.pt \
        --new-hidden 512 --epochs 3 --pos-weight 5
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
    compute_pattern_labels_batch,
    pat_labels_to_cell_labels, _get_cell_pat_index,
)


def load_backbone(ckpt_path, device):
    """Load frozen first layer from a trained board-state MLP checkpoint.

    Returns (lin_even, lin_odd, H_frozen, input_dim).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    H = ckpt['hidden_dim']
    D = ckpt['input_dim']
    lin_even = nn.Linear(D, H).to(device)
    lin_odd = nn.Linear(D, H).to(device)
    # _build_mlp uses nn.Sequential; first Linear is at index 0.
    # _build_mlp uses nn.Sequential; state-dict keys are "0.weight",
    # "0.bias", etc. (no "net." prefix).
    lin_even.weight.data = ckpt['even']['0.weight'].to(device)
    lin_even.bias.data = ckpt['even']['0.bias'].to(device)
    lin_odd.weight.data = ckpt['odd']['0.weight'].to(device)
    lin_odd.bias.data = ckpt['odd']['0.bias'].to(device)
    for p in lin_even.parameters(): p.requires_grad = False
    for p in lin_odd.parameters(): p.requires_grad = False
    return lin_even, lin_odd, H, D, ckpt.get('best_acc', None)


def prob_or_scores(pat_logits, idx, mask):
    log1m = -nn.functional.softplus(pat_logits)
    gathered = log1m[:, idx]
    gathered = gathered.masked_fill(~mask, 0.0)
    return -gathered.sum(dim=-1)


class DeepReadout(nn.Module):
    def __init__(self, H_frozen, H_new, n_patterns=960):
        super().__init__()
        self.mid = nn.Linear(H_frozen, H_new)
        self.out = nn.Linear(H_new, n_patterns)

    def forward(self, h_frozen):
        # h_frozen is already ReLU'd
        return self.out(torch.relu(self.mid(h_frozen)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True,
                        help="Path to mlp_when_H{H}_streaming.pt (frozen)")
    parser.add_argument("--new-hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--output-dir",
                        default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = parser.parse_args()

    device = get_device()
    feature_cols = list(range(N_MOVES, 2 * N_MOVES))

    lin_even, lin_odd, H_frozen, D, backbone_acc = load_backbone(args.backbone, device)
    assert D == N_MOVES, f"Expected 60-d backbone input, got {D}"
    print(f"Loaded backbone {args.backbone} (H={H_frozen}, acc={backbone_acc})")

    readout_even = DeepReadout(H_frozen, args.new_hidden).to(device)
    readout_odd = DeepReadout(H_frozen, args.new_hidden).to(device)

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device)
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)
    pw = torch.tensor([args.pos_weight], device=device)

    params = list(readout_even.parameters()) + list(readout_odd.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f and "_when60" not in f)
    if len(chunk_files) == 1:
        eval_path = chunk_files[0]; train_paths = chunk_files
        print(f"Single-chunk mode: train+eval on {chunk_files[0]}")
    else:
        eval_path = chunk_files[-1]; train_paths = chunk_files[:-1]

    def eval_pass():
        readout_even.eval(); readout_odd.eval()
        totals_pat = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        agg = {a: {n: {'c': 0, 't': 0} for n in (1, 3, 5, 10)}
               for a in ('max', 'logsumexp', 'prob_or')}
        X, Y, pos_ = _load_features(eval_path)
        X = X[:, feature_cols]
        n = min(len(X), 49 * 10000)
        rng = np.random.RandomState(0)
        si = np.sort(rng.choice(len(X), n, replace=False))
        X, Y, pos_ = X[si], Y[si], pos_[si]
        with torch.no_grad():
            for i in range(0, n, 1024):
                x = X[i:i+1024].to(device); yb = Y[i:i+1024]; p = pos_[i:i+1024]
                em = (p % 2 == 0); om = ~em
                pl = torch.zeros(len(x), 960, device=device)
                if em.any():
                    h = torch.relu(lin_even(x[em]))
                    pl[em] = readout_even(h)
                if om.any():
                    h = torch.relu(lin_odd(x[om]))
                    pl[om] = readout_odd(h)
                gp = torch.from_numpy(compute_pattern_labels_batch(
                    yb.numpy(), p.numpy(),
                    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                ).to(device)
                pred = (pl > 0); gt = (gp > 0.5)
                totals_pat['tp'] += (pred & gt).sum().item()
                totals_pat['fp'] += (pred & ~gt).sum().item()
                totals_pat['fn'] += (~pred & gt).sum().item()
                totals_pat['tn'] += (~pred & ~gt).sum().item()
                legal = pat_labels_to_cell_labels(gp, pattern_to_cell)
                gathered_ninf = pl[:, idx].masked_fill(~mask, float('-inf'))
                scores = {
                    'max':       gathered_ninf.max(dim=-1).values,
                    'logsumexp': torch.logsumexp(gathered_ninf, dim=-1),
                    'prob_or':   prob_or_scores(pl, idx, mask),
                }
                gl = (legal > 0.5).cpu().numpy()
                for name, s in scores.items():
                    cs = s.cpu().numpy()
                    for b in range(cs.shape[0]):
                        ls = set(np.where(gl[b])[0].tolist()); K = len(ls)
                        if K == 0: continue
                        r = np.argsort(-cs[b])
                        for k in (1, 3, 5, 10):
                            kk = min(k, K)
                            agg[name][k]['c'] += len(set(r[:kk].tolist()) & ls)
                            agg[name][k]['t'] += kk
        return totals_pat, agg

    def _print(pat_t, agg_t, label):
        tot = sum(pat_t.values())
        acc = (pat_t['tp'] + pat_t['tn']) / max(tot, 1)
        rec = pat_t['tp'] / max(pat_t['tp'] + pat_t['fn'], 1)
        pre = pat_t['tp'] / max(pat_t['tp'] + pat_t['fp'], 1)
        print(f"  {label}")
        print(f"    pattern: acc={acc:.4%} recall={rec:.4%} prec={pre:.4%}")
        for name in ('max', 'logsumexp', 'prob_or'):
            row = [name]
            for k in (1, 3, 5, 10):
                d = agg_t[name][k]
                row.append(f"{d['c']/max(d['t'],1):.4%}")
            print(f"    {row[0]:10s} top-1={row[1]} top-3={row[2]} top-5={row[3]} top-10={row[4]}")

    print(f"\nTraining: {len(train_paths)} chunks x {args.epochs} epochs, "
          f"H_frozen={H_frozen}, H_new={args.new_hidden}, lr={args.lr}, pw={args.pos_weight}")
    for epoch in range(1, args.epochs + 1):
        readout_even.train(); readout_odd.train()
        rng = np.random.RandomState(epoch)
        order = rng.permutation(len(train_paths))
        total_loss = 0.0; total_batches = 0
        for ci in order:
            tr_X, tr_Y, tr_pos = _load_features(train_paths[ci])
            tr_X = tr_X[:, feature_cols]
            perm = torch.randperm(len(tr_X))
            for i in range(0, len(tr_X), 1024):
                sel = perm[i:i + 1024]
                x = tr_X[sel].to(device); yb = tr_Y[sel]; p = tr_pos[sel]
                with torch.no_grad():
                    gp = torch.from_numpy(compute_pattern_labels_batch(
                        yb.numpy(), p.numpy(),
                        pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask)
                    ).to(device)
                em = (p % 2 == 0); om = ~em
                loss = torch.tensor(0.0, device=device)
                for msk, r_model, lin in [(em, readout_even, lin_even),
                                          (om, readout_odd, lin_odd)]:
                    if not msk.any(): continue
                    with torch.no_grad():
                        h = torch.relu(lin(x[msk]))
                    pl = r_model(h)
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(
                        pl, gp[msk], pos_weight=pw)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item(); total_batches += 1
            del tr_X, tr_Y, tr_pos
        avg = total_loss / max(total_batches, 1)
        pat_t, agg_t = eval_pass()
        print(f"\nEpoch {epoch}: loss={avg:.5f}", flush=True)
        _print(pat_t, agg_t, "deep-readout")

    base = os.path.splitext(os.path.basename(args.backbone))[0]
    save_dir = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,
        f"deepro_{base}_Hnew{args.new_hidden}_pw{int(args.pos_weight)}.pt")
    torch.save({
        'even': readout_even.state_dict(),
        'odd': readout_odd.state_dict(),
        'backbone_ckpt': args.backbone,
        'H_frozen': H_frozen,
        'H_new': args.new_hidden,
    }, save_path)
    print(f"\nSaved to {save_path}")
