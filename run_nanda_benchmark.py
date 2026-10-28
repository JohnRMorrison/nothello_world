"""Reproduce Nanda, Lee & Wattenberg (2023), Table 2: linear probe addition.

Their intervention (Sec. 4.1): at each intervened layer, add a fixed multiple
of the probe's vector for the TARGET state to the residual stream:

    x' <- x + alpha * d_target,   d_target = unit(W_layer[:, r, c, target])

(target = MINE/YOURS for flipping colours, EMPTY for erasing). alpha is one
fixed number for every case; the paper does not report it, so it is swept.
Each layer uses its own linear probe (main_linear_probe_L{k}.pth). The push is
applied at the last token, like the rest of this repo.

Layer sets follow their Figure 7: first N layers (N=1..8) and last N layers
(N=1..7). Layer i = resid_post of block i-1 (block 0..7).

Test cases: Li et al.'s benchmark (intervention_benchmark.pkl).
  flip   : the benchmark cases as given
  remove : same game, same square, erased instead (their "Erasing" set is not
           published; cases whose erasure does not change the legal moves are
           dropped, as in the rest of the repo)

Metric: Li et al. top-N errors = false positives + false negatives, N = number
of legal moves on the edited board.

Usage:
    python run_nanda_benchmark.py
"""
import argparse
import collections
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config, intervention, probes, data, metrics

ap = argparse.ArgumentParser()
ap.add_argument('--benchmark-path', default='intervention_benchmark.pkl')
ap.add_argument('--alphas', type=float, nargs='+',
                default=[0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
ap.add_argument('--out', default='Intervention Results/nanda_benchmark')
args = ap.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_BLOCKS = 8
ALPHAS = torch.tensor(args.alphas, dtype=torch.float32, device=DEVICE)
LAYER_SETS = ([(f'first{n}', list(range(n))) for n in range(1, N_BLOCKS + 1)]
              + [(f'last{n}', list(range(N_BLOCKS - n, N_BLOCKS)))
                 for n in range(1, N_BLOCKS)])
STOI = {c: i + 1 for i, c in enumerate(config.STOI_INDICES)}

model = intervention.load_model(device=DEVICE)
layer_probes = intervention.load_layer_probes(range(N_BLOCKS), device=DEVICE)
print(f'device {DEVICE}; probes: ' + ', '.join(
    f'block {k}={os.path.basename(intervention.layer_probe_path(k))}'
    for k in range(N_BLOCKS)))


def topn_errors(logits_last, legal_set):
    """Li et al. top-N errors (FP + FN), same as run_li_benchmark.py."""
    legal_stoi = set(legal_set) & set(config.STOI_INDICES)
    n = len(legal_stoi)
    if n == 0:
        return None
    topn = metrics.cell_logits(logits_last).argsort(descending=True)[:n]
    topn_cells = {config.STOI_INDICES[int(i)] for i in topn}
    return len(topn_cells - legal_stoi) + len(legal_stoi - topn_cells)


# ── build cases ──────────────────────────────────────────────────────────────
with open(args.benchmark_path, 'rb') as f:
    raw = pickle.load(f)
records = []
for case in raw:
    history = np.array([int(m) for m in case['history']], dtype=np.int64)
    pos_flat = int(case['pos_int'])
    for cat in ('flip', 'remove'):
        rec = data.position_from_replay(history, len(history) - 1,
                                        (pos_flat // 8, pos_flat % 8), cat)
        if rec is not None:
            records.append(rec)
cat_counts = collections.Counter((r.category, r.sub_condition) for r in records)
print(f'{len(raw)} benchmark cases -> {len(records)} records: {dict(cat_counts)}')

# ── run ──────────────────────────────────────────────────────────────────────
R, L, A = len(records), len(LAYER_SETS), len(args.alphas)
err_before = np.full(R, np.nan)
err_after = np.full((R, L, A), np.nan)
norm_h = np.zeros((R, N_BLOCKS))
proj = np.zeros((R, N_BLOCKS))

with torch.no_grad():
    for ri, rec in enumerate(records):
        tokens = torch.tensor([[STOI[m] for m in rec.move_history]],
                              dtype=torch.long, device=DEVICE)
        pos = rec.position
        r, c = rec.square
        tc = probes.board_val_to_probe_class(rec.target_val, rec.next_color, 0)
        dirs = []
        for k in range(N_BLOCKS):
            d = layer_probes[k][0, :, r, c, tc]
            dirs.append(d / d.norm())

        x0 = intervention.compute_prefix(model, tokens, 0)   # embeddings
        x = x0.clone()
        for k in range(N_BLOCKS):
            x = model.blocks[k](x)
            norm_h[ri, k] = x[0, pos].norm().item()
            proj[ri, k] = (x[0, pos] @ dirs[k]).item()
        clean_last = model.head(model.ln_f(x))[0, -1].cpu()
        err_before[ri] = topn_errors(clean_last, rec.legal_cf)

        for li, (_, layers) in enumerate(LAYER_SETS):
            x = x0.expand(A, -1, -1).clone()
            for k in range(N_BLOCKS):
                x = model.blocks[k](x)
                if k in layers:
                    x[:, pos] = x[:, pos] + ALPHAS[:, None] * dirs[k][None, :]
            logits = model.head(model.ln_f(x))[:, -1].cpu()
            for ai in range(A):
                err_after[ri, li, ai] = topn_errors(logits[ai], rec.legal_cf)
        if (ri + 1) % 200 == 0:
            print(f'  {ri + 1}/{R}', flush=True)

# ── save ─────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
np.savez_compressed(
    args.out + '.npz',
    err_before=err_before, err_after=err_after,
    alphas=np.array(args.alphas), layer_sets=np.array([n for n, _ in LAYER_SETS]),
    category=np.array([r.category for r in records]),
    sub_condition=np.array([r.sub_condition for r in records]),
    square=np.array([r.square for r in records]),
    position=np.array([r.position for r in records]),
    norm_h=norm_h, proj_on_target_dir=proj,
)

lines = []
def emit(s=''):
    print(s); lines.append(s)

emit('Nanda et al. (2023) Table 2 reproduction: linear probe ADDITION, '
     'x <- x + alpha * unit(probe vector of target state), each layer its own probe')
emit("Their Table 2: flipping colours 0.10 (null 2.723); erasing 0.02 (null 2.73)")
emit(f'records: {dict(cat_counts)}')
emit('residual norm at the last token, per block (mean): '
     + ' '.join(f'{v:.1f}' for v in norm_h.mean(0)))
cats = np.array([r.category for r in records])
for cat in ('flip', 'remove'):
    m = cats == cat
    if not m.any():
        continue
    emit(f'\n=== {cat} (n={int(m.sum())}) null intervention: {np.nanmean(err_before[m]):.3f}')
    emit('avg errors after; rows = layers intervened, cols = alpha')
    emit(f'{"layers":>8s} ' + ' '.join(f'{a:>6g}' for a in args.alphas))
    means = np.nanmean(err_after[m], axis=0)          # (L, A)
    for li, (name, _) in enumerate(LAYER_SETS):
        emit(f'{name:>8s} ' + ' '.join(f'{v:6.3f}' for v in means[li]))
    li, ai = np.unravel_index(np.nanargmin(means), means.shape)
    emit(f'best: {LAYER_SETS[li][0]}, alpha={args.alphas[ai]:g} -> {means[li, ai]:.3f}')
    all_i = [n for n, _ in LAYER_SETS].index(f'first{N_BLOCKS}')
    ai_all = int(np.nanargmin(means[all_i]))
    emit(f'all layers: best alpha={args.alphas[ai_all]:g} -> {means[all_i, ai_all]:.3f}')

with open(args.out + '.txt', 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print(f'\nsaved -> {args.out}.txt and {args.out}.npz')
