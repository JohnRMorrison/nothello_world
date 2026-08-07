"""New-Squares FROZEN-tree transfer experiment (J1B).

Complement of new_squares_j1b.py.  Instead of REFITTING per-pattern trees for
the 8 new squares (feature discovery), this keeps the EXISTING J1B bank's trees
FIXED (banks/J1_perpattern.pt: 960 base-pattern trees, ~47k leaves, 121-d base
input) and trains ONLY a fresh prob-OR readout (the per-pattern sigmoids) on
those frozen leaves to predict new-square legality.  This isolates the
readout-learning stage -- a linear-probe-on-frozen-features TRANSFER test: how
much of the new-square task the pre-existing feature basis already supports,
with zero new feature discovery.

CAVEAT (by construction): the existing trees take only the 121-d base features
(60 movable 8x8 cells) and encode NOTHING about new-square state (cells 64-71).
So ALONG-ROW rules (new-square -> new-square flanks) are INVISIBLE to the
readout; only INTO-BOARD rules (which reference base cells) are learnable.  The
frozen ceiling therefore also reflects that blindness -- it is NOT a pure
feature-discovery ablation (for that, feed the readout the raw new-square bits;
see --add-newsq-bits).

Usage:
    python new_squares_j1b_frozen.py --condition-id 2 --schedule 250,1000 \
        --num-test-games 1500 --max-positions 20000
    python new_squares_j1b_frozen.py --condition-id 3 --schedule 250,500,1000,2000,5000,20000
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import train_streaming_probe as tsp
from opening_tree_mlp import playedeven_features
from new_squares_data import CONDITIONS, N_CELLS, N_NEW, score_manifest
from new_squares_j1b import replay_positions, get_data_pool


# ----------------------------------------------------------------------------
# Readout head.  The interpretable prob-OR head (new_squares_j1b.NewSquareProbOr)
# COLLAPSES on the wide, correlated frozen basis -- it fails to fit even the
# training data (train AUC 0.5 at any epoch count), producing a false "chance"
# result.  A plain per-new-square Linear->sigmoid head (what a regularized
# logistic probe is) fits fine and recovers the transfer signal (into-board
# legality test AUC ~0.82).  So the frozen experiment uses the linear head.
# ----------------------------------------------------------------------------

class NewSquareLinear(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, N_NEW)

    def forward(self, h):
        return torch.sigmoid(self.linear(h))          # (N, N_NEW)


def train_readout_linear(H_tr, LEGAL_tr, epochs=80, lr=0.02, batch=2048,
                         weight_decay=1e-3, device='cpu', seed=0, bias_init=-2.0):
    """Class-weighted BCE on the 8 per-new-square sigmoids, L2-regularized
    (weight_decay ~ the logistic probe's alpha).  H_tr may be a bool CPU tensor;
    batches are cast to float on the fly."""
    torch.manual_seed(seed)
    probe = NewSquareLinear(H_tr.shape[1]).to(device)
    if bias_init is not None:
        torch.nn.init.constant_(probe.linear.bias, float(bias_init))
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    L = torch.from_numpy(LEGAL_tr.astype(np.float32)).to(device)
    pos = L.sum(0); neg = L.shape[0] - pos
    pos_weight = (neg / pos.clamp(min=1)).clamp(max=1000)      # balanced per new square
    N = H_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            h = H_tr[idx].to(device=device, dtype=torch.float32)
            p = probe(h).clamp(1e-6, 1 - 1e-6)
            l = L[idx]
            loss = -(pos_weight * l * torch.log(p)
                     + (1.0 - l) * torch.log(1.0 - p)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def load_frozen_bank(bank, device):
    """Load the existing per-pattern bank as a FIXED OpeningTreeMLP hidden
    layer (tree leaves only; no flanking, no recency)."""
    W_tree, b_tree, meta = tsp.load_trees(bank)
    mlp = tsp.OpeningTreeMLP(W_tree, b_tree, meta, device)
    input_dim = W_tree.shape[1]
    H_dim = len(meta)
    print(f'frozen bank: {H_dim} tree-leaf units, input_dim={input_dim}', flush=True)
    return mlp, input_dim, H_dim


def build_training_data_frozen(games, rules, input_dim, add_newsq_bits=False,
                               max_positions=None, seed=0, min_prefix=1):
    """Replay games; per position return the 121-d BASE features (what the
    frozen trees consume) + new-square legality.  With add_newsq_bits, append
    16 raw new-square bits (played + placed-as-mover) so the readout can also
    see new-square state (the non-blinded variant)."""
    Xs, LEGALs = [], []
    for moves in games:
        for (plen, fire, legal) in replay_positions(moves, rules):
            if plen < min_prefix:
                continue
            x = playedeven_features(moves[:plen], canonicalize_mover=True).astype(np.float32)
            if add_newsq_bits:
                ext = np.zeros(2 * N_NEW, dtype=np.float32)
                mp = plen % 2
                for t, c in enumerate(moves[:plen]):
                    if c >= 64:
                        k = c - 64
                        ext[k] = 1.0
                        if (t % 2) == mp:
                            ext[N_NEW + k] = 1.0
                x = np.concatenate([x, ext])
            Xs.append(x)
            LEGALs.append(legal)
    X = np.stack(Xs).astype(np.float32)
    LEGAL = np.stack(LEGALs).astype(np.float32)
    assert X.shape[1] == input_dim, f'feat dim {X.shape[1]} != expected {input_dim}'
    if max_positions is not None and X.shape[0] > max_positions:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], max_positions, replace=False)
        X, LEGAL = X[idx], LEGAL[idx]
    return X, LEGAL


def frozen_hidden(mlp, X_np, device, batch=4096, add_newsq_bits=False):
    """Frozen tree-leaf activations (bool, N x H) on CPU.  With add_newsq_bits,
    the trailing 16 raw new-square bits are concatenated as extra hidden units
    (so the readout can attend to new-square state directly)."""
    base = X_np[:, :mlp.W.shape[1]] if add_newsq_bits else X_np
    x = torch.from_numpy(np.ascontiguousarray(base)).to(device)
    H = mlp(x, batch=batch, out_device='cpu', out_dtype=torch.bool)   # (N, H) bool
    del x
    if add_newsq_bits:
        ext = torch.from_numpy(X_np[:, mlp.W.shape[1]:]).bool()       # (N, 16)
        H = torch.cat([H, ext], dim=1)
    return H


def make_score_fn_frozen(mlp, probe, device, add_newsq_bits=False):
    probe.eval()

    def score_fn(prefix):
        scores = np.zeros(N_CELLS, dtype=np.float32)
        x121 = playedeven_features(prefix, canonicalize_mover=True).astype(np.float32)
        xt = torch.from_numpy(x121[None, :]).to(device)
        h = mlp(xt, out_device=device, out_dtype=torch.float32)       # (1, H)
        if add_newsq_bits:
            ext = np.zeros(2 * N_NEW, dtype=np.float32)
            mp = len(prefix) % 2
            for t, c in enumerate(prefix):
                if c >= 64:
                    k = c - 64
                    ext[k] = 1.0
                    if (t % 2) == mp:
                        ext[N_NEW + k] = 1.0
            h = torch.cat([h, torch.from_numpy(ext[None, :]).to(device)], dim=1)
        with torch.no_grad():
            p = probe(h)[0].cpu().numpy()
        scores[64:64 + N_NEW] = p
        return scores
    return score_fn


def fit_and_score(D, rules, games, manifest, mlp, feat_dim, H_dim, args, device):
    t0 = time.time()
    X, LEGAL = build_training_data_frozen(
        games[:D], rules, feat_dim, add_newsq_bits=args.add_newsq_bits,
        max_positions=args.max_positions, seed=args.seed)
    print(f'    D={D}: {X.shape[0]} positions; frozen hidden (H={H_dim}); '
          f'training readout...', flush=True)
    H = frozen_hidden(mlp, X, device, add_newsq_bits=args.add_newsq_bits)
    probe = train_readout_linear(H, LEGAL, epochs=args.readout_epochs,
                                 lr=args.readout_lr, weight_decay=args.readout_wd,
                                 device=device, seed=args.seed,
                                 bias_init=args.readout_bias_init)
    score_fn = make_score_fn_frozen(mlp, probe, device,
                                    add_newsq_bits=args.add_newsq_bits)
    m = score_manifest(score_fn, manifest, per_bucket=True)
    print(f'    D={D}: IL_frac={m["IL_prob_frac"]:.4f} '
          f'IL_per_tgt={m["IL_prob_per_target"]:.4f} '
          f'IL_acc={m["IL_acc"]:.4f}  ({time.time()-t0:.0f}s)', flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition-id', type=int, required=True, choices=[2, 3])
    ap.add_argument('--frozen-bank', default='banks/J1_perpattern.pt')
    ap.add_argument('--add-newsq-bits', action='store_true',
                    help='also give the readout the 16 raw new-square bits '
                         '(non-blinded variant: frozen BASE trees + new-square '
                         'inputs, isolating base feature-discovery only).')
    ap.add_argument('--output-dir', default='experiments/new_squares')
    ap.add_argument('--schedule', default='250,500,1000,2000,5000,20000')
    ap.add_argument('--num-test-games', type=int, default=5000)
    ap.add_argument('--n-test-positions', type=int, default=5000)
    ap.add_argument('--max-positions', type=int, default=50000)
    ap.add_argument('--readout-epochs', type=int, default=80)
    ap.add_argument('--readout-lr', type=float, default=0.02)
    ap.add_argument('--readout-wd', type=float, default=1e-3,
                    help='L2 weight decay for the linear readout (~logistic '
                         'probe alpha; controls overfitting on the wide basis).')
    ap.add_argument('--readout-bias-init', type=float, default=-2.0,
                    help='initial readout bias (negative -> start near the base '
                         'legality rate).')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cache-dir', default='experiments/new_squares/j1b_cache')
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != 'cuda'
                          or torch.cuda.is_available()) else 'cpu')
    schedule = [int(s) for s in args.schedule.split(',') if s.strip()]
    D_max = max(schedule)
    cond_name = CONDITIONS[args.condition_id]
    print(f'Condition {args.condition_id}: {cond_name}  '
          f'{"[+newsq-bits]" if args.add_newsq_bits else "[frozen trees only]"}',
          flush=True)
    print(f'Schedule: {schedule}   device={device}', flush=True)

    mlp, input_dim, H_dim = load_frozen_bank(args.frozen_bank, device)
    feat_dim = input_dim + (2 * N_NEW if args.add_newsq_bits else 0)
    hidden_dim = H_dim + (2 * N_NEW if args.add_newsq_bits else 0)

    rules, games, manifest = get_data_pool(
        args.condition_id, D_max, args.num_test_games, args.seed,
        args.n_test_positions, args.cache_dir)

    results = {
        'condition_id': args.condition_id, 'condition_name': cond_name,
        'model': 'J1B-frozen' + ('+newsq' if args.add_newsq_bits else ''),
        'frozen_bank': args.frozen_bank, 'add_newsq_bits': args.add_newsq_bits,
        'n_train_games': schedule, 'eval_steps': schedule,
        'H_dim': hidden_dim, 'max_positions': args.max_positions,
        'readout_epochs': args.readout_epochs, 'readout_lr': args.readout_lr,
        'seed': args.seed,
        'IL_prob': [], 'IL_prob_frac': [], 'IL_prob_per_target': [],
        'IL_acc': [], 'LL_prob': [], 'LL_acc': [], 'IL_n': [],
    }
    t0 = time.time()
    for D in schedule:
        m = fit_and_score(D, rules, games, manifest, mlp, feat_dim, H_dim,
                          args, device)
        for k in ('IL_prob', 'IL_prob_frac', 'IL_prob_per_target',
                  'IL_acc', 'LL_prob', 'LL_acc', 'IL_n'):
            results[k].append(m[k])
    results['elapsed_seconds'] = time.time() - t0

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = '_newsq' if args.add_newsq_bits else ''
    out = os.path.join(args.output_dir,
                       f'j1b_frozen{suffix}_cond_{args.condition_id:03d}.json')
    json.dump(results, open(out, 'w'), indent=2)
    print(f'\nSaved {out}  ({results["elapsed_seconds"]:.0f}s)', flush=True)
    for D, f, a in zip(schedule, results['IL_prob_frac'], results['IL_acc']):
        print(f'  D={D:6d}   IL_frac={f:.4f}   IL_acc={a:.4f}', flush=True)


if __name__ == '__main__':
    main()
