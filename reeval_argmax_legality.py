"""Re-evaluate saved streaming-probe checkpoints with the argmax-legality
metric (fraction of positions where the model's top-1 move prediction is
in the ground-truth legal set), matching the metric your MLP is reported
under (98.5% legal accuracy).

Loads:
  - The probe checkpoint produced by train_streaming_probe.py (has
    'probe_state' and 'args').
  - The tree checkpoint the probe was built on (args['load_trees_from']).
  - A chunk_ext_*.npz file for evaluation positions.

Prints per-checkpoint:
  - argmax legality accuracy (per-cell prediction taken, argmax over
    64 cells, checked against ground-truth legal set).
  - position-perfect accuracy (all 64 cells binary-correct).
  - per-cell accuracy (baseline, matches streaming eval).
  - per-ply breakdown of argmax accuracy.

Usage:
  python reeval_argmax_legality.py \\
      --probe-ckpts ckpts_midgame/stream_strupo_g6000000_*.pt \\
      [--chunk-path experiments/.../feature_chunks/chunk_ext_0039.npz] \\
      [--max-positions 500000] [--ply-min 10] [--ply-max 50]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from opening_tree_mlp import (
    OpeningTreeMLP, PatternProbOrHead, LinearPatternProbOr, BOARD_CELLS
)
from flanking_patterns import load_patterns
from train_streaming_probe import (
    load_trees, process_chunk_ext_file, build_hidden_layer_batch
)


@torch.no_grad()
def evaluate_probe(probe, X, L, mlp, patterns, recent_Ks, use_relu, device,
                     batch=1024):
    """Return (argmax_acc, position_perfect_acc, per_cell_acc, per_ply_dict).

    argmax_acc:            frac of positions where argmax(probs) in legal set
    position_perfect_acc:  frac of positions where ALL 64 predictions correct
    per_cell_acc:          frac of (pos, cell) pairs correct at threshold 0.5
    """
    N = X.shape[0]
    total_argmax_hits = 0
    total_pos_perfect = 0
    total_percell_correct = 0
    total_positions = 0
    total_cells = 0
    per_ply_hits = {}
    per_ply_n = {}

    for i in range(0, N, batch):
        X_batch = X[i:i + batch]
        L_batch = L[i:i + batch]
        H = build_hidden_layer_batch(X_batch, mlp, patterns, recent_Ks,
                                          use_relu, device)
        probs = probe(H.float() if not use_relu else H)  # (B, 64) float
        probs_np = probs.cpu().numpy()
        argmax_cells = probs_np.argmax(axis=1)             # (B,)
        # argmax legality
        for j in range(X_batch.shape[0]):
            hit = int(L_batch[j, argmax_cells[j]] > 0)
            total_argmax_hits += hit
        # position-perfect
        preds_binary = (probs_np > 0.5).astype(np.uint8)
        pos_correct = (preds_binary == L_batch).all(axis=1)
        total_pos_perfect += int(pos_correct.sum())
        # per-cell
        total_percell_correct += int((preds_binary == L_batch).sum())
        total_positions += X_batch.shape[0]
        total_cells += X_batch.shape[0] * BOARD_CELLS
    return {
        'argmax_acc': total_argmax_hits / total_positions,
        'pos_perfect_acc': total_pos_perfect / total_positions,
        'per_cell_acc': total_percell_correct / total_cells,
        'n_positions': total_positions,
    }


@torch.no_grad()
def evaluate_probe_with_ply(probe, X, L, T, mlp, patterns, recent_Ks,
                                use_relu, device, batch=1024):
    """Same as evaluate_probe but also returns per-ply-bucket argmax acc."""
    N = X.shape[0]
    total_argmax_hits = 0
    total_pos_perfect = 0
    total_percell_correct = 0
    total_positions = 0
    total_cells = 0
    per_ply_hits = {}
    per_ply_n = {}

    for i in range(0, N, batch):
        X_batch = X[i:i + batch]
        L_batch = L[i:i + batch]
        T_batch = T[i:i + batch]
        H = build_hidden_layer_batch(X_batch, mlp, patterns, recent_Ks,
                                          use_relu, device)
        Hf = H.float() if not use_relu else H
        if isinstance(probe, (list, tuple)):        # ensemble: average heads
            probs = torch.stack([pr(Hf) for pr in probe]).mean(0)
        else:
            probs = probe(Hf)
        probs_np = probs.cpu().numpy()
        argmax_cells = probs_np.argmax(axis=1)
        for j in range(X_batch.shape[0]):
            hit = int(L_batch[j, argmax_cells[j]] > 0)
            total_argmax_hits += hit
            bucket = int(T_batch[j]) // 10 * 10
            per_ply_hits[bucket] = per_ply_hits.get(bucket, 0) + hit
            per_ply_n[bucket] = per_ply_n.get(bucket, 0) + 1
        preds_binary = (probs_np > 0.5).astype(np.uint8)
        pos_correct = (preds_binary == L_batch).all(axis=1)
        total_pos_perfect += int(pos_correct.sum())
        total_percell_correct += int((preds_binary == L_batch).sum())
        total_positions += X_batch.shape[0]
        total_cells += X_batch.shape[0] * BOARD_CELLS
    return {
        'argmax_acc': total_argmax_hits / total_positions,
        'pos_perfect_acc': total_pos_perfect / total_positions,
        'per_cell_acc': total_percell_correct / total_cells,
        'n_positions': total_positions,
        'ply_argmax': {b: per_ply_hits[b] / per_ply_n[b]
                        for b in sorted(per_ply_hits)},
        'ply_n': {b: per_ply_n[b] for b in sorted(per_ply_n)},
    }


def load_probe(probe_ckpt_path, device):
    """Load probe from checkpoint, reconstructing tree MLP + probe head."""
    ck = torch.load(probe_ckpt_path, map_location=device)
    saved_args = ck['args']
    tree_path = saved_args['load_trees_from']
    if not os.path.isabs(tree_path):
        tree_path = os.path.join(REPO_ROOT, tree_path)
    W, b, meta = load_trees(tree_path)
    mlp = OpeningTreeMLP(W, b, meta, device)
    patterns = load_patterns(os.path.join(
        REPO_ROOT, saved_args.get('flanking_patterns',
                                      'hand_crafted_flanking_patterns.pt')))
    hidden_dim = W.shape[0] + len(patterns)   # tree paths + 960 flanking
    probe_type = saved_args.get('probe_type', 'strupo')
    def _build():
        if probe_type == 'linpo':
            return LinearPatternProbOr(hidden_dim, patterns).to(device)
        return PatternProbOrHead(meta, patterns).to(device)
    # Ensemble: a multi-seed checkpoint stores 'probe_states' (list); a
    # single-seed one stores 'probe_state'.
    states = ck.get('probe_states', [ck['probe_state']])
    probes = []
    for st in states:
        pr = _build()
        pr.load_state_dict(st)
        pr.eval()
        probes.append(pr)
    if len(probes) > 1:
        print(f'  ensembling {len(probes)} probe heads')
    return {
        'probes': probes,
        'probe': probes[0],
        'mlp': mlp,
        'patterns': patterns,
        'saved_args': saved_args,
        'tree_path': tree_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe-ckpts', nargs='+', required=True,
                    help='Glob or list of probe checkpoint files.')
    ap.add_argument('--chunk-path',
                    default=('experiments/mathematical_transformation_experiments/'
                             'heuristic_probe_results/feature_chunks/chunk_ext_0039.npz'))
    ap.add_argument('--max-positions', type=int, default=500_000)
    ap.add_argument('--ply-min', type=int, default=10)
    ap.add_argument('--ply-max', type=int, default=50)
    ap.add_argument('--canonicalize-mover', action='store_true',
                    help='Override -- otherwise detected from probe saved_args.')
    args = ap.parse_args()

    # Expand globs.
    paths = []
    for p in args.probe_ckpts:
        expanded = sorted(glob.glob(p))
        if expanded:
            paths.extend(expanded)
        elif os.path.exists(p):
            paths.append(p)
    print(f'{len(paths)} probe checkpoint(s) to evaluate.\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load chunk once (identical for all probes)
    print(f'Loading eval chunk {args.chunk_path}...')
    # We need to load with canonicalize consistent with the FIRST probe,
    # but if probes differ (canonical vs absolute) we need to reload.
    loaded_can = None
    X_cache = None

    print(f'{"ckpt":90s}  {"argmax":>7}  {"posP":>6}  {"per-cell":>8}  {"N":>6}')
    print('-' * 130)
    for path in paths:
        try:
            info = load_probe(path, device)
        except Exception as e:
            print(f'{os.path.basename(path):90s}  FAILED: {e}')
            continue
        can_flag = (args.canonicalize_mover
                    or info['saved_args'].get('canonicalize_mover', False))
        if loaded_can != can_flag:
            X, _, T, L = process_chunk_ext_file(
                args.chunk_path, args.ply_min, args.ply_max,
                canonicalize_mover=can_flag,
                max_positions=args.max_positions,
            )
            X_cache = (X, T, L)
            loaded_can = can_flag
        X, T, L = X_cache
        recent_Ks = None
        use_relu = info['saved_args'].get('use_relu', False)
        res = evaluate_probe_with_ply(
            info['probes'], X, L, T, info['mlp'], info['patterns'],
            recent_Ks, use_relu, device,
            batch=info['saved_args'].get('batch_size', 2048),
        )
        name = os.path.basename(path)
        tree_name = os.path.basename(info['tree_path'])
        print(f'{name:90s}  {100*res["argmax_acc"]:6.2f}%  '
               f'{100*res["pos_perfect_acc"]:5.2f}%  '
               f'{100*res["per_cell_acc"]:7.4f}%  '
               f'{res["n_positions"]:>6}')
        print(f'  tree: {tree_name}')
        print(f'  ply argmax:', end=' ')
        for b in sorted(res['ply_argmax']):
            print(f'[{b:2d}-{b+9})={100*res["ply_argmax"][b]:5.1f}% (n={res["ply_n"][b]})',
                   end=' ')
        print()


if __name__ == '__main__':
    main()
