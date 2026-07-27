#!/usr/bin/env python
"""Reeval a saved board-state decoder for per-SQUARE and per-MOVE accuracy,
writing npz files in the presentation_boards.ipynb format:

  <out-prefix>_by_cell.npz : acc (8,8) in [0,1], n_games, pos_start, pos_end
  <out-prefix>_by_turn.npz : turns, per_turn_overall, per_turn_n, n_games

Mirrors analyze_nanda_probe_per_cell.py / plot_ogpt_overall_by_turn.py for the
transformer probe, but for our tree-hidden-layer decoders (--task state).

  # J1 (per-pattern; mlp hidden layer):
  python reeval_state_decoder.py \
    --decoder stream_out/J1_state_decoder.pt \
    --load-trees-from banks/J1_perpattern.pt --no-flanking --canonicalize-mover \
    --chunk-dir /workspace/feature_chunks --cache-dir /workspace/chunk_cache \
    --out-prefix notebooks/talk_data/j1_probe_accuracy

  # J3 (ordinal; leaf-index cache):
  python reeval_state_decoder.py \
    --decoder stream_out/J3_state_decoder.pt \
    --load-trees-from banks/J3_ordinal.pt --no-flanking --canonicalize-mover \
    --leaf-index-cache-dir /workspace/leaf_cache \
    --chunk-dir /workspace/feature_chunks --cache-dir /workspace/chunk_cache \
    --out-prefix notebooks/talk_data/j3_probe_accuracy
"""
import argparse, glob, os
from collections import defaultdict
import numpy as np
import torch

import train_streaming_probe as tsp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decoder', required=True,
                    help='saved --task state checkpoint (has "state_probe")')
    ap.add_argument('--load-trees-from', required=True)
    ap.add_argument('--no-flanking', action='store_true')
    ap.add_argument('--leaf-index-cache-dir', default=None)
    ap.add_argument('--chunk-dir', default='/workspace/feature_chunks')
    ap.add_argument('--cache-dir', default='/workspace/chunk_cache')
    ap.add_argument('--canonicalize-mover', action='store_true')
    ap.add_argument('--ply-min', type=int, default=5)
    ap.add_argument('--ply-max', type=int, default=54)
    ap.add_argument('--max-positions', type=int, default=500_000)
    ap.add_argument('--flanking-patterns',
                    default='hand_crafted_flanking_patterns.pt')
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--eval-chunk', default=None,
                    help='specific chunk_ext file; default = last (held-out)')
    ap.add_argument('--out-prefix', required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # --- saved 2-mode state probe (2, hidden, 64, 3) ---
    ck = torch.load(args.decoder, map_location=device, weights_only=False)
    W = ck['state_probe'].to(device)
    print(f'decoder {args.decoder}: state_probe {tuple(W.shape)}')

    # --- tree hidden layer, mirroring train_streaming_probe main() ---
    W_tree, b_tree, tree_meta = tsp.load_trees(args.load_trees_from)
    mlp = tsp.OpeningTreeMLP(W_tree, b_tree, tree_meta, device)
    leaf_build = tsp.load_leaf_build(args.load_trees_from)
    needs_ordinal = leaf_build is not None
    leaf_index = None
    if args.leaf_index_cache_dir and needs_ordinal:
        cm = np.load(os.path.join(args.leaf_index_cache_dir, 'colmap.npz'))
        leaf_index = (cm['col_tree_idx'], cm['col_nid'])
        print(f'  leaf-index cache: {len(leaf_index[0])} H cols')
    patterns = tsp.load_patterns(args.flanking_patterns)

    files = sorted(glob.glob(os.path.join(args.chunk_dir, 'chunk_ext_*.npz')))
    eval_path = args.eval_chunk or files[-1]
    print(f'eval chunk: {os.path.basename(eval_path)}  '
           f'(cap {args.max_positions})')

    if leaf_index is not None:
        X, S, T, L = tsp.load_leaf_index_chunk(
            args.leaf_index_cache_dir, eval_path, args.ply_min, args.ply_max,
            args.max_positions)
    else:
        X, S, T, L = tsp.load_chunk_cached(
            eval_path, args.ply_min, args.ply_max, args.canonicalize_mover,
            args.max_positions, needs_ordinal, cache_dir=args.cache_dir)
    N = X.shape[0]

    cell_correct = np.zeros(64, dtype=np.int64)
    n_pos = 0
    turn_correct = defaultdict(int)
    turn_n = defaultdict(int)      # positions per turn (×64 = cell predictions)
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            H = tsp.build_hidden_layer_batch(
                X[i:i + args.batch_size], mlp, patterns, None, False, device,
                no_flanking=args.no_flanking, leaf_build=leaf_build,
                leaf_index=leaf_index)
            if H.dtype != torch.float32:
                H = H.float()
            Tb_np = T[i:i + args.batch_size]
            Tb = torch.from_numpy(Tb_np.astype(np.int64)).to(device)
            Sb = torch.from_numpy(
                np.ascontiguousarray(S[i:i + args.batch_size]).astype(np.int64)
            ).to(device)
            preds = tsp._state_preds(W, H, Tb)          # (b, 64)
            correct = (preds == Sb)                     # (b, 64) bool
            cell_correct += correct.sum(0).cpu().numpy().astype(np.int64)
            n_pos += correct.shape[0]
            per_pos = correct.sum(1).cpu().numpy()       # (b,) cells right / pos
            for j in range(correct.shape[0]):
                t = int(Tb_np[j])
                turn_correct[t] += int(per_pos[j])
                turn_n[t] += 1

    # --- by square: (8, 8), row-major = OthelloBoardState.state.flatten() ---
    acc_cell = (cell_correct / max(n_pos, 1)).reshape(8, 8)
    # --- by turn: one point per ply ---
    turns = np.array(sorted(turn_correct), dtype=np.int64)
    per_turn_overall = np.array(
        [turn_correct[t] / (turn_n[t] * 64) for t in turns])
    per_turn_n = np.array([turn_n[t] for t in turns], dtype=np.int64)

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    cell_out = args.out_prefix + '_by_cell.npz'
    turn_out = args.out_prefix + '_by_turn.npz'
    np.savez(cell_out, acc=acc_cell, n_games=n_pos,
             pos_start=args.ply_min, pos_end=args.ply_max)
    np.savez(turn_out, turns=turns, per_turn_overall=per_turn_overall,
             per_turn_n=per_turn_n, n_games=n_pos)

    print(f'\noverall per-cell acc: {100*acc_cell.mean():.4f}%  '
           f'(N={n_pos} positions)')
    print(f'by-square -> {cell_out}   '
           f'(min {100*acc_cell.min():.2f}%  max {100*acc_cell.max():.2f}%)')
    print(f'by-turn   -> {turn_out}   '
           f'(turns {turns[0]}..{turns[-1]}, '
           f'acc {100*per_turn_overall.min():.2f}%..'
           f'{100*per_turn_overall.max():.2f}%)')
    print('\nNOTE: n_games is actually the POSITION count; the eval spans one '
           'position per game only if the chunk is de-duplicated.')


if __name__ == '__main__':
    main()
