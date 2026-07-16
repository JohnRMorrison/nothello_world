"""Phase 2 of the ambiguity analysis: load the precomputed consistent-board
pickle from phase 1 and run fast per-H MLP+probe evaluation.

Writes one CSV row per (game_idx, distinct_board, H).  Same column set as
consistent_board_multi_H.py.

Usage:
    python consistent_board_phase2_eval.py \\
        --boards-pkl consist_boards_k25_N300.pkl \\
        --mlp-ckpts $BASE/pattern_simple_direct_H512_playedeven.pt \\
                    $BASE/pattern_simple_direct_H1024_playedeven.pt \\
                    $BASE/pattern_simple_direct_H2048_playedeven.pt \\
                    $BASE/pattern_simple_direct_H4096_playedeven.pt \\
                    $BASE/pattern_simple_direct_H8192_playedeven.pt \\
        --probe-ckpts $BASE/probe_direct_H512_playedeven.pt \\
                      $BASE/probe_direct_H1024_playedeven.pt \\
                      $BASE/probe_direct_H2048_playedeven.pt \\
                      $BASE/probe_direct_H4096_playedeven.pt \\
                      $BASE/probe_direct_H8192_playedeven.pt \\
        --output-csv consist_board_k25_all_H.csv
"""
import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')
from train_pattern_simple import _get_cell_pat_index
from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
from compare_v4_vs_mlp import played_even_features

from consistent_board_saved_probe import (
    legal_from_state,
    board_state_target_from_state,
    load_mlp_and_probe,
    forward_hidden_scores_probe,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boards-pkl', required=True,
                    help='Phase 1 output pickle.')
    ap.add_argument('--mlp-ckpts', nargs='+', required=True)
    ap.add_argument('--probe-ckpts', nargs='+', required=True)
    ap.add_argument('--output-csv', required=True)
    ap.add_argument('--per-cell-npz-prefix', default=None,
                    help='If set, save per-cell hit accumulators and per-moveset '
                         'Jaccard stats to <prefix>_H<H>.npz, one file per H.')
    args = ap.parse_args()

    if len(args.mlp_ckpts) != len(args.probe_ckpts):
        raise SystemExit(f"{len(args.mlp_ckpts)} MLP ckpts vs "
                          f"{len(args.probe_ckpts)} probe ckpts")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading boards pickle {args.boards_pkl}...")
    with open(args.boards_pkl, 'rb') as f:
        data = pickle.load(f)
    records = data['records']
    k = data['k']
    print(f"  k={k}, {len(records)} records, "
          f"{data.get('n_with_ambiguity', '?')} ambiguous")

    triples = []
    for mlp_p, probe_p in zip(args.mlp_ckpts, args.probe_ckpts):
        print(f"\nLoading pair:")
        print(f"  MLP:   {mlp_p}")
        print(f"  Probe: {probe_p}")
        models, probes, H = load_mlp_and_probe(
            mlp_p, probe_p, input_dim=120, device=device)
        triples.append((H, models, probes))
    print(f"\nLoaded {len(triples)} H values: {[t[0] for t in triples]}")

    patterns = enumerate_flanking_patterns()
    pattern_to_cell = torch.tensor(
        [MOVE_TO_IDX[p['target']] for p in patterns],
        dtype=torch.long, device=device,
    )
    idx, mask = _get_cell_pat_index(pattern_to_cell, 60)

    print(f"\nEvaluating {len(records)} records × {len(triples)} Hs...")
    n_written = 0
    n_processed = 0
    t0 = time.time()

    # Per-H per-cell hit accumulators and per-moveset Jaccard stats.
    per_H_stats = {}
    for H, _, _ in triples:
        per_H_stats[H] = {
            'hits_uw': np.zeros(64, dtype=np.int64),
            'hits_w': np.zeros(64, dtype=np.int64),
            'n_uw': 0,
            'n_w': 0,
            'jac_mean': [], 'jac_std': [],
            'jac_min': [], 'jac_max': [],
            'jac_multi': [], 'n_boards': [],
            # For each moveset, cell -> count of boards where probe wrong.
            # Stored as (n_movesets, 64) after np.stack in the save block.
            'per_cell_wrong_counts': [],
        }

    with open(args.output_csv, 'w', newline='') as f_out:
        w = csv.writer(f_out)
        w.writerow([
            'game_idx', 'k', 'H', 'board_hash', 'board_count',
            'n_distinct_boards',
            'is_training_board',
            'top1_cell_60', 'top1_is_legal',
            'probe_cell_accuracy',
        ])

        for rec in records:
            g_idx = rec['game_idx']
            prefix = rec['prefix']
            boards = rec['boards']
            n_distinct = rec['n_distinct_boards']
            training_hash = rec['training_hash']
            n_processed += 1

            if n_distinct < 2:
                continue

            feats = played_even_features(prefix)                    # (120,)

            # Gather board targets and counts once per record (H-independent).
            board_targets_all = np.stack([
                board_state_target_from_state(s) for s, _, _ in boards.values()
            ])                                                                  # (M, 64)
            board_counts_arr = np.array(
                [c for _, _, c in boards.values()], dtype=np.int64)             # (M,)

            for H, models, probes in triples:
                cell_scores, probe_out = forward_hidden_scores_probe(
                    feats, k, models, probes, idx, mask, device)
                top1_60 = int(cell_scores.argmax().item())
                probe_argmax = probe_out.argmax(dim=-1).cpu().numpy()

                # Per-cell hit + Jaccard tracking for this (record, H)
                s = per_H_stats[H]
                error_mat = (probe_argmax[None, :] != board_targets_all)        # (M, 64)
                hits_row = (~error_mat).astype(np.int64)                        # (M, 64)
                s['hits_uw'] += hits_row.sum(axis=0)
                s['hits_w'] += (hits_row * board_counts_arr[:, None]).sum(axis=0)
                s['n_uw'] += error_mat.shape[0]
                s['n_w'] += int(board_counts_arr.sum())
                M = error_mat.shape[0]
                if M >= 2:
                    # per-cell "how many boards is probe wrong on" for this moveset
                    s['per_cell_wrong_counts'].append(
                        error_mat.sum(axis=0).astype(np.int64))
                    pair_vals = []
                    for i_ in range(M):
                        for j_ in range(i_ + 1, M):
                            inter = int((error_mat[i_] & error_mat[j_]).sum())
                            union = int((error_mat[i_] | error_mat[j_]).sum())
                            j_val = 1.0 if union == 0 else inter / union
                            pair_vals.append(j_val)
                    arr = np.asarray(pair_vals)
                    s['jac_mean'].append(float(arr.mean()))
                    s['jac_std'].append(float(arr.std()))
                    s['jac_min'].append(float(arr.min()))
                    s['jac_max'].append(float(arr.max()))
                    all_and = int(np.all(error_mat, axis=0).sum())
                    all_or = int(np.any(error_mat, axis=0).sum())
                    s['jac_multi'].append(1.0 if all_or == 0
                                           else all_and / all_or)
                    s['n_boards'].append(M)

                for b_hash, (state, next_c, count) in boards.items():
                    legal_set = legal_from_state(state, next_c)
                    top1_legal = int(top1_60 in legal_set)
                    board_target = board_state_target_from_state(state)
                    probe_acc = float((probe_argmax == board_target).mean())
                    w.writerow([
                        g_idx, k, H, b_hash.hex()[:16],
                        count, n_distinct,
                        int(b_hash == training_hash),
                        top1_60, top1_legal,
                        f"{probe_acc:.4f}",
                    ])
                    n_written += 1

            if n_processed % 25 == 0:
                print(f"  {n_processed}/{len(records)}  "
                      f"rows: {n_written}  "
                      f"({int(time.time()-t0)}s)", flush=True)
                f_out.flush()

    print()
    print(f"Done.  Wrote {n_written} rows over "
          f"{n_processed} records × {len(triples)} Hs.")
    print(f"Output: {args.output_csv}")

    if args.per_cell_npz_prefix:
        for H, s in per_H_stats.items():
            outp = f'{args.per_cell_npz_prefix}_H{H}.npz'
            wrong_counts = (np.stack(s['per_cell_wrong_counts'])
                             if s['per_cell_wrong_counts']
                             else np.zeros((0, 64), dtype=np.int64))
            np.savez(outp,
                      hits_uw=s['hits_uw'],
                      hits_w=s['hits_w'],
                      n_uw=int(s['n_uw']),
                      n_w=int(s['n_w']),
                      moveset_jaccard=np.array(s['jac_mean'], dtype=np.float64),
                      moveset_jac_std=np.array(s['jac_std'], dtype=np.float64),
                      moveset_jac_min=np.array(s['jac_min'], dtype=np.float64),
                      moveset_jac_max=np.array(s['jac_max'], dtype=np.float64),
                      moveset_jac_multi=np.array(s['jac_multi'], dtype=np.float64),
                      moveset_n_boards=np.array(s['n_boards'], dtype=np.int64),
                      per_cell_wrong_counts=wrong_counts,
                      k=k, hidden_dim=int(H), N=1)
            print(f"  {outp}  "
                  f"(pooled n_uw={s['n_uw']}; movesets={len(s['jac_mean'])})")


if __name__ == '__main__':
    main()
