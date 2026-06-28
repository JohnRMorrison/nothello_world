"""Split a multi-seed checkpoint into N individual seed ckpts compatible
with compare_v4_vs_mlp.load_mlp() and compare_mlp_seeds_*.py.

Usage:
    python split_multi_seed_ckpt.py \\
        --multi-ckpt experiments/.../multi_seed_N100_H512_playedeven.pt \\
        --output-dir experiments/.../pattern_detector_checkpoints/multi_seed_split/
"""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi-ckpt', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed-tag', default='ms',
                    help='Filename tag for individual ckpts: ms0, ms1, ...')
    ap.add_argument('--limit', type=int, default=None,
                    help='If set, only extract first N seeds')
    args = ap.parse_args()

    print(f"Loading {args.multi_ckpt}")
    ckpt = torch.load(args.multi_ckpt, map_location='cpu')
    n_seeds = ckpt['num_seeds']
    hidden = ckpt['hidden_dim']
    input_dim = ckpt['input_dim']
    n_patterns = ckpt['n_patterns']
    print(f"  num_seeds={n_seeds}  hidden={hidden}  input={input_dim}")

    os.makedirs(args.output_dir, exist_ok=True)
    n_to_write = args.limit or n_seeds
    for s in range(n_to_write):
        seed_state = ckpt['all_seeds'][s]
        out = {
            'even': seed_state['even'],
            'odd': seed_state['odd'],
            'hidden_dim': hidden,
            'input_dim': input_dim,
            'n_patterns': n_patterns,
            'mode': 'direct',
            'epoch': ckpt.get('epoch', 0),
            'best_pat_acc': ckpt['eval_acc_per_seed'][s],
        }
        # Filename pattern matches load_mlp expectations
        out_path = os.path.join(
            args.output_dir,
            f"pattern_simple_direct_H{hidden}_playedeven_{args.seed_tag}{s}.pt",
        )
        torch.save(out, out_path)
    print(f"Wrote {n_to_write} individual ckpts to {args.output_dir}")


if __name__ == '__main__':
    main()
