"""Merge selected seeds from multiple multi-seed checkpoints into one.

Each --ckpt/--take pair pulls the first --take seeds from that checkpoint.
The output is a valid multi-seed checkpoint that can be fed to
train_multi_seed_readout.py or any other eval script.

Example — combine 5 seeds from chunks 0-9 and 5 from chunks 10-19 for a
10-seed "diverse" ensemble:

    python merge_multi_seed_ckpts.py \\
        --ckpt experiments/.../multi_seed_N100_H512_playedeven_chunks0-9.pt \\
        --take 5 \\
        --ckpt experiments/.../multi_seed_N100_H512_playedeven_chunks10-19_seed1.pt \\
        --take 5 \\
        --out experiments/.../multi_seed_N10_H512_playedeven_chunks-mix.pt
"""
import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, '.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True,
                    help='Multi-seed checkpoint path (repeatable).')
    ap.add_argument('--take', type=int, action='append', required=True,
                    help='Number of seeds to take from the paired --ckpt (repeatable).')
    ap.add_argument('--out', required=True,
                    help='Output merged checkpoint path.')
    args = ap.parse_args()

    if len(args.ckpt) != len(args.take):
        raise SystemExit(f"Number of --ckpt ({len(args.ckpt)}) must match "
                          f"number of --take ({len(args.take)}).")

    merged_seeds = []
    hidden_dim = input_dim = n_patterns = None
    per_ckpt_labels = []

    for ckpt_path, k in zip(args.ckpt, args.take):
        print(f"Loading {ckpt_path}  take={k}")
        c = torch.load(ckpt_path, map_location='cpu')
        n_avail = c['num_seeds']
        if k > n_avail:
            raise SystemExit(f"--take {k} exceeds N={n_avail} in {ckpt_path}")

        if hidden_dim is None:
            hidden_dim = c['hidden_dim']
            input_dim  = c['input_dim']
            n_patterns = c['n_patterns']
        else:
            for key, cur in (('hidden_dim', c['hidden_dim']),
                             ('input_dim',  c['input_dim']),
                             ('n_patterns', c['n_patterns'])):
                prev = {'hidden_dim': hidden_dim, 'input_dim': input_dim,
                        'n_patterns': n_patterns}[key]
                if prev != cur:
                    raise SystemExit(
                        f"{key} mismatch: {prev} vs {cur} in {ckpt_path}")

        merged_seeds.extend(copy.deepcopy(c['all_seeds'][:k]))
        per_ckpt_labels.append(f"{os.path.basename(ckpt_path)}[:{k}]")

    N = len(merged_seeds)
    print(f"Merged N={N} seeds  H={hidden_dim}  input_dim={input_dim}  "
          f"n_patterns={n_patterns}")

    out = {
        'all_seeds':  merged_seeds,
        'num_seeds':  N,
        'hidden_dim': hidden_dim,
        'input_dim':  input_dim,
        'n_patterns': n_patterns,
        'source_ckpts': per_ckpt_labels,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    torch.save(out, args.out)
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
