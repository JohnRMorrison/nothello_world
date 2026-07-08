"""Merge per-depth adversarial-record files into a single adversarial_records.npz."""
import argparse
import glob
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='experiment1_by_depth')
    ap.add_argument('--output-name', default='adversarial_records.npz')
    args = ap.parse_args()

    all_games, all_turns, all_cells = [], [], []
    files = sorted(glob.glob(os.path.join(args.input_dir,
                                            'adversarial_records_depth_*.npz')))
    print(f"Merging {len(files)} depth files:")
    for f in files:
        d = np.load(f, allow_pickle=True)
        n = len(d['games'])
        print(f"  {os.path.basename(f)}: {n} records")
        all_games.extend(list(d['games']))
        all_turns.extend(list(d['turns']))
        all_cells.extend(list(d['illegal_cells']))

    print(f"Total: {len(all_games)} adversarial records")
    out = os.path.join(args.input_dir, args.output_name)
    np.savez_compressed(
        out,
        games=np.array(all_games, dtype=object),
        turns=np.array(all_turns, dtype=np.int64),
        illegal_cells=np.array(all_cells, dtype=np.int64),
    )
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
