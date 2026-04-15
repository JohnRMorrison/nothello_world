"""Combine multiple pickle files in a directory into a single games.pickle."""

import argparse
import os
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory with .pickle files")
    parser.add_argument("--output", default=None, help="Output path (default: input_dir/games.pickle)")
    args = parser.parse_args()

    output = args.output or os.path.join(args.input_dir, "games.pickle")

    games = []
    for f in sorted(os.listdir(args.input_dir)):
        if f.endswith(".pickle") and f != "games.pickle":
            path = os.path.join(args.input_dir, f)
            with open(path, "rb") as fh:
                batch = pickle.load(fh)
            print(f"{f}: {len(batch)} games")
            games.extend(batch)

    with open(output, "wb") as fh:
        pickle.dump(games, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nCombined {len(games)} games -> {output}")

if __name__ == "__main__":
    main()
