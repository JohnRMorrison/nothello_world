"""Regenerate legal_moves.pickle from existing games.pickle using multiprocessing."""
import pickle
import sys
import argparse
from multiprocessing import Pool
from generate_variant_games import VariantBoard


def process_game(args):
    variant, moves = args
    board = VariantBoard(variant)
    legal_per_turn = []
    for m in moves:
        legal_per_turn.append(board.get_valid_moves())
        board.make_move(m)
    return legal_per_turn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--games-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    games_path = f"{args.games_dir}/games.pickle"
    legal_path = f"{args.games_dir}/legal_moves.pickle"

    print(f"Loading games from {games_path}...")
    with open(games_path, "rb") as f:
        games = pickle.load(f)
    print(f"Loaded {len(games)} games")

    print(f"Processing with {args.workers} workers...")
    work = [(args.variant, g) for g in games]

    with Pool(args.workers) as pool:
        all_legal = []
        for i, result in enumerate(pool.imap(process_game, work, chunksize=1000)):
            all_legal.append(result)
            if (i + 1) % 100000 == 0:
                print(f"  {i + 1}/{len(games)}")

    print(f"Saving to {legal_path}...")
    with open(legal_path, "wb") as f:
        pickle.dump(all_legal, f)
    print("Done")


if __name__ == "__main__":
    main()
