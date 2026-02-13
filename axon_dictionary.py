"""
This file will collect model probabilities, legal and illegal moves for every position in every championship game. 

We then save to a pkl file(no JSON bc file size).

Also saves game move sequences separately for recreating the board states.

Output -> (axon_full_data.pkl):
In this structure:
{
    game_idx: {
        move_idx: {
            token_idx: { 'prob': float, 'legal': bool }
            for token_idx in range(61)
        }
    }
}
For reference: 
    gi = game index (0 to num_games-1)
    mi = move index (0 to num_moves-1)
    ti = token index (0 to 60, where 0 is the special -100 token for padding/illegal moves)
    
"""

import sys, os

# base directory where ckpts/, data/, mingpt/ live
BASE_DIR = "/engram/nklab/jrm2182/nothello_world"
sys.path.insert(0, BASE_DIR)

import torch
import torch.nn.functional as F
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState, permit
import glob, re, pickle, time

#token index and board position mappings (from mech_interp_othello_utils.py)
itos = {
    0: -100, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18,
    20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 26: 25, 27: 26, 28: 29,
    29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 37, 35: 38, 36: 39, 37: 40,
    38: 41, 39: 42, 40: 43, 41: 44, 42: 45, 43: 46, 44: 47, 45: 48, 46: 49,
    47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 56, 54: 57, 55: 58,
    56: 59, 57: 60, 58: 61, 59: 62, 60: 63,
}
stoi = {v: k for k, v in itos.items() if v != -100}
stoi[-100] = 0
stoi[-1] = 0


def parse_pgn_file(filepath):
    games = []
    with open(filepath, 'r') as f:
        content = f.read()
    for block in re.split(r'\[Event', content)[1:]:
        moves_match = re.search(r'\n((?:\d+\.\s+\w+\s+\w+\s*)+)', block)
        if moves_match:
            moves = re.findall(r'[A-H][1-8]', moves_match.group(1))
            positions = [permit(m) for m in moves]
            positions = [p for p in positions if p != -1]
            if len(positions) >= 10:
                games.append(positions)
    return games


def load_all_games(data_dir):
    pgn_files = sorted(glob.glob(f"{data_dir}/othello_championship/*.pgn"))
    all_games = []
    for i, f in enumerate(pgn_files):
        all_games.extend(parse_pgn_file(f))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pgn_files)} files, {len(all_games)} games")
    print(f"Loaded {len(all_games)} total games")
    return all_games


def collect_data(model, games, device):
    game_data = {}
    game_moves = {}
    total_pos = 0
    t0 = time.time()

    for gi, game in enumerate(games):
        board = OthelloBoardState()
        positions = {}

        for mi, move in enumerate(game):
            if mi < 4:
                board.update([move])
                continue

            valid = board.get_valid_moves()
            tokens = [stoi[m] for m in game[:mi]]
            inp = torch.tensor([tokens]).long().to(device)

            with torch.no_grad():
                logits, _ = model(inp)
            probs = F.softmax(logits[0, -1, :], dim=-1)

            pos_data = {}
            for ti in range(61):
                bp = itos.get(ti, -100)
                pos_data[ti] = {
                    'prob': probs[ti].item(),
                    'legal': (bp in valid) if bp != -100 else False
                }

            positions[mi] = pos_data
            total_pos += 1
            board.update([move])

        game_data[gi] = positions
        game_moves[gi] = game

        if (gi + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (gi + 1) / elapsed
            eta = (len(games) - gi - 1) / rate
            print(f"  {gi+1}/{len(games)} games, {total_pos:,} positions, "
                  f"{elapsed/60:.1f}m elapsed, ~{eta/60:.1f}m left")

    print(f"Done: {len(games)} games, {total_pos:,} positions, {(time.time()-t0)/60:.1f}m")
    return game_data, game_moves


if __name__ == "__main__":
    import sys

    # usage: python axon_dictionary.py <chunk_id> <total_chunks>
    # example: axon_dictionary.py 0 5   (runs first 1/5 of games)
    # or just: python axon_dictionary.py       (runs all games)
    if len(sys.argv) == 3:
        chunk_id = int(sys.argv[1])
        total_chunks = int(sys.argv[2])
    else:
        chunk_id = 0
        total_chunks = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    #load model
    mconf = GPTConfig(vocab_size=61, block_size=59, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    model.load_state_dict(torch.load(f"{BASE_DIR}/ckpts/gpt_championship.ckpt", map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded")

    #load games
    all_games = load_all_games(f"{BASE_DIR}/data")

    # split into chunks
    chunk_size = len(all_games) // total_chunks
    start = chunk_id * chunk_size
    end = len(all_games) if chunk_id == total_chunks - 1 else start + chunk_size
    games = all_games[start:end]
    print(f"Chunk {chunk_id}/{total_chunks}: games {start} to {end} ({len(games)} games)")

    #run and collect everything
    game_data, game_moves = collect_data(model, games, device)

    # remap keys back to original game indices
    remapped_data = {}
    remapped_moves = {}
    for gi, orig_idx in enumerate(range(start, start + len(games))):
        remapped_data[orig_idx] = game_data[gi]
        remapped_moves[orig_idx] = game_moves[gi]

    #save
    output = {
        'game_data': remapped_data,
        'game_moves': remapped_moves,
        'itos': itos,
        'stoi': stoi,
        'num_games': len(games),
    }

    out_file = f'{BASE_DIR}/axon_data_chunk{chunk_id}.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(output, f)

    size_gb = os.path.getsize(out_file) / (1024**3)
    print(f"Saved {out_file} ({size_gb:.2f} GB)")
