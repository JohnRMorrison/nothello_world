"""Generate synthetic Othello games locally."""
import sys, os, importlib.util, types, pickle, time, random

sys.modules['pgn'] = types.ModuleType('pgn')
spec = importlib.util.spec_from_file_location('othello_module', 'data/othello.py', submodule_search_locations=[])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
OthelloBoardState = mod.OthelloBoardState

def random_game():
    board = OthelloBoardState()
    moves = []
    for _ in range(60):
        valid = board.get_valid_moves()
        if not valid:
            break
        move = random.choice(valid)
        board.umpire(move)
        moves.append(move)
    return moves

out_dir = 'data/othello_synthetic'
os.makedirs(out_dir, exist_ok=True)

total = 100000
batch_size = 5000
start = time.time()
for batch_i in range(total // batch_size):
    games = []
    while len(games) < batch_size:
        g = random_game()
        if len(g) == 60:
            games.append(g)
    fname = os.path.join(out_dir, f'gen_local_{batch_i:03d}.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(games, f)
    elapsed = time.time() - start
    done = (batch_i + 1) * batch_size
    rate = done / elapsed
    eta = (total - done) / rate
    print(f'Batch {batch_i}: {done}/{total} games ({rate:.0f}/s, ETA {eta:.0f}s)', flush=True)
print(f'Done in {time.time()-start:.0f}s')
