import pickle, glob, os

chunks = sorted(glob.glob('axon_data_chunk*.pkl'))
print(f"Found {len(chunks)} chunks: {chunks}")

merged_data = {}
merged_moves = {}
total_games = 0

for path in chunks:
    #status message update bc Im lowkey scared of running out of memory and losing progress
    print(f"  Loading {path}...")
    with open(path, 'rb') as f:
        chunk = pickle.load(f)
    merged_data.update(chunk['game_data'])
    merged_moves.update(chunk['game_moves'])
    total_games += chunk['num_games']
    itos = chunk['itos']
    stoi = chunk['stoi']
    
print(f"Merged {total_games} games")

output = {
    'game_data': merged_data,
    'game_moves': merged_moves,
    'itos': itos,
    'stoi': stoi,
    'num_games': total_games,
}

out_file = 'axon_full_data.pkl'
with open(out_file, 'wb') as f:
    pickle.dump(output, f)

size_gb = os.path.getsize(out_file) / (1024**3)
print(f"Merged {total_games} games into {out_file} ({size_gb:.2f} GB)")


