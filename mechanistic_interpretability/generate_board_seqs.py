"""Generate board_seqs_string.pth and board_seqs_int.pth from game data.

Run from the repo root:
    python mechanistic_interpretability/generate_board_seqs.py
"""
import os
import sys
import torch

repo_root = os.path.join(os.path.dirname(__file__), "..")
repo_root = os.path.abspath(repo_root)
sys.path.insert(0, repo_root)

# get_othello() looks for ./data/ relative to CWD, so chdir to repo root
os.chdir(repo_root)

from mingpt.dataset import CharDataset
from data import get_othello

print("Loading game data...")
othello = get_othello(ood_num=-1, data_root=None, wthor=True)
train_dataset = CharDataset(othello)

full_seqs = list(filter(lambda x: len(x) == 60, train_dataset.data.sequences))
print(f"Found {len(full_seqs)} full games")

board_seqs_string = torch.tensor(full_seqs)

# Convert to model token indices
board_seqs_int = board_seqs_string.clone()
board_seqs_int[board_seqs_string < 29] += 1
board_seqs_int[(board_seqs_string >= 29) & (board_seqs_string <= 34)] -= 1
board_seqs_int[board_seqs_string > 34] -= 3

# Shuffle
indices = torch.randperm(len(board_seqs_int))
board_seqs_int = board_seqs_int[indices]
board_seqs_string = board_seqs_string[indices]

out_dir = os.path.join(repo_root, "mechanistic_interpretability")
torch.save(board_seqs_int, os.path.join(out_dir, "board_seqs_int.pth"))
torch.save(board_seqs_string, os.path.join(out_dir, "board_seqs_string.pth"))
print(f"Saved board_seqs_int.pth and board_seqs_string.pth ({board_seqs_string.shape})")
