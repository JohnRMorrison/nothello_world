"""Generate board_seqs_string.pth and board_seqs_int.pth from game data.

Run from mechanistic_interpretability/ directory:
    python generate_board_seqs.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

torch.save(board_seqs_int, "board_seqs_int.pth")
torch.save(board_seqs_string, "board_seqs_string.pth")
print(f"Saved board_seqs_int.pth and board_seqs_string.pth ({board_seqs_string.shape})")
