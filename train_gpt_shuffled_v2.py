"""Train GPT to predict the next ORIGINAL-order move from a shuffled,
parity-tagged prefix.

This is a corrected version of train_gpt_shuffled.py, which had the wrong
task (predicting the next SHUFFLED token, whose target is irreducibly
random and bounded by log(60-k)).

What this script does instead, per training sample:
  1. Take an original game M_1, M_2, ..., M_N
  2. Sample a random prefix length k in [4, max_prefix_len]
  3. Tag each cell M_i with its parity i%2 (matches your MLP's
     "played + even" features) by encoding it as the token
        token_id = cell + 60 * (i % 2)
  4. Shuffle the k tagged tokens into random order
  5. Target: the next original move M_{k+1}, also parity-tagged

Vocabulary (121 tokens):
   0  .. 59   cells on even-indexed (0,2,4,...) moves
   60 .. 119  cells on odd-indexed (1,3,5,...) moves
   120        pad

Training loss is computed only at position k-1 of each sample (the last
real input token) — everywhere else the target is -100 (ignored in CE).

Baseline reference: the MLP pattern detector achieves ~95% per-pattern
accuracy using played+even features. A transformer with the same input
should be able to match or beat that. The trivial uniform-over-60
baseline is log(60) ≈ 4.09; the network should land WELL below that.

Saves:
  ckpts/gpt_shuffled_v2_<ts>.ckpt
  logs/shuffled_v2_<ts>_final_loss.txt
"""
import os
import random
import sys
import time

import torch
from torch.utils.data import Dataset, DataLoader

from data import get_othello
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import set_seed


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShuffledPrefixDataset(Dataset):
    """For each game produces one (shuffled, parity-tagged prefix, next-move)
    sample per epoch.  Prefix length k is resampled fresh every call.

    Vocab encoding (index 0 is pad/ignore — matches the model's hardcoded
    F.cross_entropy(..., ignore_index=0) at mingpt/model.py:194):
        0           pad (and y ignore)
        1..60       cells with parity 0
        61..120     cells with parity 1
        VOCAB_SIZE = 121
        token_id = 1 + cell_index + 60 * (move_index % 2)
    """

    PAD_ID = 0
    VOCAB_SIZE = 121
    MIN_K = 4    # minimum prefix length (so the model has some context)

    def __init__(self, games, max_prefix_len=40, cell_stoi=None, sample_cap=2000):
        self.games = games
        self.max_prefix_len = max_prefix_len

        if cell_stoi is None:
            print("Building cell vocab (sampling first %d games)..." % sample_cap)
            cells = set()
            for g in games[:sample_cap]:
                cells.update(g)
            cells = sorted(cells)
            if len(cells) != 60:
                print(f"  WARN: found {len(cells)} unique cells, expected 60")
            self.cell_stoi = {c: i for i, c in enumerate(cells)}
        else:
            self.cell_stoi = cell_stoi
        print(f"  cell vocab size: {len(self.cell_stoi)}")

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        N = len(game)

        # Sample prefix length k
        max_k = min(N - 1, self.max_prefix_len)
        if max_k < self.MIN_K:
            k = max(1, N - 1)
        else:
            k = random.randint(self.MIN_K, max_k)

        # Parity-tag the first k moves: token = 1 + cell + 60 * (i%2)
        # (the +1 shift reserves 0 for pad/ignore — see class docstring)
        tagged = []
        for i in range(k):
            cell = self.cell_stoi[game[i]]
            parity = i % 2
            tagged.append(1 + cell + 60 * parity)

        # Shuffle the tagged tokens
        random.shuffle(tagged)

        # Target: M_{k+1}, tagged with its original parity k%2
        target_cell = self.cell_stoi[game[k]]
        target_parity = k % 2
        target_id = 1 + target_cell + 60 * target_parity

        # Pad input on the right; target only at position k-1.
        # PAD_ID == 0 and y-fill == 0 so the model's ignore_index=0 picks them up.
        x = tagged + [self.PAD_ID] * (self.max_prefix_len - k)
        y = [0] * self.max_prefix_len
        # Position k-1 is the last real input token; its output predicts the
        # next token, which we set to M_{k+1}.
        y[k - 1] = target_id

        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(44)

    epochs = int(os.environ.get('EPOCHS', '50'))
    batch_size = int(os.environ.get('BATCH_SIZE', '4096'))
    num_workers = int(os.environ.get('NUM_WORKERS', '16'))
    max_prefix_len = int(os.environ.get('MAX_PREFIX_LEN', '40'))
    ckpt_tag = os.environ.get('CKPT_TAG', '')

    print(f"epochs={epochs}  batch_size={batch_size}  "
          f"num_workers={num_workers}  max_prefix_len={max_prefix_len}")

    print("Loading Othello dataset...")
    othello = get_othello(ood_num=-1, data_root=None, wthor=True)
    print(f"  {len(othello.sequences)} train games, {len(othello.val)} val games")

    print("Building train dataset...")
    train_dataset = ShuffledPrefixDataset(
        othello.sequences, max_prefix_len=max_prefix_len)
    print("Building val dataset (reusing train cell vocab)...")
    val_dataset = ShuffledPrefixDataset(
        othello.val, max_prefix_len=max_prefix_len,
        cell_stoi=train_dataset.cell_stoi)

    print(f"Vocab size: {ShuffledPrefixDataset.VOCAB_SIZE}, "
          f"block_size: {max_prefix_len}")
    mconf = GPTConfig(ShuffledPrefixDataset.VOCAB_SIZE, max_prefix_len,
                      n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)

    load_ckpt = os.environ.get('LOAD_CKPT', '')
    if load_ckpt:
        print(f"Loading existing weights from {load_ckpt}")
        state = torch.load(load_ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  load_state_dict: missing={missing}, unexpected={unexpected}")

    t_start = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{ckpt_tag}" if ckpt_tag else ""
    ckpt_path = f"./ckpts/gpt_shuffled_v2{tag}_{t_start}.ckpt"
    os.makedirs('./ckpts', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    tconf = TrainerConfig(
        max_epochs=epochs,
        batch_size=batch_size,
        learning_rate=5e-4,
        lr_decay=True,
        warmup_tokens=len(train_dataset) * max_prefix_len * 5,
        final_tokens=len(train_dataset) * max_prefix_len * epochs,
        num_workers=num_workers,
        ckpt_path=ckpt_path,
    )

    trainer = Trainer(model, train_dataset, None, tconf)
    print(f"Training on device {trainer.device} for {epochs} epochs...")
    trainer.train()

    print("Computing final validation loss...")
    raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    raw_model.eval()
    loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True)
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            _, loss = raw_model(x, y)
            total_loss += loss.mean().item()
            total_batches += 1
    val_loss = total_loss / max(1, total_batches)
    print(f"FINAL VAL LOSS: {val_loss:.6f}")

    out_path = f"./logs/shuffled_v2{tag}_{t_start}_final_loss.txt"
    with open(out_path, 'w') as f:
        f.write(f"epochs {epochs}\n")
        f.write(f"batch_size {batch_size}\n")
        f.write(f"max_prefix_len {max_prefix_len}\n")
        f.write(f"ckpt {ckpt_path}\n")
        f.write(f"final_val_loss {val_loss:.6f}\n")
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
