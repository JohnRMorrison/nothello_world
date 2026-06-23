"""v3: train GPT with one shuffled context per game and 60 simultaneous
prediction "queries" — each query attends (via a per-sample attention
mask) to a different prefix of the shuffled context.  One forward pass
per game gives ~60 training signals while keeping the SET-of-played-cells
information that the MLP pattern detector uses.

Layout per sample (block_size = 2 * MAX_GAME_LEN = 120):

    positions 0..59     : CONTEXT block.  At position i sits the parity-
                          tagged token for game[p[i]] (i.e. the move that
                          ended up at shuffled position i).  Padded with 0.
    positions 60..119   : QUERY block.  Position 60+m is a single QUERY
                          token (id = 121).  Its target is the
                          parity-tagged encoding of game[m] (= M_{m+1}).

Per-sample attention mask (1 = attend, 0 = mask):

    diagonal                   1 everywhere (self-attention only; this is
                               what prevents leakage AND ensures softmax
                               never sees an all-zero row)
    context[i] -> context[j]   0 for i != j  (context positions are
                               PROCESSED INDEPENDENTLY through the L MLP
                               layers — this is what makes the model
                               truly order-blind)
    context[i] -> query[m]     0 (context never reads queries)
    query[m]   -> context[i]   1 iff p[i] < m   (sees the first m original
                                                  moves' shuffled positions)
    query[m]   -> query[m']    0 for m' != m (queries are independent)

Why no context-to-context attention?  If context positions shared info
bidirectionally, query m (attending to the first m positions) would
indirectly receive info from the OTHER context positions via the bidir
edges, leaking future moves into its prediction.  Keeping context
independent guarantees that query m can only see the SET of first-m moves
+ their parities, exactly matching the MLP pattern-detector's input.

Vocab (index 0 = pad/ignore — matches model's hardcoded ignore_index=0):
    0           pad (context filler) + y ignore
    1..60       cells with parity 0
    61..120     cells with parity 1
    121         QUERY token
    VOCAB_SIZE = 122
"""
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data import get_othello
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import set_seed


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MaskedShuffleDataset(Dataset):

    PAD_ID = 0
    QUERY_ID = 121
    VOCAB_SIZE = 122

    def __init__(self, games, max_game_len=60, cell_stoi=None, sample_cap=2000):
        self.games = games
        self.max_game_len = max_game_len
        self.block_size = 2 * max_game_len

        if cell_stoi is None:
            print(f"Building cell vocab (sampling first {sample_cap} games)...")
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
        N = min(len(game), self.max_game_len)

        T = self.block_size
        Lc = self.max_game_len  # context length

        # Initialize all-zero (= pad / ignore)
        x = np.zeros(T, dtype=np.int64)
        y = np.zeros(T, dtype=np.int64)
        mask = np.zeros((T, T), dtype=np.int8)
        # Diagonal always 1 so softmax never sees an all-zero row.
        np.fill_diagonal(mask, 1)

        if N < 2:
            return (torch.from_numpy(x),
                    torch.from_numpy(y),
                    torch.from_numpy(mask).float())

        # ----- Shuffle and build context tokens -----
        p = list(range(N))
        random.shuffle(p)  # p[i] = original move index at shuffled position i
        for i in range(N):
            cell = self.cell_stoi[game[p[i]]]
            parity = p[i] % 2
            x[i] = 1 + cell + 60 * parity

        # ----- Query tokens + targets -----
        for m in range(N):
            x[Lc + m] = self.QUERY_ID
            target_cell = self.cell_stoi[game[m]]
            target_parity = m % 2
            y[Lc + m] = 1 + target_cell + 60 * target_parity

        # ----- Attention mask -----
        # Diagonal is already 1.  We DON'T add context-to-context attention
        # (would leak future moves via the residual stream).
        # Query m attends to context positions where p[i] < m.
        p_arr = np.array(p, dtype=np.int64)        # shape (N,)
        m_grid = np.arange(N, dtype=np.int64)[:, None]   # shape (N, 1)
        # (N, N): row m, col i: 1 iff p[i] < m
        query_to_ctx = (p_arr[None, :] < m_grid).astype(np.int8)
        mask[Lc:Lc + N, :N] = query_to_ctx

        return (torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(mask).float())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(44)

    epochs = int(os.environ.get('EPOCHS', '20'))
    batch_size = int(os.environ.get('BATCH_SIZE', '1024'))
    num_workers = int(os.environ.get('NUM_WORKERS', '16'))
    max_game_len = int(os.environ.get('MAX_GAME_LEN', '60'))
    ckpt_tag = os.environ.get('CKPT_TAG', '')
    load_ckpt = os.environ.get('LOAD_CKPT', '')

    print(f"epochs={epochs}  batch_size={batch_size}  "
          f"num_workers={num_workers}  max_game_len={max_game_len}")

    print("Loading Othello dataset...")
    othello = get_othello(ood_num=-1, data_root=None, wthor=True)
    print(f"  {len(othello.sequences)} train games, {len(othello.val)} val games")

    print("Building train dataset...")
    train_dataset = MaskedShuffleDataset(
        othello.sequences, max_game_len=max_game_len)
    print("Building val dataset (reusing train cell vocab)...")
    val_dataset = MaskedShuffleDataset(
        othello.val, max_game_len=max_game_len,
        cell_stoi=train_dataset.cell_stoi)

    block_size = train_dataset.block_size
    print(f"Vocab size: {MaskedShuffleDataset.VOCAB_SIZE}, "
          f"block_size: {block_size}")
    mconf = GPTConfig(MaskedShuffleDataset.VOCAB_SIZE, block_size,
                      n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)

    if load_ckpt:
        print(f"Loading existing weights from {load_ckpt}")
        state = torch.load(load_ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  missing={missing}  unexpected={unexpected}")

    t_start = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{ckpt_tag}" if ckpt_tag else ""
    ckpt_path = f"./ckpts/gpt_shuffled_v3{tag}_{t_start}.ckpt"
    os.makedirs('./ckpts', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    tconf = TrainerConfig(
        max_epochs=epochs,
        batch_size=batch_size,
        learning_rate=5e-4,
        lr_decay=True,
        warmup_tokens=len(train_dataset) * block_size * 5,
        final_tokens=len(train_dataset) * block_size * epochs,
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
        for batch in loader:
            x, y, attn_mask = batch
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            attn_mask = attn_mask.to(trainer.device)
            _, loss = raw_model(x, y, attn_mask=attn_mask)
            total_loss += loss.mean().item()
            total_batches += 1
    val_loss = total_loss / max(1, total_batches)
    print(f"FINAL VAL LOSS: {val_loss:.6f}")

    out_path = f"./logs/shuffled_v3{tag}_{t_start}_final_loss.txt"
    with open(out_path, 'w') as f:
        f.write(f"epochs {epochs}\n")
        f.write(f"batch_size {batch_size}\n")
        f.write(f"max_game_len {max_game_len}\n")
        f.write(f"block_size {block_size}\n")
        f.write(f"ckpt {ckpt_path}\n")
        f.write(f"final_val_loss {val_loss:.6f}\n")
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
