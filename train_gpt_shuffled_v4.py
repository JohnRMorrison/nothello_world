"""v4: a cleaner MLP-analog using CELL-INDEXED positions.

Compared to v3:
  - v3 puts the moves at RANDOM sequence positions and lets the model
    figure out "ignore my sequence position, look at the token."
  - v4 fixes sequence position p (for p in 0..59) to mean "I am cell p."
    Each context slot is a SPECIFIC cell.  The token there says whether
    that cell has been played, and if so its parity.  This mirrors the
    MLP's played+even feature layout one-to-one.

Layout per sample (block_size = 2 * MAX_CELL = 120):

    positions 0..59     : CONTEXT block.  Position c corresponds to
                          cell c (a fixed assignment).  Token at position
                          c is 1+c+60*parity if cell c was played in this
                          game (parity = move's parity), else 0 (pad).
    positions 60..119   : QUERY block.  Position 60+m carries the QUERY
                          token (id 121).  Target at 60+m is the
                          parity-tagged encoding of game[m] (M_{m+1}).

Per-sample attention mask (1 = attend, 0 = mask):

    diagonal                   1 everywhere (self attention; avoids
                               all-zero softmax rows on pad slots)
    context[c] -> context[c']  0 for c != c' (cells are processed
                               independently; this is what makes the model
                               truly order-blind among the played cells)
    context[c] -> query[m]     0 (context never reads queries)
    query[m]   -> context[c]   1 iff cell c was played as one of M_1..M_m
                               (i.e., the move index when c was played < m)
    query[m]   -> query[m']    0 for m' != m (queries are independent)

Information available to query m:
  - The SET of cells played in the first m moves of the original game,
    each tagged with its move's parity bit
  - The position embedding of each visible cell-position directly encodes
    "this is cell c" (a fixed, meaningful association — just like the
    MLP's input vector having "cell c" at dimension c)
  - NO order info within parity (just like the MLP's input)

Vocab (same as v3; index 0 = pad/ignore — matches model's hardcoded
ignore_index=0):
    0           pad / y ignore / cell-not-played
    1..60       (cell c) played with parity 0  (token = 1+c)
    61..120     (cell c) played with parity 1  (token = 61+c)
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

class CellIndexedMaskedDataset(Dataset):

    PAD_ID = 0
    QUERY_ID = 121
    VOCAB_SIZE = 122
    NUM_CELLS = 60

    def __init__(self, games, cell_stoi=None, sample_cap=2000):
        self.games = games
        self.context_len = self.NUM_CELLS    # one position per cell
        self.block_size = 2 * self.NUM_CELLS  # context + queries

        if cell_stoi is None:
            print(f"Building cell vocab (sampling first {sample_cap} games)...")
            cells = set()
            for g in games[:sample_cap]:
                cells.update(g)
            cells = sorted(cells)
            if len(cells) != self.NUM_CELLS:
                print(f"  WARN: found {len(cells)} unique cells, "
                      f"expected {self.NUM_CELLS}")
            self.cell_stoi = {c: i for i, c in enumerate(cells)}
        else:
            self.cell_stoi = cell_stoi
        print(f"  cell vocab size: {len(self.cell_stoi)}")

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        N = min(len(game), self.NUM_CELLS)  # Othello game length cap

        T = self.block_size
        Lc = self.context_len

        # Init all-zero (= pad / ignore)
        x = np.zeros(T, dtype=np.int64)
        y = np.zeros(T, dtype=np.int64)
        mask = np.zeros((T, T), dtype=np.int8)
        np.fill_diagonal(mask, 1)

        if N < 2:
            return (torch.from_numpy(x),
                    torch.from_numpy(y),
                    torch.from_numpy(mask).float())

        # ----- Context tokens: position c = cell c -----
        # For each played move M_j (0-indexed: j in [0, N)),
        # the cell it occupied is c = cell_stoi[game[j]].
        # Place token (1 + c + 60 * (j%2)) at sequence position c.
        # played_at[c] = j if cell c was played as move j, else NUM_CELLS
        # (a sentinel that's >= max possible m, so cell c is never visible
        # to any query).
        played_at = np.full(self.NUM_CELLS, self.NUM_CELLS, dtype=np.int64)
        for j in range(N):
            c = self.cell_stoi[game[j]]
            played_at[c] = j
            parity = j % 2
            x[c] = 1 + c + 60 * parity

        # ----- Query tokens + targets -----
        for m in range(N):
            x[Lc + m] = self.QUERY_ID
            target_cell = self.cell_stoi[game[m]]
            target_parity = m % 2
            y[Lc + m] = 1 + target_cell + 60 * target_parity

        # ----- Attention mask -----
        # Query m attends to cell c if cell c was played as one of the
        # first m moves: played_at[c] < m.
        m_grid = np.arange(N, dtype=np.int64)[:, None]   # (N, 1)
        query_to_ctx = (played_at[None, :] < m_grid).astype(np.int8)  # (N, 60)
        mask[Lc:Lc + N, :Lc] = query_to_ctx

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
    ckpt_tag = os.environ.get('CKPT_TAG', '')
    load_ckpt = os.environ.get('LOAD_CKPT', '')

    print(f"epochs={epochs}  batch_size={batch_size}  "
          f"num_workers={num_workers}")

    print("Loading Othello dataset...")
    othello = get_othello(ood_num=-1, data_root=None, wthor=True)
    print(f"  {len(othello.sequences)} train games, {len(othello.val)} val games")

    print("Building train dataset...")
    train_dataset = CellIndexedMaskedDataset(othello.sequences)
    print("Building val dataset (reusing train cell vocab)...")
    val_dataset = CellIndexedMaskedDataset(
        othello.val, cell_stoi=train_dataset.cell_stoi)

    block_size = train_dataset.block_size
    print(f"Vocab size: {CellIndexedMaskedDataset.VOCAB_SIZE}, "
          f"block_size: {block_size}")
    mconf = GPTConfig(CellIndexedMaskedDataset.VOCAB_SIZE, block_size,
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
    ckpt_path = f"./ckpts/gpt_shuffled_v4{tag}_{t_start}.ckpt"
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

    out_path = f"./logs/shuffled_v4{tag}_{t_start}_final_loss.txt"
    with open(out_path, 'w') as f:
        f.write(f"epochs {epochs}\n")
        f.write(f"batch_size {batch_size}\n")
        f.write(f"block_size {block_size}\n")
        f.write(f"ckpt {ckpt_path}\n")
        f.write(f"final_val_loss {val_loss:.6f}\n")
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
