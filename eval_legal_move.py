"""Evaluate next-move legality and accuracy of various Othello-GPT variants.

For each held-out game and each prefix-length k, we ask:
    "Given the first k moves of this game, what does the model predict
     for the (k+1)-th move?"

We then check:
    - Is the model's top-1 prediction a LEGAL move at this game state?
    - Does it match the actually-played move M_{k+1}?
    - Is M_{k+1} in the model's top-3 predictions?
    - Is M_{k+1} in the model's top-5 predictions?

Variants:
    original : Standard Othello-GPT (input = move seq in original order)
    v2       : Shuffled-prefix, parity-tagged, predicts M_{k+1}
    v3       : Shuffled-context + per-query mask, predicts all 60 at once
    v4       : Cell-indexed context + per-query mask, predicts all 60 at once

Usage:
    python eval_legal_move.py --variant v3 \\
        --ckpt ckpts/gpt_shuffled_v3_<ts>.ckpt \\
        --num-games 500
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, '.')
from data import get_othello
from data.othello import OthelloBoardState
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig

# Per-variant model config used at training time.
VARIANT_CONFIG = {
    'original': dict(vocab_size=61,  block_size=59),
    'v2':       dict(vocab_size=121, block_size=40),
    'v3':       dict(vocab_size=122, block_size=120),
    'v4':       dict(vocab_size=122, block_size=120),
}


def load_model(variant, ckpt_path, device):
    cfg = VARIANT_CONFIG[variant]
    mconf = GPTConfig(cfg['vocab_size'], cfg['block_size'],
                      n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def build_cell_stoi(games, sample_cap=2000):
    cells = set()
    for g in games[:sample_cap]:
        cells.update(g)
    return {c: i for i, c in enumerate(sorted(cells))}


# ---------- v2 forward (one position at a time) ----------

def predict_v2(model, prefix, cell_stoi, device, max_prefix_len=40):
    k = len(prefix)
    if k == 0:
        return None
    if k > max_prefix_len:
        prefix = prefix[-max_prefix_len:]
        k = max_prefix_len
    tagged = [1 + cell_stoi[c] + 60 * (i % 2) for i, c in enumerate(prefix)]
    random.shuffle(tagged)
    x = tagged + [0] * (max_prefix_len - k)
    x = torch.tensor(x, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x)
    return logits[0, k - 1].cpu()   # (vocab,)


# ---------- v3 / v4 forward (full game in one pass) ----------

def predict_v3_full_game(model, game, cell_stoi, device, max_game_len=60):
    """Returns logits at each query position (m, vocab)."""
    from train_gpt_shuffled_v3 import MaskedShuffleDataset
    ds = MaskedShuffleDataset([list(game)], cell_stoi=cell_stoi,
                              max_game_len=max_game_len)
    x, _, mask = ds[0]
    x = x.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x, attn_mask=mask)
    return logits[0, max_game_len:max_game_len + len(game)].cpu()


def predict_v4_full_game(model, game, cell_stoi, device):
    from train_gpt_shuffled_v4 import CellIndexedMaskedDataset
    ds = CellIndexedMaskedDataset([list(game)], cell_stoi=cell_stoi)
    Lc = ds.context_len
    x, _, mask = ds[0]
    x = x.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x, attn_mask=mask)
    return logits[0, Lc:Lc + len(game)].cpu()


# ---------- original Othello-GPT forward (full game in one pass) ----------

def predict_original_full_game(model, game, train_dataset, device):
    """Returns logits at each prefix-end position (k, vocab)."""
    dix = [train_dataset.stoi[s] for s in game]
    x = torch.tensor(dix, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x)
    # logits[0, k] is the prediction conditioned on tokens 0..k (i.e. M_1..M_{k+1});
    # we want the prediction CONDITIONED ON the first k tokens, which is logits[0, k-1].
    return logits[0].cpu()


# ---------- Top-K decoding ----------

def topk_cells_v234(logits_vec, cell_stoi_inv, k=5):
    """Decode top-k token predictions to original cell values.  Returns a list
    of length <=k, with None for invalid tokens (pad/QUERY)."""
    topk = logits_vec.topk(min(k * 2, logits_vec.numel()))  # extra to filter invalid
    out = []
    for tok in topk.indices.tolist():
        if tok == 0 or tok > 120:
            continue
        cell_idx = (tok - 1) % 60
        out.append(cell_stoi_inv[cell_idx])
        if len(out) >= k:
            break
    return out


def topk_cells_original(logits_vec, train_dataset, k=5):
    topk = logits_vec.topk(min(k * 2, logits_vec.numel()))
    out = []
    for tok in topk.indices.tolist():
        if tok == 0:
            continue  # the -100 sentinel
        out.append(train_dataset.itos[tok])
        if len(out) >= k:
            break
    return out


# ---------- Main eval loop ----------

def evaluate(model, variant, val_games, helper, device, num_games):
    """helper: cell_stoi (v2/v3/v4) or train_dataset (original)."""
    if variant == 'original':
        train_dataset = helper
    else:
        cell_stoi = helper
        cell_stoi_inv = {i: c for c, i in cell_stoi.items()}

    totals = {'n': 0, 'legal1': 0, 'top1': 0, 'top3': 0, 'top5': 0}
    by_phase = {'early': dict(totals), 'mid': dict(totals), 'late': dict(totals)}

    games = val_games[:num_games]
    for game in tqdm(games, desc=f"eval {variant}"):
        N = len(game)
        if N < 2:
            continue

        # Pre-compute all per-position logits once when possible.
        if variant == 'v3':
            query_logits = predict_v3_full_game(model, game, cell_stoi, device)
        elif variant == 'v4':
            query_logits = predict_v4_full_game(model, game, cell_stoi, device)
        elif variant == 'original':
            full_logits = predict_original_full_game(model, game, train_dataset, device)

        board = OthelloBoardState()
        for m in range(N):
            # Evaluate prediction of game[m] given context game[:m].
            if m > 0:
                legal_moves = board.get_valid_moves()  # list of permit values
                actual = game[m]

                if variant in ('v3', 'v4'):
                    logits_vec = query_logits[m]
                    preds = topk_cells_v234(logits_vec, cell_stoi_inv, k=5)
                elif variant == 'v2':
                    logits_vec = predict_v2(model, list(game[:m]), cell_stoi, device)
                    preds = topk_cells_v234(logits_vec, cell_stoi_inv, k=5)
                elif variant == 'original':
                    # We want prediction given the first m tokens (game[:m]).
                    # In autoregressive: logits at position m-1 predicts token m.
                    logits_vec = full_logits[m - 1]
                    preds = topk_cells_original(logits_vec, train_dataset, k=5)

                if preds:
                    top1 = preds[0]
                else:
                    top1 = None

                # Pick the phase bucket
                if m < N / 3:        phase = 'early'
                elif m < 2 * N / 3:  phase = 'mid'
                else:                phase = 'late'

                for d in (totals, by_phase[phase]):
                    d['n']      += 1
                    d['legal1'] += int(top1 is not None and top1 in legal_moves)
                    d['top1']   += int(top1 == actual)
                    d['top3']   += int(actual in preds[:3])
                    d['top5']   += int(actual in preds[:5])

            board.update([game[m]])

    return totals, by_phase


def fmt(d):
    n = d['n'] or 1
    return (f"  legal-top1: {100*d['legal1']/n:5.2f}%  "
            f"top1: {100*d['top1']/n:5.2f}%  "
            f"top3: {100*d['top3']/n:5.2f}%  "
            f"top5: {100*d['top5']/n:5.2f}%  "
            f"(n={d['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=list(VARIANT_CONFIG), required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--num-games', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data-root', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Variant: {args.variant}  ckpt: {args.ckpt}")

    print("Loading Othello dataset (val split)...")
    othello = get_othello(ood_num=-1, data_root=args.data_root, wthor=True)
    val_games = othello.val
    print(f"  {len(val_games)} val games available")

    # Build the helper (cell_stoi for v2/v3/v4, train_dataset for original)
    if args.variant == 'original':
        print("Building CharDataset (matches train-time stoi/itos)...")
        helper = CharDataset(othello)
    else:
        print("Building cell_stoi from training games...")
        helper = build_cell_stoi(othello.sequences)

    model = load_model(args.variant, args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model with {n_params/1e6:.1f}M params")

    totals, by_phase = evaluate(model, args.variant, val_games,
                                helper, device, args.num_games)
    print(f"\n=== {args.variant} overall ===")
    print(fmt(totals))
    for phase in ('early', 'mid', 'late'):
        print(f"  {phase:5s}: {fmt(by_phase[phase])}")


if __name__ == '__main__':
    main()
