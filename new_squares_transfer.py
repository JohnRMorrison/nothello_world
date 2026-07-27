"""New Squares Transfer Experiment — GPT runner.

Tests whether OthelloGPT learns new squares faster when they fit the existing
board geometry (coherent 9th row) vs when they're spatially arbitrary
(incoherent random flanks).

The games and matched test sets are produced by new_squares_data.py (which
holds the three controls: equal new-square exposure, length-matched eval, and
n_legal-matched ceiling) and are SHARED with the MLP runner — this file only
adds the GPT-specific pieces: vocab expansion 61->69, a tokenized dataset, a
per-cell score function, and the standard-LPM retention eval.  The IL/LL
transfer metrics go through new_squares_data.score_manifest, so GPT and MLP are
scored by identical code on identical data.

Usage:
    python new_squares_transfer.py --condition-id 0 --output-dir experiments/new_squares
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from behavioral_utils import load_shard_games
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
from new_squares_data import (
    CONDITIONS, NEW_SQUARE_IDS, N_CELLS, generate_condition, score_manifest,
)

EVAL_SCHEDULE = [0, 5, 25, 50, 100, 200, 500, 1000, 2000, 5000,
                 10000, 20000, 50000, 100000]
MAX_MOVES = 60          # GPT block_size = 59 -> sequences of length 60


# ============================================================================
# Legal-move mask for the standard-LPM retention eval (GPT-specific: vocab)
# ============================================================================

def build_legal_mask(games, legal_moves, stoi, block_size, vocab_size):
    """Boolean mask (N_tokens, vocab_size) of legal moves for teacher-forced
    next-move eval.  Position t in y targets move (t+1); its legal set is
    legal_moves[game][t+1]."""
    max_key = max(k for k in stoi.keys() if k >= 0)
    stoi_map = np.full(max_key + 1, -1, dtype=np.int32)
    for k, v in stoi.items():
        if k >= 0:
            stoi_map[k] = v
    rows, cols, pos = [], [], 0
    for gi, game in enumerate(games):
        T = min(len(game) - 1, block_size)
        for t in range(T):
            move_idx = t + 1
            if move_idx < len(legal_moves[gi]):
                for p in legal_moves[gi][move_idx]:
                    if p <= max_key and stoi_map[p] >= 0:
                        rows.append(pos)
                        cols.append(stoi_map[p])
            pos += 1
    mask = np.zeros((pos, vocab_size), dtype=np.bool_)
    mask[np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)] = True
    return mask


# ============================================================================
# GPT vocab expansion + tokenized dataset
# ============================================================================

def load_base_gpt(ckpt_path, device):
    """Load the pretrained OthelloGPT directly from its state_dict.

    Skips behavioral_utils.load_model's vocab-building game generation (slow,
    and unused here — ExtendedCharDataset rebuilds the vocab, and its first 61
    tokens align with the pretrained vocab by construction).  Also makes the
    runner locally smoke-testable without that step.  n_head isn't stored in
    the weights; 8 is standard for these checkpoints.
    """
    sd = torch.load(ckpt_path, map_location=device)
    vocab_size, n_embd = sd['tok_emb.weight'].shape
    block_size = sd['pos_emb'].shape[1]
    n_layer = len(set(k.split('.')[1] for k in sd if k.startswith('blocks.')))
    mconf = GPTConfig(vocab_size, block_size, n_layer=n_layer, n_head=8,
                      n_embd=n_embd)
    model = GPT(mconf)
    model.load_state_dict(sd)
    return model.to(device).eval()


def expand_model_vocab(model, old_vocab, new_vocab, device='cpu'):
    """Expand token embedding + output head to the new vocab.  Existing weights
    preserved exactly; new rows randomly initialized (std=0.02)."""
    n_embd = model.tok_emb.weight.shape[1]
    new_tok_emb = nn.Embedding(new_vocab, n_embd).to(device)
    new_tok_emb.weight.data[:old_vocab] = model.tok_emb.weight.data
    new_tok_emb.weight.data[old_vocab:].normal_(mean=0.0, std=0.02)
    model.tok_emb = new_tok_emb
    new_head = nn.Linear(n_embd, new_vocab, bias=False).to(device)
    new_head.weight.data[:old_vocab] = model.head.weight.data
    new_head.weight.data[old_vocab:].normal_(mean=0.0, std=0.02)
    model.head = new_head
    return model


class ExtendedCharDataset(CharDataset):
    """CharDataset whose vocab includes the new square ids (64-71)."""

    def __init__(self, data):
        all_vals = set()
        for game in data:
            all_vals.update(game)
        all_vals.add(-100)                      # padding
        chars = sorted(list(all_vals))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = max(len(game) for game in data)
        self.block_size = self.max_len - 1
        self.vocab_size = len(chars)
        self.data = data
        self.shuffle_moves = False      # newer mingpt CharDataset.__getitem__ reads this
        print(f'Dataset: {len(data)} sequences, {self.vocab_size} tokens.')


# ============================================================================
# Model-specific scoring adapters (consumed by the shared score_manifest)
# ============================================================================

def make_gpt_score_fn(model, stoi, itos, block_size, device):
    """prefix (cell-id moves) -> per-cell softmax next-move probability (N_CELLS,).
    This is the GPT adapter to the model-agnostic score_manifest."""
    def score_fn(prefix_moves):
        scores = np.zeros(N_CELLS, dtype=np.float32)
        tokens = [stoi[m] for m in prefix_moves if m in stoi]
        if not tokens:
            return scores
        tokens = tokens[-block_size:]
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x)
            probs = torch.softmax(logits[0, -1, :], dim=-1).cpu().numpy()
        for tok_idx in range(probs.shape[0]):
            cell = itos.get(tok_idx)
            if cell is not None and 0 <= cell < N_CELLS:
                scores[cell] = probs[tok_idx]
        return scores
    return score_fn


@torch.no_grad()
def evaluate_standard_lpm(model, loader, mask, device):
    """Legal-prob-mass + top-1 legality on a teacher-forced loader (retention)."""
    model.eval()
    total_lpm = total_acc = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=-1)
        B, T, V = probs.shape
        V_std = min(V, mask.shape[1])
        for b in range(B):
            for t in range(T):
                if y[b, t] == 0:
                    continue
                mask_row = mask[n % mask.shape[0]]
                total_lpm += probs[b, t, :V_std][mask_row[:V_std] > 0].sum().item()
                top1 = probs[b, t, :V_std].argmax().item()
                total_acc += 1.0 if mask_row[top1] > 0 else 0.0
                n += 1
    return total_lpm / max(n, 1), total_acc / max(n, 1)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition-id", type=int, required=True,
                    choices=[0, 1, 2, 3])   # 2/3 = coherent_all/incoherent_all
    ap.add_argument("--output-dir", type=str, default="experiments/new_squares")
    ap.add_argument("--ckpt", type=str, default="./ckpts/gpt_synthetic.ckpt",
                    help="Pretrained OthelloGPT state_dict to fine-tune.")
    ap.add_argument("--skip-retention", action="store_true",
                    help="Skip the standard-LPM retention eval (smoke tests).")
    ap.add_argument("--new-square-legal-budget", type=int, default=300000,
                    help="Control 1: matched new-square exposure (positions).")
    ap.add_argument("--n-test-positions", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1,
                    help="Passes over the training games; >1 reaches "
                         "convergence without regenerating data.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regenerate-data", action="store_true",
                    help="Force re-generation of the shared data files.")
    args = ap.parse_args()

    cond_name = CONDITIONS[args.condition_id]
    print(f"Condition {args.condition_id}: {cond_name}", flush=True)
    print(f"Started at: {time.strftime('%c')}", flush=True)

    # ---- Shared, model-agnostic data (Controls 1-3 live here) ----
    games, manifest = generate_condition(
        args.condition_id, args.output_dir,
        legal_budget=args.new_square_legal_budget, seed=args.seed,
        n_test_positions=args.n_test_positions, overwrite=args.regenerate_data)
    print(f"  {len(games)} games; IL={len(manifest['IL'])} LL={len(manifest['LL'])}",
          flush=True)

    # ---- GPT setup: expand vocab, tokenized dataset ----
    print("Loading model...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_base_gpt(args.ckpt, device)
    old_vocab = model.tok_emb.weight.shape[0]

    games = [g[:MAX_MOVES] for g in games]
    n_train = int(len(games) * 0.95)
    train_games = games[:n_train]
    train_ds = ExtendedCharDataset(train_games)
    new_vocab = train_ds.vocab_size
    print(f"Expanding model vocab: {old_vocab} -> {new_vocab}", flush=True)
    model = expand_model_vocab(model, old_vocab, new_vocab, device)

    # ---- Standard-LPM retention test (GPT-specific) ----
    std_loader = std_mask = None
    if not args.skip_retention:
        print("Building standard retention test...", flush=True)
        std_games = load_shard_games(0, games_per_shard=2000)
        std_legal = []
        for g in std_games:
            b = OthelloBoardState()
            lm = []
            for move in g:
                lm.append(b.get_valid_moves())
                b.umpire(move)
            std_legal.append(lm)
        std_ds = ExtendedCharDataset(std_games)
        std_ds.stoi, std_ds.itos = train_ds.stoi, train_ds.itos
        std_ds.vocab_size = train_ds.vocab_size
        std_mask = build_legal_mask(std_games, std_legal, train_ds.stoi,
                                    train_ds.block_size, train_ds.vocab_size)
        std_loader = DataLoader(std_ds, batch_size=64, shuffle=False)

    # ---- Training ----
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    eval_at = set([s for s in EVAL_SCHEDULE if s < total_steps] + [total_steps])

    results = {
        'condition_id': args.condition_id, 'condition_name': cond_name,
        'new_square_legal_budget': args.new_square_legal_budget,
        'n_train_games': len(train_games), 'total_steps': total_steps,
        'epochs': args.epochs, 'steps_per_epoch': steps_per_epoch,
        'lr': args.lr, 'bs': args.bs, 'seed': args.seed,
        'eval_steps': [], 'IL_prob': [], 'IL_acc': [], 'LL_prob': [],
        'LL_acc': [], 'std_lpm': [], 'std_top': [], 'IL_buckets': [],
    }
    t0 = time.time()

    def do_eval(step):
        model.eval()
        score_fn = make_gpt_score_fn(model, train_ds.stoi, train_ds.itos,
                                     train_ds.block_size, device)
        m = score_manifest(score_fn, manifest, per_bucket=True)
        if args.skip_retention:
            std_lpm, std_top = 0.0, 0.0
        else:
            std_lpm, std_top = evaluate_standard_lpm(model, std_loader, std_mask, device)
        results['eval_steps'].append(step)
        results['IL_prob'].append(m['IL_prob'])
        results['IL_acc'].append(m['IL_acc'])
        results['LL_prob'].append(m['LL_prob'])
        results['LL_acc'].append(m['LL_acc'])
        results['std_lpm'].append(std_lpm)
        results['std_top'].append(std_top)
        results['IL_buckets'].append(m.get('IL_buckets', {}))
        print(f"  Step {step}: IL_prob={m['IL_prob']:.4f} IL_acc={m['IL_acc']:.4f} "
              f"LL_prob={m['LL_prob']:.4f} std_lpm={std_lpm:.4f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
        model.train()

    do_eval(0)
    model.train()
    gstep = 0
    for epoch in range(args.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gstep += 1
            if gstep in eval_at:
                do_eval(gstep)

    results['elapsed_seconds'] = time.time() - t0
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'gpt_cond_{args.condition_id:03d}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
