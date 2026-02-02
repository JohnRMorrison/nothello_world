#!/usr/bin/env python
"""
Train 5 smaller models and compare to the 8-layer OthelloGPT.
Multi-GPU CUDA script (DataParallel). Run from the project root:

    python experiments/model_size_comparison.py [OPTIONS]

Options (all optional):
    --models mlp_small,transformer_2l   Comma-separated list of models to train
                                        (default: all 5). Choices: mlp_small,
                                        mlp_medium, mlp_large, transformer_2l,
                                        transformer_4l
    --epochs 10                         Max training epochs (default: 10)
    --batch-size 256                    Per-GPU batch size (default: 256)
    --lr 3e-4                           Learning rate (default: 3e-4)
    --num-workers 4                     DataLoader workers (default: 4)
    --eval-games 200                    Games for legal-move-accuracy eval (default: 200)
    --ckpt-dir ckpts                    Checkpoint directory (default: ckpts)
    --save-plots results.png            Save comparison plots to file (default: results.png)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so imports work when run from anywhere
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data import get_othello
from data.othello import OthelloBoardState
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig
from mingpt.utils import set_seed


# ============================= MLP Model ===================================

class OthelloMLP(nn.Module):
    """MLP baseline: token embedding -> flatten -> hidden layers -> output logits."""

    def __init__(self, vocab_size, block_size, n_embd, hidden_sizes, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(dropout)

        input_dim = block_size * n_embd
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, block_size * vocab_size))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def get_block_size(self):
        return self.block_size

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, nn.Linear):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, nn.Embedding):
                    no_decay.add(fpn)
        no_decay.add("pos_emb")
        param_dict = {pn: p for pn, p in self.named_parameters()}
        decay &= param_dict.keys()
        no_decay &= param_dict.keys()
        no_decay |= param_dict.keys() - (decay | no_decay)
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        if t < self.block_size:
            pad = torch.zeros(b, self.block_size - t, dtype=idx.dtype, device=idx.device)
            idx_full = torch.cat([idx, pad], dim=1)
        else:
            idx_full = idx[:, :self.block_size]

        tok = self.tok_emb(idx_full)
        pos = self.pos_emb[:, :self.block_size, :]
        x = self.drop(tok + pos)
        x = x.view(b, -1)
        x = self.net(x)
        logits = x.view(b, self.block_size, self.vocab_size)
        logits = logits[:, :t, :]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        return logits, loss


# ============================ Training =====================================

class TrainConfig:
    def __init__(self, **kwargs):
        self.max_epochs = 10
        self.batch_size = 256
        self.learning_rate = 3e-4
        self.betas = (0.9, 0.95)
        self.grad_norm_clip = 1.0
        self.weight_decay = 0.1
        self.num_workers = 4
        for k, v in kwargs.items():
            setattr(self, k, v)


def train_model(model, train_dataset, val_dataset, config, ckpt_path, model_name="model"):
    """Training loop with DataParallel for multi-GPU CUDA."""
    device = torch.cuda.current_device()
    raw_model = model
    optimizer = raw_model.configure_optimizers(config)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        print(f"  Using {n_gpu} GPUs via DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset, shuffle=True, pin_memory=True,
        batch_size=config.batch_size, num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, pin_memory=True,
        batch_size=config.batch_size, num_workers=config.num_workers,
    )

    best_val_loss = float("inf")
    epoch_bar = tqdm(range(config.max_epochs), desc=f"{model_name}", unit="epoch")
    for epoch in epoch_bar:
        # Train
        model.train()
        train_losses = []
        batch_bar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{config.max_epochs} [train]", leave=False)
        for x, y in batch_bar:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            loss = loss.mean()  # reduce across GPUs
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            optimizer.step()
            train_losses.append(loss.item())
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"  Epoch {epoch+1}/{config.max_epochs} [val]", leave=False):
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                loss = loss.mean()
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))
        train_loss = float(np.mean(train_losses))
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(raw_model.state_dict(), ckpt_path)

    print(f"  Best val loss: {best_val_loss:.4f} — saved to {ckpt_path}")
    raw_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    return raw_model, best_val_loss


# ============================ Evaluation ===================================

@torch.no_grad()
def evaluate(model, val_dataset, device, num_games=200):
    """Compute val cross-entropy loss and legal move accuracy."""
    model.eval()
    loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

    total_loss = 0.0
    total_tokens = 0

    games = val_dataset.data
    n_eval = min(num_games, len(games))

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx * 256 >= n_eval:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        mask = y != 0
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)

    total_legal = 0
    total_preds = 0
    for game_idx in tqdm(range(n_eval), desc="  Legal-move eval", leave=False):
        game = games[game_idx]
        for pos in range(1, len(game)):
            context = game[:pos]
            x = torch.tensor(
                [val_dataset.stoi[s] for s in context], dtype=torch.long
            )[None, :].to(device)
            logits, _ = model(x)
            pred_idx = logits[0, -1, :].argmax().item()
            pred_move = val_dataset.itos[pred_idx]

            board = OthelloBoardState()
            board.update(context)
            valid = board.get_valid_moves()

            total_preds += 1
            if pred_move in valid:
                total_legal += 1

    legal_acc = total_legal / max(total_preds, 1)
    return avg_loss, legal_acc


# ============================ Main =========================================

ALL_MODEL_NAMES = ["mlp_small", "mlp_medium", "mlp_large", "transformer_2l", "transformer_4l"]


def parse_args():
    p = argparse.ArgumentParser(description="Train smaller Othello models and compare to OthelloGPT")
    p.add_argument("--models", type=str, default=",".join(ALL_MODEL_NAMES),
                    help="Comma-separated model names to train (default: all)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--ckpt-dir", type=str, default="ckpts")
    p.add_argument("--save-plots", type=str, default="results.png")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    assert torch.cuda.is_available(), "This script requires CUDA"
    device = torch.cuda.current_device()
    n_gpu = torch.cuda.device_count()
    print(f"Device: cuda:{device}  |  GPUs available: {n_gpu}")

    models_to_train = set(args.models.split(","))
    for m in models_to_train:
        assert m in ALL_MODEL_NAMES, f"Unknown model: {m}. Choose from {ALL_MODEL_NAMES}"

    ckpt_dir = os.path.join(PROJECT_ROOT, args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Data ----
    print("Loading synthetic dataset...")
    othello = get_othello(ood_num=-1, data_root=None, wthor=True)
    train_dataset = CharDataset(othello)
    val_dataset = CharDataset(othello.val)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Vocab size: {train_dataset.vocab_size}, Block size: {train_dataset.block_size}")

    vocab_size = train_dataset.vocab_size
    block_size = train_dataset.block_size

    # ---- Model factories ----
    def make_mlp(hidden_sizes):
        return OthelloMLP(vocab_size, block_size, n_embd=128, hidden_sizes=hidden_sizes)

    def make_transformer(n_layer):
        return GPT(GPTConfig(vocab_size, block_size, n_layer=n_layer, n_head=8, n_embd=512))

    model_factories = {
        "mlp_small":      lambda: make_mlp([512, 256]),
        "mlp_medium":     lambda: make_mlp([1024, 512]),
        "mlp_large":      lambda: make_mlp([2048, 1024]),
        "transformer_2l": lambda: make_transformer(2),
        "transformer_4l": lambda: make_transformer(4),
    }

    tcfg = TrainConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.1,
        grad_norm_clip=1.0,
        num_workers=args.num_workers,
    )

    # ---- Train / load each model ----
    trained_models = {}  # name -> model (on device, raw — no DataParallel)

    for name in ALL_MODEL_NAMES:
        ckpt_path = os.path.join(ckpt_dir, f"{name}.ckpt")
        model = model_factories[name]()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print(f"{name} — {n_params:,} params")

        if os.path.exists(ckpt_path):
            print(f"  Loading existing checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            model = model.to(device)
            model.eval()
            trained_models[name] = model
        elif name in models_to_train:
            print(f"  Training...")
            model, _ = train_model(model, train_dataset, val_dataset, tcfg, ckpt_path, model_name=name)
            model = model.to(device)
            model.eval()
            trained_models[name] = model
        else:
            print(f"  Skipped (not requested and no checkpoint)")

    # ---- Load OthelloGPT ----
    gpt_ckpt = os.path.join(ckpt_dir, "gpt_synthetic.ckpt")
    if os.path.exists(gpt_ckpt):
        gpt_model = GPT(GPTConfig(vocab_size, block_size, n_layer=8, n_head=8, n_embd=512))
        gpt_model.load_state_dict(torch.load(gpt_ckpt, map_location=device, weights_only=False))
        gpt_model = gpt_model.to(device)
        gpt_model.eval()
        trained_models["othello_gpt_8l"] = gpt_model
        print(f"\nLoaded OthelloGPT — {sum(p.numel() for p in gpt_model.parameters()):,} params")
    else:
        print(f"\nWarning: OthelloGPT checkpoint not found at {gpt_ckpt}, skipping")

    # ---- Evaluate ----
    results = {}
    for name, model in trained_models.items():
        print(f"\nEvaluating {name}...")
        n_params = sum(p.numel() for p in model.parameters())
        val_loss, legal_acc = evaluate(model, val_dataset, device, num_games=args.eval_games)
        results[name] = {"params": n_params, "val_loss": val_loss, "legal_acc": legal_acc}
        print(f"  params={n_params:,}, val_loss={val_loss:.4f}, legal_acc={legal_acc*100:.1f}%")

    # ---- Print table ----
    display_order = ["mlp_small", "mlp_medium", "mlp_large", "transformer_2l", "transformer_4l", "othello_gpt_8l"]
    print(f"\n{'Model':<22} {'Params':>12} {'Val Loss':>10} {'Legal Acc':>10}")
    print("-" * 56)
    for name in display_order:
        if name in results:
            r = results[name]
            print(f"{name:<22} {r['params']:>12,} {r['val_loss']:>10.4f} {r['legal_acc']*100:>9.1f}%")

    # ---- Save results JSON ----
    results_path = os.path.join(ckpt_dir, "model_size_comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ---- Save plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [n for n in display_order if n in results]
        params_list = [results[n]["params"] / 1e6 for n in names]
        losses = [results[n]["val_loss"] for n in names]
        accs = [results[n]["legal_acc"] * 100 for n in names]
        colors = ["#4c72b0" if "mlp" in n else "#dd8452" for n in names]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].barh(names, params_list, color=colors)
        axes[0].set_xlabel("Parameters (M)")
        axes[0].set_title("Model Size")

        axes[1].barh(names, losses, color=colors)
        axes[1].set_xlabel("Val Cross-Entropy Loss")
        axes[1].set_title("Validation Loss")

        axes[2].barh(names, accs, color=colors)
        axes[2].set_xlabel("Legal Move Accuracy (%)")
        axes[2].set_title("Legal Move Accuracy")

        plt.tight_layout()
        plot_path = os.path.join(PROJECT_ROOT, args.save_plots)
        plt.savefig(plot_path, dpi=150)
        print(f"Plots saved to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Could not save plots: {e}")


if __name__ == "__main__":
    main()
