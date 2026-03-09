"""
Extract Othello-GPT activations and train MLPs of varying capacity on them.

Usage:
  # Step 1: Extract activations for all layers (GPU recommended)
  python -m experiments.mathematical_transformation_experiments.mlp_on_activations \
      extract --max-games 1000000 --output-dir cached_acts_1m

  # Step 2: Train MLPs on a specific layer
  python -m experiments.mathematical_transformation_experiments.mlp_on_activations \
      train --layer 0 --acts-dir cached_acts_1m --epochs 10

  # Step 3: Plot results
  python -m experiments.mathematical_transformation_experiments.mlp_on_activations \
      plot --results-dir cached_acts_1m
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..'))

from experiments.mathematical_transformation_experiments.probe_variant_boards import (
    load_games, tokenize_games, extract_activations, seq_to_state_normal,
    STOI, ITOS,
)
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    load_model, get_board_states, states_to_labels, extract_activations_pre,
    _build_mlp,
)

VOCAB_SIZE = 61
POS_START = 5
POS_END = 54
LENGTH = POS_END - POS_START  # 49
OPTIONS = 3


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def do_extract(args):
    """Extract and save activations for all layers."""
    device = get_device()
    print(f"Device: {device}")

    model, block_size = load_model(args.ckpt_path, device)
    print(f"Loaded model from {args.ckpt_path}")

    games = load_games(max_files=args.max_files)
    if args.max_games and len(games) > args.max_games:
        games = games[:args.max_games]
    n_eval = max(int(len(games) * 0.1), 100)
    train_games = games[:len(games) - n_eval]
    eval_games = games[len(games) - n_eval:]
    print(f"Using {len(games)} games ({len(train_games)} train, {len(eval_games)} eval)")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    def collect_acts_and_labels(model, games_list, device, layer, block_size, use_pre=False):
        """Collect activations and labels in batches."""
        acts_list = []
        labels_list = []
        game_batch = 64
        for start in range(0, len(games_list), game_batch):
            if start % (game_batch * 100) == 0:
                print(f"    batch {start}/{len(games_list)}", flush=True)
            batch_games = games_list[start:start + game_batch]
            tokens = tokenize_games(batch_games, seq_len=block_size).to(device)

            with torch.no_grad():
                if use_pre:
                    h = extract_activations_pre(model, tokens)
                else:
                    h = extract_activations(model, tokens, layer)
            h = h[:, POS_START:POS_END]

            states = get_board_states(batch_games, seq_to_state_normal, POS_START, POS_END)
            labels = states_to_labels(states)
            labels = labels.reshape(len(batch_games), LENGTH, 64)

            for gi in range(len(batch_games)):
                acts_list.append(h[gi].cpu())
                labels_list.append(torch.tensor(labels[gi], dtype=torch.long))

        return torch.cat(acts_list, dim=0), torch.cat(labels_list, dim=0)

    # Build position arrays
    tr_n = len(train_games) * LENGTH
    ev_n = len(eval_games) * LENGTH
    tr_pos = np.array([POS_START + (i % LENGTH) for i in range(tr_n)], dtype=np.int16)
    ev_pos = np.array([POS_START + (i % LENGTH) for i in range(ev_n)], dtype=np.int16)

    # Extract layer 0 first (to get labels)
    print("\nExtracting layer 0...")
    t0 = time.time()
    tr_X, tr_Y = collect_acts_and_labels(model, train_games, device, 0, block_size)
    ev_X, ev_Y = collect_acts_and_labels(model, eval_games, device, 0, block_size)
    print(f"  Done in {time.time()-t0:.0f}s. Shape: tr={tr_X.shape}, ev={ev_X.shape}")

    # Save labels and positions
    np.savez_compressed(os.path.join(out_dir, 'labels_and_pos.npz'),
        tr_Y=tr_Y.numpy().astype(np.int8),
        ev_Y=ev_Y.numpy().astype(np.int8),
        tr_pos=tr_pos, ev_pos=ev_pos)
    print(f"  Saved labels")

    # Save layer 0
    np.savez_compressed(os.path.join(out_dir, 'layer0.npz'),
        tr_X=tr_X.numpy().astype(np.float16),
        ev_X=ev_X.numpy().astype(np.float16))
    print(f"  Saved layer 0")
    del tr_X, ev_X

    # Layers 1-8
    for layer in range(1, 9):
        print(f"\nExtracting layer {layer}...")
        t0 = time.time()
        tr_X, _ = collect_acts_and_labels(model, train_games, device, layer, block_size)
        ev_X, _ = collect_acts_and_labels(model, eval_games, device, layer, block_size)
        np.savez_compressed(os.path.join(out_dir, f'layer{layer}.npz'),
            tr_X=tr_X.numpy().astype(np.float16),
            ev_X=ev_X.numpy().astype(np.float16))
        print(f"  Saved layer {layer} in {time.time()-t0:.0f}s: {tr_X.shape}")
        del tr_X, ev_X

    print(f"\nAll saved to {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        size_mb = os.path.getsize(os.path.join(out_dir, f)) / 1e6
        print(f"  {f}: {size_mb:.1f} MB")


def do_train(args):
    """Train MLPs of varying hidden dim on cached activations."""
    device = get_device()
    print(f"Device: {device}")

    acts_dir = args.acts_dir

    # Load labels and positions
    lp = np.load(os.path.join(acts_dir, 'labels_and_pos.npz'))
    tr_Y = torch.tensor(lp['tr_Y'].astype(np.int64))
    ev_Y = torch.tensor(lp['ev_Y'].astype(np.int64))
    tr_pos = torch.tensor(lp['tr_pos'].astype(np.int64))
    ev_pos = torch.tensor(lp['ev_pos'].astype(np.int64))
    print(f"Labels: tr={tr_Y.shape}, ev={ev_Y.shape}")

    # Load activations for the requested layer
    layer = args.layer
    print(f"\nLoading layer {layer} activations...")
    data = np.load(os.path.join(acts_dir, f'layer{layer}.npz'))
    tr_X = torch.tensor(data['tr_X'].astype(np.float32))
    ev_X = torch.tensor(data['ev_X'].astype(np.float32))
    print(f"  Shape: tr={tr_X.shape}, ev={ev_X.shape}")
    input_dim = tr_X.shape[1]

    hidden_dims = args.hidden_dims
    epochs = args.epochs
    batch_size = args.batch_size

    results = {}
    for H in hidden_dims:
        print(f"\n{'='*60}")
        print(f"MLP H={H} on layer {layer} ({input_dim}-d), {epochs} epochs")
        print(f"{'='*60}")

        acc = _train_mlp_nanda_epochs(
            tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos,
            device, input_dim, H, epochs=epochs, batch_size=batch_size)
        results[f"H{H}"] = acc
        print(f"  -> Best accuracy: {acc:.4%}")

    # Save results
    results_file = os.path.join(acts_dir, f'mlp_results_layer{layer}.json')
    with open(results_file, 'w') as f:
        json.dump({
            'layer': layer,
            'epochs': epochs,
            'input_dim': input_dim,
            'results': results,
        }, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: MLP on layer {layer} activations ({input_dim}-d)")
    print(f"{'='*60}")
    for H in hidden_dims:
        print(f"  H={H:>5}: {results[f'H{H}']:.4%}")


def _train_mlp_nanda_epochs(tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos,
                            device, input_dim, hidden_dim,
                            lr=1e-3, epochs=10, batch_size=1024):
    """Train single-hidden-layer MLP with Nanda's even/odd split."""
    mlp_even = _build_mlp(input_dim, hidden_dim, 64 * OPTIONS, 1).to(device)
    mlp_odd = _build_mlp(input_dim, hidden_dim, 64 * OPTIONS, 1).to(device)

    all_params = list(mlp_even.parameters()) + list(mlp_odd.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=1)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        mlp_even.train()
        mlp_odd.train()
        perm = torch.randperm(len(tr_X))
        for i in range(0, len(tr_X), batch_size):
            idx = perm[i:i + batch_size]
            x = tr_X[idx].to(device)
            y = tr_Y[idx].to(device)
            pos = tr_pos[idx]
            even_mask = (pos % 2 == 0)
            odd_mask = ~even_mask

            loss = torch.tensor(0.0, device=device)
            if even_mask.any():
                logits_e = mlp_even(x[even_mask]).view(-1, 64, OPTIONS)
                loss = loss + nn.functional.cross_entropy(
                    logits_e.reshape(-1, OPTIONS), y[even_mask].reshape(-1))
            if odd_mask.any():
                logits_o = mlp_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                loss = loss + nn.functional.cross_entropy(
                    logits_o.reshape(-1, OPTIONS), y[odd_mask].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval
        mlp_even.eval()
        mlp_odd.eval()
        correct = 0
        total = 0
        losses = []
        with torch.no_grad():
            for i in range(0, len(ev_X), batch_size):
                x = ev_X[i:i + batch_size].to(device)
                y = ev_Y[i:i + batch_size].to(device)
                pos = ev_pos[i:i + batch_size]
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask

                preds = torch.zeros_like(y)
                if even_mask.any():
                    logits_e = mlp_even(x[even_mask]).view(-1, 64, OPTIONS)
                    preds[even_mask] = logits_e.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits_e.reshape(-1, OPTIONS),
                        y[even_mask].reshape(-1)).item())
                if odd_mask.any():
                    logits_o = mlp_odd(x[odd_mask]).view(-1, 64, OPTIONS)
                    preds[odd_mask] = logits_o.argmax(-1)
                    losses.append(nn.functional.cross_entropy(
                        logits_o.reshape(-1, OPTIONS),
                        y[odd_mask].reshape(-1)).item())
                correct += (preds == y).sum().item()
                total += y.numel()

        acc = correct / total
        best_acc = max(best_acc, acc)
        scheduler.step(np.mean(losses))
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}/{epochs}: acc={acc:.4%}  loss={np.mean(losses):.5f}  lr={cur_lr:.2e}",
              flush=True)

    return best_acc


def do_plot(args):
    """Plot MLP results across hidden dims on log scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    acts_dir = args.results_dir

    # Collect results from all layers
    all_results = {}
    for layer in range(9):
        rfile = os.path.join(acts_dir, f'mlp_results_layer{layer}.json')
        if os.path.exists(rfile):
            with open(rfile) as f:
                data = json.load(f)
            all_results[layer] = data['results']

    if not all_results:
        print("No results found!")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for layer in sorted(all_results.keys()):
        results = all_results[layer]
        hs = sorted([int(k[1:]) for k in results.keys()])
        accs = [results[f'H{h}'] * 100 for h in hs]
        ax.plot(hs, accs, 'o-', label=f'Layer {layer}', markersize=5)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Hidden dimension (H)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('MLP Accuracy vs Hidden Dimension on Othello-GPT Activations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    ax.set_xticklabels([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])

    out_path = os.path.join(acts_dir, 'mlp_accuracy_vs_hidden_dim.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    plt.close()

    # Print summary table
    print(f"\n{'H':>6}", end='')
    for layer in sorted(all_results.keys()):
        print(f" | Layer {layer:>1}", end='')
    print()
    print('-' * (8 + 10 * len(all_results)))
    all_hs = set()
    for r in all_results.values():
        all_hs.update(int(k[1:]) for k in r.keys())
    for h in sorted(all_hs):
        print(f"{h:>6}", end='')
        for layer in sorted(all_results.keys()):
            acc = all_results[layer].get(f'H{h}')
            if acc is not None:
                print(f" | {acc*100:>6.2f}%", end='')
            else:
                print(f" |     N/A", end='')
        print()


def main():
    p = argparse.ArgumentParser(description='MLP on Othello-GPT activations')
    sub = p.add_subparsers(dest='command')

    # Extract
    ext = sub.add_parser('extract', help='Extract and save activations')
    ext.add_argument('--ckpt-path', default='ckpts/gpt_synthetic.ckpt')
    ext.add_argument('--max-games', type=int, default=1000000)
    ext.add_argument('--max-files', type=int, default=None)
    ext.add_argument('--output-dir', default='cached_acts')

    # Train
    trn = sub.add_parser('train', help='Train MLPs on cached activations')
    trn.add_argument('--acts-dir', required=True, help='Directory with cached activations')
    trn.add_argument('--layer', type=int, default=0)
    trn.add_argument('--hidden-dims', type=int, nargs='+',
                     default=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    trn.add_argument('--epochs', type=int, default=10)
    trn.add_argument('--batch-size', type=int, default=1024)

    # Plot
    plt_p = sub.add_parser('plot', help='Plot results')
    plt_p.add_argument('--results-dir', required=True)

    args = p.parse_args()
    if args.command == 'extract':
        do_extract(args)
    elif args.command == 'train':
        do_train(args)
    elif args.command == 'plot':
        do_plot(args)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
