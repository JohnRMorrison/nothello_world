"""
Train a linear probe on the frozen MLP hidden layer to decode board state.
Same as Nanda's decoder: it tests whether the MLP's hidden representation
encodes enough information to reconstruct the board.

The probe is Linear(H, 64*3), predicting {empty, mine, opponent} for each of
the 64 cells. Two probes are trained: one for even turns (white to move) and
one for odd turns (black to move). 

Requires a trained pattern detector checkpoint from train_pattern_clean.py.

Usage:
  python probe_pattern_clean.py \
      --ckpt experiments/.../pattern_clean_H4096.pt \
      --hidden 4096 --epochs 5
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn

from train_pattern_clean import DirectMLP, load_chunk


def board_labels_to_mine_opp(board_labels, positions):
    # convert board state to turn based labels.
    # labels use absolute colors: 0=empty, 1=white, 2=black.
    # probe learns single concept (empty/mine/opponent) regardless of color,
    # so we remap based on whose turn it is:
    #   even turn (white to move): white=mine(1), black=opponent(2) : no change
    #   odd turn  (black to move): black=mine(1), white=opponent(2) : swap 1 and 2
    
    labels = board_labels.clone()
    odd = (positions % 2 == 1)
    if odd.any():
        # in black move positions --> swap white(1) and black(2)
        orig = board_labels[odd]
        labels[odd] = torch.where(orig == 1, torch.tensor(2),
                      torch.where(orig == 2, torch.tensor(1), orig))
    return labels


def get_hidden(model, x):
    # extract hidden activations --> output of the first Linear+ReLU layer.
    return torch.relu(model.net[0](x))


def train_probe(chunk_dir, chunk_prefix, device, mlp_even, mlp_odd,
                hidden_dim, epochs=5, batch_size=1024, lr=1e-3, save_path=None):

    chunk_files = sorted(
        os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir)
        if f.startswith(chunk_prefix) and f.endswith(".npz")
    )
    if not chunk_files:
        raise ValueError(f"No {chunk_prefix}*.npz files found in {chunk_dir}")

    eval_path   = chunk_files[-1]
    train_paths = chunk_files[:-1]
    n_outputs   = 64 * 3  # bc 3 classes per cell

    print(f"{len(train_paths)} train chunks, eval on {os.path.basename(eval_path)}")
    print(f"H={hidden_dim}, epochs={epochs}")

    probe_even = nn.Linear(hidden_dim, n_outputs).to(device)
    probe_odd  = nn.Linear(hidden_dim, n_outputs).to(device)
    optimizer  = torch.optim.Adam(list(probe_even.parameters()) +
                                  list(probe_odd.parameters()), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=1)

    # pre-load and sample eval data
    ev_X, ev_Y, ev_pos = load_chunk(eval_path)
    rng = np.random.RandomState(0)
    sample_idx = np.sort(rng.choice(len(ev_X), min(len(ev_X), 49 * 10000), replace=False))
    ev_X   = ev_X[sample_idx]
    ev_Y   = ev_Y[sample_idx]
    ev_pos = ev_pos[sample_idx]
    print(f"Eval: {len(ev_X)} samples")

    best_acc   = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        probe_even.train()
        probe_odd.train()
        rng = np.random.RandomState(epoch)
        chunk_order = rng.permutation(len(train_paths))
        total_loss = 0.0
        n_batches  = 0

        for ci, path_idx in enumerate(chunk_order):
            print(f"  Epoch {epoch}: chunk {ci + 1}/{len(chunk_order)}", flush=True)
            tr_X, tr_Y, tr_pos = load_chunk(train_paths[path_idx])
            perm = torch.randperm(len(tr_X))

            for i in range(0, len(tr_X), batch_size):
                idx     = perm[i:i + batch_size]
                x       = tr_X[idx].to(device)
                y_board = tr_Y[idx].to(device)
                pos     = tr_pos[idx]

                targets = board_labels_to_mine_opp(y_board, pos.to(device))

                even_mask = (pos % 2 == 0).to(device)
                odd_mask  = ~even_mask
                loss = torch.tensor(0.0, device=device)

                for mask, mlp, probe in [(even_mask, mlp_even, probe_even),
                                         (odd_mask,  mlp_odd,  probe_odd)]:
                    if not mask.any():
                        continue
                    with torch.no_grad():
                        h = get_hidden(mlp, x[mask])
                    logits = probe(h).view(-1, 64, 3)
                    loss = loss + nn.functional.cross_entropy(
                        logits.reshape(-1, 3), targets[mask].reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            del tr_X, tr_Y, tr_pos

        # eval
        probe_even.eval()
        probe_odd.eval()
        correct = total = 0
        per_cell_correct = np.zeros(64, dtype=np.int64)
        n_eval_rows = 0

        with torch.no_grad():
            for i in range(0, len(ev_X), batch_size):
                x       = ev_X[i:i + batch_size].to(device)
                y_board = ev_Y[i:i + batch_size].to(device)
                pos     = ev_pos[i:i + batch_size]

                targets = board_labels_to_mine_opp(y_board, pos.to(device))

                preds = torch.zeros_like(targets)
                for mask, mlp, probe in [(pos % 2 == 0, mlp_even, probe_even),
                                         (pos % 2 == 1, mlp_odd,  probe_odd)]:
                    mask = mask.to(device)
                    if not mask.any():
                        continue
                    h      = get_hidden(mlp, x[mask])
                    logits = probe(h).view(-1, 64, 3)
                    preds[mask] = logits.argmax(dim=-1)

                correct += (preds == targets).sum().item()
                total   += targets.numel()
                per_cell_correct += (preds == targets).sum(dim=0).cpu().numpy()
                n_eval_rows += len(targets)

        acc          = correct / total
        per_cell_acc = per_cell_correct / max(n_eval_rows, 1)
        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step(avg_loss)
        print(f"  Epoch {epoch}: acc={acc:.4%}  loss={avg_loss:.5f}", flush=True)

        if acc > best_acc:
            best_acc = acc
            best_state = {
                'even':         {k: v.cpu().clone() for k, v in probe_even.state_dict().items()},
                'odd':          {k: v.cpu().clone() for k, v in probe_odd.state_dict().items()},
                'per_cell_acc': per_cell_acc,
            }

        if best_state and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({**best_state,
                        'hidden_dim': hidden_dim,
                        'best_acc':   best_acc,
                        'epoch':      epoch}, save_path)
            print(f"  Saved {save_path}", flush=True)

    return best_acc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         required=True,
                   help="Path to pattern detector checkpoint (from train_pattern_clean.py)")
    p.add_argument("--hidden",       type=int, required=True,
                   help="Hidden dim H — must match the checkpoint")
    p.add_argument("--epochs",       type=int, default=5)
    p.add_argument("--chunk-prefix", default="chunk_")
    p.add_argument("--output-dir",   default=
        "experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the frozen MLP
    ckpt = torch.load(args.ckpt, map_location=device)
    input_dim  = ckpt.get('input_dim', 120)
    n_patterns = ckpt.get('n_patterns', 960)

    mlp_even = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    mlp_odd  = DirectMLP(input_dim, args.hidden, n_patterns).to(device)
    mlp_even.load_state_dict(ckpt['even'])
    mlp_odd.load_state_dict(ckpt['odd'])

    for p_ in list(mlp_even.parameters()) + list(mlp_odd.parameters()):
        p_.requires_grad = False
    mlp_even.eval()
    mlp_odd.eval()
    print(f"Loaded MLP from {args.ckpt} (best_acc={ckpt.get('best_acc', '?'):.4%})")

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    save_dir  = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    save_path = os.path.join(save_dir, f"probe_{ckpt_name}.pt")

    train_probe(chunk_dir, args.chunk_prefix, device, mlp_even, mlp_odd,
                hidden_dim=args.hidden, epochs=args.epochs, save_path=save_path)
