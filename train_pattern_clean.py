"""
Train an MLP to detect the 960 flanking patterns.

Architecture: Linear(input_dim, H) -> ReLU -> Linear(H, 960)

Loads precomputed chunks from precompute_chunks_clean.py
Pattern labels are computed on the fly and per batch from the stored board state 
Cant precompute because of limited disk space.

Default we train two separate models: one for even turns (white to move)
and one for odd turns (black to move). 
Use "--no-split-parity" for a single shared model.

Usage:
  python train_pattern_clean.py --hidden 4096 --epochs 1
  python train_pattern_clean.py --hidden 4096 --epochs 1 --no-split-parity
"""
import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn

from hand_crafted_flanking import enumerate_flanking_patterns
from generate_rule_games import precompute_pattern_arrays


def load_chunk(path):
    data = np.load(path)
    X   = torch.tensor(data['features'].astype(np.float32))
    Y   = torch.tensor(data['labels'].astype(np.int64))
    pos = torch.tensor(data['positions'].astype(np.int64))
    return X, Y, pos


def compute_pattern_labels(board_labels, positions, targets, terminals, opp_cells, opp_mask):
    # convert board state (N,64) + turn numbers (N,) into pattern firing labels (N,960).

    # pattern fires when: target cell is empty, opponent cells hold opponent pieces,
    # and the terminal cell holds a friendly piece. Basically when legal flanking move exists.
        
    n = len(board_labels)
    # remap board labels from {0=empty,1=white,2=black} to {0=empty,-1=white,1=black}
    flat = np.zeros((n, 64), dtype=np.int8)
    flat[board_labels == 1] = -1
    flat[board_labels == 2] = 1

    is_black = (positions % 2 == 1)
    target_empty = (flat[:, targets] == 0)
    pattern_labels = np.zeros((n, len(targets)), dtype=np.float32)

    for playing_black, mask in [(True, is_black), (False, ~is_black)]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        my_val  =  1 if playing_black else -1
        opp_val = -my_val
        sub = flat[idx]
        fires = (target_empty[idx]
                 & (sub[:, terminals] == my_val)
                 & ((sub[:, opp_cells] == opp_val) | ~opp_mask[None, :, :]).all(axis=2))
        pattern_labels[idx] = fires.astype(np.float32)

    return pattern_labels


class DirectMLP(nn.Module):
    # simple two-layer MLP: input -> hidden (ReLU) -> 960 pattern logits.
    def __init__(self, input_dim, hidden_dim, n_patterns=960):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_patterns),
        )

    def forward(self, x):
        return self.net(x)


def train(chunk_dir, chunk_prefix, device, input_dim, hidden_dim,
          pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask,
          epochs=1, batch_size=1024, lr=1e-3, pos_weight=None,
          split_parity=True, save_path=None):

    chunk_files = sorted(
        os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir)
        if f.startswith(chunk_prefix) and f.endswith(".npz")
    )
    if not chunk_files:
        raise ValueError(f"No {chunk_prefix}*.npz files found in {chunk_dir}")

    # last chunk held out for eval and the rest used for training
    eval_path   = chunk_files[-1]
    train_paths = chunk_files[:-1]
    n_patterns  = len(pat_targets)

    print(f"{len(train_paths)} train chunks, eval on {os.path.basename(eval_path)}")
    print(f"H={hidden_dim}, input_dim={input_dim}, epochs={epochs}, split_parity={split_parity}")

    model_even = DirectMLP(input_dim, hidden_dim, n_patterns).to(device)
    model_odd  = DirectMLP(input_dim, hidden_dim, n_patterns).to(device) if split_parity else model_even

    all_params = list(model_even.parameters())
    if model_odd is not model_even:
        all_params += list(model_odd.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=1)

    pw = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    if pw is not None:
        print(f"pos_weight={pos_weight}")

    # pre-load eval data and take random sample across all turn numbers
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
        model_even.train()
        model_odd.train()
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
                y_board = tr_Y[idx]
                pos     = tr_pos[idx]

                y_pat = torch.from_numpy(
                    compute_pattern_labels(y_board.numpy(), pos.numpy(),
                                          pat_targets, pat_terminals,
                                          pat_opp_cells, pat_opp_mask)
                ).to(device)

                even_mask = (pos % 2 == 0)
                odd_mask  = ~even_mask
                loss = torch.tensor(0.0, device=device)

                for mask, model in [(even_mask, model_even), (odd_mask, model_odd)]:
                    if not mask.any():
                        continue
                    logits = model(x[mask])
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(
                        logits, y_pat[mask], pos_weight=pw)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            del tr_X, tr_Y, tr_pos

        # eval
        model_even.eval()
        model_odd.eval()
        correct = tp = fp = fn = total = 0

        with torch.no_grad():
            for i in range(0, len(ev_X), batch_size):
                x       = ev_X[i:i + batch_size].to(device)
                y_board = ev_Y[i:i + batch_size]
                pos     = ev_pos[i:i + batch_size]

                y_pat = torch.from_numpy(
                    compute_pattern_labels(y_board.numpy(), pos.numpy(),
                                          pat_targets, pat_terminals,
                                          pat_opp_cells, pat_opp_mask)
                ).to(device)

                preds = torch.zeros_like(y_pat)
                if (pos % 2 == 0).any():
                    mask = (pos % 2 == 0)
                    preds[mask] = (model_even(x[mask]) > 0).float()
                if (pos % 2 == 1).any():
                    mask = (pos % 2 == 1)
                    preds[mask] = (model_odd(x[mask]) > 0).float()

                correct += (preds == y_pat).sum().item()
                total   += y_pat.numel()
                tp += ((preds == 1) & (y_pat == 1)).sum().item()
                fp += ((preds == 1) & (y_pat == 0)).sum().item()
                fn += ((preds == 0) & (y_pat == 1)).sum().item()

        acc       = correct / total
        recall    = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        avg_loss  = total_loss / max(n_batches, 1)
        lr_now    = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: acc={acc:.4%}  recall={recall:.4%}  "
              f"prec={precision:.4%}  loss={avg_loss:.5f}  lr={lr_now:.2e}", flush=True)

        if acc > best_acc:
            best_acc = acc
            best_state = {
                'even': {k: v.cpu().clone() for k, v in model_even.state_dict().items()},
                'odd':  {k: v.cpu().clone() for k, v in model_odd.state_dict().items()},
            }

        scheduler.step(avg_loss)

        if best_state and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({**best_state,
                        'feature_type': 'played+even',
                        'hidden_dim': hidden_dim,
                        'input_dim':  input_dim,
                        'n_patterns': n_patterns,
                        'best_acc':   best_acc,
                        'epoch':      epoch}, save_path)
            print(f"  Saved {save_path}", flush=True)

    return best_acc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hidden",          type=int,   default=4096)
    p.add_argument("--epochs",          type=int,   default=1)
    p.add_argument("--pos-weight",      type=float, default=None,
                   help="Upweight positive class in BCE loss (e.g. 50). "
                        "Boosts recall at the cost of precision.")
    p.add_argument("--no-split-parity", action="store_true",
                   help="Train one shared model instead of separate even/odd models")
    p.add_argument("--chunk-prefix",    default="chunk_",
                   help="Prefix for chunk filenames (default: chunk_)")
    p.add_argument("--output-dir",      default=
        "experiments/mathematical_transformation_experiments/heuristic_probe_results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    print(f"Patterns: {len(patterns)}")

    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    save_dir  = os.path.join(args.output_dir, "pattern_detector_checkpoints")
    save_name = f"pattern_clean_H{args.hidden}"
    if args.no_split_parity:
        save_name += "_single"
    save_path = os.path.join(save_dir, save_name + ".pt")

    train(chunk_dir, args.chunk_prefix, device,
          input_dim=120, hidden_dim=args.hidden,
          pat_targets=pat_targets, pat_terminals=pat_terminals,
          pat_opp_cells=pat_opp_cells, pat_opp_mask=pat_opp_mask,
          epochs=args.epochs, pos_weight=args.pos_weight,
          split_parity=not args.no_split_parity,
          save_path=save_path)
