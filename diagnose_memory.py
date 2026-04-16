"""Diagnose where memory is growing during pattern detector training.

Logs DETAILED memory breakdown after each chunk:
  - VmRSS (psutil RSS)
  - VmData (heap + data segment)
  - VmHWM (peak RSS)
  - VmSize (virtual memory)
  - Cgroup memory.usage
  - CUDA allocated
  - CUDA reserved (cached by allocator)
  - Python heap objects count

If VmRSS and cgroup diverge, we'll see where the cgroup memory is
going (likely page cache, pinned memory, or CUDA).

Usage:
    python diagnose_memory.py --mode direct --n-chunks 5
"""
import sys, os, gc, time, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn


def read_proc_status():
    """Read memory-related fields from /proc/self/status."""
    d = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("Vm", "RssAnon", "RssFile", "RssShmem")):
                key, val = line.split(":", 1)
                # val is like "12345 kB"
                try:
                    d[key] = int(val.strip().split()[0]) * 1024  # bytes
                except (ValueError, IndexError):
                    pass
    return d


def read_cgroup_memory():
    """Read cgroup memory.usage if available."""
    # Try cgroup v1 then v2
    paths = [
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory.current",
    ]
    for p in paths:
        try:
            with open(p) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            continue
    return None


def log_memory(label):
    status = read_proc_status()
    cgroup = read_cgroup_memory()
    cuda_alloc = cuda_reserved = 0
    if torch.cuda.is_available():
        cuda_alloc = torch.cuda.memory_allocated()
        cuda_reserved = torch.cuda.memory_reserved()
    n_objects = len(gc.get_objects())

    print(f"[{label}]", flush=True)
    for key in ["VmRSS", "VmData", "VmHWM", "VmSize", "RssAnon", "RssFile", "RssShmem"]:
        if key in status:
            print(f"  {key:12s}= {status[key] / 1e9:.2f} GB", flush=True)
    if cgroup is not None:
        print(f"  {'cgroup':12s}= {cgroup / 1e9:.2f} GB", flush=True)
    print(f"  {'cuda_alloc':12s}= {cuda_alloc / 1e9:.2f} GB", flush=True)
    print(f"  {'cuda_resvd':12s}= {cuda_reserved / 1e9:.2f} GB", flush=True)
    print(f"  {'py_objects':12s}= {n_objects}", flush=True)


from train_pattern_detectors import (
    _load_features_when60, _get_pattern_labels,
    labels_to_192d, BoardStateMLP, PatternDetectors, EndToEndModel,
    N_MOVES, OPTIONS,
)
from hand_crafted_flanking import enumerate_flanking_patterns
from generate_rule_games import precompute_pattern_arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="direct",
                        choices=["two-stage", "end-to-end", "direct"])
    parser.add_argument("--n-chunks", type=int, default=5,
                        help="Number of chunks to iterate")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--chunk-dir",
                        default="experiments/mathematical_transformation_experiments/"
                                "heuristic_probe_results/feature_chunks")
    args = parser.parse_args()

    log_memory("startup")

    # Setup
    patterns = enumerate_flanking_patterns()
    pat_targets, pat_terminals, pat_opp_cells, pat_opp_mask = precompute_pattern_arrays(patterns)
    feature_cols = list(range(N_MOVES, 2 * N_MOVES))

    chunk_files = sorted(os.path.join(args.chunk_dir, f)
                         for f in os.listdir(args.chunk_dir)
                         if f.endswith(".npz") and "_patterns" not in f
                         and "_when60" not in f)
    chunk_files = chunk_files[:args.n_chunks]
    print(f"Will iterate {len(chunk_files)} chunks")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build models based on mode
    if args.mode == "direct":
        mlp_even = nn.Sequential(
            nn.Linear(60, args.hidden), nn.ReLU(),
            nn.Linear(args.hidden, 960)).to(device)
        mlp_odd = nn.Sequential(
            nn.Linear(60, args.hidden), nn.ReLU(),
            nn.Linear(args.hidden, 960)).to(device)
        params = list(mlp_even.parameters()) + list(mlp_odd.parameters())
    elif args.mode == "end-to-end":
        m_e = EndToEndModel(60, args.hidden, 960).to(device)
        m_o = EndToEndModel(60, args.hidden, 960).to(device)
        params = list(m_e.parameters()) + list(m_o.parameters())
    elif args.mode == "two-stage":
        mlp_e = BoardStateMLP(60, args.hidden).to(device)
        mlp_o = BoardStateMLP(60, args.hidden).to(device)
        for p in mlp_e.parameters(): p.requires_grad = False
        for p in mlp_o.parameters(): p.requires_grad = False
        detectors = PatternDetectors(192, 960).to(device)
        params = list(detectors.parameters())

    opt = torch.optim.Adam(params, lr=1e-3)

    log_memory("after model build")

    # Iterate chunks
    for ci, path in enumerate(chunk_files):
        t0 = time.time()
        tr_X, tr_Y, tr_pos = _load_features_when60(path, feature_cols)
        pat_labels = _get_pattern_labels(
            path, tr_Y, tr_pos, pat_targets, pat_terminals,
            pat_opp_cells, pat_opp_mask)

        log_memory(f"chunk {ci} after load")

        # Run a few hundred batches
        perm = torch.randperm(len(tr_X))
        n_batches = min(500, len(tr_X) // args.batch_size)

        for bi in range(n_batches):
            i = bi * args.batch_size
            idx = perm[i:i + args.batch_size]
            x = tr_X[idx].to(device)
            pos = tr_pos[idx]
            y_pat = pat_labels[idx].float().to(device)

            if args.mode == "direct":
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask
                loss = torch.tensor(0.0, device=device)
                if even_mask.any():
                    logits = mlp_even(x[even_mask])
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(
                        logits, y_pat[even_mask])
                if odd_mask.any():
                    logits = mlp_odd(x[odd_mask])
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(
                        logits, y_pat[odd_mask])
            elif args.mode == "end-to-end":
                y_board = tr_Y[idx].to(device)
                even_mask = (pos % 2 == 0)
                odd_mask = ~even_mask
                loss = torch.tensor(0.0, device=device)
                for mask, model in [(even_mask, m_e), (odd_mask, m_o)]:
                    if not mask.any():
                        continue
                    pl, bl = model(x[mask], pos[mask])
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(pl, y_pat[mask])
                    loss = loss + 0.5 * nn.functional.cross_entropy(
                        bl.reshape(-1, OPTIONS), y_board[mask].reshape(-1))
            elif args.mode == "two-stage":
                with torch.no_grad():
                    even_mask = (pos % 2 == 0)
                    odd_mask = ~even_mask
                    pred_labels = torch.zeros(len(idx), 64, dtype=torch.long, device=device)
                    if even_mask.any():
                        pred_labels[even_mask] = mlp_e.predict_board(x[even_mask])
                    if odd_mask.any():
                        pred_labels[odd_mask] = mlp_o.predict_board(x[odd_mask])
                encoding = labels_to_192d(pred_labels, pos)
                logits = detectors(encoding)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y_pat)

            opt.zero_grad()
            loss.backward()
            opt.step()

        log_memory(f"chunk {ci} after {n_batches} batches")

        del tr_X, tr_Y, tr_pos, pat_labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_memory(f"chunk {ci} after cleanup")
        print(f"  chunk {ci} took {time.time()-t0:.1f}s\n", flush=True)


if __name__ == "__main__":
    main()
