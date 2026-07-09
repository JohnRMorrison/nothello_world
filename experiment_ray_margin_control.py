"""Turn-matched control for ray-margin at t_bias.

For each qualified adversarial position (game, C, t_bias) from the
ray-margin CSV at a given threshold, sample n_controls random val
games with length >= t_bias + 1.  At each control position, run the
layer-6 probe (parity-aware) and compute the min margin over the SAME
cell C's 8 rays (identical ray geometry, identical margin definition).

No P(C) or legality condition on controls -- they are ordinary
positions at the same turn, evaluated around the same cell.

Reports:
  - Control fraction with min ray margin > 0, > +1, distribution
  - Side-by-side with adversarial t_bias numbers
  - Paired: adversarial min minus mean of its matched controls

Per-control CSV: adv_game_id, control_game_id, turn, C,
                 min_ray_margin, mean_ray_margin
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import (
    tokenize_games, VOCAB_SIZE, extract_activations,
)
from experiment_probe_causal_analysis import DIRS, ray_cells_in_direction


def gt_probe_class(state_8x8, cell):
    v = state_8x8[cell // 8, cell % 8]
    if v == 0: return 0
    if v == -1: return 1
    return 2


def compute_ray_min_and_mean(hidden_512, turn, probe, C, state):
    """Return (min, mean) of probe logit margin over cells on C's 8 rays."""
    mode = 0 if turn % 2 == 1 else 1
    W = probe[mode]
    h = torch.from_numpy(hidden_512).float()
    logits = torch.einsum('d,drco->rco', h, W).detach().cpu().numpy()
    min_m = np.inf
    total_m = 0.0
    n = 0
    for direction in DIRS:
        for cell in ray_cells_in_direction(C, direction):
            r_, c_ = cell // 8, cell % 8
            lg = logits[r_, c_]
            gt_cls = gt_probe_class(state, cell)
            other = [lg[j] for j in range(3) if j != gt_cls]
            m = float(lg[gt_cls] - max(other))
            if m < min_m:
                min_m = m
            total_m += m
            n += 1
    return min_m, (total_m / n if n > 0 else 0.0)


def load_val_games(data_dir, num_files, min_length=1):
    files = sorted(os.listdir(data_dir))[-num_files:]
    games = []
    for fname in files:
        p = os.path.join(data_dir, fname)
        try:
            with open(p, 'rb') as f:
                batch = pickle.load(f)
        except Exception:
            continue
        for g in batch:
            if len(g) >= min_length:
                games.append(g)
    return games


def load_adv_positions(csv_path, threshold):
    """Return list of dicts with only qualified positions at the target threshold."""
    positions = []
    with open(csv_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip().split(',')
            if len(parts) < 6:
                continue
            gid, T, C, thresh_str, t_bias_str, m_str = parts
            if not t_bias_str.strip() or not m_str.strip():
                continue
            try:
                if abs(float(thresh_str) - threshold) > 1e-9:
                    continue
                positions.append({
                    'game_id': int(gid),
                    'T': int(T),
                    'C': int(C),
                    't_bias': int(t_bias_str),
                    'adv_min_margin': float(m_str),
                })
            except ValueError:
                continue
    return positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ray-margin-csv', default='ray_margin_23k.csv',
                    help='CSV from experiment_ray_margin_at_bias.py')
    ap.add_argument('--threshold', type=float, default=0.01,
                    help='Threshold to filter the ray-margin CSV rows')
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe',
                    default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--data-dir', default='./data/othello_synthetic')
    ap.add_argument('--num-data-files', type=int, default=3)
    ap.add_argument('--n-controls', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--output-csv', default='ray_margin_control.csv')
    ap.add_argument('--output-summary', default='ray_margin_control.txt')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    sd = torch.load(args.ckpt, map_location=device)
    block_size = sd["pos_emb"].shape[1]
    config = GPTConfig(VOCAB_SIZE, block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    probe = torch.load(args.probe, map_location='cpu')

    adv_positions = load_adv_positions(args.ray_margin_csv, args.threshold)
    print(f"Loaded {len(adv_positions)} adversarial positions at "
          f"threshold {args.threshold}")
    if args.limit > 0:
        adv_positions = adv_positions[:args.limit]
        print(f"Truncated to {len(adv_positions)} via --limit")
    if not adv_positions:
        print("No adversarial positions to process; exiting.")
        return

    max_t_bias = max(p['t_bias'] for p in adv_positions)
    print(f"Loading val games (need length >= {max_t_bias + 1} for the "
          f"largest t_bias in the set)")
    val_games = load_val_games(args.data_dir, args.num_data_files,
                                 min_length=1)
    print(f"Loaded {len(val_games)} val games total")
    val_lens = np.array([len(g) for g in val_games])

    rng = np.random.RandomState(args.seed)

    # Cache: control_game_idx -> layer-6 activations (block_size, 512)
    act_cache = {}

    def get_control_acts(ctrl_idx):
        if ctrl_idx in act_cache:
            return act_cache[ctrl_idx]
        g = val_games[ctrl_idx]
        L = min(len(g), block_size)
        tokens = tokenize_games([g[:L]], seq_len=block_size).to(device)
        with torch.no_grad():
            acts = extract_activations(model, tokens, args.layer)
        arr = acts.cpu().numpy()[0]
        act_cache[ctrl_idx] = arr
        return arr

    csv_rows = [['adv_game_id', 'control_game_id', 'turn', 'C',
                 'min_ray_margin', 'mean_ray_margin']]
    control_min_margins = []
    control_mean_margins = []
    paired_diffs = []
    skipped_no_controls = 0

    t0 = time.time()
    for pi, pos in enumerate(adv_positions):
        t_bias = pos['t_bias']
        C = pos['C']
        adv_game_id = pos['game_id']
        adv_min_margin = pos['adv_min_margin']

        eligible = np.where(val_lens >= t_bias + 1)[0]
        if len(eligible) < args.n_controls:
            skipped_no_controls += 1
            continue
        chosen = rng.choice(eligible, size=args.n_controls, replace=False)

        matched_mins = []
        for ctrl_idx in chosen:
            ctrl_idx = int(ctrl_idx)
            ctrl_game = val_games[ctrl_idx]
            board = OthelloBoardState()
            valid = True
            for t in range(0, t_bias + 1):
                try:
                    board.umpire(ctrl_game[t])
                except Exception:
                    valid = False
                    break
            if not valid:
                continue
            state = np.asarray(board.state, dtype=np.int8)

            acts_np = get_control_acts(ctrl_idx)
            min_m, mean_m = compute_ray_min_and_mean(
                acts_np[t_bias], t_bias, probe, C, state)
            csv_rows.append([adv_game_id, ctrl_idx, t_bias, C,
                              f"{min_m:.4f}", f"{mean_m:.4f}"])
            control_min_margins.append(min_m)
            control_mean_margins.append(mean_m)
            matched_mins.append(min_m)

        if matched_mins:
            paired_diffs.append(adv_min_margin - float(np.mean(matched_mins)))

        if (pi + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (len(adv_positions) - pi - 1) / (pi + 1)
            print(f"  {pi+1}/{len(adv_positions)}  ({elapsed:.0f}s, "
                   f"~{eta:.0f}s remaining)  cache={len(act_cache)}",
                   flush=True)

    with open(args.output_csv, 'w') as f:
        for row in csv_rows:
            f.write(','.join(str(x) for x in row) + '\n')
    print(f"Wrote per-control CSV to {args.output_csv}")

    lines = []
    lines.append(f"Threshold: P(C) > {args.threshold * 100:.2f}%")
    lines.append(f"Adversarial positions used:  {len(adv_positions) - skipped_no_controls}")
    lines.append(f"Skipped (not enough val games at that turn): "
                  f"{skipped_no_controls}")
    lines.append(f"Control positions computed:  {len(control_min_margins)}")
    lines.append("")
    if control_min_margins:
        cm = np.array(control_min_margins)
        lines.append("=== CONTROL (turn-matched, same cell C, no bias/legality filter) ===")
        lines.append(f"  Fraction min ray margin > 0:   "
                      f"{(cm > 0).sum() / len(cm) * 100:5.1f}%")
        lines.append(f"  Fraction min ray margin > +1:  "
                      f"{(cm > 1).sum() / len(cm) * 100:5.1f}%")
        lines.append(f"  Distribution of min ray margin:")
        lines.append(f"    median={np.median(cm):+.3f}  mean={cm.mean():+.3f}")
        for q in [10, 25, 50, 75, 90]:
            lines.append(f"    p{q}={np.percentile(cm, q):+.3f}")
    lines.append("")
    if paired_diffs:
        pd = np.array(paired_diffs)
        lines.append("=== PAIRED: adversarial min minus mean of matched control mins ===")
        lines.append(f"    median={np.median(pd):+.3f}  mean={pd.mean():+.3f}")
        for q in [10, 25, 50, 75, 90]:
            lines.append(f"    p{q}={np.percentile(pd, q):+.3f}")
    lines.append("")
    lines.append("(Compare to the adversarial ray-margin numbers from "
                  "experiment_ray_margin_at_bias.py.)")

    with open(args.output_summary, 'w') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote summary to {args.output_summary}")


if __name__ == '__main__':
    main()
