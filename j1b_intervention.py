#!/usr/bin/env python
"""J1B (Othello-MLP tree bank) single-piece intervention experiment.

Mirrors the collaborator's OGPT intervention protocol so the outputs are
directly comparable.  For a sampled (game, position):

  * build J1B's 121-d canonical input  [played(60), placed_as_mover(60), 0]
    from the move prefix (playedeven_features, --canonicalize-mover), and read
    off J1B's 64 legal-move probabilities (LinearPatternProbOr prob-OR);
  * for each occupied movable cell, apply the counterfactual "flip this cell's
    colour".  In J1B's moveset+parity representation a cell's colour is the
    placed_as_mover bit, so the intervention is simply TOGGLE feat[60 + i]
    (i = C64_TO_C60[cell]).  Re-read J1B's 64 probs -> intv_probs;
  * ground-truth legality (Othello engine) for the true board and the
    colour-flipped counterfactual board -> original_legal / counterfactual_legal.

There is NO 120-vs-121 mismatch here: we build our native 121-d features and run
our own J1B forward.  Output JSON matches the multi_intervention_probs schema
(keyed '1' for single interventions) so OGPT vs J1B can be compared field-for-field.

Usage:
  python j1b_intervention.py \
    --bank banks/J1_perpattern.pt --readout stream_out/J1_B.pt \
    --n-games 2000 --positions 20 30 40 --max-flips-per-pos 8 \
    --out experiments/j1b_intervention/raw_samples.json
"""
import argparse, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, '.')
import train_streaming_probe as tsp
from opening_tree_mlp import LinearPatternProbOr, playedeven_features, C64_TO_C60
from data.othello import OthelloBoardState

CENTER = [27, 28, 35, 36]                       # never legal / not movable


def load_j1b(bank, readout, flanking_patterns, device):
    W_tree, b_tree, meta = tsp.load_trees(bank)
    mlp = tsp.OpeningTreeMLP(W_tree, b_tree, meta, device)
    leaf_build = tsp.load_leaf_build(bank)      # None for J1 (non-ordinal)
    patterns = tsp.load_patterns(flanking_patterns)
    ck = torch.load(readout, map_location=device, weights_only=False)
    state = ck['probe_state'] if 'probe_state' in ck else ck['probe_states'][0]
    hidden = state['linear.weight'].shape[1]
    probe = LinearPatternProbOr(hidden, patterns).to(device)
    probe.load_state_dict(state)
    probe.eval()
    print(f"J1B loaded: hidden_dim={hidden}, leaf_build={'yes' if leaf_build else 'none'}")
    return mlp, leaf_build, patterns, probe


def j1b_probs(X_np, mlp, patterns, probe, leaf_build, device, batch_size=4096):
    """(M, 121) float32 -> (M, 64) legal-move probs; center cells set to -1."""
    out = np.empty((X_np.shape[0], 64), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X_np.shape[0], batch_size):
            H = tsp.build_hidden_layer_batch(
                X_np[i:i + batch_size], mlp, patterns, None, False, device,
                no_flanking=True, leaf_build=leaf_build, leaf_index=None)
            if H.dtype != torch.float32:
                H = H.float()
            out[i:i + batch_size] = probe(H).cpu().numpy()
    out[:, CENTER] = -1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='banks/J1_perpattern.pt')
    ap.add_argument('--readout', default='stream_out/J1_B.pt')
    ap.add_argument('--flanking-patterns', default='hand_crafted_flanking_patterns.pt')
    ap.add_argument('--n-games', type=int, default=2000)
    ap.add_argument('--num-pickle-files', type=int, default=1)
    ap.add_argument('--positions', type=int, nargs='+', default=[20, 30, 40],
                    help='Move numbers (plies) at which to intervene.')
    ap.add_argument('--max-flips-per-pos', type=int, default=8,
                    help='Randomly flip at most this many occupied movable cells '
                         'per position (0 = all occupied movable cells).')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='experiments/j1b_intervention/raw_samples.json')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    mlp, leaf_build, patterns, probe = load_j1b(
        args.bank, args.readout, args.flanking_patterns, device)

    games = tsp.load_games(num_files=args.num_pickle_files)[:args.n_games] \
        if hasattr(tsp, 'load_games') else None
    if games is None:
        # fallback loader (same format as analyze_late_ambiguity)
        import pickle, glob
        files = sorted(glob.glob('./data/othello_synthetic/*.pickle'))[-args.num_pickle_files:]
        games = []
        for fp in files:
            with open(fp, 'rb') as fh:
                games.extend(pickle.load(fh))
        games = games[:args.n_games]
    print(f"{len(games)} games loaded")

    want = set(args.positions)
    # Collect every feature vector we need to score (orig once per position, one
    # intv per flip) then run J1B in big batches.
    X_rows = []            # feature vectors
    orig_idx = []          # index into X_rows for each sample's ORIGINAL probs
    intv_idx = []          # index into X_rows for each sample's INTERVENED probs
    meta = []              # per-sample metadata (everything except the probs)

    for gi, g in enumerate(games):
        board = OthelloBoardState()
        prefix = []
        for move in g:
            valid = board.get_valid_moves()
            if not valid:
                board.update([]); valid = board.get_valid_moves()
                if not valid:
                    break
            ply = len(prefix)
            if ply in want:
                X0 = playedeven_features(prefix, canonicalize_mover=True)  # (121,)
                oi = len(X_rows); X_rows.append(X0)
                state = board.state.flatten().astype(np.int8)             # (64,)
                mover = int(board.next_hand_color)
                orig_legal = sorted(int(m) for m in valid)
                # occupied movable cells (in the feature basis) are the flip targets
                occ = [c for c in range(64)
                       if state[c] != 0 and c in C64_TO_C60]
                if args.max_flips_per_pos and len(occ) > args.max_flips_per_pos:
                    occ = sorted(random.sample(occ, args.max_flips_per_pos))
                for c in occ:
                    i = C64_TO_C60[c]
                    Xc = X0.copy()
                    Xc[60 + i] = 1.0 - Xc[60 + i]        # toggle placed_as_mover
                    # counterfactual board: flip this piece's colour
                    cf = OthelloBoardState()
                    cf.state = board.state.copy()
                    cf.next_hand_color = board.next_hand_color
                    r, col = divmod(c, 8)
                    orig_color = int(cf.state[r, col]); cf.state[r, col] = -cf.state[r, col]
                    cf_legal = sorted(int(m) for m in cf.get_valid_moves())
                    ii = len(X_rows); X_rows.append(Xc)
                    orig_idx.append(oi); intv_idx.append(ii)
                    meta.append(dict(
                        game_idx=gi, pos=ply, color=mover,
                        modifications=[[r, col, orig_color, -orig_color]],
                        original_legal=orig_legal, counterfactual_legal=cf_legal,
                        n_original_legal=len(orig_legal), n_cf_legal=len(cf_legal),
                        n_changed=len(set(orig_legal) ^ set(cf_legal)),
                        board_state=[int(v) for v in state]))
            if move not in valid:
                break
            board.update([move]); prefix.append(move)
        if (gi + 1) % 500 == 0:
            print(f"  {gi+1}/{len(games)} games, {len(meta)} samples so far", flush=True)

    if not meta:
        print("No samples (check --positions vs game lengths)."); return
    print(f"scoring {len(X_rows):,} feature vectors through J1B ...", flush=True)
    probs = j1b_probs(np.stack(X_rows).astype(np.float32),
                      mlp, patterns, probe, leaf_build, device)

    samples = []
    for k, m in enumerate(meta):
        op = probs[orig_idx[k]]; ip = probs[intv_idx[k]]
        m = dict(m)
        m['orig_probs'] = [float(x) for x in op]
        m['intv_probs'] = [float(x) for x in ip]
        samples.append(m)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({'1': samples}, fh)     # keyed '1' = single-piece interventions
    # quick summary
    dp = np.array([np.array(s['intv_probs']) - np.array(s['orig_probs'])
                   for s in samples])
    tgt = np.array([s['modifications'][0][0] * 8 + s['modifications'][0][1]
                    for s in samples])
    print(f"\nsaved {len(samples):,} single-piece interventions -> {args.out}")
    print(f"  positions: {sorted(want)}   games: {len(games)}")
    print(f"  mean |Δprob| over all cells: {np.abs(dp).mean():.5f}")
    print(f"  mean n_changed (true legality flips per intervention): "
          f"{np.mean([s['n_changed'] for s in samples]):.2f}")


if __name__ == '__main__':
    main()
