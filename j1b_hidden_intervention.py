#!/usr/bin/env python
"""J1B (tree-bank Othello-MLP) PROBE-DIRECTED HIDDEN-LAYER intervention.

Fair analog of the OGPT intervention: instead of editing the input, we edit the
47,061-d hidden layer H along the board-decoding probe's colour axis, exactly
like the OGPT activation intervention:

  H' = H + alpha * sign * dir(c)          # sign flips cell c's colour
  dir(c) = normalize( probe[mode, :, c, BLACK] - probe[mode, :, c, WHITE] )
  sign   = +1 if cell c is currently white (push toward black), -1 if black

where probe = stream_out/J1_state_decoder.pt  (the 90.5% mine/yours state
decoder, shape (2 parity-modes, 47061, 64, 3), classes 0=empty/1=white/2=black),
and mode = ply % 2.  Legality is read out from H' via LinearPatternProbOr, and
scored against the true colour-flipped board (Othello engine).

Output matches the multi_intervention_probs schema (keyed '1'), with one entry
per (game, position, flipped cell, alpha) so J1B and OGPT are compared under the
same probe-directed protocol.

Usage:
  python j1b_hidden_intervention.py \
    --bank banks/J1_perpattern.pt --readout stream_out/J1_B.pt \
    --state-decoder stream_out/J1_state_decoder.pt \
    --n-games 2000 --positions 20 30 40 --max-flips-per-pos 8 --scales 1 2 4 8 \
    --out experiments/j1b_hidden_intervention/raw_samples.json
"""
import argparse, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, '.')
import train_streaming_probe as tsp
from opening_tree_mlp import LinearPatternProbOr, playedeven_features, C64_TO_C60
from data.othello import OthelloBoardState

CENTER = [27, 28, 35, 36]
EMPTY, WHITE, BLACK = 0, 1, 2          # probe class order (matches prep_intervention_example)


def load_j1b(bank, readout, flanking_patterns, device):
    W_tree, b_tree, meta = tsp.load_trees(bank)
    mlp = tsp.OpeningTreeMLP(W_tree, b_tree, meta, device)
    leaf_build = tsp.load_leaf_build(bank)
    patterns = tsp.load_patterns(flanking_patterns)
    ck = torch.load(readout, map_location=device, weights_only=False)
    state = ck['probe_state'] if 'probe_state' in ck else ck['probe_states'][0]
    hidden = state['linear.weight'].shape[1]
    probe = LinearPatternProbOr(hidden, patterns).to(device)
    probe.load_state_dict(state); probe.eval()
    return mlp, leaf_build, patterns, probe, hidden


def load_flip_dirs(state_decoder_path, hidden, device):
    """(2, 64, hidden) unit vectors: probe[mode,:,c,BLACK]-probe[mode,:,c,WHITE]."""
    ck = torch.load(state_decoder_path, map_location='cpu', weights_only=False)
    sp = ck['state_probe']                      # (2, H, 64, 3)
    sp = sp.numpy() if hasattr(sp, 'numpy') else np.asarray(sp)
    assert sp.shape[1] == hidden, f"probe hidden {sp.shape[1]} != readout hidden {hidden}"
    d = sp[:, :, :, BLACK] - sp[:, :, :, WHITE]   # (2, H, 64)
    d = np.transpose(d, (0, 2, 1))                # (2, 64, H)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    d = d / np.maximum(n, 1e-8)
    print(f"state decoder {state_decoder_path}: acc={ck.get('final_acc')}, dirs {d.shape}")
    return torch.tensor(d, dtype=torch.float32, device=device)   # (2, 64, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='banks/J1_perpattern.pt')
    ap.add_argument('--readout', default='stream_out/J1_B.pt')
    ap.add_argument('--state-decoder', default='stream_out/J1_state_decoder.pt')
    ap.add_argument('--flanking-patterns', default='hand_crafted_flanking_patterns.pt')
    ap.add_argument('--n-games', type=int, default=2000)
    ap.add_argument('--num-pickle-files', type=int, default=1)
    ap.add_argument('--positions', type=int, nargs='+', default=[20, 30, 40])
    ap.add_argument('--max-flips-per-pos', type=int, default=8)
    ap.add_argument('--scales', type=float, nargs='+', default=[1, 2, 4, 8],
                    help='alpha multipliers on the unit probe direction.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='experiments/j1b_hidden_intervention/raw_samples.json')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    mlp, leaf_build, patterns, probe, hidden = load_j1b(
        args.bank, args.readout, args.flanking_patterns, device)
    FLIP = load_flip_dirs(args.state_decoder, hidden, device)   # (2, 64, H)

    if hasattr(tsp, 'load_games'):
        games = tsp.load_games(num_files=args.num_pickle_files)[:args.n_games]
    else:
        import pickle, glob
        files = sorted(glob.glob('./data/othello_synthetic/*.pickle'))[-args.num_pickle_files:]
        games = []
        for fp in files:
            with open(fp, 'rb') as fh:
                games.extend(pickle.load(fh))
        games = games[:args.n_games]
    print(f"{len(games)} games loaded")

    def readout_probs(H):
        """H: (B, hidden) torch -> (B, 64) numpy, center = -1."""
        if H.dtype != torch.float32:
            H = H.float()
        with torch.no_grad():
            p = probe(H).cpu().numpy()
        p[:, CENTER] = -1.0
        return p

    want = set(args.positions)
    samples = []
    n_seen = 0
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
                X0 = playedeven_features(prefix, canonicalize_mover=True).astype(np.float32)
                mode = ply % 2                                # parity mode for the probe
                state = board.state.flatten().astype(np.int8)
                mover = int(board.next_hand_color)
                orig_legal = sorted(int(m) for m in valid)
                # J1B hidden for this position (batch of 1) then baseline probs.
                H0 = tsp.build_hidden_layer_batch(
                    X0[None, :], mlp, patterns, None, False, device,
                    no_flanking=True, leaf_build=leaf_build, leaf_index=None).float()  # (1,H)
                op = readout_probs(H0)[0]
                occ = [c for c in range(64) if state[c] != 0 and c in C64_TO_C60]
                if args.max_flips_per_pos and len(occ) > args.max_flips_per_pos:
                    occ = sorted(random.sample(occ, args.max_flips_per_pos))
                for c in occ:
                    cur = int(state[c])                       # +1 black / -1 white
                    sign = -1.0 if cur > 0 else 1.0           # push toward the other colour
                    dir_c = FLIP[mode, c]                     # (H,)
                    # counterfactual true board (flip this piece's colour)
                    cf = OthelloBoardState(); cf.state = board.state.copy()
                    cf.next_hand_color = board.next_hand_color
                    r, col = divmod(c, 8); cf.state[r, col] = -cf.state[r, col]
                    cf_legal = sorted(int(m) for m in cf.get_valid_moves())
                    for a in args.scales:
                        Ha = H0 + (a * sign) * dir_c          # (1,H) intervene on hidden
                        ip = readout_probs(Ha)[0]
                        samples.append(dict(
                            game_idx=gi, pos=ply, color=mover, alpha=float(a),
                            modifications=[[r, col, cur, -cur]],
                            original_legal=orig_legal, counterfactual_legal=cf_legal,
                            n_original_legal=len(orig_legal), n_cf_legal=len(cf_legal),
                            n_changed=len(set(orig_legal) ^ set(cf_legal)),
                            board_state=[int(v) for v in state],
                            orig_probs=[float(x) for x in op],
                            intv_probs=[float(x) for x in ip]))
            if move not in valid:
                break
            board.update([move]); prefix.append(move)
        n_seen += 1
        if n_seen % 250 == 0:
            print(f"  {n_seen}/{len(games)} games, {len(samples)} samples", flush=True)

    if not samples:
        print("No samples."); return
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({'1': samples}, fh)
    dp = np.array([np.array(s['intv_probs']) - np.array(s['orig_probs']) for s in samples])
    print(f"\nsaved {len(samples):,} probe-directed hidden interventions -> {args.out}")
    print(f"  positions {sorted(want)}  scales {args.scales}")
    print(f"  mean |Δprob| (movable): {np.abs(dp[dp > -1]).mean():.4f}")


if __name__ == '__main__':
    main()
