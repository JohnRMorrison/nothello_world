#!/usr/bin/env python
"""J1B (tree-bank Othello-MLP): probe-directed hidden intervention, matched to
`mechanistic_interpretability/multi_intervention.py`'s OGPT protocol.

Same experiment as OGPT, on J1B's 47,061-d hidden layer:
  * 5 conditions x N in {1,2,3,5,8} simultaneous edits (reusing multi_intervention's
    selectors: flip/add x interacting/non-interacting, + mixed),
  * intervention = probe-direction PROJECTION  H' = H - Σ_i s_i (H·d̂_i) d̂_i,
    with s_i per-cell CALIBRATED to the minimal scale that flips the J1
    state-decoder's argmax at the modified cell (binary search, +10%, cap 10) --
    identical to OGPT's compute_per_cell_scales,
  * d̂_i from stream_out/J1_state_decoder.pt (2 parity-modes, 47061, 64, 3;
    classes 0=empty/1=white/2=black).

Metrics (per intervention) -- directly comparable to multi_intervention.py:
  probe_acc            fraction of modified cells whose state-decode argmax flips to target
  direction_acc_probe  fraction where the target-class logit rises vs current
  crosstalk            mean |Δ state-logit| on NON-modified cells
Plus legal-move metrics via the LinearPatternProbOr readout:
  legal_dir_acc        newly-legal cells whose legality prob rises + newly-illegal that fall
  legal_crosstalk      mean |Δ legal-prob| on cells whose true legality did NOT change

Usage:
  python j1b_multi_intervention.py --bank banks/J1_perpattern.pt \
    --readout stream_out/J1_B.pt --state-decoder stream_out/J1_state_decoder.pt \
    --n-games 500 --out experiments/j1b_multi/results.json
"""
import argparse, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, '.')
import train_streaming_probe as tsp
from opening_tree_mlp import LinearPatternProbOr, playedeven_features, C64_TO_C60
from data.othello import OthelloBoardState
# reuse the EXACT condition selectors + board helpers from the OGPT experiment
from mechanistic_interpretability.multi_intervention import (
    select_flip_noninteracting, select_flip_interacting,
    select_add_noninteracting, select_add_interacting, select_mixed,
    compute_legal_moves, compute_counterfactual_legal,
)

CENTER = [27, 28, 35, 36]
EMPTY, WHITE, BLACK = 0, 1, 2
CONDITIONS = [
    ("flip_noninteracting", select_flip_noninteracting),
    ("flip_interacting",    select_flip_interacting),
    ("add_noninteracting",  select_add_noninteracting),
    ("add_interacting",     select_add_interacting),
]


def val_to_class(v):
    return EMPTY if v == 0 else (WHITE if v < 0 else BLACK)


def load_j1b(bank, readout, flanking_patterns, state_decoder, device):
    W_tree, b_tree, meta = tsp.load_trees(bank)
    mlp = tsp.OpeningTreeMLP(W_tree, b_tree, meta, device)
    leaf_build = tsp.load_leaf_build(bank)
    patterns = tsp.load_patterns(flanking_patterns)
    ck = torch.load(readout, map_location=device, weights_only=False)
    st = ck['probe_state'] if 'probe_state' in ck else ck['probe_states'][0]
    hidden = st['linear.weight'].shape[1]
    probe = LinearPatternProbOr(hidden, patterns).to(device); probe.load_state_dict(st); probe.eval()
    sd = torch.load(state_decoder, map_location=device, weights_only=False)
    W = sd['state_probe']                                   # (2, H, 64, 3)
    W = torch.as_tensor(W, dtype=torch.float32, device=device)
    assert W.shape[1] == hidden, f"state-decoder H {W.shape[1]} != readout H {hidden}"
    print(f"J1B: hidden={hidden}, state-decoder acc={sd.get('final_acc')}")
    return mlp, leaf_build, patterns, probe, W, hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='banks/J1_perpattern.pt')
    ap.add_argument('--readout', default='stream_out/J1_B.pt')
    ap.add_argument('--state-decoder', default='stream_out/J1_state_decoder.pt')
    ap.add_argument('--flanking-patterns', default='hand_crafted_flanking_patterns.pt')
    ap.add_argument('--n-games', type=int, default=500)
    ap.add_argument('--num-pickle-files', type=int, default=1)
    ap.add_argument('--n-values', type=int, nargs='+', default=[1, 2, 3, 5, 8])
    ap.add_argument('--pos-range', type=int, nargs=2, default=[5, 54])
    ap.add_argument('--scale-cap', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='experiments/j1b_multi/results.json')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    mlp, leaf_build, patterns, probe, W, hidden = load_j1b(
        args.bank, args.readout, args.flanking_patterns, args.state_decoder, device)

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
    print(f"{len(games)} games")

    def hidden_of(prefix):
        X = playedeven_features(prefix, canonicalize_mover=True).astype(np.float32)
        H = tsp.build_hidden_layer_batch(X[None, :], mlp, patterns, None, False, device,
                                         no_flanking=True, leaf_build=leaf_build,
                                         leaf_index=None).float()
        return H[0]                                          # (hidden,)

    def state_logits(H, mode):
        # H:(hidden,)  W[mode]:(hidden,64,3) -> (64,3)
        return torch.einsum('d,dco->co', H, W[mode])

    def legal_probs(H):
        with torch.no_grad():
            p = probe(H[None, :]).cpu().numpy()[0]
        p[CENTER] = -1.0
        return p

    def calibrated_dirs(H, mods, mode):
        """Per-mod: (s * coeff * d_hat) vector to SUBTRACT.  s = min scale that
        flips the state-decoder argmax at the cell (binary search, +10%, cap)."""
        vecs = []
        for (r, c, orig, tgt) in mods:
            b = r * 8 + c
            tc, cc = val_to_class(tgt), val_to_class(orig)
            d = W[mode, :, b, tc] - W[mode, :, b, cc]
            nrm = d.norm()
            if nrm < 1e-8:
                vecs.append(torch.zeros_like(H)); continue
            d_hat = d / nrm
            coeff = float(H @ d_hat)
            def pred(s):
                Hm = H - s * coeff * d_hat
                return int(state_logits(Hm, mode)[b].argmax())
            hi = args.scale_cap
            if pred(0.0) == tc:
                s = 0.5
            elif pred(hi) != tc:
                s = hi
            else:
                lo = 0.0
                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    if pred(mid) == tc: hi = mid
                    else: lo = mid
                s = min(hi * 1.1, args.scale_cap)
            vecs.append(s * coeff * d_hat)
        return vecs

    def metrics(H0, s0, l0, mods, mode, cf_legal, orig_legal):
        vecs = calibrated_dirs(H0, mods, mode)
        Hp = H0.clone()
        for v in vecs:
            Hp = Hp - v
        sp = state_logits(Hp, mode).detach().cpu().numpy()   # (64,3)
        s0n = s0.detach().cpu().numpy()
        lp = legal_probs(Hp)
        mod_cells = [m[0] * 8 + m[1] for m in mods]
        # --- board-state probe metrics (comparable to multi_intervention) ---
        correct = dir_ok = 0
        for (r, c, orig, tgt) in mods:
            b = r * 8 + c; tc, cc = val_to_class(tgt), val_to_class(orig)
            if int(sp[b].argmax()) == tc: correct += 1
            if (sp[b, tc] - sp[b, cc]) > (s0n[b, tc] - s0n[b, cc]): dir_ok += 1
        d_state = np.abs(sp - s0n)
        keep = np.array([k not in mod_cells for k in range(64)])
        crosstalk = float(d_state[keep].mean())
        # --- legal-move metrics via readout ---
        changed = set(orig_legal) ^ set(cf_legal)
        nl = [lp[cc] - l0[cc] for cc in (set(cf_legal) - set(orig_legal)) if l0[cc] > -1]
        ni = [lp[cc] - l0[cc] for cc in (set(orig_legal) - set(cf_legal)) if l0[cc] > -1]
        keepL = np.array([l0[k] > -1 and k not in mod_cells and k not in changed for k in range(64)])
        dL = np.abs(lp - l0)
        legal_dir = ([1 if x > 0 else 0 for x in nl] + [1 if x < 0 else 0 for x in ni])
        return dict(
            probe_acc=correct / len(mods),
            direction_acc_probe=dir_ok / len(mods),
            crosstalk=crosstalk,
            legal_crosstalk=float(dL[keepL].mean()) if keepL.any() else 0.0,
            legal_dir_acc=float(np.mean(legal_dir)) if legal_dir else float('nan'),
            n_newly_legal=len(nl), n_newly_illegal=len(ni),
        )

    results = {c: {str(n): [] for n in args.n_values} for c, _ in CONDITIONS}
    results['mixed'] = {str(n): [] for n in args.n_values}

    for gi, g in enumerate(games):
        board = OthelloBoardState(); prefix = []
        # advance to a random sampled position in range
        target_pos = rng.randint(args.pos_range[0], min(args.pos_range[1], len(g) - 1))
        ok = True
        for t in range(target_pos):
            valid = board.get_valid_moves()
            if not valid:
                board.update([]); valid = board.get_valid_moves()
                if not valid: ok = False; break
            if g[t] not in valid: ok = False; break
            board.update([g[t]]); prefix.append(g[t])
        if not ok or len(prefix) != target_pos:
            continue
        valid = board.get_valid_moves()
        if not valid:
            continue
        pos = len(prefix); mode = pos % 2
        board2d = board.state.copy(); color = int(board.next_hand_color)
        bstate = board2d.flatten().astype(int)
        orig_legal = sorted(int(m) for m in valid)
        H0 = hidden_of(prefix)
        s0 = state_logits(H0, mode)
        l0 = legal_probs(H0)
        for n in args.n_values:
            for cond, sel in CONDITIONS:
                mods = sel(board2d, n, rng)
                if not mods:
                    continue
                cf3 = [(r, c, nw) for (r, c, ov, nw) in mods]        # (r,c,new) for legality
                cf = compute_counterfactual_legal(board2d, cf3, color)
                results[cond][str(n)].append(
                    metrics(H0, s0, l0, mods, mode, sorted(cf), orig_legal))
            mm = select_mixed(board2d, n, rng)
            mods = mm[0] if isinstance(mm, tuple) else mm
            if mods:
                cf3 = [(r, c, nw) for (r, c, ov, nw) in mods]
                cf = compute_counterfactual_legal(board2d, cf3, color)
                m = metrics(H0, s0, l0, mods, mode, sorted(cf), orig_legal)
                if isinstance(mm, tuple): m['n_flips'], m['n_adds'] = mm[1]
                results['mixed'][str(n)].append(m)
        if (gi + 1) % 100 == 0:
            print(f"  {gi+1}/{len(games)} games", flush=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh)
    # summary
    print(f"\nsaved -> {args.out}")
    print(f"{'condition':22s} " + "  ".join(f"N={n}" for n in args.n_values))
    for cond in [c for c, _ in CONDITIONS] + ['mixed']:
        row = []
        for n in args.n_values:
            rs = results[cond][str(n)]
            row.append(f"{np.mean([r['probe_acc'] for r in rs]):.2f}" if rs else " -- ")
        print(f"{cond+' (probe_acc)':22s} " + "  ".join(f"{v:>4}" for v in row))


if __name__ == '__main__':
    main()
