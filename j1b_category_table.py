#!/usr/bin/env python
"""J1B (tree-bank Othello-MLP): direction-accuracy table in the src/ taxonomy.

Fills the same chart the OGPT / dense-MLP framework produces:
  categories   = remove, add_mine, add_yours, flip
  sub          = newly_legal, newly_illegal, both
  metric       = fraction of relevant cells whose READOUT legality prob moves
                 the correct way after a probe-directed HIDDEN push:
                   newly-legal cell  -> prob should RISE
                   newly-illegal cell-> prob should FALL
                 ('both' pools newly-legal(rise) + newly-illegal(fall) cells.)

Intervention = FIXED-alpha push of the 47,061-d hidden along the J1 state-
decoder's (current_class -> target_class) axis, matching the framework's
`alpha=2.0, mode=fixed` protocol (NOT calibrated-minimal):
   H' = H + alpha * normalize( W[mode,:,cell,tgt] - W[mode,:,cell,cur] )
W = stream_out/J1_state_decoder.pt  (2 parity-modes, H, 64, 3;
classes 0=empty / 1=white(-1) / 2=black(+1)).  Legality read from H' via
LinearPatternProbOr; classification uses the true Othello engine.

N=1 only (single-cell).  One random position per game (independence).

Usage:
  python j1b_category_table.py --n-games 2000 --alpha 2.0 \
    --out experiments/j1b_category/table_a2.json
"""
import argparse, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, '.')
import train_streaming_probe as tsp
from opening_tree_mlp import LinearPatternProbOr, playedeven_features, C64_TO_C60
from data.othello import OthelloBoardState

CENTER = [27, 28, 35, 36]
EMPTY, WHITE, BLACK = 0, 1, 2
CATEGORIES = ['remove', 'add_mine', 'add_yours', 'flip']
SUBS = ['newly_legal', 'newly_illegal', 'both']


def val_to_class(v):
    return EMPTY if v == 0 else (WHITE if v < 0 else BLACK)


def classify_change(orig_legal, cf_legal):
    nl = set(cf_legal) - set(orig_legal)
    ni = set(orig_legal) - set(cf_legal)
    if nl and not ni: return 'newly_legal'
    if ni and not nl: return 'newly_illegal'
    if nl and ni:     return 'both'
    return None


def load_j1b(bank, readout, flanking_patterns, state_decoder, device):
    W_tree, b_tree, meta = tsp.load_trees(bank)
    mlp = tsp.OpeningTreeMLP(W_tree, b_tree, meta, device)
    leaf_build = tsp.load_leaf_build(bank)
    patterns = tsp.load_patterns(flanking_patterns)
    ck = torch.load(readout, map_location=device)
    st = ck['probe_state'] if 'probe_state' in ck else ck['probe_states'][0]
    hidden = st['linear.weight'].shape[1]
    probe = LinearPatternProbOr(hidden, patterns).to(device); probe.load_state_dict(st); probe.eval()
    sd = torch.load(state_decoder, map_location=device)
    W = torch.as_tensor(sd['state_probe'], dtype=torch.float32, device=device)   # (2,H,64,3)
    assert W.shape[1] == hidden, f"state-decoder H {W.shape[1]} != readout H {hidden}"
    print(f"J1B: hidden={hidden}, state-decoder acc={sd.get('final_acc')}")
    return mlp, leaf_build, patterns, probe, W, hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='banks/J1_perpattern.pt')
    ap.add_argument('--readout', default='stream_out/J1_B.pt')
    ap.add_argument('--state-decoder', default='stream_out/J1_state_decoder.pt')
    ap.add_argument('--flanking-patterns', default='hand_crafted_flanking_patterns.pt')
    ap.add_argument('--n-games', type=int, default=2000)
    ap.add_argument('--num-pickle-files', type=int, default=1)
    ap.add_argument('--pos-range', type=int, nargs=2, default=[10, 50])
    ap.add_argument('--alpha', type=float, default=2.0)
    ap.add_argument('--push', choices=['projection', 'additive'], default='projection',
                    help="projection: h' = h - alpha*(h.d_hat)*d_hat (matches the "
                         "src/ framework, mode=fixed); additive: h + alpha*d_hat.")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='experiments/j1b_category/table.json')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}  alpha={args.alpha}")
    mlp, leaf_build, patterns, probe, W, hidden = load_j1b(
        args.bank, args.readout, args.flanking_patterns, args.state_decoder, device)

    if hasattr(tsp, 'load_games'):
        games = tsp.load_games(num_files=args.num_pickle_files)[:args.n_games]
    else:
        import pickle, glob
        files = sorted(glob.glob('./data/othello_synthetic/*.pickle'))[-args.num_pickle_files:]
        games = []
        for fp in files:
            with open(fp, 'rb') as fh: games.extend(pickle.load(fh))
        games = games[:args.n_games]
    print(f"{len(games)} games")

    def hidden_of(prefix):
        X = playedeven_features(prefix, canonicalize_mover=True).astype(np.float32)
        H = tsp.build_hidden_layer_batch(X[None, :], mlp, patterns, None, False, device,
                                         no_flanking=True, leaf_build=leaf_build,
                                         leaf_index=None).float()
        return H[0]

    def readout_rows(Hmat):
        with torch.no_grad():
            p = probe(Hmat).cpu().numpy()
        p[:, CENTER] = -1.0
        return p

    # accumulators: hits/total per (category, sub)
    hits = {c: {s: 0 for s in SUBS} for c in CATEGORIES}
    tot  = {c: {s: 0 for s in SUBS} for c in CATEGORIES}
    n_int = {c: {s: 0 for s in SUBS} for c in CATEGORIES}   # # interventions
    shift_n = {c: {s: 0.0 for s in SUBS} for c in CATEGORIES}  # oriented shift, normalized dist
    shift_r = {c: {s: 0.0 for s in SUBS} for c in CATEGORIES}  # oriented shift, raw readout prob
    sv = {c: {s: [] for s in SUBS} for c in CATEGORIES}        # per-cell oriented shift (normalized)
    MOV60 = [x for x in range(64) if x not in CENTER]
    def _nd(p):                                             # -> distribution over 60 movable cells
        v = np.array([max(float(p[x]), 0.0) for x in MOV60]); s = v.sum()
        q = np.zeros(64)
        if s > 0:
            for i, x in enumerate(MOV60): q[x] = v[i] / s
        return q

    for gi, g in enumerate(games):
        board = OthelloBoardState(); prefix = []
        hi = min(args.pos_range[1], len(g) - 1)
        if hi < args.pos_range[0]:
            continue                                        # game too short to sample in range
        tp = rng.randint(args.pos_range[0], hi)
        ok = True
        for t in range(tp):
            valid = board.get_valid_moves()
            if not valid:
                board.update([]); valid = board.get_valid_moves()
                if not valid: ok = False; break
            if g[t] not in valid: ok = False; break
            board.update([g[t]]); prefix.append(g[t])
        if not ok or len(prefix) != tp:
            continue
        valid = board.get_valid_moves()
        if not valid:
            continue
        pos = len(prefix); mode = pos % 2
        mover = int(board.next_hand_color)                 # +1 black / -1 white
        state = board.state.copy(); flat = state.flatten().astype(int)
        orig_legal = sorted(int(m) for m in valid)
        H0 = hidden_of(prefix)
        l0 = readout_rows(H0[None, :])[0]

        specs = []                                          # (cat, sub, cell, d_hat, cf_legal)
        for c in range(64):
            if c in CENTER or c not in C64_TO_C60:
                continue
            r, col = divmod(c, 8); v = int(flat[c])
            if v != 0:
                cands = [('remove', 0), ('flip', -v)]
            else:
                cands = [('add_mine', mover), ('add_yours', -mover)]
            for cat, tgt_val in cands:
                cur_cls, tgt_cls = val_to_class(v), val_to_class(tgt_val)
                cf = OthelloBoardState(); cf.state = state.copy(); cf.next_hand_color = board.next_hand_color
                cf.state[r, col] = tgt_val
                cf_legal = sorted(int(m) for m in cf.get_valid_moves())
                sub = classify_change(orig_legal, cf_legal)
                if sub is None:
                    continue
                d = W[mode, :, c, tgt_cls] - W[mode, :, c, cur_cls]
                nrm = d.norm()
                if nrm < 1e-8:
                    continue
                specs.append((cat, sub, c, d / nrm, cf_legal))
        if not specs:
            continue
        if args.push == 'projection':
            Hmat = torch.stack(
                [H0 - args.alpha * float(H0 @ s[3]) * s[3] for s in specs], dim=0)
        else:
            Hmat = torch.stack([H0 + args.alpha * s[3] for s in specs], dim=0)
        lp = readout_rows(Hmat)                             # (K,64)
        q0 = _nd(l0)                                        # orig distribution (this position)
        for k, (cat, sub, c, _, cf_legal) in enumerate(specs):
            nl = [x for x in (set(cf_legal) - set(orig_legal)) if l0[x] > -1]
            ni = [x for x in (set(orig_legal) - set(cf_legal)) if l0[x] > -1]
            hh = sum(1 for x in nl if lp[k, x] - l0[x] > 0) + \
                 sum(1 for x in ni if lp[k, x] - l0[x] < 0)
            nn = len(nl) + len(ni)
            hits[cat][sub] += hh; tot[cat][sub] += nn; n_int[cat][sub] += 1
            # oriented shift: +ve = prob moved the intended way (up for NL, down for NI)
            q1 = _nd(lp[k])
            shift_r[cat][sub] += (sum(lp[k, x] - l0[x] for x in nl)
                                  + sum(l0[x] - lp[k, x] for x in ni))
            shift_n[cat][sub] += (sum(q1[x] - q0[x] for x in nl)
                                  + sum(q0[x] - q1[x] for x in ni))
            for x in nl: sv[cat][sub].append(float(q1[x] - q0[x]))
            for x in ni: sv[cat][sub].append(float(q0[x] - q1[x]))
        if (gi + 1) % 250 == 0:
            print(f"  {gi+1}/{len(games)} games", flush=True)

    out = {c: {s: dict(dir_acc=(hits[c][s] / tot[c][s]) if tot[c][s] else float('nan'),
                       mean_shift_norm=float(shift_n[c][s] / tot[c][s]) if tot[c][s] else float('nan'),
                       mean_shift_raw=float(shift_r[c][s] / tot[c][s]) if tot[c][s] else float('nan'),
                       n_cells=tot[c][s], n_interventions=n_int[c][s])
               for s in SUBS} for c in CATEGORIES}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(dict(alpha=args.alpha, table=out), fh, indent=2)
    np.savez(args.out.replace('.json', '_shifts.npz'),
             **{f"{c}__{s}": np.array(sv[c][s], dtype=np.float32)
                for c in CATEGORIES for s in SUBS})

    print(f"\nsaved -> {args.out}")
    print(f"\nJ1B tree-bank direction accuracy, N=1, fixed alpha={args.alpha}")
    print(f"{'category':10s} {'sub':14s} {'N=1':>7s} {'n_cells':>9s} {'n_int':>7s}")
    for c in CATEGORIES:
        for s in SUBS:
            e = out[c][s]
            acc = f"{e['dir_acc']:.3f}" if e['dir_acc'] == e['dir_acc'] else "  -- "
            print(f"{c:10s} {s:14s} {acc:>7s} {e['n_cells']:>9d} {e['n_interventions']:>7d}")


if __name__ == '__main__':
    main()
