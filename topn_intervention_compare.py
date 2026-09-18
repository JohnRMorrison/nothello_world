"""Top-N accuracy under causal interventions: Othello-GPT vs Othello-MLP (J1B),
on the SAME intervention set, with calibrated-minimal alpha for BOTH models.

Metric (Li/Nanda top-N, as already implemented for OGPT in
ogpt_legal_mass_shift.li_topn_accuracy):
    N = |legal set|; take the model's top-N cells; errors = false positives in
    top-N + false negatives missed; accuracy = 1 - (FP + FN) / (2N).
Top-N is scored against the COUNTERFACTUAL legal set (legal_cf), before and
after the intervention, so `topn_shift = after - before` is "how much closer
the model's ranking got to the counterfactual board".

Why both models are comparable here
-----------------------------------
Both interventions are probe-directed pushes of the SAME form, in each model's
own representation space:

    h' = h - alpha * (h . d_hat) * d_hat ,  d_hat = normalise(W[:, tgt] - W[:, cur])

  * Othello-GPT : h = residual stream at INTERVENE_LAYER, W = Nanda probe.
  * J1B         : h = the 47,061-d hidden layer,          W = J1 state decoder.

`alpha` is CALIBRATED-MINIMAL for both: the smallest alpha (binary search) that
flips that model's own board decoder on the target square.  This is the cd0
protocol the OGPT figures already use; J1B's shipped script used a FIXED
alpha=2.0, which is not comparable across spaces, so it is calibrated here.

Shared intervention set
-----------------------
One manifest drives both models.  Games come from board_seqs_string_small.npy
(cells 0-63, the move prefix J1B needs) with board_seqs_int_small.npy giving
the matching tokens OGPT needs (verified: int == STOI[string]).  For each
sampled position every non-centre square is tried in every category that
applies to it:

    empty square    -> add_mine (mover's colour), add_yours (opponent's)
    occupied square -> remove, flip

Entries whose counterfactual does not change legality are dropped, and an entry
either model cannot flip within the alpha cap is dropped from BOTH, so the two
models are always scored on an identical set.

Usage
-----
    python topn_intervention_compare.py --n-positions 5000 \
        --models j1b,ogpt --out experiments/topn_compare

PROTOCOL STATUS - READ BEFORE USING (2026-09-18)
------------------------------------------------
The OGPT side inherits ogpt_legal_mass_shift.py's parameterisation, which ties
the edit layer to the single layer-6 probe:  intervene_layer = 6 - cal_depth.
That means:

  --cal-depth 0  -> edit at resid_post block 6, calibrated with the layer-6
                    probe (native, but NOT the layer Nanda edits)
  --cal-depth 2  -> edit at resid_post block 4, but calibrated at layer 6
                    (right layer, NON-native: alpha must be large enough to
                    survive two blocks, hence the K~4 multiplier needed)

NEITHER is Nanda's protocol, which edits at layer 4 and calibrates against the
LAYER-4 (native) probe -- cf. multi_intervention.py, whose --layer-probe default
is "layer-intervene - 1, i.e. native" and --cal-depth 0 = "local (native probe)".
Doing that needs a layer-4 probe, which is not in the repo (only the layer-6
main_linear_probe.pth); train one with
    train_nanda_probe_extended.py --layer 4
which emits the same (3, 512, 8, 8, 3) format this script consumes.

Consequently, any comparison here between cal_depth settings confounds the edit
layer with the calibration target, and the cross-model (OGPT vs MLP) rows
inherit that confound.  Treat the committed numbers as provisional.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ogpt_legal_mass_shift as oms          # OGPT: model/probe/calibration/top-N
import j1b_category_table as jct             # J1B: loader + class helpers
from data.othello import OthelloBoardState

sys.path.insert(0, "experiments/mathematical_transformation_experiments")
from probe_state_pred_for_othello import STOI  # noqa: E402

CENTER = {27, 28, 35, 36}
MOVABLE = [c for c in range(64) if c not in CENTER]
CATEGORIES = ['remove', 'add_mine', 'add_yours', 'flip']
SUBS = ['newly_legal', 'newly_illegal', 'both']
# vocab token for each of the 60 scored cells, in oms.STOI_INDICES order
TOKEN_OF_CELL60 = [STOI[c] for c in oms.STOI_INDICES]


# --------------------------------------------------------------------------
# Stage A - the shared intervention manifest
# --------------------------------------------------------------------------
def build_manifest(seqs_string, n_positions, pos_lo, pos_hi, seed, categories=None):
    """One sampled position per game; every applicable (square, category)."""
    rng = random.Random(seed)
    entries = []
    positions = []
    gi = -1
    n_games = len(seqs_string)
    while len(positions) < n_positions and gi < n_games - 1:
        gi += 1
        pos = rng.randint(pos_lo, pos_hi - 1)
        game_string = seqs_string[gi]
        try:
            board_state, color = oms.replay(game_string, pos)
        except Exception:
            continue
        legal_orig = sorted(oms.legal_moves_for(board_state, color))
        if not legal_orig:
            continue

        here = []
        for cell in MOVABLE:
            r, c = cell // 8, cell % 8
            v = int(board_state[r, c])
            if v == 0:
                cands = [('add_mine', int(color)), ('add_yours', int(-color))]
            else:
                cands = [('remove', 0), ('flip', -v)]
            for cat, tgt_val in cands:
                if categories and cat not in categories:
                    continue
                cf = OthelloBoardState()
                cf.state = board_state.copy()
                cf.next_hand_color = color
                cf.state[r, c] = tgt_val
                legal_cf = sorted(int(m) for m in cf.get_valid_moves())
                sub = jct.classify_change(legal_orig, legal_cf)
                if sub is None:
                    continue                      # legality unchanged -> skip
                here.append(dict(cell=cell, cat=cat, sub=sub,
                                 orig_val=v, tgt_val=tgt_val,
                                 legal_cf=legal_cf))
        if not here:
            continue
        pid = len(positions)
        positions.append(dict(pid=pid, gi=gi, pos=pos, color=int(color),
                              legal_orig=legal_orig))
        for e in here:
            e['pid'] = pid
            entries.append(e)
    return positions, entries


# --------------------------------------------------------------------------
# calibration (identical semantics for both models)
# --------------------------------------------------------------------------
def calibrate_min_alpha(argmax_at, tgt_cls, cap, tol=1e-3, iters=40):
    """Smallest alpha in [0, cap] whose push flips the decoder argmax to tgt.
    Returns (alpha, flipped)."""
    if argmax_at(0.0) == tgt_cls:
        return 0.5, True                       # already there; small nudge (matches cdN_calibrate)
    if argmax_at(cap) != tgt_cls:
        return cap, False                      # cannot flip within budget
    lo, hi = 0.0, cap
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if argmax_at(mid) == tgt_cls:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi, True


def topn_from_scores60(scores60, legal_set):
    """oms.li_topn_accuracy on a plain score vector (rank-based -> logits or probs)."""
    return oms.li_topn_accuracy(torch.as_tensor(scores60, dtype=torch.float32), legal_set)


def topn_errors60(scores60, legal_set):
    """Nanda's top-N ERROR COUNT: |FP| + |FN| for the model's top-N cells vs
    `legal_set` (N = |legal_set|).  Related to the accuracy by
    acc = 1 - errors / (2N); Nanda reports the MEAN of this, before -> after."""
    legal_stoi = set(legal_set) & set(oms.STOI_INDICES)
    n = len(legal_stoi)
    if n == 0:
        return None
    t = torch.as_tensor(scores60, dtype=torch.float32)
    topn_idx = t.argsort(descending=True)[:n]
    topn_cells = {oms.STOI_INDICES[int(i)] for i in topn_idx}
    fp = topn_cells - legal_stoi
    fn = legal_stoi - topn_cells
    return len(fp) + len(fn)


# --------------------------------------------------------------------------
# Stage B - J1B
# --------------------------------------------------------------------------
def run_j1b(args, positions, entries, seqs_string, device):
    import train_streaming_probe as tsp
    from opening_tree_mlp import playedeven_features

    mlp, leaf_build, patterns, probe, W, hidden = jct.load_j1b(
        args.bank, args.readout, args.flanking_patterns, args.state_decoder, device)

    def hidden_of(prefix):
        X = playedeven_features(prefix, canonicalize_mover=True).astype(np.float32)
        H = tsp.build_hidden_layer_batch(X[None, :], mlp, patterns, None, False, device,
                                         no_flanking=True, leaf_build=leaf_build,
                                         leaf_index=None).float()
        return H[0]

    def scores64(Hmat):
        with torch.no_grad():
            p = probe(Hmat).cpu().numpy()
        p[:, list(CENTER)] = -1.0
        return p

    by_pid = defaultdict(list)
    for e in entries:
        by_pid[e['pid']].append(e)

    out = {}
    t0 = time.time()
    for k, P in enumerate(positions):
        prefix = [int(x) for x in seqs_string[P['gi']][: P['pos'] + 1]]
        H0 = hidden_of(prefix)
        mode = len(prefix) % 2                       # parity mode of the state decoder
        base = scores64(H0[None, :])[0]

        specs, keep = [], []
        for e in by_pid[P['pid']]:
            cell = e['cell']
            Wc = W[mode, :, cell, :]                 # (H,3)
            cur_cls = jct.val_to_class(e['orig_val'])
            tgt_cls = jct.val_to_class(e['tgt_val'])
            d = Wc[:, tgt_cls] - Wc[:, cur_cls]
            nrm = d.norm()
            if nrm < 1e-8:
                continue
            d_hat = d / nrm
            coeff = float(H0 @ d_hat)

            def argmax_at(a, _H0=H0, _c=coeff, _d=d_hat, _Wc=Wc):
                return int(torch.argmax((_H0 - a * _c * _d) @ _Wc))

            alpha, flipped = calibrate_min_alpha(argmax_at, tgt_cls, args.alpha_cap)
            specs.append(H0 - (args.K * alpha) * coeff * d_hat)
            keep.append((e, alpha, flipped))
        if not specs:
            continue
        S = scores64(torch.stack(specs, dim=0))
        for i, (e, alpha, flipped) in enumerate(keep):
            b60 = [base[c] for c in oms.STOI_INDICES]
            a60 = [S[i, c] for c in oms.STOI_INDICES]
            out[(e['pid'], e['cell'], e['cat'])] = dict(
                topn_before=topn_from_scores60(b60, e['legal_cf']),
                topn_after=topn_from_scores60(a60, e['legal_cf']),
                err_before=topn_errors60(b60, e['legal_cf']),
                err_after=topn_errors60(a60, e['legal_cf']),
                alpha=float(alpha), flipped=bool(flipped))
        if (k + 1) % 250 == 0:
            print(f"  [j1b] position {k+1}/{len(positions)}  "
                  f"scored={len(out)}  ({int(time.time()-t0)}s)", flush=True)
    return out



# --------------------------------------------------------------------------
# Stage B2 - trained MLPs (H512 played+even, H4096 move_grid)
# --------------------------------------------------------------------------
_CKPT_DIR = ('experiments/mathematical_transformation_experiments/'
             'heuristic_probe_results/pattern_detector_checkpoints')
MLP_SPECS = {
    'h512_playedeven': dict(
        mlp=f'{_CKPT_DIR}/pattern_simple_direct_H512_playedeven.pt',
        probe=f'{_CKPT_DIR}/probe_direct_H512_playedeven.pt', input='playedeven'),
    'h4096_movegrid': dict(
        mlp=f'{_CKPT_DIR}/pattern_simple_direct_H4096_move_grid.pt',
        probe=f'{_CKPT_DIR}/probe_direct_H4096_move_grid.pt', input='move_grid'),
}


def run_mlp(args, positions, entries, seqs_string, device, which):
    """Probe-directed push of the MLP's hidden layer, calibrated-minimal alpha.

    Directly analogous to the OGPT / J1B runs: d_hat comes from the model's OWN
    board-state probe for the target cell, the push is the same projection form,
    and alpha is the smallest value that flips that probe's argmax.
    """
    import torch.nn as nn
    import torch.nn.functional as F
    from train_pattern_simple import DirectMLP, _get_cell_pat_index, to_move_grid_input
    from hand_crafted_flanking import enumerate_flanking_patterns, MOVE_TO_IDX
    from compare_v4_vs_mlp import C64_TO_C60

    cfg = MLP_SPECS[which]
    mck = torch.load(cfg['mlp'], map_location=device, weights_only=False)
    pck = torch.load(cfg['probe'], map_location=device, weights_only=False)
    H = pck['hidden_dim']
    in_dim = 3600 if cfg['input'] == 'move_grid' else 120
    models, probes = {}, {}
    for par in ('even', 'odd'):
        m = DirectMLP(in_dim, H).to(device); m.load_state_dict(mck[par]); m.eval()
        pr = nn.Linear(H, 64 * 3).to(device); pr.load_state_dict(pck[par]); pr.eval()
        models[par], probes[par] = m, pr
    print(f"{which}: H={H}, input={cfg['input']}, probe acc={pck.get('best_acc'):.4f}")

    patterns = enumerate_flanking_patterns()
    p2c = torch.tensor([MOVE_TO_IDX[q['target']] for q in patterns],
                       dtype=torch.long, device=device)
    idx, mask = _get_cell_pat_index(p2c, 60)

    def make_x(prefix):
        f = torch.zeros(1, 180, device=device)
        for i, c in enumerate(prefix):
            if c not in C64_TO_C60:
                continue
            c60 = C64_TO_C60[c]
            f[0, c60] = 1.0
            f[0, 60 + c60] = (i + 1) / 60.0
            if i % 2 == 0:
                f[0, 120 + c60] = 1.0
        if cfg['input'] == 'move_grid':
            return to_move_grid_input(f)
        return torch.cat([f[:, :60], f[:, 120:180]], dim=1)

    by_pid = defaultdict(list)
    for e in entries:
        by_pid[e['pid']].append(e)

    out = {}
    t0 = time.time()
    for k, P in enumerate(positions):
        prefix = [int(x) for x in seqs_string[P['gi']][: P['pos'] + 1]]
        par = 'even' if (len(prefix) % 2 == 1) else 'odd'
        mlp, pr = models[par], probes[par]
        lin1, relu, lin2 = mlp.net[0], mlp.net[1], mlp.net[2]
        Wp = pr.weight.view(64, 3, H)                     # per-cell class directions
        with torch.no_grad():
            x = make_x(prefix)
            h0 = relu(lin1(x))[0]                          # (H,)

            def scores60(Hm):                              # (B,H) -> (B,60) prob-OR
                lg = lin2(Hm)
                l1m = -F.softplus(lg)
                g = l1m[:, idx].masked_fill(~mask[None], 0.0)
                return (-g.sum(dim=-1))

            b60 = scores60(h0[None, :])[0].cpu().numpy()
            specs, keep = [], []
            for e in by_pid[P['pid']]:
                cell = e['cell']
                cur = jct.val_to_class(e['orig_val']); tgt = jct.val_to_class(e['tgt_val'])
                d = Wp[cell, tgt] - Wp[cell, cur]
                if d.norm() < 1e-8:
                    continue
                d_hat = d / d.norm()
                coeff = float(h0 @ d_hat)

                def argmax_at(a, _h=h0, _c=coeff, _d=d_hat, _cell=cell):
                    hm = _h - a * _c * _d
                    return int(torch.argmax(pr(hm[None, :]).view(64, 3)[_cell]))

                alpha, flipped = calibrate_min_alpha(argmax_at, tgt, args.alpha_cap)
                specs.append(h0 - (args.K * alpha) * coeff * d_hat)
                keep.append((e, alpha, flipped))
            if not specs:
                continue
            S = scores60(torch.stack(specs, dim=0)).cpu().numpy()
        for i, (e, alpha, flipped) in enumerate(keep):
            out[(e['pid'], e['cell'], e['cat'])] = dict(
                topn_before=topn_from_scores60(b60, e['legal_cf']),
                topn_after=topn_from_scores60(S[i], e['legal_cf']),
                err_before=topn_errors60(b60, e['legal_cf']),
                err_after=topn_errors60(S[i], e['legal_cf']),
                alpha=float(alpha), flipped=bool(flipped))
        if (k + 1) % 250 == 0:
            print(f"  [{which}] position {k+1}/{len(positions)}  scored={len(out)}  "
                  f"({int(time.time()-t0)}s)", flush=True)
    return out


# --------------------------------------------------------------------------
# Stage C - Othello-GPT
# --------------------------------------------------------------------------
def run_ogpt(args, positions, entries, seqs_int, device):
    model = oms.load_model(args.ckpt, device)
    probe = oms.load_probe(args.probe_path, device)
    INTERVENE_LAYER = oms.intervene_layer_for(args.cal_depth, oms.PROBE_LAYER)
    probe_mode = oms.PROBE_KIND_TO_MODE[args.probe_kind]

    by_pid = defaultdict(list)
    for e in entries:
        by_pid[e['pid']].append(e)

    out = {}
    t0 = time.time()
    for k, P in enumerate(positions):
        tokens = torch.from_numpy(
            np.asarray(seqs_int[P['gi']][: P['pos'] + 1], dtype=np.int64)
        ).unsqueeze(0).to(device)
        pos = P['pos']
        with torch.no_grad():
            prefix_acts = oms.compute_prefix(model, tokens, INTERVENE_LAYER + 1)
            clean_logits, _ = oms.forward_from_prefix_capturing(
                model, prefix_acts.clone(), INTERVENE_LAYER + 1, oms.PROBE_LAYER)
            clean_last = clean_logits[0, -1]
            b60 = clean_last[TOKEN_OF_CELL60].detach().cpu().numpy()

            for e in by_pid[P['pid']]:
                cell = e['cell']; r, c = cell // 8, cell % 8
                probe_cell_W = probe[probe_mode, :, r, c, :].detach()
                cur_cls = oms.board_val_to_probe_class(e['orig_val'], P['color'], args.probe_kind)
                tgt_cls = oms.board_val_to_probe_class(e['tgt_val'], P['color'], args.probe_kind)
                d = probe_cell_W[:, tgt_cls] - probe_cell_W[:, cur_cls]
                nrm = d.norm()
                if nrm < 1e-8:
                    continue
                d_hat = d / nrm
                coeff = (prefix_acts[0, pos] @ d_hat).item()

                def argmax_at(a, _pa=prefix_acts, _p=pos, _c=coeff, _d=d_hat, _W=probe_cell_W):
                    h = _pa[0, _p] - a * _c * _d
                    return int(torch.argmax(_W.T @ h))

                alpha, flipped = calibrate_min_alpha(argmax_at, tgt_cls, args.alpha_cap)
                intv_logits, _ = oms.run_with_intervention(
                    model, prefix_acts, pos, d_hat, coeff, args.K * alpha,
                    INTERVENE_LAYER, capture_layer=oms.PROBE_LAYER)
                a60 = intv_logits[0, -1][TOKEN_OF_CELL60].detach().cpu().numpy()
                out[(e['pid'], e['cell'], e['cat'])] = dict(
                    topn_before=topn_from_scores60(b60, e['legal_cf']),
                    topn_after=topn_from_scores60(a60, e['legal_cf']),
                    err_before=topn_errors60(b60, e['legal_cf']),
                    err_after=topn_errors60(a60, e['legal_cf']),
                    alpha=float(alpha), flipped=bool(flipped))
        if (k + 1) % 100 == 0:
            print(f"  [ogpt] position {k+1}/{len(positions)}  "
                  f"scored={len(out)}  ({int(time.time()-t0)}s)", flush=True)
    return out


# --------------------------------------------------------------------------
def aggregate(entries, res, keys):
    """Mean top-N before/after/shift per (category, sub), over `keys` only."""
    acc = {c: {s: dict(n=0, b=0.0, a=0.0, eb=0.0, ea=0.0) for s in SUBS} for c in CATEGORIES}
    for e in entries:
        k = (e['pid'], e['cell'], e['cat'])
        if k not in keys:
            continue
        r = res[k]
        if r['topn_before'] is None or r['topn_after'] is None:
            continue
        if r.get('err_before') is None or r.get('err_after') is None:
            continue
        d = acc[e['cat']][e['sub']]
        d['n'] += 1; d['b'] += r['topn_before']; d['a'] += r['topn_after']
        d['eb'] += r['err_before']; d['ea'] += r['err_after']
    table = {}
    tot = dict(n=0, b=0.0, a=0.0, eb=0.0, ea=0.0)
    for c in CATEGORIES:
        for s in SUBS:
            d = acc[c][s]
            for k in tot:
                tot[k] += d[k]
    if tot['n']:
        table['ALL|pooled'] = dict(n=tot['n'],
                                   topn_before=tot['b'] / tot['n'],
                                   topn_after=tot['a'] / tot['n'],
                                   topn_shift=(tot['a'] - tot['b']) / tot['n'],
                                   err_before=tot['eb'] / tot['n'],
                                   err_after=tot['ea'] / tot['n'])
    for c in CATEGORIES:
        for s in SUBS:
            d = acc[c][s]
            if d['n']:
                table[f"{c}|{s}"] = dict(n=d['n'],
                                         topn_before=d['b'] / d['n'],
                                         topn_after=d['a'] / d['n'],
                                         topn_shift=(d['a'] - d['b']) / d['n'],
                                         err_before=d['eb'] / d['n'],
                                         err_after=d['ea'] / d['n'])
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-positions', type=int, default=5000)
    ap.add_argument('--pos-lo', type=int, default=10)
    ap.add_argument('--pos-hi', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--models', default='j1b,ogpt')
    ap.add_argument('--categories', default='flip',
                    help="comma list of remove,add_mine,add_yours,flip. Default 'flip' = "
                         "the Li et al. protocol (black<->white colour flips only).")
    ap.add_argument('--K', type=float, default=1.0,
                    help='multiple of the calibrated-minimal alpha (1 = minimal)')
    ap.add_argument('--alpha-cap', type=float, default=10.0)
    # J1B
    ap.add_argument('--bank', default='banks/J1_perpattern.pt')
    ap.add_argument('--readout', default='stream_out/J1_B.pt')
    ap.add_argument('--state-decoder', default='stream_out/J1_state_decoder.pt')
    ap.add_argument('--flanking-patterns', default='hand_crafted_flanking_patterns.pt')
    # OGPT
    ap.add_argument('--ckpt', default='ckpts/gpt_nanda_synthetic.ckpt')
    ap.add_argument('--probe-path', default='mechanistic_interpretability/main_linear_probe.pth')
    ap.add_argument('--probe-kind', default='mode0_mineyours')
    ap.add_argument('--cal-depth', type=int, default=0)
    ap.add_argument('--int-path', default='mechanistic_interpretability/board_seqs_int_small.npy')
    ap.add_argument('--string-path', default='mechanistic_interpretability/board_seqs_string_small.npy')
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', default='experiments/topn_compare')
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    want = [m.strip() for m in args.models.split(',') if m.strip()]
    print(f"device={device}  models={want}  K={args.K}x calibrated-minimal alpha")

    seqs_int = np.load(args.int_path)
    seqs_string = np.load(args.string_path)

    t0 = time.time()
    cats = {c.strip() for c in args.categories.split(',') if c.strip()}
    print(f"categories: {sorted(cats)}")
    positions, entries = build_manifest(seqs_string, args.n_positions,
                                        args.pos_lo, args.pos_hi, args.seed, cats)
    print(f"manifest: {len(positions)} positions, {len(entries)} interventions "
          f"({int(time.time()-t0)}s)")

    res = {}
    for m in want:
        print(f"--- running {m} ---", flush=True)
        if m == 'j1b':
            res[m] = run_j1b(args, positions, entries, seqs_string, device)
        elif m == 'ogpt':
            res[m] = run_ogpt(args, positions, entries, seqs_int, device)
        elif m in MLP_SPECS:
            res[m] = run_mlp(args, positions, entries, seqs_string, device, m)
        else:
            raise SystemExit(f"unknown model {m}")
        print(f"    {m}: scored {len(res[m])} interventions")

    # keep only interventions BOTH models flipped (identical set for the comparison)
    keysets = []
    for m in want:
        keysets.append({k for k, v in res[m].items() if v['flipped']})
    shared = set.intersection(*keysets) if keysets else set()
    print(f"\nshared (flipped by all requested models): {len(shared)}")

    os.makedirs(args.out, exist_ok=True)
    report = dict(n_positions=len(positions), n_interventions=len(entries),
                  n_shared=len(shared), K=args.K, cal_depth=args.cal_depth,
                  models={})
    for m in want:
        report['models'][m] = aggregate(entries, res[m], shared)
    with open(os.path.join(args.out, 'topn_compare.json'), 'w') as fh:
        json.dump(report, fh, indent=2)

    rows = ['ALL|pooled'] + [f"{c}|{s_}" for c in CATEGORIES for s_ in SUBS]
    print("\n=== mean number of top-N ERRORS vs the counterfactual board "
          f"(shared set, K={args.K}x calibrated-minimal alpha) ===")
    hdr = f"{'category|sub':<22}{'n':>9}" + ''.join(f"{m:>24}" for m in want)
    print(hdr); print('-' * len(hdr))
    for key in rows:
        if not any(key in report['models'][m] for m in want):
            continue
        n0 = next(report['models'][m][key]['n'] for m in want if key in report['models'][m])
        row = f"{key:<22}{n0:>9}"
        for m in want:
            d = report['models'][m].get(key)
            row += (f"{d['err_before']:.3f} -> {d['err_after']:.3f}".rjust(24)) if d else ' ' * 24
        print(row)

    print("\n=== top-N accuracy vs the counterfactual legal set (shared set, "
          f"K={args.K}x calibrated-minimal alpha) ===")
    hdr = f"{'category|sub':<26}" + ''.join(f"{m:>28}" for m in want)
    print(hdr); print('-' * len(hdr))
    for c in CATEGORIES:
        for s in SUBS:
            key = f"{c}|{s}"
            if not any(key in report['models'][m] for m in want):
                continue
            row = f"{key:<26}"
            for m in want:
                d = report['models'][m].get(key)
                row += (f"{d['topn_before']:.4f}->{d['topn_after']:.4f} "
                        f"({d['topn_shift']:+.4f})".rjust(28)) if d else ' ' * 28
            print(row)
    print(f"\nsaved {os.path.join(args.out, 'topn_compare.json')}")


if __name__ == '__main__':
    main()
