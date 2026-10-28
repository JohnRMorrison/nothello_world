"""Generate all Section 5.4 statistics reported in the ICLR paper.

Produces:
  - Mean raw top-N errors before / after intervention (paper: 1.37 → 0.31)
  - Fraction of positions where dP_legal > 0 (paper: 98.5%)
  - Mean abs_dP_newly_legal across all conditions (paper: 0.25%)
  - Mean P_before_newly_illegal for ni/both conditions (paper: 10.83%)
  - Mean abs_dP_newly_illegal for ni/both conditions (paper: 2.15%)
  - Table 4: per-kind mean abs_dP_newly_legal and abs_dP_newly_illegal
  - Table 5: per-square percentiles of newly_legal/newly_illegal shifts
  - MLP equivalents (95.9%, 1.11%, 3.76%, 4.03 → 3.64)

Usage:
    python generate_paper_numbers.py --n-positions 5000
    python generate_paper_numbers.py --n-positions 5000 --alpha-mult 2.0  # optional scale
"""

import argparse
import collections
import glob
import os
import pickle
import random
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from src import config, intervention, probes, metrics
from mechanistic_interpretability.multi_intervention import compute_counterfactual_legal
from data.othello import OthelloBoardState

INTV_LAYER  = 4
PROBE_LAYER = INTV_LAYER   # cal_depth=0: probe at the intervention layer (paper's setup)
PROBE_MODE  = 0
EMPTY_CLS, YOURS_CLS, MINE_CLS = 0, 1, 2   # probe mode 0 classes
ALL60 = set(config.STOI_INDICES)           # the 60 cells the model predicts
CENTER     = {27, 28, 35, 36}
VALID64    = [c for c in range(64) if c not in CENTER]
EMPTY, YOURS, MINE = 0, 1, 2


def bv2cls(val, color):
    if val == 0:
        return EMPTY
    return MINE if val == color else YOURS


def classify_change(legal_orig, legal_cf):
    added   = legal_cf - legal_orig
    removed = legal_orig - legal_cf
    if not added and not removed:
        return None
    if added and not removed:
        return 'newly_legal'
    if removed and not added:
        return 'newly_illegal'
    return 'both'


def li_topn_fp(p60, legal_cf_flat):
    """Returns FP count = |top-N predictions that are illegal|.
    Paper defines 'errors' this way. Since |top_N|=|target|=N, FP = FN = (FP+FN)/2."""
    target = set()
    for fc in legal_cf_flat:
        idx = config.STOI_INDEX_OF.get(fc)
        if idx is not None:
            target.add(idx)
    N = len(target)
    if N == 0:
        return None
    ranked = sorted(range(60), key=lambda i: float(p60[i]), reverse=True)
    top_n  = set(ranked[:N])
    return float(len(top_n - target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-positions',      type=int,  default=5000)
    ap.add_argument('--pos-lo',           type=int,  default=10)
    ap.add_argument('--pos-hi',           type=int,  default=50)
    ap.add_argument('--num-pickle-files', type=int,  default=5)
    ap.add_argument('--seed',             type=int,  default=42)
    ap.add_argument('--alpha-mult',       type=float,default=1.0,
                    help='Multiply calibrated alpha by this factor (default 1.0).')
    ap.add_argument('--out',              default='Intervention Results/paper_numbers.txt')
    ap.add_argument('--cascade-start-li', type=int, default=None,
                    help='Cascade from this starting layer in Li et al. '
                         'numbering (1-8; Li layer k = our block k-1). '
                         'Omit for the single-layer intervention at block 4.')
    ap.add_argument('--later-layers', choices=['skip_if_target', 'always'],
                    default=None,
                    help='Cascade only (required with --cascade-start-li): '
                         'what to do at a later layer whose probe already '
                         'reads the target.')
    ap.add_argument('--push', choices=['reflect', 'add'], default='reflect',
                    help='Cascade only. reflect = ours; add = Nanda et al. 2023 '
                         '(add alpha * unit target-state probe vector).')
    ap.add_argument('--size-mode', choices=['calibrated', 'fixed'],
                    default='calibrated',
                    help='calibrated = each intervention gets its own calibrated '
                         'size; fixed = one size for all (--fixed-alpha), push=add only.')
    ap.add_argument('--fixed-alpha', type=float, default=None,
                    help='The fixed 1x size (use the median calibrated size printed '
                         'by a --size-mode calibrated run).')
    ap.add_argument('--corruption', action='store_true',
                    help='Cascade only: also compute the corruption measures at '
                         'every intervened layer.')
    ap.add_argument('--save-records', default=None,
                    help='Save every intervention (kind, effect, all measures) '
                         'to this .npz path.')
    ap.add_argument('--save-dist', action='store_true',
                    help='With --save-records: also save the full 64-square move '
                         'distribution after each intervention (P_after_64, one '
                         'row per intervention) and with no intervention '
                         '(P_clean_64, one row per position; index by pos_index).')
    args = ap.parse_args()
    cascade = args.cascade_start_li is not None
    if args.corruption and not cascade:
        ap.error('--corruption needs --cascade-start-li')
    if args.push == 'add' and not cascade:
        ap.error('--push add needs --cascade-start-li')
    if args.size_mode == 'fixed':
        if args.push != 'add':
            ap.error('--size-mode fixed is only implemented for --push add')
        if args.fixed_alpha is None:
            ap.error('--size-mode fixed needs --fixed-alpha')
    elif args.fixed_alpha is not None:
        ap.error('--fixed-alpha only applies with --size-mode fixed')
    ADD_CAP = 100.0
    if cascade:
        if not 1 <= args.cascade_start_li <= 8:
            ap.error('--cascade-start-li must be in 1..8')
        if args.later_layers is None:
            ap.error('--later-layers is required with --cascade-start-li')
    elif args.later_layers is not None:
        ap.error('--later-layers only applies with --cascade-start-li')

    rng    = random.Random(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}  alpha_mult={args.alpha_mult}')

    model = intervention.load_model(device=device)
    probe = probes.load_probe(device=device)
    print(f'model + probe loaded; probe shape {tuple(probe.shape)}')

    # Block whose resid_post is edited first (single-layer: INTV_LAYER).
    start_block = (intervention.li_layer_to_block(args.cascade_start_li)
                   if cascade else INTV_LAYER)
    CASC_BLOCKS = list(range(start_block, 8))
    if cascade:
        layer_probes = intervention.load_layer_probes(
            range(start_block, 8), device=device)
        print(f'cascade: Li start layer {args.cascade_start_li} = blocks '
              f'{start_block}..7, later_layers={args.later_layers}, probes: '
              + ', '.join(f'{k}:{os.path.basename(intervention.layer_probe_path(k))}'
                          for k in range(start_block, 8)))

    files = sorted(glob.glob('./data/othello_synthetic/*.pickle'))[-args.num_pickle_files:]
    games = []
    for fp in files:
        with open(fp, 'rb') as fh:
            games.extend(pickle.load(fh))
    rng.shuffle(games)
    games = games[:max(args.n_positions * 3, 10000)]
    print(f'{len(games)} games available')

    STOI = {c: i + 1 for i, c in enumerate(config.STOI_INDICES)}

    # buckets[(condition, sub_condition)] = list of metric dicts
    buckets = collections.defaultdict(list)

    n_processed = 0
    clean_by_pos = {}     # pos_index -> 60-cell clean distribution (--save-dist)
    for g in games:
        if n_processed >= args.n_positions:
            break
        if len(g) <= args.pos_lo:
            continue

        board = OthelloBoardState()
        prefix = []
        target_pos = rng.randint(args.pos_lo, min(args.pos_hi, len(g) - 1))
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

        board2d    = board.state.copy()
        color      = int(board.next_hand_color)
        orig_legal = set(int(m) for m in valid)
        pos        = len(prefix) - 1

        token_ids = torch.tensor(
            [[STOI[m] for m in prefix]], dtype=torch.long, device=device)
        with torch.no_grad():
            prefix_acts  = intervention.compute_prefix(model, token_ids, start_block + 1)
            clean_logits = intervention.forward_from_prefix(
                model, prefix_acts.clone(), start_block + 1)
        clean_last = clean_logits[0, -1].detach().cpu()
        clean_p60  = metrics.cell_probs(clean_last)
        if args.save_dist:
            clean_by_pos[n_processed] = clean_p60.numpy().astype(np.float32)

        if args.corruption:
            # Clean probe readings of every square at every cascade layer, and
            # each square's real state (mode-0 class) on the unedited board.
            true_cls = torch.tensor(
                np.where(board2d == 0, EMPTY_CLS,
                         np.where(board2d == color, MINE_CLS, YOURS_CLS)),
                dtype=torch.long, device=device)                     # (8, 8)
            clean_read = {}
            with torch.no_grad():
                x = prefix_acts.clone()
                for k in CASC_BLOCKS:
                    if k > start_block:
                        x = model.blocks[k](x)
                    lg = torch.einsum('d,drco->rco', x[0, pos],
                                      layer_probes[k][PROBE_MODE])
                    clean_read[k] = (lg.argmax(-1), lg.softmax(-1))

        for flat_cell in VALID64:
            r, c = flat_cell // 8, flat_cell % 8
            orig = int(board2d[r, c])

            if orig == 0:
                conditions = [('add_mine', color), ('add_yours', -color)]
            else:
                conditions = [('remove', 0), ('flip', -orig)]

            for cond_name, tgt in conditions:
                cf  = compute_counterfactual_legal(board2d, [(r, c, tgt)], color)
                sub = classify_change(orig_legal, cf)
                if sub is None:
                    continue

                cc = bv2cls(orig, color)
                tc = bv2cls(tgt, color)
                if cc == tc:
                    continue

                if cascade:
                    # Calibrate once at the starting layer with its own probe,
                    # then reuse that size at every later layer.
                    probe_W = layer_probes[start_block][PROBE_MODE, :, r, c, :]
                    if args.push == 'add':
                        cap = ADD_CAP
                        base_alpha = intervention.calibrate_alpha_add(
                            prefix_acts[0, pos], probe_W, tc, alpha_cap=cap)
                    else:
                        cap = 10.0
                        base_alpha = intervention.calibrate_alpha(
                            model, prefix_acts, pos, probe_W, cc, tc,
                            intervene_layer=start_block, probe_layer=start_block,
                            alpha_cap=cap,
                        )
                    size_1x = (args.fixed_alpha if args.size_mode == 'fixed'
                               else base_alpha)
                    alpha = size_1x * args.alpha_mult
                    intv_logits, cinfo = intervention.run_cascade(
                        model, prefix_acts, pos, (r, c), cc, tc,
                        start_block, alpha, layer_probes,
                        probe_mode=PROBE_MODE, later_layers=args.later_layers,
                        push=args.push)
                    intv_p60 = metrics.cell_probs(intv_logits[0, -1].detach().cpu())
                    if args.corruption:
                        other = torch.ones(8, 8, dtype=torch.bool, device=device)
                        other[r, c] = False
                        flips_other, probe_drop, corr_mask = [], [], []
                        for k in CASC_BLOCKS:
                            lg = torch.einsum('d,drco->rco', cinfo['h_layers'][k],
                                              layer_probes[k][PROBE_MODE])
                            c_pred, c_prob = clean_read[k]
                            i_pred, i_prob = lg.argmax(-1), lg.softmax(-1)
                            changed = (i_pred != c_pred) & other          # (8, 8)
                            flips_other.append(int(changed.sum()))
                            corr_mask.append(changed.flatten().cpu().numpy())
                            pt_c = c_prob.gather(-1, true_cls[..., None])[..., 0]
                            pt_i = i_prob.gather(-1, true_cls[..., None])[..., 0]
                            probe_drop.append(float((pt_c - pt_i)[other].mean()))
                    else:
                        flips_other = probe_drop = corr_mask = None
                    # Hard flip = starting-layer probe after its push.
                    hard_flip = int(cinfo['read_target'][start_block])
                    hard_flip_final = int(cinfo['read_target'][7])
                    n_later = 7 - start_block
                    n_skipped = len(cinfo['skipped'])
                    cal_capped = int(base_alpha >= cap)
                else:
                    probe_W = probe[PROBE_MODE, :, r, c, :]
                    alpha   = intervention.calibrate_alpha(
                        model, prefix_acts, pos, probe_W, cc, tc,
                        intervene_layer=INTV_LAYER, probe_layer=PROBE_LAYER,
                        alpha_cap=10.0,
                    ) * args.alpha_mult

                    spec = intervention.InterventionSpec.from_probe(
                        probe, PROBE_MODE, r, c, cc, tc, prefix_acts[0, pos])
                    intv_logits, intv_resid = intervention.run_with_intervention(
                        model, prefix_acts, pos, [spec], [alpha],
                        intervene_layer=INTV_LAYER, capture_layer=PROBE_LAYER)
                    intv_p60 = metrics.cell_probs(intv_logits[0, -1].detach().cpu())

                    # Hard flip: did probe at PROBE_LAYER predict target class after intervention?
                    probe_logits_after = (intv_resid[0, pos].float() @ probe_W)
                    hard_flip = int(probe_logits_after.argmax().item() == tc)
                    hard_flip_final = n_later = n_skipped = cal_capped = None
                    flips_other = probe_drop = corr_mask = None
                    base_alpha = None

                # Paper "errors" = FP = |top-N ∩ illegal|. Since |top_N|=|target|=N, FP=FN=(FP+FN)/2.
                err_before = li_topn_fp(clean_p60, cf)
                err_after  = li_topn_fp(intv_p60,  cf)

                # Probability shifts
                newly_legal   = cf - orig_legal
                newly_illegal = orig_legal - cf
                P_bef_cf  = metrics.mass_on(clean_p60, cf)
                P_aft_cf  = metrics.mass_on(intv_p60, cf)
                # Moves whose legality the edit did not change.
                stay_legal   = orig_legal & cf
                stay_illegal = ALL60 - (orig_legal | cf)
                P_bef_stayL = metrics.mass_on(clean_p60, stay_legal)
                P_aft_stayL = metrics.mass_on(intv_p60, stay_legal)
                P_bef_stayI = metrics.mass_on(clean_p60, stay_illegal)
                P_aft_stayI = metrics.mass_on(intv_p60, stay_illegal)
                P_bef_nl  = metrics.mass_on(clean_p60, newly_legal)
                P_aft_nl  = metrics.mass_on(intv_p60, newly_legal)
                P_bef_ni  = metrics.mass_on(clean_p60, newly_illegal)
                P_aft_ni  = metrics.mass_on(intv_p60, newly_illegal)

                dP_legal = P_aft_cf - P_bef_cf
                dP_nl    = P_aft_nl - P_bef_nl
                dP_ni    = P_aft_ni - P_bef_ni

                # "Correct direction": intervention moved probability the right way
                # nl: want newly-legal cells to gain probability
                # ni: want newly-illegal cells to lose probability
                # both: need both
                if sub == 'newly_legal':
                    correct_dir = int(dP_nl > 0)
                elif sub == 'newly_illegal':
                    correct_dir = int(dP_ni < 0)
                else:  # 'both'
                    correct_dir = int(dP_nl > 0 and dP_ni < 0)

                # Alternative correct_dir: dP_legal > 0 (prob of all legal_cf moves increased)
                correct_dir_legal = int(dP_legal > 0)

                # Whether top-N errors strictly improved
                err_improved = int(
                    err_before is not None and err_after is not None
                    and err_after < err_before
                )

                buckets[(cond_name, sub)].append({
                    'err_before': err_before,
                    'err_after':  err_after,
                    'dP_legal':   dP_legal,
                    'dP_nl':      dP_nl,
                    'dP_ni':      dP_ni,
                    'P_bef_ni':   P_bef_ni,
                    'correct_dir':       correct_dir,
                    'correct_dir_legal': correct_dir_legal,
                    'err_improved':      err_improved,
                    'hard_flip':         hard_flip,
                    'hard_flip_final':   hard_flip_final,
                    'n_later':           n_later,
                    'n_skipped':         n_skipped,
                    'cal_capped':        cal_capped,
                    'base_alpha':        base_alpha,
                    'alpha':             alpha,
                    'pos_index':         n_processed,
                    'flips_other':       flips_other,
                    'probe_drop':        probe_drop,
                    'corr_mask':         corr_mask,
                    'P_bef_cf':          P_bef_cf,
                    'P_aft_cf':          P_aft_cf,
                    'P_bef_stayL':       P_bef_stayL,
                    'P_aft_stayL':       P_aft_stayL,
                    'P_bef_stayI':       P_bef_stayI,
                    'P_aft_stayI':       P_aft_stayI,
                    'P_bef_nl':          P_bef_nl,
                    'P_aft_nl':          P_aft_nl,
                    'P_aft_ni':          P_aft_ni,
                    'sub':        sub,
                    'cond':       cond_name,
                    'sq':         (r, c),
                    'p60_after':  (intv_p60.numpy().astype(np.float32)
                                   if args.save_dist else None),
                })

        n_processed += 1
        if n_processed % 200 == 0:
            print(f'  {n_processed}/{args.n_positions}', flush=True)

    print(f'Done processing {n_processed} positions.')

    # ── Aggregate ──────────────────────────────────────────────────────────────
    def avg(lst, key):
        vals = [x[key] for x in lst if x.get(key) is not None]
        return float(np.mean(vals)) if vals else float('nan')

    all_entries   = [e for lst in buckets.values() for e in lst]
    flip_entries  = [e for e in all_entries if e['cond'] == 'flip']
    flip_nl       = [e for e in flip_entries if e['sub'] == 'newly_legal']
    flip_ni       = [e for e in flip_entries if e['sub'] == 'newly_illegal']
    flip_both     = [e for e in flip_entries if e['sub'] == 'both']
    nl_entries    = [e for e in all_entries if e['sub'] in ('newly_legal', 'both')]
    ni_entries    = [e for e in all_entries if e['sub'] in ('newly_illegal', 'both')]

    lines = []
    def emit(s=''):
        print(s)
        lines.append(s)

    emit('=' * 70)
    emit(f'SECTION 5.4 — CAUSAL INTERVENTIONS  (n_positions={n_processed}, alpha_mult={args.alpha_mult})')
    if cascade:
        emit(f'CASCADE: Li start layer {args.cascade_start_li} (blocks {start_block}..7), '
             f'push={args.push}, later_layers={args.later_layers}')
        if args.size_mode == 'fixed':
            emit(f'  size: FIXED, 1x = {args.fixed_alpha:g} for every intervention, '
                 f'same at every layer')
        else:
            emit(f'  size: CALIBRATED per intervention at block {start_block}, '
                 f'same size reused at every later layer')
        emit('  probes: ' + ', '.join(
            f'block {k}={os.path.basename(intervention.layer_probe_path(k))}'
            for k in range(start_block, 8)))
    else:
        emit(f'SINGLE LAYER: block {INTV_LAYER}, probe {os.path.basename(config.PROBE_PATH)}')
    emit('=' * 70)

    # Top-N errors: flip-only (matches paper's 1.37 → 0.31 from run_ogpt_per_square_all_squares.py)
    eb_flip = avg(flip_entries, 'err_before')
    ea_flip = avg(flip_entries, 'err_after')
    emit(f'\nTop-N errors FLIP-ONLY (mean FP): before={eb_flip:.2f}  after={ea_flip:.2f}')
    emit(f'  → paper says: 1.37 → 0.31')

    # All conditions for reference (inflated because add_mine/add_yours have large N)
    eb = avg(all_entries, 'err_before')
    ea = avg(all_entries, 'err_after')
    emit(f'Top-N errors ALL cond (reference, FP): before={eb:.2f}  after={ea:.2f}')

    # Correct direction — multiple variants to identify which matches paper's 98.5%
    emit(f'\n── Correct direction variants ──')
    # (A) all entries, original metric: dP_nl>0 / dP_ni<0 / both
    cd_all = avg(all_entries, 'correct_dir')
    # (B) flip entries, original metric
    cd_flip = avg(flip_entries, 'correct_dir')
    # (C) nl+both entries only: dP_nl > 0
    cd_nl = avg(nl_entries, 'correct_dir')
    # (D) flip/nl only: dP_nl > 0
    cd_flip_nl = avg(flip_nl, 'correct_dir')
    # (E) all entries: abs_dP_legal > 0
    cd_legal_all = avg(all_entries, 'correct_dir_legal')
    # (F) flip entries: abs_dP_legal > 0
    cd_legal_flip = avg(flip_entries, 'correct_dir_legal')
    # (G) all entries: err_after < err_before
    cd_err_all = avg(all_entries, 'err_improved')
    # (H) flip entries: err_after < err_before
    cd_err_flip = avg(flip_entries, 'err_improved')
    emit(f'  (A) ALL conds, dP_nl>0/dP_ni<0:         {cd_all * 100:.1f}%')
    emit(f'  (B) FLIP only, dP_nl>0/dP_ni<0:         {cd_flip * 100:.1f}%')
    emit(f'  (C) nl+both only, dP_nl>0:               {cd_nl * 100:.1f}%')
    emit(f'  (D) flip/nl only, dP_nl>0:               {cd_flip_nl * 100:.1f}%')
    emit(f'  (E) ALL conds, abs_dP_legal>0:            {cd_legal_all * 100:.1f}%')
    emit(f'  (F) FLIP only, abs_dP_legal>0:            {cd_legal_flip * 100:.1f}%')
    cd_hard_all  = avg(all_entries,  'hard_flip')
    cd_hard_flip = avg(flip_entries, 'hard_flip')
    cd_hard_nl   = avg(nl_entries,   'hard_flip')
    emit(f'  (G) ALL conds, err_after<err_before:      {cd_err_all * 100:.1f}%')
    emit(f'  (H) FLIP only, err_after<err_before:      {cd_err_flip * 100:.1f}%')
    emit(f'  (I) ALL conds, hard probe flip:            {cd_hard_all * 100:.1f}%')
    emit(f'  (J) FLIP only, hard probe flip:            {cd_hard_flip * 100:.1f}%')
    emit(f'  (K) nl+both only, hard probe flip:         {cd_hard_nl * 100:.1f}%')
    emit(f'  → paper says: 98.5%  (likely from variant D or K)')
    if cascade:
        emit(f'\n── Cascade diagnostics ──')
        emit(f'  (L) ALL conds, block-7 probe reads target at the end: '
             f'{avg(all_entries, "hard_flip_final") * 100:.1f}%')
        n_later_tot = sum(e['n_later'] for e in all_entries)
        n_skip_tot  = sum(e['n_skipped'] for e in all_entries)
        emit(f'  later-layer pushes skipped (already read target): '
             f'{n_skip_tot}/{n_later_tot} = '
             f'{(n_skip_tot / n_later_tot * 100) if n_later_tot else 0:.1f}%')
        cap_txt = ADD_CAP if args.push == 'add' else 10.0
        emit(f'  starting-layer calibration hit alpha_cap={cap_txt:g}: '
             f'{avg(all_entries, "cal_capped") * 100:.1f}%')
        ba = np.array([e['base_alpha'] for e in all_entries], dtype=float)
        q = np.percentile(ba, [10, 25, 50, 75, 90])
        emit(f'  per-intervention calibrated size: 10th={q[0]:.4f} 25th={q[1]:.4f} '
             f'median={q[2]:.4f} 75th={q[3]:.4f} 90th={q[4]:.4f}')
        emit(f'  FIXED_1X_SIZE (median = smallest single size that flips the '
             f'majority at block {start_block}): {q[2]:.6f}')

    # Newly-legal shift (nl+both only — entries where newly-legal moves exist)
    dP_nl_mean = avg(nl_entries, 'dP_nl')
    emit(f'\nMean dP_newly_legal (nl+both entries): {dP_nl_mean * 100:.4f}%')
    emit(f'  → paper says: 0.25%')

    # P_before_ni and dP_ni (for ni + both sub-conditions)
    P_bef_ni_mean = avg(ni_entries, 'P_bef_ni')
    dP_ni_mean    = avg(ni_entries, 'dP_ni')
    emit(f'\nP_before_newly_illegal (ni+both): {P_bef_ni_mean * 100:.2f}%')
    emit(f'  → paper says: 10.83%')
    emit(f'dP_newly_illegal (ni+both):       {abs(dP_ni_mean) * 100:.2f}%  (paper says 2.15%)')
    if P_bef_ni_mean > 0:
        emit(f'  = {abs(dP_ni_mean) / P_bef_ni_mean * 100:.2f}% of P_before  (paper: 2.15% of 10.83%)')

    # ── TABLE 4 ────────────────────────────────────────────────────────────────
    emit('\n── TABLE 4: per kind (paper: Add opp +0.19/-3.03, Add own +0.38/-3.75, Flip +0.05/-0.86, Remove +0.38/-0.96) ──')
    COND_NAMES = {'add_mine': 'Add own piece', 'add_yours': "Add opp piece",
                  'flip':     'Flip',           'remove':   'Remove'}
    for cond in ['add_mine', 'add_yours', 'flip', 'remove']:
        nl_ents = buckets.get((cond, 'newly_legal'), []) + buckets.get((cond, 'both'), [])
        ni_ents = buckets.get((cond, 'newly_illegal'), []) + buckets.get((cond, 'both'), [])
        dP_nl_c = avg(nl_ents, 'dP_nl')
        dP_ni_c = avg(ni_ents, 'dP_ni')
        emit(f'  {COND_NAMES[cond]:18s}  NL={dP_nl_c * 100:+.4f}%  NI={dP_ni_c * 100:+.4f}%')

    # ── TABLE 5 ────────────────────────────────────────────────────────────────
    emit('\n── TABLE 5: per-square percentiles ──')
    sq_nl: dict = {}
    sq_ni: dict = {}
    for (cond, sub), ents in buckets.items():
        for e in ents:
            sq = e['sq']
            if sub in ('newly_legal', 'both'):
                sq_nl.setdefault(sq, []).append(e['dP_nl'])
            if sub in ('newly_illegal', 'both'):
                sq_ni.setdefault(sq, []).append(e['dP_ni'])

    nl_by_sq = [np.mean(v) for v in sq_nl.values() if v]
    ni_by_sq = [abs(np.mean(v)) for v in sq_ni.values() if v]

    if nl_by_sq:
        nl_arr = sorted(nl_by_sq)
        ni_arr = sorted(ni_by_sq)
        p10_nl = float(np.percentile(nl_arr, 10))
        p50_nl = float(np.percentile(nl_arr, 50))
        p90_nl = float(np.percentile(nl_arr, 90))
        p10_ni = float(np.percentile(ni_arr, 10))
        p50_ni = float(np.percentile(ni_arr, 50))
        p90_ni = float(np.percentile(ni_arr, 90))
        emit(f'  Promotion  (NL): median={p50_nl*100:.3f}%  10th={p10_nl*100:.3f}%  90th={p90_nl*100:.3f}%'
             f'  ratio={p90_nl/p10_nl if p10_nl > 0 else float("nan"):.1f}x')
        emit(f'  Suppression(NI): median={p50_ni*100:.3f}%  10th={p10_ni*100:.3f}%  90th={p90_ni*100:.3f}%'
             f'  ratio={p90_ni/p10_ni if p10_ni > 0 else float("nan"):.1f}x')
        emit(f'  → paper Table 5: Promotion: median 0.146%, 10th 0.072%, 90th 0.537%, ratio 7.4x')
        emit(f'                   Suppression: median 2.050%, 10th 1.402%, 90th 3.120%, ratio 2.2x')

    # ── Per-condition n summary ────────────────────────────────────────────────
    emit('\n── Per-condition n ──')
    for cond in ['add_mine', 'add_yours', 'flip', 'remove']:
        for sub in ['both', 'newly_illegal', 'newly_legal']:
            n = len(buckets.get((cond, sub), []))
            emit(f'  {cond}/{sub}: n={n}')

    if args.corruption:
        last = CASC_BLOCKS[-1]
        li_names = [f'L{k + 1}' for k in CASC_BLOCKS]        # Li numbering

        def corr_row(es):
            fo = np.array([e['flips_other'] for e in es], dtype=float)
            pdp = np.array([e['probe_drop'] for e in es], dtype=float)
            g = lambda k: np.array([e[k] for e in es], dtype=float)
            return dict(
                n=len(es), flips=fo.mean(0), drop=pdp.mean(0),
                stayL=(g('P_aft_stayL') - g('P_bef_stayL')).mean(),
                stayI=(g('P_aft_stayI') - g('P_bef_stayI')).mean(),
                legal_cf=(g('P_aft_cf') - g('P_bef_cf')).mean(),
            )

        emit('\n── CORRUPTION ──')
        emit('  (1) flips of other squares = number of squares other than the '
             'target whose probe reading changed (out of 63)')
        emit("  (2) probe drop = average drop in each other square's probe "
             'probability for its real state')
        emit('  (3a) moves whose legality did not change: change in total '
             'probability on still-legal moves (drop = corruption) and on '
             'still-illegal moves (rise = corruption)')
        emit('  (3b) all moves on the edited board: change in total probability '
             'on legal moves (illegal = 1 - legal, so it is the negative)')
        a = corr_row(all_entries)
        emit(f'\n  By layer, all interventions (n={a["n"]}):')
        emit('    layer  flips_other  probe_drop')
        for i, nm in enumerate(li_names):
            emit(f'    {nm:>5s}  {a["flips"][i]:11.3f}  {a["drop"][i] * 100:9.3f}%')
        emit(f'\n  Last layer ({li_names[-1]}) by kind and effect:')
        emit(f'    {"kind":10s} {"effect":14s} {"n":>6s}  {"flips":>6s}  {"drop":>7s}  '
             f'{"dP stillLegal":>13s}  {"dP stillIllegal":>15s}  {"dP legal(edited)":>16s}')
        for (cname, sname) in sorted(buckets):
            row = corr_row(buckets[(cname, sname)])
            emit(f'    {cname:10s} {sname:14s} {row["n"]:6d}  {row["flips"][-1]:6.3f}  '
                 f'{row["drop"][-1] * 100:6.3f}%  {row["stayL"] * 100:+12.3f}%  '
                 f'{row["stayI"] * 100:+14.3f}%  {row["legal_cf"] * 100:+15.3f}%')
        emit(f'    {"ALL":10s} {"":14s} {a["n"]:6d}  {a["flips"][-1]:6.3f}  '
             f'{a["drop"][-1] * 100:6.3f}%  {a["stayL"] * 100:+12.3f}%  '
             f'{a["stayI"] * 100:+14.3f}%  {a["legal_cf"] * 100:+15.3f}%')

    if args.save_records:
        os.makedirs(os.path.dirname(args.save_records) or '.', exist_ok=True)
        keys = ['err_before', 'err_after', 'dP_legal', 'dP_nl', 'dP_ni',
                'P_bef_ni', 'P_aft_ni', 'P_bef_nl', 'P_aft_nl', 'P_bef_cf',
                'P_aft_cf', 'P_bef_stayL', 'P_aft_stayL', 'P_bef_stayI',
                'P_aft_stayI', 'correct_dir', 'hard_flip', 'hard_flip_final',
                'base_alpha', 'alpha', 'pos_index']
        rec = {k: np.array([np.nan if e.get(k) is None else e[k]
                            for e in all_entries], dtype=np.float32) for k in keys}
        rec['cond'] = np.array([e['cond'] for e in all_entries])
        rec['sub'] = np.array([e['sub'] for e in all_entries])
        rec['square'] = np.array([e['sq'] for e in all_entries], dtype=np.int8)
        if args.corruption:
            rec['flips_other'] = np.array([e['flips_other'] for e in all_entries],
                                          dtype=np.int8)
            rec['probe_drop'] = np.array([e['probe_drop'] for e in all_entries],
                                         dtype=np.float32)
            rec['layer_blocks'] = np.array(CASC_BLOCKS)
            # corrupted_squares[i, layer, sq] = True if the probe reading of
            # square sq (flat 8*r + c) differs from the clean run at that layer
            # for intervention i. The intervened square itself is always False.
            # Summing over the last axis gives flips_other.
            rec['corrupted_squares'] = np.stack(
                [np.stack(e['corr_mask']) for e in all_entries]).astype(bool)
        if args.save_dist:
            # 64 columns = flat square 8*r + c; the 4 center squares are 0
            # (the model has no move for them). Softmax over the 60 moves,
            # same as every other probability in this file.
            idx = np.array(config.STOI_INDICES)
            after = np.zeros((len(all_entries), 64), dtype=np.float32)
            after[:, idx] = np.stack([e['p60_after'] for e in all_entries])
            clean = np.zeros((n_processed, 64), dtype=np.float32)
            clean[:, idx] = np.stack([clean_by_pos[i] for i in range(n_processed)])
            rec['P_after_64'] = after
            rec['P_clean_64'] = clean
        np.savez_compressed(args.save_records, **rec)
        emit(f'Saved per-intervention records → {args.save_records}')

    # Save
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    emit(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
