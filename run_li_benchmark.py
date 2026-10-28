"""Run our linear-probe intervention on Li et al.'s pre-built intervention
benchmark (intervention_benchmark.pkl) to compare numbers on the same 1001
flip-only positions used in their paper.

This lets us separate position-selection effects from method effects: any
remaining difference between our numbers and theirs is due to intervention
method (gradient-based vs. our linear calibrated-alpha approach), not the
choice of positions.
"""

import argparse
import collections
import os
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from src import config, intervention, probes, data, metrics
from src import io as iox

ap = argparse.ArgumentParser()
ap.add_argument('--intervention-layer', type=int, default=4)
ap.add_argument('--probe-path', default=None,
                help='Path to probe .pth file. Default: layer-6 probe.')
ap.add_argument('--benchmark-path', default='intervention_benchmark.pkl')
ap.add_argument('--out-suffix', default='',
                help='Suffix appended to output filenames (e.g. _L4).')
ap.add_argument('--alpha-mults', type=float, nargs='+',
                default=[1, 2, 2.5, 3, 4, 6, 8],
                help='Multipliers on the calibrated alpha.')
ap.add_argument('--cascade-start-li', type=int, default=None,
                help='Cascade from this starting layer in Li et al. numbering '
                     '(1-8; Li layer k = our block k-1), each layer using its '
                     'own linear probe. Omit for the single-layer intervention.')
ap.add_argument('--later-layers', choices=['skip_if_target', 'always'],
                default=None,
                help='Cascade only (required with --cascade-start-li): what to '
                     'do at a later layer whose probe already reads the target.')
args = ap.parse_args()

CASCADE = args.cascade_start_li is not None
if CASCADE:
    if not 1 <= args.cascade_start_li <= 8:
        ap.error('--cascade-start-li must be in 1..8')
    if args.later_layers is None:
        ap.error('--later-layers is required with --cascade-start-li')
    if args.probe_path is not None:
        ap.error('--probe-path does not apply to the cascade (it uses each '
                 "layer's own probe)")
elif args.later_layers is not None:
    ap.error('--later-layers only applies with --cascade-start-li')

ALPHA_MULTS = args.alpha_mults
START_BLOCK = (intervention.li_layer_to_block(args.cascade_start_li)
               if CASCADE else args.intervention_layer)

params = config.assemble_params(
    intervention_layer        = START_BLOCK,
    cal_depth                 = 0,
    calibration_mode          = 'per_square',
    alpha                     = 2.0,
    probe_mode                = 0,
    n_positions_per_condition = 1,
    max_games                 = 1,
    pos_lo                    = 0,
    pos_hi                    = 60,
    seed                      = 42,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}  intervention_layer={START_BLOCK}')

model  = intervention.load_model(device=DEVICE)
if CASCADE:
    layer_probes = intervention.load_layer_probes(range(START_BLOCK, 8), device=DEVICE)
    # Calibration and the starting push use the starting layer's own probe.
    probe = layer_probes[START_BLOCK]
    CASCADE_DESC = (
        f'CASCADE: Li start layer {args.cascade_start_li} (blocks {START_BLOCK}..7), '
        f'size calibrated once at block {START_BLOCK} and reused, '
        f'later_layers={args.later_layers}; probes: '
        + ', '.join(f'block {k}={os.path.basename(intervention.layer_probe_path(k))}'
                    for k in range(START_BLOCK, 8)))
else:
    probe  = probes.load_probe(probe_path=args.probe_path, device=DEVICE)
    CASCADE_DESC = (f'SINGLE LAYER: block {START_BLOCK}, probe '
                    f'{os.path.basename(args.probe_path or config.PROBE_PATH)}')
print(CASCADE_DESC)
print(f'model ready; probe shape {tuple(probe.shape)}')

# ── load benchmark ─────────────────────────────────────────────────────────────
print(f'loading benchmark from {args.benchmark_path}...')
with open(args.benchmark_path, 'rb') as f:
    raw = pickle.load(f)
print(f'{len(raw)} raw benchmark cases')

# ── build PositionRecords ──────────────────────────────────────────────────────
# ori_color encoding: 0.0 = white (board_val = -1), 2.0 = black (board_val = +1)
records = []
skipped = 0
for case in raw:
    history  = [int(m) for m in case['history']]
    pos_flat = int(case['pos_int'])          # flat board index 0-63
    r, c     = pos_flat // 8, pos_flat % 8

    game_arr = np.array(history, dtype=np.int64)
    pos      = len(history) - 1             # last move index

    rec = data.position_from_replay(game_arr, pos, (r, c), 'flip')
    if rec is None:
        skipped += 1
        continue
    records.append(rec)

print(f'{len(records)} valid records ({skipped} skipped)')

# Sub-condition breakdown
sub_counts = collections.Counter(r.sub_condition for r in records)
print('sub-condition counts:', dict(sub_counts))

# ── cache clean state ──────────────────────────────────────────────────────────
print('caching clean forward state...')
for rec in tqdm(records):
    intervention.cache_clean_state(
        model, rec, probe,
        intervention_layer = params['intervention_layer'],
        probe_layer        = params['probe_layer'],
        probe_mode         = params['probe_mode'],
        device             = DEVICE,
    )

# ── per-square calibrated alpha ────────────────────────────────────────────────
sq_alpha_cache = {}

def alpha_for(rec):
    sq_key = rec.square
    if sq_key not in sq_alpha_cache:
        r, c = sq_key
        probe_cell_W = probe[params['probe_mode'], :, r, c, :]
        a = intervention.calibrate_alpha(
            model,
            rec.extras['prefix_acts'],
            rec.position,
            probe_cell_W,
            rec.extras['current_class'],
            rec.extras['target_class'],
            intervene_layer = params['intervention_layer'],
            probe_layer     = params['probe_layer'],
            alpha_cap       = params['alpha'] if params['alpha'] > 0 else 10.0,
        )
        sq_alpha_cache[sq_key] = a
        print(f'  calibrated alpha for {sq_key}: {a:.4f}', flush=True)
    return sq_alpha_cache[sq_key]


def li_topn_errors(logits_last, legal_set):
    """Li et al. top-N metric: raw error count (FP + FN). Returns (errors, n)."""
    legal_stoi = set(legal_set) & set(config.STOI_INDICES)
    n = len(legal_stoi)
    if n == 0:
        return None, 0
    cls        = metrics.cell_logits(logits_last)
    topn       = cls.argsort(descending=True)[:n]
    topn_cells = {config.STOI_INDICES[int(i)] for i in topn}
    errors     = len(topn_cells - legal_stoi) + len(legal_stoi - topn_cells)
    return errors, n


def li_topn_accuracy(logits_last, legal_set):
    """Li et al. top-N metric: 1 - (FP + FN) / (2 * N)."""
    errors, n = li_topn_errors(logits_last, legal_set)
    if errors is None:
        return None
    return 1.0 - errors / (2.0 * n)


# ── main sweep ─────────────────────────────────────────────────────────────────
print('\nrunning sweep...')
# buckets[mult][sub] = list of per-record metric dicts
buckets = {m: collections.defaultdict(list) for m in ALPHA_MULTS}

for rec in tqdm(records, desc='li benchmark sweep'):
    base_alpha = alpha_for(rec)
    sub        = rec.sub_condition

    errors_before, n_legal = li_topn_errors(rec.extras['clean_logits_last'], rec.legal_cf)
    lb         = (1.0 - errors_before / (2.0 * n_legal)) if errors_before is not None else None
    clean_p60  = metrics.cell_probs(rec.extras['clean_logits_last'])
    p_legal_orig_clean = metrics.mass_on(clean_p60, rec.legal_orig)

    # probe score on target square before intervention
    r, c = rec.square
    probe_cell_W = probe[params['probe_mode'], :, r, c, :]   # (d_model, 3)
    h_clean = rec.extras['clean_resid_at_probe'][0, rec.position].to(probe_cell_W.device)
    logits_clean = torch.einsum('d,dl->l', h_clean, probe_cell_W)
    probs_clean  = torch.softmax(logits_clean, dim=0)
    probe_score_before = probs_clean[rec.extras['target_class']].item()

    for mult in ALPHA_MULTS:
        cinfo = None
        if CASCADE:
            intv_logits, cinfo = intervention.run_cascade(
                model,
                rec.extras['prefix_acts'],
                rec.position,
                rec.square,
                rec.extras['current_class'],
                rec.extras['target_class'],
                START_BLOCK,
                base_alpha * mult,
                layer_probes,
                probe_mode   = params['probe_mode'],
                later_layers = args.later_layers,
            )
            h_intv = cinfo['h_start'].to(probe_cell_W.device)
        else:
            intv_logits, intv_resid = intervention.run_with_intervention(
                model,
                rec.extras['prefix_acts'],
                rec.position,
                [rec.extras['spec']],
                [base_alpha * mult],
                intervene_layer = params['intervention_layer'],
                capture_layer   = params['probe_layer'],
            )
            h_intv = intv_resid[0, rec.position].to(probe_cell_W.device)
        intv_last  = intv_logits[0, -1].detach().cpu()
        logits_intv = torch.einsum('d,dl->l', h_intv, probe_cell_W)
        probs_intv  = torch.softmax(logits_intv, dim=0)
        probe_score_after = probs_intv[rec.extras['target_class']].item()

        errors_after, _ = li_topn_errors(intv_last, rec.legal_cf)
        la       = (1.0 - errors_after / (2.0 * n_legal)) if errors_after is not None else None
        li_shift = (la - lb) if (la is not None and lb is not None) else None

        m = metrics.mass_shift_metrics(
            rec.extras['clean_logits_last'], intv_last,
            rec.legal_orig, rec.legal_cf,
        )
        buckets[mult][sub].append({
            'errors_before':       errors_before,
            'errors_after':        errors_after,
            'li_acc_before':       lb,
            'li_acc_after':        la,
            'li_shift':            li_shift,
            'probe_score_before':  probe_score_before,
            'probe_score_after':   probe_score_after,
            'P_before_legal_orig': p_legal_orig_clean,
            'final_reads_target':  (int(cinfo['read_target'][7]) if cinfo else None),
            'n_later':             (7 - START_BLOCK if cinfo else None),
            'n_skipped':           (len(cinfo['skipped']) if cinfo else None),
            **m,
        })

# ── save results ───────────────────────────────────────────────────────────────
print('saving results...')

def mean_f(entries, k):
    vals = [e[k] for e in entries if e.get(k) is not None]
    return sum(vals) / len(vals) if vals else float('nan')


# Table 1: rank improvement + probability shift
rows1 = []
for mult in ALPHA_MULTS:
    for sub, entries in sorted(buckets[mult].items()):
        rows1.append([
            'flip', sub, f'{mult:g}x', len(entries),
            mean_f(entries, 'errors_before'),
            mean_f(entries, 'errors_after'),
            mean_f(entries, 'li_acc_before'),
            mean_f(entries, 'li_acc_after'),
            mean_f(entries, 'li_shift'),
            mean_f(entries, 'P_before_newly_legal'),
            mean_f(entries, 'P_after_newly_legal'),
            mean_f(entries, 'abs_dP_newly_legal'),
            mean_f(entries, 'P_before_newly_illegal'),
            mean_f(entries, 'P_after_newly_illegal'),
            mean_f(entries, 'abs_dP_newly_illegal'),
        ])

iox.save_table(
    rows1,
    filename=f'li_benchmark_results{args.out_suffix}.txt',
    headers=[
        'category', 'sub_condition', 'alpha_mult', 'n',
        'avg_errors_before', 'avg_errors_after',
        'li_acc_before', 'li_acc_after', 'li_topn_shift',
        'P_before_newly_legal', 'P_after_newly_legal', 'dP_newly_legal',
        'P_before_newly_illegal', 'P_after_newly_illegal', 'dP_newly_illegal',
    ],
    title=(
        CASCADE_DESC + '\n'
        'Li et al. benchmark (1001 flip-only positions) — our linear probe method\n'
        'avg_errors = avg (FP+FN) in top-N — Li et al. report 0.12 (natural), 0.06 (unnatural)\n'
        'li_topn_shift = change in Li top-N accuracy after intervention (higher is better)\n'
        'dP_newly_legal = probability mass gained on newly-legal cells (higher is better)\n'
        'dP_newly_illegal = probability mass change on newly-illegal cells (lower is better)'
    ),
)

# Table 2: illegal probability
rows2 = []
for mult in ALPHA_MULTS:
    for sub, entries in sorted(buckets[mult].items()):
        p_ill_before = [1.0 - e['P_before_legal_orig'] for e in entries]
        p_ill_after  = [1.0 - e['P_after_legal_cf']    for e in entries]
        dp_ill       = [a - b for a, b in zip(p_ill_after, p_ill_before)]
        n = len(entries)
        rows2.append([
            'flip', sub, f'{mult:g}x', n,
            sum(p_ill_before) / n,
            sum(p_ill_after)  / n,
            sum(dp_ill)       / n,
        ])

iox.save_table(
    rows2,
    filename=f'li_benchmark_illegal_prob{args.out_suffix}.txt',
    headers=[
        'category', 'sub_condition', 'alpha_mult', 'n',
        'P_illegal_before', 'P_illegal_after', 'dP_illegal',
    ],
    title=(
        'Li et al. benchmark — total probability mass on illegal moves\n'
        'P_illegal_before = 1 - P(legal_orig) on clean output\n'
        'P_illegal_after  = 1 - P(legal_cf) on intervened output\n'
        'dP_illegal = P_after - P_before (negative = fewer illegal moves after intervention)'
    ),
)

# Table 3: probe score (confidence of probe on target class at target square)
rows3 = []
for mult in ALPHA_MULTS:
    for sub, entries in sorted(buckets[mult].items()):
        rows3.append([
            'flip', sub, f'{mult:g}x', len(entries),
            mean_f(entries, 'probe_score_before'),
            mean_f(entries, 'probe_score_after'),
        ])

iox.save_table(
    rows3,
    filename=f'li_benchmark_probe_score{args.out_suffix}.txt',
    headers=[
        'category', 'sub_condition', 'alpha_mult', 'n',
        'probe_score_before', 'probe_score_after',
    ],
    title=(
        'Li et al. benchmark — probe confidence on target class at intervened square\n'
        'probe_score = softmax(probe[mode,:,r,c,:] @ h)[target_class]\n'
        'probe_score_before: clean forward (no intervention)\n'
        'probe_score_after:  after intervention — natural range ~0.65, random ~0.33\n'
        'At 1x we expect ~0.50 (probe just barely flipped); 2.5x should hit ~0.65'
    ),
)

if CASCADE:
    rows4 = []
    for mult in ALPHA_MULTS:
        entries = [e for es in buckets[mult].values() for e in es]
        n_later_tot = sum(e['n_later'] for e in entries)
        n_skip_tot  = sum(e['n_skipped'] for e in entries)
        rows4.append([
            f'{mult:g}x', len(entries),
            mean_f(entries, 'final_reads_target'),
            (n_skip_tot / n_later_tot) if n_later_tot else 0.0,
        ])
    iox.save_table(
        rows4,
        filename=f'li_benchmark_cascade_diag{args.out_suffix}.txt',
        headers=['alpha_mult', 'n', 'frac_block7_reads_target',
                 'frac_later_pushes_skipped'],
        title=(
            CASCADE_DESC + '\n'
            'frac_block7_reads_target = block-7 probe reads the target at the end\n'
            'frac_later_pushes_skipped = later-layer pushes skipped because that '
            "layer's probe already read the target"
        ),
    )

print('done. results saved to Intervention Results/')
