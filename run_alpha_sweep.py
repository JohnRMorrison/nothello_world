"""Alpha sweep: run intervention at 1x,2x,3x,4x,6x,8x,10x of calibrated alpha.
Records rank improvement and number of board squares whose probe prediction
changes (i.e. gets corrupted) at each level, broken down by category
(flip/remove/add_mine/add_yours) and sub-condition."""

import argparse
import os
import sys
import collections

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
ap.add_argument('--n-positions', type=int, default=200,
                help='Positions per (square, category, sub-condition).')
ap.add_argument('--out-suffix', default='',
                help='Suffix appended to output filenames (e.g. _L4).')
args = ap.parse_args()

# ── params ────────────────────────────────────────────────────────────────────
ALPHA_MULTS = [1, 2, 3, 4, 6, 8, 10]

CENTER = {(3, 3), (3, 4), (4, 3), (4, 4)}
squares = [(r, c) for r in range(8) for c in range(8) if (r, c) not in CENTER]

params = config.assemble_params(
    intervention_layer        = args.intervention_layer,
    cal_depth                 = 0,
    calibration_mode          = 'per_square',
    alpha                     = 2.0,
    probe_mode                = 0,
    n_positions_per_condition = args.n_positions,
    max_games                 = 10_000,
    pos_lo                    = 10,
    pos_hi                    = 50,
    seed                      = 42,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}  intervention_layer={args.intervention_layer}')

# ── load ──────────────────────────────────────────────────────────────────────
model = intervention.load_model(device=DEVICE)
probe = probes.load_probe(probe_path=args.probe_path, device=DEVICE)
board_seqs_int, board_seqs_string = data.load_games()
print(f'model ready; probe shape {tuple(probe.shape)}; {len(board_seqs_int)} games loaded')

# ── build database ─────────────────────────────────────────────────────────────
db = data.sample_database(
    board_seqs_string,
    squares=squares,
    n_positions_per_condition=params['n_positions_per_condition'],
    max_games=params['max_games'],
    pos_lo=params['pos_lo'],
    pos_hi=params['pos_hi'],
    seed=params['seed'],
    verbose=True,
)

records = [
    (sq_label, cat, sub, rec)
    for sq_label, cats in db.items()
    for cat, subs in cats.items()
    for sub, recs in subs.items()
    for rec in recs
]
print(f'{len(records)} total records')

# ── cache clean state ──────────────────────────────────────────────────────────
print('caching clean forward state...')
for _, _, _, rec in tqdm(records):
    intervention.cache_clean_state(
        model, rec, probe,
        intervention_layer=params['intervention_layer'],
        probe_layer=params['probe_layer'],
        probe_mode=params['probe_mode'],
        device=DEVICE,
    )

# ── per-square calibrated alpha ────────────────────────────────────────────────
sq_alpha_cache = {}

def alpha_for(sq_key, rec):
    if sq_key not in sq_alpha_cache:
        inner = sq_key[0].strip('()').split(',')
        r, c = int(inner[0]), int(inner[1])
        probe_cell_W = probe[params['probe_mode'], :, r, c, :]
        a = intervention.calibrate_alpha(
            model,
            rec.extras['prefix_acts'],
            rec.position,
            probe_cell_W,
            rec.extras['current_class'],
            rec.extras['target_class'],
            intervene_layer=params['intervention_layer'],
            probe_layer=params['probe_layer'],
            alpha_cap=params['alpha'] if params['alpha'] > 0 else 10.0,
        )
        sq_alpha_cache[sq_key] = a
        print(f'  calibrated alpha for {sq_key}: {a:.4f}', flush=True)
    return sq_alpha_cache[sq_key]


# ── corruption counting ────────────────────────────────────────────────────────
probe_weights = probe[params['probe_mode']]  # (512, 8, 8, 3) on DEVICE

def probe_preds_all_squares(h):
    """Apply probe to residual vector h (512,) → (8,8) predicted class."""
    h = h.to(probe_weights.device)
    logits = torch.einsum('d,drcl->rcl', h, probe_weights)  # (8,8,3)
    return logits.argmax(dim=-1)  # (8,8)

def count_corrupted(clean_resid, intv_resid, pos, target_r, target_c):
    """Count board squares whose probe prediction changed after intervention.

    Returns:
        n_total      - total squares with changed prediction (includes target)
        n_collateral - squares other than the target that changed
        target_flipped - 1 if target square prediction changed, 0 otherwise
    """
    h_clean = clean_resid[0, pos]
    h_intv  = intv_resid[0, pos]
    clean_pred = probe_preds_all_squares(h_clean)  # (8,8)
    intv_pred  = probe_preds_all_squares(h_intv)   # (8,8)
    changed = (clean_pred != intv_pred)
    n_total = changed.sum().item()
    target_changed = int(changed[target_r, target_c].item())
    n_collateral = n_total - target_changed
    return n_total, n_collateral, target_changed


# ── main sweep ────────────────────────────────────────────────────────────────
print('\nrunning alpha sweep...')
# buckets[mult][(sq_label, cat, sub)] = list of per-record dicts
buckets = {m: collections.defaultdict(list) for m in ALPHA_MULTS}

for sq_label, cat, sub, rec in tqdm(records, desc='alpha sweep'):
    sq_key = (sq_label, cat, sub)
    base_alpha = alpha_for(sq_key, rec)

    lb = metrics.li_topn_accuracy(rec.extras['clean_logits_last'], rec.legal_cf)
    r_tgt, c_tgt = rec.square
    clean_resid = rec.extras['clean_resid_at_probe']

    for mult in ALPHA_MULTS:
        alpha_val = base_alpha * mult
        intv_logits, intv_resid = intervention.run_with_intervention(
            model,
            rec.extras['prefix_acts'],
            rec.position,
            [rec.extras['spec']],
            [alpha_val],
            intervene_layer=params['intervention_layer'],
            capture_layer=params['probe_layer'],
        )
        intv_last  = intv_logits[0, -1].detach().cpu()
        intv_resid = intv_resid.detach().cpu()

        la = metrics.li_topn_accuracy(intv_last, rec.legal_cf)
        li_shift = (la - lb) if (la is not None and lb is not None) else None

        n_total, n_collateral, target_flipped = count_corrupted(
            clean_resid, intv_resid, rec.position, r_tgt, c_tgt,
        )

        m = metrics.mass_shift_metrics(
            rec.extras['clean_logits_last'], intv_last,
            rec.legal_orig, rec.legal_cf,
        )
        buckets[mult][sq_key].append({
            'li_shift': li_shift,
            'n_corrupted_total': n_total,
            'n_corrupted_collateral': n_collateral,
            'target_flipped': target_flipped,
            **m,
        })

# ── save results ──────────────────────────────────────────────────────────────
print('saving results...')

def mean_field(entries, k):
    vals = [e[k] for e in entries if e.get(k) is not None]
    return sum(vals) / len(vals) if vals else float('nan')

# Table 1: rank improvement at each alpha multiplier
rows1 = []
for mult in ALPHA_MULTS:
    for sq_key, entries in sorted(buckets[mult].items()):
        sq_label, cat, sub = sq_key
        rows1.append([
            sq_label, cat, sub, f'{mult}x', len(entries),
            mean_field(entries, 'li_shift'),
        ])

iox.save_table(
    rows1,
    filename=f'alpha_sweep_rank_improvement{args.out_suffix}.txt',
    headers=['square', 'category', 'sub_condition', 'alpha_mult', 'n', 'rank_improvement'],
    title='Alpha sweep — rank improvement at each multiplier of calibrated alpha',
)

# Table 2: corruption counts at each alpha multiplier
# n_corrupted_total    = total squares with changed probe prediction (includes target)
# n_corrupted_collateral = squares OTHER than target that changed (unintended side effects)
# target_flipped_rate  = fraction of positions where target square probe actually flipped
rows2 = []
for mult in ALPHA_MULTS:
    for sq_key, entries in sorted(buckets[mult].items()):
        sq_label, cat, sub = sq_key
        rows2.append([
            sq_label, cat, sub, f'{mult}x', len(entries),
            mean_field(entries, 'n_corrupted_total'),
            mean_field(entries, 'n_corrupted_collateral'),
            mean_field(entries, 'target_flipped'),
        ])

iox.save_table(
    rows2,
    filename=f'alpha_sweep_corruption{args.out_suffix}.txt',
    headers=['square', 'category', 'sub_condition', 'alpha_mult', 'n',
             'avg_squares_changed', 'avg_collateral_squares', 'target_flip_rate'],
    title=(
        'Alpha sweep — probe corruption at each multiplier of calibrated alpha\n'
        'avg_squares_changed   = mean number of board squares (out of 64) whose probe prediction changed\n'
        'avg_collateral_squares = same but excluding the target square (unintended side effects)\n'
        'target_flip_rate       = fraction of positions where the target square probe actually flipped'
    ),
)

# Table 3: probability shift to newly-legal / newly-illegal moves at each alpha
rows3 = []
for mult in ALPHA_MULTS:
    for sq_key, entries in sorted(buckets[mult].items()):
        sq_label, cat, sub = sq_key
        rows3.append([
            sq_label, cat, sub, f'{mult}x', len(entries),
            mean_field(entries, 'P_before_newly_legal'),
            mean_field(entries, 'P_after_newly_legal'),
            mean_field(entries, 'abs_dP_newly_legal'),
            mean_field(entries, 'P_before_newly_illegal'),
            mean_field(entries, 'P_after_newly_illegal'),
            mean_field(entries, 'abs_dP_newly_illegal'),
        ])

iox.save_table(
    rows3,
    filename=f'alpha_sweep_prob_shift{args.out_suffix}.txt',
    headers=[
        'square', 'category', 'sub_condition', 'alpha_mult', 'n',
        'P_before_newly_legal', 'P_after_newly_legal', 'dP_newly_legal',
        'P_before_newly_illegal', 'P_after_newly_illegal', 'dP_newly_illegal',
    ],
    title=(
        'Alpha sweep — probability shift to newly-legal / newly-illegal moves\n'
        'P_before / P_after = total probability mass on those moves before/after intervention\n'
        'dP = P_after - P_before  (positive = model gives more mass to those moves)'
    ),
)

print('done. results saved to Intervention Results/')
