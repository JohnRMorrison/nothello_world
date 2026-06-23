"""Per-hidden-node heuristic analysis for a parity-split pattern-detector MLP.

For each hidden node j in each parity MLP, compute structural fields suitable
for downstream interpretation (e.g. by an LLM):

  signature           Input-side: 6 subentries describing robust sufficient and
                       silencing sets of input cells.
    excitatory_sufficient(+core,+aux)
    inhibitory_sufficient(+core,+aux)
    bipolar_cells

  classification      Interpretability tier (clean-single, clean-bipolar,
                      clean-pattern-*, distributed, etc.)

  outputs.excites     Top output patterns by positive W2 weight, with metadata.
  outputs.inhibits    Same for negative W2.

  alone_fires         Patterns this node alone fires (at h_j_min, others silent).

The script intentionally does NOT produce a natural-language description of
each node -- semantic glosses ("flanking-prefix fragment", etc.) require
domain knowledge and are best left to an LLM consuming the JSON.

Usage:
    python node_heuristics.py --ckpt PATH [--out PATH.json]
"""
import argparse
import json
from collections import Counter

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants and utility
# ---------------------------------------------------------------------------

STATE_NAMES = ['empty', 'black', 'white']  # 0=empty (not played), 1=black, 2=white
N_CELLS = 60


def cell_name(c):
    """Convert cell index 0..59 to chess-like label (A1..H8 skipping centers)."""
    return f"{chr(65 + c // 8)}{c % 8 + 1}"


# ---------------------------------------------------------------------------
# Spatial pattern detection
# ---------------------------------------------------------------------------

def detect_pattern(cell_indices):
    """Classify the spatial arrangement of a small set of cells.

    Returns one of:
      'empty', 'single',
      'horizontal' (all in same row),
      'vertical' (all in same column),
      'diagonal' (all on same diagonal),
      'tight-cluster' (max Chebyshev distance from centroid <= 1.5),
      'loose-cluster' (<= 2.5),
      'scattered' (anything else).
    """
    if len(cell_indices) == 0:
        return 'empty'
    if len(cell_indices) == 1:
        return 'single'
    rows = np.array([c // 8 for c in cell_indices])
    cols = np.array([c % 8 for c in cell_indices])
    if len(set(rows)) == 1:
        return 'horizontal'
    if len(set(cols)) == 1:
        return 'vertical'
    if len(set(rows - cols)) == 1 or len(set(rows + cols)) == 1:
        return 'diagonal'
    cr, cc = rows.mean(), cols.mean()
    max_dist = max(max(abs(r - cr), abs(c - cc)) for r, c in zip(rows, cols))
    if max_dist <= 1.5:
        return 'tight-cluster'
    if max_dist <= 2.5:
        return 'loose-cluster'
    return 'scattered'


# ---------------------------------------------------------------------------
# Per-node signature computation
# ---------------------------------------------------------------------------

def compute_node_signature(W1, b1, j, features='playedeven'):
    """Compute the 6-category signature for hidden node j.

    Each cell has 3 states (empty/black/white). For robust firing, choose
    cells whose net_gain sum > -min_act (where net_gain = best - worst).
    For robust silencing, sum > max_act.

    Returns dict with: bias, max_activation, min_activation, fires_by_default,
    excitatory_sufficient(+core,+aux), inhibitory_sufficient(+core,+aux),
    bipolar_cells.
    """
    if features != 'playedeven':
        raise NotImplementedError(f"features={features} not yet implemented")

    W_b = W1[j, :60]                 # contribution if cell c is BLACK
    W_w = W1[j, :60] + W1[j, 60:]    # contribution if cell c is WHITE
    states = np.stack([np.zeros(60), W_b, W_w], axis=-1)  # (60, 3)
    best = states.max(axis=-1)        # per-cell max contribution
    worst = states.min(axis=-1)       # per-cell min contribution
    net_gain = best - worst           # always >= 0
    best_state = states.argmax(axis=-1)
    worst_state = states.argmin(axis=-1)

    total_best = best.sum()
    total_worst = worst.sum()
    max_act = float(total_best + b1[j])
    min_act = float(total_worst + b1[j])

    def find_pool(target):
        """Find Core and Auxiliary cells for a robust sufficient set.

        Core = cells in every minimal robust sufficient set of size K_min.
        Aux  = cells in some but not all such sets, OR cells beyond K_min that
               could substitute for an interchangeable Core cell.
        """
        if target <= 0:
            return [], []  # trivially achievable with empty set
        order = np.argsort(net_gain)[::-1]
        cum = np.cumsum(net_gain[order])
        w = np.where(cum > target)[0]
        if len(w) == 0:
            return [], []  # not achievable
        K = int(w[0]) + 1
        slack = cum[K - 1] - target
        g_next = net_gain[order[K]] if K < 60 else 0.0
        # Core: top-K cells whose net_gain > g_next + slack (their absence
        # cannot be compensated by the next-best cell).
        core_cells = [int(c) for c in order[:K] if net_gain[c] > g_next + slack]
        # Aux: top-K cells not in Core (interchangeable with each other),
        # plus cells beyond top-K that COULD substitute.
        aux_cells = [int(c) for c in order[:K] if c not in core_cells]
        # Threshold for being a viable substitute: net_gain > bottom-of-K - slack
        threshold = net_gain[order[K - 1]] - slack
        for c in order[K:]:
            if net_gain[c] > threshold:
                aux_cells.append(int(c))
            else:
                break  # sorted descending, can stop
        return core_cells, aux_cells

    ex_core, ex_aux = ([], [])
    inh_core, inh_aux = ([], [])

    # Excitatory: target is -min_act (overcome worst-case inhibition)
    if max_act > 0:
        ex_core, ex_aux = find_pool(-min_act)
    # Inhibitory: target is max_act (overcome best-case excitation)
    if min_act <= 0:
        inh_core, inh_aux = find_pool(max_act)

    # Bipolar cells: appear in both Ex (Core or Aux) and Inh (Core or Aux)
    # AND their best_state != worst_state (so the role really differs by state)
    ex_pool = set(ex_core) | set(ex_aux)
    inh_pool = set(inh_core) | set(inh_aux)
    bipolar = []
    for c in (ex_pool & inh_pool):
        if best_state[c] != worst_state[c]:
            bipolar.append((int(c), int(best_state[c]), int(worst_state[c])))

    def annotate(cells, state_array):
        return [{
            'cell': int(c),
            'cell_name': cell_name(int(c)),
            'state': STATE_NAMES[int(state_array[c])],
            'net_gain': float(net_gain[c]),
        } for c in cells]

    return {
        'bias': float(b1[j]),
        'max_activation': max_act,
        'min_activation': min_act,
        'fires_by_default': float(b1[j]) > 0,
        'excitatory_sufficient': annotate(sorted(ex_pool, key=lambda c: -net_gain[c]),
                                          best_state),
        'excitatory_sufficient_core': annotate(ex_core, best_state),
        'excitatory_sufficient_aux': annotate(ex_aux, best_state),
        'inhibitory_sufficient': annotate(sorted(inh_pool, key=lambda c: -net_gain[c]),
                                          worst_state),
        'inhibitory_sufficient_core': annotate(inh_core, worst_state),
        'inhibitory_sufficient_aux': annotate(inh_aux, worst_state),
        'bipolar_cells': [{
            'cell': c, 'cell_name': cell_name(c),
            'excited_state': STATE_NAMES[ex_s],
            'silenced_state': STATE_NAMES[inh_s],
        } for (c, ex_s, inh_s) in bipolar],
    }


# ---------------------------------------------------------------------------
# h_j_min: lower bound on hidden activation when j fires at its minimal set
# ---------------------------------------------------------------------------

def compute_h_lower_bound(W1, b1):
    """For each hidden node j, return h_j at its minimal robust sufficient input
    set (the smallest "guaranteed" activation when j fires).

    For positive-bias nodes (fire by default): h_j_min = ReLU(b1[j]).
    For negative-bias nodes: compute activation at the minimum K_min set where
    the chosen cells are in their best state and all others in their worst.
    Returns 0 for nodes that never fire robustly.
    """
    H = W1.shape[0]
    h_lower = np.zeros(H, dtype=np.float32)
    W_b = W1[:, :60]
    W_w = W1[:, :60] + W1[:, 60:]
    states = np.stack([np.zeros_like(W_b), W_b, W_w], axis=-1)
    best = states.max(axis=-1)
    worst = states.min(axis=-1)
    net_gain = best - worst
    for j in range(H):
        max_act = best[j].sum() + b1[j]
        min_act = worst[j].sum() + b1[j]
        if b1[j] > 0:
            h_lower[j] = max(0.0, float(b1[j]))
            continue
        if max_act <= 0:
            continue
        target = -min_act
        if target <= 0:
            h_lower[j] = max(0.0, float(b1[j]))
            continue
        order = np.argsort(net_gain[j])[::-1]
        cum = np.cumsum(net_gain[j][order])
        w = np.where(cum > target)[0]
        if len(w) == 0:
            continue
        K = int(w[0]) + 1
        in_S = set(order[:K].tolist())
        h_at_K = (sum(best[j, c] for c in in_S) +
                  sum(worst[j, c] for c in range(60) if c not in in_S) +
                  b1[j])
        h_lower[j] = max(0.0, float(h_at_K))
    return h_lower


# ---------------------------------------------------------------------------
# Output side: list patterns the node excites or inhibits
# ---------------------------------------------------------------------------


def compute_node_alone_fires(W2, b2, j, h_j_min, patterns=None):
    """List output patterns that hidden node j ALONE fires (when all other
    hidden nodes are silent) given j is at its minimum sufficient activation.

    Vote condition:  W2[p, j] * h_j_min + b2[p] > 0  with all h_k = 0 for k != j.

    Splits into two lists:
      alone_fires_nontrivial : patterns where b2[p] <= 0, so firing requires
                                j's contribution -- the meaningful "votes".
      alone_fires_trivial    : patterns where b2[p] > 0 (fire by default);
                                j's vote isn't needed but doesn't hurt.

    Each entry is sorted by activation margin (most decisive first).
    """
    w_col = W2[:, j]
    activations = w_col * h_j_min + b2  # (P,) final activation if j alone fires
    fires = activations > 0
    nontrivial_mask = fires & (b2 <= 0)
    trivial_mask = fires & (b2 > 0)

    def annotate(p_idx):
        d = {
            'pattern_idx': int(p_idx),
            'W2': float(w_col[p_idx]),
            'b2': float(b2[p_idx]),
            'activation_margin': float(activations[p_idx]),
        }
        if patterns is not None:
            p = patterns[p_idx]
            d['target'] = cell_name(p['target'])
            d['terminal'] = cell_name(p['terminal'])
            d['opponents'] = [cell_name(c) for c in p['opponents']]
            d['length'] = p['length']
            d['direction'] = p['direction']
        return d

    nontrivial = [annotate(p) for p in np.where(nontrivial_mask)[0]]
    nontrivial.sort(key=lambda d: -d['activation_margin'])
    trivial_count = int(trivial_mask.sum())
    return {
        'h_j_min': float(h_j_min),
        'alone_fires_nontrivial': nontrivial,
        'alone_fires_nontrivial_count': len(nontrivial),
        'alone_fires_trivial_count': trivial_count,
    }




def compute_node_outputs(W2, b2, j, patterns=None, top_k=10,
                         abs_threshold=None, rel_threshold=0.5):
    """For hidden node j, list the patterns it excites (W2 > 0) and
    inhibits (W2 < 0).

    A pattern is included if EITHER:
      - it is in the top `top_k` patterns for j by |W2[p, j]|, OR
      - |W2[p, j]| > rel_threshold * max_|W2[:, j]|, OR
      - |W2[p, j]| > abs_threshold (if abs_threshold is given)

    The criteria are explicit and configurable; the output is just the list
    of (pattern_idx, W2 value, pattern metadata) -- no interpretation.

    Returns dict with 'excites' and 'inhibits' lists.
    """
    w_col = W2[:, j]  # weights from j to each output pattern
    abs_w = np.abs(w_col)
    max_abs = float(abs_w.max())
    rel_cut = rel_threshold * max_abs

    # Precompute rank-by-magnitude once (O(P log P) total instead of per-pattern)
    sorted_desc = np.argsort(abs_w)[::-1]
    rank = np.empty_like(sorted_desc)
    rank[sorted_desc] = np.arange(len(sorted_desc))

    # Indices that pass any criterion
    in_topk = set(sorted_desc[:top_k].tolist())
    in_rel = set(np.where(abs_w > rel_cut)[0].tolist())
    if abs_threshold is not None:
        in_abs = set(np.where(abs_w > abs_threshold)[0].tolist())
    else:
        in_abs = set()
    picked = sorted(in_topk | in_rel | in_abs, key=lambda p: -abs_w[p])

    def annotate(p_idx):
        d = {
            'pattern_idx': int(p_idx),
            'W2': float(w_col[p_idx]),
            'abs_W2_rank': int(rank[p_idx]) + 1,
        }
        if patterns is not None:
            p = patterns[p_idx]
            d['target'] = cell_name(p['target'])
            d['terminal'] = cell_name(p['terminal'])
            d['opponents'] = [cell_name(c) for c in p['opponents']]
            d['length'] = p['length']
            d['direction'] = p['direction']
        return d

    excites = [annotate(p) for p in picked if w_col[p] > 0]
    inhibits = [annotate(p) for p in picked if w_col[p] < 0]
    return {
        'excites': excites,
        'inhibits': inhibits,
        'max_abs_W2': max_abs,
        'b2_irrelevant_for_node_j': True,  # b2 is per-output, not per-node
    }


# ---------------------------------------------------------------------------
# Classification criteria (explicit and documented)
# ---------------------------------------------------------------------------

def classify_node(sig):
    """Classify a node by interpretability using explicit criteria.

    Categories, checked in order (first match wins):

    1. clean-single:
         Exactly ONE cell in Ex MS Core (or Inh MS Core if Ex is empty).
         The heuristic is "fires when cell X is in state Y" (or its dual).

    2. clean-bipolar:
         2-6 bipolar cells (same cell, opposite state in Ex and Inh)
         AND those cells form a spatial pattern (not 'scattered').
         The heuristic is "this cell-region's COLOR determines fire vs
         silence" — the cells are the feature, the state is the polarity.

    3. clean-pattern-excitatory:
         Ex MS Core has 2-5 cells forming a spatial pattern (line, diagonal,
         cluster). No bipolar story.

    4. clean-pattern-inhibitory:
         Inh MS Core has 2-5 cells forming a spatial pattern.

    5. bipolar-small:
         <=5 bipolar cells but no clean spatial pattern.

    6. spatial-with-aux:
         Ex MS Core + Aux pool (combined) has <=8 cells forming a pattern
         where Core alone didn't qualify.

    7. distributed:
         None of the above. Genuinely complex per-node structure.
    """
    ex_core = sig['excitatory_sufficient_core']
    ex_aux = sig['excitatory_sufficient_aux']
    inh_core = sig['inhibitory_sufficient_core']
    bipolar = sig['bipolar_cells']

    ex_core_cells = [d['cell'] for d in ex_core]
    inh_core_cells = [d['cell'] for d in inh_core]
    bipolar_cells = [d['cell'] for d in bipolar]

    # 1. Clean single
    if len(ex_core) == 1:
        return 'clean-single', ex_core_cells, 'excitatory'
    if not ex_core and len(inh_core) == 1:
        return 'clean-single', inh_core_cells, 'inhibitory'

    # 2. Clean bipolar (2-6 cells, spatial pattern)
    if 2 <= len(bipolar) <= 6:
        pat = detect_pattern(bipolar_cells)
        if pat not in ('scattered', 'empty'):
            return 'clean-bipolar', bipolar_cells, pat

    # 3. Clean pattern excitatory
    if 2 <= len(ex_core) <= 5:
        pat = detect_pattern(ex_core_cells)
        if pat not in ('scattered', 'empty'):
            return 'clean-pattern-excitatory', ex_core_cells, pat

    # 4. Clean pattern inhibitory
    if 2 <= len(inh_core) <= 5:
        pat = detect_pattern(inh_core_cells)
        if pat not in ('scattered', 'empty'):
            return 'clean-pattern-inhibitory', inh_core_cells, pat

    # 5. Bipolar small but scattered
    if 2 <= len(bipolar) <= 5:
        return 'bipolar-small', bipolar_cells, 'scattered'

    # 6. Spatial with aux help
    if ex_core:
        with_aux = ex_core_cells + [d['cell'] for d in ex_aux][:5]
        if 3 <= len(with_aux) <= 8:
            pat = detect_pattern(with_aux)
            if pat not in ('scattered', 'empty'):
                return 'spatial-with-aux', with_aux, pat

    return 'distributed', [], ''


# ---------------------------------------------------------------------------
# Network-level analysis
# ---------------------------------------------------------------------------

def analyze_network(ckpt_path, features='playedeven', output_top_k=10,
                    output_rel_threshold=0.5):
    """Process all hidden nodes in both parity MLPs, return a list of dicts.

    Includes both the input-side signature (with classification) AND the
    output-side excite/inhibit lists per hidden node.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    try:
        from hand_crafted_flanking import enumerate_flanking_patterns
        patterns = enumerate_flanking_patterns()
    except Exception:
        patterns = None
    results = []
    for parity in ('even', 'odd'):
        if parity not in ckpt:
            continue
        sd = ckpt[parity]
        W1 = sd['net.0.weight'].numpy()
        b1 = sd['net.0.bias'].numpy()
        W2 = sd['net.2.weight'].numpy()
        b2 = sd['net.2.bias'].numpy()
        h_lower = compute_h_lower_bound(W1, b1)
        for j in range(W1.shape[0]):
            sig = compute_node_signature(W1, b1, j, features=features)
            cls = classify_node(sig)
            outputs = compute_node_outputs(W2, b2, j, patterns=patterns,
                                           top_k=output_top_k,
                                           rel_threshold=output_rel_threshold)
            alone_fires = compute_node_alone_fires(W2, b2, j, h_lower[j],
                                                   patterns=patterns)
            results.append({
                'parity': parity,
                'node': j,
                'signature': sig,
                'classification': cls[0],
                'classification_cells': [cell_name(c) for c in cls[1]],
                'pattern_or_direction': cls[2],
                'outputs': outputs,
                'alone_fires': alone_fires,
            })
    return results


def summarize(results):
    counts = Counter(r['classification'] for r in results)
    print(f"\nClassification summary across {len(results)} nodes:\n")
    interpretable = 0
    for cls in ('clean-single', 'clean-bipolar', 'clean-pattern-excitatory',
                'clean-pattern-inhibitory', 'bipolar-small', 'spatial-with-aux',
                'distributed'):
        n = counts.get(cls, 0)
        if cls != 'distributed':
            interpretable += n
        print(f"  {cls:>30}: {n:>4} ({100 * n / len(results):5.1f}%)")
    print(f"\n  {'INTERPRETABLE TOTAL':>30}: {interpretable:>4} "
          f"({100 * interpretable / len(results):5.1f}%)")


# ---------------------------------------------------------------------------
# Stdout formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pattern(e, value_field, value_label):
    """Format one pattern entry as 'pat N(target→terminal LN) field=±X.XX'."""
    geo = (f"({e['target']}→{e['terminal']} L{e['length']})"
           if 'target' in e else "")
    return f"pat {e['pattern_idx']}{geo} {value_label}={e[value_field]:+.2f}"


def show_node(r):
    """Print a compact summary for one node from structured fields only."""
    sig = r['signature']
    default = 'fires by default' if sig['fires_by_default'] else 'silent by default'
    cells = ", ".join(r['classification_cells'])
    print(f"  Node {r['parity']}-{r['node']}: {default} (bias={sig['bias']:+.2f}) "
          f"[{r['classification']} / {r['pattern_or_direction']}]"
          + (f" cells=[{cells}]" if cells else ""))

    outs = r['outputs']
    for label, key in [('EXCITES', 'excites'), ('INHIBITS', 'inhibits')]:
        items = outs[key][:5]
        if items:
            print(f"    {label}: " + ", ".join(_fmt_pattern(e, 'W2', 'W2') for e in items))

    af = r['alone_fires']
    n_nt = af['alone_fires_nontrivial_count']
    n_t = af['alone_fires_trivial_count']
    print(f"    ALONE-FIRES (h_j_min={af['h_j_min']:.2f}): "
          f"{n_nt} non-trivial" + (f", {n_t} trivial" if n_t else ""))
    top_af = af['alone_fires_nontrivial'][:5]
    if top_af:
        print("      top: " + ", ".join(
            _fmt_pattern(e, 'activation_margin', 'margin') for e in top_af))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='Path to MLP checkpoint .pt')
    ap.add_argument('--features', default='playedeven',
                    help='Feature encoding (only playedeven supported for now)')
    ap.add_argument('--out', default=None,
                    help='Output JSON path. Default: stdout summary only.')
    ap.add_argument('--show', type=int, default=10,
                    help='Show N example descriptions on stdout')
    ap.add_argument('--output-top-k', type=int, default=10,
                    help='Top-K output patterns per node by |W2|')
    ap.add_argument('--output-rel-threshold', type=float, default=0.5,
                    help='Output patterns with |W2| > rel_threshold * max |W2| per node')
    args = ap.parse_args()

    results = analyze_network(args.ckpt, features=args.features,
                              output_top_k=args.output_top_k,
                              output_rel_threshold=args.output_rel_threshold)
    summarize(results)

    if args.show > 0:
        print(f"\nFirst {args.show} node summaries:\n")
        for r in results[:args.show]:
            show_node(r)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {len(results)} node records to {args.out}")


if __name__ == '__main__':
    main()
