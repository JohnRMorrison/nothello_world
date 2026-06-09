"""All metrics: Li top-N, mass-shift metrics (5 per intervention), the
redistribution metric, log-probability distance distributions, and crosstalk
counts on the probe-layer residual.

Convention: model output `logits` has shape (vocab=61,). Index 0 = pass,
indices 1..60 = the 60 valid cells in `STOI_INDICES` order. We always strip
the pass token before computing softmax over cells.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from . import config


# ---------------------------------------------------------------------------
# Cell probabilities from logits
# ---------------------------------------------------------------------------
def cell_logits(logits_last: torch.Tensor) -> torch.Tensor:
    """`logits_last` is (vocab=61,). Returns the 60 valid-cell logits."""
    return logits_last[1:61]


def cell_probs(logits_last: torch.Tensor) -> torch.Tensor:
    return torch.softmax(cell_logits(logits_last), dim=0)


def cell_logprobs(logits_last: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(cell_logits(logits_last), dim=0)


def mass_on(probs60: torch.Tensor, cell_set) -> float:
    """Sum of probabilities over the cells in `cell_set` that are valid."""
    total = 0.0
    for c in cell_set:
        idx = config.STOI_INDEX_OF.get(c)
        if idx is not None:
            total += probs60[idx].item()
    return total


# ---------------------------------------------------------------------------
# Li et al. top-N accuracy
# ---------------------------------------------------------------------------
def li_topn_accuracy(logits_last: torch.Tensor, legal_set) -> float | None:
    """Top-N accuracy where N = |legal_set ∩ STOI|. Errors = false positives
    in top-N + false negatives missed from top-N; accuracy = 1 - errors/(2N).

    Returns None if N == 0."""
    legal_stoi = set(legal_set) & set(config.STOI_INDICES)
    n = len(legal_stoi)
    if n == 0:
        return None
    cls = cell_logits(logits_last)
    topn_idx = cls.argsort(descending=True)[:n]
    topn_cells = {config.STOI_INDICES[int(i)] for i in topn_idx}
    fp = topn_cells - legal_stoi
    fn = legal_stoi - topn_cells
    return 1.0 - (len(fp) + len(fn)) / (2.0 * n)


# ---------------------------------------------------------------------------
# Mass-shift metrics (the 5 the notebook needs, plus redistribution)
# ---------------------------------------------------------------------------
def mass_shift_metrics_from_probs(
    probs_before: torch.Tensor,
    probs_after: torch.Tensor,
    legal_orig,
    legal_cf,
) -> dict:
    """Mass-shift metrics computed from pre-normalized 60-d cell probabilities.

    Use for the MLP pipeline where `probs` are noisy-or aggregates, not a
    softmax distribution. `probs_before` and `probs_after` must already be in
    a comparable normalization (e.g. both renormalized to sum to 1, or both
    raw noisy-or — caller's choice).
    """
    legal_cf = set(legal_cf)
    legal_orig = set(legal_orig)
    newly_legal = legal_cf - legal_orig
    newly_illegal = legal_orig - legal_cf

    pb_legal = mass_on(probs_before, legal_cf)
    pa_legal = mass_on(probs_after, legal_cf)
    pb_nl = mass_on(probs_before, newly_legal)
    pa_nl = mass_on(probs_after, newly_legal)
    pb_ni = mass_on(probs_before, newly_illegal)
    pa_ni = mass_on(probs_after, newly_illegal)

    abs_legal = pa_legal - pb_legal
    abs_nl = pa_nl - pb_nl
    abs_ni = pa_ni - pb_ni

    pct_legal = (abs_legal / pa_legal) if pa_legal > 0 else float("nan")
    pct_nl = (abs_nl / pa_legal) if pa_legal > 0 else float("nan")

    added_to_nl = max(abs_nl, 0.0)
    subtracted_from_ni = max(-abs_ni, 0.0)
    redistribution = min(added_to_nl, subtracted_from_ni)

    return {
        "P_before_legal_cf": pb_legal,
        "P_after_legal_cf": pa_legal,
        "P_before_newly_legal": pb_nl,
        "P_after_newly_legal": pa_nl,
        "P_before_newly_illegal": pb_ni,
        "P_after_newly_illegal": pa_ni,
        "abs_dP_legal": abs_legal,
        "abs_dP_newly_legal": abs_nl,
        "abs_dP_newly_illegal": abs_ni,
        "pct_dP_legal": pct_legal,
        "pct_dP_newly_legal": pct_nl,
        "redistribution": redistribution,
        "n_newly_legal": len(newly_legal),
        "n_newly_illegal": len(newly_illegal),
    }


def mass_shift_metrics(
    clean_logits_last: torch.Tensor,
    intv_logits_last: torch.Tensor,
    legal_orig,
    legal_cf,
) -> dict:
    """Softmax-over-cells version (the categorical-distribution case used
    by OGPT). For MLP use `mass_shift_metrics_from_probs` directly.

    Return a dict with:

      P_before_legal_cf:       mass on counterfactual-legal cells, pre.
      P_after_legal_cf:        mass on counterfactual-legal cells, post.
      P_before_newly_legal:    mass on newly-legal cells, pre.
      P_after_newly_legal:     mass on newly-legal cells, post.
      P_before_newly_illegal:  mass on newly-illegal cells, pre.
      P_after_newly_illegal:   mass on newly-illegal cells, post.

      abs_dP_legal:            P_after_legal_cf - P_before_legal_cf.
      abs_dP_newly_legal:      P_after_newly_legal - P_before_newly_legal.
      abs_dP_newly_illegal:    P_after_newly_illegal - P_before_newly_illegal.
                               (Negative when the intervention pushed correctly.)

      pct_dP_legal:            abs_dP_legal / P_after_legal_cf (NaN if denom 0).
      pct_dP_newly_legal:      abs_dP_newly_legal / P_after_legal_cf.

      redistribution:          min(abs_dP_newly_legal,
                                    -abs_dP_newly_illegal)
                               — captures whether mass moved in the correct
                               direction, gated by the weaker arm.
    """
    pb = cell_probs(clean_logits_last)
    pa = cell_probs(intv_logits_last)
    return mass_shift_metrics_from_probs(pb, pa, legal_orig, legal_cf)


# ---------------------------------------------------------------------------
# Log-probability distance helpers
# ---------------------------------------------------------------------------
def logprob_distance_above_max_illegal(
    logits_last: torch.Tensor,
    candidate_cells,
    legal_set,
) -> list[float]:
    """For each cell in `candidate_cells`, return its log-prob minus the
    maximum log-prob over cells that are NOT in `legal_set`.

    Positive ⇒ candidate cell has higher prob than every illegal cell."""
    lp = cell_logprobs(logits_last)
    legal_stoi = set(legal_set) & set(config.STOI_INDICES)
    illegal_idxs = [i for i, c in enumerate(config.STOI_INDICES)
                    if c not in legal_stoi]
    if not illegal_idxs:
        return [float("inf") for _ in candidate_cells]
    max_illegal_lp = max(lp[i].item() for i in illegal_idxs)
    out = []
    for c in candidate_cells:
        idx = config.STOI_INDEX_OF.get(c)
        if idx is None:
            continue
        out.append(lp[idx].item() - max_illegal_lp)
    return out


def logprob_distance_below_min_legal(
    logits_last: torch.Tensor,
    candidate_cells,
    legal_set,
) -> list[float]:
    """For each cell in `candidate_cells`, return min_legal_logprob -
    cell_logprob.

    Positive ⇒ candidate cell has lower prob than every legal cell."""
    lp = cell_logprobs(logits_last)
    legal_stoi = set(legal_set) & set(config.STOI_INDICES)
    legal_idxs = [config.STOI_INDEX_OF[c] for c in legal_stoi]
    if not legal_idxs:
        return [float("-inf") for _ in candidate_cells]
    min_legal_lp = min(lp[i].item() for i in legal_idxs)
    out = []
    for c in candidate_cells:
        idx = config.STOI_INDEX_OF.get(c)
        if idx is None:
            continue
        out.append(min_legal_lp - lp[idx].item())
    return out


# ---------------------------------------------------------------------------
# Crosstalk count (cells whose probe argmax changed)
# ---------------------------------------------------------------------------
def crosstalk_count(
    clean_resid: torch.Tensor,
    intv_resid: torch.Tensor,
    probe_full_mode: torch.Tensor,
    pos: int,
    target_cells,
    *,
    exclude_center: bool = True,
) -> int:
    """Number of cells, excluding `target_cells` (and optionally center cells),
    whose probe argmax differs between clean and intervened residuals at `pos`.

    `clean_resid`, `intv_resid`: (1, T, d_model).
    `probe_full_mode`: (d_model, 8, 8, 3).
    `target_cells`: iterable of flat-cell-indices 0..63.
    """
    # Inputs may live on different devices (e.g. cached clean on CPU, intv
    # on MPS). Align everything to the probe's device.
    device = probe_full_mode.device
    h_clean = clean_resid[0, pos].to(device)
    h_intv = intv_resid[0, pos].to(device)
    logits_clean = torch.einsum("d,drco->rco", h_clean, probe_full_mode)
    logits_intv = torch.einsum("d,drco->rco", h_intv, probe_full_mode)
    pred_clean = logits_clean.argmax(-1)
    pred_intv = logits_intv.argmax(-1)
    changed = (pred_clean != pred_intv)
    target_set = set(target_cells)
    count = 0
    for r in range(8):
        for c in range(8):
            cell = r * 8 + c
            if cell in target_set:
                continue
            if exclude_center and cell in config.CENTER_CELLS:
                continue
            if bool(changed[r, c]):
                count += 1
    return count


def any_other_cell_changed(
    clean_resid: torch.Tensor,
    intv_resid: torch.Tensor,
    probe_full_mode: torch.Tensor,
    pos: int,
    target_cells,
    *,
    exclude_center: bool = True,
) -> bool:
    """Boolean version: did the intervention flip at least one non-target,
    non-center cell's probe argmax?"""
    return crosstalk_count(
        clean_resid, intv_resid, probe_full_mode, pos, target_cells,
        exclude_center=exclude_center,
    ) > 0


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------
def mean_std(values, ddof: int = 1) -> tuple[float, float]:
    """Mean and (sample) std, ignoring NaN/None. Returns (nan, 0) if empty."""
    arr = np.array([v for v in values
                    if v is not None and not (isinstance(v, float) and math.isnan(v))],
                   dtype=float)
    if len(arr) == 0:
        return float("nan"), 0.0
    if len(arr) == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=ddof))
