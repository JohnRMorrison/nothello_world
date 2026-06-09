"""Forward passes, per-cell binary-search calibration, single-cell and
multi-cell intervention application.

All layer numbers come in as arguments — nothing is hardcoded outside config.

Convention recap:
  `intervention_layer` = mingpt block index whose resid_post we modify
  `probe_layer`        = mingpt block index whose resid_post the probe reads
  `cal_depth`          = probe_layer - intervention_layer

`compute_prefix(model, tokens, intervention_layer + 1)` runs blocks[:N+1]
and produces resid_post of block N. To then forward to the output we run
`forward_from_prefix(model, x, intervention_layer + 1)` which runs the
remaining blocks + ln_f + head.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from mingpt.model import GPT, GPTConfig  # noqa: E402

from . import config, probes


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str = None, device: str = "cpu",
               n_layer: int = 8, n_head: int = 8, n_embd: int = 512):
    """Load the mingpt OthelloGPT model from a checkpoint."""
    ckpt_path = ckpt_path or config.CKPT_PATH
    cfg = GPTConfig(vocab_size=61, block_size=59,
                    n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    model = GPT(cfg)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Partial forward passes
# ---------------------------------------------------------------------------
def compute_prefix(model, input_tokens, layer_intervene: int) -> torch.Tensor:
    """Run embedding + blocks[:layer_intervene]. Returns the residual at the
    intervention point with shape (1, T, d_model)."""
    b, t = input_tokens.size()
    tok = model.tok_emb(input_tokens)
    pos = model.pos_emb[:, :t, :]
    x = model.drop(tok + pos)
    for block in model.blocks[:layer_intervene]:
        x = block(x)
    return x


def forward_from_prefix(model, x: torch.Tensor, layer_intervene: int) -> torch.Tensor:
    """Continue forward from `x` (= resid at block `layer_intervene - 1`)
    through remaining blocks + ln_f + head. Returns logits (1, T, vocab)."""
    for block in model.blocks[layer_intervene:]:
        x = block(x)
    x = model.ln_f(x)
    return model.head(x)


def forward_from_prefix_capturing(
    model, x: torch.Tensor, layer_intervene: int, capture_layer: int,
):
    """Like forward_from_prefix but also captures resid_post of mingpt block
    `capture_layer`. Returns (logits, resid_at_capture_layer).

    Edge case: when `capture_layer == layer_intervene - 1`, the input `x`
    already IS resid_post of the capture layer — we clone it before any
    further blocks run. This is the cd=0 setting.
    """
    resid_at_capture = None
    if capture_layer == layer_intervene - 1:
        resid_at_capture = x.clone()
    for i, block in enumerate(model.blocks[layer_intervene:],
                              start=layer_intervene):
        x = block(x)
        if i == capture_layer:
            resid_at_capture = x.clone()
    x = model.ln_f(x)
    logits = model.head(x)
    return logits, resid_at_capture


# ---------------------------------------------------------------------------
# Calibration: binary-search the minimum alpha that flips the probe argmax
# ---------------------------------------------------------------------------
def calibrate_alpha(
    model,
    prefix_acts: torch.Tensor,
    pos: int,
    probe_cell_W: torch.Tensor,
    current_class: int,
    target_class: int,
    intervene_layer: int,
    probe_layer: int,
    *,
    alpha_cap: float = 10.0,
    n_iters: int = 20,
    safety_margin: float = 1.1,
) -> float:
    """Find the minimum alpha in [0, alpha_cap] that, after intervention at
    `intervene_layer` along direction `d = probe_cell_W[:, target] -
    probe_cell_W[:, current]`, makes the probe argmax = target_class at
    `probe_layer`.

    Returns the bound times `safety_margin`, clamped to alpha_cap.

    Works for any cal_depth (= probe_layer - intervene_layer) >= 0. cd=0 short-
    circuits the forward propagation: the probe read is computed directly on
    the modified prefix activation.
    """
    model_device = next(model.parameters()).device
    if prefix_acts.device != model_device:
        prefix_acts = prefix_acts.to(model_device)
    if probe_cell_W.device != model_device:
        probe_cell_W = probe_cell_W.to(model_device)
    h = prefix_acts[0, pos].detach()
    flip_dir = probe_cell_W[:, target_class] - probe_cell_W[:, current_class]
    d_hat = flip_dir / flip_dir.norm()
    coeff = (h @ d_hat).item()

    cal_depth = probe_layer - intervene_layer
    blocks_to_run = model.blocks[intervene_layer + 1: probe_layer + 1]
    # When cal_depth == 0, blocks_to_run is empty -> no forward propagation.

    def probe_argmax_at_scale(s: float) -> int:
        h_mod = h - s * coeff * d_hat
        if cal_depth == 0:
            logits = probe_cell_W.T @ h_mod   # (3,)
        else:
            x = prefix_acts.clone()
            x[0, pos] = h_mod
            with torch.no_grad():
                for block in blocks_to_run:
                    x = block(x)
                act = x[0, pos]
            logits = probe_cell_W.T @ act
        return int(logits.argmax().item())

    # Triviality / feasibility checks.
    if probe_argmax_at_scale(0.0) == target_class:
        return 0.5  # already at target; small nudge for downstream propagation
    if probe_argmax_at_scale(alpha_cap) != target_class:
        return alpha_cap  # cannot flip within budget

    lo, hi = 0.0, alpha_cap
    for _ in range(n_iters):
        mid = (lo + hi) / 2.0
        if probe_argmax_at_scale(mid) == target_class:
            hi = mid
        else:
            lo = mid
    return min(hi * safety_margin, alpha_cap)


# ---------------------------------------------------------------------------
# Single-cell intervention apply
# ---------------------------------------------------------------------------
@dataclass
class InterventionSpec:
    """All info needed to apply one single-cell intervention on a prefix."""
    cell: tuple[int, int]               # (r, c)
    current_class: int
    target_class: int
    flip_dir: torch.Tensor               # raw direction (d_model,)
    d_hat: torch.Tensor                  # unit direction
    coeff: float                         # h @ d_hat at the intervention point

    @classmethod
    def from_probe(cls, probe: torch.Tensor, mode: int,
                   r: int, c: int,
                   current_class: int, target_class: int,
                   h_at_intervention: torch.Tensor):
        flip_dir = (probe[mode, :, r, c, target_class]
                    - probe[mode, :, r, c, current_class])
        d_hat = flip_dir / flip_dir.norm()
        # Align h to d_hat's device for the coefficient computation.
        h_dev = h_at_intervention.to(d_hat.device)
        coeff = (h_dev @ d_hat).item()
        return cls((r, c), current_class, target_class, flip_dir.detach(),
                   d_hat.detach(), coeff)

    def to(self, device):
        """Return a copy with `flip_dir` and `d_hat` on `device`."""
        if self.flip_dir.device == torch.device(device):
            return self
        return InterventionSpec(
            self.cell, self.current_class, self.target_class,
            self.flip_dir.to(device), self.d_hat.to(device), self.coeff,
        )


def apply_intervention(
    prefix_acts: torch.Tensor,
    pos: int,
    specs: list,
    alphas: list,
):
    """Apply one or more interventions to a copy of `prefix_acts` at sequence
    position `pos`. Returns the modified prefix (not in place).

    `specs` and `alphas` must be same length. Specs are moved to
    `prefix_acts.device` as needed so callers can keep them on CPU.
    """
    device = prefix_acts.device
    x = prefix_acts.clone()
    h = x[0, pos]
    for spec, alpha in zip(specs, alphas):
        # Use the spec's coefficient relative to the ORIGINAL prefix (so
        # multiple interventions stack via independent projections), matching
        # multi_intervention.py's convention.
        s = spec.to(device)
        h = h - alpha * s.coeff * s.d_hat
    x[0, pos] = h
    return x


# ---------------------------------------------------------------------------
# High-level: run with intervention and return logits + (optional) residual
# ---------------------------------------------------------------------------
def run_with_intervention(
    model,
    prefix_acts: torch.Tensor,
    pos: int,
    specs: list,
    alphas: list,
    intervene_layer: int,
    capture_layer: int = None,
):
    """Apply interventions to `prefix_acts` then run through the rest of the
    model. If `capture_layer` is given, also returns the residual at that
    block's resid_post.

    `prefix_acts` can be on any device; it's moved to the model's device for
    the forward pass and the returned tensors stay on that device. (Callers
    are expected to .cpu() the results if storing many of them.)

    Returns:
      logits (1, T, vocab)
      resid_at_capture (1, T, d_model) or None
    """
    model_device = next(model.parameters()).device
    if prefix_acts.device != model_device:
        prefix_acts = prefix_acts.to(model_device)
    with torch.no_grad():
        x = apply_intervention(prefix_acts, pos, specs, alphas)
        if capture_layer is None:
            return forward_from_prefix(model, x, intervene_layer + 1), None
        return forward_from_prefix_capturing(
            model, x, intervene_layer + 1, capture_layer,
        )


# ---------------------------------------------------------------------------
# Cache builder for a PositionRecord
# ---------------------------------------------------------------------------
def cache_clean_state(
    model,
    record,           # data.PositionRecord
    probe: torch.Tensor,
    *,
    intervention_layer: int,
    probe_layer: int,
    probe_mode: int = 0,
    device: str = "cpu",
):
    """Populate `record.extras` with:
      input_tokens, prefix_acts (at intervention_layer), clean_logits_last,
      clean_resid_at_probe, intervention_direction (spec), current_class,
      target_class.

    Idempotent; safe to call multiple times.
    """
    if record.extras.get("cached_for") == (intervention_layer, probe_layer,
                                            probe_mode):
        return record

    # Build the model input from the move history.
    move_history = record.move_history
    # Map flat cell indices 0..63 to mingpt vocab indices 1..60 (0 = pass).
    # The board_seqs_int file already has this mapping, but here we have only
    # move history. Build the mapping on the fly.
    stoi = {c: i + 1 for i, c in enumerate(config.STOI_INDICES)}
    token_ids = torch.tensor(
        [[stoi[m] for m in move_history]], dtype=torch.long, device=device,
    )
    pos = record.position

    with torch.no_grad():
        prefix_acts = compute_prefix(model, token_ids, intervention_layer + 1)
        # Clean forward + capture probe-layer resid.
        clean_logits, clean_resid_at_probe = forward_from_prefix_capturing(
            model, prefix_acts.clone(), intervention_layer + 1, probe_layer,
        )
        clean_logits_last = clean_logits[0, -1].detach().cpu()
        clean_resid_at_probe = clean_resid_at_probe.detach()

    # Build the intervention direction (no alpha yet).
    r, c = record.square
    cur_cls = probes.board_val_to_probe_class(
        record.orig_val, record.next_color, probe_mode)
    tgt_cls = probes.board_val_to_probe_class(
        record.target_val, record.next_color, probe_mode)
    spec = InterventionSpec.from_probe(
        probe, probe_mode, r, c, cur_cls, tgt_cls,
        h_at_intervention=prefix_acts[0, pos],
    )

    # Store cached tensors on CPU to avoid OOM at 36k records. Forward
    # passes will move them to the model's device on demand.
    record.extras.update({
        "cached_for": (intervention_layer, probe_layer, probe_mode),
        "input_tokens": token_ids.cpu(),
        "prefix_acts": prefix_acts.cpu(),
        "clean_logits_last": clean_logits_last,
        "clean_resid_at_probe": clean_resid_at_probe.cpu(),
        "current_class": cur_cls,
        "target_class": tgt_cls,
        "spec": spec.to("cpu"),
    })
    return record
