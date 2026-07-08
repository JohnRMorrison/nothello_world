"""1-layer pattern-detector MLP intervention pipeline.

This mirrors `src/intervention.py` for the OGPT model, but for the
single-hidden-layer MLP pattern detectors trained by `train_pattern_simple.py`.

Architectural recap (DirectMLP):

    raw 180-d features  ->  some feature view (e.g. 120-d when+even,
                            3600-d move_grid)
              |
              v
        Linear(input_dim, H)
              |
              v
            ReLU
              |
              v
        Linear(H, 960)            <- 960 flanking-pattern logits
              |
              v
        prob_or aggregation       <- 60 cell probabilities (noisy-or)

For interventions we modify the H-dim hidden along a probe-derived direction.
The probe (separately trained by `probe_pattern_models.py`) is parity-split:
one nn.Linear(H, 192) for even positions, one for odd. Class convention is
**0 = empty, 1 = mine, 2 = yours**; "mine" depends on whose turn it is via
`pos % 2`. (Different from the Nanda probe used for OGPT, which has mine and
yours swapped — see `probes.board_val_to_probe_class` for the OGPT side.)

The relevant feature constructors are vendored from `train_pattern_simple.py`
to keep this file self-contained and to avoid a hard dependency on that
script's module-level globals.

**Checkpoint availability**: the pattern-detector and probe checkpoints are
~hundreds of MB each and live on the cluster. They are NOT bundled with this
repo. Functions in this module raise a helpful error if the files are
missing — see `MLP_CKPT_PATHS` / `MLP_PROBE_PATHS` for the expected paths.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from . import config


# ---------------------------------------------------------------------------
# Expected checkpoint locations
# ---------------------------------------------------------------------------
_PAT_CKPT_DIR = os.path.join(
    _REPO_ROOT,
    "experiments/mathematical_transformation_experiments/"
    "heuristic_probe_results/pattern_detector_checkpoints",
)
MLP_CKPT_PATHS = {
    "wheneven_H512": os.path.join(
        _PAT_CKPT_DIR, "pattern_simple_direct_H512_wheneven.pt"),
    "wheneven_H1024": os.path.join(
        _PAT_CKPT_DIR, "pattern_simple_direct_H1024_wheneven.pt"),
    "movegrid_H4096": os.path.join(
        _PAT_CKPT_DIR, "pattern_simple_direct_H4096_move_grid.pt"),
    "playedeven_H512": os.path.join(
        _PAT_CKPT_DIR, "pattern_simple_direct_H512_playedeven.pt"),
    "playedeven_H4096": os.path.join(
        _PAT_CKPT_DIR, "pattern_clean_H4096.pt"),
}
MLP_PROBE_PATHS = {
    "wheneven_H512": os.path.join(
        _PAT_CKPT_DIR, "probe_direct_H512_wheneven.pt"),
    "wheneven_H1024": os.path.join(
        _PAT_CKPT_DIR, "probe_direct_H1024_wheneven.pt"),
    "movegrid_H4096": os.path.join(
        _PAT_CKPT_DIR, "probe_direct_H4096_move_grid.pt"),
    "playedeven_H512": os.path.join(
        _PAT_CKPT_DIR, "probe_direct_H512_playedeven.pt"),
    "playedeven_H4096": os.path.join(
        _PAT_CKPT_DIR, "probe_pattern_clean_H4096.pt"),
}


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class DirectMLP(nn.Module):
    """input_dim -> Linear -> ReLU -> Linear -> 960 pattern logits."""

    def __init__(self, input_dim: int, hidden_dim: int, n_patterns: int = 960):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_patterns),
        )

    def forward(self, x):
        return self.net(x)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-output-layer hidden activation (post-ReLU)."""
        return torch.relu(self.net[0](x))

    def from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """Apply only the output layer."""
        return self.net[2](h)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
N_MOVES = config.N_VALID_CELLS                       # = 60
_STOI_INDEX_OF = config.STOI_INDEX_OF
_CENTER_CELLS = config.CENTER_CELLS


def build_raw_180d_features(move_history) -> np.ndarray:
    """Build the raw 180-d features [played(60), when(60), even(60)] from
    a list of move cell indices (0..63).

    Convention (matches `mlp_intervention.build_180d_features`):
      - `played[i]` = 1 if cell-index i has been played
      - `when[i]`   = (s + 1) / 60 where s is the move-step (0-indexed) at
                      which the cell was played; 0 otherwise.
      - `even[i]`   = 1 if the move-step was even (s % 2 == 0), 0 otherwise.

    Note: in `train_pattern_simple`'s `to_move_grid_input`, step 0 is treated
    as the white-played step. This is an inverted convention from the actual
    game (where move 0 is black) but it's what the precompute and training
    code use, so we replicate it here for consistency.
    """
    played = np.zeros(N_MOVES, dtype=np.float32)
    when = np.zeros(N_MOVES, dtype=np.float32)
    even = np.zeros(N_MOVES, dtype=np.float32)
    for s, move in enumerate(move_history):
        move = int(move)
        if move not in _STOI_INDEX_OF:
            continue   # center cell, skip
        idx = _STOI_INDEX_OF[move]
        played[idx] = 1.0
        when[idx] = (s + 1) / 60.0
        even[idx] = 1.0 if (s % 2 == 0) else 0.0
    return np.concatenate([played, when, even])


def to_when_even_features(raw_180d: np.ndarray) -> np.ndarray:
    """120-d view: [when, even]. Matches feature_type='when+even'."""
    return raw_180d[60:180]


def to_played_even_features(raw_180d: np.ndarray) -> np.ndarray:
    """120-d view: [played, even]. Matches feature_type='played+even'."""
    return np.concatenate([raw_180d[:60], raw_180d[120:180]])


def to_move_grid_features(raw_180d: np.ndarray) -> np.ndarray:
    """3600-d signed move grid: grid[cell, move_num] = +1 if black, -1 if
    white, 0 otherwise. Flat-shaped (60 cells × 60 move-numbers).

    Vendored from `train_pattern_simple.to_move_grid_input`.
    """
    played = raw_180d[:60]
    when = raw_180d[60:120]
    even = raw_180d[120:180]

    move_num = np.clip((when * 60.0 - 1.0).round().astype(np.int64), 0, 59)
    color_pm = (1.0 - 2.0 * even) * played   # +1 black, -1 white, 0 unplayed

    grid = np.zeros((60, 60), dtype=np.float32)
    rows = np.arange(60)
    grid[rows, move_num] = color_pm
    return grid.reshape(60 * 60)


# Feature-type registry: maps the human-readable name to (input_dim, fn).
FEATURE_VIEWS: dict[str, tuple[int, Callable]] = {
    "when+even": (120, to_when_even_features),
    "played+even": (120, to_played_even_features),
    "move_grid": (3600, to_move_grid_features),
}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_mlp(ckpt_path: str, hidden_dim: int = None, device: str = "cpu"):
    """Load a saved DirectMLP checkpoint (even/odd state dicts).

    Returns dict with keys:
      model_even, model_odd, hidden_dim, input_dim, feature_type
    `feature_type` is inferred from input_dim if not explicit in the ckpt.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"MLP checkpoint not found: {ckpt_path}\n"
            "These checkpoints live on the cluster; copy with scp:\n"
            "  scp -P 55 jrm2182@axon-remote.rc.zi.columbia.edu:"
            "/engram/nklab/jrm2182/nothello_world/"
            "experiments/.../pattern_detector_checkpoints/"
            f"{os.path.basename(ckpt_path)} {ckpt_path}"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = int(ckpt.get("input_dim",
                             # heuristic fallback: weight shape of first Linear
                             ckpt["even"]["net.0.weight"].shape[1]))
    H = hidden_dim or int(ckpt["even"]["net.0.weight"].shape[0])
    n_patterns = int(ckpt["even"]["net.2.weight"].shape[0])

    me = DirectMLP(input_dim, H, n_patterns).to(device)
    mo = DirectMLP(input_dim, H, n_patterns).to(device)
    me.load_state_dict(ckpt["even"]); me.eval()
    mo.load_state_dict(ckpt["odd"]); mo.eval()

    # Infer feature type from input_dim (only the two we expose).
    feature_type = ckpt.get("feature_type") or {120: "when+even", 3600: "move_grid"}.get(input_dim)
    if feature_type is None:
        raise ValueError(
            f"Unrecognized MLP input_dim={input_dim}; only when+even (120) "
            f"and move_grid (3600) are supported in this module."
        )
    return {
        "model_even": me, "model_odd": mo,
        "hidden_dim": H, "input_dim": input_dim,
        "feature_type": feature_type, "n_patterns": n_patterns,
    }


def load_mlp_probe(probe_path: str, hidden_dim: int, device: str = "cpu"):
    """Load a saved Nanda-style probe on the MLP hidden state.

    Returns dict with keys: probe_even, probe_odd. Each is a (192, H) tensor
    of the linear layer weight (no bias used in direction extraction).
    """
    if not os.path.exists(probe_path):
        raise FileNotFoundError(
            f"MLP probe not found: {probe_path}\n"
            "These probes are on the cluster (see help in load_mlp)."
        )
    ckpt = torch.load(probe_path, map_location=device, weights_only=False)
    # Probes were saved as nn.Linear state dicts (key 'weight' / 'bias').
    def _weight(sd):
        if "weight" in sd:
            return sd["weight"].to(device)
        # Some variants nest the linear inside `proj.weight`
        if "proj.weight" in sd:
            return sd["proj.weight"].to(device)
        raise KeyError(f"Cannot find probe weight in keys: {list(sd.keys())}")
    we = _weight(ckpt["even"])
    wo = _weight(ckpt["odd"])
    if we.shape[1] != hidden_dim:
        raise ValueError(
            f"Probe hidden_dim={we.shape[1]} != requested {hidden_dim}")
    return {"probe_even": we.detach(), "probe_odd": wo.detach()}


# ---------------------------------------------------------------------------
# Class indexing under the MLP probe convention (0=empty, 1=mine, 2=yours)
# ---------------------------------------------------------------------------
def board_val_to_mlp_probe_class(val: int, pos: int) -> int:
    """Map board value (+1/-1/0) at position `pos` to the MLP probe's class.

    MLP probe convention: 0 = empty, 1 = mine, 2 = yours.
    "Mine" is +1 (black) iff pos % 2 == 1, else -1 (white).
    """
    if val == 0:
        return 0
    is_black_to_move = (pos % 2 == 1)
    mine = 1 if is_black_to_move else -1
    return 1 if val == mine else 2


def mlp_direction(
    probe_weight: torch.Tensor,
    r: int, c: int,
    current_class: int, target_class: int,
) -> torch.Tensor:
    """Direction in H-dim hidden space:
        d = w[cell, target_class] - w[cell, current_class]
    probe_weight: (192, H). Reshape internally to (64, 3, H)."""
    H = probe_weight.shape[1]
    W = probe_weight.view(64, 3, H)
    cell = r * 8 + c
    return W[cell, target_class] - W[cell, current_class]


# ---------------------------------------------------------------------------
# prob_or aggregation (960 pattern logits -> 60 cell probabilities)
# ---------------------------------------------------------------------------
def _build_pattern_to_cell():
    """Cache the pattern_to_cell mapping built from
    `hand_crafted_flanking.enumerate_flanking_patterns()`. Returns
    (idx_padded, mask) for noisy-or aggregation."""
    import hand_crafted_flanking as hcf
    patterns = hcf.enumerate_flanking_patterns()
    # pattern_to_cell[p_idx] = MOVE_TO_IDX[target] (0..59)
    pat_to_cell = np.array(
        [hcf.MOVE_TO_IDX[p["target"]] for p in patterns], dtype=np.int64)

    # For each cell c, the indices of patterns that target c.
    groups = [np.where(pat_to_cell == c)[0] for c in range(60)]
    K = max(len(g) for g in groups)
    idx_padded = np.zeros((60, K), dtype=np.int64)
    mask = np.zeros((60, K), dtype=bool)
    for c, g in enumerate(groups):
        idx_padded[c, :len(g)] = g
        mask[c, :len(g)] = True
    return torch.from_numpy(idx_padded), torch.from_numpy(mask)


_PATTERN_CELL_CACHE = None


def get_pattern_to_cell_mapping(device: str = "cpu"):
    """Returns (idx_padded, mask) on `device`. Cached after first call."""
    global _PATTERN_CELL_CACHE
    if _PATTERN_CELL_CACHE is None:
        _PATTERN_CELL_CACHE = _build_pattern_to_cell()
    idx, mask = _PATTERN_CELL_CACHE
    return idx.to(device), mask.to(device)


def prob_or_aggregate(
    pat_logits: torch.Tensor,
    idx_padded: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Aggregate 960 pattern logits -> 60 cell probs via noisy-or:
        p_cell = 1 - prod_k (1 - sigmoid(logit_k))
    pat_logits: (...batch..., 960)
    idx_padded: (60, K)
    mask:       (60, K)
    Returns:    (...batch..., 60)
    """
    log1m = -F.softplus(pat_logits)              # log(1 - sigmoid(x))
    g = log1m[..., idx_padded]                    # (..., 60, K)
    g = g.masked_fill(~mask, 0.0)
    return 1.0 - torch.exp(g.sum(dim=-1))


def prob_or_logits(cell_probs: torch.Tensor) -> torch.Tensor:
    """Map noisy-or cell probs -> log-odds so we can reuse all the existing
    OGPT cell-logit metrics. Returns a (..., 60) tensor of logits.

    To match the OGPT-style "60 cell logits => softmax => 60 cell probs"
    convention used by `metrics.cell_logits` etc., we instead just store the
    cell probabilities directly and use a thin shim in `mlp_compute_metrics`.
    """
    eps = 1e-9
    p = cell_probs.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


# ---------------------------------------------------------------------------
# High-level: cached clean state + intervention
# ---------------------------------------------------------------------------
@dataclass
class MLPInterventionSpec:
    """Direction + projection for one MLP intervention."""
    cell: tuple[int, int]
    current_class: int
    target_class: int
    flip_dir: torch.Tensor
    d_hat: torch.Tensor
    coeff: float

    @classmethod
    def from_probe(cls, probe_weight: torch.Tensor,
                   r: int, c: int,
                   current_class: int, target_class: int,
                   h_clean: torch.Tensor):
        flip_dir = mlp_direction(probe_weight, r, c, current_class, target_class)
        d_hat = flip_dir / flip_dir.norm()
        coeff = float(h_clean @ d_hat)
        return cls((r, c), current_class, target_class, flip_dir.detach(),
                   d_hat.detach(), coeff)


def model_and_probe_for_pos(
    bundle: dict, probe_bundle: dict, pos: int,
) -> tuple[DirectMLP, torch.Tensor]:
    """Pick (model, probe weight) based on position parity."""
    if pos % 2 == 0:
        return bundle["model_even"], probe_bundle["probe_even"]
    return bundle["model_odd"], probe_bundle["probe_odd"]


def cache_mlp_clean_state(
    rec,                       # data.PositionRecord
    bundle: dict,              # output of load_mlp
    probe_bundle: dict,        # output of load_mlp_probe
    *,
    device: str = "cpu",
):
    """Populate rec.extras with MLP-specific cached values:
      mlp_features      — (input_dim,) tensor
      mlp_hidden_clean  — (H,) hidden activation
      mlp_clean_probs   — (60,) noisy-or cell probabilities
      mlp_clean_logits  — (60,) log-odds derived from clean_probs (for the
                          OGPT metric interfaces that expect cell logits)
      mlp_spec          — MLPInterventionSpec for this record
      mlp_current_class — MLP probe class index for orig_val
      mlp_target_class  — MLP probe class index for target_val
    """
    feat_view = FEATURE_VIEWS[bundle["feature_type"]][1]
    raw = build_raw_180d_features(rec.move_history)
    feat = torch.from_numpy(feat_view(raw)).to(device).unsqueeze(0)   # (1, D)

    model, probe_w = model_and_probe_for_pos(bundle, probe_bundle, rec.position)
    with torch.no_grad():
        h = model.hidden(feat)[0]                           # (H,)
        pat_logits = model.from_hidden(h.unsqueeze(0))[0]    # (960,)
        idx, mask = get_pattern_to_cell_mapping(device=device)
        cell_probs = prob_or_aggregate(pat_logits, idx, mask)

    r, c = rec.square
    cur_cls = board_val_to_mlp_probe_class(rec.orig_val, rec.position)
    tgt_cls = board_val_to_mlp_probe_class(rec.target_val, rec.position)
    spec = MLPInterventionSpec.from_probe(
        probe_w, r, c, cur_cls, tgt_cls, h_clean=h,
    )

    rec.extras.update({
        "mlp_features": feat,
        "mlp_hidden_clean": h,
        "mlp_clean_probs": cell_probs,
        "mlp_clean_logits": prob_or_logits(cell_probs),
        "mlp_spec": spec,
        "mlp_current_class": cur_cls,
        "mlp_target_class": tgt_cls,
    })
    return rec


def run_mlp_intervention(
    rec,
    bundle: dict,
    probe_bundle: dict,
    specs: list,            # one or more MLPInterventionSpec
    alphas: list,
    *,
    device: str = "cpu",
):
    """Apply MLP intervention(s) and return (cell_probs, cell_logits).
    cell_probs is the noisy-or aggregation of post-intervention pattern logits.
    cell_logits is log-odds of those probs, for reuse with OGPT metrics.
    """
    model, _ = model_and_probe_for_pos(bundle, probe_bundle, rec.position)
    h = rec.extras["mlp_hidden_clean"].clone()
    for spec, a in zip(specs, alphas):
        h = h - a * spec.coeff * spec.d_hat
    with torch.no_grad():
        pat_logits = model.from_hidden(h.unsqueeze(0))[0]
        idx, mask = get_pattern_to_cell_mapping(device=device)
        cell_probs = prob_or_aggregate(pat_logits, idx, mask)
    cell_logits = prob_or_logits(cell_probs)
    return cell_probs, cell_logits


def normalize_probs(probs_60: torch.Tensor) -> torch.Tensor:
    """Renormalize noisy-or cell probabilities to sum to 1, so the same
    mass-shift metric semantics apply to MLP and OGPT.

    Without renormalization, the MLP's per-cell legality probabilities don't
    constitute a categorical distribution — comparing 'mass shift' would mix
    in changes to the global P(legal=*) signal. Renormalizing converts the
    independent-per-cell signal into a categorical 'which cell is the next
    move?' distribution, which is directly comparable to OGPT's softmax.
    """
    s = probs_60.sum()
    if s == 0:
        return probs_60
    return probs_60 / s


def li_topn_from_probs(probs_60: torch.Tensor, legal_set):
    """Li top-N for a probability vector (not logits). Mirrors
    `metrics.li_topn_accuracy` but skips the softmax step."""
    from . import config
    legal_stoi = set(legal_set) & set(config.STOI_INDICES)
    n = len(legal_stoi)
    if n == 0:
        return None
    topn_idx = probs_60.argsort(descending=True)[:n]
    topn_cells = {config.STOI_INDICES[int(i)] for i in topn_idx}
    fp = topn_cells - legal_stoi
    fn = legal_stoi - topn_cells
    return 1.0 - (len(fp) + len(fn)) / (2.0 * n)
