"""Defaults + paths + constants for the intervention experiments.

The notebook builds a `params` dict that overrides these for each run; nothing
in the experimental code path should read directly from this module except as
a source of defaults. In particular, *no layer numbers are hardcoded* outside
of this file — everything that depends on a layer takes it as an argument.
"""

import os

# --- Default experiment parameters (overridable in the notebook) -----------
DEFAULT_INTERVENTION_LAYER = 4
# cal_depth = number of transformer blocks between intervention and probe.
# probe_layer = intervention_layer + cal_depth.
# cal_depth = 0: intervene + decode at the same layer (no propagation).
# cal_depth = 2: Nanda's published setup (intervene L4, probe L6).
DEFAULT_CAL_DEPTH = 0
# How alpha is chosen per intervention:
#   "fixed"        — same alpha for every position/square (Nanda's protocol)
#   "per_square"   — binary-search a single alpha for each target square,
#                    reused across positions on that square
#   "per_position" — binary-search alpha for every (square, position) pair
DEFAULT_CALIBRATION_MODE = "fixed"
DEFAULT_ALPHA = 2.0

# Probe convention. Mode 0 (and mode 1) use mine/yours/empty labels (~99%
# probe accuracy at layer 6). Mode 2 uses absolute labels and is degraded.
DEFAULT_PROBE_MODE = 0
PROBE_LAYER_NANDA = 6   # layer at which `main_linear_probe.pth` is trained

# Database build params.
DEFAULT_N_POSITIONS_PER_CONDITION = 1000
DEFAULT_MAX_GAMES = 10_000
DEFAULT_POS_LO = 10
DEFAULT_POS_HI = 50    # exclusive
DEFAULT_SEED = 42

# Repo paths (resolved relative to repo root).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_PATH = os.path.join(_REPO_ROOT, "ckpts/gpt_nanda_synthetic.ckpt")
PROBE_PATH = os.path.join(
    _REPO_ROOT, "mechanistic_interpretability/main_linear_probe.pth")
INT_DATA_PATH = os.path.join(
    _REPO_ROOT, "mechanistic_interpretability/board_seqs_int_small.npy")
STRING_DATA_PATH = os.path.join(
    _REPO_ROOT, "mechanistic_interpretability/board_seqs_string_small.npy")

# Output folder for all text tables, figures, saved databases.
OUTPUT_DIR = os.path.join(_REPO_ROOT, "Intervention Results")

# --- Board / token constants (these are derived from the model spec; not
# parameters in any meaningful sense, so they live here) -------------------
# The 60 valid (non-center) cells, as flat 0–63 indices.
STOI_INDICES = list(range(27)) + list(range(29, 35)) + list(range(37, 64))
STOI_INDEX_OF = {c: i for i, c in enumerate(STOI_INDICES)}
CENTER_CELLS = {27, 28, 35, 36}        # never an intervention target
ALPHA_LABELS = "ABCDEFGH"
N_VALID_CELLS = 60

# --- Condition taxonomy ---------------------------------------------------
# Categories: what kind of board edit. (add is split by added color.)
CATEGORIES = ["remove", "add_mine", "add_yours", "flip"]

# Sub-conditions: how the legality set changes.
SUB_CONDITIONS = ["newly_legal", "newly_illegal", "both"]
# newly_legal:  |legal_cf \ legal_orig| >= 1, |legal_orig \ legal_cf| == 0
# newly_illegal: |legal_cf \ legal_orig| == 0, |legal_orig \ legal_cf| >= 1
# both:         both sets nonempty


def assemble_params(**overrides) -> dict:
    """Build a params dict starting from defaults; overrides win.

    Returns a fresh dict; callers can mutate it freely.
    """
    params = {
        "intervention_layer": DEFAULT_INTERVENTION_LAYER,
        "cal_depth": DEFAULT_CAL_DEPTH,
        "calibration_mode": DEFAULT_CALIBRATION_MODE,
        "alpha": DEFAULT_ALPHA,
        "probe_mode": DEFAULT_PROBE_MODE,
        "n_positions_per_condition": DEFAULT_N_POSITIONS_PER_CONDITION,
        "max_games": DEFAULT_MAX_GAMES,
        "pos_lo": DEFAULT_POS_LO,
        "pos_hi": DEFAULT_POS_HI,
        "seed": DEFAULT_SEED,
        # Derived but exposed for convenience; recomputed if user changes the
        # underlying fields.
        "probe_layer": (DEFAULT_INTERVENTION_LAYER + DEFAULT_CAL_DEPTH),
    }
    params.update(overrides)
    # Re-derive probe_layer if user supplied intervention_layer or cal_depth.
    params["probe_layer"] = (
        params["intervention_layer"] + params["cal_depth"]
    )
    return params
