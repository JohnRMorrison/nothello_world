# Othello-GPT board code: CCGP abstraction analysis

**Question.** Decodability of the board from Othello-GPT does not by itself prove a
"world model" — a linear probe could be reading a bag of context-specific
correlates rather than a unified state variable. We use **CCGP** (Cross-Condition
Generalization Performance; Bernardi/Fusi et al. 2020, "the geometry of
abstraction") to test whether the board code is **abstract**: train a decoder on
one region of a condition, test on a held-out region. If it transfers, the
variable is encoded in a reusable, low-dimensional format — a genuine state
variable, not a per-context detector.

## Setup

- **Model / activations.** Nanda synthetic Othello-GPT (8 layers, d=512).
  Representation = residual stream `resid_post` at **layer 6**.
- **Positions.** 100,000 synthetic games, **one probed ply per game**
  (independent test set), plies **5–53**.
- **Frame.** **mine/yours, parity-split** — separate probes for even/odd plies on
  the absolute `{empty, white, black}` labels. Within a fixed parity, absolute
  color == mine/yours, so this is Nanda's mine/yours frame (the frame in which the
  board is linearly decodable). Numbers below average the two parities.
- **Probe.** Logistic regression, **class-balanced binary** per (cell, class)
  "is cell C == color X". CCGP and the within-condition ceiling use the **same
  matched train size**, so the Gap is a fair comparison.
- **Metrics.**
  - **CCGP** = held-out-condition decode accuracy (train one region, test the other).
  - **Within** = within-condition ceiling at the matched train size.
  - **Gap = Within − CCGP.** Small Gap = abstract/transferable; large Gap =
    context-bound. `Within` here is high (~0.92–0.99), so probes are fully powered.

## Conditions

Each condition defines a nuisance variable **V** that a genuine board code should
be invariant to (decode C the same regardless of V). Train on one value of V,
test on the other.

| Condition | Split (train → test) | Invariance tested |
|---|---|---|
| **null** | random split (control) | none — CCGP should ≈ Within; validates the estimator is unbiased |
| **context** | diagonal-opposite cell empty → occupied (and reverse) | is C's state read independently of a distant cell? |
| **crowd** | few occupied neighbors → many | invariance to local crowding |
| **frontier** | C interior (all neighbors filled) → C adjacent to an empty square | invariance to being on the frontier of play |
| **flip** | C never net-flipped → C net-flipped (current color ≠ placement color) | Markov: is current state coded independent of capture history? |
| **recency** | C recently played → long-settled (moves-since-placement) | do settled squares decode like fresh ones? |
| **phase** | leave-one-ply-bin-out over 4 bins (**interpolation**; held-out bin is surrounded by training bins) | stability across game phase, with support on both sides |
| **phase_fwd** | train earliest bins → test **latest** bin (**extrapolate forward** in game-time) | does the code learned early cover late boards it never saw? |
| **phase_bwd** | train latest bins → test **earliest** bin (**extrapolate backward**) | does the code learned late cover early boards it never saw? |
| **spatial** | leave-cells-out: train a single decoder pooled over some squares → decode held-out squares | is there **one shared** "is-mine" direction reused across all 64 squares (translation-invariance)? |

## Results (Othello-GPT, layer 6, N=100k)

Gap ordered small → large:

| Condition | CCGP | Within | **Gap** |
|---|---|---|---|
| null (control) | 0.990 | 0.990 | **0.000** |
| phase_fwd (→ late) | 0.981 | 0.985 | **0.003** |
| recency | 0.988 | 0.992 | **0.004** |
| context | 0.984 | 0.989 | **0.005** |
| phase (interpolation) | 0.889 | 0.925 | **0.031** |
| flip | 0.930 | 0.992 | **0.062** |
| frontier | 0.923 | 0.985 | **0.062** |
| crowd | 0.906 | 0.971 | **0.063** |
| phase_bwd (→ early) | 0.862 | 0.943 | **0.081** |
| spatial | 0.621 | 0.983 | **0.364** |

## Interpretation

**The per-cell board code is strongly abstract.** For the invariances that matter
most to a "state variable" reading — **context (0.005), recency (0.004),
forward-phase (0.003)**, and interpolated **phase (0.031)** — the Gap is near
zero. A probe trained in one context/time decodes the same cell in another. This
is the positive evidence for a genuine, reusable board-state representation rather
than context-specific detectors.

**Mild, real dependencies (Gap ~0.06).** `flip`, `crowd`, and `frontier` each cost
~6 points of transfer. So the code is *not perfectly* invariant to whether C was
just captured, or to local crowding / being on the frontier — there is a small
context signature, but the state is still mostly recovered.

**Phase transfer is asymmetric.** `phase_fwd` (train early → test late) Gap 0.003
vs `phase_bwd` (train late → test early) Gap 0.081. The code generalizes **forward
in game-time but not backward**: what is learned on rich late-game boards does not
cover sparse early boards, while early-board structure carries forward. (This
reconciles the common observation that a probe tested on move numbers it never saw
does poorly — that is the *backward* / into-the-unseen-early-regime direction.)
Note within-bin decodability itself falls with ply (early ~0.96 → late ~0.90), the
expected board-decode ply-decay; the Gap is normalized against that.

**`spatial` is large (0.36) but must be read carefully — it is NOT evidence
against a world model.** This mode asks whether **one** "is-mine" direction is
shared across all 64 squares. The pooled decoder is given the activation but not
*which* square, so a large Gap simply means **each square has its own code
direction** — which is expected of essentially any model ("A is mine" and "B is
mine" are genuinely different features). It characterizes the code as **per-cell**,
not translation-invariant; it does not undermine the per-cell abstraction shown by
every other condition. (Validated on synthetic data: a truly shared direction
yields Gap ≈ 0; independent per-cell codes yield Gap ≈ 0.4, matching OGPT.)

## Bottom line

Othello-GPT's board representation behaves like a **genuine, per-square mine/yours
state variable**: each square's state is decoded consistently across game phase
(interpolated), distant context, and recency (Gap ≈ 0), with only mild sensitivity
to local crowding, frontier status, and flip history (~0.06), and one asymmetry —
it extrapolates forward in game-time but not backward. The board code is *not* a
single translation-shared direction (per-cell, not per-board), which is expected
and not a mark against the world-model interpretation.

*Controls / rigor: `null` Gap = 0.000 with `Within` ≈ 0.99 confirms the estimator
is unbiased and the probes are fully powered at N=100k; results are stable at
N=30k. One probed position per game keeps the test set independent.*

*Reproduce: `run_ccgp_matrix.py` / `compute_ccgp.py` (`--mode-data shared`), which
run OGPT and the comparison models on identical positions.*
