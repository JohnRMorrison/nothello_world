# J1B pipeline — train & evaluate the interpretable tree-bank Othello-MLP

**J1B** is the interpretable counterpart to OthelloGPT: a two-part model that
predicts move legality from a move-set feature vector.

1. **Hidden layer** = one `sklearn` `DecisionTreeClassifier` per flanking pattern
   (960 patterns); every tree leaf becomes a hidden unit.
2. **Readout** = a linear prob-OR head (`LinearPatternProbOr`, "linpo") trained
   by streaming over millions of games to predict per-cell legality.

Everything runs on the cluster (SLURM). conda env: **`othello`**. Repo root on
the pod: `/engram/nklab/jrm2182/nothello_world`.

---

## Pipeline at a glance

```
data/othello_synthetic/*.pickle  +  hand_crafted_flanking_patterns.pt
        │
        ▼  STAGE 1  train_midgame_tree_legal.sh  →  midgame_tree_mlp.py
ckpts_midgame/midgame_leg_pattern_trees_*.pt          (the tree bank = hidden layer)
        │
        ▼  STAGE 2  train_streaming_probe.sh    →  train_streaming_probe.py
ckpts_midgame/stream_linpo_g6000000_ep3_<tag>_j<JOBID>_<ts>.pt   (the readout)
        │
        ▼  EVAL     reeval_argmax_legality.py
argmax-legality accuracy (+ per-ply breakdown)
```

Wrappers are `sbatch` scripts; each `python` backend can also be run directly.

---

## Stage 1 — fit the trees (hidden layer)

Wrapper **`train_midgame_tree_legal.sh`** (job `midgame_leg`) → **`midgame_tree_mlp.py`**.
Fits one decision tree per flanking pattern (per-pattern, mover-canonical, no
recency for J1B):

```bash
STAGE=full CANONICALIZE_MOVER=1 \
  sbatch --time=06:00:00 train_midgame_tree_legal.sh <variant> 20000 5000 15 50 10 50 50
#                                                     └ positional: NUM_TRAIN NUM_TEST MAX_DEPTH MIN_LEAF PLY_MIN PLY_MAX TOP_K
```

- Key flags it sets for the pattern bank: `--tree-target patterns --pattern-n-trees 1
  --include-flanking-patterns hand_crafted_flanking_patterns.pt`.
- The variant options (per-pattern / grouped / recency / base) are documented in
  the header of `train_midgame_tree_legal.sh`.
- **Output:** `ckpts_midgame/midgame_leg_pattern_trees_no_recent_canonical_g20000_d15_ml50_p10-50.pt`
  (the exact name encodes games/depth/leaf/ply). This is the tree bank you pass
  to Stage 2.
- Tree fitting is **not** mid-fit resumable — keep `--time=06:00:00` (a `--tree-cache`
  path lets a resubmit skip re-fitting once a fit completes).

## Stage 2 — train the readout (streaming)

Wrapper **`train_streaming_probe.sh`** (job `stream_probe`) → **`train_streaming_probe.py`**.
Loads the tree bank, streams games, trains the `linpo` legality head:

```bash
LOAD_TREES=ckpts_midgame/midgame_leg_pattern_trees_no_recent_canonical_g20000_d15_ml50_p10-50.pt \
  PROBE_TYPE=linpo CANONICALIZE_MOVER=1 RECENT_KS="" \
  DATA_SOURCE=chunk-ext NUM_GAMES=6000000 EPOCHS=3 PLY_MIN=10 PLY_MAX=50 \
  CHECKPOINT_EVERY=1 \
  sbatch --time=06:00:00 train_streaming_probe.sh
```

- `DATA_SOURCE=chunk-ext` reads pre-simulated feature chunks from
  `CHUNK_DIR` (default `experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks`);
  `DATA_SOURCE=pickle` streams raw games from `data/othello_synthetic` instead.
- `RECENT_KS=""` + `CANONICALIZE_MOVER=1` **must match** the tree bank (no-recent + canonical).
  `PLY_MIN/PLY_MAX` must match the bank's `p{lo}-{hi}`.
- **Resumable per chunk:** re-submit the *identical* env-var command to continue
  (a config-keyed sidecar under `ckpts_midgame/resume/` tracks the last chunk;
  the log prints `RESUMED from … chunk N`). Chunks are slow (~45 min each) → a 6h
  window does ~7–8 chunks.
- **Output:** `ckpts_midgame/stream_linpo_g6000000_ep3_<treetag>_j<JOBID>_<ts>.pt`
  (only written on successful completion).
- Variants: `PROBE_TYPE=strupo` (sparse per-pattern head), `NO_FLANKING=1`
  (tree-only hidden layer), `train_streaming_state.sh` (board-**state** decoder).

## Evaluation — argmax legality

**`reeval_argmax_legality.py`** — the headline metric (per-cell prediction, argmax
over cells, is-it-legal), matching the MLP's legality accuracy. Loads the probe
checkpoint + the tree bank it references.

```bash
sbatch --mem=48G --wrap "python reeval_argmax_legality.py \
  --probe-ckpts ckpts_midgame/stream_linpo_g6000000_ep3_<treetag>_j*.pt \
  --ply-min 10 --ply-max 50"
```

- Run via `sbatch --mem=48G` (it OOMs on the login node).
- Prints overall argmax-legality accuracy (single-seed linpo ≈ **98.7%**) plus a
  per-ply breakdown.
- The streaming log's own `eval per-cell acc:` is a *different* (higher) metric —
  don't compare it against these argmax numbers.

---

## Inputs / artifacts

| what | where |
|---|---|
| training games | `data/othello_synthetic/*.pickle` (~24M games) |
| flanking patterns | `hand_crafted_flanking_patterns.pt` |
| feature chunks (chunk-ext) | `experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks/chunk_ext_*.npz` |
| tree bank (Stage 1 out) | `ckpts_midgame/midgame_leg_pattern_trees_*.pt` |
| readout (Stage 2 out) | `ckpts_midgame/stream_linpo_*_j*.pt` |

Backends you can run directly (add `--help`): `midgame_tree_mlp.py`,
`train_streaming_probe.py`, `train_streaming_state.py`, `reeval_argmax_legality.py`.
