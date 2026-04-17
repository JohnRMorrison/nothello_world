# 2×2 Factorial Transfer Learning Experiment

A causal test of whether OthelloGPT's extracted heuristic rules are load-bearing
for next-move prediction, versus correlational artifacts of a deeper "world
model" computation. This document explains the scientific design, how to run
the pipeline, and how to interpret the results.

---

## 1. Scientific question

[Li et al. (2023)](https://arxiv.org/abs/2210.13382) argue that OthelloGPT
develops an internal "world model" — a linearly decodable representation of
the 8×8 board — which they interpret as evidence that the model understands
the game rather than memorizing patterns. Follow-up work (e.g., Singh et al.
2025, "Automatically Finding Rule-Based Neurons") extracts human-readable
IF-THEN rules from individual neurons, suggesting the model also encodes local
heuristics (e.g., *"if F3 was flipped and G3 is not empty, promote F5"*).

The scientific question is whether these heuristics are **causally engaged**
during prediction, or whether they are epiphenomenal — present in activations
but not load-bearing for the output. Our strategy: construct modified games
whose rules align with specific extracted heuristics, and measure whether the
pretrained model adapts to them faster than to matched-but-unaligned controls.

All interventions are at the **game-rule level**, not the weight level. This
is crucial:

- A weight edit ("negate neurons X, Y, Z") has no well-defined fine-tuning
  target — you can't train on data that doesn't exist.
- Changing the game's rules creates a valid data-generating process. The
  only thing that differs between conditions is how well the pretrained
  model's internal circuitry already matches the new rules.

---

## 2. The 2×2 design

Each modified game adds one or more **move-forbiddance rules** to standard
Othello:

> *"Standard Othello, except: whenever some board-state condition `C` holds,
> moves in the set `S` are forbidden (even if otherwise legal)."*

The condition `C` is called the **antecedent**; the forbidden set `S` is the
**consequent**. Each can independently be either *aligned* (derived from the
model) or *random*, giving a 2×2 factorial:

|                          | Cons: aligned (DLA argmax)            | Cons: random             |
|--------------------------|---------------------------------------|--------------------------|
| **Ant: aligned** (rule)  | **B₁** full alignment                 | **B₂** antecedent-only   |
| **Ant: random** (matched)| **B₃** consequent-only                | **C** no alignment       |

**Pairwise contrasts** each isolate a specific claim:

| Contrast       | What it isolates                          |
|----------------|-------------------------------------------|
| B₁ vs B₂       | The *consequent* alignment effect         |
| B₁ vs B₃       | The *antecedent* alignment effect         |
| B₁ vs C        | The *combined* alignment effect           |
| B₁ vs C        | (orthogonal decomposition: sum of above)  |

A from-scratch training arm per condition addresses the "B₁ is just an easier
game" critique: if B₁ is easier only because its rules match the model's
pretrained machinery, then a from-scratch model should see no such advantage.

---

## 3. Quadruple construction

For each selected neuron we co-construct **all four arms** in a single
"quadruple," so they share structure and differ only on the intended axes.

**Construction order** (chosen to avoid circular self-reference constraints
among the exclusion sets):

1. `Ant_aligned` ← `parse_rule_conditions(rule_str)`. Squares: `S_A`.
2. `Cons_aligned` ← DLA argmax (`W_U @ W_out[:, neuron]`) over valid squares,
   skipping tautological targets and anything in `S_A`.
3. `Ant_random` ← frequency-matched random reassignment of `S_A` (best-of-N,
   see §4), with `S_A ∪ {Cons_aligned}` forbidden as replacement squares.
   Squares: `S_R`.
4. `Cons_random` ← uniform sample from valid squares excluding
   `S_A ∪ S_R ∪ {Cons_aligned}`.

**Invariants** verified per quadruple (`verify_quadruple` in
`build_restriction_configs.py`):

- `Cons_aligned ∉ S_A ∪ S_R`
- `Cons_random ∉ S_A ∪ S_R`
- `Cons_aligned ≠ Cons_random`
- `len(Ant_aligned) == len(Ant_random)` (same number of conjuncts)

This means the *consequent square* never appears as a condition square in any
of the four arms — the model can't satisfy the restriction by trivially
recognizing its own output square.

**If any arm cannot be satisfied** (e.g., no non-tautological DLA target
exists, or random-antecedent sampling fails), the neuron is dropped with a
diagnostic message. The over-selection pool (3K candidates) typically keeps
K survivors; use `--strict-K` to hard-fail otherwise.

---

## 3a. Multi-square consequents (`--n-forbidden-squares`)

By default each restriction forbids only **one** square when it fires. With
a typical board offering ~8 legal moves, the pretrained model's argmax almost
never hits that exact square — step-0 violation rates are ~2–3%, and the
model adapts in <50 gradient steps regardless of condition. The alignment
signal is undetectable because the task is too easy.

The `--n-forbidden-squares N` flag (default 5, exposed as `N_FORBIDDEN` in
`run_2x2.sh`) instructs `build_restriction_configs.py` to select the **top-N
DLA squares** per neuron as the aligned consequent, and N uniformly random
squares (disjoint from aligned) as the random consequent. When a restriction
fires, all N squares become illegal simultaneously.

**Why this amplifies the alignment signal:**

- **Higher step-0 violation rate.** With N=5, the model must avoid 5 of ~8
  legal moves per fired position. The aligned targets are exactly the squares
  the neuron *promotes* via DLA — so the pretrained model is most likely to
  predict those. Step-0 `violation_rate_when_fires` should be substantially
  higher for aligned consequents (B₁, B₃) than random ones (B₂, C).
- **Harder task.** Redistributing probability mass away from 5 squares is
  much harder than from 1. The model can no longer trivially satisfy the
  restriction; it must genuinely learn the antecedent to know *when* the
  restriction applies.
- **More room for alignment to help.** If the pretrained model's recognition
  circuitry already computes the antecedent, it has a head start for B₁
  (both antecedent and consequent aligned) vs C (neither aligned).

**Construction changes:** `choose_aligned_consequent` walks the DLA-sorted
target list and picks the top N non-tautological squares (excluding antecedent
squares). `choose_random_consequent` samples N random squares (excluding
antecedent squares and the aligned set). All self-reference exclusion
invariants extend to sets: no consequent square (aligned or random) may
appear in any antecedent.

**JSON schema:** restrictions now carry `forbidden_positions: [int, ...]` and
`forbidden_squares: [str, ...]` (lists). The legacy `forbidden_position` /
`forbidden_square` keys still exist (pointing to the first target) for
backward compatibility.

**Recommended value:** N=5 for a publishable run. N=1 recovers the original
single-square design. Values above ~8 risk making the task too hard (most
legal moves forbidden → frequent fallback to full legal set).

---

## 4. Firing-rate matching

"Firing rate" = fraction of board positions at which an antecedent evaluates
to True, estimated on a held-out sample of `--tautology-games` standard
Othello games.

By construction:

- B₁ and B₂ share antecedent → **identical** firing rates.
- B₃ and C share antecedent → identical firing rates.
- B₁ vs B₃ is matched via best-of-`--n-random-attempts` (default 50) random
  square reassignments, picking the one with firing rate closest to B₁.

After construction, a **hard-fail gate** aborts config generation if any
quadruple's `|fire_rate_aligned - fire_rate_random|` exceeds
`--max-firing-rate-diff` (default 0.05). This prevents a confound where, e.g.,
the aligned antecedent fires 3× more often than the random one, making B₁
intrinsically harder or easier.

If the gate fires, options are:

- Raise `--max-firing-rate-diff` (accept noisier matching)
- Raise `--n-random-attempts` (more thorough search)
- Loosen selection (`--layers`, `--K`, `--min-score` in source rules)

---

## 5. From-scratch baseline

For each of the four conditions, we run two training modes:

- **`ft`**: fine-tune from the pretrained `gpt_synthetic.ckpt` checkpoint.
- **`scratch`** (alias: `rnd`): train from random initialization with the
  same architecture, data, and hyperparameters. LR can be overridden via
  `--lr-scratch`.

**Interpretation logic:**

| Observation                                                 | Conclusion                                              |
|-------------------------------------------------------------|---------------------------------------------------------|
| B₁-ft ≪ C-ft AND B₁-ft ≪ B₁-scratch                         | Pretrained heuristics are causally engaged              |
| B₁-ft ≪ C-ft BUT B₁-ft ≈ B₁-scratch                         | B₁ is just an easier game; alignment claim unsupported  |
| B₁-ft ≈ C-ft                                                | No detectable alignment effect                          |
| B₁-ft ≪ B₂-ft AND B₁-ft ≈ B₃-ft                             | Consequent alignment dominates; antecedent doesn't help |
| B₁-ft ≪ B₃-ft AND B₁-ft ≈ B₂-ft                             | Antecedent alignment dominates                          |
| B₁-ft ≪ B₂-ft ≈ B₃-ft ≪ C-ft                                | Both factors contribute, roughly additively             |

---

## 6. How to run

### Quick start

```bash
cd experiments/transfer_learning_experiments
bash run_2x2.sh
```

This runs the full pipeline (configs → games → finetune → plot) and writes
everything under `runs/2x2_<timestamp>/`.

### Smoke test first

Before a full run (≈24 trainings at ~5k steps each), verify the pipeline end
to end with small sizes:

```bash
K=2 N_RUNS=1 MAX_STEPS=200 NUM_GAMES=10000 EVAL_GAMES=50 bash run_2x2.sh
```

Expect completion in <15 min on one modern GPU.

### Environment overrides

All pipeline knobs are exposed as env vars in `run_2x2.sh`:

| Variable                  | Default                                                     | Purpose                                                     |
|---------------------------|-------------------------------------------------------------|-------------------------------------------------------------|
| `RULES_FILE`              | `../reverse_engineering_experiments/rules_085_200_2-6.json` | Source rules JSON from `extract_rules.py`                   |
| `CKPT`                    | `../../ckpts/gpt_synthetic.ckpt`                            | Pretrained OthelloGPT checkpoint                            |
| `LAYERS`                  | `2-5`                                                       | Layers to draw neurons from                                 |
| `K`                       | `20`                                                        | Target number of quadruples                                 |
| `N_FORBIDDEN`             | `5`                                                         | Squares forbidden per restriction (see §3a)                 |
| `N_RUNS`                  | `3`                                                         | Seeds per sweep (8 sweeps × N_RUNS = total runs)            |
| `NUM_GAMES`               | `500000`                                                    | Games generated per condition                               |
| `MAX_STEPS`               | `5000`                                                      | Gradient steps per sweep                                    |
| `EVAL_GAMES`              | `200`                                                       | Eval games per evaluation pass                              |
| `BATCH_SIZE`              | `16`                                                        | Matches existing protocol                                   |
| `LR`                      | `3e-4`                                                      | Fine-tune LR                                                |
| `LR_SCRATCH`              | `5e-4`                                                      | From-scratch LR (usually higher is better)                  |
| `MIN_CONDITIONS`          | `2`                                                         | Minimum conjuncts per rule (avoids single-feature cases)    |
| `TAUTOLOGY_THRESHOLD`     | `0.85`                                                      | Max `P(ant fires | target legal)`                           |
| `MAX_FIRING_RATE_DIFF`    | `0.05`                                                      | Hard-fail threshold for B₁ vs B₃ firing-rate mismatch       |
| `OUTPUT_ROOT`             | `runs`                                                      | Parent directory for all artifacts                          |

### Full evaluation workflow

Reasonable parameter set for a publishable run (adjust to taste):

| Parameter       | Value        | Rationale                                               |
|-----------------|--------------|---------------------------------------------------------|
| `K`             | 5            | Fewer restrictions with multi-square consequents         |
| `N_FORBIDDEN`   | 5            | Squares forbidden per fired restriction (see §3a)        |
| `N_RUNS`        | 3            | Seeds for mean ± SEM error bars                         |
| `NUM_GAMES`     | 500,000      | Standard OthelloGPT-scale training corpus per condition |
| `MAX_STEPS`     | 5,000        | Enough for ft curves to plateau                         |
| `LR` / `LR_SCRATCH` | 3e-4 / 5e-4 | Scratch benefits from slightly higher LR             |
| `EVAL_GAMES`    | 200          | ~5,000 positions per eval — stable statistics           |
| `EVAL_EVERY`    | 50 steps     | 100 eval points per run                                 |

Total compute: 8 training sweeps × 3 seeds = **24 trainings × 5,000 steps** at batch-size 16. On a single modern GPU (A100 / L40 / RTX 4090), expect **~20–30 minutes per sweep**, so ~10–15 hours end-to-end. Parallelize across GPUs by running sweeps concurrently (the script is sequential by default; split the inner loop by hand or use a scheduler).

#### Option 1 — one command, end to end

```bash
cd experiments/transfer_learning_experiments

K=5 N_FORBIDDEN=5 N_RUNS=3 NUM_GAMES=500000 MAX_STEPS=5000 \
    EVAL_GAMES=200 EVAL_EVERY=50 \
    BATCH_SIZE=16 LR=1e-5 LR_SCRATCH=5e-4 \
    MAX_FIRING_RATE_DIFF=0.05 TAUTOLOGY_THRESHOLD=0.85 \
    RULES_FILE=../reverse_engineering_experiments/rules_085_200_2-6.json \
    bash run_2x2.sh 2>&1 | tee full_run.log
```

All artifacts land under `runs/2x2_<timestamp>/`. Watch `full_run.log` for per-sweep progress.

#### Option 2 — staged execution (recommended for a real run)

Runs each stage separately so you can inspect before proceeding. Useful if training takes a while and you want to verify configs / games first.

**Step 1: Build configs** (~1 min)

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
BASE=runs/2x2_${STAMP}
mkdir -p ${BASE}/configs ${BASE}/data ${BASE}/results ${BASE}/figures

python build_restriction_configs.py \
    --rules ../reverse_engineering_experiments/rules_085_200_2-6.json \
    --ckpt ../../ckpts/gpt_synthetic.ckpt \
    --layers 2-5 \
    --K 20 \
    --min-conditions 2 \
    --tautology-threshold 0.85 \
    --max-firing-rate-diff 0.05 \
    --output-dir ${BASE}/configs
```

**Sanity check:** open `${BASE}/configs/manifest.json`. Confirm:
- 20 entries in `per_quadruple`.
- `fire_rate_diff` < 0.05 on every row.
- `cons_aligned_square ≠ cons_random_square` on every row.
- `tautology_score` ≤ 0.85 on every row.

**Step 2: Generate games for all four conditions** (~30 min each; run serially or parallel)

```bash
for C in B1 B2 B3 C; do
    python generate_restricted_games.py \
        --config ${BASE}/configs/${C}.json \
        --output-dir ${BASE}/data/${C} \
        --num-games 500000
done
```

**Sanity check:** each `${BASE}/data/{B1,B2,B3,C}/` should contain ~5 pickles of ~100k games each. Inspect mean game length with a quick Python snippet:

```bash
python -c "
import pickle, glob, numpy as np
for cond in ['B1','B2','B3','C']:
    games = []
    for p in glob.glob(f'${BASE}/data/{cond}/*.pickle'):
        games += pickle.load(open(p,'rb'))
    lens = [len(g) for g in games[:5000]]
    print(f'{cond}: n={len(games)} mean_len={np.mean(lens):.1f}')
"
```

Game lengths should average 55–60 moves. If any condition drops below 45, the restrictions are too aggressive — lower `K` or raise `--max-firing-rate-diff` to admit lower-firing-rate configs.

**Step 3: Run 8 training sweeps × 3 seeds = 24 trainings** (~10–15 hours on one GPU)

```bash
for C in B1 B2 B3 C; do
    for MODE in ft scratch; do
        python finetune_and_evaluate.py \
            --games-dir ${BASE}/data/${C} \
            --config ${BASE}/configs/${C}.json \
            --label ${C}_${MODE} \
            --condition ${C} \
            --mode ${MODE} \
            --ckpt ../../ckpts/gpt_synthetic.ckpt \
            --runs 3 \
            --max-steps 5000 \
            --eval-games 200 \
            --eval-every 50 \
            --batch-size 16 \
            --lr 3e-4 \
            --lr-scratch 5e-4 \
            --output-dir ${BASE}/results \
            2>&1 | tee -a ${BASE}/train.log
    done
done
```

**Parallelizing across GPUs:** if you have N GPUs, export `CUDA_VISIBLE_DEVICES` and run condition/mode pairs concurrently. Example with 4 GPUs (one condition per GPU, both modes sequential):

```bash
for i in 0 1 2 3; do
    C=(B1 B2 B3 C)[$i]
    CUDA_VISIBLE_DEVICES=$i bash -c "
        for MODE in ft scratch; do
            python finetune_and_evaluate.py [...args...] --label ${C}_\${MODE} --condition ${C} --mode \${MODE}
        done
    " &
done
wait
```

**Sanity check after each sweep:** `top1_legal_when_fires` step-0 on `*_ft_*` should be 0.70–0.95 (room for improvement); on `*_scratch_*` should be near chance (0.05–0.20).

**Step 4: Plot all figures** (~30 s)

```bash
python plot_transfer_curves.py \
    --curves-dir ${BASE}/results \
    --out ${BASE}/figures \
    --metric top1_legal_when_fires \
    --smooth 3
```

This produces:
- `grid_top1_legal_when_fires.png` — the **headline 2×2 grid** (rows = antecedent, cols = consequent, ft/scratch overlaid per cell). This is the figure for the paper.
- `grid_top1_legal.png`, `grid_violation_rate_when_fires.png`, `grid_legal_mass.png` — secondary grids for sensitivity.
- `headline_top1_legal_when_fires.png` — all four ft curves on one axis.

#### Post-run analysis

Pull the key numbers into a table:

```bash
python -c "
import json, glob, os, numpy as np
results = {}
for p in sorted(glob.glob('${BASE}/results/curves_*.json')):
    d = json.load(open(p))
    key = (d['condition'], d['mode_canonical'])
    finals = [d['curves'][r][-1].get('top1_legal_when_fires') for r in d['curves']]
    finals = [v for v in finals if v is not None]
    results[key] = (np.mean(finals), np.std(finals) / np.sqrt(len(finals)))
print(f'{\"Cond\":<4} {\"Mode\":<8} {\"Final top1_legal_when_fires (mean ± SEM)\":<40}')
for (c, m), (mean, sem) in sorted(results.items()):
    print(f'{c:<4} {m:<8} {mean:.4f} ± {sem:.4f}')
"
```

Primary contrasts to report (mean ± SEM on the final-step value):
- **B₁-ft vs C-ft** — the headline alignment effect.
- **B₁-ft vs B₂-ft** — consequent-alignment contribution.
- **B₁-ft vs B₃-ft** — antecedent-alignment contribution.
- **X-ft vs X-scratch** (per condition) — confirms pretrained weights carry the structure.

See §8 (Interpretation guide) for decision rules on what each pattern implies.

---

### Running stages individually

```bash
# Configs only:
python build_restriction_configs.py \
    --rules ../reverse_engineering_experiments/rules_085_200_2-6.json \
    --output-dir configs/2x2_run1

# Games for one condition:
python generate_restricted_games.py \
    --config configs/2x2_run1/B1.json \
    --output-dir data/2x2_run1/B1

# One sweep (3 seeds):
python finetune_and_evaluate.py \
    --games-dir data/2x2_run1/B1 \
    --config configs/2x2_run1/B1.json \
    --label B1_ft --condition B1 --mode ft --runs 3 \
    --output-dir results/2x2_run1

# Plot the whole results directory:
python plot_transfer_curves.py \
    --curves-dir results/2x2_run1 \
    --out figures/2x2_run1
```

---

## 7. Output layout

```
runs/2x2_<timestamp>/
├── configs/
│   ├── B1.json        # Aligned ant + Aligned cons
│   ├── B2.json        # Aligned ant + Random cons
│   ├── B3.json        # Random ant  + Aligned cons
│   ├── C.json         # Random ant  + Random cons
│   └── manifest.json  # Meta + quadruple_id index
├── data/
│   ├── B1/*.pickle    # Restricted games per condition
│   ├── B2/...
│   ├── B3/...
│   └── C/...
├── results/
│   └── curves_<cond>_<mode>_<label>.json   # One per sweep (contains N_RUNS run-curves)
└── figures/
    ├── grid_top1_legal.png       # 2x2 grid: ft + scratch per cell
    ├── grid_top1_prob.png
    ├── grid_legal_mass.png
    └── headline_top1_legal.png   # Single-axis overlay of all 4 ft curves
```

### Config JSON schema

```json
{
  "label": "B1",
  "description": "Aligned antecedent + aligned consequent (full heuristic alignment)",
  "num_restrictions": 20,
  "meta": { "K_target": 20, "K_actual": 20, "layers": [2,3,4,5], ... },
  "restrictions": [
    {
      "id": "L5N123_r0",
      "quadruple_id": "L5N123_r0",
      "arm": "B1",
      "source_neuron": "L5N123",
      "antecedent_kind": "aligned",
      "consequent_kind": "aligned",
      "conditions": [{"square": "F3", "feature_type": "flipped", "polarity": true}, ...],
      "forbidden_positions": [45, 37, 21, 53, 12],
      "forbidden_squares": ["F5", "E5", "C5", "G5", "B4"],
      "forbidden_position": 45,
      "forbidden_square": "F5",
      "targets": [{"target_board_pos": 45, "target_square": "F5", "dla_value": 0.0253, "dla_rank": 1, "tautology_score": 0.73}, ...],
      "fire_rate": 0.12,
      "rule_str": "(F3_flipped) AND (G3_not_empty)",
      "influence_score": 1.934183,
      "neuron_f1": 0.9612,
      "rule_source": {"layer": 5, "neuron": 123, "rule_idx": 0}
    }
  ]
}
```

### Join key

**`quadruple_id`** is the canonical join across arms. Every restriction in
`B1.json`, `B2.json`, `B3.json`, `C.json` with the same `quadruple_id`
corresponds to the same source neuron and thus the same matched quadruple.
Do *not* join on `source_neuron` alone — the same neuron can in principle
appear in multiple quadruples (future extension).

Result JSONs embed `condition` and `mode_canonical` fields; the plotter
groups on these automatically.

---

## 8. Interpretation guide

Treat each pairwise contrast as a hypothesis test. Assume curves are
compared at the same step budget.

**Primary effect (B₁ ft vs C ft):**
- If B₁ converges substantially faster or higher → heuristic alignment
  confers a measurable advantage. This is the headline result.

**Factorial decomposition:**
- If B₂ ≈ B₁ and B₃ ≈ C → *consequent* alignment drives the effect.
  The model's write directions are load-bearing; its recognition machinery
  is not (in this data regime).
- If B₃ ≈ B₁ and B₂ ≈ C → *antecedent* alignment drives the effect.
  The model's recognition machinery transfers; its write directions do not.
- If B₂ and B₃ are both intermediate → both factors contribute; report
  their relative magnitudes.

**From-scratch control (X-ft vs X-scratch, per condition):**
- X-ft ≫ X-scratch → the pretrained weights carry transferable structure.
- X-ft ≈ X-scratch → the advantage is the game itself, not the pretraining.

Combining these: the strongest positive result is
**B₁-ft ≫ {B₂-ft, B₃-ft, C-ft, B₁-scratch}** — full alignment beats every
partial or ablated variant.

---

## 9. Known limitations

- **No structural-rule control (Condition A).** The original design envisioned
  a third condition with game-mechanics changes (e.g., "no diagonal flips"),
  but that code is not yet in the repo. The 2×2 stands alone but is
  strengthened by adding A as an anchor once available.
- **Single source rules file.** Results may be sensitive to rule extraction
  parameters (`min_score`, `top_n_influential`, `tree_type`). Reproducing
  across multiple rule extractions would strengthen generalization claims.
- **Monosemanticity assumption.** Extracted rules treat neurons as
  approximately monosemantic. Polysemanticity or feature superposition would
  mean DLA-argmax under-approximates the true consequent set.
- **Firing-rate matching is surface-level.** Two antecedents with the same
  fire rate can still differ in *which* positions they fire at, which
  interacts with the consequent. The `--max-firing-rate-diff` gate catches
  gross mismatches but not finer-grained distributional differences.
- **Tautology threshold is a knob.** Lowering it drops more neurons; raising
  it admits near-trivial restrictions. We default to 0.85 as a pragmatic
  middle ground.

---

## 10. References

- Li et al. (2023), *Emergent World Representations: Exploring a Sequence
  Model Trained on a Synthetic Task*, ICLR.
  <https://arxiv.org/abs/2210.13382>
- Singh et al. (2025), *Automatically Finding Rule-Based Neurons in
  OthelloGPT* (cited informally as "Automatically Finding Rule-Base").
- Hazineh et al. (2023), *Linear Latent World Models in Simple Transformers:
  A Case Study on Othello-GPT*. <https://arxiv.org/abs/2310.07582>
- Karpathy, *minGPT* (model code basis). <https://github.com/karpathy/minGPT>
