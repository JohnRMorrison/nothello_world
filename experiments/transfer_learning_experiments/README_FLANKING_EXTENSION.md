# Flanking Extension to the 2×2 Factorial (2×3 design)

A follow-up to the bag-of-heuristics 2×2 experiment (see `README_2x2.md`).
The 2×2 measures causal engagement of human-readable IF-THEN rules
extracted from OthelloGPT neurons via decision trees. This extension adds
a **third antecedent type** — the 960 analytically-enumerated flanking
patterns from `hand_crafted_flanking.py` — reusing the existing arms'
aligned/random consequent squares. Old code and runs are untouched;
everything new lives alongside them.

---

## 1. Scientific question

The 2×2 answers: *are extracted bag-of-heuristics rules causally
engaged in the model's next-move prediction?* This extension asks a
cross-family follow-up: *is that engagement better explained by the
bag-of-heuristics rules or by the 960 flanking patterns?*

Flanking patterns form a complete legal-move oracle (any legal Othello
move satisfies at least one pattern). The hand-crafted 960-unit network
built from them achieves ~99.8% per-cell legality accuracy with zero
learned parameters. If the model's internal circuitry more closely
mirrors flanking geometry than bag-of-heuristics rules, then
restrictions whose antecedents are flanking patterns should show a
larger alignment effect than restrictions whose antecedents are
neuron-rule conjunctions — when the **consequents are identical**.

---

## 2. The 2×3 design

We hold the existing 2×2's aligned/random consequent squares fixed
**per quadruple** and add two new arms that swap in a flanking
antecedent:

|                               | Cons: aligned (DLA argmax) | Cons: random |
|-------------------------------|----------------------------|--------------|
| **Ant: heuristic-aligned**    | B₁ *(existing)*            | B₂ *(existing)* |
| **Ant: random** (freq-matched)| B₃ *(existing)*            | C *(existing)* |
| **Ant: flanking**             | **F₁** (new)               | **F₂** (new) |

Each of the K heuristic quadruples is matched to a single flanking
pattern, length-stratified and firing-rate-matched. The row's
aligned-consequent and random-consequent square sets are inherited
verbatim from B₁/B₂ so the only factor that differs across rows is the
antecedent source.

### Contrasts enabled

| Contrast    | Isolates                                                   |
|-------------|------------------------------------------------------------|
| B₁ vs F₁    | **Headline.** Heuristic vs flanking antecedent, same aligned consequent. Which antecedent class does the pretrained circuitry already compute? |
| F₁ vs C     | Full flanking-alignment effect vs null control.            |
| F₁ vs B₃    | Flanking antecedent vs frequency-matched random, both with aligned consequent. |
| B₂ vs F₂    | Negative-control version of the headline (random consequents — expect smaller gap if the effect is real). |
| (B₁ − C) vs (F₁ − C) | **Gap-of-gaps.** Direct magnitude comparison of the two rule families' alignment effects. |

Each arm also runs in `scratch` mode (from-scratch training, same
architecture) so a cross-mode comparison rules out "this arm is just an
easier game" artifacts.

---

## 3. Flanking pattern selection (one per existing quadruple)

For each of the K existing quadruples `q` we pick one flanking pattern
`p` subject to three constraints, applied in order:

1. **Length stratification.** Each row is assigned a *target length*
   from `[1,2,3,4,5,6]` cycling across rows. With K=20 that gives
   roughly 3–4 rows per length bucket. Longer flanks are geometrically
   rarer, so the target length is honored as a hard constraint (we fall
   back to the nearest length only when no disjoint pattern of the
   target length is available).
2. **Firing-rate match.** Within the length bucket, we pick the
   pattern whose empirical firing rate on standard Othello positions
   most closely matches `q.fire_rate_aligned` (the rate of the neuron's
   rule). We flag (but still emit) any row whose |diff| exceeds
   `--max-firing-rate-diff` (default 0.05, inherited from
   `run_2x2.sh`).
3. **No-overlap with consequent.** The pattern's target, terminal, and
   opponent cells must all be disjoint from
   `cons_aligned_squares ∪ cons_random_squares` for that row —
   preserving the "consequent never in antecedent" invariant the
   heuristic run enforces.

### Why firing-rate parity is imperfect

The extracted heuristic rules fire at ~10–30% per position. Flanking
patterns span a much wider range: length-1 patterns fire at
~3–8%, length-6 patterns at <0.5%. Matching a heuristic rule with a
firing rate of 0.30 to a same-length flanking pattern is usually
infeasible. We honor stratification and report firing-rate diffs as
a known confound. The gap-of-gaps plot still interprets cleanly —
both B and F arms are compared against the same C control, so the
absolute firing rate cancels in the contrast.

---

## 4. Files added

All under `experiments/transfer_learning_experiments/`.

| File | Purpose |
|------|---------|
| `flanking_rule_adapter.py` | Converts a flanking pattern into the `conditions` / `rule_str` format consumed by `restriction_utils.parse_rule_conditions`. Includes a round-trip self-test (`python flanking_rule_adapter.py`). |
| `augment_configs_with_flanking.py` | Reads an existing run's `configs/manifest.json` + `configs/B{1,2}.json` and emits `F1.json`, `F2.json`, and `flanking_manifest.json`. No pretrained checkpoint required. |
| `run_flanking_extension.sh` | Orchestrator. Requires `BASE_RUN=<existing 2×2 dir>`. Inherits all env-var knobs from `run_2x2.sh`. Skips configs / data / training per-sweep if already present. |
| `plot_2x3.py` | 2×3 grid + headline overlay (B₁ vs F₁ vs C) + gap-of-gaps plot. Imports and reuses `plot_transfer_curves.py` helpers. |
| `zero_shot_flanking_vs_heuristics.py` | Companion comparison with no fine-tuning. Trains a linear heuristic readout on an 80% split; compares it, the 960-flanking predictor, and the pretrained GPT against ground-truth legal moves on the 20% test split. |

### Files reused (unchanged)

- `restriction_utils.py` — rule parsing and restriction evaluation.
- `generate_restricted_games.py` — called unchanged on F1/F2 configs.
- `finetune_and_evaluate.py` — called unchanged on F1/F2 games.
- `hand_crafted_flanking.py` — `enumerate_flanking_patterns`, `HandCraftedFlanking`, `encode_board` imported.
- `plot_transfer_curves.py` — helpers imported by `plot_2x3.py`.
- `build_restriction_configs.py` — **not** modified; its output is consumed.
- `reverse_engineering_experiments/heuristic_legal_move_predictor.py` — helpers imported by the zero-shot companion.

---

## 5. Running it

### Prerequisite

A completed (or at-least-config-built) 2×2 run, e.g.
`runs/2x2_20260415_160147/`. If you don't have one, run `bash run_2x2.sh`
first.

### Full run (matches 2×2 compute budget)

```bash
BASE_RUN=runs/2x2_20260415_160147 bash run_flanking_extension.sh
```

This produces:

```
<BASE_RUN>/
├── configs/
│   ├── B1.json, B2.json, B3.json, C.json   (pre-existing)
│   ├── F1.json, F2.json                    (new)
│   ├── flanking_manifest.json              (new — pattern selections + diffs)
│   └── flanking_fire_rates.json            (cache — safe to delete)
├── data/
│   ├── B1/, B2/, B3/, C/                   (pre-existing)
│   └── F1/, F2/                            (new)
├── results/
│   ├── curves_B*_{ft,scratch}_*.json       (pre-existing)
│   └── curves_F{1,2}_{ft,scratch}_*.json   (new)
└── figures/
    ├── grid_*_2x3.png                      (new)
    ├── headline_B1_vs_F1.png               (new)
    └── gap_comparison.png                  (new)
```

### Smoke test (~15 min on a laptop-class GPU)

```bash
BASE_RUN=runs/2x2_20260417_135000 \
    N_RUNS=1 MAX_STEPS=200 NUM_GAMES=10000 EVAL_GAMES=50 \
    bash run_flanking_extension.sh
```

### Env-var knobs

All flags from `run_2x2.sh` are available; the extension also recognizes:

| Var | Default | Meaning |
|-----|---------|---------|
| `BASE_RUN`           | *(required)* | Path to the existing 2×2 run |
| `MAX_FIRING_RATE_DIFF` | `0.05`     | Soft threshold for pattern fire-rate match |
| `SNAPSHOT_GAMES`     | `200`      | Games used to estimate flanking fire rates |

---

## 6. Zero-shot companion

The fine-tuning extension answers the causal question (does the model
*adapt* faster to flanking-based game rules?). The zero-shot companion
answers the correlational question (do the model's *existing*
predictions track flanking or heuristics more closely?).

```bash
python zero_shot_flanking_vs_heuristics.py \
    --rules ../reverse_engineering_experiments/rules_085_200_2-6.json \
    --ckpt ../../ckpts/gpt_synthetic.ckpt \
    --n-games 1000 \
    --output runs/.../zero_shot_flanking_vs_heuristics.json
```

Reports, on a held-out 20% split of random Othello positions:

- Per-cell accuracy / F1 of the 960-flanking predictor vs ground truth.
- Per-cell accuracy / F1 of a trained bag-of-heuristics linear readout
  (identical to the one in `heuristic_legal_move_predictor.py`).
- Pretrained GPT's per-cell legality (softmax ≥ threshold) and top-1
  legality rate.
- Pairwise cell-level agreement between all three predictors.
- **Error-explanation rate**: on positions where GPT's top-1 move is
  illegal, what fraction of those errors does flanking vs heuristic
  also endorse (by marking the target square as legal)?

If GPT agrees more with flanking than with heuristics at the cell level,
that's a zero-shot analog of the fine-tuning hypothesis. If GPT's errors
are better explained by one predictor than the other, that's a sharper
variant of the same signal.

---

## 7. Interpretation guide

**Fine-tuning (2×3):**

| Observation                              | Interpretation |
|------------------------------------------|---------------|
| F₁-ft gap vs C-ft > B₁-ft gap vs C-ft    | Flanking antecedents are easier for the pretrained model — supports flanking hypothesis. |
| F₁-ft ≈ B₁-ft                            | Both antecedent families are comparably aligned with the model. |
| F₁-ft gap < B₁-ft gap                    | Bag-of-heuristics is the better model of what the model uses. |
| F₁-scratch matches F₁-ft                 | F₁ isn't benefiting from pretraining; the "alignment" isn't specific to the pretrained circuitry. |
| B₁-scratch matches B₁-ft and F₁-scratch matches F₁-ft | Both arms are just "easier games"; neither claim of causal engagement is supported. |

**Zero-shot:**

| Observation                                     | Interpretation |
|-------------------------------------------------|---------------|
| Flanking F1 ≫ Heuristic F1                      | Flanking is a cleaner legal-move oracle (expected by design). |
| GPT ↔ flanking agreement ≫ GPT ↔ heuristic agreement | The pretrained model's cell-level predictions track flanking geometry more closely. |
| flanking_explanation_rate > heuristic_explanation_rate | GPT's illegal-move errors look more like flanking false-positives than heuristic false-positives — supports flanking hypothesis. |

The **cleanest joint signal** — the one that would most strongly support
the flanking hypothesis — is: F₁-ft drops below B₁-ft on the fine-tune
curves AND flanking agreement > heuristic agreement on zero-shot AND
flanking_explanation_rate > heuristic_explanation_rate on errors. All
three failing directions would falsify the hypothesis.

---

## 8. Known limitations

- **Firing-rate mismatch.** See §3. Long-flank patterns fire far below
  typical heuristic-rule rates; the gap-of-gaps contrast partially
  cancels this, but the absolute firing-rate diff per row is a confound
  to disclose in any writeup.
- **One pattern per row.** Unlike the heuristic arm (which draws on 902
  neurons, selecting K=20 by influence), we sample K flanking patterns
  deterministically by stratification + fire-rate proximity. Alternative
  sampling (e.g. random-of-equal-fire-rate) would stress-test the
  robustness of the F₁-vs-B₁ contrast but is not included.
- **Length-3+ patterns are rare.** For K=20 the length-3, 4, 5, 6
  buckets contribute fewer rows than lengths 1–2; the gap between
  heuristics and flanking may be driven primarily by short-flank
  patterns. Stratified analysis of F₁ curves by pattern length is a
  natural follow-up.
- **Zero-shot uses a single linear readout.** The heuristic predictor is
  linear in neuron-firing features. A more expressive readout (e.g. a
  shallow MLP) could close the F1 gap somewhat and change the
  agreement-rate comparison; the default replicates the baseline in
  `heuristic_legal_move_predictor.py`.
