# P(C) vs. board-corruption lag cross-correlation — code map & cleanup guide

This README documents **all** the code behind the experiment that measures the
lag cross-correlation between **P(C)** (OthelloGPT's probability on an *illegal*
move `C`) and **board corruption** (the linear probe's cross-entropy loss /
margin), at relative turn offsets **ρ(−1), ρ(0), ρ(+1)**.

It is written so a collaborator can (a) find the code that produced each number,
and (b) know what is core vs. auxiliary vs. safe to delete.

---

## 1. What the experiment measures

For each adversarial position `(game, T, C)` we walk the same-parity turns of the
player who owns `C` and, at each such turn `t`, record `P(C, t)` and a corruption
metric. We then correlate `P(C, t)` with `corruption(t + Δ)`:

- **ρ(−1)** — corruption on the **previous** move predicts P(C).
- **ρ(0)**  — corruption on the **same** move.
- **ρ(+1)** — P(C) predicts corruption on the **next** move.

Reading: **ρ(+1) > ρ(−1)** ⟹ P(C) *leads* corruption (P(C) predicts loss); the
script calls this `S = median[ρ(+1) − ρ(−1)] > 0` = **"bias-first."**

We use the **median** Pearson ρ across positions (each position's series is
short). Report the IQR alongside (the script/notebook do), and Spearman as a
linearity check.

### Headline results (measured; runtime prints, not hardcoded)

Filter: games with **≥5 moves by the relevant player**, n = **18,443**.

| corruption scope | ρ(−1) | ρ(0) | ρ(+1) | reading |
|---|---|---|---|---|
| whole board (`loss_all64`) | −0.100 | +0.419 | +0.771 | P(C) leads |
| C's rays (`loss_ray`) | −0.100 | +0.583 | +0.784 | P(C) leads |
| critical cell (`loss_crit`, n=15,086) | +0.255 | +0.899 | +0.815 | P(C) leads |
| critical cell, probe <0.9 confident (n=5,162) | +0.607 | +0.856 | +0.791 | ρ(0) peaks |

**Conclusion:** P(C) is not well explained by board corruption; if anything the
lead–lag goes the other way (P(C) predicts corruption).

---

## 2. Data-flow / how to reproduce

```
data/othello_synthetic/*.pickle  +  ckpts/gpt_nanda_synthetic.ckpt
        │
        ▼  experiment1_adversarial_rate_by_depth.py   (run once per depth D)
experiment1_by_depth/adversarial_records_depth_DD.npz
        │
        ▼  merge_by_depth_records.py
experiment1_by_depth/adversarial_records.npz     {games, turns, illegal_cells}
        │
        ├───────────────────────────────────────────────┐
        ▼                                                 ▼
experiment_lag_crosscorr.py  (CLI/batch)     notebooks/adversarial_games.ipynb  (FULL)
   → lag_crosscorr.csv, lag_crosscorr.txt        → inline ρ table + figures
```

Both consumers also load `ckpts/gpt_nanda_synthetic.ckpt` and
`mechanistic_interpretability/main_linear_probe.pth`, and import shared helpers
from `experiment_probe_causal_analysis.py` and
`experiments/mathematical_transformation_experiments/probe_state_pred_for_othello.py`.

### Commands

```bash
# 1. Generate adversarial positions (repeat per depth D; example D=10)
python experiment1_adversarial_rate_by_depth.py --depth 10   # -> adversarial_records_depth_10.npz
# 2. Merge all depths
python merge_by_depth_records.py                             # -> experiment1_by_depth/adversarial_records.npz
# 3a. CLI variant (reduced: crit + rays, margin + loss, all/ill slices)
python experiment_lag_crosscorr.py \
    --adversarial-dir experiment1_by_depth \
    --ckpt ckpts/gpt_nanda_synthetic.ckpt \
    --probe mechanistic_interpretability/main_linear_probe.pth \
    --output-csv lag_crosscorr.csv --output-summary lag_crosscorr.txt
# 3b. FULL experiment (all scopes + <0.9 + >=5 filter + the numbers above)
#     Run notebooks/adversarial_games.ipynb  (Section 5 / cells 6, 35, 39)
```

> **Important:** the CLI script `experiment_lag_crosscorr.py` is the **reduced**
> version. The **whole-board scope, the probe<0.9 restriction, the ≥5-move
> filter, and the median-ρ table with the numbers above live ONLY in the
> notebook.** Treat the notebook as the source of record for the paper numbers.

---

## 3. CORE files (this experiment)

| File | Role | In → Out |
|---|---|---|
| `notebooks/adversarial_games.ipynb` | **PRIMARY — full experiment + all figures.** Cell 6 computes per-position `loss_all64`/`loss_ray`/`loss_crit` (+ `p_true_crit`); cell 35 applies `>= 5` filter (n=18,443) and prints the `median_curve` ρ table (board/ray/critical, n=15,086); cell 39 applies the `< 0.9` restriction (n=5,162). Imports `lag_correlations` from `experiment_lag_crosscorr.py`. | npz + ckpt + probe → inline figures/prints |
| `experiment_lag_crosscorr.py` | **CLI/batch variant + home of `lag_correlations()` (line 146)** the notebook imports. Computes per-position lag ρ for `margin_crit, margin_mean, loss_crit, loss_mean` × {`all`,`ill`} slices × {pearson, spearman}, with median/IQR and `S = median[ρ(+1)−ρ(−1)]`. | npz + ckpt + probe → `lag_crosscorr.csv`, `lag_crosscorr.txt` |
| `experiment_probe_causal_analysis.py` | **Core dependency (library here).** Defines the critical cell & C's rays: `next_hand_color_at_turn, flank_providing_directions, critical_errors_for_direction, DIRS, ray_cells_in_direction`. | (imported) |
| `experiments/mathematical_transformation_experiments/probe_state_pred_for_othello.py` | **Core dependency.** `tokenize_games, VOCAB_SIZE, extract_activations`. | (imported) |
| `experiment1_adversarial_rate_by_depth.py` | **Data-gen step 1.** Beam-searches adversarial positions per depth. | synthetic games + ckpt → `adversarial_records_depth_DD.npz` |
| `merge_by_depth_records.py` | **Data-gen step 2.** Merges per-depth files. | `adversarial_records_depth_*.npz` → `adversarial_records.npz` |

### Input artifacts (keep; present on disk)
- `ckpts/gpt_nanda_synthetic.ckpt` — OthelloGPT (mingpt format).
- `mechanistic_interpretability/main_linear_probe.pth` — linear probe, shape `(3,512,8,8,3)`.
- `experiment1_by_depth/adversarial_records.npz` — keys `games`, `turns`, `illegal_cells`.

---

## 4. AUXILIARY files (related adversarial/corruption analyses — NOT this ρ experiment)

Same `adversarial_records.npz` and/or the causal helpers, but a *different*
question. Keep if you keep the broader project; not needed to reproduce the ρ
table.

- `experiment_transition_point.py` → `transition.csv`; `plot_transition_point.py` (plots it) — L→I transition-point analysis.
- `experiment_probe_prob_evolution.py` — move-by-move probability heatmaps.
- `experiment_c_legal_at_bias.py`, `experiment_c_legal_at_error_turn.py` — is C legal under the probe board.
- `experiment_ray_margin_at_bias.py` — ray margins at t_bias.
- `experiment_precedence_aggregate.py`, `experiment_precedence_table.py`, `experiment_within_episode_precedence.py` — bias-vs-corruption precedence.
- `experiment_three_way_classify.py`, `experiment_2x2_classify.py` — position classification.
- `experiment1_any_cell_from_enumeration.py` — opening-coverage claim.
- `experiment_probe_causal_visualize.py`, `plot_actual_vs_probe.py`, `plot_game1_trajectory.py` — single-game / board visualizations.
- `experiment_probe_on_adversarial.py` — probe accuracy adversarial vs control.
- `print_topk_at_turn.py`, `print_mlp_output_at_turn.py` — debug printers.
- `experiment1_adversarial_rate.py` — **alternate** enumeration pipeline writing `experiment1_data/cell_XX.npz` (different from the `by_depth` pipeline used here — do not confuse the two).

---

## 5. Cleanup candidates (safe to delete for THIS experiment)

Not referenced by the ρ pipeline; verify before removing if the wider project needs them.

- `plot_corruption_v2.py` — superseded by `plot_corruption_losses.py` (belongs to the separate MLP-corruption pipeline).
- `prob_evolution_v2.txt` — stale text dump.
- Separate MLP/corruption **game-generation** pipeline (does **not** feed `adversarial_records.npz`): `generate_adversarial_games.py`, `generate_corruption_games.py`, `plot_corruption_losses.py`, and the `*_v2` shell scripts (`corruption_v2_*.sh`, `variant_*_v2*.sh`, `train_gpt_shuffled_v2.*`, `measure_divergence_v2.sh`, `run_bakeoff.sh`, `bakeoff_out/`).
- Root notebooks unrelated to this experiment: `Othello_GPT_Circuits.ipynb`, `heatmap_visual.ipynb`, `intervening_probe_interact_column.ipynb`, `plot_attribution_via_intervention_othello.ipynb`.
- Regenerated outputs (fine to delete; recreated on run): `lag_crosscorr.csv`, `lag_crosscorr.txt`, `transition.csv`.

---

## 6. Methodology notes / limitations (per the median-Pearson question)

- **Median Pearson is reasonable but discards spread.** Each position's ρ is over a *short* same-parity series (often <10 points), so per-position ρ is noisy and undefined under zero variance. The median is robust to that; **always report the IQR** (the script/notebook already do) so the reader sees dispersion, not just the center.
- **Pearson assumes linearity.** `experiment_lag_crosscorr.py` also computes **Spearman** — cite it to show the lead–lag ordering isn't a linearity artifact.
- **Selection.** The critical-cell rows drop the ~18% of positions with no critical cell (n falls 18,443 → 15,086), and the <0.9 row conditions on probe uncertainty (n=5,162) — state both restrictions when quoting those ρ's.
- **Lead–lag ≠ causation.** ρ(+1) > ρ(−1) is consistent with "P(C) precedes corruption," but a shared upstream driver rising over the trajectory could produce the same ordering; the claim is directional-correlational, not causal.

---

## 7. TL;DR for cleanup

Keep: the 6 CORE files in §3 + the 3 input artifacts. The notebook
`notebooks/adversarial_games.ipynb` is the source of record for the reported
numbers; `experiment_lag_crosscorr.py` is its reduced CLI sibling. Everything in
§4 is optional (broader project), and §5 is deletable for this experiment.
