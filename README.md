# Transfer-tasks branch

This is a focused subset of the `othello_world` repository containing only
the code needed to run the two transfer-task experiments shown in **Fig 1e,f**
of the paper.

## What the experiments test

> If [the model] maintained an internal board representation, it should learn
> variants with spatially **coherent** rules faster than variants with spatially
> **incoherent** rules. It should also learn variants that add new squares in
> spatially coherent ways (e.g., a ninth row) faster than variants that add
> them in spatially incoherent ways (e.g., with random neighbors in each
> direction). We found the **opposite** (Fig. 1e,f).

Each task fine-tunes a pretrained Othello-GPT (`ckpts/gpt_synthetic.ckpt`) on
a corrupted variant of Othello and tracks four behaviours over training:

- **LL** — Legal training, Legal test (probability mass on standard Othello legal moves)
- **IL** — Impossible (corrupted) training, Legal test (forgetting of legal moves)
- **LI** — Legal training, Impossible test (mass on the now-illegal cells under corruption)
- **II** — Impossible training, Impossible test (mass on the cells legal under the new variant)

## Tasks

### Task A — `new_squares_experiment.py` (Fig 1f)

Adds eight new squares to the Othello board, each with five new flanking rules.

- **Coherent** (`--condition-id 0`): the eight new squares form a *ninth row*
  (A9–H9). Their five rules use standard Othello geometry extended into the
  new row.
- **Incoherent** (`--condition-id 1`): the eight new squares get random
  neighbors in each direction. Same number of new rules; spatial structure
  destroyed.

The model's vocab is expanded from 61 → 69 (8 new randomly-initialized token
embeddings) for both conditions.

### Task B — `incoherent_rules_experiment.py` (Fig 1e)

Replaces N existing flanking rules with spatially-transformed versions:

- **Coherent** (conditions 0–5): shift opponents and terminal by `(+1, +1)`,
  keeping the target cell fixed. Rules stay locally consistent — they describe
  a flanking line in a shifted location.
- **Incoherent** (conditions 6–11): cross-wire opponents and terminal between
  distant rule pairs. Same target cell, but the line of opponents and terminal
  comes from a spatially unrelated rule.

12 conditions = 6 N values (`[25, 75, 100, 125, 150, 175]`) × {coherent,
incoherent}. The 100-rule condition is the headline result.

## Layout

```
new_squares_experiment.py            Task A
new_squares_experiment.sh            SLURM array job (0–1)
incoherent_rules_experiment.py       Task B
incoherent_rules_experiment.sh       SLURM array job (0–11)
hand_crafted_flanking.py             enumerates 960 flanking patterns
transfer_utils.py                    shared helpers used by Task B:
                                      precompute_pattern_arrays_extended,
                                      generate_games_extended,
                                      collect_three_test_sets,
                                      evaluate_on_test_sets,
                                      build_standard_lpm_test,
                                      prepare_lpm_test, evaluate_lpm,
                                      place_piece_no_flip, build_legal_mask
behavioral_utils.py                  load_model, basic constants
data/othello.py                      OthelloBoardState
mingpt/                              Karpathy's minGPT (model + training)
ckpts/gpt_synthetic.ckpt             pretrained Othello-GPT (97 MB)
plot_fig1ef.py                       reads JSON outputs, plots Fig 1e,f
```

## Running on SLURM

```bash
sbatch --array=0-1  new_squares_experiment.sh        # Task A: coherent vs incoherent
sbatch --array=0-11 incoherent_rules_experiment.sh   # Task B: 6 scales × 2 variants
```

Each condition takes ~6 hours on a single GPU. Outputs land in
`experiments/new_squares/cond_*.json` and
`experiments/incoherent_rules/cond_*.json` respectively. Each JSON contains
the eval schedule (`eval_steps`) and parallel time series for `LL_prob`,
`LL_acc`, `IL_prob`, `IL_acc`, `LI_prob`, `LI_acc`, `std_lpm`, `cor_lpm`.

## Plotting

```bash
python plot_fig1ef.py --output figs/fig1ef.png
```

Two panels:
- **(e)** Coherent vs incoherent rule corruption at `n_rules=100`
  (use `--all-scales` to draw every n_rules level)
- **(f)** Coherent (ninth row) vs incoherent (random neighbors) for new squares

Y-axis is `IL_acc` — the fraction of held-out positions where the model's top-1
predicted move is legal under the corrupted rules. Lower = the model learned
the new rules less.

## Reference (parent repo)

`experiments/experiment_log.txt` in the parent `othello_world` repo
documents the broader sensitivity / parameter-search experiments (section A7)
and the coherent-scale sweep (section A10) that this branch focuses on.
