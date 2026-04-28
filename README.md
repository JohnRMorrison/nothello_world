# Transfer-tasks branch

This is a focused subset of the `othello_world` repository containing only the code needed to run two transfer-task experiments, described below.

Each task fine-tunes a pretrained Othello-GPT (`ckpts/gpt_synthetic.ckpt`) on a corrupted variant of Othello and tracks the following metrics during training:

Probability mass on cells of each category (averaged over the test set's positions):

- **LL_prob** — cells legal under both standard and corrupted rules
- **IL_prob** — cells illegal under standard rules but legal under corrupted rules 
- **LI_prob** — cells legal under standard rules but illegal under corrupted rules 
- **STD_prob** — cells legal under standard rules
- **COR_prob** — cells legal under corrupted rules



Top-1 accuracy: fraction of positions where the argmax cell is legal under
the corrupted rules. Computed on each test-set's positions:

- **LL_acc**, **IL_acc**, **LI_acc**



## Tasks

### Task A — `new_squares_experiment.py` (Fig 1f)

Adds eight new squares to the Othello board, each with five new flanking rules.

- **Coherent** (`--condition-id 0`): the eight new squares form a *ninth row* (A9–H9). Their five rules use standard Othello geometry extended into the new row.
- **Incoherent** (`--condition-id 1`): the eight new squares get random neighbors in each direction. Same number of new rules, but spatial structure is destroyed.

The model's vocab is expanded from 61 → 69 (8 new randomly-initialized token embeddings) for both conditions.

### Task B — `incoherent_rules_experiment.py` (Fig 1e)

Replaces N existing flanking rules with spatially-transformed versions. The script takes `--variant` and `--n-rules` (default 100):

- **`--variant coherent`**: shift opponents and terminal by `(+1, +1)`,  keeping the target cell fixed. Rules stay locally consistent — they describe a flanking line in a shifted location.
- **`--variant incoherent`**: cross-wire opponents and terminal between  distant rule pairs. Same target cell, but the line of opponents and terminal comes from a spatially unrelated rule.

#### Further experiments

Two additional variants are available for follow-up analyses:

- **`--variant proximal_nonlinear`**: replace each rule's opponents and terminal with random king-move neighbors of the target. Tests proximity without linearity.

- **`--variant distal_linear`**: donate opponents and terminal from a valid flanking line whose target is far away (Manhattan distance ≥ 3). Tests linearity without proximity.

## Layout

```
new_squares_experiment.py            Task A
new_squares_experiment.sh            SLURM array job (0–1)
incoherent_rules_experiment.py       Task B
incoherent_rules_experiment.sh       SLURM submitter; takes <variant> [n_rules]
rules_960.py                         enumerates 960 flanking patterns + board constants
transfer_utils.py                    shared helpers: rule eval / game gen
                                      (precompute_pattern_arrays_extended,
                                      generate_games_extended,
                                      collect_three_test_sets,
                                      evaluate_on_test_sets,
                                      build_standard_lpm_test,
                                      prepare_lpm_test, evaluate_lpm,
                                      place_piece_no_flip), finetune-eval
                                      primitives (build_legal_mask, evaluate),
                                      and load_model / load_shard_games
data/othello.py                      OthelloBoardState
mingpt/                              Karpathy's minGPT (model + training)
ckpts/gpt_synthetic.ckpt             pretrained Othello-GPT (97 MB)
plot_fig1ef.py                       reads JSON outputs, plots Fig 1e,f
```

## Running on SLURM

```bash
# Task A: coherent vs incoherent ninth-row (2 conditions)
sbatch --array=0-1 new_squares_experiment.sh

# Task B: one (variant, n_rules) pair per submission
sbatch incoherent_rules_experiment.sh coherent 100
sbatch incoherent_rules_experiment.sh incoherent 100
# To sweep scales for Fig 1e:
for n in 25 75 100 125 150 175; do
    sbatch incoherent_rules_experiment.sh coherent   $n
    sbatch incoherent_rules_experiment.sh incoherent $n
done
```

Each run takes ~6 hours on a single GPU. Outputs:
- Task A → `experiments/new_squares/cond_*.json`
- Task B → `experiments/incoherent_rules/{variant}_n{n_rules}.json`
  (e.g., `coherent_n100.json`)

Each JSON contains the eval schedule (`eval_steps`) and parallel time series
for `LL_prob`, `LL_acc`, `IL_prob`, `IL_acc`, `LI_prob`, `LI_acc`, `STD_prob`,
`COR_prob`.

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
