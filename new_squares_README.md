# New-Squares Transfer Experiment

Does OthelloGPT learn **new** board squares faster when they fit the existing
board geometry (**coherent**) than when they're spatially arbitrary
(**incoherent**)?  We add 8 brand-new squares (a 9th row, cells 64–71) with
hand-made "flanking" rules and fine-tune the base model on games that use them,
measuring how quickly the model starts predicting the new squares as legal.

Game generation is **built in** — the runner generates the shared games and the
matched test manifest itself (nothing to run separately).

## Files

| File | Role |
|---|---|
| `new_squares_data.py` | Model-agnostic **data generation**: games, new-square rules, matched test manifest, and the IL/LL scoring (`score_manifest`). Pure data — no torch. Has its own `main()` if you want to pre-generate. |
| `new_squares_transfer.py` | **GPT runner**: expands vocab 61→69, fine-tunes the base ckpt, evaluates on a schedule, writes `gpt_cond_00X.json`. Calls `generate_condition` from the data module, so it generates games on first run. |
| `new_squares_experiment.sh` | SLURM wrapper (`STAGE=data|train|full`, one array task per condition). |

## Conditions (`--condition-id`)

| id | name | new-square rules |
|----|------|------------------|
| 0 | `coherent` | one coherent into-board flank per new square (the "9th row" analog) |
| 1 | `incoherent` | one random/arbitrary flank per new square (spatially incoherent) |
| 2 | `coherent_all` | **ALL** coherent into-board + along-row geometric relationships per new square |
| 3 | `incoherent_all` | random flanks, count/chain-length **matched** to `coherent_all`'s per-square distribution (unequal in the same distribution) |

Conditions 2/3 are the "all-rules" version added most recently.

## Three controls (all in `new_squares_data.py`, so GPT & MLP share them)

1. **Equal new-square exposure** (`--new-square-legal-budget N`): each condition's
   training set is grown until it holds exactly `N` positions where ≥1 new square
   is legal. A data balance — no loss reweighting.
2. **Length-matched eval**: every test position is tagged with the firing rule's
   chain length, so coherent vs incoherent are compared at equal pattern length.
3. **Matched ceiling**: test positions tagged with `n_legal` so both conditions
   are scored against comparable difficulty.

## Dependencies (already in the repo/fork)

- `mingpt/model.py` (`GPT`, `GPTConfig`), `mingpt/dataset.py` (`CharDataset`)
- `data/othello.py` (`OthelloBoardState`)
- `behavioral_utils.py` (`load_shard_games`) — retention eval only
- Base checkpoint: `ckpts/gpt_synthetic.ckpt`
- Retention data: the synthetic `gen10e5__*.pickle` shards (used by
  `load_shard_games`); skip with `--skip-retention` if unavailable.

## Quickstart

Run one condition end-to-end (generates games, fine-tunes, evaluates):

```bash
# coherent_all, 3 epochs, 1M new-square-legal budget
python new_squares_transfer.py \
  --condition-id 2 --output-dir experiments/new_squares \
  --new-square-legal-budget 1000000 --epochs 3 --lr 5e-5 --bs 16 --seed 42
```

All four conditions:

```bash
for C in 0 1 2 3; do
  python new_squares_transfer.py --condition-id $C \
    --output-dir experiments/new_squares \
    --new-square-legal-budget 1000000 --epochs 3 --seed 42
done
```

On SLURM (`--array=0-3` runs all four conditions, one per array task):

```bash
# stale-proof: generate on CPU, then train on GPU
STAGE=data  sbatch --array=0-3 --gres=gpu:0 --mem=16G new_squares_experiment.sh
STAGE=train sbatch --array=0-3 new_squares_experiment.sh
```

Useful flags: `--skip-retention` (no `gen10e5` shards needed),
`--regenerate-data` (force fresh games), `--ckpt <path>` (different base model).

Pre-generate data only (no GPU):

```bash
python new_squares_data.py --condition-id 2 --output-dir experiments/new_squares \
  --new-square-legal-budget 1000000 --n-test-positions 5000 --seed 42
```

## Output — `experiments/new_squares/gpt_cond_00X.json`

Per-eval-step arrays (index into `eval_steps`), plus run metadata:

| key | meaning |
|---|---|
| `eval_steps` | training step at each eval (0 … `total_steps`) |
| `IL_prob` | **new-square legal probability mass** — the transfer learning curve |
| `IL_acc` | fraction of IL test positions where a new square is top-1 |
| `LL_prob`, `LL_acc` | same metrics on *legacy* (original-board) legal moves |
| `std_lpm`, `std_top` | **retention** on standard games (legal-prob-mass / top-1); `0` if `--skip-retention` |
| `IL_buckets` | IL metrics split by rule chain length |
| `total_steps`, `steps_per_epoch`, `epochs`, `lr`, `bs`, `seed`, `new_square_legal_budget`, `n_train_games` | run config |

`IL_prob` vs `eval_steps` (or vs `eval_steps/total_steps` for a batch-matched
x-axis) is the headline coherent-vs-incoherent curve.

## Reference numbers (all-rules run, epochs=3, budget=1e6)

| condition | IL_acc | IL_prob (final) | LL_prob | std_lpm (retention) |
|---|---|---|---|---|
| `coherent_all` | 0.339 | 0.114 | 0.999 | 0.841 |
| `incoherent_all` | 0.265 | 0.112 | 0.994 | 0.794 |

Coherence helps: +~7 pts IL accuracy and +~5 pts retention.
