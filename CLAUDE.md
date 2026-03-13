# CLAUDE.md

## Project Overview

This is a research project investigating the "emergent world model" claims of OthelloGPT, based on the paper [Emergent World Representations (Li et al., ICLR 2023)](https://arxiv.org/abs/2210.13382). A GPT model is trained to predict legal next moves in Othello from move sequences alone — the original paper argues it learns an internal board representation ("world model"). This project contains experiments that test, challenge, and extend those claims.

The core model code in `mingpt/` is based on [Karpathy's minGPT](https://github.com/karpathy/minGPT) and the reverse engineering utilities in `experiments/reverse_engineering_experiments/OthelloReverseEngineering/` are ported from external repos. **Do not refactor or restructure these third-party directories.**

## Setup

```bash
conda env create -f environment.yml
conda activate othello
python -m ipykernel install --user --name othello --display-name "othello"
mkdir -p ckpts/battery_othello
```

Key dependencies: PyTorch, TransformerLens, NumPy, scikit-learn, matplotlib, pgnparser.

## Architecture

### Model (`mingpt/`)

- **model.py**: GPT transformer (8 layers, 8 heads, 512 embed dim, 2048 MLP dim). Variants: `GPT` (next-move prediction), `GPTforProbing` (returns hidden states at a specific layer), `GPTforIntervention` (two-stage forward for activation patching).
- **trainer.py / probe_trainer.py**: Training loops for the main model and probes respectively. AdamW with cosine LR decay.
- **probe_model.py**: Linear (`BatteryProbeClassification`) and two-layer probes that predict 64 board positions x 3 classes (white/blank/black) from activations.
- **dataset.py**: `CharDataset` — tokenizes move sequences. Vocab: 61 tokens (60 valid board positions + padding). Padding token is -100.

### Data (`data/`)

- **othello.py**: `OthelloBoardState` class (game engine) and `Othello` class (dataset loader). Board: 8x8, values -1/0/+1 (white/empty/black). 60 playable positions (center 4 squares start filled).
- **othello_synthetic/**: ~240 pickle files, each with ~100k random-legal games.
- **othello_championship/**: Championship PGN files.

### Checkpoints (`ckpts/`)

- `gpt_synthetic.ckpt` / `gpt_championship.ckpt`: Pre-trained 8-layer OthelloGPT.
- `battery_othello/`: Probe checkpoints organized by experiment/layer.

## Experiments (`experiments/`)

### Probing Board State (`probe_board_state.ipynb`)
Linear probes recover board state from GPT activations at each layer. Tests whether the model encodes a linear world model in the residual stream.

### Move Legality Analysis (`move_legality_analysis.ipynb`)
Analyzes how often OthelloGPT assigns high probability to illegal moves. Breaks down error rates by game phase, board position, and confidence level.

### Model Size Comparison (`model_size_comparison.py`)
Trains smaller baselines (MLP-small/medium/large, Transformer-2L/4L) and compares them against the full 8-layer model. Tests whether the world model is scale-dependent.

### Mathematical Transformation Experiments (`mathematical_transformation_experiments/`)
Tests whether transformers can learn arbitrary mathematical functions of game sequences, and whether they develop board-state representations while doing so.
- **transforms.py**: Six transform classes of increasing difficulty: `DotProduct`, `MaxProjection`, `ReluFeatures`, `Quadratic`, `Periodic`, `SparseParity`.
- **generate_labels.py**: Applies transforms to game data to produce binary classification labels.
- **train_boolean_classifier.py**: Trains a GPT classifier on transformed labels.
- **train_state_predictor.py**: Regression variant — predicts sum of a random state vector.
- **probe_board_state.py**: Probes the trained classifier's activations for implicit board representations.

### Reverse Engineering Experiments (`reverse_engineering_experiments/`)
Extracts human-readable IF-THEN rules from individual neurons using decision trees, then tests if those rules alone can predict legal moves.
- **extract_rules.py**: Fits decision trees to neuron activations, outputs rules as JSON. Supports ranking neurons by F1 score or influence (DLA + ablation).
- **heuristic_legal_move_predictor.py**: Evaluates extracted rules as a standalone legal-move predictor without the neural network.
- **OthelloReverseEngineering/**: Third-party utilities (~4600 lines) for circuit analysis, feature extraction, and board operations. Do not modify.
- **rules_*.json**: Pre-extracted rule sets with varying parameters (threshold, depth, layers).

## Key Commands

```bash
# Train probes (example: nonlinear probe, layer 6, championship model)
python train_probe_othello.py --layer 6 --twolayer --mid_dim 64 --championship

# Train probes for all layers and configurations
bash produce_probes.sh

# Extract rules from neurons (see extract_rules.py for args)
python experiments/reverse_engineering_experiments/extract_rules.py

# Train mathematical transform classifier
python experiments/mathematical_transformation_experiments/train_boolean_classifier.py
```

## Conventions

- Seed is set to 42 via `mingpt.utils.set_seed(42)` in scripts.
- Probing datasets use 80/20 train/test split.
- Board state encoding: 0=white, 1=blank, 2=black (for probing targets).
- Move encoding: 60 tokens in lexicographic order of board positions, skipping the 4 center squares (D3, D4, E3, E4).
- Training requires GPU(s); default setting uses 8 GPUs with ~12GB each.
