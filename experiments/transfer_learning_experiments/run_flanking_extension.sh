#!/usr/bin/env bash
# Extend an existing 2x2 run with F1/F2 flanking-antecedent arms.
#
# Pipeline (mirrors run_2x2.sh, but only for F1 + F2; the B1/B2/B3/C arms
# are assumed to already exist in --base-run):
#
#   1. augment_configs_with_flanking.py  -> configs/F1.json, F2.json
#   2. generate_restricted_games.py x2   -> data/F1/, data/F2/
#   3. finetune_and_evaluate.py x4 sweeps (F1|F2) x (ft|scratch), N_RUNS seeds
#                                         -> results/curves_F{1,2}_{ft,scratch}_*.json
#   4. plot_2x3.py                        -> figures/
#
# All outputs land inside <base-run>/, next to the existing B* artifacts, so
# downstream plotters see a single unified set of curves.
#
# Usage:
#   BASE_RUN=runs/2x2_20260415_160147 bash run_flanking_extension.sh
#   BASE_RUN=runs/2x2_20260415_160147 K=20 N_RUNS=3 bash run_flanking_extension.sh
#
#   # Smoke test (15 min on modest hardware):
#   BASE_RUN=<small_existing_run> N_RUNS=1 MAX_STEPS=200 NUM_GAMES=10000 \
#       EVAL_GAMES=50 bash run_flanking_extension.sh
#
# Resume behavior mirrors run_2x2.sh: configs, data, and results are each
# skipped if already present. Configs are inherited from the B-run; game
# generation and training are incremental per (arm, mode, seed).
#
# Environment overrides (all optional; defaults match run_2x2.sh):
#   BASE_RUN                   (required: path to an existing 2x2 run)
#   CKPT                       (pretrained checkpoint for fine-tuning)
#   N_RUNS, NUM_GAMES          (seeds per sweep, games per condition)
#   MAX_STEPS, EVAL_GAMES      (training steps, eval games per pass)
#   BATCH_SIZE, LR, LR_SCRATCH (training knobs)
#   EVAL_EVERY, EVAL_EVERY_EARLY, EVAL_EARLY_UNTIL (eval schedule)
#   MAX_FIRING_RATE_DIFF       (soft threshold; patterns over this are flagged
#                               but still emitted — see augment script)
#   SNAPSHOT_GAMES             (games used to estimate flanking fire rates)

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

# ---- Required: path to an existing 2x2 run -----------------------------
if [[ -z "${BASE_RUN:-}" ]]; then
    echo "ERROR: BASE_RUN must be set to an existing 2x2 run directory."
    echo "       e.g. BASE_RUN=runs/2x2_20260415_160147 bash $0"
    exit 1
fi

if [[ ! -d "${BASE_RUN}" ]]; then
    echo "ERROR: BASE_RUN directory does not exist: ${BASE_RUN}"
    exit 1
fi

# ---- Inherit knobs from run_2x2.sh defaults ----------------------------
CKPT="${CKPT:-../../ckpts/gpt_synthetic.ckpt}"
N_RUNS="${N_RUNS:-3}"
NUM_GAMES="${NUM_GAMES:-500000}"
MAX_STEPS="${MAX_STEPS:-5000}"
EVAL_GAMES="${EVAL_GAMES:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-3e-4}"
LR_SCRATCH="${LR_SCRATCH:-5e-4}"
EVAL_EVERY="${EVAL_EVERY:-50}"
EVAL_EVERY_EARLY="${EVAL_EVERY_EARLY:-5}"
EVAL_EARLY_UNTIL="${EVAL_EARLY_UNTIL:-200}"
MAX_FIRING_RATE_DIFF="${MAX_FIRING_RATE_DIFF:-0.05}"
SNAPSHOT_GAMES="${SNAPSHOT_GAMES:-200}"

CONFIGS_DIR="${BASE_RUN}/configs"
DATA_DIR="${BASE_RUN}/data"
RESULTS_DIR="${BASE_RUN}/results"
FIGURES_DIR="${BASE_RUN}/figures"
mkdir -p "${CONFIGS_DIR}" "${DATA_DIR}" "${RESULTS_DIR}" "${FIGURES_DIR}"

# ---- Verify base B-run is complete enough to extend --------------------
for F in manifest.json B1.json B2.json; do
    if [[ ! -f "${CONFIGS_DIR}/${F}" ]]; then
        echo "ERROR: base run is missing ${CONFIGS_DIR}/${F}"
        echo "       run_flanking_extension.sh requires a completed config build"
        echo "       (run run_2x2.sh first, or at minimum its config step)."
        exit 1
    fi
done

echo "=============================================================="
echo "Flanking extension (F1, F2) on top of 2x2 run"
echo "  base:           ${BASE_RUN}"
echo "  runs=${N_RUNS}  num_games=${NUM_GAMES}  max_steps=${MAX_STEPS}"
echo "  max_fire_diff=${MAX_FIRING_RATE_DIFF}  snapshot_games=${SNAPSHOT_GAMES}"
echo "=============================================================="

# ---- 1. Build F1 / F2 configs (augment existing manifest) --------------
CONFIGS_EXIST=true
for C in F1 F2; do
    [[ -f "${CONFIGS_DIR}/${C}.json" ]] || CONFIGS_EXIST=false
done

if $CONFIGS_EXIST; then
    echo
    echo "[1/4] F1 / F2 configs already exist — skipping."
else
    echo
    echo "[1/4] Building F1 / F2 configs (flanking antecedents)..."
    python augment_configs_with_flanking.py \
        --base-run "${BASE_RUN}" \
        --snapshot-games "${SNAPSHOT_GAMES}" \
        --max-firing-rate-diff "${MAX_FIRING_RATE_DIFF}"
fi

# ---- 2. Generate restricted games (F1, F2) -----------------------------
echo
echo "[2/4] Generating restricted games for F1 / F2..."
for C in F1 F2; do
    if [[ -d "${DATA_DIR}/${C}" ]]; then
        PICKLE_COUNT=$(find "${DATA_DIR}/${C}" -name '*.pickle' | wc -l | tr -d ' ')
    else
        PICKLE_COUNT=0
    fi
    if [[ "${PICKLE_COUNT}" -gt 0 ]]; then
        echo "  -> condition ${C}: ${PICKLE_COUNT} pickle files found — skipping."
    else
        echo "  -> condition ${C}: generating..."
        python generate_restricted_games.py \
            --config "${CONFIGS_DIR}/${C}.json" \
            --output-dir "${DATA_DIR}/${C}" \
            --num-games "${NUM_GAMES}"
    fi
done

# ---- 3. Finetune + evaluate (F1 / F2 x {ft, scratch} x N_RUNS) ---------
echo
echo "[3/4] Finetuning F1 / F2 (${N_RUNS} seeds per sweep, 4 sweeps)..."
for C in F1 F2; do
    for MODE in ft scratch; do
        LABEL="${C}_${MODE}"
        RESULT_PATTERN="${RESULTS_DIR}/curves_${C}_${MODE}_*.json"
        EXISTING=$(ls ${RESULT_PATTERN} 2>/dev/null | head -1 || true)
        if [[ -n "${EXISTING}" ]]; then
            echo "  -> ${LABEL}: results found ($(basename "${EXISTING}")) — skipping."
            continue
        fi
        echo "  -> ${LABEL}: training..."
        python finetune_and_evaluate.py \
            --games-dir "${DATA_DIR}/${C}" \
            --config "${CONFIGS_DIR}/${C}.json" \
            --label "${LABEL}" \
            --condition "${C}" \
            --mode "${MODE}" \
            --ckpt "${CKPT}" \
            --runs "${N_RUNS}" \
            --max-steps "${MAX_STEPS}" \
            --eval-games "${EVAL_GAMES}" \
            --batch-size "${BATCH_SIZE}" \
            --lr "${LR}" \
            --lr-scratch "${LR_SCRATCH}" \
            --eval-every "${EVAL_EVERY}" \
            --eval-every-early "${EVAL_EVERY_EARLY}" \
            --eval-early-until "${EVAL_EARLY_UNTIL}" \
            --output-dir "${RESULTS_DIR}"
    done
done

# ---- 4. Plot 2x3 ------------------------------------------------------
echo
echo "[4/4] Plotting 2x3 (B1/B2/B3/C + F1/F2)..."
python plot_2x3.py \
    --curves-dir "${RESULTS_DIR}" \
    --out "${FIGURES_DIR}"

echo
echo "Done. Flanking artifacts under: ${BASE_RUN}"
