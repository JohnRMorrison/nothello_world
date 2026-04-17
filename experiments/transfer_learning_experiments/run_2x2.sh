#!/usr/bin/env bash
# Orchestrate the full 2x2 factorial transfer learning experiment.
#
# Pipeline:
#   1. build_restriction_configs.py      -> configs/{B1,B2,B3,C}.json
#   2. generate_restricted_games.py x4   -> data/{B1,B2,B3,C}/
#   3. finetune_and_evaluate.py x8 sweeps x N_RUNS seeds
#                                         -> results/
#   4. plot_transfer_curves.py            -> figures/
#
# Modes:
#   Fresh run (default):   bash run_2x2.sh
#   Resume a partial run:  RESUME=runs/2x2_20260415_160147 bash run_2x2.sh
#
# When resuming, the script detects what's already completed and skips it:
#   - Configs:  skipped if configs/{B1,B2,B3,C}.json all exist
#   - Games:    skipped per-condition if data/{C}/*.pickle files exist
#   - Training: skipped per-sweep if results/curves_{C}_{MODE}_*.json exists
#   - Plotting: always re-run (fast, uses whatever results exist)
#
# Environment overrides (all optional):
#   RESUME (path to existing run dir — enables resume mode)
#   RULES_FILE, CKPT, LAYERS, K, N_RUNS, NUM_GAMES,
#   MAX_STEPS, EVAL_GAMES, BATCH_SIZE, LR, LR_SCRATCH,
#   MIN_CONDITIONS, TAUTOLOGY_THRESHOLD, MAX_FIRING_RATE_DIFF,
#   OUTPUT_ROOT (parent dir for configs/data/results/figures; ignored if RESUME set)

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

# ---- Defaults (override via env) ---------------------------------------
RULES_FILE="${RULES_FILE:-../reverse_engineering_experiments/rules_085_200_2-6.json}"
CKPT="${CKPT:-../../ckpts/gpt_synthetic.ckpt}"
LAYERS="${LAYERS:-2-5}"
K="${K:-20}"
MIN_CONDITIONS="${MIN_CONDITIONS:-2}"
TAUTOLOGY_THRESHOLD="${TAUTOLOGY_THRESHOLD:-0.85}"
MAX_FIRING_RATE_DIFF="${MAX_FIRING_RATE_DIFF:-0.05}"
N_FORBIDDEN="${N_FORBIDDEN:-5}"

N_RUNS="${N_RUNS:-3}"
NUM_GAMES="${NUM_GAMES:-500000}"

MAX_STEPS="${MAX_STEPS:-5000}"
EVAL_GAMES="${EVAL_GAMES:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-3e-4}"
LR_SCRATCH="${LR_SCRATCH:-5e-4}"
EVAL_EVERY="${EVAL_EVERY:-50}"

# ---- Resolve base directory --------------------------------------------
if [[ -n "${RESUME:-}" ]]; then
    BASE="${RESUME}"
    if [[ ! -d "${BASE}" ]]; then
        echo "ERROR: resume directory does not exist: ${BASE}"
        exit 1
    fi
    echo "RESUMING from: ${BASE}"
else
    STAMP="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
    BASE="${OUTPUT_ROOT}/2x2_${STAMP}"
fi

CONFIGS_DIR="${BASE}/configs"
DATA_DIR="${BASE}/data"
RESULTS_DIR="${BASE}/results"
FIGURES_DIR="${BASE}/figures"
mkdir -p "${CONFIGS_DIR}" "${DATA_DIR}" "${RESULTS_DIR}" "${FIGURES_DIR}"

echo "=============================================================="
echo "2x2 factorial transfer learning experiment"
echo "  base:        ${BASE}"
echo "  rules:       ${RULES_FILE}"
echo "  K=${K}  layers=${LAYERS}  runs=${N_RUNS}  n_forbidden=${N_FORBIDDEN}"
echo "  max_steps=${MAX_STEPS}  num_games=${NUM_GAMES}"
echo "=============================================================="

# ---- 1. Build configs --------------------------------------------------
CONFIGS_EXIST=true
for C in B1 B2 B3 C; do
    [[ -f "${CONFIGS_DIR}/${C}.json" ]] || CONFIGS_EXIST=false
done

if $CONFIGS_EXIST; then
    echo
    echo "[1/4] Configs already exist — skipping."
else
    echo
    echo "[1/4] Building restriction configs..."
    python build_restriction_configs.py \
        --rules "${RULES_FILE}" \
        --ckpt "${CKPT}" \
        --layers "${LAYERS}" \
        --K "${K}" \
        --n-forbidden-squares "${N_FORBIDDEN}" \
        --min-conditions "${MIN_CONDITIONS}" \
        --tautology-threshold "${TAUTOLOGY_THRESHOLD}" \
        --max-firing-rate-diff "${MAX_FIRING_RATE_DIFF}" \
        --output-dir "${CONFIGS_DIR}"
fi

# ---- 2. Generate restricted games (4 conditions) ----------------------
echo
echo "[2/4] Generating restricted games..."
for C in B1 B2 B3 C; do
    PICKLE_COUNT=$(find "${DATA_DIR}/${C}" -name '*.pickle' 2>/dev/null | wc -l | tr -d ' ' || echo 0)
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

# ---- 3. Finetune + evaluate (4 conditions x {ft, scratch} x N_RUNS) ---
echo
echo "[3/4] Finetuning (${N_RUNS} seeds per sweep, 8 sweeps)..."
for C in B1 B2 B3 C; do
    for MODE in ft scratch; do
        LABEL="${C}_${MODE}"
        # Check if results already exist for this sweep.
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
            --output-dir "${RESULTS_DIR}"
    done
done

# ---- 4. Plot ----------------------------------------------------------
echo
echo "[4/4] Plotting..."
python plot_transfer_curves.py \
    --curves-dir "${RESULTS_DIR}" \
    --out "${FIGURES_DIR}"

echo
echo "Done. Artifacts under: ${BASE}"
