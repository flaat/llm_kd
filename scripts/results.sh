#!/usr/bin/env bash
set -euo pipefail

# Edit these lists as needed
DATASETS=(
    "adult"
    #"titanic"
    #"california"
    #"diabetes"
)
WORKER_MODELS=(
    "unsloth_qwen_0.5B"
    "unsloth_qwen3_0.6B"
    "unsloth_llama_1B-Instruct"
    "unsloth_deepseek_r1_qwen_1.5B"
    "unsloth_qwen3_1.7B"
    "unsloth_llama_3B-Instruct"
    "unsloth_qwen_3B"
)
REFINER_MODELS=(
    "unsloth_qwen_0.5B"
    "unsloth_qwen3_0.6B"
    "unsloth_llama_1B-Instruct"
    "unsloth_deepseek_r1_qwen_1.5B"
    "unsloth_qwen3_1.7B"
    "unsloth_llama_3B-Instruct"
    "unsloth_qwen_3B"
)

MAX_EXAMPLES=200

# Single-run / loop options
REFINER=0                # 1 to enable refiner
WORKER_FINETUNED=1       # applies to all worker runs
REFINER_FINETUNED=0      # applies to all refiner runs

# Override to run exactly one pair instead of loops (leave empty to use loops)
DATASET_NAME_OVERRIDE=""
WORKER_MODEL_OVERRIDE=""
REFINER_MODEL_OVERRIDE=""

RUN_VALIDATION=0      # set to 1 to run validation mode
OUTPUT_DIR=""         # optional output dir for validation mode

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

if [[ $RUN_VALIDATION -eq 1 ]]; then
  if [[ ${#WORKER_MODELS[@]} -eq 0 ]]; then
    echo "No worker models provided for validation." >&2
    exit 1
  fi
  OUT_DIR_ARG=()
  if [[ -n "$OUTPUT_DIR" ]]; then
    OUT_DIR_ARG=(--output-dir "$OUTPUT_DIR")
  fi
  "$PYTHON_BIN" results.py \
    --validation \
    --datasets "${DATASETS[@]}" \
    --models "${WORKER_MODELS[@]}" \
    --max-examples "$MAX_EXAMPLES" \
    "${OUT_DIR_ARG[@]}"
  exit 0
fi

# If overrides are set, run single pair and exit
if [[ -n "$DATASET_NAME_OVERRIDE" && -n "$WORKER_MODEL_OVERRIDE" ]]; then
  REFINER_ARGS=()
  if [[ $REFINER -eq 1 ]]; then
    if [[ -z "$REFINER_MODEL_OVERRIDE" ]]; then
      echo "Set REFINER_MODEL_OVERRIDE when REFINER=1." >&2
      exit 1
    fi
    REFINER_ARGS=(--refiner --refiner-model "$REFINER_MODEL_OVERRIDE")
    if [[ $REFINER_FINETUNED -eq 1 ]]; then
      REFINER_ARGS+=("--refiner-finetuned")
    fi
  fi
  WORKER_FT_ARG=()
  if [[ $WORKER_FINETUNED -eq 1 ]]; then
    WORKER_FT_ARG=(--worker-finetuned)
  fi
  "$PYTHON_BIN" results.py \
    --dataset-name "$DATASET_NAME_OVERRIDE" \
    --worker-model "$WORKER_MODEL_OVERRIDE" \
    --max-examples "$MAX_EXAMPLES" \
    "${WORKER_FT_ARG[@]}" \
    "${REFINER_ARGS[@]}"
  exit 0
fi

# Loop over lists
for DATASET in "${DATASETS[@]}"; do
  for WORKER in "${WORKER_MODELS[@]}"; do
    if [[ $REFINER -eq 1 ]]; then
      for REFINER_MODEL in "${REFINER_MODELS[@]}"; do
        REFINER_ARGS=(--refiner --refiner-model "$REFINER_MODEL")
        if [[ $REFINER_FINETUNED -eq 1 ]]; then
          REFINER_ARGS+=("--refiner-finetuned")
        fi
        WORKER_FT_ARG=()
        if [[ $WORKER_FINETUNED -eq 1 ]]; then
          WORKER_FT_ARG=(--worker-finetuned)
        fi
        "$PYTHON_BIN" results.py \
          --dataset-name "$DATASET" \
          --worker-model "$WORKER" \
          --max-examples "$MAX_EXAMPLES" \
          "${WORKER_FT_ARG[@]}" \
          "${REFINER_ARGS[@]}"
      done
    else
      WORKER_FT_ARG=()
      if [[ $WORKER_FINETUNED -eq 1 ]]; then
        WORKER_FT_ARG=(--worker-finetuned)
      fi
      "$PYTHON_BIN" results.py \
        --dataset-name "$DATASET" \
        --worker-model "$WORKER" \
        --max-examples "$MAX_EXAMPLES" \
        "${WORKER_FT_ARG[@]}"
    fi
  done
done

