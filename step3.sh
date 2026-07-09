#!/bin/bash
#SBATCH --job-name=mi_step3_coding
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00
#SBATCH --output=output_step3_%j.txt
#SBATCH --error=error_step3_%j.txt

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model served by vLLM.}"
OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:8000/v1}"

cd "$PROJECT_DIR"

echo "=== Step 3: merge therapist turns and split client turns by Quick Key changes ==="
python -u step3.py \
  --input_dir "${STEP2_OUT_DIR:-$PROJECT_DIR/speaker_corrected_txt}" \
  --out_dir "${STEP3_OUT_DIR:-$PROJECT_DIR/speaker_merged_turns}" \
  --segmentation_unit "${SEGMENTATION_UNIT:-none}" \
  --coding_mode "${CODING_MODE:-llm}" \
  --model "$MODEL_PATH" \
  --openai_api_base "$OPENAI_API_BASE" \
  --batch_size "${BATCH_SIZE:-20}"

echo "=== Step 3 complete ==="
