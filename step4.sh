#!/bin/bash
#SBATCH --job-name=mi_step4_components
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=3-10:00:00
#SBATCH --output=output_step4_%j.txt
#SBATCH --error=error_step4_%j.txt

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model served by vLLM.}"
OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:8000/v1}"

cd "$PROJECT_DIR"

echo "=== Step 4: session-component segmentation and final cleaned output ==="
python -u step4.py \
  --root_dir "${STEP3_OUT_DIR:-$PROJECT_DIR/speaker_merged_turns}" \
  --session_type "${SESSION_TYPE:-auto}" \
  --out_dir_mode central \
  --out_dir "${STEP4_FULL_OUT_DIR:-$PROJECT_DIR/component_segments}" \
  --clean_out_dir "${FINAL_OUT_DIR:-$PROJECT_DIR/cleaned-final}" \
  --models "$MODEL_PATH" \
  --openai_api_base "$OPENAI_API_BASE"

echo "=== Step 4 complete ==="
