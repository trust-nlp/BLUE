#!/bin/bash
#SBATCH --job-name=mi_step2_roles
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=output_step2_%j.txt
#SBATCH --error=error_step2_%j.txt

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model served by vLLM.}"
OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:8000/v1}"

cd "$PROJECT_DIR"

echo "=== Step 2: LLM speaker relabeling from txt transcripts ==="
python -u step2.py \
  --root_dir \
    "${STEP1_BMI_DIR:-$PROJECT_DIR/step1_output/BMI_output}" \
    "${STEP1_SFAS_DIR:-$PROJECT_DIR/step1_output/SAFS_output}" \
    "${STEP1_BAI_DIR:-$PROJECT_DIR/step1_output/BAI}" \
  --out_dir_mode central \
  --out_dir "${STEP2_OUT_DIR:-$PROJECT_DIR/speaker_corrected_txt}" \
  --model "$MODEL_PATH" \
  --openai_api_base "$OPENAI_API_BASE" \
  --batch_size "${BATCH_SIZE:-10}" \
  --max_files_per_root "${MAX_FILES_PER_ROOT:-5}" \
  --skip_existing

echo "=== Step 2 complete ==="
