#!/bin/bash
#SBATCH --job-name=mi_step3_coding
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=output_step3_%j.txt
#SBATCH --error=error_step3_%j.txt

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model served by vLLM.}"
OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:8000/v1}"

cd "$PROJECT_DIR"

# The motivation second pass (MOT_PASSES judgments per TB/ALT/OTH unit, thinking on)
# takes most of the time: about 30 min per session on one server with 3 passes.
# MOT_PASSES=0 skips it (first pass only).
echo "=== Step 3: code client utterances, re-judge motivation, build segments ==="
python -u step3.py \
  --input_dir "${STEP2_OUT_DIR:-$PROJECT_DIR/speaker_corrected_txt}" \
  --out_dir "${STEP3_OUT_DIR:-$PROJECT_DIR/speaker_merged_turns}" \
  --segmentation_unit "${SEGMENTATION_UNIT:-punctuation}" \
  --coding_mode "${CODING_MODE:-llm}" \
  --model "$MODEL_PATH" \
  --openai_api_base "$OPENAI_API_BASE" \
  --batch_size "${BATCH_SIZE:-20}" \
  --context_utts "${CONTEXT_UTTS:-10}" \
  --max_retries "${MAX_RETRIES:-2}" \
  --after_utts "${AFTER_UTTS:-3}" \
  --temperature "${TEMPERATURE:-0}" \
  --mot_passes "${MOT_PASSES:-3}" \
  --mot_before "${MOT_BEFORE:-15}" \
  --mot_after "${MOT_AFTER:-5}" \
  --mot_workers "${MOT_WORKERS:-32}"

echo "=== Step 3 complete ==="
