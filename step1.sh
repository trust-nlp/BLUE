#!/bin/bash
#SBATCH --job-name=mi_step1_asr
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=0-20:00:00
#SBATCH --output=output_step1_%j.txt
#SBATCH --error=error_step1_%j.txt

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
AUDIO_DIR="${AUDIO_DIR:?Set AUDIO_DIR to the folder or file containing session audio.}"
DATASET_NAME="${DATASET_NAME:-sessions}"
STEP1_OUT_DIR="${STEP1_OUT_DIR:-$PROJECT_DIR/step1_output/$DATASET_NAME}"
WHISPER_ENV="${WHISPER_ENV:-whisperx2}"
: "${HF_TOKEN:?Set HF_TOKEN before submitting this job. It is required for diarization.}"

source activate "$WHISPER_ENV"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}"

cd "$PROJECT_DIR"

echo "=== Step 1: WhisperX ASR + diarization ==="
python -u step1.py \
  --audio "$AUDIO_DIR" \
  --out_dir "$STEP1_OUT_DIR" \
  --model "${WHISPER_MODEL:-large-v3}" \
  --language "${LANGUAGE:-en}" \
  --device "${DEVICE:-cuda}" \
  --compute_type "${COMPUTE_TYPE:-float16}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --diarize \
  --min_speakers "${MIN_SPEAKERS:-2}" \
  --max_speakers "${MAX_SPEAKERS:-2}" \
  --hf_token "$HF_TOKEN"

echo "=== Step 1 complete ==="
