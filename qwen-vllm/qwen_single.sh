#!/bin/bash
#SBATCH --job-name=qwen-single
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --output=output_qwen_single_%j.txt
#SBATCH --error=error_qwen_single_%j.txt

# Single-node vLLM server for Qwen/Qwen3.8-27B (replaces the 4-node rank0-3
# setup used for Qwen3-235B). Pick the GPU type at submit time, e.g.:
#   sbatch --gres=gpu:h100_80gb:2 qwen_single.sh
#   sbatch --gres=gpu:rtx_6000:4  qwen_single.sh   (also set TP_SIZE=4)
# Steps 2-4 must pass the same MODEL_PATH string as --model / --models.
# To keep the GPUs allocated independently of vLLM (e.g. for debugging), get
# them with `salloc --no-shell ...` and run this script inside the allocation
# (see README.md).

set -euo pipefail

: "${CONDA_SH:?Set CONDA_SH to your conda.sh path.}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model directory, e.g. /path/to/Qwen3.8-27B.}"

CONDA_ENV="${CONDA_ENV:-vllm_qwen}"
TP_SIZE="${TP_SIZE:-${SLURM_GPUS_ON_NODE:-2}}"
PORT="${PORT:-8000}"
# Step 4 sends a whole session (up to ~40k tokens) plus up to 32k output tokens.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# Qwen3.8 thinks by default. The qwen3 parser splits <think> reasoning out of
# message.content so the pipeline's JSON parsing only sees the answer.
# --language-model-only skips the vision encoder (transcripts are text only).
# --enable-prefix-caching reuses the long shared coding prompt across batches.
REASONING_PARSER="${REASONING_PARSER-qwen3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Qwen3.8's linear-attention layers JIT-compile a flashinfer kernel on first
# use and need nvcc from a CUDA toolkit matching torch's CUDA version (12.8).
# Set CUDA_HOME to that toolkit (e.g. a conda cuda-nvcc 12.8 install) if nvcc
# is not already on PATH.
if [ -n "${CUDA_HOME:-}" ]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
fi
# If the node's gcc has no C++ front end (cc1plus), nvcc and flashinfer need
# another g++. CC is left unset on purpose: Triton compiles C launcher stubs
# with it. Set HOST_CXX to such a g++ (e.g. $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
# from conda's gxx_linux-64) if the system g++ cannot be used.
if [ -n "${HOST_CXX:-}" ]; then
  export CXX="$HOST_CXX"
  export NVCC_PREPEND_FLAGS="-ccbin $HOST_CXX ${NVCC_PREPEND_FLAGS:-}"
fi

start_server() {
  local reasoning_args=()
  if [ -n "$REASONING_PARSER" ]; then
    reasoning_args=(--reasoning-parser "$REASONING_PARSER")
  fi
  echo "[INFO] host=$(hostname) port=$PORT model=$MODEL_PATH tp=$TP_SIZE max_len=$MAX_MODEL_LEN"
  echo "[INFO] set OPENAI_API_BASE=http://$(hostname):$PORT/v1 for step2-4"
  # shellcheck disable=SC2086
  vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --trust-remote-code \
    --language-model-only \
    --enable-prefix-caching \
    "${reasoning_args[@]}" \
    $EXTRA_ARGS
}

nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
start_server
