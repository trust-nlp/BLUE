#!/bin/bash
#SBATCH --job-name=qwen3-r3
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=408G
#SBATCH --gres=gpu:8
#SBATCH --time=3-00:00:00
#SBATCH --output=output_qwen_r3_%j.txt
#SBATCH --error=error_qwen_r3_%j.txt

set -euo pipefail

: "${CONDA_SH:?Set CONDA_SH to your conda.sh path.}"
: "${MODEL_PATH:?Set MODEL_PATH to the Qwen model path.}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the head node address.}"

CONDA_ENV="${CONDA_ENV:-vllm_qwen}"
MASTER_PORT="${MASTER_PORT:-29500}"
TP_SIZE="${TP_SIZE:-8}"
PP_SIZE="${PP_SIZE:-4}"
NNODES="${NNODES:-4}"
VLLM_IFACE="${VLLM_IFACE:-eth0}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="$VLLM_IFACE"
export GLOO_SOCKET_IFNAME="$VLLM_IFACE"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
ulimit -l unlimited

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

vllm serve "$MODEL_PATH" \
  --tensor-parallel-size "$TP_SIZE" --pipeline-parallel-size "$PP_SIZE" \
  --nnodes "$NNODES" --node-rank 3 \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --headless \
  --trust-remote-code
