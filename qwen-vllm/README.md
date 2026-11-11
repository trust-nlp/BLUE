# Qwen vLLM Server Scripts

This folder contains the Slurm scripts used to start the Qwen vLLM server for
the pipeline.

## Single node (Qwen3.8-27B)

`qwen_single.sh` serves `Qwen/Qwen3.8-27B` on one node, text only
(`--language-model-only`). Choose the GPUs at submit time and set `TP_SIZE` to
the GPU count:

```bash
export CONDA_SH=/path/to/conda.sh
export MODEL_PATH=/path/to/Qwen3.8-27B
export CUDA_HOME=/path/to/cuda-12.8     # only if nvcc is not on PATH (needed by flashinfer)
export HOST_CXX=/path/to/g++           # only if the system g++ cannot compile CUDA code
sbatch --gres=gpu:h100_80gb:2 qwen_single.sh                 # 2x H100 80GB
TP_SIZE=4 sbatch --gres=gpu:rtx_6000:4 qwen_single.sh         # 4x RTX 6000
```

To keep the GPUs allocated to you independently of vLLM (for debugging),
allocate them with `salloc` and run the script inside the allocation:

```bash
salloc --no-shell --gres=gpu:h100_80gb:2 --cpus-per-task=16 --mem=128G --time=3-00:00:00
srun --jobid=<jobid> --gpus=2 --cpus-per-task=16 bash qwen-vllm/qwen_single.sh   # in tmux
srun --jobid=<jobid> --overlap --pty bash      # extra debug shell next to vLLM
scancel <jobid>                                # release the GPUs
```

Do not start vLLM with `--overlap`: on this cluster (Slurm 24.05) an
`--overlap` step only sees one of the allocated GPUs, so TP=2 fails.

The log prints the `OPENAI_API_BASE` to use for Steps 2-4.
`REASONING_PARSER` (default `qwen3`) moves thinking text out of the answer so
the pipeline's JSON parsing is not affected; set it to `""` to disable.

## Multi node (Qwen3-235B)

The setup uses four nodes:

```text
qwen3_rank0.sh  head node, node rank 0
qwen3_rank1.sh  worker node, node rank 1
qwen3_rank2.sh  worker node, node rank 2
qwen3_rank3.sh  worker node, node rank 3
```

The Step 2, Step 3, and Step 4 scripts call this server through an
OpenAI-compatible API endpoint. Set that endpoint with:

```bash
export OPENAI_API_BASE=http://your-vllm-host:8000/v1
```

Before running these scripts on another cluster, update:

- Slurm partition and node names
- conda environment name
- model path
- network interface names
- master address and port
- tensor and pipeline parallel settings
- time and memory limits

The scripts intentionally do not hard-code internal IP addresses or node names.
Set the required environment variables before submitting:

```bash
export CONDA_SH=/path/to/conda.sh
export CONDA_ENV=vllm_qwen
export MODEL_PATH=/path/to/Qwen-model
export MASTER_ADDR=<head-node-address>
export MASTER_PORT=29500
export VLLM_IFACE=eth0

sbatch qwen3_rank0.sh
sbatch qwen3_rank1.sh
sbatch qwen3_rank2.sh
sbatch qwen3_rank3.sh
```

If your cluster requires specific nodes, pass that through Slurm when
submitting, or add local-only `#SBATCH --nodelist=...` lines that you do not
commit to a public repository.

These scripts do not contain transcript data, participant data, or model
outputs. They are included only to document how the local Qwen model server was
started for this pipeline.
