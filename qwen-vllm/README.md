# Qwen vLLM Server Scripts

This folder contains the Slurm scripts used to start the Qwen vLLM server for
the pipeline.

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
