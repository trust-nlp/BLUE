# MI Session Processing Pipeline

This repository converts recorded BMI/BAI and SFAS counseling sessions into
clean, session-level CSV files. It contains four processing steps:

```text
audio
  -> Step 1: transcript
  -> Step 2: therapist/patient roles
  -> Step 3: MI-coded segments
  -> Step 4: session components and final clean CSV
```

No participant data, audio, transcripts, generated CSVs, model files, tokens,
or cluster-specific private information should be committed to this repository.

## Final Output

Each session produces:

```text
cleaned-final/<session_id>/<session_id>.cleaned_step4.csv
```

The final columns are:

```text
segment_id, role, behavior, motivation, time, text, session_component
```

- `role`: `therapist` or `patient`
- `behavior`: `th`, `tb`, `alt`, `oth`, or `nr`
- `motivation`: `-` (avoid), `=` (neutral), or `+` (approach)
- `time`: `P` (past), `C` (current), or `F` (future)
- `session_component`: a predefined BMI/BAI or SFAS intervention component

## Requirements

- Python 3
- WhisperX and PyTorch for Step 1
- An OpenAI-compatible language-model endpoint for Steps 2-4
- Slurm if using the included `.sh` job scripts

Install the Python packages required by the scripts in your own environments.
Do not store API tokens or private model paths in this repository.

## Local Model with vLLM

Steps 2-4 can use a locally deployed model through vLLM's OpenAI-compatible
API. Recommended model families for this pipeline are:

- Qwen3.5
- Gemma 4

Use an instruction-tuned checkpoint that fits the available GPUs. For
transcript-only processing, Qwen3.5 can be served in language-model-only mode
to avoid loading unused multimodal components.

Example single-node server:

```bash
vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --language-model-only
```

Then configure the pipeline:

```bash
export MODEL_PATH="/path/to/instruction-model"
export OPENAI_API_BASE="http://localhost:8000/v1"
```

The `qwen-vllm/` directory contains a four-node example with ranks 0-3. Its
default configuration is four nodes, tensor parallel size 8, and pipeline
parallel size 4. Adjust those values for the selected model and hardware.

## Step 1: Transcribe Audio

Purpose: transcribe each recording and optionally identify speakers.

Input:

```text
audio files
```

Default output:

```text
step1_output/<dataset>/<session_id>/<session_id>.txt
```

Run:

```bash
export AUDIO_DIR="/path/to/audio"
export HF_TOKEN="your-hugging-face-token"
export DATASET_NAME="sessions"
sbatch step1.sh
```

Step 1 writes only transcript text by default. Use
`--save_debug_outputs` when JSON and segment-level debug CSVs are needed.

## Step 2: Identify Speaker Roles

Purpose: assign every transcript line to either `therapist` or `patient`.
Unreliable diarization labels such as `SPEAKER_00` are not preserved.

Input:

```text
step1_output/**/*.txt
```

Output:

```text
speaker_corrected_txt/<session_id>/<session_id>.speaker_corrected.csv
```

Run:

```bash
sbatch step2.sh
```

## Step 3: Create MI-Coded Segments

Purpose: code patient speech using the MI Quick Key and create analytically
consistent segments.

- Consecutive therapist speech is merged.
- Patient speech is split whenever behavior, motivation, or time changes.
- Only patient segments coded `tb`, `alt`, or `oth` receive motivation and time
  ratings.

Input:

```text
speaker_corrected_txt/**/*.speaker_corrected.csv
```

Output:

```text
speaker_merged_turns/<session_id>/<session_id>.coded_segments.csv
```

Run:

```bash
export SEGMENTATION_UNIT="punctuation"
sbatch step3.sh
```

`SEGMENTATION_UNIT=none` labels the existing units directly.
`SEGMENTATION_UNIT=punctuation` splits by sentence-ending punctuation, labels
the pieces, and merges adjacent pieces with identical codes.

## Step 4: Assign Session Components

Purpose: identify predefined intervention components and propagate each
component name from its starting `segment_id` to the next component boundary.
This is protocol-component segmentation, not open-ended topic generation.

BMI/BAI and SFAS use separate prompts containing:

- the allowed session components
- full component definitions
- canonical opening patterns
- strict boundary rules

Input:

```text
speaker_merged_turns/**/*.coded_segments.csv
```

Default output:

```text
cleaned-final/<session_id>/<session_id>.cleaned_step4.csv
```

Run:

```bash
sbatch step4.sh
```

The optional `--write_full_output` flag also writes debugging metadata under
`component_segments/`. It is not needed for the normal final output.

## Configuration

The shell scripts use environment variables instead of private hardcoded
paths. Important variables include:

```text
AUDIO_DIR
HF_TOKEN
MODEL_PATH
OPENAI_API_BASE
STEP1_OUT_DIR
STEP2_OUT_DIR
STEP3_OUT_DIR
FINAL_OUT_DIR
```

Review each `.sh` file before submission and adjust Slurm resources,
environments, model paths, and endpoint settings for the local system.
