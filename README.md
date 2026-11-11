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
- `behavior`: `th`, `TB`, `ALT`, `OTH`, `nr`, or `NC` (no code)
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

Start with a smaller instruction-tuned checkpoint and evaluate it on a few
representative sessions. Use a larger checkpoint only if the smaller model does
not label speaker roles, MI codes, or session-component boundaries reliably.
For transcript-only processing, Qwen3.5 can be served in language-model-only
mode to avoid loading unused multimodal components.

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

- Therapist speech is never merged; each therapist transcript line stays its
  own segment.
- Patient speech is coded with the surrounding conversation (therapist and
  patient lines) as read-only context; `CONTEXT_UTTS` (default 10) sets how
  many preceding lines are included before each batch.
- Patient speech is split whenever behavior, motivation, or time changes.
- The coding prompt follows the MISC-BE manual (`MIBE manual.docx`): its
  behavior/motivation/time definitions, decision rules, and examples.
- There is no keyword fallback. Utterances with a missing or invalid model
  code are retried (`MAX_RETRIES`, default 2); if still invalid they are coded
  `NC` with `evidence` starting `model_failed:`.
- Only patient segments coded `tb`, `alt`, or `oth` receive motivation and time
  ratings.
- The first pass codes batches of `BATCH_SIZE` client units (thinking off,
  temperature 0) and also shows `AFTER_UTTS` (default 3) following lines as
  read-only context. The prompt adds rules for deciding motivation from
  context and for client units that are often wrongly coded `nr`.
- Motivation second pass: every unit coded TB/ALT/OTH is judged again on its
  own, only for motivation, with `MOT_BEFORE` (15) lines before and
  `MOT_AFTER` (5) lines after it, thinking on, temperature 0.6. This is done
  `MOT_PASSES` (default 3) times and majority-voted; a tie gives `=`, and a
  unit with no valid answer keeps its first-pass motivation. Behavior and time
  are not changed. The votes are saved next to the output as
  `<session_id>.motivation_votes.json`. Set `MOT_PASSES=0` to skip this pass.
  It takes most of Step 3's time (about 30 minutes per session on one server).

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
sbatch step3.sh
```

`SEGMENTATION_UNIT=punctuation` (default) splits patient lines by sentence-ending
punctuation, labels the pieces, and merges adjacent pieces with identical codes.
Therapist lines are not split.
`SEGMENTATION_UNIT=none` labels the existing transcript lines directly.

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

- Each segment is sent to the model as `<segment_id>: [therapist|patient] <text>`.
- The prompts tell the model that a recording may start mid-session or end
  early. Segments before the first detected component (or all segments, if the
  model output cannot be parsed) get `session_component = Unknown`.
- The model's full response (reasoning and JSON) is saved next to the final
  CSV as `<session_id>.step4_raw.json`.

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
