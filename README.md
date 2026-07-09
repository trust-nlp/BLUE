# MI Session Processing Pipeline

This repository contains a four-step workflow for turning recorded counseling
sessions into a clean table that can be reviewed or analyzed.

The pipeline was built for Motivational Interviewing (MI) style sessions,
including BMI/BAI sessions and SFAS sessions. It uses speech transcription,
speaker-role labeling, MI coding, and topic segmentation.

No private audio, transcripts, CSV outputs, model outputs, logs, or participant
data should be committed to this repository.

## Final Result

The final output is a cleaned CSV file for each session:

```text
cleaned-final/<session_id>/<session_id>.cleaned_step4.csv
```

Each final CSV contains only these columns:

```text
segment_id, role, behavior, motivation, time, text, topic
```

In plain language, the final file says:

- who is speaking: therapist or patient
- what the patient statement is about
- whether the statement is about avoidance, neutrality, or approach
- whether it refers to the past, present, or future
- what the actual transcript text says
- what broader session topic the segment belongs to

## Overview of the Four Steps

```text
audio files
  -> Step 1 transcript text
  -> Step 2 therapist/patient speaker roles
  -> Step 3 MI-coded conversation segments
  -> Step 4 final cleaned topic-labeled CSV
```

Each step produces the input needed by the next step.

## Step 1: Transcribe Audio

Script:

```bash
sbatch step1.sh
```

Program:

```text
step1.py
```

### Purpose

Step 1 converts audio recordings into text transcripts. It uses WhisperX for
automatic speech recognition. If diarization is enabled, it can also attach
temporary speaker tags such as `[SPEAKER_00]`.

Those temporary speaker tags are not trusted as final speaker labels. They are
only used as rough transcript structure. Step 2 replaces them with meaningful
roles: `therapist` and `patient`.

### Input

Audio files, for example:

```text
*.wav
*.mp3
*.m4a
*.flac
*.mp4
```

The input folder is set in `step1.sh` with the `--audio` argument.

### Default Output

Step 1 now outputs only the transcript text needed by the rest of the pipeline:

```text
step1_output/<dataset>/<session_id>/<session_id>.txt
```

This is enough for Step 2. The pipeline does not need JSON or timestamp CSV
files for normal operation.

### Optional Debug Output

If timestamp-level debugging is needed, run `step1.py` with:

```bash
--save_debug_outputs
```

That optional flag also writes:

```text
<session_id>.json
<session_id>.segments.csv
```

These debug files are not required for the normal Step 1 -> Step 4 workflow.

### Important Setup Note

If diarization is enabled, Step 1 needs a Hugging Face token for pyannote.
Do not put this token in the repository. Set it in the shell before submitting
the job:

```bash
export HF_TOKEN=your_token_here
sbatch step1.sh
```

## Step 2: Identify Therapist and Patient Turns

Script:

```bash
sbatch step2.sh
```

Program:

```text
step2.py
```

### Purpose

Step 2 decides whether each transcript line was spoken by the therapist or the
patient.

This is necessary because automatic speaker labels such as `[SPEAKER_00]` and
`[SPEAKER_01]` are unreliable. A numeric speaker label does not tell us who is
the therapist and who is the patient.

Step 2 uses a language model to label each utterance as:

```text
therapist
patient
```

### Input

Step 1 transcript text files:

```text
step1_output/**/*.txt
```

### Output

One CSV per session:

```text
speaker_corrected_txt/<session_id>/<session_id>.speaker_corrected.csv
```

The main useful columns are:

```text
utt_id, line_no, role, text
```

Meaning:

- `utt_id`: the utterance number within the transcript
- `line_no`: the original line number in the Step 1 text file
- `role`: `therapist` or `patient`
- `text`: the transcript text

### Why This Step Matters

Later MI coding rules treat therapist and patient speech differently. For
example, therapist utterances are not coded for patient motivation. Patient
utterances are.

## Step 3: Create MI-Coded Segments

Script:

```bash
sbatch step3.sh
```

Program:

```text
step3.py
```

### Purpose

Step 3 converts speaker-labeled utterances into MI-coded segments.

It applies a Coding Quick Key:

```text
Behavior:
th  = therapist utterance
TB  = target behavior, such as alcohol use
ALT = alternative reinforcer
OTH = independent reinforcer
nr  = not relevant

Motivation:
- = avoid
= = neutral
+ = approach

Time:
P = past
C = current
F = future
```

### Input

Step 2 speaker-role CSV files:

```text
speaker_corrected_txt/**/*.speaker_corrected.csv
```

### Output

One CSV per session:

```text
speaker_merged_turns/<session_id>/<session_id>.coded_segments.csv
```

The main useful columns are:

```text
segment_id, role, behavior, motivation, time, text
```

### What Step 3 Does

For therapist speech:

- consecutive therapist lines are merged together
- they receive behavior code `th`
- they do not receive motivation or time labels

For patient speech:

- patient utterances are coded for behavior
- patient utterances with `TB`, `ALT`, or `OTH` also receive motivation and time
- if behavior, motivation, or time changes, a new segment starts
- if behavior, motivation, and time stay the same, adjacent patient utterances
  can be merged

### Segmentation Mode

Step 3 supports two modes:

```text
--segmentation_unit none
--segmentation_unit punctuation
```

`none` keeps transcript lines as the base unit.

`punctuation` first splits transcript lines by sentence-ending punctuation,
then codes and merges them.

## Step 4: Add Topic Labels and Write the Final Clean File

Script:

```bash
sbatch step4.sh
```

Program:

```text
step4.py
```

### Purpose

Step 4 adds a broad session topic to each Step 3 segment. It identifies topic
boundaries and then fills the topic label downward until the next topic begins.

For example, once a section starts as `Introduction`, following rows keep that
topic until another topic boundary is detected.

### Input

Step 3 coded segment CSV files:

```text
speaker_merged_turns/**/*.coded_segments.csv
```

### Default Output

Step 4 writes the final cleaned CSV files:

```text
cleaned-final/<session_id>/<session_id>.cleaned_step4.csv
```

The final output columns are:

```text
segment_id, role, behavior, motivation, time, text, topic
```

This is the main result intended for review or downstream analysis.

### Optional Full Debug Output

Step 4 does not write full debug CSVs by default.

If debugging metadata is needed, run `step4.py` with:

```bash
--write_full_output
```

That optional mode writes fuller Step 4 files under:

```text
topic_segments/
```

These full files are not needed for the normal cleaned final output.

## What Should Be Shared on GitHub

This repository is intended to share only the code and instructions.

Safe to share:

```text
README.md
step1.py
step1.sh
step2.py
step2.sh
step3.py
step3.sh
step4.py
step4.sh
qwen-vllm/
.gitignore
```

Do not share:

```text
audio recordings
raw transcripts
intermediate CSV files
final cleaned CSV files
JSON outputs
Slurm logs
Hugging Face tokens
model checkpoints
participant data
```

The `.gitignore` file is configured to prevent common data and output files
from being accidentally committed.

## Running the Full Pipeline

Run the steps in order:

```bash
sbatch step1.sh
sbatch step2.sh
sbatch step3.sh
sbatch step4.sh
```

Before running on a new system, update the paths inside the `.sh` files so they
point to the correct audio folders, output folders, model endpoint, and compute
environment.

The public scripts avoid hard-coding internal IP addresses, node names, and
private filesystem paths. Configure them with environment variables such as:

```bash
export AUDIO_DIR=/path/to/audio
export HF_TOKEN=your_huggingface_token
export MODEL_PATH=/path/to/Qwen-model
export OPENAI_API_BASE=http://your-vllm-host:8000/v1
```

## Starting the Local Qwen Model Server

Steps 2, 3, and 4 use an OpenAI-compatible language-model endpoint. In this
project, that endpoint was served by Qwen through vLLM.

The Slurm scripts used to start that server are in:

```text
qwen-vllm/
```

These scripts are cluster-specific examples. They show a four-node setup for
Qwen, but users should set their own node placement, model paths, conda
environment names, network settings, and Slurm resource requests.

## Short Summary

Step 1 makes transcripts from audio.

Step 2 identifies who is speaking: therapist or patient.

Step 3 turns the conversation into MI-coded segments.

Step 4 adds topic labels and writes the final clean CSV.
