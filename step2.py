#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: relabel transcript speaker roles with an LLM.

This version reads WhisperX plain-text transcript files, removes the existing
speaker tags because they are unreliable, and labels utterances in batches of
10 using only prior context. The next batch receives the previous batch's
corrected labels as context. Each utterance must be assigned to either the
therapist or patient.

Output:
  <base>.speaker_corrected.csv
"""

import argparse
import csv
import fnmatch
import glob
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


PROMPT_VERSION = "speaker_role_txt_v3_binary_previous_context_batch10"
BATCH_SIZE_DEFAULT = 10
ROLE_LABELS = {
    "therapist": "therapist",
    "patient": "patient",
}
VALID_ROLES = set(ROLE_LABELS)
OLD_SPEAKER_RE = re.compile(r"^\s*(?:\[[^\]]+\]\s*)+")
THERAPIST_CUE_RE = re.compile(
    r"\b("
    r"can you|could you|would you|tell me|let'?s|we'?re going to|"
    r"on a scale|how|what|why|when|where|do you|did you|have you|"
    r"are you|is that|does that|feedback|session|goal|goals"
    r")\b",
    re.IGNORECASE,
)
PATIENT_CUE_RE = re.compile(r"\b(i|i'm|i've|i'd|me|my|mine|we|we're|our)\b", re.IGNORECASE)


SYSTEM_PROMPT = (
    "You are an expert clinical-dialog annotation assistant. "
    "You classify each utterance as therapist or patient using only the provided "
    "prior context and current utterances. Return only valid JSON."
)


USER_PROMPT_TEMPLATE = """\
You are correcting speaker labels in a transcript.

Important:
- Use ONLY the previous context and the current 10 utterances below.
- Do NOT use future context.
- Label every current utterance.
- Classify speaker role as therapist or patient; do not invent numeric speaker IDs.

Allowed labels:
- therapist
- patient

General cues:
- Therapists often ask structured questions, summarize, reflect, explain session purpose, introduce feedback, ask permission, or transition topics.
- Patients often answer questions, describe their own drinking/drug use, goals, feelings, history, consequences, or preferences.
- Very short fragments like "Um", "Yeah", "Okay", or broken transcript pieces still must be assigned to the most likely speaker from context.

Previous corrected context:
{previous_context}

Current utterances to label:
{current_utterances}

Return JSON only, no markdown fences, with exactly this shape:
{{
  "utterances": [
    {{
      "utt_id": 0,
      "role": "therapist"
    }}
  ]
}}

Rules:
- Include every current utt_id exactly once.
- role must be therapist or patient.
- Never output uncertain, unknown, unclear, other, or any third speaker class.
"""


def strip_old_speaker(text: str) -> str:
    return OLD_SPEAKER_RE.sub("", text).strip()


def read_txt(path: str) -> List[Dict[str, object]]:
    utterances: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            text = strip_old_speaker(raw)
            if not text:
                continue
            utterances.append({
                "utt_id": len(utterances),
                "line_no": line_no,
                "text": text,
            })
    return utterances


def extract_json(text: str) -> Optional[Dict]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except Exception:
        return None


def fallback_role(text: str, previous_role: Optional[str] = None) -> str:
    stripped = text.strip()
    if stripped.endswith("?") or THERAPIST_CUE_RE.search(stripped):
        return "therapist"
    if PATIENT_CUE_RE.search(stripped):
        return "patient"
    return previous_role if previous_role in VALID_ROLES else "patient"


def normalize_role(role: object) -> Optional[str]:
    if not isinstance(role, str):
        return None
    role = role.strip().lower()
    return role if role in VALID_ROLES else None


def format_previous_context(records: List[Dict[str, object]]) -> str:
    if not records:
        return "(none)"
    lines = []
    for rec in records:
        role = str(rec.get("role", "patient"))
        label = ROLE_LABELS.get(role, ROLE_LABELS["patient"])
        lines.append(f'{rec["utt_id"]}: [{label}] {rec["text"]}')
    return "\n".join(lines)


def format_current_utterances(batch: List[Dict[str, object]]) -> str:
    return "\n".join(f'{row["utt_id"]}: {row["text"]}' for row in batch)


def call_llm(
    client: OpenAI,
    model: str,
    previous_context: List[Dict[str, object]],
    batch: List[Dict[str, object]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Tuple[Optional[Dict], str, float]:
    prompt = USER_PROMPT_TEMPLATE.format(
        previous_context=format_previous_context(previous_context),
        current_utterances=format_current_utterances(batch),
    )
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    latency = round(time.time() - t0, 3)
    raw = response.choices[0].message.content or ""
    return extract_json(raw), raw, latency


def labels_from_parsed(parsed: Optional[Dict], batch: List[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    by_id: Dict[int, Dict[str, object]] = {}
    if isinstance(parsed, dict):
        for item in parsed.get("utterances", []):
            if not isinstance(item, dict):
                continue
            try:
                utt_id = int(item.get("utt_id"))
            except Exception:
                continue
            role = normalize_role(item.get("role"))
            if role is None:
                continue
            by_id[utt_id] = {
                "role": role,
            }

    labels: Dict[int, Dict[str, object]] = {}
    previous_role: Optional[str] = None
    for row in batch:
        utt_id = int(row["utt_id"])
        label = by_id.get(utt_id)
        if label is None:
            label = {
                "role": fallback_role(str(row["text"]), previous_role),
                "fallback": "missing or invalid model label; assigned by fallback heuristic",
            }
        labels[utt_id] = label
        previous_role = str(label["role"])
    return labels


def output_path(txt_path: str, root_dir: str, out_dir: str, mode: str) -> str:
    base = os.path.basename(txt_path)
    stem = base[:-4] if base.lower().endswith(".txt") else base
    if mode == "per_folder":
        parent = os.path.dirname(txt_path)
    elif mode == "flat":
        parent = out_dir
    else:
        rel = os.path.relpath(os.path.dirname(txt_path), root_dir)
        parent = os.path.join(out_dir, rel)
    os.makedirs(parent, exist_ok=True)
    return os.path.join(parent, f"{stem}.speaker_corrected.csv")


def write_outputs(
    out_csv: str,
    input_txt: str,
    model: str,
    utterances: List[Dict[str, object]],
    labels: Dict[int, Dict[str, object]],
    batch_logs: List[Dict[str, object]],
    batch_size: int,
) -> None:
    fieldnames = [
        "input_txt", "model", "prompt_version", "batch_size",
        "utt_id", "line_no", "role", "text", "fallback",
    ]
    tmp = out_csv + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in utterances:
            utt_id = int(row["utt_id"])
            role = str(labels.get(utt_id, {}).get("role", "patient"))
            rec = labels.get(utt_id, {})
            writer.writerow({
                "input_txt": input_txt,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "batch_size": batch_size,
                "utt_id": utt_id,
                "line_no": row["line_no"],
                "role": ROLE_LABELS.get(role, ROLE_LABELS["patient"]),
                "text": row["text"],
                "fallback": rec.get("fallback", ""),
            })
    os.replace(tmp, out_csv)


def filter_files(files: List[str], include: Optional[List[str]], exclude: Optional[List[str]]) -> List[str]:
    if include:
        files = [
            f for f in files
            if any(fnmatch.fnmatch(f, pat) or pat in f for pat in include)
        ]
    if exclude:
        files = [
            f for f in files
            if not any(fnmatch.fnmatch(f, pat) or pat in f for pat in exclude)
        ]
    return files


def limit_files_per_root(files_with_roots: List[Tuple[str, str]], max_files_per_root: int) -> List[Tuple[str, str]]:
    if max_files_per_root <= 0:
        return files_with_roots
    counts: Dict[str, int] = {}
    limited: List[Tuple[str, str]] = []
    for root, path in files_with_roots:
        count = counts.get(root, 0)
        if count >= max_files_per_root:
            continue
        limited.append((root, path))
        counts[root] = count + 1
    return limited


def parse_args():
    parser = argparse.ArgumentParser(description="Relabel txt transcript speaker roles in 10-line batches")
    parser.add_argument(
        "--root_dir",
        nargs="+",
        default=[
            "step1_output/BMI_output",
            "step1_output/SAFS_output",
            "step1_output/BAI",
        ],
        help="One or more directories containing transcript .txt files",
    )
    parser.add_argument("--glob", default="**/*.txt")
    parser.add_argument("--out_dir", default="speaker_corrected_txt")
    parser.add_argument("--out_dir_mode", choices=["per_folder", "central", "flat"], default="central")
    parser.add_argument("--model", required=True, help="Model id served by vLLM/OpenAI-compatible API")
    parser.add_argument("--openai_api_base", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="token-abc123")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--include", nargs="+", default=None)
    parser.add_argument("--exclude", nargs="+", default=["*.speaker_corrected.csv"])
    parser.add_argument(
        "--max_files_per_root",
        type=int,
        default=5,
        help="Maximum txt files to process per root_dir; use 0 to disable the limit",
    )
    parser.add_argument("--range", nargs=2, type=int, default=None, metavar=("START", "END"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    files_with_roots: List[Tuple[str, str]] = []
    for root_dir in args.root_dir:
        pattern = os.path.join(root_dir, args.glob)
        for path in glob.glob(pattern, recursive=True):
            files_with_roots.append((root_dir, path))
    files_with_roots = sorted(files_with_roots, key=lambda x: x[1])
    filtered_files = set(filter_files([path for _, path in files_with_roots], args.include, args.exclude))
    files_with_roots = [(root, path) for root, path in files_with_roots if path in filtered_files]
    files_with_roots = limit_files_per_root(files_with_roots, args.max_files_per_root)
    if args.range:
        files_with_roots = files_with_roots[args.range[0]:args.range[1]]
    root_by_file = {path: root for root, path in files_with_roots}
    files = [path for _, path in files_with_roots]
    if not files:
        raise RuntimeError(f"No transcript txt files found under: {args.root_dir}")

    print(f"[INFO] Found {len(files)} txt files")
    print(f"[INFO] base_url: {args.openai_api_base}")
    print(f"[INFO] model: {args.model}")
    print(f"[INFO] prompt_version: {PROMPT_VERSION}")
    print(f"[INFO] batch_size: {args.batch_size}")
    print(f"[INFO] max_files_per_root: {args.max_files_per_root}")
    for root_dir in args.root_dir:
        root_count = sum(1 for root, _ in files_with_roots if root == root_dir)
        print(f"[INFO] root files: {root_count} {root_dir}")

    client = OpenAI(api_key=args.api_key, base_url=args.openai_api_base)

    for txt_path in files:
        root_dir = root_by_file.get(txt_path, args.root_dir[0])
        out_csv = output_path(txt_path, root_dir, args.out_dir, args.out_dir_mode)
        if args.skip_existing and os.path.exists(out_csv):
            print(f"[SKIP] {txt_path}")
            continue

        print(f"\n[FILE] {txt_path}")
        utterances = read_txt(txt_path)
        if not utterances:
            print("[WARN] no non-empty utterances")
            continue

        labels: Dict[int, Dict[str, object]] = {}
        batch_logs: List[Dict[str, object]] = []
        previous_batch_records: List[Dict[str, object]] = []

        for start in range(0, len(utterances), args.batch_size):
            batch = utterances[start:start + args.batch_size]
            batch_id = len(batch_logs)
            try:
                parsed, raw_output, latency = call_llm(
                    client=client,
                    model=args.model,
                    previous_context=previous_batch_records,
                    batch=batch,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                parsed_ok = parsed is not None
                print(
                    f"[BATCH] {batch_id} utt={batch[0]['utt_id']}-{batch[-1]['utt_id']} "
                    f"latency={latency}s parsed_ok={parsed_ok}"
                )
            except Exception as exc:
                parsed = None
                raw_output = repr(exc)
                latency = 0.0
                parsed_ok = False
                print(f"[ERR ] batch={batch_id} {exc}")

            batch_labels = labels_from_parsed(parsed, batch)
            labels.update(batch_labels)
            previous_batch_records = [
                {
                    "utt_id": row["utt_id"],
                    "text": row["text"],
                    "role": batch_labels[int(row["utt_id"])]["role"],
                }
                for row in batch
            ]
            batch_logs.append({
                "batch_id": batch_id,
                "start_utt_id": batch[0]["utt_id"],
                "end_utt_id": batch[-1]["utt_id"],
                "latency_sec": latency,
                "parsed_ok": parsed_ok,
                "parsed_json": parsed,
                "raw_output": raw_output,
            })

        write_outputs(
            out_csv=out_csv,
            input_txt=txt_path,
            model=args.model,
            utterances=utterances,
            labels=labels,
            batch_logs=batch_logs,
            batch_size=args.batch_size,
        )
        print(f"[OUT ] {out_csv}")


if __name__ == "__main__":
    main()
