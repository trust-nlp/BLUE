#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified topic segmentation runner for AAAA-BLUE2+3.

For each Step 3 *.coded_segments.csv file, this program chooses the BMI/BAI or
SFAS topic prompt by path unless --session_type overrides it.
"""

import argparse
import asyncio
import csv
import fnmatch
import glob
import json
import os
import time
import re
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI


DEFAULT_CLEAN_FIELDS = ["segment_id", "role", "behavior", "motivation", "time", "text", "topic"]

BMI_SEGMENT_PROMPT = r"""
You are segmenting a Brief Motivational Intervention (BMI/BAI) transcript.

Input items are Step 3 coded segments in the form:
<segment_id>: <segment text>

Identify the start segment_id for each topic section that is present.
Do not invent missing sections. Use exact segment_id values from the input.

Allowed BMI/BAI sections:
- Introduction
- Decisional Balance Exercise
- Feedback: Drinking Data and BAC
- Feedback: Consequences and Risks
- Feedback Summary
- Envision the Future
- Establish Goals
- Strategies for Reducing Drinking/Drug Use
- Conclusion of Session

Return:
### Reasoning Process
Briefly explain the section boundaries.

### Final JSON
{
  "Introduction": {"start_index": 0, "start_utterance": "..."},
  "Decisional Balance Exercise": {"start_index": 12, "start_utterance": "..."}
}
"""

SFAS_SEGMENT_PROMPT = r"""
You are segmenting a Substance-Free Activity Session (SFAS) transcript.

Input items are Step 3 coded segments in the form:
<segment_id>: <segment text>

Identify the start segment_id for each topic section that is present.
SFAS sections may appear out of order and may appear more than once.
Do not invent missing sections. Use exact segment_id values from the input.

Allowed SFAS sections:
- Introduction
- College/Career Goals
- Alcohol/Drug Impact
- Graduation Benefits
- Career Requirements
- Career Activities
- Time Allocation
- Substance-Free Activities
- Summary/Goal Setting

Return:
### Reasoning Process
Briefly explain the section boundaries.

### Final JSON
[
  {"section": "Introduction", "start_index": 0, "start_utterance": "..."},
  {"section": "College/Career Goals", "start_index": 8, "start_utterance": "..."}
]
"""

SESSION_ALIASES = {
    "bmi": "bmi",
    "bai": "bmi",
    "blue2": "bmi",
    "blue-2": "bmi",
    "blue3": "bmi",
    "blue-3": "bmi",
    "sfas": "sfas",
    "safs": "sfas",
}

STEP1_TYPE_ROOTS = [
    ("sfas", "step1_output/SAFS_output"),
    ("bmi", "step1_output/BMI_output"),
    ("bmi", "step1_output/BAI"),
]


def normalize_session_type(value: str) -> str:
    key = value.strip().lower().replace("_", "-")
    if key == "auto":
        return "auto"
    if key not in SESSION_ALIASES:
        raise ValueError(f"Unknown session type: {value}")
    return SESSION_ALIASES[key]


def infer_session_type(path: str) -> str:
    lowered = path.lower()
    parts = [part for part in lowered.replace("\\", "/").split("/") if part]
    joined = " ".join(parts)
    if any(token in joined for token in ("sfas", "safs")):
        return "sfas"
    if any(token in joined for token in ("bmi", "bai", "blue-2", "blue2", "blue-3", "blue3")):
        return "bmi"
    sample_id = os.path.basename(path)
    for suffix in (".topic_segmentation.csv", ".coded_segments.csv", ".segments.csv"):
        if sample_id.endswith(suffix):
            sample_id = sample_id[: -len(suffix)]
    if sample_id in ("", os.path.basename(path)):
        sample_id = os.path.basename(os.path.dirname(path))
    for session_type, root in STEP1_TYPE_ROOTS:
        if os.path.isdir(os.path.join(root, sample_id)):
            return session_type
    print(f"[WARN] Cannot infer session type from path; using BMI/BAI schema: {path}")
    return "bmi"


def config_for_session_type(session_type: str) -> Dict[str, str]:
    if session_type == "sfas":
        return {"prompt": SFAS_SEGMENT_PROMPT, "json_kind": "list"}
    if session_type == "bmi":
        return {"prompt": BMI_SEGMENT_PROMPT, "json_kind": "dict"}
    raise ValueError(f"Unsupported normalized session type: {session_type}")


def build_transcript(rows: List[Tuple[int, str]]) -> str:
    return "\n".join(f"{idx}: {text}" for idx, text in rows)


def extract_json(text: str, json_kind: str):
    final_json_match = re.search(r"###\s*Final JSON\s*\n", text, re.IGNORECASE)
    search_text = text[final_json_match.end():] if final_json_match else text
    search_text = re.sub(r"^```(?:json)?\s*", "", search_text.lstrip())
    search_text = re.sub(r"\s*```\s*$", "", search_text)

    if json_kind == "list":
        start, end = search_text.find("["), search_text.rfind("]")
    else:
        start, end = search_text.find("{"), search_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(search_text[start:end + 1])
    except Exception:
        return None


def filter_files(files: List[Tuple[str, str]], include: Optional[List[str]], exclude: Optional[List[str]]) -> List[Tuple[str, str]]:
    if include:
        files = [
            item for item in files
            if any(fnmatch.fnmatch(item[1], pat) or pat in item[1] for pat in include)
        ]
    if exclude:
        files = [
            item for item in files
            if not any(fnmatch.fnmatch(item[1], pat) or pat in item[1] for pat in exclude)
        ]
    return files


def collect_csv_files(root_dirs: List[str], glob_pattern: str) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    for root_dir in root_dirs:
        pattern = os.path.join(root_dir, glob_pattern)
        for path in sorted(glob.glob(pattern, recursive=True)):
            files.append((root_dir, path))
    return sorted(files, key=lambda item: item[1])


def output_path_for_csv(csv_path: str, root_dir: str, out_dir_mode: str, out_dir: str) -> str:
    base = os.path.basename(csv_path).replace(".coded_segments.csv", "").replace(".segments.csv", "")
    filename = f"{base}.topic_segmentation.csv"
    if out_dir_mode == "per_folder":
        return os.path.join(os.path.dirname(csv_path), filename)
    if out_dir_mode == "flat":
        return os.path.join(out_dir, filename)
    rel = os.path.relpath(os.path.dirname(csv_path), root_dir)
    return os.path.join(out_dir, rel, filename)


def clean_output_path_for_topic_csv(topic_csv_path: str, out_dir: str, clean_out_dir: str) -> str:
    rel = os.path.relpath(topic_csv_path, out_dir)
    cleaned_name = os.path.basename(rel).replace(".topic_segmentation.csv", ".cleaned_step4.csv")
    rel_parent = os.path.dirname(rel)
    return os.path.join(clean_out_dir, rel_parent, cleaned_name)


def boundaries_from_parsed(parsed) -> List[Dict[str, object]]:
    boundaries: List[Dict[str, object]] = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                start = int(item.get("start_index"))
            except Exception:
                continue
            boundaries.append({
                "start_index": start,
                "topic": item.get("section", ""),
                "topic_start_utterance": item.get("start_utterance", ""),
            })
    elif isinstance(parsed, dict):
        for section, item in parsed.items():
            if not isinstance(item, dict):
                continue
            try:
                start = int(item.get("start_index"))
            except Exception:
                continue
            boundaries.append({
                "start_index": start,
                "topic": section,
                "topic_start_utterance": item.get("start_utterance", ""),
            })
    return sorted(boundaries, key=lambda row: row["start_index"])


def topic_for_index(index: int, boundaries: List[Dict[str, object]]) -> Dict[str, object]:
    current: Optional[Dict[str, object]] = None
    for boundary in boundaries:
        if int(boundary["start_index"]) <= index:
            current = boundary
        else:
            break
    if current is None:
        return {"topic": "", "topic_start_index": "", "topic_start_utterance": ""}
    return {
        "topic": current.get("topic", ""),
        "topic_start_index": current.get("start_index", ""),
        "topic_start_utterance": current.get("topic_start_utterance", ""),
    }


def read_source_csv(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_coded_segments_csv(path: str) -> List[Tuple[int, str]]:
    """
    Return [(segment_id, text), ...] sorted by segment_id.
    The topic model uses these Step 3 segment ids as start_index values.
    """
    rows: List[Tuple[int, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "segment_id" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(f"CSV must contain columns segment_id and text. Got: {reader.fieldnames}")
        for r in reader:
            try:
                segment_id = int(r["segment_id"])
            except Exception:
                continue
            text = r.get("text") or ""
            if text.strip() == "":
                continue
            rows.append((segment_id, text))
    rows.sort(key=lambda x: x[0])
    return rows


def write_topic_csv(
    path: str,
    source_rows: List[Dict[str, object]],
    session_type: str,
    model: str,
    parsed_ok: bool,
    parsed,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    boundaries = boundaries_from_parsed(parsed)
    base_fields = list(source_rows[0].keys()) if source_rows else ["segment_id", "role", "code", "behavior", "motivation", "time", "text"]
    extra_fields = [
        "session_type", "model", "parsed_ok", "topic",
        "topic_start_index", "topic_start_utterance",
    ]
    fieldnames = base_fields + [field for field in extra_fields if field not in base_fields]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in source_rows:
            try:
                segment_id = int(row.get("segment_id", ""))
            except Exception:
                segment_id = -1
            out = dict(row)
            out.update({
                "session_type": session_type,
                "model": model,
                "parsed_ok": parsed_ok,
            })
            out.update(topic_for_index(segment_id, boundaries))
            writer.writerow(out)
    os.replace(tmp, path)


def write_clean_topic_csv_from_rows(
    path: str,
    source_rows: List[Dict[str, object]],
    parsed,
    fields: List[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    boundaries = boundaries_from_parsed(parsed)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in source_rows:
            try:
                segment_id = int(row.get("segment_id", ""))
            except Exception:
                segment_id = -1
            out = dict(row)
            out.update(topic_for_index(segment_id, boundaries))
            writer.writerow({field: out.get(field, "") for field in fields})
    os.replace(tmp, path)


def write_clean_topic_csv(topic_csv_path: str, clean_csv_path: str, fields: List[str]) -> None:
    os.makedirs(os.path.dirname(clean_csv_path), exist_ok=True)
    tmp = clean_csv_path + ".tmp"
    with open(topic_csv_path, "r", encoding="utf-8", newline="") as in_f:
        reader = csv.DictReader(in_f)
        missing = [field for field in fields if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Cannot write cleaned CSV; missing columns {missing} in {topic_csv_path}")
        with open(tmp, "w", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, clean_csv_path)


async def call_chat(
    client: AsyncOpenAI,
    model: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert annotation assistant. "
                    "Follow the output format exactly as specified in the user prompt: "
                    "first output your reasoning under '### Reasoning Process', "
                    "then output the JSON under '### Final JSON'."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return response.choices[0].message.content or ""


async def main_async(args):
    api_key = args.api_key or "token-abc123"
    base_url = args.openai_api_base or "http://localhost:8000/v1"
    session_override = normalize_session_type(args.session_type)

    csv_files = collect_csv_files(args.root_dir, args.glob)
    csv_files = filter_files(csv_files, args.include, args.exclude)
    if args.range:
        start_idx, end_idx = args.range
        csv_files = csv_files[start_idx:end_idx]
    if args.max_files > 0:
        csv_files = csv_files[:args.max_files]
    if not csv_files:
        raise RuntimeError("No CSV files remain after filtering.")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    print(f"[INFO] Found {len(csv_files)} csv files after filtering.")
    print(f"[INFO] base_url: {base_url}")
    print(f"[INFO] models: {args.models}")
    print(f"[INFO] session_type: {args.session_type}")

    for root_dir, csv_path in csv_files:
        session_type = infer_session_type(csv_path) if session_override == "auto" else session_override
        cfg = config_for_session_type(session_type)

        try:
            rows = read_coded_segments_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] Failed reading {csv_path}: {exc}")
            continue

        transcript = build_transcript(rows)
        user_content = (
            cfg["prompt"]
            + "\n\nThe numbered items below are Step 3 coded segments. "
            "Use these segment_id numbers as start_index values in the final JSON."
            + "\n\n--- Coded Segments ---\n"
            + transcript
        )

        full_out_path = output_path_for_csv(csv_path, root_dir, args.out_dir_mode, args.out_dir)
        clean_out_path = clean_output_path_for_topic_csv(full_out_path, args.out_dir, args.clean_out_dir)
        if args.skip_existing and os.path.exists(clean_out_path):
            print(f"[SKIP] {csv_path} (already exists)")
            continue
        source_rows = read_source_csv(csv_path)

        print(f"\n[FILE] {csv_path}")
        print(f"[TYPE] {session_type}")
        if args.write_full_output:
            print(f"[FULL] {full_out_path}")
        print(f"[OUT ] {clean_out_path}")

        for model in args.models:
            t0 = time.time()
            try:
                content = await call_chat(
                    client=client,
                    model=model,
                    user_content=user_content,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                parsed = extract_json(content, cfg["json_kind"])
                parsed_ok = parsed is not None
                if args.write_full_output:
                    write_topic_csv(full_out_path, source_rows, session_type, model, parsed_ok, parsed)
                write_clean_topic_csv_from_rows(clean_out_path, source_rows, parsed, DEFAULT_CLEAN_FIELDS)
                print(f"[DONE] {model} ({round(time.time() - t0, 3)}s) parsed_ok={parsed_ok}")
            except Exception as exc:
                if args.write_full_output:
                    write_topic_csv(full_out_path, source_rows, session_type, model, False, None)
                write_clean_topic_csv_from_rows(clean_out_path, source_rows, None, DEFAULT_CLEAN_FIELDS)
                print(f"[ERR ] {model}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified BMI/BAI/SFAS topic segmentation runner")
    parser.add_argument(
        "--root_dir",
        nargs="+",
        default=[
            "speaker_merged_turns",
        ],
        help="One or more directories containing Step 3 *.coded_segments.csv files",
    )
    parser.add_argument("--glob", default="**/*.coded_segments.csv")
    parser.add_argument(
        "--session_type",
        choices=["auto", "bmi", "bai", "sfas", "safs"],
        default="auto",
        help="auto infers per file from path; bai uses the BMI prompt",
    )
    parser.add_argument(
        "--out_dir_mode",
        choices=["per_folder", "central", "flat"],
        default="central",
        help="per_folder writes next to CSV; flat writes all files in out_dir; central preserves relative structure",
    )
    parser.add_argument("--out_dir", default="topic_segments")
    parser.add_argument("--clean_out_dir", default="cleaned-final")
    parser.add_argument("--write_full_output", action="store_true", help="Also write full Step 4 CSVs under --out_dir")
    parser.add_argument("--models", nargs="+", required=True, help="One or more model ids supported by your vLLM server")
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--api_key", default="", help="vLLM serve --api-key")
    parser.add_argument("--openai_api_base", default="", help="e.g., http://localhost:8000/v1")
    parser.add_argument("--include", nargs="+", default=None, metavar="PATTERN")
    parser.add_argument("--exclude", nargs="+", default=None, metavar="PATTERN")
    parser.add_argument("--range", nargs=2, type=int, default=None, metavar=("START", "END"))
    parser.add_argument("--max_files", type=int, default=0, help="Maximum total files to process; 0 means no limit")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
