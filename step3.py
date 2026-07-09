#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: merge therapist utterances and split client turns by coding changes.

Rules:
  - First choose segmentation_unit:
    none = keep one transcript line as one coding unit.
    punctuation = split each line into sentence-like units by punctuation.
  - Therapist utterances are only merged when consecutive.
  - Client utterances are coded with behavior, motivation, and time.
  - Consecutive client utterances are merged only when behavior, motivation,
    and time are all unchanged.

Coding Quick Key:
  Behavior:
    th = therapist utterance
    TB = target behavior (alcohol)
    ALT = alternative reinforcer
    OTH = independent reinforcer
    nr = not relevant
  Motivation:
    - = avoid
    = = neutral
    + = approach
  Time:
    P = past
    C = current
    F = future

Only client utterances with behavior TB, ALT, or OTH receive motivation and
time ratings. Client utterances coded nr have motivation/time set to null.
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


LABEL_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)\s*$")
SENTENCE_RE = re.compile(r"[^.!?。！？;；]+(?:[.!?。！？;；]+|$)")
ROLE_FROM_LABEL = {
    "therapist": "therapist",
    "patient": "patient",
}
LABEL_FROM_ROLE = {
    "therapist": "therapist",
    "patient": "patient",
}
VALID_BEHAVIORS = {"TB", "ALT", "OTH", "nr"}
VALID_MOTIVATIONS = {"-", "=", "+"}
VALID_TIMES = {"P", "C", "F"}

SYSTEM_PROMPT = (
    "You are an expert clinical-dialog coding assistant. "
    "You code client utterances using the supplied Coding Quick Key. "
    "Return only valid JSON."
)

USER_PROMPT_TEMPLATE = """\
Code each CLIENT utterance using this Coding Quick Key.

Behavior:
- TB = target behavior, alcohol use or drinking
- ALT = alternative reinforcer, a healthier or goal-consistent alternative activity/reinforcer
- OTH = independent reinforcer, another valued independent reinforcer not best captured by TB or ALT
- nr = not relevant to behavior/reinforcement content

Motivation:
- "-" = avoid
- "=" = neutral
- "+" = approach

Time:
- P = past
- C = current
- F = future

Important rules:
- Only utterances with behavior TB, ALT, or OTH get motivation and time.
- If behavior is nr, motivation and time must be null.
- Use motivation/time only for the client utterance itself, not for therapist wording.
- Return one code for every utt_id.

Client utterances:
{utterances}

Return JSON only, no markdown fences, with exactly this shape:
{{
  "utterances": [
    {{
      "utt_id": 0,
      "behavior": "TB",
      "motivation": "+",
      "time": "C",
      "evidence": "short reason"
    }}
  ]
}}
"""


def normalize_role(label: str) -> Optional[str]:
    return ROLE_FROM_LABEL.get(label.strip().lower()) or ROLE_FROM_LABEL.get(label.strip())


def parse_corrected_txt(path: str) -> List[Dict[str, object]]:
    utterances: List[Dict[str, object]] = []
    previous_role: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            match = LABEL_RE.match(line)
            if match:
                role = normalize_role(match.group(1)) or previous_role or "patient"
                text = match.group(2).strip()
            else:
                role = previous_role or "patient"
                text = line
            if not text:
                continue
            utterances.append({
                "utt_id": len(utterances),
                "line_no": line_no,
                "role": role,
                "label": LABEL_FROM_ROLE[role],
                "text": text,
            })
            previous_role = role
    return utterances


def parse_corrected_csv(path: str) -> List[Dict[str, object]]:
    utterances: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = normalize_role(str(row.get("role", ""))) or "patient"
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            try:
                utt_id = int(row.get("utt_id", len(utterances)))
            except Exception:
                utt_id = len(utterances)
            try:
                line_no = int(row.get("line_no", utt_id + 1))
            except Exception:
                line_no = utt_id + 1
            utterances.append({
                "utt_id": utt_id,
                "line_no": line_no,
                "role": role,
                "label": LABEL_FROM_ROLE[role],
                "text": text,
            })
    return utterances


def parse_corrected_input(path: str) -> List[Dict[str, object]]:
    if path.lower().endswith(".csv"):
        return parse_corrected_csv(path)
    return parse_corrected_txt(path)


def split_text_by_punctuation(text: str) -> List[str]:
    parts = [match.group(0).strip() for match in SENTENCE_RE.finditer(text)]
    parts = [part for part in parts if part]
    return parts or [text.strip()]


def split_utterances(utterances: List[Dict[str, object]], segmentation_unit: str) -> List[Dict[str, object]]:
    if segmentation_unit == "none":
        for idx, row in enumerate(utterances):
            row["unit_id"] = idx
            row["source_utt_id"] = row["utt_id"]
            row["source_unit_index"] = 0
        return utterances

    split_rows: List[Dict[str, object]] = []
    for row in utterances:
        parts = split_text_by_punctuation(str(row["text"]))
        for unit_index, part in enumerate(parts):
            split_rows.append({
                "utt_id": len(split_rows),
                "unit_id": len(split_rows),
                "source_utt_id": row["utt_id"],
                "source_unit_index": unit_index,
                "line_no": row["line_no"],
                "role": row["role"],
                "label": row["label"],
                "text": part,
            })
    return split_rows


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


def normalize_behavior(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    upper = value.upper()
    if upper in {"TB", "ALT", "OTH"}:
        return upper
    if value.lower() in {"nr", "not relevant", "not_relevant"}:
        return "nr"
    return None


def normalize_motivation(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    aliases = {
        "avoid": "-",
        "negative": "-",
        "neutral": "=",
        "approach": "+",
        "positive": "+",
    }
    value = aliases.get(value, value)
    return value if value in VALID_MOTIVATIONS else None


def normalize_time(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().upper()
    aliases = {
        "PAST": "P",
        "CURRENT": "C",
        "PRESENT": "C",
        "FUTURE": "F",
    }
    value = aliases.get(value, value)
    return value if value in VALID_TIMES else None


def heuristic_code(text: str) -> Dict[str, object]:
    lowered = text.lower()
    if re.search(r"\b(alcohol|drink|drinking|drunk|beer|liquor|vodka|wine)\b", lowered):
        behavior = "TB"
    elif re.search(r"\b(study|class|homework|school|exercise|gym|meeting|work|job|internship|career|goal)\b", lowered):
        behavior = "ALT"
    elif re.search(r"\b(friend|family|hobby|music|art|sport|club|relationship|health)\b", lowered):
        behavior = "OTH"
    else:
        return {
            "behavior": "nr",
            "motivation": None,
            "time": None,
            "evidence": "heuristic: no behavior or reinforcer cue",
        }

    if re.search(r"\b(don't want|do not want|avoid|stop|quit|cut back|less|worry|concern|bad|problem|hard)\b", lowered):
        motivation = "-"
    elif re.search(r"\b(want|like|enjoy|benefit|good|important|need|plan|try|going to|hope)\b", lowered):
        motivation = "+"
    else:
        motivation = "="

    if re.search(r"\b(used to|was|were|did|last|before|previous|past|had)\b", lowered):
        code_time = "P"
    elif re.search(r"\b(will|going to|plan|future|tomorrow|next|later|hope|want to)\b", lowered):
        code_time = "F"
    else:
        code_time = "C"

    return {
        "behavior": behavior,
        "motivation": motivation,
        "time": code_time,
        "evidence": "heuristic keyword fallback",
    }


def format_client_utterances(batch: List[Dict[str, object]]) -> str:
    return "\n".join(f'{row["utt_id"]}: {row["text"]}' for row in batch)


def call_llm(
    client: OpenAI,
    model: str,
    batch: List[Dict[str, object]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Tuple[Optional[Dict], str, float]:
    prompt = USER_PROMPT_TEMPLATE.format(utterances=format_client_utterances(batch))
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


def normalize_code(item: Dict[str, object], text: str) -> Dict[str, object]:
    behavior = normalize_behavior(item.get("behavior"))
    if behavior is None:
        return heuristic_code(text)
    if behavior == "nr":
        return {
            "behavior": "nr",
            "motivation": None,
            "time": None,
            "evidence": str(item.get("evidence") or "").strip(),
        }
    motivation = normalize_motivation(item.get("motivation"))
    code_time = normalize_time(item.get("time"))
    if motivation is None or code_time is None:
        fallback = heuristic_code(text)
        fallback["evidence"] = "invalid motivation/time; " + str(fallback.get("evidence", ""))
        return fallback
    return {
        "behavior": behavior,
        "motivation": motivation,
        "time": code_time,
        "evidence": str(item.get("evidence") or "").strip(),
    }


def codes_from_parsed(parsed: Optional[Dict], batch: List[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    by_id: Dict[int, Dict[str, object]] = {}
    if isinstance(parsed, dict):
        for item in parsed.get("utterances", []):
            if not isinstance(item, dict):
                continue
            try:
                utt_id = int(item.get("utt_id"))
            except Exception:
                continue
            by_id[utt_id] = item

    codes: Dict[int, Dict[str, object]] = {}
    for row in batch:
        utt_id = int(row["utt_id"])
        raw_item = by_id.get(utt_id)
        if raw_item is None:
            code = heuristic_code(str(row["text"]))
            code["evidence"] = "missing model code; " + str(code.get("evidence", ""))
        else:
            code = normalize_code(raw_item, str(row["text"]))
        codes[utt_id] = code
    return codes


def code_client_utterances(
    utterances: List[Dict[str, object]],
    coding_mode: str,
    model: Optional[str],
    openai_api_base: str,
    api_key: str,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Tuple[Dict[int, Dict[str, object]], List[Dict[str, object]]]:
    patient_rows = [row for row in utterances if row["role"] == "patient"]
    codes: Dict[int, Dict[str, object]] = {}
    logs: List[Dict[str, object]] = []

    if coding_mode == "heuristic":
        for row in patient_rows:
            codes[int(row["utt_id"])] = heuristic_code(str(row["text"]))
        return codes, logs

    if not model:
        raise ValueError("--model is required when --coding_mode llm")

    client = OpenAI(api_key=api_key, base_url=openai_api_base)
    for start in range(0, len(patient_rows), batch_size):
        batch = patient_rows[start:start + batch_size]
        batch_id = len(logs)
        try:
            parsed, raw_output, latency = call_llm(
                client=client,
                model=model,
                batch=batch,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            parsed_ok = parsed is not None
            print(
                f"[CODE] batch={batch_id} patient_utt={batch[0]['utt_id']}-{batch[-1]['utt_id']} "
                f"latency={latency}s parsed_ok={parsed_ok}"
            )
        except Exception as exc:
            parsed = None
            raw_output = repr(exc)
            latency = 0.0
            parsed_ok = False
            print(f"[ERR ] coding batch={batch_id} {exc}")

        batch_codes = codes_from_parsed(parsed, batch)
        codes.update(batch_codes)
        logs.append({
            "batch_id": batch_id,
            "start_utt_id": batch[0]["utt_id"],
            "end_utt_id": batch[-1]["utt_id"],
            "latency_sec": latency,
            "parsed_ok": parsed_ok,
            "parsed_json": parsed,
            "raw_output": raw_output,
        })

    return codes, logs


def code_key(role: str, code: Optional[Dict[str, object]]) -> Tuple[object, object, object]:
    if role == "therapist":
        return ("th", None, None)
    if not code:
        return ("nr", None, None)
    return (code.get("behavior"), code.get("motivation"), code.get("time"))


def display_code(role: str, code: Optional[Dict[str, object]]) -> str:
    if role == "therapist":
        return "th"
    behavior, motivation, code_time = code_key(role, code)
    if behavior == "nr":
        return "nr"
    return f"{behavior}{motivation}{code_time}"


def build_segments(utterances: List[Dict[str, object]], codes: Dict[int, Dict[str, object]]) -> List[Dict[str, object]]:
    return build_segments_with_policy(utterances, codes, merge_patient_same_code=True)


def build_segments_with_policy(
    utterances: List[Dict[str, object]],
    codes: Dict[int, Dict[str, object]],
    merge_patient_same_code: bool,
) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    for utt in utterances:
        role = str(utt["role"])
        utt_id = int(utt["utt_id"])
        code = None if role == "therapist" else codes.get(utt_id, heuristic_code(str(utt["text"])))
        key = code_key(role, code)

        can_merge = (
            bool(segments)
            and segments[-1]["role"] == role
            and (
                role == "therapist"
                or (merge_patient_same_code and tuple(segments[-1]["code_key"]) == key)
            )
        )
        if can_merge:
            segments[-1]["text"] = f'{segments[-1]["text"]} {utt["text"]}'.strip()
            segments[-1]["source_utt_ids"].append(utt_id)
            segments[-1]["source_line_nos"].append(utt["line_no"])
            if role == "patient":
                segments[-1]["utterance_codes"].append({
                    "utt_id": utt_id,
                    **code,
                })
            continue

        segment = {
            "segment_id": len(segments),
            "role": role,
            "label": LABEL_FROM_ROLE[role],
            "code": display_code(role, code),
            "code_key": list(key),
            "text": str(utt["text"]),
            "source_utt_ids": [utt_id],
            "source_line_nos": [utt["line_no"]],
        }
        if role == "patient":
            segment["behavior"] = code["behavior"]
            segment["motivation"] = code["motivation"]
            segment["time"] = code["time"]
            segment["utterance_codes"] = [{
                "utt_id": utt_id,
                **code,
            }]
        else:
            segment["behavior"] = "th"
            segment["motivation"] = None
            segment["time"] = None
        segments.append(segment)
    return segments


def output_path(input_path: str, input_dir: str, out_dir: str) -> str:
    rel_dir = os.path.relpath(os.path.dirname(input_path), input_dir)
    parent = os.path.join(out_dir, rel_dir)
    os.makedirs(parent, exist_ok=True)
    base = os.path.basename(input_path)
    stem = base.replace(".speaker_corrected.txt", "")
    stem = stem.replace(".speaker_corrected.csv", "")
    return os.path.join(parent, f"{stem}.coded_segments.csv")


def write_outputs(
    out_csv: str,
    input_txt: str,
    utterances: List[Dict[str, object]],
    segments: List[Dict[str, object]],
    coding_mode: str,
    segmentation_unit: str,
    coding_logs: List[Dict[str, object]],
) -> None:
    fieldnames = [
        "input_file", "coding_mode", "segmentation_unit",
        "segment_id", "role", "code", "behavior", "motivation", "time",
        "text", "source_utt_ids", "source_line_nos", "utterance_codes",
    ]
    tmp = out_csv + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for segment in segments:
            writer.writerow({
                "input_file": input_txt,
                "coding_mode": coding_mode,
                "segmentation_unit": segmentation_unit,
                "segment_id": segment.get("segment_id", ""),
                "role": segment.get("role", ""),
                "code": segment.get("code", ""),
                "behavior": segment.get("behavior", ""),
                "motivation": segment.get("motivation", ""),
                "time": segment.get("time", ""),
                "text": segment.get("text", ""),
                "source_utt_ids": json.dumps(segment.get("source_utt_ids", []), ensure_ascii=False),
                "source_line_nos": json.dumps(segment.get("source_line_nos", []), ensure_ascii=False),
                "utterance_codes": json.dumps(segment.get("utterance_codes", []), ensure_ascii=False),
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


def parse_args():
    parser = argparse.ArgumentParser(description="Split speaker-corrected transcripts by Quick Key coding changes")
    parser.add_argument("--input_dir", default="speaker_corrected_txt")
    parser.add_argument("--glob", default="**/*.speaker_corrected.csv")
    parser.add_argument("--out_dir", default="speaker_merged_turns")
    parser.add_argument(
        "--segmentation_unit",
        choices=["none", "punctuation"],
        default="none",
        help="none keeps transcript lines intact; punctuation splits each line by sentence-ending punctuation before coding",
    )
    parser.add_argument("--coding_mode", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--model", default=None, help="Model id served by vLLM/OpenAI-compatible API")
    parser.add_argument("--openai_api_base", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="token-abc123")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--include", nargs="+", default=None)
    parser.add_argument("--exclude", nargs="+", default=None)
    parser.add_argument("--range", nargs=2, type=int, default=None, metavar=("START", "END"))
    parser.add_argument("--max_files", type=int, default=0, help="Maximum files to process; 0 means no limit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    files = sorted(glob.glob(os.path.join(args.input_dir, args.glob), recursive=True))
    files = filter_files(files, args.include, args.exclude)
    if args.range:
        files = files[args.range[0]:args.range[1]]
    if args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        raise RuntimeError(f"No speaker-corrected csv files found under: {args.input_dir}")

    print(f"[INFO] Found {len(files)} speaker-corrected csv files")
    print(f"[INFO] coding_mode: {args.coding_mode}")
    print(f"[INFO] segmentation_unit: {args.segmentation_unit}")
    for path in files:
        print(f"\n[FILE] {path}")
        out_csv = output_path(path, args.input_dir, args.out_dir)
        raw_utterances = parse_corrected_input(path)
        utterances = split_utterances(raw_utterances, args.segmentation_unit)
        codes, coding_logs = code_client_utterances(
            utterances=utterances,
            coding_mode=args.coding_mode,
            model=args.model,
            openai_api_base=args.openai_api_base,
            api_key=args.api_key,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        segments = build_segments_with_policy(
            utterances=utterances,
            codes=codes,
            merge_patient_same_code=True,
        )
        write_outputs(
            out_csv=out_csv,
            input_txt=path,
            utterances=utterances,
            segments=segments,
            coding_mode=args.coding_mode,
            segmentation_unit=args.segmentation_unit,
            coding_logs=coding_logs,
        )
        print(f"[OUT ] {out_csv}")


if __name__ == "__main__":
    main()
