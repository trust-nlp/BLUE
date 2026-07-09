#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
import omegaconf

torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])

import whisperx
from whisperx.diarize import DiarizationPipeline


def export_segments_csv(segments: List[Dict[str, Any]], out_csv: Path) -> None:
    """
    Export WhisperX segments to CSV.
    Each row is a segment with start/end/speaker/text.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "start", "end", "speaker", "text"])
        for i, seg in enumerate(segments):
            w.writerow([
                i,
                float(seg.get("start", 0.0)),
                float(seg.get("end", 0.0)),
                seg.get("speaker", ""),
                (seg.get("text", "") or "").strip()
            ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, required=True,
                    help="Path to an audio file OR a directory containing audio files")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory")
    ap.add_argument("--model", type=str, default="large-v2",
                    help="Whisper model name, e.g., tiny/base/small/medium/large-v2")
    ap.add_argument("--language", type=str, default=None,
                    help="Force language code (e.g., en). If not set, Whisper will detect.")
    ap.add_argument("--device", type=str, default="cuda",
                    help="cuda or cpu")
    ap.add_argument("--compute_type", type=str, default="float16",
                    help="float16 / int8 / int8_float16 depending on your setup")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Larger is faster but needs more VRAM")
    ap.add_argument("--diarize", action="store_true",
                    help="Enable speaker diarization (requires HF token + accepted pyannote agreements)")
    ap.add_argument("--min_speakers", type=int, default=2)
    ap.add_argument("--max_speakers", type=int, default=2)
    ap.add_argument("--hf_token", type=str, default=None,
                    help="Hugging Face token; if not set, will use env HF_TOKEN")
    ap.add_argument("--save_debug_outputs", action="store_true",
                    help="Also save WhisperX JSON and timestamp CSV outputs for debugging")

    args = ap.parse_args()

    audio_path = Path(args.audio)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect audio files
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mp4"}
    if audio_path.is_dir():
        audio_files = sorted([p for p in audio_path.rglob("*") if p.suffix.lower() in exts])
        if not audio_files:
            raise RuntimeError(f"No audio files found in directory: {audio_path}")
    else:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        audio_files = [audio_path]

    # Resolve HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.diarize and not hf_token:
        raise RuntimeError("Diarization requested but no HF token provided. Set HF_TOKEN or pass --hf_token.")

    # Load ASR model once
    device = args.device
    compute_type = args.compute_type

    print(f"[INFO] Loading WhisperX model={args.model} device={device} compute_type={compute_type}")
    model = whisperx.load_model(args.model, device=device, compute_type=compute_type, language=args.language)

    # Optional: diarization pipeline (loaded once)
    diarize_pipeline = None
    if args.diarize:
        print("[INFO] Loading diarization pipeline (pyannote via HF token)")

        diarize_pipeline = DiarizationPipeline(use_auth_token=hf_token, device=device)


    for p in audio_files:
        rel_name = p.stem
        this_out = out_dir / rel_name
        this_out.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] Processing: {p}")
        # 1) ASR
        audio = whisperx.load_audio(str(p))
        result = model.transcribe(audio, batch_size=args.batch_size)

        # 2) Alignment (improves timestamps; also helps diarization assignment quality)
        lang = result.get("language", args.language) or "en"
        print(f"[INFO] Detected language: {lang}")

        print("[INFO] Loading alignment model")
        align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
        print("[INFO] Running alignment")
        result_aligned = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
            return_char_alignments=False
        )

        segments = result_aligned["segments"]

        # 3) Diarization + speaker assignment
        if args.diarize and diarize_pipeline is not None:
            print("[INFO] Running diarization")
            diarize_segments = diarize_pipeline(
                audio,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers
            )
            print("[INFO] Assigning speakers to segments")
            segments = whisperx.assign_word_speakers(diarize_segments, result_aligned)["segments"]

        # 4) Save outputs. The downstream pipeline only needs the plain text transcript.
        out_json = this_out / f"{rel_name}.json"
        out_csv = this_out / f"{rel_name}.segments.csv"
        out_txt = this_out / f"{rel_name}.txt"

        print(f"[INFO] Writing: {out_txt}")
        with out_txt.open("w", encoding="utf-8") as f:
            for seg in segments:
                spk = seg.get("speaker", "")
                line = (seg.get("text", "") or "").strip()
                if spk:
                    f.write(f"[{spk}] {line}\n")
                else:
                    f.write(f"{line}\n")

        if args.save_debug_outputs:
            print(f"[INFO] Writing debug output: {out_json}")
            payload = {
                "audio_path": str(p),
                "language": lang,
                "segments": segments,
            }
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            print(f"[INFO] Writing debug output: {out_csv}")
            export_segments_csv(segments, out_csv)

        print(f"[DONE] {p} -> {this_out}")

    print("\n[ALL DONE]")


if __name__ == "__main__":
    # Better CUDA perf if you want; safe default:
    torch.set_float32_matmul_precision("high")
    main()
