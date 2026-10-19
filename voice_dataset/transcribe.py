"""Transcribe real Mandarin/Japanese recordings to TXT, JSONL and SRT.

Runs a locally downloaded faster-whisper ASR model. NO cloud upload.
Use --language zh or ja for language-specific files, or auto otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from .core import windows, clip_id
from .audio_io import read_mono_pcm16

SUPPORTED = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".mp4"}


def execute(*args):
    return subprocess.run(list(map(str, args)), check=True, text=True,
                          capture_output=True).stdout


def duration(path: Path) -> float:
    return float(execute("ffprobe", "-v", "error", "-show_entries",
                         "format=duration", "-of",
                         "default=noprint_wrappers=1:nokey=1", path).strip())


def srt_time(seconds: float) -> str:
    milliseconds = round(max(0.0, seconds) * 1000)
    h, rem = divmod(milliseconds, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def owned_segment_parts(segment, window, source_duration: float):
    """Keep only words whose midpoints belong to this window's core.

    Word ownership avoids losing short words at window boundaries while
    preventing the same word appearing twice due to overlapped reads.
    """
    words = [w for w in (getattr(segment, "words", None) or [])
             if w.start is not None and w.end is not None
             and w.end > w.start]
    if words:
        kept = []
        for w in words:
            start = window.read_start + float(w.start)
            end = min(source_duration, window.read_start + float(w.end))
            if window.core_start <= (start + end) / 2 < window.core_end:
                kept.append((start, end, str(w.word)))
        if not kept:
            return []
        return [(kept[0][0], kept[-1][1], "".join(w[2] for w in kept).strip())]
    start = window.read_start + float(segment.start)
    end = min(source_duration, window.read_start + float(segment.end))
    if window.core_start <= (start + end) / 2 < window.core_end:
        return [(start, end, str(segment.text).strip())]
    return []


def write_result(rows: list[dict], prefix: Path):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["start"], r["end"]))
    with prefix.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with prefix.with_suffix(".txt").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f'[{srt_time(row["start"])} - {srt_time(row["end"])}] '
                    f'{row["text"]}\n')
    with prefix.with_suffix(".srt").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            f.write(f'{i}\n{srt_time(row["start"])} --> '
                    f'{srt_time(row["end"])}\n{row["text"]}\n\n')


def transcribe_file(model, path: Path, output: Path, args):
    total = duration(path)
    rows = []
    for i, window in enumerate(windows(total, args.window_seconds,
                                       args.overlap_seconds), 1):
        with tempfile.TemporaryDirectory(prefix="asr-window-") as d:
            wav = Path(d) / "audio.wav"
            execute("ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-ss", f"{window.read_start:.3f}", "-i", path,
                    "-t", f"{window.read_end-window.read_start:.3f}",
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav)
            segments, info = model.transcribe(
                read_mono_pcm16(wav), language=None if args.language == "auto" else args.language,
                beam_size=args.beam_size, vad_filter=True,
                word_timestamps=True, condition_on_previous_text=False)
            for segment in segments:
                for start, end, text in owned_segment_parts(segment, window, total):
                    if not text or end <= start:
                        continue
                    rows.append(dict(start=round(start, 3),
                                     end=round(end, 3), text=text,
                                     language=info.language,
                                     language_probability=round(float(info.language_probability), 4),
                                     source=str(path.name)))
        print(f"{path.name}: window {i} done", flush=True)
    write_result(rows, output)
    print(f"Saved {len(rows)} lines: {output}.[jsonl|txt|srt]", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="A file or folder of audio")
    p.add_argument("--output", required=True, help="Transcription output directory")
    p.add_argument("--model", default="small", help="small, medium, large-v3, etc.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--language", default="auto", choices=["auto", "zh", "ja"])
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--window-seconds", type=float, default=120)
    p.add_argument("--overlap-seconds", type=float, default=4)
    args = p.parse_args()
    if args.window_seconds <= 0 or args.overlap_seconds < 0:
        p.error("Invalid window/overlap parameters")
    target = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if target.is_file():
        files = [target]
    elif target.is_dir():
        if target == output or target in output.parents:
            p.error("Output must be outside input folder")
        files = sorted(x for x in target.rglob("*")
                       if x.suffix.lower() in SUPPORTED and x.is_file())
    else:
        p.error("Input does not exist")
    if not files:
        p.error("No input audio files")
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type)
    for f in files:
        key = clip_id(str(f), 0.0, 0.0)[:8]
        transcribe_file(model, f, output / (f.stem + "_" + key), args)


if __name__ == "__main__":
    main()
