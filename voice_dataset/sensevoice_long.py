"""Turn authorized single-speaker long recordings into SenseVoice-aligned clips.

VAD: FunASR FSMN-VAD in overlapping 120-second analysis windows.
ASR: FunASR SenseVoiceSmall on <=27-second utterance clips.
Outputs clips/, per-source manifests/, and a combined metadata.jsonl.
No diarization is done: each source MUST contain one speaker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

from .core import add_context, windows
from .sensevoice_smoketest import resolve_ffmpeg

EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".mp4"}
LANG_RE = re.compile(r"<\|(zh|ja|en|yue|ko)\|>")
TAG_RE = re.compile(r"<\|([^|]+)\|>")
DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def ffmpeg_duration(path: Path, executable: str) -> float:
    result = subprocess.run([executable, "-i", str(path)],
                            capture_output=True, text=True, errors="replace")
    match = DURATION_RE.search(result.stderr)
    if not match:
        raise RuntimeError("Cannot read duration from FFmpeg header: " + result.stderr[-400:])
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def extract(executable: str, source: Path, wav: Path,
            start: float, end: float) -> None:
    if end <= start:
        raise ValueError("Invalid clipping bounds")
    result = subprocess.run(
        [executable, "-nostdin", "-v", "error", "-y",
         "-ss", f"{start:.3f}", "-i", str(source),
         "-t", f"{end - start:.3f}", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(wav)],
        capture_output=True, text=True, errors="replace")
    if result.returncode or not wav.is_file() or wav.stat().st_size <= 44:
        raise RuntimeError("FFmpeg export failed: " + result.stderr.strip()[-400:])


def intervals_union(intervals, gap: float = 0.15):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def make_regions_detailed(intervals, min_s: float, max_s: float, join_gap: float):
    """Group neighboring VAD turns, return (start, end, review_reason).

    Short isolated turns are kept for review; long uninterrupted speech is
    hard-cut only into *review* regions and is never silently promoted to
    a training sample. An interval can join a previous interval only when
    its original gap and combined length satisfy the limits.
    """
    if not (0 < min_s < max_s and 0 <= join_gap):
        raise ValueError("Require 0 < min_s < max_s and nonnegative join_gap")
    regions = []
    for start, end in intervals_union(intervals):
        if end - start > max_s:
            cursor = start
            while cursor < end - 1e-3:
                stop = min(cursor + max_s, end)
                regions.append((cursor, stop, "forced_cut"))
                cursor = stop
        elif (regions and regions[-1][2] is None and
              0 <= start - regions[-1][1] <= join_gap and
              end - regions[-1][0] <= max_s):
            prev = regions[-1]
            regions[-1] = (prev[0], end, None)
        else:
            regions.append((start, end, None))
    return [(round(a, 3), round(b, 3),
             reason or ("too_short" if b - a < min_s else None))
            for a, b, reason in regions]


def make_regions(intervals, min_s: float, max_s: float, join_gap: float):
    """Legacy three-tuple API for existing test fixtures."""
    return [(a, b, reason is not None)
            for a, b, reason in make_regions_detailed(intervals, min_s, max_s, join_gap)]


def vad_intervals(path: Path, vad, executable: str, duration: float,
                  window_seconds: float, overlap_seconds: float):
    intervals = []
    chunks = windows(duration, window_seconds, overlap_seconds)
    for i, w in enumerate(chunks, 1):
        with tempfile.TemporaryDirectory(prefix="voxcpm-vad-") as tmp:
            wav = Path(tmp) / "window.wav"
            extract(executable, path, wav, w.read_start, w.read_end)
            result = vad.generate(input=str(wav))
        if not result or not isinstance(result[0].get("value"), list):
            raise RuntimeError(f"Unexpected FSMN-VAD output for window {i}: {str(result)[:300]}")
        for times in result[0]["value"]:
            if not isinstance(times, (list, tuple)) or len(times) != 2:
                continue
            a = w.read_start + float(times[0]) / 1000
            b = w.read_start + float(times[1]) / 1000
            # Assign each moment once; union across adjacent windows afterward.
            a = max(w.core_start, a)
            b = min(w.core_end, b)
            if b > a:
                intervals.append((a, b))
        print(f"[VAD] {path.name} window {i}/{len(chunks)}", flush=True)
    return intervals_union(intervals)


def raw_language(text: str, requested: str) -> str:
    match = LANG_RE.search(text)
    return match.group(1) if match else (requested if requested != "auto" else "unknown")


def source_files(input_path: Path, speaker_id: str | None):
    if input_path.is_file():
        if not speaker_id:
            raise ValueError("--speaker-id is required for a single recording")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", speaker_id):
            raise ValueError("Use an ASCII speaker ID containing letters, digits, _ or -")
        return [(input_path, speaker_id, f"{speaker_id}/{input_path.name}")]
    files = []
    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTS:
            rel = path.relative_to(input_path)
            if len(rel.parts) < 2:
                raise ValueError("Folder input needs raw/<speaker_id>/<recording>")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", rel.parts[0]):
                raise ValueError("Speaker directories must be safe ASCII IDs: " + rel.parts[0])
            files.append((path, rel.parts[0], rel.as_posix()))
    return files


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(path)


def process_recording(path: Path, speaker: str, source: str,
                      output: Path, vad, asr, executable: str, args):
    source_key = hashlib.sha256(source.encode()).hexdigest()[:16]
    manifest = output / "manifests" / f"{source_key}.jsonl"
    if manifest.is_file() and not args.overwrite:
        print("[SKIP] completed recording:", source, flush=True)
        return
    length = ffmpeg_duration(path, executable)
    intervals = vad_intervals(path, vad, executable, length,
                              args.window_seconds, args.overlap_seconds)
    regions = make_regions_detailed(intervals, args.min_seconds,
                                    args.max_seconds, args.join_gap_seconds)
    records = []
    for i, (start, end, boundary_reason) in enumerate(regions, 1):
        # Keep moderate speech-boundary padding, never exceed 29 s ASR limit.
        audio_start = max(0, start - args.padding_seconds)
        audio_end = min(length, end + args.padding_seconds)
        if audio_end - audio_start > 29 and boundary_reason is None:
            boundary_reason = "asr_window_too_long"
        status_for_reason = {
            "too_short": "review_short",
            "forced_cut": "review_forced_cut",
            "asr_window_too_long": "review_asr_window",
        }
        item = dict(speaker_id=speaker, source=source,
                    start=round(audio_start, 3), end=round(audio_end, 3),
                    vad_start=start, vad_end=end,
                    boundary_reason=boundary_reason,
                    language="unknown", text="", audio=None,
                    context_before=[], context_after=[],
                    prediction_tags=[], acoustic_emotion_candidate=None,
                    human_verified=False, label_source="SenseVoiceSmall+FSMN-VAD",
                    status=status_for_reason.get(boundary_reason, "pending"))
        if boundary_reason:
            records.append(item)
            continue
        clip_id = hashlib.sha256(f"{source}|{audio_start:.3f}|{audio_end:.3f}".encode()).hexdigest()[:20]
        clip_rel = Path("clips") / speaker / f"{clip_id}.wav"
        clip_path = output / clip_rel
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            extract(executable, path, clip_path, audio_start, audio_end)
            result = asr.generate(input=str(clip_path), language=args.language,
                                  use_itn=False, batch_size_s=30)
            if not result or not result[0].get("text"):
                raise ValueError("empty ASR result")
            raw = str(result[0]["text"])
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
            text = rich_transcription_postprocess(raw).strip()
            language = raw_language(raw, args.language)
            tags = TAG_RE.findall(raw)
            item.update(text=text, language=language,
                        prediction_tags=tags,
                        acoustic_emotion_candidate=next((x for x in tags if x in
                            {"happy", "sad", "angry", "neutral", "fearful", "surprised", "disgusted"}), None))
            if not text or language not in {"zh", "ja"}:
                item["status"] = "review_asr_or_language"
            else:
                item["audio"] = clip_rel.as_posix()
                item["status"] = "accepted"
        except Exception as error:
            if "out of memory" in str(error).lower() or "cuda error" in str(error).lower():
                raise RuntimeError("ASR GPU exhausted; reduce maximum segment length or free VRAM") from error
            item["status"] = "review_runtime_error"
            item["error"] = f"{type(error).__name__}: {str(error)[:160]}"
        records.append(item)
        if i % 10 == 0:
            print(f"[ASR] {source} region {i}/{len(regions)}", flush=True)
    # Context should never flow across different source recordings.
    add_context(records)
    write_jsonl(manifest, records)
    print(f"[DONE] {source}: {sum(x['status']=='accepted' for x in records)} accepted / {len(records)}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Recording file or raw/<speaker_id>/ folder")
    p.add_argument("--speaker-id", help="Required when --input is one file")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="iic/SenseVoiceSmall")
    p.add_argument("--vad-model", default="fsmn-vad")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--vad-device", default="cpu")
    p.add_argument("--language", choices=["auto", "zh", "ja"], default="auto")
    p.add_argument("--window-seconds", type=float, default=120)
    p.add_argument("--overlap-seconds", type=float, default=3)
    p.add_argument("--min-seconds", type=float, default=3)
    p.add_argument("--max-seconds", type=float, default=25)
    p.add_argument("--join-gap-seconds", type=float, default=0.7)
    p.add_argument("--padding-seconds", type=float, default=0.12)
    p.add_argument("--ffmpeg-exe", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if not (0 < a.window_seconds and 0 <= a.overlap_seconds < a.window_seconds
            and 1 <= a.min_seconds < a.max_seconds <= 27
            and 0 <= a.padding_seconds <= 0.5):
        p.error("Invalid processing times")
    source_path = Path(a.input).expanduser().resolve()
    output = Path(a.output).expanduser().resolve()
    if not source_path.exists() or (source_path.is_dir() and output.is_relative_to(source_path)):
        p.error("Input path missing or output nested inside source")
    files = source_files(source_path, a.speaker_id)
    if not files:
        p.error("No recordings found")
    from funasr import AutoModel
    executable = resolve_ffmpeg(a.ffmpeg_exe)
    print("[FFmpeg]", executable, flush=True)
    vad = AutoModel(model=a.vad_model, device=a.vad_device, disable_update=True)
    asr = AutoModel(model=a.model, device=a.device, disable_update=True)
    for path, speaker, source in files:
        process_recording(path, speaker, source, output, vad, asr, executable, a)
    rows = []
    for manifest in sorted((output / "manifests").glob("*.jsonl")):
        rows.extend(json.loads(s) for s in manifest.read_text(encoding="utf-8").splitlines() if s.strip())
    write_jsonl(output / "metadata.jsonl", rows)
    print("[TOTAL] metadata rows:", len(rows), flush=True)


if __name__ == "__main__":
    main()
