"""Produce VoxCPM2 LoRA JSONL from speaker-labelled short-clip metadata.

Input is the metadata.jsonl produced by voice_dataset.pipeline (Whisper)
or voice_dataset.sensevoice_long (SenseVoice + FSMN VAD).
This exporter performs *no* transcription: text must already match its audio.

Output: train.jsonl / val.jsonl in the official VoxCPM2 schema, 16-kHz
mono PCM WAV clips, audit.jsonl and quality_report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def ffmpeg_binary() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        import shutil
        cmd = shutil.which("ffmpeg")
        if cmd is None:
            raise RuntimeError("Install imageio-ffmpeg or FFmpeg to process audio")
        return cmd


def decode_pcm(path: Path, command: str, rate: int = 16000) -> np.ndarray:
    result = subprocess.run(
        [command, "-nostdin", "-v", "error", "-i", str(path),
         "-vn", "-ac", "1", "-ar", str(rate),
         "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"],
        capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[:500])
    raw = result.stdout
    if not raw:
        raise ValueError("empty decoded audio")
    return np.frombuffer(raw[:len(raw)//2*2], dtype="<i2").copy()


def trailing_silence_ms(samples: np.ndarray, rate: int, dbfs: float = -45.0,
                        frame_ms: int = 20) -> int:
    """Diagnostic only: lower-amplitude speech may also be counted as silence."""
    frame_len = max(1, round(rate * frame_ms / 1000))
    threshold = 32768.0 * 10 ** (dbfs / 20)
    last = len(samples)
    while last:
        start = max(0, last - frame_len)
        if np.max(np.abs(samples[start:last].astype(np.int32))) > threshold:
            break
        last = start
    return round((len(samples) - last) * 1000 / rate)


def clean_audio(pcm: np.ndarray, args) -> tuple[np.ndarray, dict]:
    sample_rate = 16000
    length_s = len(pcm) / sample_rate
    if not args.min_seconds <= length_s <= args.max_seconds:
        raise ValueError("duration_out_of_range")
    abs_samples = np.abs(pcm.astype(np.int32))
    rms = math.sqrt(float(np.mean(pcm.astype(np.float64) ** 2)))
    rms_dbfs = 20 * math.log10(max(rms / 32768.0, 1e-10))
    clipped_pct = 100.0 * float(np.count_nonzero(abs_samples >= 32700)) / len(pcm)
    if rms_dbfs < args.min_rms_dbfs:
        raise ValueError("too_quiet_review")
    if clipped_pct > args.max_clipped_pct:
        raise ValueError("clipping_review")
    tail_ms = trailing_silence_ms(pcm, sample_rate, args.silence_dbfs)
    # Tail trimming is conservative, to avoid eliminating breathy/quiet syllables.
    trimmed_ms = 0
    if tail_ms > args.max_tail_ms:
        trim_ms = max(0, tail_ms - args.keep_tail_ms)
        trim_count = min(len(pcm), round(trim_ms * sample_rate / 1000))
        if trim_count:
            pcm = pcm[:-trim_count]
            trimmed_ms = round(trim_count * 1000 / sample_rate)
    out_seconds = len(pcm) / sample_rate
    if not args.min_seconds <= out_seconds <= args.max_seconds:
        raise ValueError("duration_after_trim_review")
    return pcm, dict(duration=round(out_seconds, 3),
                     rms_dbfs=round(rms_dbfs, 2),
                     clipped_percent=round(clipped_pct, 4),
                     original_tail_ms=tail_ms, trimmed_tail_ms=trimmed_ms)


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.wav")
    with wave.open(str(temp), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.astype("<i2", copy=False).tobytes())
    os.replace(temp, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def speaker_slug(speaker: str) -> str:
    return speaker if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", speaker) else "spk_" + hashlib.sha256(speaker.encode()).hexdigest()[:12]


def load_candidates(metadata: Path, root: Path, output: Path, args) -> tuple[list[dict], list[dict]]:
    result = []
    audit = []
    command = ffmpeg_binary()
    allowed_languages = set(args.languages.split(","))
    with metadata.open(encoding="utf-8-sig") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not_an_object")
                speaker = str(row.get("speaker_id") or "").strip()
                text = str(row.get("text") or "").strip()
                language = str(row.get("language") or "").lower().strip()
                src = str(row.get("source") or "").strip()
                audio = row.get("audio")
                if row.get("status", "accepted") != "accepted":
                    raise ValueError("source_not_accepted")
                if not speaker or not src or not isinstance(audio, str) or not audio:
                    raise ValueError("missing_speaker_source_or_audio")
                if language not in allowed_languages:
                    raise ValueError("language_excluded_or_uncertain")
                if not text or "<|" in text:
                    raise ValueError("empty_or_tagged_transcription")
                if args.verified_only and row.get("human_verified") is not True:
                    raise ValueError("not_human_verified")
                path = (root / audio).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError("unsafe_or_missing_audio_path")
                pcm = decode_pcm(path, command)
                pcm, metrics = clean_audio(pcm, args)
                # Stable ID across runs and ordered by individual source segment.
                key = f"{speaker}|{src}|{row.get('start','')}|{row.get('end','')}|{audio}"
                filename = hashlib.sha256(key.encode()).hexdigest()[:20] + ".wav"
                relative = Path("clips") / speaker_slug(speaker) / filename
                out_path = output / relative
                write_wav(out_path, pcm)
                result.append(dict(audio=str(out_path.resolve()), text=text, language=language,
                                   speaker_id=speaker, source=src, source_audio=audio,
                                   unique_id=filename[:-4], **metrics,
                                   human_verified=row.get("human_verified") is True))
            except (ValueError, RuntimeError, json.JSONDecodeError, OSError, TypeError) as ex:
                audit.append(dict(line=number, reason=str(ex)[:160],
                                  speaker_id=str(row.get("speaker_id", "")) if isinstance(locals().get("row"), dict) else ""))
                continue
    return result, audit


def split_sessions(items: list[dict], seed: int, val_fraction: float) -> tuple[list[dict], list[dict], list[str]]:
    by_speaker = defaultdict(lambda: defaultdict(list))
    for item in items:
        by_speaker[item["speaker_id"]][item["source"]].append(item)
    train, val, no_val = [], [], []
    for speaker, sessions in sorted(by_speaker.items()):
        source_names = sorted(sessions, key=lambda s: stable_int(f"{seed}|{speaker}|{s}"))
        if len(source_names) <= 1 or val_fraction == 0:
            no_val.append(speaker)
            val_sources = set()
        else:
            n_val = max(1, min(len(source_names) - 1,
                               round(len(source_names) * val_fraction)))
            val_sources = set(source_names[:n_val])
        for src, rows in sessions.items():
            (val if src in val_sources else train).extend(rows)
    return train, val, no_val


def make_manifest(items: list[dict], fraction: float, seed: int) -> tuple[list[dict], int]:
    by_speaker = defaultdict(list)
    for item in items:
        by_speaker[item["speaker_id"]].append(item)
    output, refs = [], 0
    for item in sorted(items, key=lambda r: (r["speaker_id"], r["source"], r["unique_id"])):
        sample = dict(audio=item["audio"], text=item["text"],
                      duration=item["duration"], dataset_id=0)
        possible = [x for x in by_speaker[item["speaker_id"]]
                    if x["unique_id"] != item["unique_id"]]
        if possible and (stable_int(f"{seed}|use_ref|{item['unique_id']}") % 10000)/10000 < fraction:
            different_session = [x for x in possible if x["source"] != item["source"]]
            possible = different_session or possible
            possible.sort(key=lambda x: stable_int(f"{seed}|ref|{item['unique_id']}|{x['unique_id']}"))
            sample["ref_audio"] = possible[0]["audio"]
            refs += 1
        output.append(sample)
    return output, refs


def build(args) -> dict:
    metadata = Path(args.metadata).expanduser().resolve()
    root = Path(args.input_root).expanduser().resolve() if args.input_root else metadata.parent
    output = Path(args.output).expanduser().resolve()
    if not metadata.is_file() or not root.is_dir():
        raise FileNotFoundError("Metadata file or input root does not exist")
    if not (0 <= args.val_fraction < 1 and 0 <= args.ref_fraction <= 1):
        raise ValueError("Split/ref fractions must be within [0,1], validation < 1")
    if args.min_seconds < 1 or args.max_seconds > 30 or args.min_seconds >= args.max_seconds:
        raise ValueError("Allowed duration bounds: 1 <= min < max <= 30")
    if output == root:
        raise ValueError("Output folder must differ from input root")
    if not output.exists():
        output.mkdir(parents=True)
    samples, audit = load_candidates(metadata, root, output, args)
    train, val, no_val = split_sessions(samples, args.seed, args.val_fraction)
    train_json, ref_train = make_manifest(train, args.ref_fraction, args.seed)
    val_json, ref_val = make_manifest(val, args.ref_fraction, args.seed)
    atomic_jsonl(output / "train.jsonl", train_json)
    atomic_jsonl(output / "val.jsonl", val_json)
    atomic_jsonl(output / "audit.jsonl", audit)
    # Do not leak timestamps, source identity or ASR pseudo-label confidence
    # into the official VoxCPM training manifest; keep provenance separate.
    atomic_jsonl(output / "provenance.jsonl", [
        dict(audio=item["audio"], speaker_id=item["speaker_id"], source=item["source"],
             source_audio=item["source_audio"], language=item["language"],
             human_verified=item["human_verified"], rms_dbfs=item["rms_dbfs"],
             clipped_percent=item["clipped_percent"], original_tail_ms=item["original_tail_ms"],
             trimmed_tail_ms=item["trimmed_tail_ms"])
        for item in samples])
    report = dict(total_accepted=len(samples), excluded=len(audit),
                  train=len(train_json), val=len(val_json),
                  train_ref_audio=ref_train, val_ref_audio=ref_val,
                  no_validation_speakers=no_val,
                  languages=dict(Counter(x["language"] for x in samples)),
                  rejected_reasons=dict(Counter(x["reason"] for x in audit)),
                  note="No ASR accuracy or speaker purity is guaranteed; manually review before LoRA.")
    (output / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",
                                                 encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True, help="Path to metadata.jsonl")
    p.add_argument("--input-root", default=None, help="Root for relative audio paths (default metadata parent)")
    p.add_argument("--output", required=True, help="Export folder, outside input root preferred")
    p.add_argument("--languages", default="zh", help="zh (default) or zh,ja for ablation")
    p.add_argument("--verified-only", action="store_true")
    p.add_argument("--min-seconds", type=float, default=3)
    p.add_argument("--max-seconds", type=float, default=25)
    p.add_argument("--min-rms-dbfs", type=float, default=-48)
    p.add_argument("--max-clipped-pct", type=float, default=0.1)
    p.add_argument("--silence-dbfs", type=float, default=-45)
    p.add_argument("--max-tail-ms", type=int, default=500)
    p.add_argument("--keep-tail-ms", type=int, default=200)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--ref-fraction", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
