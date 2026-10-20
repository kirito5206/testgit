"""Quick REAL SenseVoiceSmall ASR test on a short excerpt.

Uses ModelScope-hosted weights via FunASR, not Hugging Face.
Unlike the mock integration tests, transcription text comes from a real model.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def resolve_ffmpeg(explicit_path: str | None = None) -> str:
    """Prefer a known FFmpeg binary bundled by imageio-ffmpeg.

    This avoids picking up stale or incompatible executables on Windows PATH.
    """
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        located = shutil.which(explicit_path)
        if located:
            return located
        raise FileNotFoundError(f"FFmpeg executable not found: {explicit_path}")

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        located = shutil.which("ffmpeg")
        if located:
            return located
        raise FileNotFoundError(
            "No FFmpeg binary found. Run: python -m pip install imageio-ffmpeg"
        )


def extract_pcm16(input_audio: Path, output_wav: Path,
                  start: float, seconds: float,
                  ffmpeg_exe: str = "ffmpeg") -> None:
    command = [
        ffmpeg_exe, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-ss", f"{start:.3f}", "-i", str(input_audio),
        "-t", f"{seconds:.3f}", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(output_wav),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg ({ffmpeg_exe}) failed with exit code "
            f"{result.returncode}:\n{result.stderr.strip()}\n"
            "Install imageio-ffmpeg or use --ffmpeg-exe to specify "
            "a newer FFmpeg executable."
        )
    if not output_wav.is_file() or output_wav.stat().st_size == 0:
        raise RuntimeError("FFmpeg produced an empty output file")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing MP3/WAV/etc.")
    parser.add_argument("--output", default="sensevoice_test.json")
    parser.add_argument("--model", default="iic/SenseVoiceSmall",
                        help="ModelScope model ID or already-downloaded local model folder")
    parser.add_argument("--device", default="cuda:0",
                        help="cuda:0 on NVIDIA laptop, or cpu")
    parser.add_argument("--language", choices=["auto", "zh", "ja"],
                        default="auto")
    parser.add_argument("--ffmpeg-exe", default=None,
                        help="Override FFmpeg executable; default prefers imageio-ffmpeg")
    parser.add_argument("--start", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.start < 0 or not (0 < args.seconds <= 29):
        parser.error("--start must be nonnegative; --seconds must be between 0 and 29")
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"Input not found: {input_path}")

    # Import lazily so --help and argument validation work without FunASR.
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    ffmpeg_exe = resolve_ffmpeg(args.ffmpeg_exe)
    print(f"FFmpeg executable: {ffmpeg_exe}", flush=True)
    with tempfile.TemporaryDirectory(prefix="sensevoice-") as folder:
        window = Path(folder) / "excerpt.wav"
        extract_pcm16(input_path, window, args.start, args.seconds, ffmpeg_exe)
        print(f"Loading SenseVoiceSmall: {args.model}", flush=True)
        model = AutoModel(model=args.model, device=args.device,
                          disable_update=True)
        result = model.generate(input=str(window), language=args.language,
                                use_itn=True, batch_size_s=30)
    if not result or "text" not in result[0]:
        raise RuntimeError("SenseVoice returned no transcription")
    raw = str(result[0]["text"])
    clean = rich_transcription_postprocess(raw)
    tags = re.findall(r"<\|([^|]+)\|>", raw)
    output = dict(source=input_path.name, start_s=args.start,
                  end_s=args.start + args.seconds, model=args.model,
                  requested_language=args.language, text=clean,
                  raw_model_output=raw, predicted_tags=tags,
                  verified=False, timestamp_type="excerpt_boundary_only")
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print("Recognized text:", clean)
    print("Raw model tags:", tags)
    print("Output:", output_path)
    print("Important: timestamps are excerpt boundaries, not word timestamps.")


if __name__ == "__main__":
    main()
