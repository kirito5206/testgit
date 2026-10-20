"""Tests for the Windows-safe FFmpeg helper; no ASR weights required."""
from __future__ import annotations

import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from voice_dataset.sensevoice_smoketest import extract_pcm16, resolve_ffmpeg


class FFmpegTests(unittest.TestCase):
    def test_explicit_nonexistent_binary_fails(self):
        with self.assertRaises(FileNotFoundError):
            resolve_ffmpeg("nonexistent-ffmpeg-binary-test-2026")

    def test_short_wav_decoding(self):
        try:
            ffmpeg_exe = resolve_ffmpeg()
        except FileNotFoundError:
            self.skipTest("No FFmpeg binary installed")
        with tempfile.TemporaryDirectory() as temp:
            input_wav = Path(temp) / "tone.wav"
            output_wav = Path(temp) / "out.wav"
            with wave.open(str(input_wav), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(b"".join(
                    struct.pack("<h", int(10000 * math.sin(i * 2 * math.pi * 440 / 16000)))
                    for i in range(16000)
                ))
            extract_pcm16(input_wav, output_wav, 0.2, 0.3, ffmpeg_exe)
            with wave.open(str(output_wav), "rb") as f:
                self.assertEqual(f.getframerate(), 16000)
                self.assertEqual(f.getnchannels(), 1)
                self.assertEqual(f.getsampwidth(), 2)
                self.assertGreater(f.getnframes(), 4000)


if __name__ == "__main__":
    unittest.main()
