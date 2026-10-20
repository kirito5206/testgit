"""End-to-end FFmpeg + recording manifest smoke test with fake VAD/ASR.

Audio is synthesized locally; models are faked, and no real transcript,
private voice data, torch or model weights are required.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from voice_dataset.sensevoice_long import (
    extract, ffmpeg_duration, process_recording, source_files,
)
from voice_dataset.sensevoice_smoketest import resolve_ffmpeg


class FakeVad:
    def generate(self, input):
        return [{"value": [[1000, 6500]]}]


class FakeAsr:
    def generate(self, input, **kwargs):
        return [{"text": "<|zh|><|neutral|>这是合成测试文字。"}]


class LongPipelineTest(unittest.TestCase):
    def test_ffmpeg_and_per_recording_manifest(self):
        try:
            binary = resolve_ffmpeg()
        except FileNotFoundError:
            self.skipTest("FFmpeg not installed")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            file = base / "fixture.wav"
            with wave.open(str(file), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                samples = bytearray()
                for i in range(16000 * 8):
                    value = int(9000 * math.sin(2 * math.pi * 220 * i / 16000))
                    samples.extend(value.to_bytes(2, "little", signed=True))
                f.writeframes(samples)
            duration = ffmpeg_duration(file, binary)
            self.assertAlmostEqual(duration, 8, delta=0.03)
            excerpt = base / "excerpt.wav"
            extract(binary, file, excerpt, 1, 6.5)
            self.assertTrue(excerpt.is_file())
            with wave.open(str(excerpt), "rb") as f:
                self.assertEqual(f.getframerate(), 16000)
                self.assertEqual(f.getnchannels(), 1)
            output = base / "processed"
            args = types.SimpleNamespace(window_seconds=120, overlap_seconds=3,
                                         min_seconds=3, max_seconds=25,
                                         join_gap_seconds=0.7, padding_seconds=0.12,
                                         overwrite=False, language="auto")
            def postprocess(raw):
                return raw.replace("<|zh|>", "").replace("<|neutral|>", "")
            # Patch only FunASR's postprocess import; audio decode is real.
            funasr = types.ModuleType("funasr")
            utils = types.ModuleType("funasr.utils")
            pp = types.ModuleType("funasr.utils.postprocess_utils")
            pp.rich_transcription_postprocess = postprocess
            with patch.dict(sys.modules, {
                "funasr": funasr,
                "funasr.utils": utils,
                "funasr.utils.postprocess_utils": pp,
            }):
                process_recording(file, "speaker_001", "speaker_001/fixture.wav",
                                  output, FakeVad(), FakeAsr(), binary, args)
            manifest_files = list((output / "manifests").glob("*.jsonl"))
            self.assertEqual(len(manifest_files), 1)
            rows = [json.loads(line) for line in manifest_files[0].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "accepted")
            self.assertEqual(rows[0]["language"], "zh")
            self.assertEqual(rows[0]["text"], "这是合成测试文字。")
            self.assertEqual(rows[0]["speaker_id"], "speaker_001")
            self.assertFalse(rows[0]["human_verified"])
            self.assertTrue((output / rows[0]["audio"]).exists())
            # Completed manifests are skipped instead of overwriting.
            process_recording(file, "speaker_001", "speaker_001/fixture.wav",
                              output, FakeVad(), FakeAsr(), binary, args)
            self.assertEqual(manifest_files[0].read_text(encoding="utf-8").count("\n"), 1)

    def test_invalid_speaker_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "fake.wav"
            wav_path.write_bytes(b"no real audio required")
            with self.assertRaises(ValueError):
                source_files(wav_path, "../other_speaker")


if __name__ == "__main__":
    unittest.main()
