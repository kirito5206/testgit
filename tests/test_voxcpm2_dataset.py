"""Synthetic tests for VoxCPM2 export (no ML models, no private voices)."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from voice_dataset.sensevoice_long import intervals_union, make_regions, raw_language
from voice_dataset.voxcpm_export import build, trailing_silence_ms


def wav(path, seconds=4.0, tail=0.65):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = round(seconds * 16000)
    t = np.arange(count, dtype=np.float32) / 16000
    signal = np.round(np.sin(2 * math.pi * 180 * t) * 7000).astype("<i2")
    signal[-round(tail * 16000):] = 0
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(signal.tobytes())


class VoxCPMBuilderTests(unittest.TestCase):
    def test_vad_intervals_and_code_tags(self):
        self.assertEqual(intervals_union([(1, 2), (1.9, 3), (7, 8)]),
                         [(1, 3), (7, 8)])
        regions = make_regions([(0, 2), (2.4, 5)], min_s=3, max_s=25, join_gap=0.7)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0], (0, 5, False))
        self.assertEqual(raw_language("<|ja|><|neutral|>こんにちは", "auto"), "ja")
        self.assertEqual(raw_language("你好", "auto"), "unknown")

    def test_tail_detector(self):
        pcm = np.concatenate((np.ones(16000, dtype=np.int16) * 7000,
                              np.zeros(16000, dtype=np.int16)))
        self.assertGreaterEqual(trailing_silence_ms(pcm, 16000), 980)

    def test_official_manifests_no_cross_session_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "processed"
            root.mkdir()
            rows = []
            for session in ("session_a", "session_b", "session_c"):
                for j in (1, 2):
                    path = root / "clips" / "speaker_001" / f"{session}_{j}.wav"
                    wav(path)
                    rows.append(dict(audio=path.relative_to(root).as_posix(),
                        text=f"你好，这是会话{session}中的语音{j}。", speaker_id="speaker_001",
                        source=f"speaker_001/{session}.mp3", language="zh",
                        status="accepted", human_verified=True,
                        start=float(j * 10), end=float(j * 10 + 4)))
            # Also test that a Japanese row is excluded under zh-only default.
            ja_path = root / "clips" / "speaker_002" / "ja.wav"
            wav(ja_path)
            rows.append(dict(audio=ja_path.relative_to(root).as_posix(),
                text="こんにちは", speaker_id="speaker_002",
                source="speaker_002/session_ja.mp3", language="ja",
                status="accepted", human_verified=False))
            metadata = root / "metadata.jsonl"
            metadata.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                                encoding="utf-8")
            out = Path(d) / "voxcpm2"
            args = argparse.Namespace(metadata=str(metadata), input_root=None,
                output=str(out), languages="zh", verified_only=True,
                min_seconds=3, max_seconds=25, min_rms_dbfs=-48,
                max_clipped_pct=0.1, silence_dbfs=-45,
                max_tail_ms=500, keep_tail_ms=200, val_fraction=0.34,
                ref_fraction=1.0, seed=2026)
            report = build(args)
            self.assertEqual(report["total_accepted"], 6)
            self.assertEqual(report["excluded"], 1)
            train = [json.loads(x) for x in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            val = [json.loads(x) for x in (out / "val.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(train) + len(val), 6)
            self.assertGreater(len(val), 0)
            provenance = [json.loads(x) for x in (out / "provenance.jsonl").read_text(encoding="utf-8").splitlines()]
            lookup = {x["audio"]: x for x in provenance}
            train_sources = {lookup[x["audio"]]["source"] for x in train}
            val_sources = {lookup[x["audio"]]["source"] for x in val}
            self.assertTrue(train_sources.isdisjoint(val_sources))
            self.assertTrue(all(set(x) <= {"audio", "text", "duration", "dataset_id", "ref_audio"}
                                for x in train + val))
            for part in (train, val):
                available = {r["audio"] for r in part}
                for row in part:
                    self.assertTrue(Path(row["audio"]).is_file())
                    self.assertGreaterEqual(row["duration"], 3)
                    if "ref_audio" in row:
                        self.assertIn(row["ref_audio"], available)
                        self.assertNotEqual(row["ref_audio"], row["audio"])
                        self.assertEqual(lookup[row["audio"]]["speaker_id"],
                                         lookup[row["ref_audio"]]["speaker_id"])
            # Deterministic split / reference assignments.
            self.assertEqual(report, build(args))
            train_after = (out / "train.jsonl").read_text(encoding="utf-8")
            self.assertEqual(train_after,
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train))


if __name__ == "__main__":
    unittest.main()
