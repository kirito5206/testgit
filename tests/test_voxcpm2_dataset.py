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

from voice_dataset.sensevoice_long import intervals_union, make_regions, make_regions_detailed, raw_language
from voice_dataset.inspect_boundaries import diagnostic
from voice_dataset.estimate_short_recovery import estimate, simulate, project_vad_bounds
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

    def test_boundary_classification(self):
        # Short bursts are merged only when nearby and under duration limit.
        regions = make_regions_detailed([(0, 1.2), (1.7, 3.9), (8, 8.8)],
                                        min_s=3, max_s=25, join_gap=0.7)
        self.assertEqual(regions, [(0, 3.9, None), (8, 8.8, "too_short")])
        # Uninterrupted >25s speech is NEVER silently accepted.
        regions = make_regions_detailed([(10, 63)], 3, 25, 0.7)
        self.assertEqual(len(regions), 3)
        self.assertTrue(all(x[2] == "forced_cut" for x in regions))

    def test_legacy_boundary_report(self):
        with tempfile.TemporaryDirectory() as d:
            metadata = Path(d) / "metadata.jsonl"
            rows = [
                {"status": "accepted", "start": 0, "end": 5},
                {"status": "review_boundary", "start": 10, "end": 11.8},
                {"status": "review_boundary", "start": 20, "end": 45.2},
                {"status": "review_short", "boundary_reason": "too_short",
                 "start": 50, "end": 51.2},
            ]
            metadata.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            report = diagnostic(metadata)
            self.assertEqual(report["status_counts"]["review_boundary"], 2)
            self.assertEqual(report["legacy_duration_estimates_not_ground_truth"]["likely_short"], 1)
            self.assertEqual(report["legacy_duration_estimates_not_ground_truth"]["likely_forced_cut"], 1)
            self.assertEqual(report["explicit_boundary_reasons"]["too_short"], 1)
            self.assertEqual(report["legacy_review_boundary_rows"], 2)

    def test_recovery_gap_scenarios(self):
        rows = [
            dict(start=0.0, end=3.2, status="accepted", line=1, precise=False),
            dict(start=4.08, end=5.88, status="review_boundary", line=2, precise=False),
            dict(start=8.5, end=10.0, status="review_boundary", line=3, precise=False),
        ]
        intervals = project_vad_bounds(rows, 0.12)
        with_gap_07, _, _ = simulate(intervals, 0.7, 25)
        with_gap_12, reassess, _ = simulate(intervals, 1.2, 25)
        self.assertNotIn(2, with_gap_07)
        self.assertIn(2, with_gap_12)
        self.assertIn(1, reassess)  # Existing accepted clip must also be re-ASR'd.
        self.assertNotIn(3, with_gap_12)

    def test_short_recovery_grouped_by_source(self):
        with tempfile.TemporaryDirectory() as d:
            metadata = Path(d) / "metadata.jsonl"
            records = [
                dict(start=0, end=4, status="accepted", speaker_id="s1", source="a"),
                dict(start=4.8, end=6, status="review_boundary", speaker_id="s1", source="a"),
                dict(start=6.2, end=7, status="review_boundary", speaker_id="s1", source="b"),
            ]
            metadata.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            out = estimate(metadata, gaps=(0.7, 1.2))
            self.assertEqual(out["original_short_rows"], 2)
            self.assertEqual(out["scenarios"][0]["candidate_short_rows_recovered"], 0)
            self.assertEqual(out["scenarios"][1]["candidate_short_rows_recovered"], 1)

    def test_short_only_does_not_merge_two_long_clips(self):
        rows = [
            dict(start=0.0, end=5.0, status="accepted", line=1, precise=True),
            dict(start=5.8, end=10.8, status="accepted", line=2, precise=True),
            dict(start=11.6, end=12.7, status="review_short", line=3, precise=True),
        ]
        _, selected_short, groups_short = simulate(
            rows, 2, 25, merge_policy="short-only")
        self.assertEqual(selected_short, {2, 3})
        self.assertEqual(groups_short[0]["count"], 2)
        _, selected_all, groups_all = simulate(
            rows, 2, 25, merge_policy="all")
        self.assertEqual(selected_all, {1, 2, 3})
        self.assertEqual(groups_all[0]["count"], 3)
        vad_short = make_regions_detailed(
            [(0, 5), (5.8, 10.8), (11.6, 12.7)], 3, 25, 2,
            merge_policy="short-only")
        self.assertEqual(len(vad_short), 2)
        self.assertEqual(vad_short[0], (0, 5, None))
        self.assertEqual(vad_short[1], (5.8, 12.7, None))
        vad_all = make_regions_detailed(
            [(0, 5), (5.8, 10.8), (11.6, 12.7)], 3, 25, 2,
            merge_policy="all")
        self.assertEqual(len(vad_all), 1)

    def test_estimator_default_matches_short_only_policy(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "metadata.jsonl"
            records = [
                dict(start=0.0, end=5.0, status="accepted", speaker_id="s", source="a",
                     vad_start=0.0, vad_end=5.0),
                dict(start=5.8, end=10.8, status="accepted", speaker_id="s", source="a",
                     vad_start=5.8, vad_end=10.8),
                dict(start=11.6, end=12.7, status="review_short", speaker_id="s", source="a",
                     vad_start=11.6, vad_end=12.7),
            ]
            path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            short = estimate(path, gaps=(2.0,))
            legacy = estimate(path, gaps=(2.0,), merge_policy="all")
            self.assertEqual(short["merge_policy"], "short-only")
            self.assertEqual(short["scenarios"][0]["candidate_rows_to_retranscribe"], 2)
            self.assertEqual(legacy["scenarios"][0]["candidate_rows_to_retranscribe"], 3)

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
            rows.append(dict(audio=None, text="", speaker_id="speaker_001",
                source="speaker_001/session_a.mp3", language="zh",
                status="review_boundary", start=7.0, end=32.0,
                human_verified=False))
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
            self.assertEqual(report["excluded"], 2)
            self.assertEqual(report["upstream_rejected_statuses"], {"review_boundary": 1})
            self.assertEqual(report["rejected_reasons"]["upstream_status:review_boundary"], 1)
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
