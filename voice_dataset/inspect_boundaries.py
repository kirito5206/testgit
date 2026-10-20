"""Privacy-preserving local diagnostics for rejected VAD boundaries.

Uses only metadata.jsonl; does not load audio, ASR weights or private text.
Legacy review_boundary rows cannot definitively distinguish long/short cuts:
duration heuristics are estimates only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def length_bucket(seconds: float) -> str:
    if seconds < 1:
        return "under_1s"
    if seconds < 2:
        return "1_to_2s"
    if seconds < 3:
        return "2_to_3s"
    if seconds < 5:
        return "3_to_5s"
    if seconds < 10:
        return "5_to_10s"
    if seconds < 20:
        return "10_to_20s"
    if seconds < 25:
        return "20_to_25s"
    return "25s_or_more"


def diagnostic(metadata: Path, min_seconds: float = 3.0,
               max_seconds: float = 25.0, padding_seconds: float = 0.12) -> dict:
    status = Counter()
    buckets = Counter()
    reasons = Counter()
    legacy_estimates = Counter()
    legacy_examples = []
    segments = 0
    legacy = 0
    durations = []
    for number, line in enumerate(metadata.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        st = str(obj.get("status", "unknown"))
        status[st] += 1
        if not st.startswith("review_"):
            continue
        segments += 1
        try:
            start = float(obj["start"])
            end = float(obj["end"])
            d = max(0.0, end - start)
        except (ValueError, TypeError, KeyError):
            continue
        durations.append(d)
        buckets[length_bucket(d)] += 1
        if obj.get("boundary_reason"):
            reasons[str(obj["boundary_reason"])] += 1
        if st == "review_short":
            reasons["too_short"] += (not bool(obj.get("boundary_reason")))
        elif st == "review_forced_cut":
            reasons["forced_cut"] += (not bool(obj.get("boundary_reason")))
        elif st == "review_boundary":
            legacy += 1
            # Historical timestamps already include up to +/- padding_seconds,
            # so these categories cannot be regarded as exact VAD causes.
            if d < min_seconds + 2 * padding_seconds + 0.05:
                guess = "likely_short"
            elif d >= max_seconds - 0.05:
                guess = "likely_forced_cut"
            else:
                guess = "ambiguous"
            legacy_estimates[guess] += 1
            if len(legacy_examples) < 12:
                legacy_examples.append({
                    "row": number, "start_s": round(start, 3),
                    "end_s": round(end, 3), "length_s": round(d, 3),
                    "estimate": guess
                })
    return {
        "total_rows": sum(status.values()),
        "status_counts": dict(status),
        "review_rows": segments,
        "review_duration_bins": dict(sorted(buckets.items())),
        "explicit_boundary_reasons": dict(reasons),
        "legacy_review_boundary_rows": legacy,
        "legacy_duration_estimates_not_ground_truth": dict(legacy_estimates),
        "legacy_examples_no_text_or_audio": legacy_examples,
        "review_duration_seconds_sum_not_speech_ground_truth": round(sum(durations), 2),
        "next_step": ("If mainly likely_short: tune --join-gap-seconds, then rerun ASR "
                      "with --overwrite into a NEW output folder. If mainly "
                      "likely_forced_cut: inspect long speech/singing boundaries; "
                      "do not automatically promote forced cuts to training."),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True)
    p.add_argument("--min-seconds", type=float, default=3)
    p.add_argument("--max-seconds", type=float, default=25)
    p.add_argument("--padding-seconds", type=float, default=0.12)
    p.add_argument("--output", default=None, help="Optional safe statistics JSON file")
    args = p.parse_args()
    path = Path(args.metadata).expanduser()
    if not path.is_file():
        p.error("Cannot find metadata.jsonl")
    report = diagnostic(path, args.min_seconds, args.max_seconds, args.padding_seconds)
    content = json.dumps(report, indent=2, ensure_ascii=False)
    print(content)
    if args.output:
        Path(args.output).write_text(content + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
