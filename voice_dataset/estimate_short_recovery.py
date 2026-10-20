"""Estimate recoverable short-VAD fragments without running ASR or changing data.

Legacy metadata records already include up to 0.12s leading/trailing padding.
Estimated gaps therefore add padding back; results are only a dry-run estimate.
No audio or transcribed text is read into the output.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_intervals(metadata: Path):
    grouped = defaultdict(list)
    for line_no, line in enumerate(metadata.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        source = str(row.get("source") or "")
        speaker = str(row.get("speaker_id") or "")
        if not source or not speaker:
            continue
        try:
            start = float(row.get("vad_start", row["start"]))
            end = float(row.get("vad_end", row["end"]))
        except (ValueError, TypeError, KeyError):
            continue
        if end <= start:
            continue
        precise = ("vad_start" in row and "vad_end" in row)
        status = str(row.get("status") or "unknown")
        grouped[(speaker, source)].append(dict(start=start, end=end,
             status=status, line=line_no, precise=precise))
    for k in grouped:
        grouped[k].sort(key=lambda x: (x["start"], x["end"], x["line"]))
    return grouped


def project_vad_bounds(rows, padding_s: float):
    """Derive approximate *unpadded* VAD bounds for old manifest rows.

    Clamp to previous/next midpoints only when ranges overlap; do not infer
    missing speech intervals or pretend that heuristic times are ground truth.
    """
    output = []
    for row in rows:
        if row["precise"]:
            start, end = row["start"], row["end"]
        else:
            start = row["start"] + padding_s
            end = row["end"] - padding_s
            if end <= start:
                start, end = row["start"], row["end"]
        output.append(dict(row, start=start, end=end))
    return output


def simulate(rows, gap_s: float, max_seconds: float,
             merge_policy: str = "short-only", min_seconds: float = 3.0):
    """Approximate short-only or legacy-all grouping of adjacent regions.

    The policy is the same as sensevoice_long.make_regions_detailed, except
    that legacy metadata no longer preserves original raw VAD intervals.
    Only rows labelled as boundary-short can be recognized with certainty
    from legacy data; full VAD/ASR reprocessing is required for validation.
    """
    if merge_policy not in {"short-only", "all"}:
        raise ValueError("merge_policy must be short-only or all")
    groups = []
    for row in rows:
        current_length = row["end"] - row["start"]
        previous_length = (groups[-1][-1]["end"] - groups[-1][0]["start"]
                           if groups else 0.0)
        current_is_short = (
            current_length < min_seconds or
            row["status"] in {"review_boundary", "review_short"}
        )
        if (groups and
            0 <= row["start"] - groups[-1][-1]["end"] <= gap_s and
            row["end"] - groups[-1][0]["start"] <= max_seconds and
            (merge_policy == "all" or previous_length < min_seconds or
             current_is_short)):
            groups[-1].append(row)
        else:
            groups.append([row])
    recoverable, needs_reasr, new_groups = set(), set(), []
    for group in groups:
        start, end = group[0]["start"], group[-1]["end"]
        flagged = [r for r in group if r["status"] in
                   {"review_boundary", "review_short"}]
        if len(group) >= 2 and flagged and min_seconds <= end - start <= max_seconds:
            recoverable.update(x["line"] for x in flagged)
            needs_reasr.update(x["line"] for x in group)
            new_groups.append(dict(start_s=round(start, 3), end_s=round(end, 3),
                                   count=len(group), recovered=len(flagged)))
    return recoverable, needs_reasr, new_groups


def estimate(metadata: Path, gaps=(0.7, 1.2, 1.5),
             padding_s: float = 0.12, max_seconds: float = 25.0,
             merge_policy: str = "short-only", min_seconds: float = 3.0):
    originals = read_intervals(metadata)
    total_short = sum(row["status"] in {"review_boundary", "review_short"}
                      for rows in originals.values() for row in rows)
    projection = {k: project_vad_bounds(v, padding_s)
                  for k, v in originals.items()}
    trials = []
    for gap in gaps:
        rescued = set()
        re_asr = set()
        examples = []
        for (speaker, source), rows in projection.items():
            recovered, selected, groups = simulate(
                rows, gap, max_seconds, merge_policy, min_seconds)
            rescued.update((speaker, source, x) for x in recovered)
            re_asr.update((speaker, source, x) for x in selected)
            if len(examples) < 10:
                examples.extend(dict(speaker_id=speaker,
                                     source=source, **g) for g in groups[:10-len(examples)])
        trials.append(dict(join_gap_seconds=gap,
                           candidate_short_rows_recovered=len(rescued),
                           candidate_rows_to_retranscribe=len(re_asr),
                           short_rows_still_unrecovered=total_short-len(rescued),
                           group_examples_no_transcript=examples))
    return dict(total_metadata_rows=sum(map(len, originals.values())),
                original_short_rows=total_short,
                legacy_timestamps_approximate=any(not r["precise"]
                    for rows in originals.values() for r in rows),
                merge_policy=merge_policy,
                method="metadata-only adjacent grouping, per speaker/source",
                warning=("Estimates are not validated VAD/ASR and do not certify word "
                         "boundaries. Re-transcribe complete merged audio before training."),
                scenarios=trials)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True)
    p.add_argument("--gaps", default="0.7,1.2,1.5")
    p.add_argument("--padding-seconds", type=float, default=0.12)
    p.add_argument("--max-seconds", type=float, default=25.0)
    p.add_argument("--min-seconds", type=float, default=3.0)
    p.add_argument("--merge-policy", choices=["short-only", "all"],
                   default="short-only",
                   help="short-only: rescue short clips; all: reproduce legacy estimates")
    p.add_argument("--output", help="Optional path for private local JSON statistics")
    args = p.parse_args()
    if args.padding_seconds < 0 or not 0 < args.min_seconds < args.max_seconds:
        p.error("Invalid padding or maximum duration")
    gaps = tuple(float(x) for x in args.gaps.split(","))
    if any(x < 0 or x > 3 for x in gaps):
        p.error("Gaps must be between 0 and 3 seconds")
    report = estimate(Path(args.metadata), gaps, args.padding_seconds,
                      args.max_seconds, args.merge_policy, args.min_seconds)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
