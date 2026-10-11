"""Pure helpers for windowing, speech segmentation and contextual metadata."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    core_start: float
    core_end: float
    read_start: float
    read_end: float


def windows(duration: float, seconds: float = 120.0, overlap: float = 4.0):
    if duration <= 0 or seconds <= 0 or overlap < 0:
        raise ValueError("duration and seconds must be positive; overlap nonnegative")
    out = []
    t = 0.0
    while t < duration:
        end = min(t + seconds, duration)
        out.append(Window(t, end, max(0.0, t - overlap),
                          min(duration, end + overlap)))
        t = end
    return out


def clip_id(source: str, start: float, end: float) -> str:
    value = f"{source}|{start:.3f}|{end:.3f}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:20]


def join_text(first: str, second: str) -> str:
    first, second = first.strip(), second.strip()
    if not first:
        return second
    if not second:
        return first
    if re.search(r"[a-zA-Z0-9]$", first) and re.match(r"[a-zA-Z0-9]", second):
        return first + " " + second
    return first + second


def split_asr_segment(segment, max_seconds: float = 15.0):
    """Return (start, end, text) tuples in chunk-local time.

    Long segments are split only at word timestamps. If unavailable, the long
    segment is returned intact and later excluded from training.
    """
    start, end = float(segment.start), float(segment.end)
    text = str(segment.text).strip()
    if not text or end <= start:
        return []
    if end - start <= max_seconds:
        return [(start, end, text)]
    words = [w for w in (getattr(segment, "words", None) or [])
             if w.start is not None and w.end is not None and w.end > w.start]
    if not words:
        return [(start, end, text)]
    result = []
    batch = []
    for word in words:
        if batch and float(word.end) - float(batch[0].start) > max_seconds:
            result.append((float(batch[0].start), float(batch[-1].end),
                           "".join(str(x.word) for x in batch).strip()))
            batch = []
        batch.append(word)
    if batch:
        result.append((float(batch[0].start), float(batch[-1].end),
                       "".join(str(x.word) for x in batch).strip()))
    return [item for item in result if item[2]]


def merge_short(items: list[dict], min_seconds: float, max_seconds: float,
                max_gap: float = 0.5):
    """Merge nearby short phrases from the same source/language."""
    out = []
    for item in sorted(items, key=lambda x: (x["start"], x["end"])):
        if out:
            prev = out[-1]
            gap = item["start"] - prev["end"]
            if (prev["language"] == item["language"]
                    and 0 <= gap <= max_gap
                    and item["end"] - prev["start"] <= max_seconds
                    and (prev["end"] - prev["start"] < min_seconds
                         or item["end"] - item["start"] < min_seconds)):
                prev["end"] = item["end"]
                prev["text"] = join_text(prev["text"], item["text"])
                continue
        out.append(dict(item))
    return out


def add_context(items: list[dict], before_count: int = 2,
                after_count: int = 1, max_gap_seconds: float = 120):
    """Context is drawn only from the same source recording."""
    for i, item in enumerate(items):
        before = []
        for j in range(i - 1, max(-1, i - before_count - 1), -1):
            other = items[j]
            if item["start"] - other["end"] > max_gap_seconds:
                break
            before.insert(0, other["text"])
        after = []
        for j in range(i + 1, min(len(items), i + after_count + 1)):
            other = items[j]
            if other["start"] - item["end"] > max_gap_seconds:
                break
            after.append(other["text"])
        item["context_before"] = before
        item["context_after"] = after
    return items
