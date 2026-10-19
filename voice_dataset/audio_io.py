"""Decode FFmpeg-generated mono 16-bit PCM WAV to the format Whisper expects.

Pass a NumPy waveform directly to faster-whisper, bypassing PyAV's version-
sensitive av.open API. ffmpeg is already used to create the temporary WAV.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_mono_pcm16(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Return a 1D, contiguous, float32 waveform normalized to [-1, 1].

    Source must be mono signed 16-bit PCM with the requested sample rate.
    """
    with wave.open(str(path), "rb") as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError("Expected mono signed 16-bit PCM WAV")
        if f.getframerate() != sample_rate:
            raise ValueError(f"Expected {sample_rate} Hz, got {f.getframerate()} Hz")
        raw = f.readframes(f.getnframes())
    if not raw:
        raise ValueError("Audio window is empty")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
