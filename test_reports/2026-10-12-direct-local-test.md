# Direct local test of uploaded audio — 2026-10-12

This test was run **directly in the current container**, not on AutoDL. The user audio remains private; no WAV, MP3, transcripts, or voice embeddings were uploaded to GitHub. The audio was 41m33.910s, MP3, 44.1kHz stereo, 99,865,695 bytes.

| Test | Result |
|---|---|
| FFmpeg full recording decode | **PASS** |
| Full 120-s window plan, ±4-s overlap | **PASS: 21 windows** |
| Actual decode of all 21 overlapping windows | **PASS: 21/21**, maximum duration difference 12 ms |
| 24kHz mono 16-bit WAV extraction | **PASS**, 12-second clip |
| Python repository utility unit tests | **PASS: 4/4** |
| 19-second source excerpt processed by repository pipeline | **PASS with mocked ASR:** 2 WAV clips (4.0/4.4s), metadata, speaker ID, context and re-run skipping verified |
| Acoustic feature sampling | **DONE:** 6 eight-second samples, pitch/RMS/peak analyzed with Parselmouth |
| True ASR transcription | **NOT RUN:** no pretrained weights locally; model download network blocked |
| Emotion inference | **NOT RUN:** no pretrained emotion model locally |
| AutoDL GPU execution | **NOT RUN** |

FFmpeg low-level detection (`silencedetect=noise=-38dB:d=0.55`) produced 891 below-threshold intervals totaling 1088.72 s. **This does not imply 1088.72 s of non-speech.** Quiet speech and music may be below the threshold, so the pipeline must use validated VAD and ASR.

Representative eight-second samples (time s → RMS dBFS / raw pitch median Hz): 0→-41.09/239.1; 300→-41.33/201.0; 600→-39.35/311.9; 1250→-29.74/240.6; 2000→-37.33/259.8; 2450→-40.12/218.3. Pitch values are exploratory only and have **not** been attributed to voice-only frames or emotion classes.

### Interpretation

The real media input, audio decoding, windowing, WAV extraction and metadata logic work on this file. **Actual recognized text, speaker style, Chinese/Japanese identification, and emotional labels have not been verified.** Mock test sentences are synthetic and must not be treated as the recording's transcript.

Follow-up: run real ASR with downloaded weights (AutoDL is optional; a local 4060 can work with smaller/quantized models), manually validate word/segment accuracy and use a validated emotion classifier before using data for TTS training.
