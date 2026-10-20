# Long-audio Chinese / Japanese dataset builder (v0.1)

This project builds short audio + text + context training data from **authorized single-speaker recordings**. The future target is **Chinese-only personalized expressive speech synthesis**.

## Install (AutoDL / RTX 4090)
Run:

    apt-get update && apt-get install -y ffmpeg
    conda create -n voice-data python=3.11 -y
    conda activate voice-data
    pip install -r requirements-voice.txt

## Input folder
Organize recordings as:

    raw/
      person_001/
        live_001.mp3
        live_002.wav
      person_002/
        japanese_001.flac

Each recording contains a single primary speaker; each speaker folder must be a consistent identity.

## Process
From repository root, run:

    python -m voice_dataset.pipeline --input /root/autodl-tmp/raw --output /root/autodl-tmp/processed --device cuda --compute-type float16 --model large-v3

Local CPU smoke test:

    python -m voice_dataset.pipeline --input ./raw --output ./processed --device cpu --compute-type int8 --model small

The first run downloads the model. The script needs ffprobe/ffmpeg in PATH.

## Output
- clips/<speaker_id>/<stable-id>.wav: 24kHz mono clips
- manifests/<recording>.jsonl: each recording's transcription metadata
- metadata.jsonl: all records

Each row keeps speaker_id, source, timestamps, text, language, nearby text context and QA status. Unsupported languages and incorrectly sized segments are flagged for manual review, not saved as training audio. A rerun skips completed recording manifests; use --overwrite to regenerate.

## Limitations
v0.1 runs faster-whisper automatic Chinese/Japanese ASR per 120-second window; mixed-language switching *inside* a window can be missed. Transcript word timestamps are approximate. No automatic emotion/prosody annotations exist yet; explicit null fields avoid treating predictions as truth. Verify pronunciation, speaker identity and text before training. An interrupted file reprocesses from the beginning.

## Tests

    python -m unittest discover -s tests -v

## Next steps
1. Quality filters: clipped audio, long silence, ASR confidence, split-boundary repair.
2. emotion2vec / SenseVoice pseudo-labels, Parselmouth pitch and energy.
3. Context-aware semantic intent inference with human verification.
4. Chinese TTS training exports and personalized style profiles.

Never commit private audio, models, speaker embeddings, or generated datasets to a public GitHub repository. Use secure storage on AutoDL; obtain consent from voice owners and disclose generated speech.
