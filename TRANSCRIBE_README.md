# Real speech transcription (Chinese / Japanese)

A standalone speech recognition tool that uses **actual** faster-whisper model inference, not mock text. The user's RTX 4060 laptop can execute it after downloading a model. Original audio stays on the machine where the command is run.

## Install

Python 3.11 is recommended. Install FFmpeg for your operating system.

```bash
pip install -r requirements-voice.txt
```

## Usage — laptop RTX 4060 (8 GB VRAM)

```bash
python -m voice_dataset.transcribe --input "D:/recordings/sample.mp3" --output "D:/recordings/asr" --model small --device cuda --compute-type float16 --language auto
```

For a known Japanese-only recording: `--language ja`. For Chinese-only: `--language zh`. Run `--model medium` if small quality is insufficient and memory allows. For debugging on CPU: `--device cpu --compute-type int8`.

## Usage — AutoDL (24GB+ NVIDIA GPU)

```bash
python -m voice_dataset.transcribe --input /root/autodl-tmp/raw/person_001/recording.mp3 --output /root/autodl-tmp/asr --model large-v3 --device cuda --compute-type float16 --language auto
```

On the first run, faster-whisper downloads model weights from Hugging Face; internet/model access is required. Output formats:

- `*.txt`: timestamped readable transcription
- `*.jsonl`: original start/end seconds, text, language guess and language probability
- `*.srt`: subtitle format

## Important validation notes

- Uses overlapping windows to reduce errors at long-audio boundaries, keeping word midpoints only in their unique core windows.
- In mixed-language audio, automatic detection occurs **per window**, not perfectly per word. Manually verify Chinese/Japanese code switching.
- Timestamps and words are automatically predicted; sample human review is required before TTS model training.
- The original recording and generated text must be protected. **Do not push raw media or private transcriptions to public GitHub.**
- This command is NOT yet executable in the ChatGPT container because the speech recognition weights are not locally installed and model downloads fail in the current container.
