# Try SenseVoiceSmall without Hugging Face

Alternative Mandarin/Japanese speech recognition route using the ModelScope
checkpoint `iic/SenseVoiceSmall`.

This is a short-excerpt **real-model** smoke test, not the complete 41-minute
training dataset build.

## Windows PowerShell (RTX 4060 laptop)

Run in your local `testgit` Git checkout:

```powershell
conda create -n sensevoice python=3.11 -y
conda activate sensevoice
python -m pip install -U funasr modelscope
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Install a CUDA-compatible version of PyTorch and torchaudio from the
[official PyTorch selector](https://pytorch.org/get-started/locally/) if
`CUDA available` is False. Make sure `ffmpeg` is on PATH.

Download model files through ModelScope (not Hugging Face):

```powershell
modelscope download --model iic/SenseVoiceSmall --local_dir .\models\SenseVoiceSmall
```

Run 20s from the 30-second position using the local model:

```powershell
python -m voice_dataset.sensevoice_smoketest --input ".\your_recording.mp3" --output ".\sensevoice_test.json" --model ".\models\SenseVoiceSmall" --start 30 --seconds 20 --language auto --device cuda:0
```

The JSON contains predicted transcription and the original SenseVoice markup
tokens, including any emotion and sound-event indicators. Such tokens are
pseudo-labels, **not verified emotion ground truth**. The start/end times
refer to the *whole 20-second test excerpt*; do not use them as word boundaries.

If the environment lacks a configured CUDA PyTorch stack, set `--device cpu`
for a slower first functional test.

Do not commit personal audio, transcripts or speaker data to the public repo.

## Follow-up implementation

Once this real 20s transcription succeeds, add validated VAD-based batching
for the full recording and export proper VAD-region timestamps. No word-level
alignment or language-switch detection is claimed by this smoke test.
