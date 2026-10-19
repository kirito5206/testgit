"""Dataset generation CLI: single-speaker long Chinese/Japanese recordings."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from .core import windows, split_asr_segment, merge_short, add_context, clip_id
from .audio_io import read_mono_pcm16

AUDIO_EXT={".wav",".mp3",".flac",".m4a",".ogg",".mp4"}

def ffmpeg(*args):
    subprocess.run(["ffmpeg","-nostdin","-v","error","-y",*map(str,args)],check=True)

def duration(path):
    p=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1",str(path)],capture_output=True,text=True,check=True)
    return float(p.stdout.strip())

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    with temp.open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    temp.replace(path)

def process(path, root, out, model, args):
    relative=path.relative_to(root)
    if len(relative.parts)<2:
        raise ValueError("Audio must be in raw/<speaker_id>/<recording>")
    speaker=relative.parts[0]
    manifest=out/"manifests"/("__".join(relative.parts)+".jsonl")
    if manifest.exists() and not args.overwrite:
        print("skip:",relative)
        return
    total=duration(path)
    candidates=[]
    for index,w in enumerate(windows(total,args.window_seconds,args.overlap_seconds)):
        with tempfile.TemporaryDirectory() as d:
            clip=Path(d)/"window.wav"
            ffmpeg("-ss",f"{w.read_start:.3f}","-i",path,"-t",
                f"{w.read_end-w.read_start:.3f}","-ac","1","-ar","16000","-c:a","pcm_s16le",clip)
            segments,info=model.transcribe(read_mono_pcm16(clip),
                language=None if args.language=="auto" else args.language,
                vad_filter=True,word_timestamps=True,
                condition_on_previous_text=False)
            for seg in segments:
                for s,e,t in split_asr_segment(seg,args.max_seconds):
                    s+=w.read_start
                    e+=w.read_start
                    if not w.core_start <= (s+e)/2 < w.core_end: continue
                    candidates.append(dict(start=round(max(0,s),3),
                        end=round(min(total,e),3),text=t,language=info.language))
    candidates=merge_short(candidates,args.min_seconds,args.max_seconds)
    candidates.sort(key=lambda x:x["start"])
    add_context(candidates)
    rows=[]
    for item in candidates:
        item.update(speaker_id=speaker,source=relative.as_posix(),
            acoustic_emotion=None,semantic_intent=None,prosody=None,
            label_source="faster-whisper",human_verified=False)
        length=item["end"]-item["start"]
        status=("review_duration" if not args.min_seconds<=length<=args.max_seconds
                else "review_language" if item["language"] not in ("zh","ja")
                else "accepted")
        item["status"]=status
        item["audio"]=None
        if status=="accepted":
            target=out/"clips"/speaker/(clip_id(relative.as_posix(),item["start"],item["end"])+".wav")
            target.parent.mkdir(parents=True,exist_ok=True)
            ffmpeg("-ss",f'{item["start"]:.3f}',"-i",path,"-t",f"{length:.3f}",
                "-ac","1","-ar",str(args.sample_rate),"-c:a","pcm_s16le",target)
            item["audio"]=target.relative_to(out).as_posix()
        rows.append(item)
    write_jsonl(manifest,rows)
    print("processed:",relative,len(rows),"segments")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--model",default="large-v3")
    p.add_argument("--device",default="cuda")
    p.add_argument("--compute-type",default="float16")
    p.add_argument("--language",choices=["auto","zh","ja"],default="auto")
    p.add_argument("--window-seconds",type=float,default=120)
    p.add_argument("--overlap-seconds",type=float,default=4)
    p.add_argument("--min-seconds",type=float,default=2)
    p.add_argument("--max-seconds",type=float,default=15)
    p.add_argument("--sample-rate",type=int,default=24000)
    p.add_argument("--overwrite",action="store_true")
    a=p.parse_args()
    if a.window_seconds<=0 or a.overlap_seconds<0 or a.min_seconds<=0 or a.max_seconds<a.min_seconds:
        p.error("Invalid time parameters")
    from faster_whisper import WhisperModel
    root=Path(a.input).resolve()
    out=Path(a.output).resolve()
    if not root.is_dir() or root==out or root in out.parents: p.error("Invalid input/output folders")
    model=WhisperModel(a.model,device=a.device,compute_type=a.compute_type)
    files=sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in AUDIO_EXT)
    for path in files: process(path,root,out,model,a)
    rows=[]
    for manifest in sorted((out/"manifests").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    write_jsonl(out/"metadata.jsonl",rows)
    print("total records:",len(rows))

if __name__=="__main__": main()
