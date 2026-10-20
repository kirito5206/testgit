"""Prepare a local browser-based human audit for speech/transcript pairs.

No model calls, network calls, or uploads are performed. The browser exports
an audit_results.json file, which can be applied to a *new* metadata JSONL.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import shutil
from pathlib import Path


DECISIONS = {"pending", "pass", "corrected", "uncertain", "reject"}
ERROR_TYPES = {
    "omission": "漏字",
    "substitution": "错字/错词",
    "repetition": "重复字词",
    "hallucination": "识别出音频中没有的内容",
    "truncation": "句首/句尾截断",
    "name_number": "人名/数字/专有名词",
    "punctuation": "标点或断句",
    "other": "其他",
}


def row_fingerprint(row: dict) -> str:
    fields = [str(row.get(k) if row.get(k) is not None else "")
              for k in ("source", "start", "end", "audio", "text")]
    payload = json.dumps(fields, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[tuple[int, dict]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Metadata line {line_no} is not a JSON object")
            rows.append((line_no, row))
    return rows


def resolve_audio(root: Path, value: str) -> Path:
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file missing: {path}")
    if not candidate.is_absolute():
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"Relative audio path escapes metadata root: {value}") from exc
    return path


def _card(index: int, item: dict, local_audio: str) -> str:
    esc = lambda value: html.escape(str(value if value is not None else ""), quote=True)
    original_text = str(item.get("text") or "")
    error_options = "\n".join(
        f'<label class="error"><input type="checkbox" value="{esc(code)}"> {html.escape(label)}</label>'
        for code, label in ERROR_TYPES.items()
    )
    return f"""
    <section class="card sample"
      data-metadata-line="{int(item['metadata_line'])}"
      data-row-fingerprint="{esc(item['row_fingerprint'])}"
      data-source="{esc(item.get('source', ''))}"
      data-start="{esc(item.get('start', ''))}"
      data-end="{esc(item.get('end', ''))}"
      data-audio="{esc(item.get('audio', ''))}"
      data-original-text="{esc(original_text)}">
      <header><h2>样本 {index:02d}</h2><span>metadata 第 {int(item['metadata_line'])} 行</span></header>
      <p class="source">{html.escape(str(item.get('source') or '未知来源'))}
        · {html.escape(str(item.get('start', '?')))}–{html.escape(str(item.get('end', '?')))} 秒</p>
      <audio controls preload="none" src="{esc(local_audio)}"></audio>
      <label class="field">原始转写（供对照）
        <pre>{html.escape(original_text)}</pre>
      </label>
      <label class="field">核验后的转写（可以直接编辑）
        <textarea class="review-text" rows="3" spellcheck="false">{html.escape(original_text)}</textarea>
      </label>
      <label class="field">审核结论
        <select class="decision">
          <option value="pending">尚未审核</option>
          <option value="pass">正确，无需修改</option>
          <option value="corrected">已修正文字</option>
          <option value="uncertain">仍听不清 / 不确定</option>
          <option value="reject">音频或切分不适合训练</option>
        </select>
      </label>
      <fieldset><legend>错误类型（可多选）</legend>{error_options}</fieldset>
      <label class="field">备注（可选）
        <textarea class="notes" rows="2" placeholder="例如：句尾被截断；人名需复核"></textarea>
      </label>
    </section>"""


def prepare(metadata_path: Path, output_dir: Path, count: int = 30,
            seed: int = 20261020) -> dict:
    metadata_path = metadata_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if count < 1:
        raise ValueError("--count must be at least 1")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    root = metadata_path.parent
    eligible = []
    for line_no, row in read_jsonl(metadata_path):
        if row.get("status", "accepted") != "accepted":
            continue
        if row.get("human_verified") is True:
            continue
        audio = row.get("audio")
        text = str(row.get("text") or "").strip()
        if not isinstance(audio, str) or not audio.strip() or not text:
            continue
        eligible.append({
            "metadata_line": line_no,
            "row_fingerprint": row_fingerprint(row),
            "audio": audio,
            "source": row.get("source", ""),
            "start": row.get("start"),
            "end": row.get("end"),
            "text": text,
        })
    if not eligible:
        raise ValueError("No unverified accepted rows with both audio and text were found")
    selected = random.Random(seed).sample(eligible, min(count, len(eligible)))
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    cards = []
    for index, item in enumerate(selected, 1):
        src = resolve_audio(root, item["audio"])
        suffix = src.suffix.lower() if src.suffix else ".wav"
        local_name = f"{index:03d}{suffix}"
        shutil.copy2(src, audio_dir / local_name)
        manifest_item = dict(item)
        manifest_item["local_audio"] = f"audio/{local_name}"
        manifest.append(manifest_item)
        cards.append(_card(index, item, f"audio/{local_name}"))

    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    page = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>语音数据人工核验</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,"Microsoft YaHei",sans-serif}
body{max-width:980px;margin:0 auto;padding:18px;line-height:1.55}
.notice{padding:12px;border:1px solid #8888;border-radius:10px;margin:12px 0 20px}
.card{border:1px solid #8888;border-radius:12px;padding:16px;margin:16px 0}
header{display:flex;justify-content:space-between;gap:12px;align-items:baseline}
h1{font-size:1.6rem}h2{font-size:1.2rem;margin:0}
.source{font-size:.9rem;opacity:.8;overflow-wrap:anywhere}
audio{width:100%;margin:8px 0}
.field{display:block;margin:14px 0;font-weight:600}
textarea,pre{box-sizing:border-box;width:100%;font:inherit;padding:10px;border:1px solid #8888;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere}
textarea{font-weight:400}
pre{margin:6px 0;min-height:2em}
select,button{font:inherit;padding:9px;border-radius:8px}
fieldset{border:1px solid #8888;border-radius:8px;display:flex;gap:8px;flex-wrap:wrap}
.error{font-size:.9rem}
button{cursor:pointer;margin:6px 8px 6px 0}
.primary{font-weight:700}
.small{font-size:.9rem;opacity:.8}
</style></head><body>
<h1>语音—文本人工核验</h1>
<div class="notice">
<strong>本地离线审核：</strong>音频与文本都在本机，不会上传。每条从头到尾播放；
若修改了文字，结论选“已修正文字”。听不清就选“不确定”，不要猜。
核验完成后点击“导出核验结果 JSON”，再把下载的 JSON 交给 apply 命令。
</div>
<div class="notice small" id="progress">已完成结论：0 / __COUNT__</div>
<div id="samples">__CARDS__</div>
<button class="primary" id="export">导出核验结果 JSON</button>
<button id="count">刷新完成数量</button>
<p class="small">“通过”表示听到的每个字都与文本一致；标点可按项目规范调整。
“拒绝”表示音频切分或质量不适合训练，不只是文字有错。</p>
<script>
const cards=[...document.querySelectorAll(".sample")];
function updateProgress(){
 const done=cards.filter(c=>c.querySelector(".decision").value!=="pending").length;
 document.getElementById("progress").textContent="已完成结论："+done+" / "+cards.length;
}
cards.forEach(c=>c.querySelector(".decision").addEventListener("change",updateProgress));
document.getElementById("count").addEventListener("click",updateProgress);
document.getElementById("export").addEventListener("click",()=>{
 const result=cards.map(c=>({
  metadata_line:Number(c.dataset.metadataLine),
  row_fingerprint:c.dataset.rowFingerprint,
  source:c.dataset.source,
  start:c.dataset.start,
  end:c.dataset.end,
  audio:c.dataset.audio,
  original_text:c.dataset.originalText,
  reviewed_text:c.querySelector(".review-text").value.trim(),
  decision:c.querySelector(".decision").value,
  error_types:[...c.querySelectorAll('input[type="checkbox"]:checked')].map(x=>x.value),
  notes:c.querySelector(".notes").value.trim()
 }));
 const blob=new Blob([JSON.stringify(result,null,2)+"\n"],{type:"application/json;charset=utf-8"});
 const url=URL.createObjectURL(blob),a=document.createElement("a");
 a.href=url;a.download="audit_results.json";document.body.appendChild(a);a.click();a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),1500);
});
updateProgress();
</script></body></html>"""
    page = page.replace("__COUNT__", str(len(selected)))
    page = page.replace("__CARDS__", "\n".join(cards))
    (output_dir / "review.html").write_text(page, encoding="utf-8")
    report = {
        "metadata": str(metadata_path),
        "output": str(output_dir),
        "eligible_unverified_rows": len(eligible),
        "sampled_rows": len(selected),
        "seed": seed,
        "open_in_browser": str(output_dir / "review.html"),
        "instructions": "Open review.html locally, audit each audio, export audit_results.json, then run the apply subcommand.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def apply_audit(metadata_path: Path, audit_path: Path, output_path: Path) -> dict:
    metadata_path = metadata_path.expanduser().resolve()
    audit_path = audit_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not metadata_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("Metadata or audit JSON file does not exist")
    original = read_jsonl(metadata_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    if not isinstance(audit, list):
        raise ValueError("Audit JSON must be a list exported by review.html")

    by_line = {line_no: row for line_no, row in original}
    seen = set()
    validated = []
    for item in audit:
        if not isinstance(item, dict):
            raise ValueError("Every audit result must be a JSON object")
        try:
            line_no = int(item["metadata_line"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Audit result has no valid metadata_line") from exc
        if line_no in seen:
            raise ValueError(f"Duplicate audit entry for metadata line {line_no}")
        seen.add(line_no)
        row = by_line.get(line_no)
        if row is None or row_fingerprint(row) != item.get("row_fingerprint"):
            raise ValueError(
                f"Metadata line {line_no} changed since sampling; regenerate the audit sample")
        decision = item.get("decision", "pending")
        if decision not in DECISIONS:
            raise ValueError(f"Unknown decision on metadata line {line_no}: {decision}")
        validated.append((line_no, item, row))

    counts = {"pass": 0, "corrected": 0, "uncertain": 0, "reject": 0, "pending": 0}
    by_line_update = {}
    for line_no, item, row in validated:
        decision = item.get("decision", "pending")
        counts[decision] += 1
        new_row = dict(row)
        reviewed_text = str(item.get("reviewed_text") or "").strip()
        notes = str(item.get("notes") or "").strip()
        errors = item.get("error_types", [])
        if not isinstance(errors, list):
            errors = []
        errors = [str(x) for x in errors if str(x) in ERROR_TYPES]
        if decision in {"pass", "corrected"}:
            if not reviewed_text:
                raise ValueError(f"Reviewed text is empty on metadata line {line_no}")
            effective = "corrected" if reviewed_text != str(row.get("text") or "").strip() else "pass"
            new_row["text"] = reviewed_text
            new_row["human_verified"] = True
            new_row["human_review_status"] = effective
            new_row["human_review_error_types"] = errors
            new_row["human_review_notes"] = notes
            counts[decision] -= 1
            counts[effective] += 1
        elif decision in {"uncertain", "reject"}:
            new_row["human_verified"] = False
            new_row["human_review_status"] = decision
            new_row["human_review_error_types"] = errors
            new_row["human_review_notes"] = notes
            new_row["status"] = f"review_human_{decision}"
        by_line_update[line_no] = new_row

    updated = [by_line_update.get(line_no, row) for line_no, row in original]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        for row in updated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(output_path)
    report = {
        "output_metadata": str(output_path),
        "rows_in_metadata": len(updated),
        "review_results": counts,
        "human_verified_accepted_rows_in_output": sum(
            row.get("status", "accepted") == "accepted" and row.get("human_verified") is True
            for row in updated),
        "note": "Only pass/corrected results are marked human_verified. uncertain/reject rows receive review statuses and are excluded by the exporter.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_prepare = sub.add_parser("prepare", help="Create local review.html and copy sampled audio")
    p_prepare.add_argument("--metadata", required=True)
    p_prepare.add_argument("--output", required=True)
    p_prepare.add_argument("--count", type=int, default=30)
    p_prepare.add_argument("--seed", type=int, default=20261020)
    p_apply = sub.add_parser("apply", help="Apply exported audit JSON to a new metadata JSONL")
    p_apply.add_argument("--metadata", required=True)
    p_apply.add_argument("--audit", required=True)
    p_apply.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(Path(args.metadata), Path(args.output), args.count, args.seed)
    else:
        apply_audit(Path(args.metadata), Path(args.audit), Path(args.output))


if __name__ == "__main__":
    main()
