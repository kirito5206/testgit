# VoxCPM2 LoRA：长录音到训练数据集

目标：使用已获授权的单人中文、日文长录音，最终生成**中文**个性化语音。首次只训练中文，后续用中文+日文做对照实验。

## 1. Windows 安装

本脚本已加入独立分支。先在 PowerShell 执行：

~~~powershell
cd "D:\CODE\music\testgit"
git fetch origin
git switch feat/voxcpm2-lora-dataset

$python = "D:\miniconda\envs\sensevoice311\python.exe"
& $python -m pip install -r requirements-voxcpm-data.txt
& $python -m unittest discover -s tests -p "test_voxcpm2_dataset.py" -v
~~~

这个依赖文件不修改 PyTorch/CUDA；使用你已经配置的 RTX 4060 环境。
imageio-ffmpeg 提供兼容的 FFmpeg 二进制文件，避开系统 PATH 中不兼容的旧 FFmpeg。

## 2. 用 SenseVoiceSmall + FSMN-VAD 把长录音做成短片段

你现在的单个 MP3 只要确实是同一位说话人即可运行：

~~~powershell
& $python -m voice_dataset.sensevoice_long --input ".\稀贩1姐弟.mp3" --speaker-id speaker_001 --output ".\processed" --model "iic/SenseVoiceSmall" --vad-model "fsmn-vad" --language zh --device cuda:0 --vad-device cpu --window-seconds 120 --min-seconds 3 --max-seconds 25
~~~

混合中日语的单条录音用 --language auto。若整段为日文，用 --language ja。

对多个录音，先组织原始数据：

~~~text
raw/
  speaker_001/
    001.mp3
    002.mp3
  speaker_002/
    001.mp3
~~~

~~~powershell
& $python -m voice_dataset.sensevoice_long --input ".\raw" --output ".\processed" --language auto --device cuda:0
~~~

输出有 processed/clips/、processed/manifests/ 和 processed/metadata.jsonl。
每个原始文件结束后保存清单，默认跳过已处理文件；--overwrite 重新处理。
每一行保留说话人 ID、原录音的时间戳、转写、语言、声学标签候选、前后文、
human_verified=false 和审查状态。

音频用重叠窗口进行 VAD，随后每条目标片段做 SenseVoice 真正 ASR。
无法放心切分的连续长语音、过短片段会标记 review_boundary，不冒充合格训练片段。
这里的时间戳是**声学 VAD 区间时间**，不是已经验证过的逐字对齐时间。

## 3. 导出官方 VoxCPM2 LoRA JSONL

~~~powershell
& $python -m voice_dataset.voxcpm_export --metadata ".\processed\metadata.jsonl" --output ".\voxcpm2_data" --languages zh --min-seconds 3 --max-seconds 25 --ref-fraction 0.4 --val-fraction 0.15
~~~

输出文件：

~~~text
voxcpm2_data/
  clips/speaker_001/*.wav  # 16000 Hz mono PCM16
  train.jsonl
  val.jsonl
  provenance.jsonl
  audit.jsonl
  quality_report.json
~~~

官方 VoxCPM2 训练字段包括 audio, text, 可选 ref_audio、duration、dataset_id。
示例为格式展示，不对应你的真实录音：

~~~json
{"audio":"/absolute/voxcpm2_data/clips/speaker_001/abcd.wav","text":"实验终于成功了。","ref_audio":"/absolute/voxcpm2_data/clips/speaker_001/efgh.wav","duration":4.2,"dataset_id":0}
~~~

本脚本读取转写后再检查真实音频时长、RMS、电平削波、尾部静音；拒绝异常
样本并输出 audit.jsonl，不会直接把错误转写写成“正确标注”。
ref_audio 只从**同一说话人且同一 train/val 分组内的另一段录音**中选取。
每个说话人的数据按**原始录音文件**划分 train/val，杜绝相同源会话跨组泄漏。
如果只有一个源文件，该说话人全部进入 train，val 可能为空：
详情见 quality_report.json 的 no_validation_speakers。

正式训练前必须人工审核并设置 human_verified=true。然后启用：

~~~powershell
& $python -m voice_dataset.voxcpm_export --metadata ".\processed\metadata.jsonl" --output ".\voxcpm2_verified" --languages zh --verified-only
~~~

默认语言仅 zh；要比较使用日语额外风格数据时才设置 --languages zh,ja。
不能把日文目标音频配对中文译文。音频导出后训练路径是**绝对路径**，
因此最好在最终运行 VoxCPM2 的 AutoDL 上执行最后一步导出。

## 4. 开始 VoxCPM2 LoRA

将通过审查的数据私密上传 AutoDL，下载官方 VoxCPM2。
官方训练配置中的关键字段：

~~~yaml
pretrained_path: /root/autodl-tmp/models/VoxCPM2
train_manifest: /root/autodl-tmp/voxcpm2_verified/train.jsonl
val_manifest: /root/autodl-tmp/voxcpm2_verified/val.jsonl
sample_rate: 16000
out_sample_rate: 48000
batch_size: 1
grad_accum_steps: 8
num_workers: 2
num_iters: 500
max_steps: 500
max_batch_tokens: 4096
learning_rate: 0.0001
weight_decay: 0.01
warmup_steps: 50
log_interval: 10
valid_interval: 100
save_interval: 100
save_path: /root/autodl-tmp/checkpoints/voxcpm2_lora
tensorboard: /root/autodl-tmp/logs/voxcpm2_lora
lambdas:
  loss/diff: 1.0
  loss/stop: 1.0
lora:
  enable_lm: true
  enable_dit: true
  enable_proj: false
  r: 32
  alpha: 32
  dropout: 0.0
~~~

24GB GPU 配置为初步建议，尚未在你的 AutoDL 实例训练验证。
若 val.jsonl 为空，不能把“有验证集”当作已完成，请增加另一录制会话并审核。

官方训练命令（在官方 VoxCPM 仓库中执行）：

~~~bash
python scripts/train_voxcpm_finetune.py --config_path conf/my_lora.yaml
~~~

官方资料：https://voxcpm.readthedocs.io/zh-cn/latest/finetuning/finetune.html

## 限制及安全

- 单文件仅一人说话是假设，不是模型已经验证的事实；不要混入其他人物。
- SenseVoice 声学情绪标签是模型猜测，不能代替语义与人物表达标注。
- 没有人工核对转写之前，不应宣称语音文本严格对应。
- 训练清单的声学路径是本机绝对路径，迁移机器要重新导出。
- 禁止将未授权人物声音、原音频、私密转写或声纹上传公开 GitHub。


## 5. 排查大批 review_boundary：无需重新跑 41 分钟 ASR

旧版长录音处理脚本将过短（<3 秒）和长语音强制切段两种情况
都标记为 review_boundary。可以直接对现有 metadata.jsonl 统计：

~~~powershell
cd "D:\CODE\music\testgit"
git pull origin feat/voxcpm2-lora-dataset
& "D:\miniconda\envs\sensevoice311\python.exe" -m voice_dataset.inspect_boundaries --metadata ".\processed\metadata.jsonl"
~~~

这条命令**不会加载模型、不生成或修改训练样本**，也不会上传个人录音。
主要关注输出里的
legacy_duration_estimates_not_ground_truth 和 review_duration_bins。
旧版 metadata 保存的是有边缘填充的片段时长，因此分类只是启发式估计：

- likely_short：更像短于最小训练时长的片段；
- likely_forced_cut：更像超过 VAD 最大长度的连续语音；
- ambiguous：仅靠旧版时间信息不能确认。

如果短段占多数，可以在**新的输出目录**重跑 SenseVoice+VAD，并尝试增大
--join-gap-seconds（例如从 0.7 改为 1.2）。
合并后的文本必须重新 ASR，不允许直接复用两个短段旧文本。
如果 forced_cut 占多数，需要在适当的停顿处重新分句，
不能直接把硬切的语音自动放入 VoxCPM2 训练数据。

新版处理后的 metadata 将显式使用 review_short / review_forced_cut，
并提供 boundary_reason、vad_start、vad_end，方便后续定点审核。
旧版已有音频和元数据不会被新的诊断命令篡改。
