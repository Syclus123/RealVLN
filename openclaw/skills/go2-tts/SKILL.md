---
name: go2_tts_tencent
description: Text-to-Speech synthesis using Tencent Cloud TTS WebSocket API. Convert text to natural-sounding speech with multiple voice options.
---

# Go2 TTS - 腾讯云语音合成技能

使用腾讯云实时语音合成 WebSocket API 将文本转换为自然语音。支持多种音色选择，高质量语音输出。

## 功能特点

- 🎤 **高质量语音合成**：基于腾讯云 TTS 引擎
- 🎙️ **多种音色**：20+ 种音色选择（男女声、童声、方言等）
- ⚡ **实时合成**：WebSocket 流式合成，响应快速
- 💾 **自动保存**：合成音频自动保存到媒体目录
- 🔧 **灵活配置**：支持自定义音色、格式、输出路径

## 使用方法

### 基础用法

```bash
# 合成语音（默认音色: 101016智瑞-沉稳男声）
python3 scripts/synthesize.py "你好，我是勾勾"

# 指定其他音色
python3 scripts/synthesize.py "你好，我是勾勾" --voice 101010

# 指定输出路径和格式
python3 scripts/synthesize.py "你好" --output ~/music/hello.mp3 --codec mp3
```

### 查看可用音色

```bash
python3 scripts/synthesize.py --list-voices
```

### 在 OpenClaw 中使用

**场景**：用户要求语音回复

```yaml
# 1. 合成语音
exec:
  command: python3 /home/unitree/openclaw/skills/go2-tts-tencent/scripts/synthesize.py "你好，我是勾勾，很高兴为你服务！"

# 2. 发送语音给用户（自动保存到 ~/.openclaw/media/）
message:
  action: send
  media: "~/.openclaw/media/tts_tencent_20260306_120000.mp3"
```

## 可用音色

| 音色ID | 描述 |
|--------|------|
| 101001 | 智瑜 - 温柔女声 |
| 101002 | 智聆 - 标准女声 |
| 101003 | 智美 - 活泼女声 |
| 101004 | 智云 - 沉稳男声 |
| 101005 | 智莉 - 甜美女声 |
| 101006 | 智言 - 知性女声 |
| 101007 | 智娜 - 方言女声 |
| 101008 | 智琪 - 新闻女声 |
| 101009 | 智芸 - 可爱女声 |
| 101010 | 智华 - 磁性男声 |
| 101011 | 智燕 - 严厉女声 |
| 101012 | 智丹 - 柔和女声 |
| 101013 | 智霞 - 自然女声 |
| 101014 | 智婷 - 活力女声 |
| 101015 | 智刚 - 硬朗男声 |
| 101016 | 智瑞 - 沉稳男声 ⭐默认 |
| 101017 | 智萌 - 童声 |
| 101018 | 智栋 - 青年男声 |
| 101019 | 智杰 - 成熟男声 |
| 101020 | 智浩 - 亲切男声 |

更多音色请参考腾讯云官方文档。

## 参数说明

| 参数 | 短参数 | 默认值 | 说明 |
|------|--------|--------|------|
| `text` | - | 必填 | 要合成的文本 |
| `--voice` | `-v` | 101016 | 音色ID (默认: 智瑞-沉稳男声) |
| `--codec` | `-c` | mp3 | 音频格式 (mp3 或 pcm) |
| `--output` | `-o` | 自动生成 | 输出文件路径 |
| `--list-voices` | `-l` | - | 列出可用音色 |

## 音频格式

- **mp3**: 压缩格式，文件小，兼容性好 ⭐推荐
- **pcm**: 无损格式，文件大，音质好

采样率：16000Hz

## 完整实战示例

### 场景：机器狗打招呼

用户说："说句话听听"

```yaml
# 步骤 1：合成问候语音（使用默认音色-智萌童声）
exec:
  command: python3 /home/unitree/openclaw/skills/go2-tts/scripts/synthesize.py "你好！我是勾勾，一只宇树Go2机器狗。我会拍照、走路、还能听懂你说话！"

# 步骤 2：发送给用户（文件路径从输出中提取）
message:
  action: send
  media: "~/.openclaw/media/tts_tencent_20260306_120000.mp3"
```

### Python 代码调用

```python
from scripts.synthesize import text_to_speech

# 合成语音（使用默认音色-智萌童声）
audio_path = text_to_speech("你好，我是勾勾")
print(f"语音已保存: {audio_path}")
```

## 与 ASR 技能配合使用

结合语音识别 (`go2-asr`) 和语音合成 (`go2-tts-tencent`)，实现完整的语音交互：

```yaml
# 1. 识别用户语音
exec:
  command: python3 /home/unitree/openclaw/skills/go2-asr/scripts/transcribe.py /path/to/audio.ogg
# 输出: "你好勾勾"

# 2. 处理文本，生成回复...

# 3. 合成回复语音
exec:
  command: python3 /home/unitree/openclaw/skills/go2-tts/scripts/synthesize.py "你好！有什么可以帮你的吗？"

# 4. 发送语音回复
message:
  action: send
  media: "~/.openclaw/media/tts_tencent_xxx.mp3"
```

## 文件保存

合成音频自动保存到：`~/.openclaw/media/tts_tencent_YYYYMMDD_HHMMSS.mp3`

## 依赖

```bash
pip install websocket-client
```

## 文件结构

```
go2-tts/
├── SKILL.md                 # 本文档
└── scripts/
    └── synthesize.py        # 语音合成脚本
```

## 配置说明

API 配置已内置在脚本中：
- APPID: 1325896784
- SecretId/Key: 已配置

如需更换账号，请修改脚本顶部的配置区。

## 注意事项

1. **文本长度**：单次合成建议不超过 500 字符
2. **网络要求**：需要访问腾讯云 TTS 服务
3. **免费额度**：腾讯云 TTS 有免费调用额度

## 相关技能

- `go2-asr`：语音识别（语音→文字）
- `go2-tts`：语音合成（文字→语音）
- `go2-vision`：视觉（拍照）
- `unitree-go2`：运动控制

## 参考

- 腾讯云 TTS 文档: https://cloud.tencent.com/document/product/1073
- 音色列表: https://cloud.tencent.com/document/product/1073/92627
