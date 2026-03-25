---
name: go2_asr
description: Speech-to-text recognition using DashScope qwen3-asr-flash API. Transcribe audio files from local path or URL.
---

# Go2 ASR - 语音识别技能

使用阿里云 DashScope 的 `qwen3-asr-flash` 模型进行语音识别，支持本地音频文件和 URL。

## 功能特点

- 🎤 **高精度识别**：基于 DashScope qwen3-asr-flash 模型
- 📁 **本地文件支持**：支持 ogg, mp3, wav 等格式
- 🌐 **URL 支持**：可直接识别网络音频
- 🔧 **base64 编码**：自动处理本地文件上传

## 使用方法

### 识别本地音频文件

```bash
# 识别本地音频文件
python3 scripts/transcribe.py /path/to/audio.ogg

# 识别 Feishu 接收到的语音消息
python3 scripts/transcribe.py ~/.openclaw/media/inbound/xxx.ogg
```

### 识别网络音频 URL

```bash
python3 scripts/transcribe.py "https://example.com/audio.mp3"
```

### 在 OpenClaw 中使用

**场景**：用户发送语音消息，需要转录为文字

```yaml
# 用户发送了语音，文件保存在 inbound 目录
exec:
  command: python3 /home/unitree/openclaw/skills/go2-asr/scripts/transcribe.py ~/.openclaw/media/inbound/xxx.ogg

# 输出会打印识别结果
```

## API 信息

- **服务**: DashScope 阿里云
- **模型**: qwen3-asr-flash
- **API URL**: https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation

## 完整实战示例

### Feishu 语音消息处理流程

用户发送语音 → 保存到 `~/.openclaw/media/inbound/` → 调用 ASR 识别 → 回复文字内容

```yaml
# 步骤 1：识别语音（假设文件路径已知）
exec:
  command: python3 /home/unitree/openclaw/skills/go2-asr/scripts/transcribe.py ~/.openclaw/media/inbound/voice_message.ogg

# 步骤 2：获取识别结果并回复用户
# 识别结果会输出到 stdout，可以捕获后发送给用户
```

### Python 代码调用

```python
from scripts.transcribe import transcribe_audio

# 识别本地文件
text = transcribe_audio("/path/to/audio.ogg")
print(f"识别结果: {text}")

# 识别 URL
text = transcribe_audio("https://example.com/audio.mp3")
print(f"识别结果: {text}")
```

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `audio` | 音频文件路径或 URL | `~/audio.ogg` 或 `https://...` |
| `--api-key` | DashScope API Key（可选） | `sk-xxxxx` |

## 支持的音频格式

- OGG (Opus) - Feishu 默认格式
- MP3
- WAV
- 其他常见音频格式

## 响应格式

成功时输出识别的纯文本：
```
✅ 识别成功!
📝 识别结果: 你好，我是勾勾
你好，我是勾勾
```

## 错误处理

- 文件不存在 → 返回 None，exit code 1
- API 请求失败 → 打印错误信息，exit code 1
- 超时 → 打印超时信息，exit code 1

## 依赖

```bash
pip install requests
```

## 文件结构

```
go2-asr/
├── SKILL.md                 # 本文档
└── scripts/
    └── transcribe.py        # 语音识别脚本
```

## 注意事项

1. **API Key 安全**：默认使用内置 Key，可通过 `--api-key` 参数自定义
2. **文件大小**：大文件建议先压缩或使用 URL 方式
3. **网络要求**：需要访问阿里云 DashScope 服务
4. **超时设置**：默认 30 秒超时

## 参考

- DashScope 文档: https://dashscope.aliyun.com/
- qwen3-asr-flash 模型文档
