---
name: go2_volcengine_voice
description: Real-time voice conversation using Volcengine (ByteDance) API. Speech-to-Text -> Chat -> Text-to-Speech pipeline.
---

# Go2 Volcengine Voice - 火山引擎语音对话

基于火山引擎（字节跳动）API 的实时语音对话技能，实现：
- 🎤 **语音输入** (ASR) - 语音识别转文字
- 💬 **大模型对话** (Chat) - 豆包大模型
- 🔊 **语音输出** (TTS) - 语音合成

## 功能特点

- 🎙️ **完整语音对话流程**：语音 → 文字 → AI回复 → 语音
- 🤖 **豆包大模型**：使用火山方舟大模型平台
- 🎵 **高质量语音**：支持多种音色选择
- 🔧 **灵活配置**：支持环境变量或参数配置

## 前置要求

1. **火山引擎账号**：注册 https://console.volcengine.com/
2. **开通服务**：
   - 语音识别 (ASR)
   - 语音合成 (TTS)
   - 大模型对话 (方舟)
3. **获取密钥**：
   - API Key
   - App ID
   - Access Token

## 配置方法

### 方式1：环境变量（推荐）

```bash
export VOLCENGINE_API_KEY="your_api_key"
export VOLCENGINE_APP_ID="your_app_id"
export VOLCENGINE_ACCESS_TOKEN="your_access_token"
```

### 方式2：命令行参数

```bash
python3 scripts/voice_chat.py input.wav \
  --api-key "your_api_key" \
  --app-id "your_app_id" \
  --access-token "your_access_token"
```

## 使用方法

### 基础用法

```bash
# 语音对话（需要配置环境变量）
python3 scripts/voice_chat.py /path/to/input.wav

# 指定输出文件
python3 scripts/voice_chat.py input.wav --output reply.wav

# 自动播放回复
python3 scripts/voice_chat.py input.wav --play
```

### 完整示例

```bash
# 1. 录制或准备输入语音
# 2. 执行语音对话
python3 /home/unitree/openclaw/skills/go2-volcengine-voice/scripts/voice_chat.py \
  /home/unitree/.openclaw/media/input.wav \
  --output /tmp/ai_reply.wav \
  --play
```

### 在 OpenClaw 中使用

```yaml
# 1. 录制用户语音
# 2. 调用火山引擎语音对话
exec:
  command: |
    export VOLCENGINE_API_KEY="${VOLCENGINE_API_KEY}" && \
    python3 /home/unitree/openclaw/skills/go2-volcengine-voice/scripts/voice_chat.py \
    /home/unitree/.openclaw/media/user_voice.wav \
    --output /tmp/ai_reply.wav \
    --play

# 3. 发送 AI 文字回复给用户
message:
  action: send
  message: "AI回复内容"
```

## 工作流程

```
┌─────────────┐    ASR    ┌─────────────┐   Chat   ┌─────────────┐   TTS   ┌─────────────┐
│  用户语音    │ ────────> │   识别文字   │ ───────> │  AI 回复    │ ──────> │  合成语音    │
│  input.wav  │           │  user_text  │          │  ai_reply   │         │ output.wav  │
└─────────────┘           └─────────────┘          └─────────────┘         └─────────────┘
                                                                                   │
                                                                                   ▼
                                                                            ┌─────────────┐
                                                                            │  扬声器播放   │
                                                                            └─────────────┘
```

## API 说明

### ASR 语音识别

- **接口**: `POST https://openspeech.bytedance.com/api/v1/auc`
- **格式**: WAV, 16kHz, 16bit
- **支持语言**: 中文、英文等

### Chat 大模型对话

- **接口**: `POST https://ark.cn-beijing.volces.com/api/v3/chat/completions`
- **模型**: doubao-lite-4k, doubao-pro-4k 等
- **功能**: 自然语言理解、知识问答、对话生成

### TTS 语音合成

- **接口**: `POST https://openspeech.bytedance.com/api/v1/tts`
- **音色**: BV001_streaming, BV002_streaming 等
- **格式**: WAV, 16kHz

## 音色列表

| 音色代码 | 描述 |
|---------|------|
| BV001_streaming | 中文女声（默认） |
| BV002_streaming | 中文男声 |
| BV701_streaming | 英文女声 |
| ... | 更多音色请参考火山引擎文档 |

## 参数说明

| 参数 | 环境变量 | 说明 |
|------|---------|------|
| `--api-key` | `VOLCENGINE_API_KEY` | 火山引擎 API Key |
| `--app-id` | `VOLCENGINE_APP_ID` | 火山引擎 App ID |
| `--access-token` | `VOLCENGINE_ACCESS_TOKEN` | 火山引擎 Access Token |
| `--output` | - | 输出音频文件路径 |
| `--play` | - | 自动播放输出语音 |

## 文件结构

```
go2-volcengine-voice/
├── SKILL.md                 # 本文档
└── scripts/
    └── voice_chat.py        # 语音对话脚本
```

## 依赖

```bash
pip install requests sounddevice numpy websockets
```

## 完整对话示例

```python
from scripts.voice_chat import VolcengineVoiceClient

# 创建客户端
client = VolcengineVoiceClient(
    api_key="your_api_key",
    app_id="your_app_id",
    access_token="your_access_token"
)

# 执行语音对话
user_text, ai_reply, success = client.voice_chat(
    input_audio_path="/path/to/input.wav",
    output_audio_path="/path/to/output.wav"
)

if success:
    print(f"用户: {user_text}")
    print(f"AI: {ai_reply}")
    # 播放音频...
```

## 故障排查

### 401 Unauthorized
- 检查 API Key 是否正确
- 检查 Access Token 是否过期

### 无法识别语音
- 确保音频格式为 WAV, 16kHz, 16bit
- 检查音频文件是否损坏

### 无声音输出
- 检查机器狗扬声器设置
- 尝试使用 `go2-audio-play` skill 测试扬声器

## 参考文档

- 火山引擎语音识别: https://www.volcengine.com/docs/6561
- 火山引擎语音合成: https://www.volcengine.com/docs/6561
- 火山方舟大模型: https://www.volcengine.com/docs/82379

## 相关技能

- `go2-asr`：本地语音识别（备用）
- `go2-tts`：本地语音合成（备用）
- `go2-audio-play`：扬声器播放
- `go2-volcengine-voice`：火山引擎云端语音对话（本技能）
