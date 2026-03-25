---
name: go2_feishu_audio
description: Send audio files as voice messages to Feishu (Lark). Converts audio to Opus format, uploads to Feishu server, and sends as playable voice message.
---

# Go2 Feishu Audio - 飞书语音消息发送技能

将音频文件（MP3/WAV）作为语音消息发送到飞书聊天中，用户可以直接点击播放，无需下载文件。

## 功能特点

- 🎤 **语音消息格式**：发送可直接播放的语音消息，而非文件
- 🔄 **自动格式转换**：将 MP3/WAV 自动转换为飞书支持的 Opus 格式
- 📤 **自动上传**：上传到飞书服务器获取 file_key
- 💬 **消息发送**：发送标准的飞书语音消息
- 🔧 **完整流程**：一站式完成转换-上传-发送

## 工作原理

飞书语音消息需要三步：
1. **格式转换**：将音频转为 Opus 格式（飞书要求）
2. **文件上传**：通过 `/im/v1/files` API 上传获取 `file_key`
3. **消息发送**：通过 `/im/v1/messages` API 发送 `msg_type: "audio"` 消息

## 使用方法

### 基础用法

```bash
# 发送语音消息（chat_id 从环境变量获取）
python3 scripts/send_voice.py /path/to/audio.mp3

# 指定 chat_id
python3 scripts/send_voice.py /path/to/audio.mp3 --chat-id ou_xxxxxxxx

# 指定音频时长
python3 scripts/send_voice.py /path/to/audio.mp3 --duration 5
```

### 分步使用

如果只想执行其中某一步：

```bash
# 1. 仅转换格式
python3 scripts/convert_opus.py input.mp3 --output output.opus

# 2. 仅上传文件（获取 file_key）
python3 scripts/send_audio.py input.opus --chat-id ou_xxx

# 3. 完整流程（转换+上传+发送）
python3 scripts/send_voice.py input.mp3 --chat-id ou_xxx
```

### 在 OpenClaw 中使用

**场景**：TTS 合成语音后，发送为语音消息

```yaml
# 步骤 1：合成语音
exec:
  command: python3 /home/unitree/openclaw/skills/go2-tts/scripts/synthesize.py "你好，我是勾勾！"
# 输出: /home/unitree/.openclaw/media/tts_tencent_xxx.mp3

# 步骤 2：发送为飞书语音消息
exec:
  command: python3 /home/unitree/openclaw/skills/go2-feishu-audio/scripts/send_voice.py ~/.openclaw/media/tts_tencent_xxx.mp3
```

## 环境变量配置

为了方便使用，可以设置以下环境变量：

```bash
# 飞书机器人配置（必须）
export FEISHU_APP_ID="cli_xxxxxxxx"
export FEISHU_APP_SECRET="xxxxxxxx"

# 默认聊天 ID（可选）
export FEISHU_CHAT_ID="ou_xxxxxxxx"
```

## 依赖

- Python 3.6+
- `requests` 库
- `ffmpeg` 或 `opusenc`（用于音频格式转换）

安装 ffmpeg：
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
```

## 文件结构

```
go2-feishu-audio/
├── SKILL.md                    # 本文档
└── scripts/
    ├── send_voice.py           # 完整流程（推荐）
    ├── send_audio.py           # 上传+发送（需 Opus 格式）
    └── convert_opus.py         # 格式转换
```

## 参数说明

### send_voice.py（完整流程）

| 参数 | 短参数 | 说明 |
|------|--------|------|
| `audio` | - | 音频文件路径（mp3/wav/opus） |
| `--chat-id` | `-c` | 目标聊天 ID |
| `--duration` | `-d` | 音频时长（秒） |
| `--app-id` | - | 飞书 App ID |
| `--app-secret` | - | 飞书 App Secret |

### send_audio.py（上传+发送）

| 参数 | 短参数 | 说明 |
|------|--------|------|
| `audio` | - | Opus 格式音频文件路径 |
| `--chat-id` | `-c` | 目标聊天 ID |
| `--duration` | `-d` | 音频时长（秒） |

## 完整实战示例

### 场景：机器狗说话并发送语音

```yaml
# 1. 合成语音（使用腾讯云 TTS）
exec:
  command: python3 /home/unitree/openclaw/skills/go2-tts/scripts/synthesize.py "你好！我是勾勾，一只会说话的机器狗！" --voice 101003
# 输出示例: /home/unitree/.openclaw/media/tts_tencent_20260306_120000.mp3

# 2. 发送为飞书语音消息
exec:
  command: |
    export FEISHU_CHAT_ID="${CHAT_ID}" && \
    python3 /home/unitree/openclaw/skills/go2-feishu-audio/scripts/send_voice.py \
    ~/.openclaw/media/tts_tencent_20260306_120000.mp3
```

### Python 代码调用

```python
from scripts.send_voice import send_voice_message

# 发送语音消息
success = send_voice_message(
    audio_path="/path/to/audio.mp3",
    chat_id="ou_xxxxxxxx",
    duration=5
)

if success:
    print("语音消息发送成功！")
else:
    print("发送失败")
```

## 与 TTS 技能配合使用

结合语音合成 (`go2-tts`) 和语音消息发送 (`go2-feishu-audio`)：

```bash
#!/bin/bash
# tts_and_send.sh - 合成语音并发送

TEXT="你好，我是勾勾！"
CHAT_ID="ou_xxxxxxxx"

# 1. 合成语音
AUDIO_FILE=$(python3 /home/unitree/openclaw/skills/go2-tts/scripts/synthesize.py "$TEXT" 2>&1 | tail -1)

# 2. 发送语音消息
python3 /home/unitree/openclaw/skills/go2-feishu-audio/scripts/send_voice.py "$AUDIO_FILE" --chat-id "$CHAT_ID"
```

## 飞书 API 参考

- [上传文件](https://open.feishu.cn/document/server-docs/im-v1/file/create)
- [发送消息](https://open.feishu.cn/document/server-docs/im-v1/message/create)
- [消息内容格式](https://open.feishu.cn/document/server-docs/im-v1/message-content-description/create_json)

## 注意事项

1. **音频格式**：飞书语音消息要求 Opus 格式，脚本会自动转换
2. **文件大小**：建议不超过 10MB
3. **时长限制**：语音消息建议不超过 60 秒
4. **API 权限**：需要机器人有发送消息的权限
5. **Token 有效期**：tenant_access_token 有效期 2 小时，脚本会自动刷新

## 故障排查

### 转换失败
- 检查是否安装了 ffmpeg：`which ffmpeg`
- 尝试手动转换：`ffmpeg -i input.mp3 -c:a libopus output.opus`

### 上传失败
- 检查 App ID 和 App Secret 是否正确
- 检查网络是否能访问飞书 API

### 发送失败
- 检查 chat_id 是否正确
- 检查机器人是否有发送权限
- 检查 token 是否过期

## 相关技能

- `go2-tts`：语音合成（文字→语音）
- `go2-feishu-audio`：发送语音消息（语音→飞书）
- `go2-asr`：语音识别（语音→文字）

组合使用可实现完整的语音交互！
