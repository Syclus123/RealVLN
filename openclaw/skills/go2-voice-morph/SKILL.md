---
name: go2_voice_morph
description: Voice morphing using ASR + TTS pipeline. Convert speech to different voices by transcribing to text and resynthesizing with target voice.
---

# Go2 Voice Morph - 语音变形技能

基于 **ASR + TTS** 实现语音转换，完全避免了 ElevenLabs 的 Chinglish 口音问题。

**核心原理**: 语音 → 文字(ASR) → 新语音(TTS)

## 🎯 功能特点

- 🎭 **自然转换**: 基于腾讯云 TTS，中文发音标准
- 🎙️ **多种音色**: 9 种声音可选（童声、男声、女声等）
- 🔊 **完整流程**: 语音输入 → 识别 → 转换 → 播放
- 📱 **飞书集成**: 接收飞书语音，转换后播放
- ⚡ **无需 API Key**: 使用内置的 ASR/TTS 配置

## 🔧 前置要求

已安装依赖技能：
- `go2-asr`: 语音识别
- `go2-tts`: 语音合成  
- `go2-audio-play`: 扬声器播放

系统依赖：
```bash
sudo apt-get install ffmpeg
```

## 🚀 使用方法

### 基础用法

```bash
# 转换音频文件（默认 101016智瑞-沉稳男声）
python3 scripts/voice_morph.py /path/to/audio.ogg

# 指定转换为男声
python3 scripts/voice_morph.py /path/to/audio.ogg --voice man

# 转换为女声
python3 scripts/voice_morph.py /path/to/audio.ogg --voice girl

# 转换为童声
python3 scripts/voice_morph.py /path/to/audio.ogg --voice tong

# 不播放，只生成文件
python3 scripts/voice_morph.py /path/to/audio.ogg --no-play
```

### 查看可用声音

```bash
python3 scripts/voice_morph.py --list-voices
```

**可用声音**:

| 名称 | 描述 | 音色ID |
|------|------|--------|
| `default` | **智瑞-沉稳男声 (默认)** ⭐ | **101016** |
| `man` | 智瑞-沉稳男声 | 101016 |
| `tong` | 智萌-童声 | 101017 |
| `boy` | 智栋-青年男声 | 101018 |
| `girl` | 智美-活泼女声 | 101003 |
| `woman` | 智聆-标准女声 | 101002 |
| `sweet` | 智莉-甜美女声 | 101005 |
| `cute` | 智芸-可爱女声 | 101009 |
| `news` | 智琪-新闻女声 | 101008 |
| `magnetic` | 智华-磁性男声 | 101010 |

### 飞书语音处理

```bash
# 接收飞书语音并转换播放
python3 scripts/feishu_voice_morph.py "https://open.feishu.cn/.../audio.opus"

# 指定转换目标声音
python3 scripts/feishu_voice_morph.py "https://open.feishu.cn/.../audio.opus" --voice tong
```

## 📝 在 OpenClaw 中使用

### 场景 1: 飞书语音变声

当通过飞书收到语音消息时：

```yaml
# 假设语音文件已下载到 ~/.openclaw/media/inbound/voice.ogg
exec:
  command: |
    python3 /home/unitree/openclaw/skills/go2-voice-morph/scripts/voice_morph.py \
    ~/.openclaw/media/inbound/voice.ogg \
    --voice girl
```

### 场景 2: 用户命令触发

用户说: "把我的声音变成男生"

```yaml
# 1. 录制用户语音到 /tmp/user_voice.wav
# 2. 转换并播放
exec:
  command: |
    python3 /home/unitree/openclaw/skills/go2-voice-morph/scripts/voice_morph.py \
    /tmp/user_voice.wav \
    --voice tong
```

## 🔊 完整流程示例

### 单人测试流程

```bash
# 1. 先合成一段测试语音
python3 /home/unitree/openclaw/skills/go2-tts/scripts/synthesize.py \
    "你好，测试语音变形功能！" --voice 101003

# 输出: ~/.openclaw/media/tts_tencent_xxx.mp3

# 2. 用 voice morph 转换声音
python3 /home/unitree/openclaw/skills/go2-voice-morph/scripts/voice_morph.py \
    ~/.openclaw/media/tts_tencent_xxx.mp3 \
    --voice tong
```

### 处理流程图解

```
输入音频 (OGG/WAV/MP3)
    ↓
┌─────────────────┐
│  ASR 语音识别    │  ← DashScope qwen3-asr-flash
│  音频 → 文字     │
└─────────────────┘
    ↓
识别文字: "你好，我是勾勾"
    ↓
┌─────────────────┐
│  TTS 语音合成    │  ← 腾讯云 TTS
│  文字 → 新语音   │
│  使用目标音色    │
└─────────────────┘
    ↓
输出音频 (MP3)
    ↓
┌─────────────────┐
│  狗扬声器播放    │  ← go2-audio-play
└─────────────────┘
```

## 📁 文件结构

```
go2-voice-morph/
├── SKILL.md                           # 本文档
└── scripts/
    ├── voice_morph.py                # 核心变形脚本
    └── feishu_voice_morph.py         # 飞书完整流程
```

## 🔍 技术细节

### ASR 部分 (go2-asr)

- **服务**: DashScope 阿里云
- **模型**: qwen3-asr-flash
- **输入**: OGG/MP3/WAV/URL
- **输出**: 中文文字

### TTS 部分 (go2-tts)

- **服务**: 腾讯云
- **模型**: 实时语音合成
- **输入**: 中文文字
- **输出**: MP3 (16000Hz)

### 音频播放 (go2-audio-play)

- **设备**: 24 (狗扬声器)
- **格式**: WAV, 48kHz, 单声道
- **处理**: MP3 → FFmpeg → WAV → 播放

## 🐛 故障排查

### ASR 识别失败
- 检查网络是否能访问阿里云 DashScope
- 检查音频文件是否损坏
- 尝试其他音频格式

### TTS 合成失败
- 检查网络是否能访问腾讯云
- 检查文字长度（建议不超过 500 字符）
- 检查音色 ID 是否正确

### 没有声音输出
1. 测试扬声器：
   ```bash
   python3 /home/unitree/openclaw/skills/go2-audio-play/scripts/play.py \
       /home/unitree/openclaw/skills/go2-audio-play/sounds/dog_barking.wav
   ```

2. 检查音量：
   ```bash
   amixer sget 'I2S2 Playback Audio Channels'
   ```

### FFmpeg 错误
- 安装 ffmpeg: `sudo apt-get install ffmpeg`

## 💰 费用说明

本技能使用现有 API：
- **ASR**: DashScope qwen3-asr-flash（有一定免费额度）
- **TTS**: 腾讯云 TTS（有一定免费额度）

通常个人使用不会产生费用。

## 🔗 相关技能

- `go2-asr`: 语音识别
- `go2-tts`: 语音合成
- `go2-audio-play`: 扬声器播放
- `go2-feishu-audio`: 飞书语音发送

## 📚 参考

- DashScope 文档: https://dashscope.aliyun.com/
- 腾讯云 TTS 文档: https://cloud.tencent.com/document/product/1073
