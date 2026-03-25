---
name: go2_record_to_feishu
description: Record audio from Go2 microphone and send to Feishu as voice messages. Requires explicit start/stop commands to avoid continuous recording.
---

# Go2 Record to Feishu - 麦克风录音发送到飞书

从机器狗麦克风录音，自动发送到飞书作为语音消息。
**重要：需要显式指令启动/关停，避免持续录音发送消息。**

## 🎯 功能特点

- 🎙️ **麦克风录音**：从机器狗麦克风录制音频
- 📤 **自动发送**：录音完成后自动发送到飞书作为语音消息
- ⏱️ **分段录音**：每段固定时长（默认5秒），循环录制发送
- 🎛️ **启停控制**：需要显式指令启动和停止，防止误触发
- 🔒 **安全设计**：不会自动启动，避免隐私问题和消息轰炸

## 🔧 前置要求

依赖技能：
- `go2-feishu-audio`：发送语音消息到飞书

系统依赖：
```bash
pip3 install sounddevice numpy
```

## 🚀 使用方法

### 基础用法

```bash
# 启动录音（需要显式执行）
python3 scripts/record_and_send.py start

# 停止录音
python3 scripts/record_and_send.py stop

# 查看状态
python3 scripts/record_and_send.py status

# 自定义每段录音时长（比如10秒）
python3 scripts/record_and_send.py start --duration 10
```

### 在 OpenClaw 中使用

**场景1：用户说"启动录音"**

用户："启动录音"

```yaml
exec:
  command: python3 /home/unitree/openclaw/skills/go2-record-to-feishu/scripts/record_and_send.py start
```

**场景2：用户说"关停"**

用户："关停"

```yaml
exec:
  command: python3 /home/unitree/openclaw/skills/go2-record-to-feishu/scripts/record_and_send.py stop
```

## 📋 工作流程

```
用户说"启动录音"
    ↓
开始录音循环
    ↓
┌─────────────────────────────────────────┐
│  🎤 录音 5 秒                           │
│     ↓                                   │
│  📤 自动发送到飞书（语音消息）           │
│     ↓                                   │
│  🔄 继续下一段录音...                    │
└─────────────────────────────────────────┘
    ↓
用户说"关停" → 停止录音循环
```

## ⚙️ 配置参数

| 参数 | 短参数 | 说明 | 默认值 |
|------|--------|------|--------|
| `action` | - | 操作: start/stop/status | 必填 |
| `--device` | `-d` | 麦克风设备ID | 自动查找 |
| `--duration` | `-t` | 每段录音时长（秒） | 5 |
| `--chat-id` | `-c` | 飞书聊天ID | 环境变量 |

## 🛡️ 安全设计

**为什么需要显式启停？**

1. **隐私保护**：不会偷偷录音
2. **避免消息轰炸**：不会无限发送消息到飞书
3. **用户控制**：用户明确知道何时开始/结束

**使用规则**：
- 必须通过文字指令 `"启动录音"` 启动
- 必须通过文字指令 `"关停"` 停止
- 不会自动启动或停止

## 📁 文件结构

```
go2-record-to-feishu/
├── SKILL.md                           # 本文档
└── scripts/
    └── record_and_send.py             # 录音并发送脚本
```

## 🐛 故障排查

### 找不到麦克风
```bash
# 查看可用音频设备
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

### 录音失败
- 检查麦克风权限
- 检查设备ID是否正确

### 发送失败
- 检查 `go2-feishu-audio` 技能是否配置正确
- 检查飞书 App ID 和 Secret

## 🔗 相关技能

- `go2-feishu-audio`：发送语音消息到飞书
- `go2-asr`：语音识别
- `go2-tts`：语音合成

## 💡 使用场景

- 🏠 **远程监控**：出门在外，让机器狗录音传回家
- 🎙️ **语音对讲**：和家里的人语音交流
- 📝 **会议记录**：录下会议内容自动发送到飞书
- 🐕 **宠物互动**：录下狗狗的叫声分享给朋友
