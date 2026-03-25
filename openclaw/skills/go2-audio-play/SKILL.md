---
name: go2_audio_play
description: Play audio files through Go2 robot dog's speaker using sounddevice. Based on RealtimeAPI SDK implementation.
---

# Go2 Audio Play - 机器狗扬声器播放技能

通过机器狗的扬声器播放音频文件（WAV），使用 sounddevice 库直接控制硬件音频输出。

**完全按照 RealtimeAPI SDK 实现**

## 硬件配置（来自 SDK）

```python
AUDIO_OUTPUT_DEVICE_ID = 24   # 输出设备ID
HW_SAMPLE_RATE = 48000        # 硬件采样率 48kHz
WS_SAMPLE_RATE = 16000        # 网络音频采样率 16kHz
RESAMPLE_RATIO = 3            # 48k / 16k = 3
FRAME_DURATION_MS = 20        # 每帧 20ms
blocksize = 960               # 48000 * 0.02
```

## 使用方法

### 播放声音

```bash
# 播放狗叫声
python3 scripts/play.py dog_barking.wav

# 查看可用声音
python3 scripts/play.py --list
```

### 在 OpenClaw 中使用

```yaml
# 播放狗叫声
exec:
  command: python3 /home/unitree/openclaw/skills/go2-audio-play/scripts/play.py dog_barking.wav
```

## 添加声音文件

将 WAV 文件放入 `sounds/` 目录即可播放。

## 技术细节

### 音频处理流程

1. **读取 WAV 文件**：获取采样率、声道数、采样宽度
2. **格式转换**：
   - 非 16-bit 采样 → 转换为 16-bit
   - 立体声 → 转换为单声道（取平均）
3. **重采样**：
   - 16kHz → 48kHz（SDK 方式：`np.repeat(data, 3)`）
   - 48kHz → 直接使用
   - 其他采样率 → 线性插值
4. **播放**：使用 `sounddevice.OutputStream` 写入设备 24

### 与 SDK 的差异

| 参数 | SDK | 本 Skill |
|------|-----|----------|
| device | 24 | 24 ✓ |
| samplerate | 48000 | 48000 ✓ |
| channels | 1 | 1 ✓ |
| dtype | int16 | int16 ✓ |
| blocksize | 960 | 960 ✓ |
| 重采样 | np.repeat(x, 3) | np.repeat(x, 3) ✓ |

## 文件结构

```
go2-audio-play/
├── SKILL.md                 # 本文档
├── scripts/
│   └── play.py             # 播放脚本
└── sounds/
    └── dog_barking.wav     # 示例声音
```

## 依赖

```bash
pip install sounddevice numpy
```

## 故障排查

### 没有声音

1. **检查混音器设置**：
   ```bash
   # 检查关键混音器
   amixer sget 'DSPK1 Mux'
   amixer sget 'I2S2 Mux'
   amixer sget 'I2S2 Playback Audio Channels'
   ```

2. **检查音量**：
   ```bash
   # 确保音量不是 0
   amixer sget 'DSPK1 Audio Channels'
   amixer sget 'I2S2 Playback Audio Channels'
   ```

3. **检查设备占用**：
   ```bash
   lsof /dev/snd/*
   ```

4. **直接测试 ALSA**：
   ```bash
   aplay -D hw:1,0 sounds/dog_barking.wav
   ```

### 参考

- RealtimeAPI SDK: `/home/unitree/RealtimeAPI-sdk-python/src/audio_client/sdk.py`
- 关键函数: `_play_audio_loop()`
