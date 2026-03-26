---
name: go2_guard_dog
description: A guard dog skill for Go2 robot - barks at strangers approaching the door, retreats when they get too close, stops barking when they leave.
---

# Go2 Guard Dog - 看家狗

让机器狗成为忠实的看家狗！当陌生人靠近时发出警告，太靠近时后退并保持警惕。

## 功能特点

🐕 **智能警戒模式**
- 👀 **2米警戒**：发现陌生人靠近（2米）→ 汪汪叫警告
- ⚠️ **1米危险**：陌生人太近（<1米）→ 慢慢后退 + 持续警告
- ✅ **解除警戒**：陌生人走远（>2米）→ 停止叫声，恢复平静

🎯 **YOLO 人体检测**
- 使用 YOLOv8 实时检测画面中的人
- 通过 BoT-SORT 算法追踪人员
- 根据人在画面中的位置估算距离

🔊 **语音警告**
- 使用机器狗扬声器播放逼真的狗叫声
- 非阻塞式音频播放，不影响视觉检测

## 工作原理

```
持续拍摄 → YOLO检测人员 → 估算距离 → 决策 → 执行动作
                          ↓
              ┌─────────────────────┐
              │ 距离 > 2米          │ → 停止叫声，监视状态
              │ 距离 1-2米          │ → 持续汪汪叫（警告）
              │ 距离 < 1米          │ → 后退 + 汪汪叫（危险）
              └─────────────────────┘
```

**距离估算：**
- 基于人员在画面中的高度（y2 坐标）
- y2 越大（越靠近画面底部）→ 距离越近
- 经过校准映射到实际距离（米）

## 使用方法

### 命令行启动

```bash
# 启动看家模式
python3 scripts/guard_dog.py

# 调试模式（只显示检测，不移动/不叫）
python3 scripts/guard_dog.py --test

# 自定义距离阈值
python3 scripts/guard_dog.py --warn-distance 2.0 --danger-distance 0.8

# 显示帮助
python3 scripts/guard_dog.py --help
```

### OpenClaw 中使用

```yaml
# 启动看家模式
exec:
  command: python3 /home/unitree/RealVLN/openclaw/skills/go2-guard-dog/scripts/guard_dog.py
```

## 参数配置

在 `guard_dog.py` 中可以调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `WARN_DISTANCE` | 2.0 | 警戒距离（米），超过此距离不叫 |
| `DANGER_DISTANCE` | 1.0 | 危险距离（米），小于此距离后退 |
| `BARK_INTERVAL` | 2.0 | 叫声间隔（秒）|
| `RETREAT_SPEED` | -0.3 | 后退速度（负值为后退）|

## 状态机

```
          ┌───────────┐
          │   IDLE    │ ←── 初始状态，监视中
          │  (平静)   │
          └─────┬─────┘
                │ 检测到人员
                │ 距离 > WARN_DISTANCE
                ↓
          ┌───────────┐
          │  WATCHING │ ←── 有人但很远，不动作
          │  (观察)   │
          └─────┬─────┘
                │ 距离 <= WARN_DISTANCE
                │ 距离 > DANGER_DISTANCE
                ↓
          ┌───────────┐     距离继续减小
          │  BARKING  │ ←──────────────────┐
          │  (警告)   │                     │
          │ 汪汪叫     │                     │
          └─────┬─────┘                     │
                │ 距离 <= DANGER_DISTANCE   │
                ↓                           │
          ┌───────────┐                     │
          │ RETREATING│ ────────────────────┘
          │ (后退警告)│ 距离增加到安全范围
          │ 后退+汪汪叫│
          └─────┬─────┘
                │ 距离 > WARN_DISTANCE
                ↓
          ┌───────────┐
          │   IDLE    │
          └───────────┘
```

## 文件结构

```
go2-guard-dog/
├── SKILL.md                    # 本文档
├── scripts/
│   └── guard_dog.py           # 主程序
└── sounds/
    └── dog_barking.wav        # 狗叫声（从 go2-audio-play 复用）
```

## 依赖

- unitree-sdk2py（宇树 SDK）
- ultralytics（YOLO 检测）
- opencv-python（图像处理）
- numpy
- sounddevice（音频播放）

## 技术细节

### 距离映射校准

由于机器狗没有深度传感器，距离通过画面中人像的高度估算：

```python
# 经验值校准（根据实际测试调整）
# y2_pos: 人体底部在画面中的相对位置 (0.0~1.0)
# 映射到实际距离（米）
DISTANCE_MAP = {
    0.70: 3.0,   # y2=0.70 → 约3米远
    0.80: 2.0,   # y2=0.80 → 约2米（警戒）
    0.90: 1.0,   # y2=0.90 → 约1米（危险）
    0.95: 0.5,   # y2=0.95 → 约0.5米（太近）
}
```

### 音频播放

- 使用独立的音频线程，避免阻塞主视觉循环
- 支持重叠播放（正在叫的时候触发新的叫声会排队）
- 音频文件复用 `go2-audio-play/sounds/dog_barking.wav`

### 运动控制

- 后退使用平滑过渡（EMA 滤波）
- 后退时保持面对目标（原地后退）
- 停止时立即刹车

## 安全提醒

⚠️ **使用时请注意：**
- 确保机器狗站在稳定的地面上
- 测试时从远距离开始，逐步靠近
- 确保后退路径上没有障碍物
- 随时准备手动停止（Ctrl+C）
- 首次使用建议在开阔平地测试

## 故障排除

| 问题 | 解决 |
|------|------|
| 检测不到人 | 确保光线充足，人员全身入镜 |
| 距离判断不准 | 调整 `DISTANCE_MAP` 校准值 |
| 叫声不响 | 检查扬声器音量，`amixer sget 'I2S2 Playback Audio Channels'` |
| 后退时摇晃 | 减小 `RETREAT_SPEED` 或增大 `SMOOTH_FACTOR` |
| 频繁误触发 | 增大 `CONFIDENCE_THRESHOLD` 或调整距离阈值 |

## 未来扩展

- 🤖 **人脸识别**：识别家人 vs 陌生人
- 🗣️ **语音播报**：除了狗叫还可以说"请离开"
- 📹 **录像取证**：检测到陌生人时自动录像
- 📱 **远程通知**：发送警报到手机
- 🌙 **夜视模式**：低光环境下的检测

---

🐕 **让机器狗守护你的家！**
