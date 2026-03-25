---
name: go2_follow_person
description: Automatically follow a person using camera detection and movement control. The robot dog captures images, detects person's position, and moves accordingly to maintain following distance.
---

# Go2 Follow Person - 自动跟随人

让机器狗自动识别人并跟随移动！

## 功能特点

- 📷 **实时视觉**：持续拍照识别人在画面中的位置
- 🎯 **智能跟踪**：根据人位置自动调整移动方向
- 🔄 **自动搜索**：如果丢失目标，自动转圈搜索
- ⚡ **距离保持**：维持理想跟随距离

## 工作原理

```
拍照 → 检测人位置 → 决策 → 移动 → 循环
```

**位置判断：**
- 人在画面 **左边** → 机器狗 **左转**
- 人在画面 **右边** → 机器狗 **右转**
- 人在画面 **中间但远** → 机器狗 **前进**
- 人在画面 **中间且近** → 机器狗 **停止**

## 使用方法

### 基础用法

```bash
# 启动自动跟随模式
python3 scripts/follow_person.py

# 测试模式（只拍照分析，不移动）
python3 scripts/follow_person.py --test

# 跟随30秒后自动停止
python3 scripts/follow_person.py --duration 30
```

### 在 OpenClaw 中使用

```yaml
# 启动自动跟随
exec:
  command: python3 /home/unitree/openclaw/skills/go2-follow-person/scripts/follow_person.py
```

## 参数配置

在 `follow_person.py` 中可以调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CENTER_THRESHOLD` | 0.2 | 中心区域阈值（画面宽度的20%）|
| `FOLLOW_DISTANCE` | 0.3 | 理想跟随距离（画面高度的30%）|
| `MOVE_INTERVAL` | 2.0 | 移动间隔（秒）|
| `CAPTURE_INTERVAL` | 1.0 | 拍照间隔（秒）|

## 技术实现

### 当前版本
- 使用 **混合检测器** (HOG + VLM)
  - **HOG**：OpenCV 内置，毫秒级速度，适合全身检测
  - **VLM**：视觉语言模型，秒级速度，能检测部分身体
  - **策略**：先用 HOG 快速检测，失败再用 VLM 精确检测
- 基于人在画面中的 **相对位置** 决策

### 待优化（TODO）
- [x] 集成目标检测模型（HOG + VLM 已完成）
- [ ] 深度估计（判断实际距离）
- [ ] 多目标跟踪（跟特定的人）
- [ ] 避障功能（跟随时不撞到东西）
- [ ] 语音交互（"跟我来"指令）

## 安全提醒

⚠️ **使用时请注意：**
- 确保周围空间充足
- 远离楼梯和障碍物
- 随时准备手动停止（Ctrl+C）
- 首次使用建议在开阔平地测试

## 依赖

- go2-vision (拍照)
- unitree-go2 (运动控制)
- OpenCV (可选，用于高级检测)

## 文件结构

```
go2-follow-person/
├── SKILL.md                    # 本文档
├── scripts/
│   └── follow_person.py        # 主程序
└── models/                     # (可选) 检测模型
    └── yolo_person.pt
```

## 示例场景

### 场景1：跟我去会议室
```
用户："跟我走"
→ 启动跟随模式
→ 机器狗识别人并跟随
→ 到达后说"停"（或自动停止）
```

### 场景2：自动巡逻
```
用户："跟着我在办公室转一圈"
→ 机器狗跟随用户行走
→ 同时可以拍照记录
```

## 故障排除

| 问题 | 解决 |
|------|------|
| 识别不到人 | 确保光线充足，人进入画面 |
| 移动太频繁 | 增大 MOVE_INTERVAL |
| 跟随距离不合适 | 调整 FOLLOW_DISTANCE |
| 转弯角度太大 | 减小转弯角度参数 |

## 未来扩展

- 🤖 **手势控制**：举手=停止，挥手=跟我走
- 🗣️ **语音跟随**：说"跟我来"自动启动
- 👥 **多人识别**：选择跟随特定的人
- 🗺️ **路径记忆**：记住跟随路线并重复

---

🐕 **让机器狗成为你的忠实跟班！**
