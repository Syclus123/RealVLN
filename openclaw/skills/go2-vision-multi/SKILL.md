---
name: go2_vision_multi
description: Capture multiple images from Go2 robot dog's front camera while rotating step-by-step (stop-then-capture), save to timestamped folder, and analyze with VLM for comprehensive 360° scene summary.
---

# Go2 Vision Multi - 转圈拍照与 VLM 总结

**改进版**：转一点 → 停下来 → 拍照 → 继续转

避免边转边拍导致的照片模糊问题，每张照片都在静止状态下拍摄，保证清晰度！

**工作流程：**
1. 🔄 **旋转** → 转动一定角度
2. ⏹️ **停止** → 等待稳定
3. 📷 **拍照** → 静止状态拍摄（清晰）
4. 🔁 **重复** → 直到转完一圈

这样每张照片都清晰锐利，VLM 分析更准确！

## 功能特点

- 📸 **清晰拍摄**：转-停-拍模式，每张照片都在静止状态下拍摄，不会模糊
- 📁 **时间戳文件夹**：每次拍摄创建独立文件夹，便于管理
- 🤖 **VLM 多图分析**：支持将多张照片一起传给 VLM 进行综合总结
- ⚡ **灵活配置**：支持自定义网络接口、照片数量、旋转角度
- 🔄 **自动回位**：拍摄完成后自动转回起始方向

## 使用方法

### 基础用法 - 转圈拍照（转-停-拍）

```bash
# 转一圈拍8张照片（默认360度，每45度一张）
python3 scripts/capture_multi.py

# 右转一圈拍摄
python3 scripts/capture_multi.py --angle -360

# 拍摄 12 张照片（每30度一张）
python3 scripts/capture_multi.py --count 12

# 只拍4张（每90度一张）
python3 scripts/capture_multi.py --count 4

# 指定网络接口
python3 scripts/capture_multi.py --network eth0
```

### 完整示例 - 拍摄 + VLM 总结

**场景**：用户说 "转一圈并拍摄多张照片，总结一下周围有什么"

```yaml
# 1. 执行转圈
exec:
  command: python3 /home/unitree/openclaw/skills/unitree-go2/scripts/go2_control.py turn_left --angle 360

# 2. 连续拍摄多张照片（转圈过程中）
exec:
  command: python3 /home/unitree/openclaw/skills/go2-vision-multi/scripts/capture_multi.py --count 8 --interval 0.5

# 3. 使用 image 工具分析多张照片（VLM 会收到所有图片）
image:
  images:
    - "~/.openclaw/media/go2_capture_20260306_164830/go2_001.jpg"
    - "~/.openclaw/media/go2_capture_20260306_164830/go2_002.jpg"
    - "~/.openclaw/media/go2_capture_20260306_164830/go2_003.jpg"
    - "~/.openclaw/media/go2_capture_20260306_164830/go2_004.jpg"
    - "~/.openclaw/media/go2_capture_20260306_164830/go2_005.jpg"
    - "~/.openclaw/media/go2_capture_20260306_164830/go2_006.jpg"
    - "~/.openclaw/media/go2_capture_20260306_164830/go2_007.jpg"
    - "~/.openclaw/media/go2_capture_20260306_164830/go2_008.jpg"
  prompt: "这系列照片是我转圈过程中拍摄的，请综合分析所有照片，总结周围的环境、人物、物体等。注意照片之间的关联性和连续性。"

# 4. 发送所有照片给用户
message:
  action: send
  media: "~/.openclaw/media/go2_capture_20260306_164830/go2_001.jpg"
  
message:
  action: send  
  media: "~/.openclaw/media/go2_capture_20260306_164830/go2_002.jpg"
# ... 以此类推
```

## 文件保存结构

```
~/.openclaw/media/
├── go2_camera.jpg                    # 单张拍摄（旧技能）
└── go2_capture_20260306_164830/      # 多拍文件夹（新技能）
    ├── go2_001.jpg
    ├── go2_002.jpg
    ├── go2_003.jpg
    ├── go2_004.jpg
    └── go2_005.jpg
```

## 参数说明

| 参数 | 短参数 | 默认值 | 说明 |
|------|--------|--------|------|
| `--count` | `-c` | 8 | 拍摄照片数量 |
| `--angle` | `-a` | 360 | 旋转角度（正数左转，负数右转） |
| `--network` | `-n` | eth0 | 网络接口 |

**计算方式：**
- 每步旋转角度 = `angle / count`
- 例如：8张照片转360度 = 每步转45度

## VLM 多图分析技巧

### 提示词建议

**场景总结**：
```
这系列照片是我在转圈过程中连续拍摄的，请：
1. 分析每张照片的内容
2. 找出照片之间的关联
3. 综合描述周围的整体环境
4. 列出看到的人物、物体和场景
```

**人物追踪**：
```
这些照片拍摄于不同角度，请帮我：
1. 识别是否有人物出现在多张照片中
2. 追踪他们的位置和动作变化
3. 总结这个空间的人员活动情况
```

**环境映射**：
```
我通过转圈拍摄了这系列照片，请帮我：
1. 构建周围环境的 360° 全景描述
2. 指出各个方向的物体和特征
3. 描述空间的布局和结构
```

## 与单张拍摄的区别

| 特性 | go2-vision (单张) | go2-vision-multi (多张) |
|------|------------------|-------------------------|
| 用途 | 快速抓拍当前画面 | 连续记录、全景扫描 |
| 保存位置 | 单文件，覆盖保存 | 时间戳文件夹，保留历史 |
| VLM 使用 | 单图分析 | 多图联合分析 |
| 典型场景 | "拍张照片" | "转一圈看看周围" |

## 完整实战示例

### 场景：办公室 360° 扫描

用户指令："在办公室转一圈，多拍几张照片，看看都有谁在"

**执行流程**（转-停-拍模式，照片清晰）：

```yaml
# 步骤 1：转圈拍照（自动转一点→停→拍→继续）
exec:
  command: python3 /home/unitree/openclaw/skills/go2-vision-multi/scripts/capture_multi.py --count 8

# 步骤 2：获取生成的文件夹路径（从输出中提取）
# 例如：/home/unitree/.openclaw/media/go2_capture_20260306_165500

# 步骤 3：VLM 分析所有照片
image:
  images:
    - "~/.openclaw/media/go2_capture_20260306_165500/go2_001.jpg"
    - "~/.openclaw/media/go2_capture_20260306_165500/go2_002.jpg"
    - "~/.openclaw/media/go2_capture_20260306_165500/go2_003.jpg"
    - "~/.openclaw/media/go2_capture_20260306_165500/go2_004.jpg"
    - "~/.openclaw/media/go2_capture_20260306_165500/go2_005.jpg"
    - "~/.openclaw/media/go2_capture_20260306_165500/go2_006.jpg"
    - "~/.openclaw/media/go2_capture_20260306_165500/go2_007.jpg"
    - "~/.openclaw/media/go2_capture_20260306_165500/go2_008.jpg"
  prompt: "这是我在办公室转圈过程中拍摄的 8 张照片，每张照片都是在静止状态下拍摄的，很清晰。请综合分析：1) 每张照片中的主要内容；2) 识别出现在多张照片中的同一个人；3) 总结办公室的人员分布和活动情况。"

# 步骤 4：回复用户分析结果，并发送代表性照片
```

## Feishu 消息发送

多张照片可以通过多次调用 `message` 工具发送：

```yaml
# 发送第一张照片
message:
  action: send
  media: "~/.openclaw/media/go2_capture_20260306_165500/go2_001.jpg"

# 发送第二张照片  
message:
  action: send
  media: "~/.openclaw/media/go2_capture_20260306_165500/go2_002.jpg"

# ... 依此类推
```

或使用批量路径（如果 message 工具支持）：

```yaml
message:
  action: send
  paths:
    - "~/.openclaw/media/go2_capture_20260306_165500/go2_001.jpg"
    - "~/.openclaw/media/go2_capture_20260306_165500/go2_002.jpg"
```

## 注意事项

1. **存储空间**：连续拍摄会占用更多存储空间，定期清理旧文件夹
2. **照片清晰**：新模式「转-停-拍」保证每张照片都在静止状态下拍摄，不会模糊
3. **拍摄时间**：总时长 ≈ 旋转时间 + 拍照时间 × 张数，8张照片约需 30-40 秒
4. **VLM 限制**：部分 VLM 模型有图片数量限制，建议单次分析不超过 10 张照片
5. **稳定性**：每次拍照前等待 0.3 秒稳定，确保照片清晰锐利

## 依赖

- Python 3.6+
- unitree_sdk2py
- OpenClaw 环境

## 参考

- 单张拍摄技能：`/home/unitree/openclaw/skills/go2-vision/`
- Go2 控制技能：`/home/unitree/openclaw/skills/unitree-go2/`
