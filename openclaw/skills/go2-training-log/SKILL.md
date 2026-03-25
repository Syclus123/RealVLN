---
name: go2_training_log
description: 机器狗每日训练日志，自动记录互动并生成小红书内容
---

# Go2 Training Log - 每日训练日志

## 功能

自动记录主人与机器狗的每日互动，生成小红书风格的训练日志，定时发布。

## 工作流程

```
白天互动 → 自动记录 → 23:30生成总结 → 主人确认 → 发布小红书
```

## 记录类型

- `command` - 执行的命令
- `skill_created` - 创建的新技能
- `skill_learned` - 学习的功能
- `action` - 执行的动作
- `conversation` - 对话互动
- `task_created` - 创建的任务

## 定时任务

**每晚 23:00**（晚上 11 点）：
1. 生成当日训练总结
2. 生成小红书文案
3. 发送飞书消息给主人确认
4. 等待确认后发布

## 使用方法

### 手动记录
```bash
python3 scripts/training_log.py --log "内容" --type command
```

### 查看今日记录
```bash
python3 scripts/training_log.py --show
```

### 生成总结（测试）
```bash
python3 scripts/training_log.py --generate
```

## 文件结构

```
go2-training-log/
├── SKILL.md
├── scripts/
│   ├── training_log.py      # 主程序
│   └── daily_cron.sh        # 定时任务脚本
└── logs/                     # 日志存储（自动创建）
    ├── training_log_2026-03-07.json
    └── pending_review.json
```

## 配置

需要在 OpenClaw 中设置定时任务：
```cron
30 23 * * * /home/unitree/openclaw/skills/go2-training-log/scripts/daily_cron.sh
```

## 小红书发布流程

1. 生成文案后保存到待确认文件
2. 发送飞书消息给主人，包含文案预览
3. 主人回复"确认"后发布
4. 主人回复"修改"则等待新指令
5. 主人回复"取消"则跳过今日发布

## 文案风格

以机器狗勾勾的第一人称口吻：
- 可爱、忠诚、积极
- 记录学到的技能和执行的命令
- 表达对主人的感谢
- 使用 emoji 和网络用语

## 示例文案

```
🐕 勾勾的训练日志 #2026/03/07

今天是2026年03月07日，主人又教我新本领啦！

📊 今日训练统计：
• 执行指令: 5 次
• 新技能开发: 2 个
• 对话互动: 12 轮

🎯 今日新技能：
✅ 自动跟随人技能
✅ 小红书每日日志发布

🎮 执行的任务：
▸ 转圈拍照识别人
▸ 播放音乐
▸ 查看小红书状态

今天又变聪明了一点点，开心！🦾

#赛博训狗师 #机器狗 #AI训练 #Go2 #宇树
```
