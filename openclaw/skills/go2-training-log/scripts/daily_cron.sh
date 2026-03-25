#!/bin/bash
# 每日训练日志定时任务脚本
# 每晚 23:00 执行

cd /home/unitree/openclaw/skills/go2-training-log

# 生成今日总结
python3 scripts/training_log.py --generate

# 发送飞书消息通知主人确认
# 这里会调用 OpenClaw 的消息功能
