#!/bin/bash
# 设置每日训练日志定时任务
# 每晚 23:00 执行

CRON_JOB="0 23 * * * /home/unitree/openclaw/skills/go2-training-log/scripts/daily_cron.sh >> /home/unitree/.openclaw/training_logs/cron.log 2>&1"

# 检查是否已存在
crontab -l 2>/dev/null | grep -q "training_log" && {
    echo "定时任务已存在，更新时间为 23:00..."
    crontab -l 2>/dev/null | grep -v "training_log" | crontab -
}

# 添加新任务
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "✅ 定时任务已设置：每晚 23:00 生成训练日志"
echo ""
echo "当前 crontab:"
crontab -l | grep training_log
