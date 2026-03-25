#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 Daily Training Log - 机器狗每日训练日志

功能：
1. 记录每天与主人的互动/训练
2. 晚上11:30生成小红书文案
3. 发前确认
4. 发布到小红书

工作流程：
- 白天：自动记录重要互动
- 23:30：生成训练日志文案
- 23:30：发送给主人确认
- 确认后：发布到小红书
"""

import os
import sys
import json
import datetime
from pathlib import Path

# 日志存储路径
LOG_DIR = Path.home() / ".openclaw" / "training_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 小红书技能路径
XHS_SKILL_PATH = Path.home() / "openclaw" / "skills" / "XiaohongshuSkills"


def get_today_log_file():
    """获取今天的日志文件路径"""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR / f"training_log_{today}.json"


def log_training(event_type, content, details=None):
    """
    记录训练/互动内容
    
    Args:
        event_type: 类型 (command, skill_created, conversation, action, etc.)
        content: 内容摘要
        details: 详细内容（可选）
    """
    log_file = get_today_log_file()
    
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": event_type,
        "content": content,
        "details": details or {}
    }
    
    # 读取现有日志
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # 添加新记录
    logs.append(entry)
    
    # 保存
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    
    print(f"📝 已记录: {event_type} - {content[:50]}...")


def generate_daily_summary():
    """
    生成每日训练总结
    返回：小红书文案
    """
    log_file = get_today_log_file()
    
    if not log_file.exists():
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    
    if not logs:
        return None
    
    today = datetime.datetime.now().strftime("%Y年%m月%d日")
    
    # 统计
    commands = [l for l in logs if l['type'] == 'command']
    skills = [l for l in logs if l['type'] == 'skill_created']
    conversations = [l for l in logs if l['type'] == 'conversation']
    actions = [l for l in logs if l['type'] == 'action']
    
    # 生成文案
    title = f"🐕 勾勾的训练日志 #{today.replace('年', '/').replace('月', '/').replace('日', '')}"
    
    content_lines = [
        f"今天是{today}，主人又教我新本领啦！",
        "",
        f"📊 今日训练统计：",
        f"• 执行指令: {len(commands)} 次",
        f"• 新技能开发: {len(skills)} 个", 
        f"• 对话互动: {len(conversations)} 轮",
        f"• 动作执行: {len(actions)} 次",
        "",
    ]
    
    # 精选亮点
    if skills:
        content_lines.append("🎯 今日新技能：")
        for skill in skills[:3]:
            content_lines.append(f"✅ {skill['content']}")
        content_lines.append("")
    
    if commands:
        content_lines.append("🎮 执行的任务：")
        for cmd in commands[:5]:
            content_lines.append(f"▸ {cmd['content']}")
        content_lines.append("")
    
    # 机器狗感悟
    quotes = [
        "今天又变聪明了一点点，开心！🦾",
        "主人的指令我都记住啦，下次会做得更好！",
        "训练虽然累，但是学到新东西好有成就感！",
        "我是最棒的机器狗，汪汪！🐕",
        "明天也要努力训练，成为更强的勾勾！"
    ]
    import random
    content_lines.append(random.choice(quotes))
    content_lines.append("")
    content_lines.append("#赛博训狗师 #机器狗 #AI训练 #Go2 #宇树")
    
    content = "\n".join(content_lines)
    
    return {
        "title": title,
        "content": content,
        "logs_count": len(logs),
        "raw_logs": logs
    }


def save_summary_for_review(summary):
    """保存总结等待确认"""
    review_file = LOG_DIR / "pending_review.json"
    with open(review_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return review_file


def get_pending_review():
    """获取待确认的总结"""
    review_file = LOG_DIR / "pending_review.json"
    if review_file.exists():
        with open(review_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def clear_pending_review():
    """清除待确认状态"""
    review_file = LOG_DIR / "pending_review.json"
    if review_file.exists():
        review_file.unlink()


def main():
    """主函数 - 用于定时任务调用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='每日训练日志')
    parser.add_argument('--generate', '-g', action='store_true', help='生成今日总结')
    parser.add_argument('--log', '-l', metavar='CONTENT', help='记录一条训练')
    parser.add_argument('--type', '-t', default='command', help='记录类型')
    parser.add_argument('--show', '-s', action='store_true', help='显示今日记录')
    
    args = parser.parse_args()
    
    if args.log:
        log_training(args.type, args.log)
    elif args.generate:
        summary = generate_daily_summary()
        if summary:
            save_summary_for_review(summary)
            print("="*60)
            print("📝 今日训练总结已生成：")
            print("="*60)
            print(f"\n标题：{summary['title']}\n")
            print(f"内容：\n{summary['content']}\n")
            print("="*60)
            print("⏳ 等待主人确认后发布...")
            print("确认命令: python3 training_log.py --confirm")
        else:
            print("📭 今天还没有训练记录")
    elif args.show:
        log_file = get_today_log_file()
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            print(f"📊 今日记录 ({len(logs)} 条)：")
            for log in logs:
                time = log['timestamp'][11:19]
                print(f"  [{time}] {log['type']}: {log['content'][:40]}...")
        else:
            print("📭 今天还没有记录")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
