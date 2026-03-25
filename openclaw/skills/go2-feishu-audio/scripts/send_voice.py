#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 Feishu Audio - 发送飞书语音消息（完整流程）
1. 转换音频格式为 Opus
2. 上传到飞书
3. 发送语音消息

使用方法：
    python3 send_voice.py /path/to/audio.mp3 --chat-id ou_xxx
"""
import sys
import os
import argparse

# 导入同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from send_audio import FeishuAudioSender, get_chat_id_from_env
from convert_opus import convert_to_opus


def send_voice_message(audio_path, chat_id=None, duration=0, app_id=None, app_secret=None):
    """
    发送语音消息的完整流程
    
    Args:
        audio_path: 音频文件路径（支持 mp3, wav, opus 等）
        chat_id: 目标聊天 ID（默认从环境变量获取）
        duration: 音频时长（秒）
        app_id: 飞书 App ID
        app_secret: 飞书 App Secret
    
    Returns:
        成功返回 True，失败返回 False
    """
    # 1. 转换音频格式
    opus_path = convert_to_opus(audio_path)
    if not opus_path:
        print("❌ 音频格式转换失败", file=sys.stderr)
        return False
    
    # 2. 获取 chat_id
    if not chat_id:
        chat_id = get_chat_id_from_env()
    
    if not chat_id:
        print("❌ 请提供 chat_id 或设置环境变量", file=sys.stderr)
        return False
    
    # 3. 发送语音消息
    sender = FeishuAudioSender(app_id=app_id, app_secret=app_secret)
    return sender.send_audio(opus_path, chat_id, duration)


def main():
    parser = argparse.ArgumentParser(description="发送飞书语音消息（完整流程）")
    parser.add_argument("audio", help="音频文件路径（mp3/wav/opus）")
    parser.add_argument("--chat-id", "-c", help="目标聊天 ID（默认从环境变量获取）")
    parser.add_argument("--duration", "-d", type=int, default=0, help="音频时长（秒）")
    parser.add_argument("--app-id", help="飞书 App ID")
    parser.add_argument("--app-secret", help="飞书 App Secret")
    
    args = parser.parse_args()
    
    success = send_voice_message(
        audio_path=args.audio,
        chat_id=args.chat_id,
        duration=args.duration,
        app_id=args.app_id,
        app_secret=args.app_secret
    )
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
