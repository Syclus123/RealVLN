#!/usr/bin/env python3
"""
飞书语音消息 → Voice Morph → 狗扬声器播放

完整流程:
1. 从飞书下载语音消息
2. ASR 识别为文字
3. TTS 用目标声音合成新语音
4. 狗扬声器播放

使用方法:
    python3 scripts/feishu_voice_morph.py <audio_url> [--voice <voice_name>]
"""

import os
import sys
import argparse
import requests
import subprocess

MORPH_SCRIPT = "/home/unitree/openclaw/skills/go2-voice-morph/scripts/voice_morph.py"
DOWNLOAD_DIR = "/tmp/feishu_audio"


def download_audio(url: str, output_path: str) -> bool:
    """下载音频文件"""
    try:
        print(f"📥 正在下载飞书语音...")
        print(f"   URL: {url[:50]}...")
        
        response = requests.get(url, timeout=30, stream=True)
        
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ 下载成功")
            return True
        else:
            print(f"❌ 下载失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 下载错误: {e}")
        return False


def process_feishu_voice(
    audio_url: str,
    voice: str = "default",
    play: bool = True
) -> bool:
    """
    处理飞书语音消息的完整流程
    
    Args:
        audio_url: 飞书语音文件下载 URL
        voice: 目标声音名称
        play: 是否自动播放
    
    Returns:
        是否成功
    """
    # 创建下载目录
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # 下载文件
    input_path = f"{DOWNLOAD_DIR}/feishu_{os.getpid()}.opus"
    
    if not download_audio(audio_url, input_path):
        return False
    
    print()
    
    # 执行 voice morph
    cmd = [
        "python3", MORPH_SCRIPT,
        input_path,
        "--voice", voice
    ]
    
    if not play:
        cmd.append("--no-play")
    
    result = subprocess.run(cmd)
    
    # 清理输入文件
    if os.path.exists(input_path):
        os.remove(input_path)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="飞书语音 → Voice Morph (ASR+TTS) → 播放"
    )
    parser.add_argument(
        "url",
        help="飞书语音文件下载 URL"
    )
    parser.add_argument(
        "--voice", "-v",
        default="default",
        help="目标声音 (default/man/tong/boy/girl/woman/sweet/cute/news/magnetic)。默认: 101016智瑞"
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="只转换，不播放"
    )
    
    args = parser.parse_args()
    
    success = process_feishu_voice(
        audio_url=args.url,
        voice=args.voice,
        play=not args.no_play
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
