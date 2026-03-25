#!/usr/bin/env python3
"""
Go2 Voice Morph - 语音变形技能
基于 ASR + TTS 实现语音转换

流程: 输入音频 → ASR识别 → 文字 → TTS合成(目标声音) → 输出音频 → 播放
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# 路径配置
ASR_SCRIPT = "/home/unitree/openclaw/skills/go2-asr/scripts/transcribe.py"
TTS_SCRIPT = "/home/unitree/openclaw/skills/go2-tts/scripts/synthesize.py"
PLAY_SCRIPT = "/home/unitree/openclaw/skills/go2-audio-play/scripts/play.py"

# TTS 音色配置
# 默认声音: 101016 - 智瑞 - 沉稳男声 (用户指定)
TTS_VOICES = {
    "default": "101016",     # 智瑞 - 沉稳男声 (默认)
    "man": "101016",         # 智瑞 - 沉稳男声
    "tong": "101017",        # 智萌 - 童声
    "boy": "101018",         # 智栋 - 青年男声
    "girl": "101003",        # 智美 - 活泼女声
    "woman": "101002",       # 智聆 - 标准女声
    "sweet": "101005",       # 智莉 - 甜美女声
    "cute": "101009",        # 智芸 - 可爱女声
    "news": "101008",        # 智琪 - 新闻女声
    "magnetic": "101010",    # 智华 - 磁性男声
}


def transcribe_audio(audio_path: str) -> Optional[str]:
    """
    使用 ASR 识别音频为文字
    
    Args:
        audio_path: 音频文件路径或 URL
    
    Returns:
        识别出的文字，失败返回 None
    """
    try:
        print(f"🎤 正在识别语音...")
        print(f"   输入: {audio_path}")
        
        cmd = ["python3", ASR_SCRIPT, audio_path]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"❌ ASR 识别失败: {result.stderr}")
            return None
        
        # 解析输出，获取识别结果
        # 输出格式: 最后一行是识别结果
        lines = result.stdout.strip().split('\n')
        
        # 查找 "📝 识别结果:" 后面的内容
        text = None
        for i, line in enumerate(lines):
            if "📝 识别结果:" in line:
                # 下一行是实际结果
                if i + 1 < len(lines):
                    text = lines[i + 1].strip()
                    break
        
        # 如果没找到标记，取最后一行非空行
        if not text:
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith(('🎤', '✅', '❌', '📝', '文件大小')):
                    text = line
                    break
        
        if text:
            print(f"✅ 识别成功!")
            print(f"📝 识别结果: {text}")
            return text
        else:
            print(f"❌ 未能解析识别结果")
            print(f"   输出: {result.stdout}")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ ASR 识别超时")
        return None
    except Exception as e:
        print(f"❌ ASR 错误: {e}")
        return None


def synthesize_speech(text: str, voice_id: str = "101017") -> Optional[str]:
    """
    使用 TTS 合成语音
    
    Args:
        text: 要合成的文字
        voice_id: TTS 音色 ID
    
    Returns:
        输出音频文件路径，失败返回 None
    """
    try:
        print(f"🔊 正在合成语音...")
        print(f"   文字: {text[:30]}..." if len(text) > 30 else f"   文字: {text}")
        print(f"   音色: {voice_id}")
        
        cmd = ["python3", TTS_SCRIPT, text, "--voice", voice_id]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"❌ TTS 合成失败: {result.stderr}")
            return None
        
        # 解析输出，获取文件路径
        # 输出格式包含: /home/unitree/.openclaw/media/tts_tencent_xxx.mp3
        output_path = None
        
        for line in result.stdout.split('\n'):
            line = line.strip()
            if 'tts_tencent_' in line and line.endswith('.mp3'):
                # 检查是否是文件路径
                if line.startswith('/') or line.startswith('~/'):
                    output_path = line.replace('~/', os.path.expanduser('~') + '/')
                    break
        
        # 如果没找到，从 stdout 最后一行找
        if not output_path:
            lines = [l.strip() for l in result.stdout.split('\n') if l.strip()]
            for line in reversed(lines):
                if '.mp3' in line and ('media' in line or 'tts' in line):
                    output_path = line.replace('~/', os.path.expanduser('~') + '/')
                    break
        
        if output_path and os.path.exists(output_path):
            print(f"✅ 合成成功!")
            print(f"📁 文件: {output_path}")
            return output_path
        else:
            print(f"❌ 未能获取合成文件路径")
            print(f"   输出: {result.stdout}")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ TTS 合成超时")
        return None
    except Exception as e:
        print(f"❌ TTS 错误: {e}")
        return None


def play_audio(audio_path: str) -> bool:
    """
    使用 go2-audio-play 播放音频
    
    Args:
        audio_path: 音频文件路径
    
    Returns:
        是否成功
    """
    try:
        print(f"🔊 正在播放...")
        
        # 检查是否是 MP3，如果是需要转换
        if audio_path.endswith('.mp3'):
            # 转换为 WAV
            wav_path = f"/tmp/voice_morph_{os.getpid()}.wav"
            
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-af", "adelay=120",  # 添加 120ms 前导静音
                "-ar", "48000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                wav_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if result.returncode != 0:
                print(f"❌ 转换为 WAV 失败: {result.stderr}")
                return False
            
            play_path = wav_path
        else:
            play_path = audio_path
            wav_path = None
        
        # 播放
        subprocess.run(["python3", PLAY_SCRIPT, play_path], check=True)
        
        # 清理临时文件
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 播放失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 播放错误: {e}")
        return False


def voice_morph(
    audio_path: str,
    voice: str = "tong",
    play: bool = True
) -> Optional[str]:
    """
    语音变形主函数
    
    流程: 音频 → ASR → 文字 → TTS → 音频
    
    Args:
        audio_path: 输入音频路径或 URL
        voice: 目标声音名称
        play: 是否自动播放
    
    Returns:
        输出音频文件路径，失败返回 None
    """
    print("=" * 50)
    print("🎭 Go2 Voice Morph - 语音变形")
    print("=" * 50)
    print()
    
    # Step 1: ASR 识别
    text = transcribe_audio(audio_path)
    if not text:
        print("❌ 语音识别失败，流程终止")
        return None
    
    print()
    
    # Step 2: TTS 合成
    voice_id = TTS_VOICES.get(voice.lower(), voice)
    output_path = synthesize_speech(text, voice_id)
    if not output_path:
        print("❌ 语音合成失败，流程终止")
        return None
    
    print()
    
    # Step 3: 播放（可选）
    if play:
        play_audio(output_path)
    
    print()
    print("=" * 50)
    print("✅ Voice Morph 完成!")
    print("=" * 50)
    
    return output_path


def list_voices():
    """列出可用的声音"""
    print("🎭 可用声音列表:")
    print()
    print(f"  {'名称':<12} {'描述':<15} {'音色ID'}")
    print(f"  {'-'*12} {'-'*15} {'-'*10}")
    descriptions = {
        "default": "智瑞-沉稳男声（默认）",
        "man": "智瑞-沉稳男声",
        "tong": "智萌-童声",
        "boy": "智栋-青年男声",
        "girl": "智美-活泼女声",
        "woman": "智聆-标准女声",
        "sweet": "智莉-甜美女声",
        "cute": "智芸-可爱女声",
        "news": "智琪-新闻女声",
        "magnetic": "智华-磁性男声",
    }
    for name, voice_id in TTS_VOICES.items():
        desc = descriptions.get(name, "")
        print(f"  {name:<12} {desc:<15} {voice_id}")
    print()
    print("使用: --voice tong/boy/man/girl/...")


def main():
    parser = argparse.ArgumentParser(
        description="语音变形 - ASR + TTS 实现声音转换"
    )
    parser.add_argument(
        "audio",
        nargs="?",
        help="输入音频文件路径或 URL"
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
    parser.add_argument(
        "--list-voices", "-l",
        action="store_true",
        help="列出可用声音"
    )
    
    args = parser.parse_args()
    
    # 列出声音
    if args.list_voices:
        list_voices()
        return
    
    # 检查音频文件
    if not args.audio:
        parser.print_help()
        sys.exit(1)
    
    # 执行语音变形
    result = voice_morph(
        audio_path=args.audio,
        voice=args.voice,
        play=not args.no_play
    )
    
    if result:
        print(f"\n📁 输出文件: {result}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
