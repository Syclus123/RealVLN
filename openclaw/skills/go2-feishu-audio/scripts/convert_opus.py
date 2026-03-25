#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频格式转换工具
将 MP3/WAV 转换为飞书支持的 Opus 格式
"""
import os
import sys
import subprocess
import tempfile


def convert_to_opus(input_path, output_path=None):
    """
    将音频文件转换为 Opus 格式（飞书语音消息要求）
    
    Args:
        input_path: 输入音频文件路径
        output_path: 输出 opus 文件路径（默认自动生成）
    
    Returns:
        opus 文件路径，失败返回 None
    """
    input_path = os.path.expanduser(input_path)
    
    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}", file=sys.stderr)
        return None
    
    # 自动生成输出路径
    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}.opus"
    
    output_path = os.path.expanduser(output_path)
    
    print(f"🔄 正在转换音频格式...")
    print(f"   输入: {input_path}")
    print(f"   输出: {output_path}")
    
    # 尝试使用 ffmpeg 转换
    try:
        # ffmpeg -i input.mp3 -c:a libopus -b:a 24k output.opus
        cmd = [
            "ffmpeg",
            "-y",  # 覆盖输出文件
            "-i", input_path,
            "-c:a", "libopus",
            "-b:a", "24k",
            "-ar", "16000",  # 采样率 16kHz
            "-ac", "1",      # 单声道
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"✅ 转换成功: {output_path} ({size} bytes)")
            return output_path
        else:
            print(f"⚠️ ffmpeg 转换失败: {result.stderr}", file=sys.stderr)
            # 继续尝试其他方法
            
    except FileNotFoundError:
        print(f"⚠️ ffmpeg 未安装，尝试其他方法...", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ 转换错误: {e}", file=sys.stderr)
    
    # 尝试使用 opusenc
    try:
        cmd = [
            "opusenc",
            "--bitrate", "24",
            input_path,
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"✅ 转换成功 (opusenc): {output_path} ({size} bytes)")
            return output_path
        else:
            print(f"⚠️ opusenc 转换失败: {result.stderr}", file=sys.stderr)
            
    except FileNotFoundError:
        print(f"⚠️ opusenc 未安装", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ 转换错误: {e}", file=sys.stderr)
    
    # 如果没有转换工具，直接复制文件（飞书可能也支持其他格式）
    print(f"⚠️ 没有找到音频转换工具，尝试直接上传原文件...")
    return input_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="音频格式转换为 Opus")
    parser.add_argument("input", help="输入音频文件")
    parser.add_argument("--output", "-o", help="输出 opus 文件路径")
    
    args = parser.parse_args()
    
    result = convert_to_opus(args.input, args.output)
    
    if result:
        print(result)
        sys.exit(0)
    else:
        sys.exit(1)
