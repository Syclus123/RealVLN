#!/usr/bin/env python3
"""
Go2 Record to Feishu - 录音并发送到飞书
从麦克风录音，自动发送到飞书作为语音消息
"""
import os
import sys
import time
import wave
import signal
import argparse
import subprocess
import sounddevice as sd
import numpy as np
from pathlib import Path

# 配置
AUDIO_OUTPUT_DEVICE_ID = 24  # 狗的扬声器（但我们要找麦克风）
SAMPLE_RATE = 16000         # 录音采样率 16kHz
CHANNELS = 1                # 单声道
DTYPE = np.int16
CHUNK_DURATION = 5          # 每段录音时长（秒）

# 文件路径
RECORD_DIR = os.path.expanduser("~/.openclaw/media/recordings")
SEND_VOICE_SCRIPT = "/home/unitree/openclaw/skills/go2-feishu-audio/scripts/send_voice.py"

# 全局控制变量
running = False


def find_input_device():
    """查找输入设备（麦克风）"""
    # 使用 default 设备（通常是正确的麦克风）
    try:
        default_device = sd.default.device[0]  # 输入设备
        if default_device is not None:
            device_info = sd.query_devices(default_device)
            print(f"🎤 使用默认输入设备: {default_device} - {device_info['name']}")
            return default_device
    except:
        pass
    
    # 备选：查找第一个有输入通道的设备
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"🎤 找到输入设备: {i} - {device['name']}")
            return i
    return None


def record_audio(duration, output_path, device_id):
    """
    录制音频
    
    Args:
        duration: 录音时长（秒）
        output_path: 输出文件路径
        device_id: 输入设备ID
    """
    print(f"🎤 录音 {duration} 秒...")
    
    # 计算样本数
    num_samples = int(duration * SAMPLE_RATE)
    
    # 录制音频
    recording = sd.rec(
        num_samples,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        device=device_id
    )
    
    # 等待录音完成
    sd.wait()
    
    # 保存为 WAV 文件
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(recording.tobytes())
    
    print(f"✅ 录音完成: {output_path}")
    return output_path


def send_to_feishu(audio_path, duration, chat_id=None):
    """
    发送音频到飞书
    
    Args:
        audio_path: 音频文件路径
        duration: 音频时长（秒）
        chat_id: 飞书聊天ID（可选）
    """
    try:
        cmd = [
            "python3", SEND_VOICE_SCRIPT,
            audio_path,
            "--duration", str(duration)
        ]
        
        # 如果有 chat_id，添加到命令
        if chat_id:
            cmd.extend(["--chat-id", chat_id])
        
        # 设置环境变量
        env = os.environ.copy()
        if chat_id:
            env["FEISHU_CHAT_ID"] = chat_id
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        
        if result.returncode == 0:
            print("✅ 已发送到飞书")
            return True
        else:
            print(f"❌ 发送失败: {result.stderr}")
            print(f"   输出: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"❌ 发送错误: {e}")
        return False


def recording_loop(device_id, chat_id=None):
    """
    录音循环 - 持续录音并发送
    
    Args:
        device_id: 输入设备ID
        chat_id: 飞书聊天ID（可选）
    """
    global running
    
    running = True
    os.makedirs(RECORD_DIR, exist_ok=True)
    
    print("=" * 50)
    print("🎙️ 录音循环已启动")
    print("=" * 50)
    print(f"   设备: {device_id}")
    print(f"   采样率: {SAMPLE_RATE}Hz")
    print(f"   每段时长: {CHUNK_DURATION}秒")
    print("   发送 '关停' 停止录音")
    print("=" * 50)
    print()
    
    segment = 0
    
    while running:
        segment += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RECORD_DIR, f"record_{timestamp}_{segment}.wav")
        
        try:
            # 录音
            record_audio(CHUNK_DURATION, output_path, device_id)
            
            # 发送到飞书
            send_to_feishu(output_path, CHUNK_DURATION, chat_id)
            
            # 清理临时文件（可选）
            # os.remove(output_path)
            
        except Exception as e:
            print(f"❌ 录音/发送失败: {e}")
            time.sleep(1)
    
    print()
    print("=" * 50)
    print("🛑 录音循环已停止")
    print("=" * 50)


def stop_recording():
    """停止录音"""
    global running
    running = False
    print("🛑 正在停止录音...")


def main():
    parser = argparse.ArgumentParser(
        description="录音并发送到飞书 - 机器狗麦克风 → 飞书语音消息"
    )
    parser.add_argument(
        "action",
        choices=["start", "stop", "status"],
        help="操作: start=启动录音, stop=停止录音, status=查看状态"
    )
    parser.add_argument(
        "--device", "-d",
        type=int,
        help="输入设备ID（默认自动查找）"
    )
    parser.add_argument(
        "--duration", "-t",
        type=int,
        default=5,
        help="每段录音时长（秒），默认5秒"
    )
    parser.add_argument(
        "--chat-id", "-c",
        help="飞书聊天ID"
    )
    
    args = parser.parse_args()
    
    if args.action == "status":
        global running
        status = "运行中" if running else "已停止"
        print(f"📊 录音状态: {status}")
        return
    
    if args.action == "stop":
        stop_recording()
        return
    
    if args.action == "start":
        # 查找输入设备
        device_id = args.device
        if device_id is None:
            device_id = find_input_device()
            if device_id is None:
                print("❌ 未找到输入设备（麦克风）")
                sys.exit(1)
        
        # 设置录音时长
        global CHUNK_DURATION
        CHUNK_DURATION = args.duration
        
        # 启动录音循环
        try:
            recording_loop(device_id, args.chat_id)
        except KeyboardInterrupt:
            print("\n🛑 用户中断")
            stop_recording()


if __name__ == "__main__":
    main()
