#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 Audio Play - 播放声音到机器狗扬声器
完全按照 RealtimeAPI SDK 实现
"""
import sys
import os
import argparse
import wave
import sounddevice as sd
import numpy as np


# AUDIO DEVICE
device_names = [
    "8-mic Speakerphone: USB Audio (hw:0,0)",
    "Jabra Link 370: USB Audio (hw:0,0)", # 
    "Jabra SPEAK 510 USB: Audio (hw:0,0)", #
    "default"
    ]

def get_device_by_exact_name(exact_name, kind='any'):
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if kind == 'input' and dev['max_input_channels'] == 0:
            continue
        if kind == 'output' and dev['max_output_channels'] == 0:
            continue
        if dev['name'] in exact_name:
            return i, dev['name']
    raise ValueError(f"未找到名称完全匹配 '{exact_name}' 的设备")

AUDIO_INPUT_DEVICE_ID, AUDIO_DEVICE_NAME = get_device_by_exact_name(device_names)
AUDIO_OUTPUT_DEVICE_ID = AUDIO_INPUT_DEVICE_ID

#AUDIO_OUTPUT_DEVICE_ID=get_device_by_exact_name('8-mic Speakerphone: USB Audio (hw:0,0)')
#AUDIO_OUTPUT_DEVICE_ID = "plughw:0,0"
#AUDIO_OUTPUT_DEVICE_ID = 0
#AUDIO_OUTPUT_DEVICE_ID = 'Jabra'
print('AUDIO_OUTPUT_DEVICE_ID:', AUDIO_OUTPUT_DEVICE_ID)



# 机器狗音频设备配置（来自 SDK）
#AUDIO_OUTPUT_DEVICE_ID = 24   # 输出设备ID
HW_SAMPLE_RATE = 48000        # 硬件采样率 48kHz
WS_SAMPLE_RATE = 16000        # 服务器/网络音频采样率 16kHz
RESAMPLE_RATIO = HW_SAMPLE_RATE // WS_SAMPLE_RATE  # 3
FRAME_DURATION_MS = 20        # 每帧时长 20ms

# 声音文件目录
SOUNDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sounds")


def list_available_sounds():
    """列出可用的声音文件"""
    print("🎵 可用声音文件：")
    print()
    
    if not os.path.exists(SOUNDS_DIR):
        print(f"  ❌ 声音目录不存在: {SOUNDS_DIR}")
        return
    
    sounds = [f for f in os.listdir(SOUNDS_DIR) if f.endswith(('.wav', '.mp3', '.ogg'))]
    
    if not sounds:
        print(f"  ❌ 没有找到声音文件")
        return
    
    for i, sound in enumerate(sorted(sounds), 1):
        file_path = os.path.join(SOUNDS_DIR, sound)
        size = os.path.getsize(file_path)
        print(f"  {i}. {sound:30} ({size} bytes)")
    
    print()
    print(f"使用: python3 play.py <声音文件名>")


def play_audio(pcm_data, input_sample_rate):
    """
    播放音频数据（完全按照 SDK 方式）
    
    Args:
        pcm_data: numpy int16 数组
        input_sample_rate: 输入音频采样率
    """
    # 重采样到 48kHz（SDK 方式：每个样本重复 RESAMPLE_RATIO 次）
    if input_sample_rate == WS_SAMPLE_RATE:
        # 16k -> 48k
        pcm_48k = np.repeat(pcm_data, RESAMPLE_RATIO)
    elif input_sample_rate == HW_SAMPLE_RATE:
        # 已经是 48k
        pcm_48k = pcm_data
    else:
        # 其他采样率，线性插值
        ratio = HW_SAMPLE_RATE / input_sample_rate
        new_length = int(len(pcm_data) * ratio)
        pcm_48k = np.interp(
            np.linspace(0, len(pcm_data), new_length),
            np.arange(len(pcm_data)),
            pcm_data
        ).astype(np.int16)
    
    # 计算 blocksize (SDK 方式: 48000 * 0.02 = 960)
    blocksize = int(HW_SAMPLE_RATE * FRAME_DURATION_MS / 1000)
    
    print(f"🔊 启动音频流...")
    print(f"   设备: {AUDIO_OUTPUT_DEVICE_ID}")
    print(f"   采样率: {HW_SAMPLE_RATE}Hz")
    print(f"   声道: 1")
    print(f"   blocksize: {blocksize}")
    print(f"   数据长度: {len(pcm_48k)} 样本")
    
    # 创建输出流（完全按照 SDK）
    stream = sd.OutputStream(
        device=AUDIO_OUTPUT_DEVICE_ID,
        samplerate=HW_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=blocksize,
    )
    
    stream.start()
    print(f"🔊 音频播放已启动")
    
    # 写入音频数据
    #stream.write(pcm_48k)
  
    # 替换为这两行：
    pcm_48k_2d = pcm_48k.reshape(-1, 1) # 强转为 (N, 1) 的二维数组，明确告诉它这是单声道
    stream.write(pcm_48k_2d)
 
    # 等待播放完成
    sd.wait()
    
    stream.stop()
    stream.close()
    print(f"✅ 播放完成")


def play_wav(file_path):
    """
    播放 WAV 文件
    
    Args:
        file_path: WAV 文件路径
    
    Returns:
        成功返回 True，失败返回 False
    """
    try:
        # 打开 WAV 文件
        with wave.open(file_path, 'rb') as wf:
            # 获取音频参数
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            
            print(f"🎵 正在播放: {os.path.basename(file_path)}")
            print(f"   原始格式: {channels}ch, {sample_rate}Hz, {sample_width*8}bit")
            print(f"   时长: {n_frames / sample_rate:.2f}秒")
            
            # 读取音频数据
            audio_data = wf.readframes(n_frames)
            
            # 转换为 numpy 数组 (16-bit PCM)
            if sample_width == 2:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                print(f"⚠️ 转换采样宽度: {sample_width} -> 2")
                audio_array = np.frombuffer(audio_data, dtype=np.int8).astype(np.int16) * 256
            
            # 如果是立体声，转换为单声道
            if channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            # 播放音频
            play_audio(audio_array, sample_rate)
            
            return True
            
    except FileNotFoundError:
        print(f"❌ 文件不存在: {file_path}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ 播放错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def play_sound(sound_name):
    """
    播放声音文件
    
    Args:
        sound_name: 声音文件名或路径
    
    Returns:
        成功返回 True，失败返回 False
    """
    # 如果是完整路径，直接使用
    if os.path.isabs(sound_name) and os.path.exists(sound_name):
        file_path = sound_name
    else:
        # 从 sounds 目录查找
        file_path = os.path.join(SOUNDS_DIR, sound_name)
        
        # 如果不存在，尝试添加 .wav 后缀
        if not os.path.exists(file_path) and not sound_name.endswith(('.wav', '.mp3', '.ogg')):
            file_path = os.path.join(SOUNDS_DIR, sound_name + '.wav')
    
    if not os.path.exists(file_path):
        print(f"❌ 声音文件不存在: {sound_name}", file=sys.stderr)
        print(f"   搜索路径: {file_path}", file=sys.stderr)
        return False
    
    # 播放 WAV 文件
    if file_path.endswith('.wav'):
        return play_wav(file_path)
    else:
        print(f"⚠️ 当前仅支持 WAV 格式")
        return False


def main():
    parser = argparse.ArgumentParser(description="Go2 音频播放 - 播放声音到机器狗扬声器")
    parser.add_argument("sound", nargs="?", help="声音文件名或路径")
    parser.add_argument("--list", "-l", action="store_true", help="列出可用声音")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_sounds()
        sys.exit(0)
    
    if not args.sound:
        print("❌ 请提供声音文件名，或使用 --list 查看可用声音", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # 播放声音
    success = play_sound(args.sound)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
