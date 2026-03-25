#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pyaudio
import numpy as np
import sherpa_onnx
import wave

MODEL_DIR = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
MIC_RATE = 16000
MODEL_RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 6
DEVICE_INDEX = 24  # 填入你的目标麦克风序号
CHUNK = int(MIC_RATE * 0.1)

def create_keyword_spotter():
    print("🚀 正在加载 Sherpa-onnx 中文唤醒词模型...")
    kws = sherpa_onnx.KeywordSpotter(
        tokens=f"{MODEL_DIR}/tokens.txt",
        encoder=f"{MODEL_DIR}/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        decoder=f"{MODEL_DIR}/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        joiner=f"{MODEL_DIR}/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
        keywords_file="my_keywords.txt",
        num_threads=2,
        provider="cpu",
    )
    return kws

def main():
    kws = create_keyword_spotter()
    audio = pyaudio.PyAudio()
    
    print("\n🎤 正在以 48kHz 打开麦克风...")
    try:
        mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=MIC_RATE, input=True, frames_per_buffer=CHUNK, input_device_index=DEVICE_INDEX)
    except Exception as e:
        print(f"\n❌ 麦克风打开失败: {e}")
        return

    stream = kws.create_stream()
    print("\n🟢 监听中... (按 Ctrl+C 退出)")
    print("-" * 50)

    debug_frames = [] # 👈 增加：用于收集测试音频的列表

    try:
        while True:
            pcm_data = mic_stream.read(CHUNK, exception_on_overflow=False)
            samples_int16 = np.frombuffer(pcm_data, dtype=np.int16)
            
            # 【这是你现在的通道剥离代码】
            samples_int16 = samples_int16[0::CHANNELS]
            
            # 👈 增加：把剥离后准备喂给 AI 的数据存起来
            debug_frames.append(samples_int16.tobytes())
            
            # 👈 增加：存满 5 秒 (50个 chunk) 就自动保存并退出
            #if len(debug_frames) >= 50:
            #    print("\n\n💾 正在保存 AI 听到的原始音频用于分析...")
            #    with wave.open("ai_debug_listen.wav", "wb") as wf:
            #        wf.setnchannels(1)
            #        wf.setsampwidth(2)
            #        wf.setframerate(16000)
            #        wf.writeframes(b''.join(debug_frames))
            #    print("✅ 已保存至 ai_debug_listen.wav！请用电脑播放听一下。")
                #os._exit(0)


            
            # ================= 下面不用改，保持原样 =================
            # 计算音量的均值（底噪）和峰值（人声）
            vol_mean = np.abs(samples_int16).mean()
            vol_max = np.abs(samples_int16).max()
            
            # 动态打印数据：帮你科学排查麦克风状态！
            vol_bar = "█" * int(vol_mean / 50) 
            print(f"\r\033[K🎤 底噪:{vol_mean:.0f} | 峰值:{vol_max:.0f} | {vol_bar:<20}", end="", flush=True)

            # 转换为浮点数并去除直流偏置
            samples_float32 = samples_int16.astype(np.float32) / 32768.0
            #samples_float32 = samples_float32 - np.mean(samples_float32)

            # 发送给 AI 引擎
            stream.accept_waveform(MODEL_RATE, samples_float32)
            while kws.is_ready(stream):
                kws.decode_stream(stream)
            
            result = kws.get_result(stream)
            keyword_str = getattr(result, 'keyword', str(result)).strip()
            
            if keyword_str:
                print(f"\n\n🔥 [唤醒触发!] 识别到指令: '{keyword_str}'")
                print("-" * 50)
                kws.reset_stream(stream)
                
    except KeyboardInterrupt:
        print("\n\n🛑 接收到退出信号，正在安全终结进程...")
        # 【核心修复】直接硬核杀死进程，绕过 Jetson 恶心的 ALSA 内存释放 Bug
        os._exit(0)

if __name__ == "__main__":
    import ctypes
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    def py_error_handler(filename, line, function, err, fmt): pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    try:
        asound = ctypes.cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
    except:
        pass

    main()
