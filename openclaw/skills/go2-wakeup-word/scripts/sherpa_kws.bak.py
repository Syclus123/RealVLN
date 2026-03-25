#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyaudio
import numpy as np
import sherpa_onnx

# ==========================================
# 模型与麦克风配置
# ==========================================
MODEL_DIR = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000          # Sherpa-onnx 要求 16kHz
CHUNK = 1600          # 每次读取 100ms 音频数据

def create_keyword_spotter():
    print("🚀 正在加载 Sherpa-onnx 中文唤醒词模型...")
    # 初始化 KeywordSpotter (纯 CPU 运行，完全不抢占 GPU)
    kws = sherpa_onnx.KeywordSpotter(
        tokens=f"{MODEL_DIR}/tokens.txt",
        encoder=f"{MODEL_DIR}/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        decoder=f"{MODEL_DIR}/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        joiner=f"{MODEL_DIR}/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
        keywords_file="my_keywords.txt",  # 指向你刚才创建的词表
        num_threads=2,                    # 分配两个 CPU 核心专门跑语音
        provider="cpu",
    )
    return kws

def main():
    kws = create_keyword_spotter()
    
    # 初始化音频流
    audio = pyaudio.PyAudio()
    print("\n🎤 正在打开麦克风...")
    try:
        mic_stream = audio.open(format=FORMAT, 
                                channels=CHANNELS, 
                                rate=RATE, 
                                input=True, 
                                frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"\n❌ 麦克风打开失败，请检查声卡输入: {e}")
        return

    # 创建一个独立的识别数据流
    stream = kws.create_stream()

    print("\n🟢 监听中... 请说出你在 my_keywords.txt 里配置的词 (例如 '狗子跟上')")
    print("-" * 50)

    try:
        while True:

            # 1. 读取音频数据
            pcm_data = mic_stream.read(CHUNK, exception_on_overflow=False)
            
            # 2. 转换为 float32 并消除直流偏置
            samples_int16 = np.frombuffer(pcm_data, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32) / 32768.0
            samples_float32 = samples_float32 - np.mean(samples_float32)

            # 👇 --- 新增：暴力噪声门 (Noise Gate) --- 👇
            # 如果音量绝对值小于 0.05（可根据实际嗡嗡声大小微调，如 0.03~0.1），直接视为静音
            samples_float32[np.abs(samples_float32) < 0.05] = 0.0

            # (可选) 打印音量柱状图
            volume = np.abs(samples_int16).mean()
            vol_bar = "█" * int(volume / 50) 
            print(f"\r\033[K🎤 实时音量: [{vol_bar:<30}]", end="", flush=True)

            # 3. 将净化后的音频塞进模型流
            stream.accept_waveform(RATE, samples_float32)


        
            # 4. 只要模型准备好了，就开始解码
            while kws.is_ready(stream):
                kws.decode_stream(stream)
            
            # 5. 获取结果，如果命中会返回我们配置的 @ 后面的文字
            result = kws.get_result(stream)
            print('result:', result)
            
            # 提取真实的唤醒词字符串
            keyword_str = getattr(result, 'keyword', str(result)).strip()
            
            if keyword_str:
                print(f"\n\n🔥 [唤醒触发!] 识别到指令: '{keyword_str}'")

            if result:
                print(f"\n\n🔥 [唤醒触发!] 识别到指令: '{result}'")
                
                # 在这里可以加入控制狗坐下、站起、或激活 YOLO 的全局变量
                if "狗子跟上" in result:
                    print("🤖 机器狗: '收到，进入跟随模式！'")
                elif "你好问问" in result:
                    print("🤖 机器狗: '你好，主人！'")
                
                print("-" * 50)
                # 识别到之后，立刻重置流的状态，防止同一个词被疯狂连环触发
                kws.reset_stream(stream)
                
    except KeyboardInterrupt:
        print("\n🛑 停止监听。")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()

if __name__ == "__main__":
    # 屏蔽底层 ALSA C语言库那些烦人的警告
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
