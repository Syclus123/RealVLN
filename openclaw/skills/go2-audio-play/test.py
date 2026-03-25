import sounddevice as sd
import numpy as np
from scipy import signal
from scipy.io import wavfile  # 新增：导入 wav 文件读写模块

# ==========================================
# 步骤 1：设置你的 Jabra 设备 ID
# ==========================================
INPUT_DEVICE = 25   
OUTPUT_DEVICE = 0  

sd.default.device = (INPUT_DEVICE, OUTPUT_DEVICE)

# ==========================================
# 步骤 2：定义物理参数
# ==========================================
record_fs = 16000     
play_fs = 48000       
duration = 3          
channels = 1          

if __name__ == "__main__":
    print(f"\n[录音中] 请对着麦克风说话，持续 {duration} 秒...")
    
    # 获取原始 int16 格式的录音矩阵
    myrecording = sd.rec(int(duration * record_fs), 
                         samplerate=record_fs, 
                         channels=channels, 
                         dtype='int16')
    sd.wait()
    print("[录音完毕]")

    # ==========================================
    # 步骤 3：保存原始音频为 WAV 文件
    # ==========================================
    save_path_16k = "original_record_16k.wav"
    # wavfile.write 的参数顺序是：(文件路径, 采样率, 音频数据矩阵)
    wavfile.write(save_path_16k, record_fs, myrecording)
    print(f"[文件保存] 原始 16kHz 录音已保存至当前目录：{save_path_16k}")

    # ==========================================
    # 步骤 4：在内存中重采样 (16k -> 48k)
    # ==========================================
    print("\n正在进行重采样处理 (16kHz -> 48kHz)...")
    target_length = int(len(myrecording) * (play_fs / record_fs))
    resampled_audio = signal.resample(myrecording, target_length)
    resampled_audio = np.int16(resampled_audio)

    # 💡 可选：如果你也想把给喇叭听的 48k 版本保存下来对比，取消下面两行的注释
    # save_path_48k = "resampled_play_48k.wav"
    # wavfile.write(save_path_48k, play_fs, resampled_audio)

    # ==========================================
    # 步骤 5：播放处理后的音频
    # ==========================================
    print(f"[播放中] 正在以 {play_fs}Hz 回放...")
    sd.play(resampled_audio, samplerate=play_fs)
    sd.wait()
    print("[播放完毕]")
