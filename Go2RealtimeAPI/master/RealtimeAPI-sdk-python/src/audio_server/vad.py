from collections import deque
from dataclasses import dataclass, field
import numpy as np
import torch
import os

# ══════════════════════════════════════════════
#  VAD 语音活动检测
# ══════════════════════════════════════════════

@dataclass
class RMSVADConfig:
    sample_rate: int = 16000
    frame_duration_ms: int = 20
    energy_threshold: float = 500.0
    speech_min_frames: int = 10   # 确认语音最少帧数 (200ms)
    silence_max_frames: int = 30  # 静音超时帧数 (600ms)
    pre_speech_frames: int = 5    # 预缓冲帧数 (防切头)
    # 长时间静音后底噪漂移会导致“静音”帧仍超阈值，speech_stopped 永不触发
    max_speech_duration_frames: int = 3000  # 最大语音时长兜底 (60s)，超时强制 speech_stopped
    adaptive_silence_window: int = 50      # 自适应静音：最近 N 帧的 min(rms)，低于其 1.5 倍视为静音

  
class RMSVoiceActivityDetector:
    """基于 RMS 能量的 VAD. 状态机: IDLE → MAYBE_SPEECH → SPEECH → IDLE."""

    def __init__(self, config: RMSVADConfig | None = None):
        self.cfg = config or RMSVADConfig()
        self.frame_bytes = self.cfg.sample_rate * 2 * self.cfg.frame_duration_ms // 1000
        self.reset()

    def reset(self) -> None:
        self._state = "IDLE"
        self._speech_count = 0
        self._silence_count = 0
        self._ring: list[bytes] = []
        self._speech_audio = bytearray()
        self._speech_frame_count = 0
        self._recent_rms: deque = deque(maxlen=getattr(
            self.cfg, "adaptive_silence_window", 50
        ))

    def _rms(self, pcm: bytes) -> float:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0

    def process_frame(self, frame: bytes) -> list[dict]:
        """送入一帧, 返回事件列表."""
        events: list[dict] = []
        is_speech = self._rms(frame) >= self.cfg.energy_threshold

        if self._state == "IDLE":
            self._ring.append(frame)
            if len(self._ring) > self.cfg.pre_speech_frames:
                self._ring.pop(0)
            if is_speech:
                self._state = "MAYBE_SPEECH"
                self._speech_count = 1
                self._silence_count = 0

        elif self._state == "MAYBE_SPEECH":
            # 继续缓存, 确认后一并写入 speech_audio
            self._ring.append(frame)
            if is_speech:
                self._speech_count += 1
                if self._speech_count >= self.cfg.speech_min_frames:
                    self._state = "SPEECH"
                    self._speech_audio = bytearray(b"".join(self._ring))
                    self._ring.clear()
                    events.append({"event": "speech_started"})
            else:
                # 回退 IDLE, 恢复环形缓冲上限
                self._state = "IDLE"
                self._speech_count = 0
                while len(self._ring) > self.cfg.pre_speech_frames:
                    self._ring.pop(0)

        elif self._state == "SPEECH":
            self._speech_audio.extend(frame)
            rms = self._rms(frame)
            self._speech_frame_count += 1

            # 兜底：超长语音强制结束，避免长时间静音后底噪漂移导致永不 speech_stopped
            max_frames = getattr(
                self.cfg, "max_speech_duration_frames", 3000
            )
            if self._speech_frame_count >= max_frames:
                audio = bytes(self._speech_audio)
                self.reset()
                events.append({"event": "speech_stopped", "audio": audio})
                return events

            # 自适应静音：长时间静音后底噪可能高于固定阈值，用近期最小 RMS 做相对判断
            self._recent_rms.append(rms)
            min_history = min(10, self._recent_rms.maxlen)
            if len(self._recent_rms) >= min_history:
                recent_min = min(self._recent_rms)
                is_silence = (
                    rms < self.cfg.energy_threshold
                    or (recent_min > 0 and rms < max(100.0, recent_min * 1.5))
                )
            else:
                is_silence = not is_speech

            if not is_silence:
                self._silence_count = 0
            else:
                self._silence_count += 1
                if self._silence_count >= self.cfg.silence_max_frames:
                    audio = bytes(self._speech_audio)
                    self.reset()
                    events.append({"event": "speech_stopped", "audio": audio})

        return events


# ══════════════════════════════════════════════
#  Silero VAD 语音活动检测 (适配版)
# ══════════════════════════════════════════════

@dataclass
class SileroVADConfig:
    sample_rate: int = 16000
    # Silero 在 16k 下最佳窗口是 512 采样点 (32ms)
    # 我们不再强制外部传入 20ms，内部会自动缓冲
    threshold: float = 0.5        # 判定为人声的概率阈值
    
    # 状态机计数器 (注意：现在每"帧"代表一次 32ms 的推理)
    # 32ms * 5 = 160ms 确认
    min_speech_frames: int = 5    
    # 32ms * 20 = 640ms 静音超时
    max_silence_frames: int = 20  
    # 预缓冲 (防切头), 32ms * 10 = 320ms
    pre_speech_frames: int = 10   
    # 长时间静音后可能无法正确检测结束，兜底：最大语音时长 (帧数，32ms/帧)，约 60s
    max_speech_duration_frames: int = 1875

    # 模型路径 (可选，None则自动下载)
    model_path: str | None = None 

class SileroVoiceActivityDetector:
    """
    基于 Silero VAD 的检测器. 
    保留了原有的状态机逻辑: IDLE → MAYBE_SPEECH → SPEECH → IDLE.
    """

    def __init__(self, config: SileroVADConfig | None = None):
        self.cfg = config or SileroVADConfig()
        
        # Silero 硬性要求: 16k采样率下窗口必须是 512
        self.window_size_samples = 512 if self.cfg.sample_rate == 16000 else 256
        
        # 加载模型
        self._load_model()
        
        # 内部缓冲，用于处理输入帧长与 Silero 要求不一致的问题
        self._raw_buffer = bytearray()
        
        self.reset()

    def _load_model(self):
        print("正在加载 Silero VAD 模型...")
        try:
            # 优先尝试加载本地 JIT 模型 (在 Go2 上推荐)
            if self.cfg.model_path and os.path.exists(self.cfg.model_path):
                self.model = torch.jit.load(self.cfg.model_path)
            else:

                # 否则从 Hub 加载
                local_model_dir = '/home/unitree/snakers4-silero-vad-fcf78bc'
                if not os.path.isdir(local_model_dir):
                    self.model, _ = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=False,
                        trust_repo=True,
                        onnx=False # 使用 PyTorch 推理
                    )
                else:
                    self.model, _ = torch.hub.load(
                        repo_or_dir=local_model_dir,
                        source='local',
                        model='silero_vad',
                        force_reload=False,
                        trust_repo=True,
                        onnx=False # 使用 PyTorch 推理
                    )

            self.model.eval()
            print("Silero VAD 模型加载成功")
        except Exception as e:
            print(f"Silero VAD 加载失败: {e}")
            raise e

    def reset(self) -> None:
        self._state = "IDLE"
        self._speech_count = 0
        self._silence_count = 0
        
        # 环形缓冲区 (存储的是已经切分好的 512 长度的 bytes)
        self._ring: list[bytes] = []
        
        # 最终输出的音频
        self._speech_audio = bytearray()
        self._speech_frame_count = 0

        # 清空内部对齐缓冲
        self._raw_buffer = bytearray()
        
        # 重置模型内部状态 (LSTM hidden states)
        if hasattr(self, 'model'):
            self.model.reset_states()

    def _get_speech_prob(self, pcm_chunk: bytes) -> float:
        """运行模型获取人声概率"""
        # 1. 转换 bytes -> float32 tensor
        audio_int16 = np.frombuffer(pcm_chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_float32)
        
        # 2. 增加维度 [Time] -> [1, Time]
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
            
        # 3. 推理
        with torch.no_grad():
            # item() 取出标量
            prob = self.model(tensor, self.cfg.sample_rate).item()
            
        return prob

    def process_frame(self, frame: bytes) -> list[dict]:
        """
        送入任意长度的 frame, 返回事件列表.
        注意：因为内部有缓冲，输入一帧不一定马上有输出，或者可能输出多个事件。
        """
        events: list[dict] = []
        
        # 1. 将新数据加入原始缓冲
        self._raw_buffer.extend(frame)
        
        # 2. 只要缓冲够 512 个点 (1024 bytes @ 16bit)，就切出来跑一次模型
        chunk_bytes_len = self.window_size_samples * 2  # 16bit = 2 bytes
        
        while len(self._raw_buffer) >= chunk_bytes_len:
            # 取出一个 chunk
            current_chunk = bytes(self._raw_buffer[:chunk_bytes_len])
            # 从缓冲中移除
            del self._raw_buffer[:chunk_bytes_len]
            
            # --- 核心逻辑 ---
            prob = self._get_speech_prob(current_chunk)
            is_speech = prob >= self.cfg.threshold
            
            # 状态机逻辑 (跟你原来的代码保持一致，只是对象换成了 current_chunk)
            self._update_state_machine(current_chunk, is_speech, events)
            
        return events

    def _update_state_machine(self, chunk: bytes, is_speech: bool, events: list[dict]):
        """原来 process_frame 里的状态机逻辑移到这里"""
        
        if self._state == "IDLE":
            self._ring.append(chunk)
            if len(self._ring) > self.cfg.pre_speech_frames:
                self._ring.pop(0)
                
            if is_speech:
                self._state = "MAYBE_SPEECH"
                self._speech_count = 1
                self._silence_count = 0

        elif self._state == "MAYBE_SPEECH":
            self._ring.append(chunk)
            if is_speech:
                self._speech_count += 1
                if self._speech_count >= self.cfg.min_speech_frames:
                    self._state = "SPEECH"
                    # 拼接环形缓冲区里的“切头”数据
                    self._speech_audio = bytearray(b"".join(self._ring))
                    self._ring.clear()
                    events.append({"event": "speech_started"})
            else:
                # 误触发，回退
                self._state = "IDLE"
                self._speech_count = 0
                # 恢复环形缓冲上限，防止无限增长
                while len(self._ring) > self.cfg.pre_speech_frames:
                    self._ring.pop(0)

        elif self._state == "SPEECH":
            self._speech_audio.extend(chunk)
            self._speech_frame_count += 1
            max_frames = getattr(
                self.cfg, "max_speech_duration_frames", 1875
            )
            if self._speech_frame_count >= max_frames:
                audio = bytes(self._speech_audio)
                events.append({"event": "speech_stopped", "audio": audio})
                self.reset()
                return
            if is_speech:
                self._silence_count = 0
            else:
                self._silence_count += 1
                if self._silence_count >= self.cfg.max_silence_frames:
                    audio = bytes(self._speech_audio)
                    events.append({"event": "speech_stopped", "audio": audio})
                    self.reset()