import os
import base64
import struct
import numpy as np
from io import BytesIO

import dashscope
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

def pcm_to_wav_bytes(pcm_data, sample_rate=16000, channels=1, bits_per_sample=16):
    """
    将 PCM 数据转换为 WAV 字节流
    
    参数:
        pcm_data: bytes 或 numpy array
        sample_rate: 采样率
        channels: 声道数
        bits_per_sample: 位深度 (16, 24, 32)
    """
    if isinstance(pcm_data, np.ndarray):
        # 转换为 bytes
        if pcm_data.dtype == np.int16:
            pcm_bytes = pcm_data.tobytes()
        elif pcm_data.dtype == np.float32:
            # 归一化到 [-1, 1] 的浮点数转整数
            pcm_data = (pcm_data * 32767).astype(np.int16)
            pcm_bytes = pcm_data.tobytes()
        else:
            raise ValueError(f"不支持的 dtype: {pcm_data.dtype}")
    else:
        pcm_bytes = pcm_data
    
    # 计算数据大小
    data_size = len(pcm_bytes)
    
    # WAV 文件头
    # 1. RIFF 头
    riff_chunk = b'RIFF'
    file_size = 36 + data_size
    wave_format = b'WAVE'
    
    # 2. fmt 子块
    fmt_chunk = b'fmt '
    fmt_size = 16
    audio_format = 1  # PCM = 1
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    
    # 3. data 子块
    data_chunk = b'data'
    
    # 构建 WAV 文件
    wav_bytes = BytesIO()
    
    # RIFF 头
    wav_bytes.write(riff_chunk)
    wav_bytes.write(struct.pack('<I', file_size))
    wav_bytes.write(wave_format)
    
    # fmt 子块
    wav_bytes.write(fmt_chunk)
    wav_bytes.write(struct.pack('<I', fmt_size))
    wav_bytes.write(struct.pack('<H', audio_format))
    wav_bytes.write(struct.pack('<H', channels))
    wav_bytes.write(struct.pack('<I', sample_rate))
    wav_bytes.write(struct.pack('<I', byte_rate))
    wav_bytes.write(struct.pack('<H', block_align))
    wav_bytes.write(struct.pack('<H', bits_per_sample))
    
    # data 子块
    wav_bytes.write(data_chunk)
    wav_bytes.write(struct.pack('<I', data_size))
    wav_bytes.write(pcm_bytes)
    
    return wav_bytes.getvalue()


class Qwen3ASR:
    def __init__(self, logger, api_key=os.getenv("DASHSCOPE_API_KEY"), model_name="qwen3-asr-flash"):
        self.logger = logger
        self.api_key = api_key
        self.model_name = model_name
        self.audio_mime_type = "audio/wav"
        self.retry_num = 2
        
    async def recognize(self, audio_bytes, sample_rate=16000, channels=1, bits_per_sample=16):
        """调用 ASR 语音识别. 输入完整语音段 PCM(int16 mono), 返回文本."""
        self.logger.info(f"[ASR] {len(audio_bytes)} bytes ({len(audio_bytes) / 2 / sample_rate:.2f}s)")

        wav_bytes = pcm_to_wav_bytes(audio_bytes, sample_rate=sample_rate, channels=channels, bits_per_sample=bits_per_sample)
        base64_str = base64.b64encode(wav_bytes).decode()
        data_uri = f"data:{self.audio_mime_type};base64,{base64_str}"
        messages = [
            {
                "role": "system", 
                "content": [
                    {"text": ""}
                    ]
            },  # 配置定制化识别的 Context

            {
                "role": "user",
                "content": [
                    {"audio": data_uri},
                ]
            }
        ]

        retry_num = self.retry_num

        text = ""
        
        while retry_num > 0:
            try:
                response = dashscope.MultiModalConversation.call(
                    api_key=self.api_key,
                    model=self.model_name,
                    messages=messages,
                    result_format="message"
                    )
                
                if response.status_code == 200:
                    text = response["output"]["choices"][0]["message"].content[0]["text"]
                    break
                else:
                    retry_num -= 1
                    self.logger.warning(
                        f"LLM 调用失败: status={response.status_code}, "
                        f"msg={getattr(response, 'message', '')}"
                    )
                    
            except Exception as e:
                retry_num -= 1
                self.logger.error(f"ASR 调用异常: {e}")
        
        return text