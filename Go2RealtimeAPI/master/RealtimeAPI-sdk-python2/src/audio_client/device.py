import os

# 强制 PortAudio 使用 ALSA plughw 插件，规避 PaAlsaStreamComponent_BeginPolling
# assertion failure (USB 音频设备文件描述符变化导致 abort)
os.environ.setdefault("PA_ALSA_PLUGHW", "1")

import sounddevice as sd  # noqa: E402
from typing import List

# ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:2,0)

AUDIO_INPUT_DEVICE_ID = None
AUDIO_OUTPUT_DEVICE_ID = None


AUDIO_INPUT_SAMPLE_RATE = 16000
AUDIO_OUTPUT_SAMPLE_RATE = 16000
AUDIO_INPUT_DEVICE_CHANNALS = 6
AUDIO_OUTPUT_DEVICE_CHANNALS = 2