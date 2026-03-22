import sounddevice as sd
from typing import List

# ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:0,0)
# Uses "respeaker_direct" ALSA device (plug only, no dmix/dsnoop)
# to avoid PortAudio PaAlsaStreamComponent_BeginPolling assertion failure
# caused by dmix/dsnoop changing poll fd counts at runtime.

AUDIO_INPUT_DEVICE_ID = "respeaker_direct"
AUDIO_OUTPUT_DEVICE_ID = "respeaker_direct"


AUDIO_INPUT_SAMPLE_RATE = 16000
AUDIO_OUTPUT_SAMPLE_RATE = 16000
AUDIO_INPUT_DEVICE_CHANNALS = 6
AUDIO_OUTPUT_DEVICE_CHANNALS = 2