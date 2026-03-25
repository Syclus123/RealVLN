import sounddevice as sd
from typing import List, Optional, Tuple

RESPEAKER_KEYWORDS = ["ReSpeaker", "respeaker"]


def _find_device(keywords: List[str], kind: str) -> Tuple[int, str]:
    """Search sounddevice list for the first device whose name contains any keyword."""
    devices = sd.query_devices()
    for kw in keywords:
        for i, dev in enumerate(devices):
            if kind == "input" and dev["max_input_channels"] == 0:
                continue
            if kind == "output" and dev["max_output_channels"] == 0:
                continue
            if kw in dev["name"]:
                return i, dev["name"]
    available = [
        f"  {i}: {d['name']} (in={d['max_input_channels']}, out={d['max_output_channels']})"
        for i, d in enumerate(devices)
    ]
    raise RuntimeError(
        f"未找到包含 {keywords} 的 {kind} 设备。\n可用设备:\n" + "\n".join(available)
    )


AUDIO_INPUT_DEVICE_ID, _in_name = _find_device(RESPEAKER_KEYWORDS, "input")
AUDIO_OUTPUT_DEVICE_ID, _out_name = _find_device(RESPEAKER_KEYWORDS, "output")

AUDIO_INPUT_SAMPLE_RATE = 16000
AUDIO_OUTPUT_SAMPLE_RATE = 16000
AUDIO_INPUT_DEVICE_CHANNALS = 6
AUDIO_OUTPUT_DEVICE_CHANNALS = 2