import sounddevice as sd
import asyncio
import numpy as np
import logging
import os

os.environ["TENCENT_TTS_SECRET_ID"] = "AKIDCsKYYIn46tRlriKhpibGgrmTzHK8h16M"
os.environ["TENCENT_TTS_SECRET_KEY"] = "4sSowpTt77tSq2zjMmzzmMjsnbCYfYDg"

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

# JABRA
AUDIO_INPUT_DEVICE_CHANNALS = 1
AUDIO_OUTPUT_DEVICE_CHANNALS = 2
AUDIO_INPUT_SAMPLE_RATE = 16000
AUDIO_OUTPUT_SAMPLE_RATE = 48000

# HIKVISON
AUDIO_INPUT_DEVICE_CHANNALS = 2
AUDIO_OUTPUT_DEVICE_CHANNALS = 1
AUDIO_INPUT_SAMPLE_RATE = 48000
AUDIO_OUTPUT_SAMPLE_RATE = 48000

# ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:2,0)
AUDIO_INPUT_DEVICE_CHANNALS = 6
AUDIO_OUTPUT_DEVICE_CHANNALS = 2
AUDIO_INPUT_SAMPLE_RATE = 16000
AUDIO_OUTPUT_SAMPLE_RATE = 16000

logger = logging.getLogger("aplay_unit_test")
logger.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(_console_handler)

_file_handler = logging.FileHandler("aplay_unit_test.log", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(_file_handler)
logger.info("日志文件已创建: aplay_unit_test.log")

from .tts_client import TencentTTS

async def _stream_play(tts_client, text):
    tts_sr = 16000
    out_sr = AUDIO_OUTPUT_SAMPLE_RATE
    ratio = out_sr / tts_sr
    stream = sd.OutputStream(samplerate=out_sr,
        channels=AUDIO_OUTPUT_DEVICE_CHANNALS, dtype="int16", blocksize=int(out_sr*0.02))
    stream.start(); loop = asyncio.get_event_loop(); total = 0
    try:
        async for chunk in tts_client.synthesize_stream(text):
            total += len(chunk)
            arr = np.frombuffer(chunk, dtype=np.int16)
            if ratio != 1.0:
                idx = np.arange(0, len(arr), 1.0/ratio)
                arr = arr[np.clip(idx.astype(int), 0, len(arr)-1)]
            pcm = arr.reshape(-1,1) if AUDIO_OUTPUT_DEVICE_CHANNALS==1 else np.column_stack([arr]*AUDIO_OUTPUT_DEVICE_CHANNALS)
            await loop.run_in_executor(None, stream.write, pcm)
    finally:
        await asyncio.sleep(0.1); stream.stop(); stream.close()
        logger.info(f"🔊 TTS 完成, {total} bytes")


async def worker_loop(tts_client, text):
    while True:
        try: await _stream_play(tts_client, text)
        except asyncio.CancelledError: raise
        except Exception as e: logger.error(f"播报异常: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", nargs="+", help="text")
    args = parser.parse_args()

    tts_client = TencentTTS()

    asyncio.run(worker_loop(tts_client, args.text[0]))