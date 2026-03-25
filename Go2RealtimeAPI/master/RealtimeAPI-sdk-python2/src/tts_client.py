"""腾讯 TTS 流式客户端 — 异步版."""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import ssl
import time
import urllib.parse
import uuid
from typing import AsyncIterator
import os

import websockets

logger = logging.getLogger("tts")

# ================= 配置区 =================
# 在腾讯云控制台获取：https://console.cloud.tencent.com/cam/capi
APPID = "1325896784"
SECRET_ID = os.getenv("TENCENT_TTS_SECRET_ID")
SECRET_KEY = os.getenv("TENCENT_TTS_SECRET_KEY")

# 合成配置
VOICE_TYPE = 101003  # 音色ID


class TencentTTS:
    """腾讯云 TTS — 异步流式版本."""

    def __init__(self, sample_rate: int = 16000):
        self._ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE
        self.sample_rate = sample_rate

    def _generate_url(self, text: str, sample_rate: int = 16000) -> str:
        """生成带签名的 wss 请求地址."""
        params = {
            "Action": "TextToStreamAudioWS",
            "AppId": int(APPID),
            "SecretId": SECRET_ID,
            "Timestamp": int(time.time()),
            "Expired": int(time.time()) + 86400,
            "SessionId": str(uuid.uuid1()),
            "Text": text,
            "VoiceType": VOICE_TYPE,
            "Codec": "pcm",
            "SampleRate": sample_rate,
            "EnableSubtitle": "True",
            "Volume": 0,
            "Speed": 0,
        }

        sorted_params = sorted(params.items(), key=lambda d: d[0])
        query_str = "&".join([f"{k}={v}" for k, v in sorted_params])
        sign_source = f"GETtts.cloud.tencent.com/stream_ws?{query_str}"

        hmac_code = hmac.new(
            SECRET_KEY.encode("utf-8"),
            sign_source.encode("utf-8"),
            hashlib.sha1,
        ).digest()
        signature = base64.b64encode(hmac_code).decode("utf-8")

        params["Signature"] = signature
        final_query = urllib.parse.urlencode(params)
        return f"wss://tts.cloud.tencent.com/stream_ws?{final_query}"

    async def synthesize_stream(
        self,
        text: str,
        sample_rate: int = 16000,
    ) -> AsyncIterator[bytes]:
        """
        流式合成：连接腾讯 TTS WebSocket，每收到一段音频立即 yield。
        
        Usage:
            tts = TencentTTS()
            async for chunk in tts.synthesize_stream("你好世界"):
                # chunk 是 PCM int16 bytes
                send_to_client(chunk)
        """
        url = self._generate_url(text, sample_rate)
        logger.debug(f"TTS 连接: {url[:80]}...")

        async with websockets.connect(url, ssl=self._ssl_ctx) as ws:
            async for message in ws:
                # ── 文本帧：状态/字幕 ──
                if isinstance(message, str):
                    resp = json.loads(message)
                    code = resp.get("code", -1)

                    if code != 0:
                        logger.error(f"TTS 错误: {resp.get('message')}")
                        break

                    # final == 1 表示合成结束
                    if resp.get("final") == 1:
                        logger.info("TTS 合成完毕")
                        break

                # ── 二进制帧：PCM 音频数据 ──
                else:
                    yield message  # ✅ 一收到就 yield，零延迟

    # ── 保留兼容接口（同步写文件，用于独立测试） ──
    def start(self, text: str, output_file: str) -> None:
        """同步接口：合成并写入文件（向后兼容）."""
        async def _run():
            with open(output_file, "wb") as f:
                async for chunk in self.synthesize_stream(text):
                    f.write(chunk)

        asyncio.run(_run())