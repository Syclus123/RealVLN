"""音频 WebSocket 服务端.

流程: 音频流入 → VAD 端点检测 → ASR(整段) → 文本输出
"""

import argparse
import asyncio
import base64
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
from urllib.parse import parse_qs, urlparse
import time


import os

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)


import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

from .vad import RMSVADConfig as VADConfig
from .vad import RMSVoiceActivityDetector as VoiceActivityDetector
from .asr import Qwen3ASR as ASRModule


# ──────────────────────────────────────────────
# 日志
# ──────────────────────────────────────────────
logger = logging.getLogger("audio_server")
logger.setLevel(logging.DEBUG)

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(_console)

_file_handler = logging.FileHandler("audio_server.log", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(_file_handler)

# ══════════════════════════════════════════════
#  会话
# ══════════════════════════════════════════════

@dataclass
class SessionConfig:
    sample_rate: int = 16000
    channels: int = 1
    voice_id: str = "default"
    subtitle: bool = False


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    user_id: str = ""
    config: SessionConfig = field(default_factory=SessionConfig)

    ws: Optional[WebSocketServerProtocol] = None
    vad: VoiceActivityDetector = field(default_factory=VoiceActivityDetector)
    asr: ASRModule = field(default_factory=lambda: ASRModule(logger=logger))
    audio_remainder: bytes = b""  # 不足一帧的残余字节
    pipeline_task: Optional[asyncio.Task] = None

    # ✅ 新增
    audio_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

# ══════════════════════════════════════════════
#  服务器
# ══════════════════════════════════════════════

DEBUG_INPUT_AUDIO_STREAM = False

def append_pcm_to_file(output_path, pcm):
    with open(output_path, "ab") as wf:
        wf.write(pcm)

class AudioWebSocketServer:

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        path: str = "/ws",
        valid_tokens: set[str] | None = None,
    ):
        self.host = host
        self.port = port
        self.path = path
        self.valid_tokens = valid_tokens
        self.sessions: dict[str, Session] = {}

    # ── 工具 ──

    async def _send(self, ws: WebSocketServerProtocol, msg: dict) -> bool:
        """发送 JSON, 成功返回 True."""
        try:
            await ws.send(json.dumps(msg, ensure_ascii=False))
            return True
        except websockets.exceptions.ConnectionClosed:
            return False

    def _authenticate(self, ws: WebSocketServerProtocol) -> bool:
        if self.valid_tokens is None:
            return True
        auth = ws.request_headers.get("Authorization", "")
        return auth.startswith("Bearer ") and auth[7:] in self.valid_tokens

    @staticmethod
    def _extract_user_id(ws: WebSocketServerProtocol) -> str:
        path = ws.request.path if ws.request else ""
        qs = parse_qs(urlparse(path).query)
        return qs.get("user_id", [f"anon_{uuid.uuid4().hex[:8]}"])[0]

    # ── 连接 ──
    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        if not self._authenticate(ws):
            await self._send(ws, {"type": "error", "code": "auth_failed", "message": "Invalid token"})
            await ws.close(4001, "Unauthorized")
            return

        user_id = self._extract_user_id(ws)
        session = Session(user_id=user_id, ws=ws)
        self.sessions[session.session_id] = session
        logger.info(f"✅ 连接: user={user_id}, session={session.session_id}")

        await self._send(ws, {"type": "session.created", "session_id": session.session_id})

        # ✅ 启动独立的音频处理协程
        audio_processor = asyncio.create_task(
            self._audio_process_loop(session),
            name=f"audio_processor_{session.session_id}",
        )

        try:
            async for raw in ws:
                await self._route(session, raw)
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"连接关闭: {e}")
        finally:
            # 清理
            audio_processor.cancel()
            if session.pipeline_task and not session.pipeline_task.done():
                session.pipeline_task.cancel()

            # 等待任务结束（带超时）
            for t in [audio_processor, session.pipeline_task]:
                if t and not t.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(t), timeout=2.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass

            self.sessions.pop(session.session_id, None)
            logger.info(f"🔌 会话结束: {session.session_id}")

    async def _route(self, session: Session, raw: str | bytes) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = payload.get("type", "")

        if msg_type == "input_audio_buffer.append":
            audio_b64 = payload.get("audio", "")
            if audio_b64:
                try:
                    pcm = base64.b64decode(audio_b64)
                    if DEBUG_INPUT_AUDIO_STREAM:
                        append_pcm_to_file(f"debug_input_audio{session.session_id}.pcm", pcm)

                    # ✅ 用无限队列，绝不丢包
                    await session.audio_queue.put(pcm)
                except Exception as e:
                    logger.warning(f"音频解码失败: {e}")


        elif msg_type == "session.update":
            await self._on_session_update(session, payload)

        elif msg_type == "ping":
            await self._send(session.ws, {
                "type": "pong",
                "event_id": payload.get("event_id", ""),
            })

    async def _audio_process_loop(self, session: Session) -> None:
        """独立协程：从队列读取音频 → VAD → 触发 pipeline."""
        logger.info(f"🎧 音频处理协程启动: {session.session_id}")

        try:
            while True:
                try:
                    pcm = await asyncio.wait_for(
                        session.audio_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                # ✅ 批量取出队列中所有积压的数据，减少处理延迟
                chunks = [pcm]
                while not session.audio_queue.empty():
                    try:
                        chunks.append(session.audio_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                
                buf = session.audio_remainder + b"".join(chunks)
                frame_size = session.vad.frame_bytes

                while len(buf) >= frame_size:
                    frame = buf[:frame_size]
                    buf = buf[frame_size:]

                    for evt in session.vad.process_frame(frame):
                        #logger.debug(f"evt: f{evt}")
                        if evt["event"] == "speech_started":
                            await self._on_speech_started(session)
                        elif evt["event"] == "speech_stopped":
                            await self._on_speech_stopped(session, evt["audio"])

                # ✅ 保留不足一帧的余数
                session.audio_remainder = buf

        except asyncio.CancelledError:
            logger.info(f"🎧 音频处理协程退出: {session.session_id}")
            raise

    # ── session.update ──

    async def _on_session_update(self, session: Session, payload: dict) -> None:
        sess = payload.get("session", {})
        audio_cfg = sess.get("audio", {})
        meta = sess.get("meta_data", {})

        inp_rate = audio_cfg.get("input", {}).get("format", {}).get("rate")
        if inp_rate:
            session.config.sample_rate = int(inp_rate)

        voice = audio_cfg.get("output", {}).get("voice")
        if voice:
            session.config.voice_id = voice

        if "channel" in meta:
            session.config.channels = int(meta["channel"])
        if "subtitle" in meta:
            session.config.subtitle = bool(meta["subtitle"])

        # 采样率变化需重建 VAD
        session.vad = VoiceActivityDetector(VADConfig(sample_rate=session.config.sample_rate))
        session.audio_remainder = b""

        logger.info(f"📋 会话更新: rate={session.config.sample_rate}, voice={session.config.voice_id}")
        await self._send(session.ws, {"type": "session.updated"})

    # ── 语音事件 ──
    async def _on_speech_started(self, session: Session) -> None:
        """用户开始说话 → 打断旧 pipeline."""
        if session.pipeline_task and not session.pipeline_task.done():
            logger.info("⚡ 用户打断")
            session.pipeline_task.cancel()
            session._cancel_event.set()
            

        await self._send(session.ws, {
            "type": "input_audio_buffer.speech_started",
        })

    async def _on_speech_stopped(self, session: Session, speech_audio: bytes) -> None:
        """用户说完 → 启动新 pipeline."""
        
        # ✅ 检查音频是否足够长（过短可能是误触发）
        min_audio_bytes = int(session.config.sample_rate * 2 * 0.3)  # 至少 300ms
        if len(speech_audio) < min_audio_bytes:
            logger.info(f"⚠️ 音频太短 ({len(speech_audio)} bytes)，跳过")
            return

        # 取消旧 pipeline
        if session.pipeline_task and not session.pipeline_task.done():
            session.pipeline_task.cancel()
            session._cancel_event.set()
            try:
                await asyncio.wait_for(session.pipeline_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        session._cancel_event.clear()
        
        logger.info(f"🎙️ 收到语音: {len(speech_audio)} bytes, "
                     f"{len(speech_audio) / (session.config.sample_rate * 2):.1f}s")
        
        session.pipeline_task = asyncio.create_task(
            self._pipeline(session, speech_audio)
        )

    # ── 核心 ASR Pipeline: 
    async def _pipeline(self, session: Session, speech_audio: bytes = None, reply_text = None) -> None:
        ws = session.ws
        if ws is None:
            return

        response_id = uuid.uuid4().hex[:12]

        await self._send(ws, {"type": "input_audio_buffer.speech_stopped"})

        try:
            # ── ASR ──
            logger.info("🔄 ASR ...")
            user_text = await session.asr.recognize(speech_audio, session.config.sample_rate)
            
            await asyncio.sleep(0)  # 显式让出，检查 CancelledError
            
            logger.info(f"✅ ASR: {user_text}")

            await self._send(ws, {
                "type": "response.output_text.delta",
                "delta": user_text,
            })

            await self._send(ws, {
                "type": "response.output_text.done",
            })

            await self._send(ws, {
                "type": "response.done",
            })

        except asyncio.CancelledError:
            logger.info("⚡ Pipeline 被打断")
            # ✅ 通知客户端当前回复已中断
            await self._send(ws, {
                "type": "response.cancelled",
                "response_id": response_id,
            })
            return 
        except Exception as e:
            logger.error(f"Pipeline 异常: {e}", exc_info=True)
            await self._send(ws, {"type": "error", "code": "pipeline_error", "message": str(e)})

    # ── 启动 ──

    async def start(self) -> None:
        logger.info(f"🚀 ws://{self.host}:{self.port}{self.path}")
        async with websockets.serve(
            self._handler, self.host, self.port,
            ping_interval=None, ping_timeout=None,
            max_size=10 * 1024 * 1024,
        ):
            await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--path", default="/ws")
    parser.add_argument("--tokens", nargs="*")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    _console.setLevel(getattr(logging, args.log_level))

    server = AudioWebSocketServer(
        host=args.host, port=args.port, path=args.path,
        valid_tokens=set(args.tokens) if args.tokens else None,
    )

    print(f"Audio WebSocket Server | ws://{args.host}:{args.port}{args.path}")
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n✅ 已退出")


if __name__ == "__main__":
    main()
