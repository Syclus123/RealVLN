"""音频 WebSocket 客户端主模块.

提供 SDK 级别的 AudioWebSocketClient 类，支持：
- 事件回调（通过 EventHandler）
- 同步 / 异步两种运行方式
- 可自定义日志配置
"""

import asyncio
import base64
import json
import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import numpy as np
import sounddevice as sd
import websockets
from websockets.client import WebSocketClientProtocol

from .config import ClientConfig

from .device import (
    AUDIO_INPUT_DEVICE_ID,
    AUDIO_OUTPUT_DEVICE_ID,
    AUDIO_INPUT_DEVICE_CHANNALS,
    AUDIO_OUTPUT_DEVICE_CHANNALS,
    AUDIO_INPUT_SAMPLE_RATE,
    AUDIO_OUTPUT_SAMPLE_RATE,
)

# 模块级 logger，默认不添加任何 handler（由调用方决定日志输出方式）
logger = logging.getLogger("audio_client")


# ---------------------------------------------------------------------------
# 事件回调基类
# ---------------------------------------------------------------------------

class EventHandler:
    """事件回调基类.
    
    调用方通过继承此类并重写感兴趣的方法来接收事件。
    所有回调方法均为 **同步** 方法，SDK 内部会自动通过线程池
    （``run_in_executor``）调度执行，**不会阻塞事件循环**。
    因此，回调中可以安全地执行网络 IO、数据库访问等耗时操作。
    
    .. note::
        回调在线程池中执行，如果多个回调间共享状态，
        请自行做好线程安全保护（如使用 ``threading.Lock``）。
    
    示例::
    
        class MyHandler(EventHandler):
            def on_session_created(self, session: dict):
                print("会话已创建")
            
            def on_ai_transcript_delta(self, text: str):
                print(f"AI: {text}", end="", flush=True)
            
            def on_user_transcript(self, text: str):
                # 可以安全地做网络 IO，不会阻塞事件循环
                import requests
                requests.post("https://api.example.com/transcript", json={"text": text})
            
            def on_error(self, code: str, message: str):
                print(f"错误 [{code}]: {message}")
    """

    def on_session_created(self, session: dict) -> None:
        """会话已创建.
        
        Args:
            session: 服务端返回的完整 session.created 消息体
        """

    def on_session_updated(self, session: dict) -> None:
        """会话配置已更新.
        
        Args:
            session: 服务端返回的完整 session.updated 消息体
        """

    def on_audio_playing(self, pcm_bytes: bytes) -> None:
        """收到一帧下行音频数据（已 base64 解码的 PCM）.
        
        Args:
            pcm_bytes: PCM 音频数据
        """

    def on_speech_started(self, event: dict) -> None:
        """VAD 检测到用户开始说话.
        
        Args:
            event: 服务端返回的完整 input_audio_buffer.speech_started 消息体
        """

    def on_speech_stopped(self, event: dict) -> None:
        """VAD 检测到用户停止说话.
        
        Args:
            event: 服务端返回的完整 input_audio_buffer.speech_stopped 消息体
        """

    def on_ai_transcript_delta(self, text: str) -> None:
        """AI 回复字幕增量.
        
        Args:
            text: 增量文本
        """

    def on_ai_transcript_done(self, event: dict) -> None:
        """AI 回复字幕输出完成.
        
        Args:
            event: 服务端返回的完整 response.output_audio_transcript.done 消息体，
                   通常包含 ``transcript`` 字段（完整文本）
        """

    def on_user_transcript(self, text: str) -> None:
        """用户语音识别结果.
        
        Args:
            text: 识别文本
        """

    def on_user_transcript_done(self, event: dict) -> None:
        """用户语音识别完成.
        
        Args:
            event: 服务端返回的完整 response.output_text.done 消息体
        """

    def on_error(self, code: str, message: str) -> None:
        """服务端返回错误.
        
        Args:
            code: 错误码
            message: 错误信息
        """

    def on_query_completed(self, item_id: str) -> None:
        """服务端已收到文本查询请求.
        
        对应信令: conversation.item.query.completed
        
        Args:
            item_id: 服务端返回的会话项 ID
        """

    def on_signal(self, signal_type: str, payload: dict) -> None:
        """未被以上方法处理的其他信令.
        
        Args:
            signal_type: 信令类型
            payload: 完整消息体
        """

    def on_connected(self) -> None:
        """WebSocket 连接成功."""

    def on_disconnected(self, reason: str) -> None:
        """WebSocket 连接断开.
        
        Args:
            reason: 断开原因
        """


# ---------------------------------------------------------------------------
# 日志辅助
# ---------------------------------------------------------------------------

def setup_logging(
    *,
    console_level: str = "INFO",
    log_file: Optional[str] = None,
    file_level: str = "DEBUG",
) -> None:
    """配置 audio_client 的日志输出.
    
    此函数可由 SDK 调用方按需调用；如果不调用，则 audio_client
    不会产生任何日志输出（遵循库的最佳实践）。
    
    Args:
        console_level: 控制台日志级别
        log_file: 日志文件路径，为 None 则不写文件
        file_level: 文件日志级别
    """
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 避免重复添加
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)

    if log_file:
        # 检查是否已经添加了同路径的文件 handler
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler) and h.baseFilename.endswith(log_file):
                return
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info(f"日志文件已创建: {log_file}")


# ---------------------------------------------------------------------------
# 核心客户端
# ---------------------------------------------------------------------------

class AudioWebSocketClient:
    """音频 WebSocket 客户端.
    
    实现与 Air Gateway 的 WebSocket 连接，支持：
    - 会话管理（session.update）
    - 音频上行（麦克风采集 -> PCM -> base64 -> input_audio_buffer.append）
    - 音频下行（response.output_audio.delta -> base64解码 -> PCM -> 播放）
    - 心跳保活（ping/pong）
    - VAD 事件处理
    - 事件回调（通过 EventHandler）
    
    快速开始::
    
        from audio_client import AudioWebSocketClient, ClientConfig, EventHandler, setup_logging
        
        # 可选：配置日志
        setup_logging(console_level="INFO", log_file="audio_client.log")
        
        # 自定义事件处理
        class MyHandler(EventHandler):
            def on_ai_transcript_delta(self, text: str):
                print(text, end="", flush=True)
            
            def on_user_transcript(self, text: str):
                print(f"\\n用户: {text}")
        
        # 初始化并运行
        config = ClientConfig(
            ws_url="ws://your-server/realtime",
            token="your-token",
            user_id="user_001",
        )
        client = AudioWebSocketClient(config, event_handler=MyHandler())
        client.run_forever()  # 阻塞运行，Ctrl+C 退出

    三种静音机制对比:
    ┌─────────────────────┬──────────────────┬──────────────────┬────────────────────┐
    │                     │ mute_microphone   │ cancel_response  │ mute_speaker ★     │
    ├─────────────────────┼──────────────────┼──────────────────┼────────────────────┤
    │ 影响方向            │ 上行（麦克风）    │ 下行（服务端）    │ 下行（扬声器）     │
    │ 数据流              │ 丢弃采集数据     │ 丢弃+停止生成    │ 暂停播放，不丢弃   │
    │ 服务端感知          │ 收不到音频       │ 收到cancel信令   │ 无感知             │
    │ 可恢复              │ ✅ unmute        │ ❌ 不可恢复      │ ✅ unmute 后补播   │
    │ 使用场景            │ 本地TTS时        │ 打断AI回复       │ 意图分类等待期     │
    └─────────────────────┴──────────────────┴──────────────────┴────────────────────┘

    """

    # 音频发送间隔（毫秒）
    AUDIO_SEND_INTERVAL_MS = 20
    # 上行音频包日志打印间隔
    UPLINK_AUDIO_LOG_INTERVAL = 10
    # 下行音频包日志打印间隔
    DOWNLINK_AUDIO_LOG_INTERVAL = 100

    def __init__(
        self,
        config: ClientConfig,
        event_handler: Optional[EventHandler] = None,
    ) -> None:
        """初始化客户端.
        
        Args:
            config: 客户端配置
            event_handler: 事件回调处理器，为 None 时使用默认空实现
        """
        self.config = config
        self.config.validate()

        self.event_handler = event_handler or EventHandler()

        self.ws: Optional[WebSocketClientProtocol] = None
        self.input_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.output_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._tasks: list[asyncio.Task] = []
        self._audio_buffer: bytes = b""

        # 音频包计数器
        self._uplink_audio_count = 0
        self._downlink_audio_count = 0

        # cancel 静音标志：cancel_response 后置 True，收到 speech_started 后恢复
        self._cancel_muted = False

        # 麦克风静音标志：mute 后麦克风硬件仍运行，但不再采集上行音频
        self._mic_muted = False

        # ★ 新增：扬声器静音标志, 不cancel_response
        self._speaker_muted = False

        # 音频采样率对齐
        self.ws_sample_rate = config.sample_rate  # 服务器要求的采样率 (16000)
        self.input_resample_ratio = AUDIO_INPUT_SAMPLE_RATE // self.ws_sample_rate
        self.output_resample_ratio = AUDIO_OUTPUT_SAMPLE_RATE // self.ws_sample_rate

        # 事件回调专用线程池，避免与 play_audio_loop 的 stream.write（默认线程池）争用导致 recv 卡顿
        self._callback_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="audio_client_cb",
        )

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """同步阻塞运行客户端.
        
        内部创建事件循环并运行，直到收到 SIGINT/SIGTERM 信号或调用 stop()。
        适用于脚本 / CLI / 简单集成场景。
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 注册信号处理
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop)

        try:
            loop.run_until_complete(self.run_async())
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()

    async def run_async(self) -> None:
        """异步运行客户端（连接 + 麦克风）.
        
        如果你已经有自己的事件循环，可以直接 await 此方法。
        """
        self.running = True
        self._loop = asyncio.get_event_loop()

        mic_task = asyncio.create_task(self._start_microphone(), name="microphone")
        self._tasks.append(mic_task)

        try:
            await asyncio.gather(
                self._connect(),
                mic_task,
            )
        except asyncio.CancelledError:
            logger.info("主运行循环被取消")

    def stop(self) -> None:
        """停止客户端（线程安全，可从任意线程调用）."""
        logger.info("正在停止客户端...")
        self.running = False

        for task in self._tasks:
            if not task.done():
                task.cancel()

    async def stop_async(self) -> None:
        """异步停止客户端并等待资源清理完成."""
        self.stop()
        await self._cleanup()

    async def cancel_response(self) -> None:
        """取消当前正在进行的 AI 响应（发送 response.cancel 信令）.
        
        客户端主动调用此方法，可以中断服务端正在生成的回复，
        同时清空本地的下行音频播放队列。
        
        符合 OpenAI Realtime API 协议规范。
        """
        if not self.ws or not self.running:
            logger.warning("cancel_response: 客户端未连接，忽略操作")
            return

        # 进入静音模式：清空播放队列并丢弃后续下行音频，直到 speech_started
        self._cancel_muted = True

        # ★ 确保 speaker 也 unmute（避免残留 muted 状态）
        self._speaker_muted = False

        await self._clear_output_queue()
        logger.info("cancel_response: 已进入静音模式，等待下次 speech_started 恢复")

        cancel_msg = {
            "type": "response.cancel",
            "event_id": f"evt_cancel_{int(time.time() * 1000)}",
        }
        try:
            await self._send_message(cancel_msg)
            logger.info("已发送 response.cancel 信令")
        except Exception as e:
            logger.error(f"发送 response.cancel 失败: {e}")

    def mute_microphone(self) -> None:
        """静音麦克风（停止采集上行音频，但不关闭硬件设备）.
        
        调用后麦克风硬件仍保持打开状态，但采集到的音频数据会被丢弃，
        不再发送到服务端。恢复采集请调用 :meth:`unmute_microphone`。
        
        此方法是线程安全的，可在任意线程中调用。
        """
        self._mic_muted = True
        logger.info("🎤 麦克风已静音（停止采集上行音频）")

    def unmute_microphone(self) -> None:
        """取消麦克风静音（恢复采集上行音频）.
        
        此方法是线程安全的，可在任意线程中调用。
        """
        self._mic_muted = False
        logger.info("🎤 麦克风已恢复采集")

    @property
    def is_microphone_muted(self) -> bool:
        """当前麦克风是否处于静音状态."""
        return self._mic_muted

    # ★★★ 新增：扬声器静音控制 ★★★

    def mute_speaker(self) -> None:
        """静音扬声器（暂停消费 output_queue，数据留在 queue 中不丢弃）.

        行为:
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
        │  mute_speaker()                                          │
        │    │                                                     │
        │    ├→ _play_audio_loop 暂停从 queue 取数据               │
        │    ├→ 服务端继续推送音频 → 数据在 queue 中积累            │
        │    ├→ 声卡 underrun → 自动静音（不需要写零）             │
        │    │                                                     │
        │    ├─ 最终 unmute_speaker():                             │
        │    │   queue 中积累的数据按顺序播放                       │
        │    │   用户听到完整音频（只是前面有一小段静默）            │
        │    │                                                     │
        │    └─ 最终 cancel_response():                            │
        │        queue 被清空，积累的数据全部丢弃                   │
        │        用户什么都听不到 ✅                                │
        │                                                          │
        └──────────────────────────────────────────────────────────┘

        与 cancel_response 的区别:
          cancel_response: 不可逆，丢弃数据，停止服务端生成
          mute_speaker:    可逆，保留数据，服务端无感知

        线程安全，可从任意线程调用。
        """
        self._speaker_muted = True
        logger.info("🔇 扬声器已静音（暂停播放，数据缓冲在 queue 中）")

    def unmute_speaker(self) -> None:
        """恢复扬声器（恢复消费 output_queue，积累的数据会自动补播）.

        补播行为:
          mute 期间积累了 N 个音频包在 queue 中
          unmute 后 _play_audio_loop 立即恢复取数据并播放
          声卡 write 是阻塞的，N 个包会顺序播放
          用户感知: 前面一小段静默 → 后续音频完整连续

        线程安全，可从任意线程调用。
        """
        self._speaker_muted = False
        logger.info("🔊 扬声器已恢复（queue 中积累的数据将自动播放）")

    @property
    def is_speaker_muted(self) -> bool:
        """当前扬声器是否处于静音状态."""
        return self._speaker_muted
    
    def cancel_response_sync(self) -> None:
        """取消当前正在进行的 AI 响应（同步版本，线程安全）.
        
        可在任意线程中调用，内部会将操作提交到事件循环。
        """
        if self._loop and self.running:
            asyncio.run_coroutine_threadsafe(
                self.cancel_response(),
                self._loop,
            )

    async def send_text_query(self, text: str) -> None:
        """发送文本查询，主动请求 AI 回复（发送 conversation.item.query 信令）.
        
        用户通过此方法发送文字消息，服务端收到后会生成 AI 回复。
        符合 conversation.item.query 协议规范。
        
        Args:
            text: 用户输入的文本内容
        """
        if not self.ws or not self.running:
            logger.warning("send_text_query: 客户端未连接，忽略操作")
            return

        query_msg = {
            "type": "conversation.item.query",
            "id": f"msg_{int(time.time() * 1000)}",
            "role": "user",
            "content": {
                "type": "input_text",
                "text": text,
            },
        }
        try:
            await self._send_message(query_msg)
            logger.info(f"已发送文本查询: {text}")
        except Exception as e:
            logger.error(f"发送文本查询失败: {e}")

    def send_text_query_sync(self, text: str) -> None:
        """发送文本查询（同步版本，线程安全）.
        
        可在任意线程中调用，内部会将操作提交到事件循环。
        
        Args:
            text: 用户输入的文本内容
        """
        if self._loop and self.running:
            asyncio.run_coroutine_threadsafe(
                self.send_text_query(text),
                self._loop,
            )

    # ------------------------------------------------------------------
    # 回调调度
    # ------------------------------------------------------------------

    def _run_callback_safe(self, callback, args) -> None:
        """在线程池中执行回调并捕获异常，用于 fire-and-forget 时记录错误."""
        try:
            callback(*args)
        except Exception as e:
            logger.error(f"回调执行异常 [{getattr(callback, '__name__', 'unknown')}]: {e}")

    async def _emit(self, callback, *args, block: bool = True) -> None:
        """将事件回调提交到专用线程池执行，避免与 stream.write 争用导致 recv 卡顿.

        Args:
            callback: EventHandler 上的回调方法
            *args: 传递给回调的参数
            block: 若 True 则等待回调结束再返回；若 False 则提交后立即返回，不阻塞 recv（用于高频率回调如 on_audio_playing）
        """
        loop = asyncio.get_event_loop()
        if block:
            try:
                await loop.run_in_executor(
                    self._callback_executor,
                    self._run_callback_safe,
                    callback,
                    args,
                )
            except Exception as e:
                logger.error(f"回调调度异常 [{getattr(callback, '__name__', 'unknown')}]: {e}")
        else:
            loop.run_in_executor(
                self._callback_executor,
                self._run_callback_safe,
                callback,
                args,
            )

    # ------------------------------------------------------------------
    # WebSocket 连接
    # ------------------------------------------------------------------

    def _build_ws_url(self) -> str:
        """构建带 user_id 参数的 WebSocket URL."""
        url = urlparse(self.config.ws_url)
        qs = parse_qs(url.query)
        qs["user_id"] = [self.config.user_id]
        qs["model"] = [self.config.agent_id]
        new_query = urlencode(qs, doseq=True)
        return urlunparse(url._replace(query=new_query))

    async def _connect(self) -> None:
        """建立 WebSocket 连接并启动收发循环."""
        ws_url = self._build_ws_url()
        logger.info(f"正在连接: {ws_url}")

        headers = {
            "Authorization": f"Bearer {self.config.token}",
        }
        logger.debug(f"连接 Headers: {headers}")

        try:
            self.ws = await websockets.connect(
                ws_url,
                additional_headers=headers,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=5,
            )
            self.running = True
            self._loop = asyncio.get_event_loop()
            logger.info("WebSocket 连接成功")
            await self._emit(self.event_handler.on_connected)

            self._tasks = [
                asyncio.create_task(self._recv_loop(), name="recv_loop"),
                asyncio.create_task(self._send_audio_loop(), name="send_audio_loop"),
                asyncio.create_task(self._play_audio_loop(), name="play_audio_loop"),
                asyncio.create_task(self._ping_loop(), name="ping_loop"),
            ]

            await asyncio.gather(*self._tasks, return_exceptions=True)
        except websockets.exceptions.ConnectionClosed as e:
            code = getattr(e, "code", None)
            reason = getattr(e, "reason", None) or str(e)
            logger.warning("连接已关闭 code=%s reason=%r", code, reason)
            await self._emit(self.event_handler.on_disconnected, str(e))
        except asyncio.CancelledError:
            logger.info("连接任务被取消")
            await self._emit(self.event_handler.on_disconnected, "任务被取消")
        except Exception as e:
            logger.error(f"连接错误: {e}")
            await self._emit(self.event_handler.on_disconnected, str(e))
        finally:
            self.running = False
            await self._cleanup()

    async def _cleanup(self) -> None:
        """清理资源."""
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._tasks.clear()

        if self._callback_executor is not None:
            self._callback_executor.shutdown(wait=True)
            self._callback_executor = None

        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

    # ------------------------------------------------------------------
    # 消息收发
    # ------------------------------------------------------------------

    async def _recv_loop(self) -> None:
        """接收消息循环."""
        assert self.ws is not None

        try:
            async for message in self.ws:
                if not self.running:
                    break
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            # code 1000=正常关闭 1001=端点离开 1006=异常断开(无close帧) 4xxx=服务端自定义(如超时)
            logger.info(
                "接收循环：连接已关闭 code=%s reason=%r",
                getattr(e, "code", None),
                getattr(e, "reason", None),
            )
        except asyncio.CancelledError:
            logger.info("接收循环：任务被取消")
            raise

    async def _handle_message(self, message: str | bytes) -> None:
        """处理接收到的消息."""
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("无法解析 JSON 消息")
            return

        msg_type = payload.get("type", "")

        if msg_type == "session.created":
            logger.info(f"📩 收到信令: {msg_type}")
            logger.debug(f"📩 信令详情: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            logger.info("✓ 会话已创建")
            await self._emit(self.event_handler.on_session_created, payload)
            await self._send_session_update()

        elif msg_type == "session.updated":
            logger.info(f"📩 收到信令: {msg_type}")
            logger.debug(f"📩 信令详情: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            logger.info("✓ 会话配置已更新")
            await self._emit(self.event_handler.on_session_updated, payload)

        elif msg_type == "response.output_audio.delta":
            delta = payload.get("delta")
            if delta:
                try:
                    pcm_bytes = base64.b64decode(delta)

                    # cancel 静音模式下丢弃所有下行音频
                    if self._cancel_muted:
                        self._downlink_audio_count += 1
                        if self._downlink_audio_count % self.DOWNLINK_AUDIO_LOG_INTERVAL == 0:
                            logger.debug(
                                f"📥 静音模式: 已丢弃 {self._downlink_audio_count} 个下行音频包"
                            )
                        return

                    await self.output_queue.put(pcm_bytes)

                    self._downlink_audio_count += 1
                    if self._downlink_audio_count % self.DOWNLINK_AUDIO_LOG_INTERVAL == 0:
                        logger.info(
                            f"📥 下行音频包: 已接收 {self._downlink_audio_count} 个包, "
                            f"本次大小: {len(pcm_bytes)} bytes"
                        )
                    # 高频率回调不阻塞 recv，避免积压导致“卡顿”和 VAD 日志“突然出现”
                    await self._emit(
                        self.event_handler.on_audio_playing,
                        pcm_bytes,
                        block=False,
                    )
                except Exception as e:
                    logger.error(f"解码音频数据失败: {e}")

        elif msg_type == "input_audio_buffer.speech_started":
            # 服务端 VAD 检测到语音开始会发此信令；若 recv 之前积压了较多消息，会“延迟”打印，属正常
            logger.info("🎤 VAD: 检测到语音开始，清空播放缓冲区")
            # 收到 speech_started 时解除 cancel 静音模式
            if self._cancel_muted:
                self._cancel_muted = False
                logger.info("🎤 cancel 静音模式已解除")
            await self._clear_output_queue()
            await self._emit(self.event_handler.on_speech_started, payload)

        elif msg_type == "input_audio_buffer.speech_stopped":
            logger.info("🎤 VAD: 检测到语音结束")
            await self._emit(self.event_handler.on_speech_stopped, payload)

        elif msg_type == "response.output_audio_transcript.delta":
            delta = payload.get("delta", "")
            if delta:
                await self._emit(self.event_handler.on_ai_transcript_delta, delta)

        elif msg_type == "response.output_audio_transcript.done":
            await self._emit(self.event_handler.on_ai_transcript_done, payload)

        elif msg_type == "response.output_text.delta":
            transcript = payload.get("delta", "")
            if transcript:
                logger.info(f"用户字幕识别: {transcript}")
                await self._emit(self.event_handler.on_user_transcript, transcript)
        elif msg_type == "response.output_text.done":
            logger.info("用户字幕识别完成")
            await self._emit(self.event_handler.on_user_transcript_done, payload)
        elif msg_type == "conversation.item.query.completed":
            item_id = payload.get("item_id", "")
            logger.info(f"📩 收到信令: {msg_type}, item_id={item_id}")
            logger.debug(f"📩 信令详情: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            await self._emit(self.event_handler.on_query_completed, item_id)

        elif msg_type == "error":
            error_msg = payload.get("message", "未知错误")
            error_code = payload.get("code", "")
            logger.error(f"❌ 服务端错误 [{error_code}]: {error_msg}")
            await self._emit(self.event_handler.on_error, error_code, error_msg)

        elif msg_type == "pong":
            logger.debug("收到 pong")

        else:
            logger.info(f"📩 收到信令: {msg_type}")
            logger.debug(f"📩 信令详情: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            await self._emit(self.event_handler.on_signal, msg_type, payload)

    async def _clear_output_queue(self) -> None:
        """清空下行音频播放队列."""
        cleared_count = 0
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        if cleared_count > 0:
            logger.info(f"已清空 {cleared_count} 个待播放音频包")

    async def _send_message(self, msg: dict) -> None:
        """发送消息并记录日志."""
        assert self.ws is not None

        msg_type = msg.get("type", "")
        msg_json = json.dumps(msg)

        await self.ws.send(msg_json)

        if msg_type == "input_audio_buffer.append":
            self._uplink_audio_count += 1
            if self._uplink_audio_count % self.UPLINK_AUDIO_LOG_INTERVAL == 0:
                audio_len = len(msg.get("audio", ""))
                logger.info(
                    f"📤 上行音频包: 已发送 {self._uplink_audio_count} 个包, "
                    f"本次大小: {audio_len} bytes (base64)"
                )
        elif msg_type == "ping":
            logger.debug(f"📤 发送信令: {msg_type}")
        else:
            logger.info(f"📤 发送信令: {msg_type}")
            logger.debug(f"📤 信令详情: {json.dumps(msg, ensure_ascii=False, indent=2)}")

    async def _send_session_update(self) -> None:
        """发送 session.update 消息配置会话."""
        assert self.ws is not None

        cfg = self.config
        update_msg = {
            "type": "session.update",
            "session": {
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": cfg.sample_rate,
                        }
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": cfg.sample_rate,
                        },
                        "voice": cfg.tone_id,
                    },
                },
                "meta_data": {
                    "channel": cfg.channels,
                    "subtitle": cfg.subtitle,
                    "stream_subtitle": cfg.stream_subtitle,
                    "business_id": cfg.business_id,
                    "disable_greeting": cfg.disable_greeting,
                    "scene_type": cfg.scene_type,
                },
            },
        }

        # 只有在不禁用问候语时才添加问候语字段
        if not cfg.disable_greeting:
            update_msg["session"]["meta_data"]["greeting"] = "你好, 我是你的 AI 助手, 有什么可以帮助你的吗?"

        await self._send_message(update_msg)

    # ------------------------------------------------------------------
    # 心跳
    # ------------------------------------------------------------------

    async def _ping_loop(self) -> None:
        """心跳循环."""
        assert self.ws is not None

        try:
            while self.running:
                try:
                    ping_msg = {
                        "type": "ping",
                        "event_id": f"evt_ping_{int(time.time() * 1000)}",
                    }
                    await self._send_message(ping_msg)
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(
                        "心跳循环：连接已关闭 code=%s reason=%r",
                        getattr(e, "code", None),
                        getattr(e, "reason", None),
                    )
                    break
                except Exception as e:
                    logger.warning(f"发送心跳失败: {e}")
                    break

                await asyncio.sleep(self.config.ping_interval_sec)
        except asyncio.CancelledError:
            logger.info("心跳循环：任务被取消")
            raise

    # ------------------------------------------------------------------
    # 音频上行
    # ------------------------------------------------------------------

    def _calc_frame_bytes(self) -> int:
        """计算每帧音频的字节数.
        
        帧长度 = 采样率 * 通道数 * 每样本字节数 * 帧时长(秒)
        例如: 16000 * 1 * 2 * 0.02 = 640 字节 (20ms 一帧)
        """
        cfg = self.config
        bytes_per_sample = cfg.bits_per_sample // 8
        samples_per_frame = int(cfg.sample_rate * self.AUDIO_SEND_INTERVAL_MS / 1000)
        return samples_per_frame * cfg.channels * bytes_per_sample

    async def _send_audio_loop(self) -> None:
        """音频上行循环（按 20ms 间隔持续发送，即使无数据也发送静音帧）."""
        assert self.ws is not None

        cfg = self.config
        frame_bytes = self._calc_frame_bytes()
        logger.info(f"音频帧配置: 间隔={self.AUDIO_SEND_INTERVAL_MS}ms, 帧大小={frame_bytes}字节")

        try:
            while self.running:
                send_start_time = time.monotonic()

                # 从队列取出所有可用数据到缓冲区
                while not self.input_queue.empty():
                    try:
                        pcm_data = self.input_queue.get_nowait()
                        if pcm_data:
                            self._audio_buffer += pcm_data
                    except asyncio.QueueEmpty:
                        break

                if len(self._audio_buffer) < frame_bytes:
                    # 数据不足一帧，跳过本次发送
                    await asyncio.sleep(self.AUDIO_SEND_INTERVAL_MS / 1000)
                    continue

                # 取 20ms 整数倍的数据一次性发送
                frames_to_send = (len(self._audio_buffer) // frame_bytes) * frame_bytes
                audio_to_send = self._audio_buffer[:frames_to_send]
                self._audio_buffer = self._audio_buffer[frames_to_send:]

                base64_data = base64.b64encode(audio_to_send).decode("utf-8")
                msg = {
                    "type": "input_audio_buffer.append",
                    "audio": base64_data,
                    "format": {
                        "sample_rate": cfg.sample_rate,
                        "channels": cfg.channels,
                        "bit_depth": cfg.bits_per_sample,
                        "encoding": "pcm",
                    },
                }

                try:
                    await self._send_message(msg)
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(
                        "音频发送循环：连接已关闭 code=%s reason=%r",
                        getattr(e, "code", None),
                        getattr(e, "reason", None),
                    )
                    break
                except Exception as e:
                    logger.error(f"发送音频数据失败: {e}")
                    break

                elapsed = time.monotonic() - send_start_time
                sleep_time = (self.AUDIO_SEND_INTERVAL_MS / 1000) - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            logger.info("音频发送循环：任务被取消")
            raise

    # ------------------------------------------------------------------
    # 音频下行播放
    # ------------------------------------------------------------------

    async def _play_audio_loop(self) -> None:
        """音频下行播放循环.

        ★ 改动: speaker_muted 时暂停消费 queue.

        数据流:
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
        │  服务端 ──→ _handle_message ──→ output_queue ──→ 这里    │
        │                                      ▲                   │
        │                                      │                   │
        │                              speaker_muted?              │
        │                              │            │              │
        │                             Yes           No             │
        │                              │            │              │
        │                      sleep(10ms)    queue.get()          │
        │                      数据留在queue   取出 → 播放          │
        │                              │            │              │
        │                              │            ▼              │
        │                              │      resample → write     │
        │                              │                           │
        │  cancel_response → queue.clear() → 数据全部丢弃          │
        │  unmute_speaker → 恢复 get() → 积累的数据顺序播放        │
        │                                                          │
        └──────────────────────────────────────────────────────────┘

        为什么不写静音帧?
          sounddevice OutputStream 在没有 write 调用时会自动 underrun，
          声卡输出静音。不需要显式写零。
          这样代码更简洁，且避免了"写零但丢弃原始数据"的bug。
        """
        stream = sd.OutputStream(
            device=AUDIO_OUTPUT_DEVICE_ID,
            samplerate=AUDIO_OUTPUT_SAMPLE_RATE,
            channels=AUDIO_OUTPUT_DEVICE_CHANNALS,
            dtype="int16",
            blocksize=int(
                AUDIO_OUTPUT_SAMPLE_RATE
                * self.AUDIO_SEND_INTERVAL_MS
                / 1000
            ),
        )
        stream.start()
        logger.info("🔊 音频播放已启动")

        try:
            while self.running:
                # ★★★ 核心改动：speaker 静音时暂停消费 queue ★★★
                if self._speaker_muted:
                    # 不从 queue 取数据，数据在 queue 中自然积累
                    # 声卡 underrun → 自动静音
                    await asyncio.sleep(0.01)
                    continue

                try:
                    data = await asyncio.wait_for(
                        self.output_queue.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    continue

                if not data:
                    continue

                pcm_16k = np.frombuffer(data, dtype=np.int16)
                pcm = np.repeat(pcm_16k, self.output_resample_ratio)

                if AUDIO_OUTPUT_DEVICE_CHANNALS == 1:
                    pcm = pcm.reshape(-1, 1)
                else:
                    pcm = np.column_stack(
                        [pcm] * AUDIO_OUTPUT_DEVICE_CHANNALS
                    )

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, stream.write, pcm)

        except asyncio.CancelledError:
            logger.info("音频播放循环：任务被取消")
        finally:
            stream.stop()
            stream.close()
            logger.info("🔊 音频播放已停止")

    # ------------------------------------------------------------------
    # 麦克风采集
    # ------------------------------------------------------------------

    def _input_audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """麦克风输入回调."""
        if status:
            logger.warning(f"音频输入状态: {status}")
            return

        # 1. 降采样
        # indata shape 是 (frames, channels)，我们取第一个通道并降采样
        resampled_data = indata[::self.input_resample_ratio, 0]

        # 静音模式下发送全 0 静音帧，保持上行音频流连续
        if self._mic_muted:
            pcm = bytes(len(resampled_data) * 2)  # int16 每样本 2 字节，全 0
        else:
            # 2. float32 -> int16 PCM
            pcm = (resampled_data.copy() * 32767).astype(np.int16).tobytes()

        if self._loop and self.running:
            asyncio.run_coroutine_threadsafe(
                self.input_queue.put(pcm),
                self._loop,
            )

    async def _start_microphone(self) -> None:
        """启动麦克风采集."""
        cfg = self.config
        block_size = int(cfg.sample_rate * cfg.frame_duration_ms / 1000)

        stream = sd.InputStream(
            device=AUDIO_INPUT_DEVICE_ID,
            samplerate=AUDIO_INPUT_SAMPLE_RATE,
            channels=AUDIO_INPUT_DEVICE_CHANNALS,
            dtype="float32",
            blocksize=int(AUDIO_INPUT_SAMPLE_RATE * self.AUDIO_SEND_INTERVAL_MS / 1000),
            callback=self._input_audio_callback,
        )
        stream.start()
        logger.info("🎤 麦克风已开启")

        try:
            while self.running:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("麦克风循环：任务被取消")
        finally:
            stream.stop()
            stream.close()
            logger.info("🎤 麦克风已关闭")


