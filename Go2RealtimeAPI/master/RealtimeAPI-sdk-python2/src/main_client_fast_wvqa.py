"""
基于 main_client_fast.py，新增 VQA（视觉问答）意图处理。

核心逻辑 (预判 + VQA):
  speech_started   → OPEN, unmute, reset
  partial=nav      → cancel_response() 立即止损 + TTS 入队
  partial=chat     → unmute_speaker() 立即放行
  partial=vqa      → cancel_response() + TTS "好的，让我看看"
                     → 等拿到完整文本后启动 VQA pipeline (拍照 → VLM → TTS 播报)
  speech_stopped   → 已决策? 跳过 : mute_speaker() 封堵
  transcript_done  →
    已决策=nav      → 写 nav_log, 跳过
    已决策=chat     → 跳过 (已放行)
    已决策=vqa      → 启动 VQA pipeline (用完整文本)
    未决策          → 等待预判完成或超时兜底
  timeout_guard    → unmute_speaker() 兜底
"""

import asyncio
import json
import logging
import re
import time
import enum
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid
import argparse
import sys

import os

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

import aiofiles
import numpy as np

from audio_client import (
    AudioWebSocketClient,
    ClientConfig,
    EventHandler,
    setup_logging,
)

from .intentllm import QwenLLM, call_llm_api
from .vlm import QwenVLM, call_vlm_api
from .tts_client import TencentTTS
from .functions import GO2_TOOLS

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient
    HAS_CAMERA_SDK = True
except ImportError:
    HAS_CAMERA_SDK = False

logger = logging.getLogger("main_client")
logger.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(_console_handler)

_file_handler = logging.FileHandler("main_client.log", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(_file_handler)
logger.info("日志文件已创建: main_client.log")


intent_llm_system_prompt = '''
你是一个搭载了视觉感知和移动导航模块的智能机器狗。你必须将用户的每一句话解析为对应的工具调用，不能输出任何工具调用之外的自然语言废话。

你拥有以下三个工具：
1. 【nav (导航与寻物)】：负责空间移动、带路、寻找物品的位置。（关键词：去、找、带我到、哪儿）
2. 【vqa (视觉问答)】：负责分析当前视野内的图像，识别眼前物品。（关键词：这是什么、看看、描述眼前）
3. 【chat (闲聊与百科)】：负责日常寒暄、情感交流、回答通用百科知识。（场景：不需要移动、不需要看图的纯对话）

【意图判断与边界划分】（重要！）
- 用户："我的可乐在哪儿？" -> 寻找实体位置，调用 `nav`。
- 用户："我手里这瓶可乐过期了吗？" -> 需要查看眼前画面，调用 `vqa`。
- 用户："可乐是谁发明的？" -> 通用知识百科，调用 `chat`。
- 用户："今天天气真好，带我去阳台。" -> 包含移动意图，调用 `nav`。
- 用户："你好啊，笨笨。" -> 日常打招呼，调用 `chat`。

【分类防错指南】
请小心区分"眼前实体问答"和"通用知识问答"：
- 问："苹果的营养价值是什么？" -> 属于通用百科，调用 `chat`。
- 问："桌子上的苹果红了吗？" -> 针对眼前具体的苹果，调用 `vqa`。
- 问："帮我找个苹果吃。" -> 需要寻找位置，调用 `nav`。
- 问："你叫什么名字？" -> 闲聊，调用 `chat`。
'''


# ═══════════════════════════════════════════════════════════════════════════
# Go2 Camera
# ═══════════════════════════════════════════════════════════════════════════

class Go2Camera:
    """Go2 前置摄像头封装，支持多次拍照复用同一 VideoClient。"""

    def __init__(self, network_interface="eth0"):
        self._network_interface = network_interface
        self._initialized = False
        self._video_client = None
        self._output_path = os.path.expanduser("~/.openclaw/media/go2_camera.jpg")
        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)

    def initialize(self) -> bool:
        if not HAS_CAMERA_SDK:
            logger.error("unitree_sdk2py 未安装，无法使用 Go2 摄像头")
            return False
        if self._initialized:
            return True
        try:
            ChannelFactoryInitialize(0, self._network_interface)
            self._video_client = VideoClient()
            self._video_client.SetTimeout(3.0)
            self._video_client.Init()
            self._initialized = True
            logger.info(f"📷 Go2 摄像头初始化成功 (接口: {self._network_interface})")
            return True
        except Exception as e:
            logger.error(f"📷 Go2 摄像头初始化失败: {e}")
            return False

    def capture(self) -> Optional[str]:
        """拍照并保存到本地，返回图片路径。阻塞调用，需在线程中运行。"""
        if not self._initialized and not self.initialize():
            return None
        if os.path.exists(self._output_path):
            os.remove(self._output_path)
        try:
            code, data = self._video_client.GetImageSample()
            if code != 0:
                logger.error(f"获取图像失败，错误码: {code}")
                return None
            with open(self._output_path, "wb") as f:
                f.write(bytes(data))
            size = os.path.getsize(self._output_path)
            logger.info(f"📷 图像已捕获: {self._output_path} ({size} bytes)")
            return self._output_path
        except Exception as e:
            logger.error(f"📷 拍照异常: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# Enums & Data
# ═══════════════════════════════════════════════════════════════════════════

class ConflictPolicy(enum.Enum):
    INTERRUPT = "interrupt"
    WAIT      = "wait"

class PlaybackSource(enum.Enum):
    NAVIGATION_INTENT = "navigation_intent"
    NAVIGATION_STATUS = "navigation_status"
    VQA_INTENT        = "vqa_intent"
    VQA_RESPONSE      = "vqa_response"

class PreIntent(enum.Enum):
    LIKELY_NAV  = "likely_nav"
    LIKELY_CHAT = "likely_chat"
    LIKELY_VQA  = "likely_vqa"
    UNCERTAIN   = "uncertain"

class GateState(enum.Enum):
    OPEN             = "open"
    PENDING          = "pending"
    DECIDED_PLAY     = "decided_play"
    DECIDED_DISCARD  = "decided_discard"

@dataclass
class PlaybackTask:
    priority: int
    text: str
    source: PlaybackSource
    conflict_policy: ConflictPolicy = ConflictPolicy.WAIT
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    def __lt__(self, other):
        return self.priority < other.priority


# ═══════════════════════════════════════════════════════════════════════════
# AudioGate — 简化状态机
# ═══════════════════════════════════════════════════════════════════════════

class AudioGate:
    DECISION_WINDOW_SEC = 3.0

    def __init__(self):
        self._state = GateState.OPEN
        self._lock = threading.Lock()
        self._pending_start: float = 0

    @property
    def state(self) -> GateState:
        with self._lock:
            return self._state

    @property
    def is_blocking(self) -> bool:
        with self._lock:
            return self._state in (
                GateState.PENDING,
                GateState.DECIDED_DISCARD,
            )

    def on_speech_started(self) -> GateState:
        with self._lock:
            old = self._state
            self._state = GateState.OPEN
            self._pending_start = 0
        if old != GateState.OPEN:
            logger.info(f"🔄 Gate: {old.value} → OPEN (speech started)")
        return GateState.OPEN

    def on_speech_stopped(self) -> tuple[GateState, GateState]:
        with self._lock:
            old = self._state
            if self._state == GateState.OPEN:
                self._state = GateState.PENDING
                self._pending_start = time.time()
                logger.info("⏳ Gate: OPEN → PENDING")
            return old, self._state

    def decide_play(self) -> GateState:
        with self._lock:
            old = self._state
            self._state = GateState.DECIDED_PLAY
            elapsed = ""
            if self._pending_start > 0:
                elapsed = f", 等待 {(time.time()-self._pending_start)*1000:.0f}ms"
            logger.info(f"✅ Gate: {old.value} → DECIDED_PLAY{elapsed}")
            return old

    def decide_discard(self) -> GateState:
        with self._lock:
            old = self._state
            self._state = GateState.DECIDED_DISCARD
            elapsed = ""
            if self._pending_start > 0:
                elapsed = f", 等待 {(time.time()-self._pending_start)*1000:.0f}ms"
            logger.info(f"🗑️ Gate: {old.value} → DECIDED_DISCARD{elapsed}")
            return old

    def reset(self) -> None:
        with self._lock:
            self._state = GateState.OPEN
            self._pending_start = 0

    def get_remaining_budget(self) -> float:
        with self._lock:
            if self._pending_start > 0:
                elapsed = time.time() - self._pending_start
                return max(self.DECISION_WINDOW_SEC - elapsed, 0.1)
            return self.DECISION_WINDOW_SEC


# ═══════════════════════════════════════════════════════════════════════════
# PartialIntentPredictor — partial ASR + LLM (新增 VQA 支持)
# ═══════════════════════════════════════════════════════════════════════════

class PartialIntentPredictor:
    MIN_CHARS = 2
    DEBOUNCE_SEC = 0.15
    PREDICT_TIMEOUT = 2.0
    MIN_DELTA_CHARS = 2

    def __init__(self, llm: QwenLLM):
        self._llm = llm
        self._lock = threading.Lock()
        self._version: int = 0
        self._last_submitted_text: str = ""
        self._debounce_task: Optional[asyncio.Task] = None
        self._predict_task: Optional[asyncio.Task] = None
        self._pre_intent: PreIntent = PreIntent.UNCERTAIN
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._on_nav_detected: Optional[callable] = None
        self._on_chat_detected: Optional[callable] = None
        self._on_vqa_detected: Optional[callable] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_nav_callback(self, cb) -> None:
        self._on_nav_detected = cb

    def set_chat_callback(self, cb) -> None:
        self._on_chat_detected = cb

    def set_vqa_callback(self, cb) -> None:
        self._on_vqa_detected = cb

    @property
    def pre_intent(self) -> PreIntent:
        with self._lock:
            return self._pre_intent

    def reset(self) -> None:
        with self._lock:
            self._version += 1
            self._pre_intent = PreIntent.UNCERTAIN
            self._last_submitted_text = ""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        if self._predict_task and not self._predict_task.done():
            self._predict_task.cancel()

    def on_partial_text(self, accumulated_text: str, gate_state: GateState) -> None:
        text = accumulated_text.strip()
        if len(text) < self.MIN_CHARS:
            return
        if gate_state in (GateState.DECIDED_PLAY, GateState.DECIDED_DISCARD):
            return
        with self._lock:
            if self._pre_intent in (PreIntent.LIKELY_NAV, PreIntent.LIKELY_VQA):
                return
            delta = len(text) - len(self._last_submitted_text)
            has_running = self._predict_task and not self._predict_task.done()
            if self._last_submitted_text and delta < self.MIN_DELTA_CHARS and has_running:
                return

        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._start_debounce(text), self._loop,
            )

    async def _start_debounce(self, text: str) -> None:
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(
            self._debounce_then_predict(text)
        )

    async def _debounce_then_predict(self, text: str) -> None:
        try:
            await asyncio.sleep(self.DEBOUNCE_SEC)
        except asyncio.CancelledError:
            return
        if self._predict_task and not self._predict_task.done():
            self._predict_task.cancel()
        with self._lock:
            version = self._version
            self._last_submitted_text = text
        logger.info(f"🔮 Partial 预判启动: v{version} \"{text}\"")
        self._predict_task = asyncio.create_task(
            self._do_predict(text, version)
        )

    async def _do_predict(self, text: str, version: int) -> None:
        try:
            result_text, tool_call = await asyncio.wait_for(
                call_llm_api(
                    logger=logger, llm=self._llm, user_text=text,
                    conversation_history=[], system_prompt=intent_llm_system_prompt,
                    tools=GO2_TOOLS,
                ),
                timeout=self.PREDICT_TIMEOUT,
            )
            with self._lock:
                if version != self._version:
                    return
            intent = self._parse_result(result_text, tool_call)
            with self._lock:
                old = self._pre_intent
                self._pre_intent = intent
            logger.info(f"🔮 Partial 预判结果: v{version} \"{text}\" → {intent.value}")
            if intent == PreIntent.LIKELY_NAV and self._on_nav_detected:
                self._on_nav_detected()
            elif intent == PreIntent.LIKELY_VQA and self._on_vqa_detected:
                self._on_vqa_detected()
            elif intent == PreIntent.LIKELY_CHAT and self._on_chat_detected:
                self._on_chat_detected()
        except asyncio.TimeoutError:
            logger.warning(f"🔮 预判超时: v{version}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"🔮 预判异常: {e}")

    @staticmethod
    def _parse_result(result_text, tool_call) -> PreIntent:
        if result_text is None and tool_call:
            func_name = tool_call.get("function", {}).get("name", "")
            if func_name == "nav":
                return PreIntent.LIKELY_NAV
            elif func_name == "vqa":
                return PreIntent.LIKELY_VQA
            else:
                return PreIntent.LIKELY_CHAT
        return PreIntent.UNCERTAIN

    def cancel_pending(self) -> None:
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        if self._predict_task and not self._predict_task.done():
            self._predict_task.cancel()

    def stop(self) -> None:
        self.reset()


# ═══════════════════════════════════════════════════════════════════════════
# FastIntentClassifier — 仅用于提取目的地的工具类
# ═══════════════════════════════════════════════════════════════════════════

class FastIntentClassifier:
    @staticmethod
    def _extract_dest(text):
        for p in [
            re.compile(r"(?:导航|带我去?|领我去?)\s*(?:到|去|往)\s*(.{1,20})"),
            re.compile(r"(?:去|到)\s*(.{1,15})"),
        ]:
            m = p.search(text)
            if m:
                d = m.group(1).strip().rstrip("。，.!?吧吗呢啊的")
                if d:
                    return d
        return "目的地"


# ═══════════════════════════════════════════════════════════════════════════
# PlaybackCoordinator
# ═══════════════════════════════════════════════════════════════════════════

class PlaybackCoordinator:
    def __init__(self, tts_client, client=None):
        self.tts_client = tts_client
        self._client = client
        self._queue = asyncio.PriorityQueue()
        self._server_audio_playing = False
        self._server_audio_done = asyncio.Event()
        self._server_audio_done.set()
        self._local_playing = False
        self._cancel_current = asyncio.Event()
        self._task = None
        self._loop = None

    def set_client(self, c):
        self._client = c

    @property
    def is_local_playing(self):
        return self._local_playing

    async def start(self):
        self._loop = asyncio.get_event_loop()
        self._server_audio_done = asyncio.Event()
        self._server_audio_done.set()
        self._cancel_current = asyncio.Event()
        self._queue = asyncio.PriorityQueue()
        self._task = asyncio.create_task(self._worker_loop())
        logger.info("🎛️  PlaybackCoordinator 已启动")

    async def stop(self):
        if self._task and not self._task.done():
            self._cancel_current.set()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def notify_server_audio_started(self):
        self._server_audio_playing = True
        if self._loop:
            self._loop.call_soon_threadsafe(self._server_audio_done.clear)

    def notify_server_audio_done(self):
        self._server_audio_playing = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._server_audio_done.set)

    async def _worker_loop(self):
        while True:
            task = await self._queue.get()
            if not task.text:
                continue
            try:
                await self._execute_task(task)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"播报异常: {e}")

    async def _execute_task(self, task):
        client = self._client
        if not client or not client.running:
            return
        if task.conflict_policy == ConflictPolicy.INTERRUPT:
            if self._server_audio_playing:
                await client.cancel_response()
                await asyncio.sleep(0.05)
        elif task.conflict_policy == ConflictPolicy.WAIT:
            if self._server_audio_playing:
                try:
                    await asyncio.wait_for(self._server_audio_done.wait(), timeout=30)
                    await asyncio.sleep(0.3)
                except asyncio.TimeoutError:
                    await client.cancel_response()
        self._local_playing = True
        self._cancel_current.clear()
        await client.cancel_response()
        await asyncio.sleep(0.05)
        client.mute_microphone()
        try:
            await self._stream_play(client, task.text)
        finally:
            client.unmute_microphone()
            client._cancel_muted = False
            self._local_playing = False
            logger.info("🎛️  播报完成")

    async def _stream_play(self, client, text):
        """TTS 播放：将 PCM 推入 SDK 的 output_queue / 单一 OutputStream，避免 PortAudio ALSA 多流断言崩溃。"""
        tts_sr = self.tts_client.sample_rate
        ws_sr = client.config.sample_rate
        ratio = ws_sr / tts_sr
        total = 0
        try:
            async for chunk in self.tts_client.synthesize_stream(text):
                if self._cancel_current.is_set() or not client.running:
                    break
                total += len(chunk)
                arr = np.frombuffer(chunk, dtype=np.int16)
                if ratio != 1.0:
                    idx = np.arange(0, len(arr), 1.0 / ratio)
                    arr = arr[np.clip(idx.astype(int), 0, len(arr) - 1)]
                pcm_bytes = arr.astype(np.int16).tobytes()
                await client.output_queue.put(pcm_bytes)
            timeout = 30.0
            deadline = time.monotonic() + timeout
            while not client.output_queue.empty() and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
        finally:
            await asyncio.sleep(0.1)
            logger.info(f"🔊 TTS 完成, {total} bytes")


# ═══════════════════════════════════════════════════════════════════════════
# NavStatusPoller
# ═══════════════════════════════════════════════════════════════════════════

class NavStatusPoller:
    def __init__(self, status_file="./nav_status/nav_status.jsonl",
                 poll_interval=2.0, coordinator=None):
        self.status_file = Path(status_file)
        self.poll_interval = poll_interval
        self._coordinator = coordinator
        self._file_offset = 0
        self._processed_ids = set()
        self._task = None

    def set_coordinator(self, c):
        self._coordinator = c

    async def start(self):
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        if self.status_file.exists():
            self._file_offset = self.status_file.stat().st_size
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self):
        while True:
            try:
                await self._check_file()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"轮询异常: {e}")
            await asyncio.sleep(self.poll_interval)

    async def _check_file(self):
        if not self.status_file.exists():
            return
        sz = self.status_file.stat().st_size
        if sz <= self._file_offset:
            return
        async with aiofiles.open(self.status_file, "r", encoding="utf-8") as f:
            await f.seek(self._file_offset)
            content = await f.read()
            self._file_offset = sz
        for line in content.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                logger.info(f"收到导航返回状态{rec}")
            except Exception:
                continue
            rid = rec.get("id", "")
            if rid and rid in self._processed_ids:
                continue
            if rid:
                self._processed_ids.add(rid)
            await self._handle(rec)

    async def _handle(self, rec):
        msg = rec.get("message", "")
        st = rec.get("status", "")
        if not msg or not self._coordinator:
            return
        pol, pri = (
            (ConflictPolicy.INTERRUPT, 1)
            if st in ("turn", "warning", "reroute")
            else (ConflictPolicy.WAIT, 5)
        )
        await self._coordinator._queue.put(PlaybackTask(
            priority=pri, text=msg,
            source=PlaybackSource.NAVIGATION_STATUS,
            conflict_policy=pol, metadata=rec,
        ))


# ═══════════════════════════════════════════════════════════════════════════
# IntentEventHandler — 预判 + VQA pipeline
# ═══════════════════════════════════════════════════════════════════════════

class IntentEventHandler(EventHandler):
    """
    核心决策逻辑 (预判 + VQA):

    speech_started   → OPEN, unmute, reset all
    partial=nav      → cancel_response() + TTS 入队 (立即生效)
    partial=chat     → unmute_speaker() (立即放行)
    partial=vqa      → cancel_response() + TTS "好的，让我看看"
    speech_stopped   → 已决策? 跳过 : mute_speaker() 封堵，并启动 timeout_guard
    transcript_done  →
      已决策=nav      → 写 nav_log, 跳过
      已决策=chat     → 跳过 (已放行)
      已决策=vqa      → 用完整文本启动 VQA pipeline (拍照 → VLM → TTS)
      未决策          → 等待预判完成或超时后再兜底放行
    timeout_guard    → 超时后若仍未决策则 unmute_speaker() 兜底
    """

    def __init__(self, llm=None, coordinator=None, camera=None, vlm=None):
        super().__init__()
        self._client: Optional[AudioWebSocketClient] = None
        self._llm = llm
        self._coordinator = coordinator
        self._camera: Optional[Go2Camera] = camera
        self._vlm: Optional[QwenVLM] = vlm
        self._audio_gate = AudioGate()

        self._already_decided = False
        self._already_cancelled = False
        self._decided_intent: Optional[str] = None  # "nav" / "vqa" / "chat"
        self._partial_nav_dest = ""
        self._pending_final_text: Optional[str] = None
        self._vqa_task: Optional[asyncio.Task] = None

        self._partial_predictor = None
        if llm:
            self._partial_predictor = PartialIntentPredictor(llm)
            self._partial_predictor.set_nav_callback(self._on_partial_nav)
            self._partial_predictor.set_chat_callback(self._on_partial_chat)
            self._partial_predictor.set_vqa_callback(self._on_partial_vqa)

        self._user_text_buffer = ""
        self._buffer_lock = threading.Lock()
        self._server_response_active = False
        self._timeout_task = None

    def set_client(self, client: AudioWebSocketClient):
        self._client = client
        if self._partial_predictor and client and client._loop:
            self._partial_predictor.set_loop(client._loop)

    def set_coordinator(self, c):
        self._coordinator = c

    # ─────────────────────────────────────────
    #  Partial 预判回调
    # ─────────────────────────────────────────

    def _on_partial_nav(self):
        if self._already_decided:
            return

        logger.info("🚨 Partial=NAV → 立即 cancel!")

        self._already_decided = True
        self._already_cancelled = True
        self._decided_intent = "nav"
        self._audio_gate.decide_discard()

        client = self._client
        if not client:
            return

        client.cancel_response()
        logger.info("🚨 cancel_response() 已发送 (partial)")

        with self._buffer_lock:
            current_text = self._user_text_buffer.strip()
        dest = FastIntentClassifier._extract_dest(current_text) if current_text else "目的地"
        self._partial_nav_dest = dest

        if self._pending_final_text:
            self._run_async_nowait(
                self._write_nav_log(self._partial_nav_dest, self._pending_final_text)
            )
            self._pending_final_text = None

        if self._coordinator:
            tts = "好的，正在为您导航，请稍候。"
            self._run_async_nowait(
                self._coordinator._queue.put(PlaybackTask(
                    priority=2, text=tts,
                    source=PlaybackSource.NAVIGATION_INTENT,
                    conflict_policy=ConflictPolicy.INTERRUPT,
                    metadata={"destination": dest, "original_text": current_text,
                              "trigger": "partial"},
                ))
            )
            logger.info(f"🚨 TTS 已入队 (partial): dest=\"{dest}\"")

    def _on_partial_chat(self):
        if self._already_decided:
            return

        logger.info("💬 Partial=CHAT → 立即 unmute 放行!")

        self._already_decided = True
        self._decided_intent = "chat"
        self._audio_gate.decide_play()

        if self._client:
            self._client.unmute_speaker()
            logger.info("🔊 unmute_speaker (partial chat)")

    def _on_partial_vqa(self):
        """
        partial=VQA → 立即 cancel_response + TTS 即时反馈。
        VQA pipeline 在拿到完整文本后启动 (transcript_done 或 _pending_final_text)。
        """
        if self._already_decided:
            return

        logger.info("👁️ Partial=VQA → 立即 cancel!")

        self._already_decided = True
        self._already_cancelled = True
        self._decided_intent = "vqa"
        self._audio_gate.decide_discard()

        client = self._client
        if not client:
            return

        client.cancel_response()
        logger.info("👁️ cancel_response() 已发送 (partial vqa)")

        if self._coordinator:
            self._run_async_nowait(
                self._coordinator._queue.put(PlaybackTask(
                    priority=2, text="好的，让我看看。",
                    source=PlaybackSource.VQA_INTENT,
                    conflict_policy=ConflictPolicy.INTERRUPT,
                ))
            )
            logger.info("👁️ VQA 即时反馈 TTS 已入队")

        if self._pending_final_text:
            self._run_async_nowait(self._start_vqa_pipeline(self._pending_final_text))
            self._pending_final_text = None

    # ─────────────────────────────────────────
    #  VQA Pipeline: 拍照 → VLM → TTS
    # ─────────────────────────────────────────

    async def _start_vqa_pipeline(self, user_text: str):
        if self._vqa_task and not self._vqa_task.done():
            self._vqa_task.cancel()
        self._vqa_task = asyncio.create_task(self._process_vqa(user_text))

    async def _process_vqa(self, user_text: str):
        """完整 VQA 流程：拍照 → 调用 VLM → TTS 播报回答。"""
        try:
            logger.info(f"📷 VQA pipeline 启动, query=\"{user_text}\"")

            image_path = await asyncio.to_thread(self._camera.capture)
            if not image_path:
                logger.error("📷 VQA: 拍照失败")
                if self._coordinator:
                    await self._coordinator._queue.put(PlaybackTask(
                        priority=3, text="抱歉，摄像头拍照失败了，无法查看。",
                        source=PlaybackSource.VQA_RESPONSE,
                        conflict_policy=ConflictPolicy.WAIT,
                    ))
                return

            logger.info(f"🧠 VQA: 调用 VLM, image={image_path}")
            vlm_response = await call_vlm_api(
                logger=logger,
                vlm=self._vlm,
                user_image_path=image_path,
                user_text=user_text,
                conversation_history=[],
            )

            if not vlm_response:
                logger.error("🧠 VQA: VLM 返回为空")
                if self._coordinator:
                    await self._coordinator._queue.put(PlaybackTask(
                        priority=3, text="抱歉，视觉分析失败了，请稍后再试。",
                        source=PlaybackSource.VQA_RESPONSE,
                        conflict_policy=ConflictPolicy.WAIT,
                    ))
                return

            logger.info(f"🗣️ VQA: VLM 回答 ({len(vlm_response)} chars): "
                        f"\"{vlm_response[:100]}{'...' if len(vlm_response)>100 else ''}\"")

            if self._coordinator:
                await self._coordinator._queue.put(PlaybackTask(
                    priority=3, text=vlm_response,
                    source=PlaybackSource.VQA_RESPONSE,
                    conflict_policy=ConflictPolicy.WAIT,
                    metadata={"user_query": user_text, "image_path": image_path},
                ))

        except asyncio.CancelledError:
            logger.info("📷 VQA pipeline 被取消")
        except Exception as e:
            logger.error(f"📷 VQA pipeline 异常: {e}")

    # ─────────────────────────────────────────
    #  桥接方法
    # ─────────────────────────────────────────

    def _run_async(self, coro):
        if self._client and self._client._loop:
            f = asyncio.run_coroutine_threadsafe(coro, self._client._loop)
            try:
                f.result(timeout=30)
            except Exception as e:
                logger.error(f"桥接异常: {e}")

    def _run_async_nowait(self, coro):
        if self._client and self._client._loop:
            asyncio.run_coroutine_threadsafe(coro, self._client._loop)

    # ═════════════════════════════════════════
    # 用户开始说话
    # ═════════════════════════════════════════

    def on_speech_started(self, event: dict):
        logger.info("🎙️ 用户开始说话")

        self._audio_gate.on_speech_started()

        self._already_decided = False
        self._already_cancelled = False
        self._decided_intent = None
        self._partial_nav_dest = ""
        self._pending_final_text = None

        if self._vqa_task and not self._vqa_task.done():
            self._vqa_task.cancel()
            logger.info("👁️ 取消上一轮 VQA pipeline")

        if self._client:
            self._client.unmute_speaker()

        if self._partial_predictor:
            self._partial_predictor.reset()

        with self._buffer_lock:
            self._user_text_buffer = ""

        self._server_response_active = False
        if self._coordinator:
            self._coordinator.notify_server_audio_done()

        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()

    # ═════════════════════════════════════════
    # Partial ASR
    # ═════════════════════════════════════════

    def on_user_transcript(self, text: str):
        with self._buffer_lock:
            self._user_text_buffer = text
            accumulated = self._user_text_buffer

        if self._already_decided:
            return

        if self._partial_predictor:
            self._partial_predictor.on_partial_text(
                accumulated, self._audio_gate.state
            )

        logger.debug(
            f"   partial: \"{accumulated}\" "
            f"(gate={self._audio_gate.state.value}, "
            f"decided={self._already_decided})"
        )

    # ═════════════════════════════════════════
    # speech_stopped — 已决策时跳过 mute
    # ═════════════════════════════════════════

    def on_speech_stopped(self, event: dict):
        logger.info("🎤 用户停止说话")

        if self._already_decided:
            logger.info(f"🎤 已决策 ({self._decided_intent}), 跳过 mute")
            return

        old, new = self._audio_gate.on_speech_stopped()

        if self._client and new == GateState.PENDING:
            self._client.mute_speaker()
            logger.info("🔇 speech_stopped → mute_speaker (等待预判结果)")
            self._run_async_nowait(self._start_timeout_guard())

    # ═════════════════════════════════════════
    # Final ASR
    # ═════════════════════════════════════════

    def on_user_transcript_done(self, event: dict):
        with self._buffer_lock:
            full_text = self._user_text_buffer.strip()
            self._user_text_buffer = ""

        pre_intent = PreIntent.UNCERTAIN
        if self._partial_predictor:
            pre_intent = self._partial_predictor.pre_intent

        logger.info(
            f"📝 Final ASR: \"{full_text}\" | "
            f"pre={pre_intent.value} | "
            f"decided={self._already_decided} | "
            f"intent={self._decided_intent} | "
            f"cancelled={self._already_cancelled}"
        )

        # ═══ 情况 1: 已决策=NAV ═══
        if self._already_cancelled and self._decided_intent == "nav":
            if self._partial_predictor:
                self._partial_predictor.cancel_pending()
            logger.info(
                f"✅ partial=NAV 已处理, 跳过 | dest=\"{self._partial_nav_dest}\""
            )
            if full_text:
                self._run_async_nowait(
                    self._write_nav_log(self._partial_nav_dest, full_text)
                )
            return

        # ═══ 情况 2: 已决策=VQA → 启动 VQA pipeline ═══
        if self._already_cancelled and self._decided_intent == "vqa":
            if self._partial_predictor:
                self._partial_predictor.cancel_pending()
            query = full_text or "请描述你看到的画面"
            logger.info(f"👁️ partial=VQA 已处理, 启动 VQA pipeline | query=\"{query}\"")
            self._run_async_nowait(self._start_vqa_pipeline(query))
            return

        # ═══ 情况 3: 已决策=CHAT ═══
        if self._already_decided and self._decided_intent == "chat":
            if self._partial_predictor:
                self._partial_predictor.cancel_pending()
            logger.info("✅ partial=CHAT 已放行, 跳过")
            return

        # ═══ 情况 4: 预判未决 → 等待预判结果或超时兜底 ═══
        logger.info("⚠️ 预判未决，等待预判结果或超时后再兜底放行（不取消进行中的预判）")
        self._pending_final_text = full_text or None

    # ═════════════════════════════════════════
    # 超时守护
    # ═════════════════════════════════════════

    async def _start_timeout_guard(self):
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
        self._timeout_task = asyncio.create_task(self._timeout_loop())

    async def _timeout_loop(self):
        try:
            await asyncio.sleep(self._audio_gate.get_remaining_budget())
            if not self._already_decided:
                logger.warning("⏰ 预判超时 → 兜底放行")
                self._already_decided = True
                self._decided_intent = "chat"
                self._audio_gate.decide_play()
                if self._client:
                    self._client.unmute_speaker()
                self._pending_final_text = None
        except asyncio.CancelledError:
            pass

    # ═════════════════════════════════════════
    # 服务端音频状态追踪
    # ═════════════════════════════════════════

    def on_audio_playing(self, pcm_bytes: bytes):
        if not self._server_response_active:
            self._server_response_active = True
            if self._coordinator:
                self._coordinator.notify_server_audio_started()

    def on_signal(self, signal_type: str, payload: dict):
        if signal_type in ("response.output_audio.done", "response.done"):
            self._server_response_active = False
            if self._coordinator:
                self._coordinator.notify_server_audio_done()
            if self._audio_gate.state == GateState.DECIDED_DISCARD:
                self._audio_gate.reset()

    # ═════════════════════════════════════════
    # 写导航日志
    # ═════════════════════════════════════════

    async def _write_nav_log(self, dest, text):
        d = Path("./nav_logs")
        d.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(d / "navigation.jsonl", "a", encoding="utf-8") as f:
            await f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "type": "navigation",
                "destination": dest,
                "original_text": text,
            }, ensure_ascii=False) + "\n")

    # ─────────────────────────────────────────
    #  其他事件
    # ─────────────────────────────────────────

    def on_connected(self):
        logger.info("✅ 已连接")
        if self._client and self._client._loop and self._partial_predictor:
            self._partial_predictor.set_loop(self._client._loop)

    def on_ai_transcript_delta(self, text):
        if not self._audio_gate.is_blocking:
            print(text, end="", flush=True)

    def on_ai_transcript_done(self, event):
        if not self._audio_gate.is_blocking:
            print()

    def on_error(self, code, message):
        logger.error(f"❌ [{code}]: {message}")


# ═══════════════════════════════════════════════════════════════════════════
# 启动
# ═══════════════════════════════════════════════════════════════════════════

async def async_main(config: ClientConfig, network_interface: str = "eth0"):
    intent_llm = QwenLLM(logger)
    vlm = QwenVLM(logger)
    tts_client = TencentTTS()

    camera = Go2Camera(network_interface)
    camera.initialize()

    coordinator = PlaybackCoordinator(tts_client=tts_client)
    handler = IntentEventHandler(
        llm=intent_llm, coordinator=coordinator,
        camera=camera, vlm=vlm,
    )
    poller = NavStatusPoller(coordinator=coordinator)

    client = AudioWebSocketClient(config, event_handler=handler)

    handler.set_client(client)
    handler.set_coordinator(coordinator)
    coordinator.set_client(client)
    poller.set_coordinator(coordinator)

    await coordinator.start()
    await poller.start()

    try:
        await client.run_async()
    except asyncio.CancelledError:
        logger.info("🛑 任务被取消")
    finally:
        logger.info("🛑 开始清理...")
        if handler._partial_predictor:
            handler._partial_predictor.stop()
        if handler._vqa_task and not handler._vqa_task.done():
            handler._vqa_task.cancel()
        await poller.stop()
        await coordinator.stop()
        logger.info("✅ 清理完成")


def main():
    """命令行入口."""
    parser = argparse.ArgumentParser(
        description="Audio WebSocket Client for Air Gateway (with VQA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  main-client-wvqa

  # 指定服务器地址和网络接口
  main-client-wvqa --ws-url ws://192.168.1.100:8080/ws --network-interface eth0

  # 使用配置文件
  main-client-wvqa --config config.json
        """,
    )

    parser.add_argument("--ws-url", help="WebSocket 服务器地址")
    parser.add_argument("--user-id", help="用户 ID")
    parser.add_argument("--token", help="认证 Token")
    parser.add_argument("--agent-id", help="Agent ID")
    parser.add_argument("--tone-id", help="音色 ID")
    parser.add_argument("--config", "-c", help="配置文件路径（JSON 格式）")
    parser.add_argument("--sample-rate", type=int, help="采样率（默认 16000）")
    parser.add_argument("--log-level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="日志级别")
    parser.add_argument("--generate-config", metavar="PATH",
                        help="生成示例配置文件到指定路径")
    parser.add_argument("--robot-action", action="store_true",
                        help="启用机器人动作")
    parser.add_argument("--network-interface", default="eth0",
                        help="Go2 摄像头网络接口 (默认 eth0)")

    args = parser.parse_args()

    # --- 1. 生成配置文件模式 ---
    if args.generate_config:
        config = ClientConfig()
        config.save(args.generate_config)
        print(f"✓ 已生成配置文件: {args.generate_config}")
        return

    # --- 2. 加载与覆盖配置 ---
    config = ClientConfig.load(args.config) if args.config else ClientConfig()

    if args.ws_url:
        config.ws_url = args.ws_url
    if args.token:
        config.token = args.token
    if args.agent_id:
        config.agent_id = args.agent_id
    if args.tone_id:
        config.tone_id = args.tone_id
    if args.sample_rate:
        config.sample_rate = args.sample_rate
    if args.robot_action is not None:
        config.robot_action = args.robot_action

    if args.user_id:
        config.user_id = args.user_id
    elif not config.user_id:
        config.user_id = f"python_client_{str(uuid.uuid4())[:8]}"

    # --- 3. 验证与日志设置 ---
    try:
        config.validate()
    except ValueError as e:
        print(f"❌ 配置错误: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 4. 打印启动信息 ---
    print("=" * 50)
    print("Audio WebSocket Client (with VQA)")
    print("=" * 50)
    print(f"  服务器: {config.ws_url}")
    print(f"  用户ID: {config.user_id}")
    print(f"  采样率: {config.sample_rate} Hz")
    print(f"  摄像头: {args.network_interface}")
    print(f"  日志级别: {args.log_level}")
    print("=" * 50)
    print("按 Ctrl+C 退出")

    setup_logging(console_level="INFO", log_file="audio_client.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        asyncio.run(async_main(config, args.network_interface))
    except KeyboardInterrupt:
        print("\n✅ 已退出")


if __name__ == "__main__":
    main()

