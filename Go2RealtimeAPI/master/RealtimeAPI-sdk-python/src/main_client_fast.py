"""
使用 SDK 原生 mute_speaker/unmute_speaker 实现音频拦截。
TTS 播放复用 SDK 的 output_queue / 单一 OutputStream，避免 PortAudio ALSA 多流断言崩溃。

核心逻辑 (仅预判):
  speech_started   → OPEN, unmute, reset
  partial=nav      → cancel_response() 立即止损 + TTS 入队
  partial=chat     → unmute_speaker() 立即放行
  speech_stopped   → 已决策? 跳过 : mute_speaker() 封堵
  transcript_done  →
    已决策=nav      → 跳过 (TTS 已在队列)
    已决策=chat     → 跳过 (已放行)
    未决策          → 兜底放行 (当作 chat)
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
from .tts_client import TencentTTS
from .functions import GO2_TOOLS, ACTION_TOOLS

ACTION_FUNC_NAMES = {t["function"]["name"] for t in ACTION_TOOLS}

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
你是一个搭载了视觉感知、移动导航和动作控制模块的智能机器狗。你必须将用户的每一句话解析为对应的工具调用，不能输出任何工具调用之外的自然语言废话。

你拥有以下四类工具：
1. 【动作控制工具】：负责机器狗的即时动作控制，包括站立(stand_up)、趴下(stand_down)、平衡站立(balance)、恢复站立(recovery)、阻尼模式(damp)、停止(stop)、向前走(move_forward)、后退(move_backward)、横向移动(move_lateral)、左转(turn_left)、右转(turn_right)、左空翻(left_flip)、后空翻(back_flip)、自由行走(free_walk)、倒立(handstand)、跳跃(free_jump)、蹦跳(free_bound)、自动避障(free_avoid)、直立行走(walk_upright)、交叉步(cross_step)、坐下(sit)、打招呼(hello)、伸展(stretch)。当用户要求机器狗执行具体动作时，直接调用对应的动作工具。
2. 【nav (导航与寻物)】：负责空间移动到具体目的地、带路、寻找物品的位置。（关键词：去某个地方、找某个东西、带我到、哪儿）
3. 【vqa (视觉问答)】：负责分析当前视野内的图像，识别眼前物品。（关键词：这是什么、看看、描述眼前）
4. 【chat (闲聊与百科)】：负责日常寒暄、情感交流、回答通用百科知识。（场景：不需要移动、不需要看图的纯对话）

【动作控制 vs 导航的区别】（非常重要！）
- 动作控制：用户要求的是"立即执行一个动作"，没有具体的目的地。例如"向前走"、"停下来"、"趴下"、"翻个跟头"。
- 导航：用户要求的是"去某个地方"或"找某个东西"，有明确的目标地点或物品。例如"去客厅"、"帮我找水杯"。

【意图判断与边界划分】（重要！）
- 用户："向前走" -> 即时动作，调用 `move_forward`。
- 用户："停下来" -> 即时动作，调用 `stop`。
- 用户："趴下" -> 即时动作，调用 `stand_down`。
- 用户："翻个跟头" -> 即时动作，调用 `back_flip`。
- 用户："左转" -> 即时动作，调用 `turn_left`。
- 用户："站起来" -> 即时动作，调用 `stand_up`。
- 用户："跳一个" -> 即时动作，调用 `free_jump`。
- 用户："我的可乐在哪儿？" -> 寻找实体位置，调用 `nav`。
- 用户："我手里这瓶可乐过期了吗？" -> 需要查看眼前画面，调用 `vqa`。
- 用户："可乐是谁发明的？" -> 通用知识百科，调用 `chat`。
- 用户："今天天气真好，带我去阳台。" -> 去具体目的地，调用 `nav`。
- 用户："你好啊，笨笨。" -> 日常打招呼，调用 `chat`。

【分类防错指南】
请小心区分"动作控制"和"导航"：
- 问："向前走" -> 没有目的地，属于即时动作控制，调用 `move_forward`。
- 问："往前走到客厅" -> 有目的地"客厅"，属于导航，调用 `nav`。
- 问："停" -> 即时动作控制，调用 `stop`。
- 问："帮我找个苹果吃。" -> 需要寻找位置，调用 `nav`。

请小心区分"眼前实体问答"和"通用知识问答"：
- 问："苹果的营养价值是什么？" -> 属于通用百科，调用 `chat`。
- 问："桌子上的苹果红了吗？" -> 针对眼前具体的苹果，调用 `vqa`。
- 问："你叫什么名字？" -> 闲聊，调用 `chat`。
'''


# ═══════════════════════════════════════════════════════════════════════════
# Enums & Data
# ═══════════════════════════════════════════════════════════════════════════

class ConflictPolicy(enum.Enum):
    INTERRUPT = "interrupt"
    WAIT      = "wait"

class PlaybackSource(enum.Enum):
    NAVIGATION_INTENT = "navigation_intent"
    NAVIGATION_STATUS = "navigation_status"

class PreIntent(enum.Enum):
    LIKELY_NAV     = "likely_nav"
    LIKELY_CHAT    = "likely_chat"
    LIKELY_CONTROL = "likely_control"
    UNCERTAIN      = "uncertain"

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
# PartialIntentPredictor — partial ASR + LLM
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
        self._last_tool_call: Optional[dict] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._on_nav_detected: Optional[callable] = None
        self._on_chat_detected: Optional[callable] = None
        self._on_control_detected: Optional[callable] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_nav_callback(self, cb) -> None:
        self._on_nav_detected = cb

    def set_chat_callback(self, cb) -> None:
        self._on_chat_detected = cb

    def set_control_callback(self, cb) -> None:
        self._on_control_detected = cb

    @property
    def pre_intent(self) -> PreIntent:
        with self._lock:
            return self._pre_intent

    @property
    def last_tool_call(self) -> Optional[dict]:
        with self._lock:
            return self._last_tool_call

    def reset(self) -> None:
        with self._lock:
            self._version += 1
            self._pre_intent = PreIntent.UNCERTAIN
            self._last_tool_call = None
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
            if self._pre_intent in (PreIntent.LIKELY_NAV, PreIntent.LIKELY_CONTROL):
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
                self._last_tool_call = tool_call
            intent = self._parse_result(result_text, tool_call)
            with self._lock:
                old = self._pre_intent
                self._pre_intent = intent
            logger.info(f"🔮 Partial 预判结果: v{version} \"{text}\" → {intent.value}")
            if intent == PreIntent.LIKELY_CONTROL and self._on_control_detected:
                self._on_control_detected()
            elif intent == PreIntent.LIKELY_NAV and self._on_nav_detected:
                self._on_nav_detected()
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
            if func_name in ACTION_FUNC_NAMES:
                return PreIntent.LIKELY_CONTROL
            if func_name == "nav":
                return PreIntent.LIKELY_NAV
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
        """TTS 播放：将 PCM 推入 SDK 的 output_queue，由同一 OutputStream 播放，避免多流触发 PortAudio ALSA 断言 (PaAlsaStreamComponent_BeginPolling)."""
        tts_sr = self.tts_client.sample_rate
        ws_sr = client.config.sample_rate  # SDK 播放循环期望的采样率
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
                # 推入 SDK 的 output_queue，格式为 ws_sr 单声道 int16（与服务器下行一致）
                pcm_bytes = arr.astype(np.int16).tobytes()
                await client.output_queue.put(pcm_bytes)
            # 等待队列被播放循环消费完（避免提前结束导致"播放完成"误判）
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
# IntentEventHandler — 仅预判，不做终判
# ═══════════════════════════════════════════════════════════════════════════

class IntentEventHandler(EventHandler):
    """
    核心决策逻辑 (仅预判):

    speech_started   → OPEN, unmute, reset all
    partial=nav      → cancel_response() + TTS 入队 (立即生效)
    partial=chat     → unmute_speaker() (立即放行)
    speech_stopped   → 已决策? 跳过 : mute_speaker() 封堵，并启动 timeout_guard
    transcript_done  →
      已决策=nav      → 跳过 (TTS 已在队列)
      已决策=chat     → 跳过 (已放行)
      未决策          → 不立即兜底；等待进行中的预判完成，或由 timeout_guard 超时后兜底放行
    timeout_guard    → 超时后若仍未决策则 unmute_speaker() 兜底
    """

    def __init__(self, llm=None, coordinator=None):
        super().__init__()
        self._client: Optional[AudioWebSocketClient] = None
        self._llm = llm
        self._coordinator = coordinator
        self._audio_gate = AudioGate()

        # 决策标记
        self._already_decided = False       # 预判已做出决策 (nav 或 chat)
        self._already_cancelled = False     # 已发送 cancel_response (nav)
        self._partial_nav_dest = ""
        # 预判未决时暂存 transcript_done 的 full_text，供延迟出 nav 时写 nav_log
        self._pending_final_text: Optional[str] = None

        self._partial_control_func = ""

        self._partial_predictor = None
        if llm:
            self._partial_predictor = PartialIntentPredictor(llm)
            self._partial_predictor.set_nav_callback(self._on_partial_nav)
            self._partial_predictor.set_chat_callback(self._on_partial_chat)
            self._partial_predictor.set_control_callback(self._on_partial_control)

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
        """
        partial=NAV → 立即 cancel_response + TTS 入队

        不等 final 确认, 直接止损.
        风险: ~5% 误判, 用户需重说一次.
        收益: 导航响应快 ~800ms, 节省服务端算力.
        """
        if self._already_decided:
            return

        logger.info("🚨 Partial=NAV → 立即 cancel!")

        self._already_decided = True
        self._already_cancelled = True
        self._audio_gate.decide_discard()

        client = self._client
        if not client:
            return

        client.cancel_response_sync()
        logger.info("🚨 cancel_response() 已发送 (partial)")

        with self._buffer_lock:
            current_text = self._user_text_buffer.strip()
        dest = FastIntentClassifier._extract_dest(current_text) if current_text else "目的地"
        self._partial_nav_dest = dest

        # 若 transcript_done 已先到（预判未决后延迟出 nav），用暂存的 full_text 写 nav_log，避免漏写
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
        """
        partial=CHAT → 立即 unmute, 放行服务端音频.

        不等 final 确认, 直接放行.
        """
        if self._already_decided:
            return

        logger.info("💬 Partial=CHAT → 立即 unmute 放行!")

        self._already_decided = True
        self._audio_gate.decide_play()

        if self._client:
            self._client.unmute_speaker()
            logger.info("🔊 unmute_speaker (partial chat)")

    def _on_partial_control(self):
        """
        partial=CONTROL → 立即 cancel_response + 写 control log + TTS 反馈.

        与 nav 类似：立即止损，将动作函数名写入 navigation.jsonl。
        """
        if self._already_decided:
            return

        tool_call = self._partial_predictor.last_tool_call if self._partial_predictor else None
        func_name = ""
        func_args = {}
        if tool_call:
            func_name = tool_call.get("function", {}).get("name", "")
            args_str = tool_call.get("function", {}).get("arguments", "")
            if args_str and isinstance(args_str, str):
                try:
                    func_args = json.loads(args_str)
                except Exception:
                    func_args = {}
            elif isinstance(args_str, dict):
                func_args = args_str

        if not func_name:
            return

        logger.info(f"🎮 Partial=CONTROL → 动作: {func_name}, 参数: {func_args}")

        self._already_decided = True
        self._already_cancelled = True
        self._partial_control_func = func_name
        self._audio_gate.decide_discard()

        client = self._client
        if not client:
            return

        client.cancel_response_sync()
        logger.info("🎮 cancel_response() 已发送 (partial control)")

        with self._buffer_lock:
            current_text = self._user_text_buffer.strip()

        if self._pending_final_text:
            self._run_async_nowait(
                self._write_control_log(func_name, self._pending_final_text, func_args)
            )
            self._pending_final_text = None

        if self._coordinator:
            action_desc = self._get_action_description(func_name)
            tts = f"好的，正在执行{action_desc}。"
            self._run_async_nowait(
                self._coordinator._queue.put(PlaybackTask(
                    priority=2, text=tts,
                    source=PlaybackSource.NAVIGATION_INTENT,
                    conflict_policy=ConflictPolicy.INTERRUPT,
                    metadata={"action": func_name, "original_text": current_text,
                              "trigger": "partial_control"},
                ))
            )
            logger.info(f"🎮 TTS 已入队 (partial control): action=\"{func_name}\"")

    @staticmethod
    def _get_action_description(func_name: str) -> str:
        _ACTION_DESC = {
            "stand_up": "站立", "stand_down": "趴下", "balance": "平衡站立",
            "recovery": "恢复站立", "damp": "关节放松", "stop": "停止",
            "move_forward": "向前走", "move_backward": "后退",
            "move_lateral": "横向移动",
            "turn_left": "左转", "turn_right": "右转",
            "left_flip": "左空翻", "back_flip": "后空翻",
            "free_walk": "自由行走", "handstand": "倒立",
            "free_jump": "跳跃", "free_bound": "蹦跳",
            "free_avoid": "自动避障", "walk_upright": "直立行走",
            "cross_step": "交叉步", "sit": "坐下",
            "hello": "打招呼", "stretch": "伸展",
        }
        return _ACTION_DESC.get(func_name, func_name)

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

        # 重置所有决策标记
        self._already_decided = False
        self._already_cancelled = False
        self._partial_nav_dest = ""
        self._partial_control_func = ""
        self._pending_final_text = None

        # 确保恢复扬声器
        if self._client:
            self._client.unmute_speaker()

        # 重置 predictor
        if self._partial_predictor:
            self._partial_predictor.reset()

        # 清空缓冲
        with self._buffer_lock:
            self._user_text_buffer = ""

        # 重置服务端状态
        self._server_response_active = False
        if self._coordinator:
            self._coordinator.notify_server_audio_done()

        # 取消上一轮超时任务
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()

    # ═════════════════════════════════════════
    # Partial ASR
    # ═════════════════════════════════════════

    def on_user_transcript(self, text: str):
        with self._buffer_lock:
            self._user_text_buffer = text
            accumulated = self._user_text_buffer

        # 如果已经决策, 不需要再预判
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
        """
        VAD 检测到用户停止说话.

        已决策 → 不需要 mute (nav 已 cancel, chat 已 unmute).
        未决策 → mute_speaker 封堵漏音窗口, 等预判结果或兜底.
        """
        logger.info("🎤 用户停止说话")

        if self._already_decided:
            logger.info(f"🎤 已决策, 跳过 mute (cancelled={self._already_cancelled})")
            return

        old, new = self._audio_gate.on_speech_stopped()

        if self._client and new == GateState.PENDING:
            self._client.mute_speaker()
            logger.info("🔇 speech_stopped → mute_speaker (等待预判结果)")
            # 启动超时守护：若在决策窗口内预判未出结果，由超时后兜底放行
            self._run_async_nowait(self._start_timeout_guard())

    # ═════════════════════════════════════════
    # Final ASR — 不再调用 LLM
    # ═════════════════════════════════════════

    def on_user_transcript_done(self, event: dict):
        """
        Final ASR 到达, 三种情况:

        1. _already_decided=True, cancelled=True  → nav 已处理, 跳过
        2. _already_decided=True, cancelled=False → chat 已放行, 跳过
        3. _already_decided=False                 → 预判未决: 不立即兜底，等待预判完成或超时后再兜底放行
        """
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
            f"cancelled={self._already_cancelled}"
        )

        # ═══ 情况 1a: 已决策=CONTROL (partial 已 cancel + TTS) ═══
        if self._already_cancelled and self._partial_control_func:
            if self._partial_predictor:
                self._partial_predictor.cancel_pending()
            logger.info(
                f"✅ partial=CONTROL 已处理, 跳过 | action=\"{self._partial_control_func}\""
            )
            if full_text:
                tool_call = self._partial_predictor.last_tool_call if self._partial_predictor else None
                func_args = {}
                if tool_call:
                    args_str = tool_call.get("function", {}).get("arguments", "")
                    if args_str and isinstance(args_str, str):
                        try:
                            func_args = json.loads(args_str)
                        except Exception:
                            func_args = {}
                    elif isinstance(args_str, dict):
                        func_args = args_str
                self._run_async_nowait(
                    self._write_control_log(self._partial_control_func, full_text, func_args)
                )
            return

        # ═══ 情况 1b: 已决策=NAV (partial 已 cancel + TTS) ═══
        if self._already_cancelled:
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

        # ═══ 情况 2: 已决策=CHAT (partial 已 unmute) ═══
        if self._already_decided:
            if self._partial_predictor:
                self._partial_predictor.cancel_pending()
            logger.info("✅ partial=CHAT 已放行, 跳过")
            return

        # ═══ 情况 3: 预判未决 → 先等预判完成，不取消 in-flight 预测；超时由 timeout_guard 兜底放行 ═══
        logger.info("⚠️ 预判未决，等待预判结果或超时后再兜底放行（不取消进行中的预判）")
        # 暂存 full_text：若之后 predictor 延迟返回 nav，_on_partial_nav 可据此写 nav_log
        self._pending_final_text = full_text or None
        # 不 cancel_pending()，让可能仍在跑的 _do_predict 出结果；若超时则由 speech_stopped 时启动的 timeout_guard 兜底

    # ═════════════════════════════════════════
    # 超时守护
    # ═════════════════════════════════════════

    async def _start_timeout_guard(self):
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
        self._timeout_task = asyncio.create_task(self._timeout_loop())

    async def _timeout_loop(self):
        """预判超时 → 兜底放行"""
        try:
            await asyncio.sleep(self._audio_gate.get_remaining_budget())
            if not self._already_decided:
                logger.warning("⏰ 预判超时 → 兜底放行")
                self._already_decided = True
                self._audio_gate.decide_play()
                if self._client:
                    self._client.unmute_speaker()
                # 兜底当作 chat，清除未决时暂存的 full_text，避免被误当 nav 写日志
                self._pending_final_text = None
        except asyncio.CancelledError:
            pass

    # ═════════════════════════════════════════
    # 服务端音频状态追踪
    # ═════════════════════════════════════════

    def on_audio_playing(self, pcm_bytes: bytes):
        """纯追踪, 不做拦截 (拦截由 mute_speaker 完成)."""
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

    async def _write_control_log(self, func_name, text, func_args=None):
        d = Path("./nav_logs")
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "type": "control",
            "destination": func_name,
            "original_text": func_name,
        }
        if func_args:
            record["parameters"] = func_args
        async with aiofiles.open(d / "navigation.jsonl", "a", encoding="utf-8") as f:
            await f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"📝 Control log 已写入: {func_name}")

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

async def async_main(config: ClientConfig):
    intent_llm = QwenLLM(logger)
    tts_client = TencentTTS()

    coordinator = PlaybackCoordinator(tts_client=tts_client)
    handler = IntentEventHandler(llm=intent_llm, coordinator=coordinator)
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
        await poller.stop()
        await coordinator.stop()
        logger.info("✅ 清理完成")


def main():
    """命令行入口."""
    parser = argparse.ArgumentParser(
        description="Audio WebSocket Client for Air Gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  main-client

  # 指定服务器地址
  main-client --ws-url ws://192.168.1.100:8080/ws

  # 使用配置文件
  main-client --config config.json
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
    print("Audio WebSocket Client")
    print("=" * 50)
    print(f"  服务器: {config.ws_url}")
    print(f"  用户ID: {config.user_id}")
    print(f"  采样率: {config.sample_rate} Hz")
    print(f"  日志级别: {args.log_level}")
    print("=" * 50)
    print("按 Ctrl+C 退出")

    setup_logging(console_level="INFO", log_file="audio_client.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        asyncio.run(async_main(config))
    except KeyboardInterrupt:
        print("\n✅ 已退出")


if __name__ == "__main__":
    main()
