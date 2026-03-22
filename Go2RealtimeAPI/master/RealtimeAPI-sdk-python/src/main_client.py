"""main_client.py

使用 SDK 原生 mute_speaker/unmute_speaker 实现音频拦截。

核心逻辑:
  speech_started   → OPEN, unmute, reset
  partial=nav      → cancel_response() 立即止损 + TTS 入队
  partial=chat     → 不操作, 等 final
  speech_stopped   → mute_speaker() 立即封堵
  transcript_done  →
    已 cancel      → 跳过分类, 可选确认日志
    未 cancel      → final LLM
      chat         → unmute_speaker()
      nav          → cancel_response() + TTS
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
import sounddevice as sd

from audio_client import (
    AudioWebSocketClient,
    ClientConfig,
    EventHandler,
    setup_logging,
)
from audio_client.device import (
    AUDIO_OUTPUT_DEVICE_ID,
    AUDIO_OUTPUT_DEVICE_CHANNALS,
    AUDIO_OUTPUT_SAMPLE_RATE,
)

from .intentllm import QwenLLM, call_llm_api
from .tts_client import TencentTTS
from .functions import GO2_TOOLS

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


intent_llm_system_prompt='''
你是一个搭载了视觉感知和移动导航模块的智能机器狗。你必须将用户的每一句话解析为对应的工具调用，不能输出任何工具调用之外的自然语言废话。

你拥有以下三个工具：
1. 【nav (导航与寻物)】：负责空间移动、带路、寻找物品的位置。（关键词：去、找、带我到、哪儿）
2. 【vqa (视觉问答)】：负责分析当前视野内的图像，识别眼前物品。（关键词：这是什么、看看、描述眼前）
3. 【chat (闲聊与百科)】：负责日常寒暄、情感交流、回答通用百科知识。（场景：不需要移动、不需要看图的纯对话）

【意图判断与边界划分】（重要！）
- 用户：“我的可乐在哪儿？” -> 寻找实体位置，调用 `nav`。
- 用户：“我手里这瓶可乐过期了吗？” -> 需要查看眼前画面，调用 `vqa`。
- 用户：“可乐是谁发明的？” -> 通用知识百科，调用 `chat`。
- 用户：“今天天气真好，带我去阳台。” -> 包含移动意图，调用 `nav`。
- 用户：“你好啊，笨笨。” -> 日常打招呼，调用 `chat`。

【分类防错指南】
请小心区分“眼前实体问答”和“通用知识问答”：
- 问：“苹果的营养价值是什么？” -> 属于通用百科，调用 `chat`。
- 问：“桌子上的苹果红了吗？” -> 针对眼前具体的苹果，调用 `vqa`。
- 问：“帮我找个苹果吃。” -> 需要寻找位置，调用 `nav`。
- 问：“你叫什么名字？” -> 闲聊，调用 `chat`。
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
    LIKELY_NAV  = "likely_nav"
    LIKELY_CHAT = "likely_chat"
    UNCERTAIN   = "uncertain"

# ★ 改动: 去掉 SHIELDING, 简化为 4 态
class GateState(enum.Enum):
    OPEN             = "open"              # 正常播放
    PENDING          = "pending"           # 已 mute, 等分类
    DECIDED_PLAY     = "decided_play"      # 放行
    DECIDED_DISCARD  = "decided_discard"   # 丢弃

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
    """
    简化状态机, 无 side-effect.

    状态转换:
      OPEN ──speech_stopped──→ PENDING
      OPEN ──partial_nav────→ DECIDED_DISCARD  (提前 cancel)
      PENDING ──final=chat──→ DECIDED_PLAY
      PENDING ──final=nav───→ DECIDED_DISCARD
      DECIDED_* ──reset─────→ OPEN
    """

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
            # 如果已经是 DECIDED_DISCARD (partial cancel), 保持不变
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
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._on_nav_detected: Optional[callable] = None
        self._on_chat_detected: Optional[callable] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_nav_callback(self, cb) -> None:
        self._on_nav_detected = cb

    def set_chat_callback(self, cb) -> None:
        self._on_chat_detected = cb

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
        # ★ 改动: 去掉 SHIELDING, 只在 DECIDED 状态时跳过
        if gate_state in (GateState.DECIDED_PLAY, GateState.DECIDED_DISCARD):
            return
        with self._lock:
            if self._pre_intent == PreIntent.LIKELY_NAV:
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
            # TODO add more functions
            return PreIntent.LIKELY_NAV if func_name == "nav" else PreIntent.LIKELY_CHAT
        return PreIntent.UNCERTAIN

    def cancel_pending(self) -> None:
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        if self._predict_task and not self._predict_task.done():
            self._predict_task.cancel()

    def stop(self) -> None:
        self.reset()


# ═══════════════════════════════════════════════════════════════════════════
# FastIntentClassifier — final 分类
# ═══════════════════════════════════════════════════════════════════════════

class FastIntentClassifier:
    def __init__(self, llm: QwenLLM):
        self._llm = llm

    async def classify(self, text, pre_intent, timeout=2.0):
        try:
            result_text, tool_call = await asyncio.wait_for(
                call_llm_api(
                    logger=logger, llm=self._llm, user_text=text,
                    conversation_history=[], system_prompt=intent_llm_system_prompt,
                    tools=GO2_TOOLS,
                ),
                timeout=timeout,
            )
            return self._parse(result_text, tool_call, text)
        except asyncio.TimeoutError:
            if pre_intent == PreIntent.LIKELY_NAV:
                return "navigation", self._extract_dest(text), None
            return "chat", "", None
        except Exception as e:
            logger.error(f"分类异常: {e}")
            return "chat", "", None

    @staticmethod
    def _parse(result_text, tool_call, text):
        if result_text is None and tool_call:
            fn = tool_call.get("function", {}).get("name", "")
            args = tool_call.get("function", {}).get("arguments", {})
            if isinstance(args, str):
                try: args = json.loads(args)
                except: args = {}
            
            # TODO add more functions
            if fn == "nav":
                return "navigation", args.get("destination", FastIntentClassifier._extract_dest(text)), tool_call
            else:
                return "chat", "", None
            # return "unknown_tool", "", tool_call
        return "chat", "", None

    @staticmethod
    def _extract_dest(text):
        for p in [
            re.compile(r"(?:导航|带我去?|领我去?)\s*(?:到|去|往)\s*(.{1,20})"),
            re.compile(r"(?:去|到)\s*(.{1,15})"),
        ]:
            m = p.search(text)
            if m:
                d = m.group(1).strip().rstrip("。，.!?吧吗呢啊的")
                if d: return d
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

    def set_client(self, c): self._client = c
    @property
    def is_local_playing(self): return self._local_playing

    async def start(self):
        self._loop = asyncio.get_event_loop()
        self._server_audio_done = asyncio.Event(); self._server_audio_done.set()
        self._cancel_current = asyncio.Event()
        self._queue = asyncio.PriorityQueue()
        self._task = asyncio.create_task(self._worker_loop())
        logger.info("🎛️  PlaybackCoordinator 已启动")

    async def stop(self):
        if self._task and not self._task.done():
            self._cancel_current.set(); self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass

    def notify_server_audio_started(self):
        self._server_audio_playing = True
        if self._loop: self._loop.call_soon_threadsafe(self._server_audio_done.clear)

    def notify_server_audio_done(self):
        self._server_audio_playing = False
        if self._loop: self._loop.call_soon_threadsafe(self._server_audio_done.set)

    async def _worker_loop(self):
        while True:
            task = await self._queue.get()
            if not task.text: continue
            try: await self._execute_task(task)
            except asyncio.CancelledError: raise
            except Exception as e: logger.error(f"播报异常: {e}")

    async def _execute_task(self, task):
        client = self._client
        if not client or not client.running: return
        if task.conflict_policy == ConflictPolicy.INTERRUPT:
            if self._server_audio_playing:
                await client.cancel_response(); await asyncio.sleep(0.05)
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
        try: await self._stream_play(client, task.text)
        finally:
            client.unmute_microphone()
            client._cancel_muted = False
            self._local_playing = False
            logger.info("🎛️  播报完成")

    async def _stream_play(self, client, text):
        tts_sr = self.tts_client.sample_rate
        out_sr = AUDIO_OUTPUT_SAMPLE_RATE
        ratio = out_sr / tts_sr
        stream = sd.OutputStream(device=AUDIO_OUTPUT_DEVICE_ID, samplerate=out_sr,
            channels=AUDIO_OUTPUT_DEVICE_CHANNALS, dtype="int16", blocksize=int(out_sr*0.02))
        stream.start(); loop = asyncio.get_event_loop(); total = 0
        try:
            async for chunk in self.tts_client.synthesize_stream(text):
                if self._cancel_current.is_set() or not client.running: break
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


# ═══════════════════════════════════════════════════════════════════════════
# NavStatusPoller
# ═══════════════════════════════════════════════════════════════════════════

class NavStatusPoller:
    def __init__(self, status_file="./nav_status/nav_status.jsonl", poll_interval=2.0, coordinator=None):
        self.status_file = Path(status_file); self.poll_interval = poll_interval
        self._coordinator = coordinator; self._file_offset = 0
        self._processed_ids = set(); self._task = None
    def set_coordinator(self, c): self._coordinator = c
    async def start(self):
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        if self.status_file.exists(): self._file_offset = self.status_file.stat().st_size
        self._task = asyncio.create_task(self._poll_loop())
    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
    async def _poll_loop(self):
        while True:
            try: await self._check_file()
            except asyncio.CancelledError: raise
            except Exception as e: logger.error(f"轮询异常: {e}")
            await asyncio.sleep(self.poll_interval)
    async def _check_file(self):
        if not self.status_file.exists(): return
        sz = self.status_file.stat().st_size
        if sz <= self._file_offset: return
        async with aiofiles.open(self.status_file, "r", encoding="utf-8") as f:
            await f.seek(self._file_offset); content = await f.read(); self._file_offset = sz
        for line in content.strip().splitlines():
            line = line.strip()
            if not line: continue
            try: rec = json.loads(line); logger.info(f"收到导航返回状态{rec}")
            except: continue
            rid = rec.get("id","")
            if rid and rid in self._processed_ids: continue
            if rid: self._processed_ids.add(rid)
            await self._handle(rec)
    async def _handle(self, rec):
        msg = rec.get("message",""); st = rec.get("status","")
        if not msg or not self._coordinator: return
        pol, pri = (ConflictPolicy.INTERRUPT, 1) if st in ("turn","warning","reroute") else (ConflictPolicy.WAIT, 5)
        await self._coordinator._queue.put(PlaybackTask(priority=pri, text=msg, source=PlaybackSource.NAVIGATION_STATUS, conflict_policy=pol, metadata=rec))


# ═══════════════════════════════════════════════════════════════════════════
# IntentEventHandler
# ═══════════════════════════════════════════════════════════════════════════

class IntentEventHandler(EventHandler):
    """
    ★ 核心决策逻辑:

    speech_started   → OPEN, unmute, reset all
    partial=nav      → cancel_response() 立即止损 + TTS 入队
    partial=chat     → 不操作
    speech_stopped   → mute_speaker() 立即封堵漏音窗口
    transcript_done  →
      已 cancel      → 跳过分类 (TTS 已在队列)
      未 cancel      → final LLM
        chat         → unmute_speaker()
        nav          → cancel_response() + TTS
    timeout_guard    → unmute_speaker() 兜底
    """

    def __init__(self, llm=None, coordinator=None):
        super().__init__()
        self._client: Optional[AudioWebSocketClient] = None
        self._llm = llm
        self._coordinator = coordinator
        self._audio_gate = AudioGate()

        # ★ 新增: 本轮是否已经 cancel 过
        self._already_cancelled = False
        self._partial_nav_dest = ""

        self._partial_predictor = None
        if llm:
            self._partial_predictor = PartialIntentPredictor(llm)
            self._partial_predictor.set_nav_callback(self._on_partial_nav)
            self._partial_predictor.set_chat_callback(self._on_partial_chat)

        self._fast_classifier = FastIntentClassifier(llm) if llm else None

        self._user_text_buffer = ""
        self._buffer_lock = threading.Lock()
        self._server_response_active = False
        self._timeout_task = None
        self._classify_task = None

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
        ★ 改动: partial LLM=nav → 立即 cancel_response + TTS 入队

        不等 final 确认, 直接止损.
        风险: ~5% 误判, 用户需重说一次.
        收益: 导航响应快 ~800ms, 节省服务端算力.
        """
        if self._already_cancelled:
            return  # 幂等

        logger.info("🚨 Partial=NAV → 立即 cancel!")

        # 标记
        self._already_cancelled = True
        self._audio_gate.decide_discard()

        client = self._client
        if not client:
            return

        # 立即 cancel
        client.cancel_response()
        logger.info("🚨 cancel_response() 已发送 (partial)")

        # 提取目的地 (从当前 buffer)
        with self._buffer_lock:
            current_text = self._user_text_buffer.strip()
        dest = FastIntentClassifier._extract_dest(current_text) if current_text else "目的地"
        self._partial_nav_dest = dest

        # TTS 入队
        if self._coordinator:
            tts = (
                f"好的，正在为您导航，请稍候。"
            )
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
        """partial LLM=chat → 什么都不做, 等 final."""
        logger.info("🔮 Partial=CHAT → 不操作, 等 final")

    # ─────────────────────────────────────────
    #  桥接方法
    # ─────────────────────────────────────────

    def _run_async(self, coro):
        if self._client and self._client._loop:
            f = asyncio.run_coroutine_threadsafe(coro, self._client._loop)
            try: f.result(timeout=30)
            except Exception as e: logger.error(f"桥接异常: {e}")

    def _run_async_nowait(self, coro):
        if self._client and self._client._loop:
            asyncio.run_coroutine_threadsafe(coro, self._client._loop)

    # ═════════════════════════════════════════
    # 用户开始说话
    # ═════════════════════════════════════════

    def on_speech_started(self, event: dict):
        logger.info("🎙️ 用户开始说话")

        # 重置状态机
        self._audio_gate.on_speech_started()

        # ★ 重置 cancel 标记
        self._already_cancelled = False
        self._partial_nav_dest = ""

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

        # 取消上一轮任务
        if self._classify_task and not self._classify_task.done():
            self._classify_task.cancel()
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()

    # ═════════════════════════════════════════
    # Partial ASR
    # ═════════════════════════════════════════

    def on_user_transcript(self, text: str):
        with self._buffer_lock:
            self._user_text_buffer = text 
            accumulated = self._user_text_buffer

        # 如果已经 cancel, 不需要再预判
        if self._already_cancelled:
            return

        if self._partial_predictor:
            self._partial_predictor.on_partial_text(
                accumulated, self._audio_gate.state
            )

        logger.debug(
            f"   partial: \"{accumulated}\" "
            f"(gate={self._audio_gate.state.value}, "
            f"cancelled={self._already_cancelled}, "
            f"spk_muted={self._client.is_speaker_muted if self._client else '?'})"
        )

    # ═════════════════════════════════════════
    # ★ 新增: speech_stopped — 立即 mute
    # ═════════════════════════════════════════

    def on_speech_stopped(self, event: dict):
        """
        VAD 检测到用户停止说话 → 立即 mute_speaker.

        为什么不等 transcript_done?
          speech_stopped 之后, 服务端并行:
          1. ASR 识别 → transcript_done (~300ms)
          2. AI 生成  → audio.delta    (~100ms)
          audio.delta 可能先到 → 如果 speaker 没 mute → 漏音

        如果已经 cancel (partial=nav):
          cancel 已经阻止了服务端生成, 无需 mute.
        """
        logger.info("🎤 用户停止说话")

        # 如果已经 cancel, 不需要额外操作
        if self._already_cancelled:
            logger.info("🎤 已 cancel, 跳过 mute")
            return

        # gate → PENDING
        old, new = self._audio_gate.on_speech_stopped()

        # 立即 mute
        if self._client and new == GateState.PENDING:
            self._client.mute_speaker()
            logger.info("🔇 speech_stopped → mute_speaker (封堵漏音窗口)")

    # ═════════════════════════════════════════
    # Final ASR
    # ═════════════════════════════════════════

    def on_user_transcript_done(self, event: dict):
        """
        Final ASR 到达 → 两个分支:

        分支 A: _already_cancelled = True (partial=nav 已触发)
          → 跳过分类, TTS 已在队列
          → 可选: 异步确认日志

        分支 B: _already_cancelled = False
          → 启动 final LLM 分类
          → chat → unmute
          → nav  → cancel + TTS
        """
        with self._buffer_lock:
            full_text = self._user_text_buffer.strip()
            self._user_text_buffer = ""

        # 停止 predictor
        if self._partial_predictor:
            self._partial_predictor.cancel_pending()

        if not full_text:
            # 空文本 → 恢复播放
            if not self._already_cancelled:
                self._audio_gate.decide_play()
                if self._client:
                    self._client.unmute_speaker()
            return

        pre_intent = PreIntent.UNCERTAIN
        if self._partial_predictor:
            pre_intent = self._partial_predictor.pre_intent

        logger.info(
            f"📝 Final ASR: \"{full_text}\" | "
            f"pre={pre_intent.value} | "
            f"cancelled={self._already_cancelled} | "
            f"gate={self._audio_gate.state.value}"
        )

        # ════════════════════════════════════
        # 分支 A: 已经 cancel (partial=nav)
        # ════════════════════════════════════
        if self._already_cancelled:
            logger.info(
                f"✅ 已在 partial 阶段 cancel, 跳过分类 | "
                f"dest=\"{self._partial_nav_dest}\""
            )
            # 可选: 用 final 文本做确认 (仅记日志, 不影响决策)
            self._run_async_nowait(self._confirm_and_log(full_text))
            return

        # ════════════════════════════════════
        # 分支 B: 未 cancel → 启动分类
        # ════════════════════════════════════

        # 兜底: 确保 speaker 已 muted
        #   正常情况 speech_stopped 已经 muted
        #   但某些 SDK 可能不触发 speech_stopped 就直接来 transcript_done
        if self._client and not self._client.is_speaker_muted:
            self._client.mute_speaker()
            logger.info("🔇 兜底 mute (speech_stopped 可能未触发)")

        # 确保 gate 在 PENDING
        if self._audio_gate.state == GateState.OPEN:
            self._audio_gate.on_speech_stopped()

        # 启动超时守护
        self._run_async_nowait(self._start_timeout_guard())

        # 启动分类
        self._run_async_nowait(
            self._classify_and_decide(full_text, pre_intent)
        )

    # ═════════════════════════════════════════
    # 异步: 确认日志 (partial cancel 后)
    # ═════════════════════════════════════════

    async def _confirm_and_log(self, full_text: str):
        """
        partial 已经 cancel 后, 用 final 文本确认一下.
        仅记日志, 不影响决策.
        如果不一致, 记 warning (用于调优 prompt).
        """
        if not self._fast_classifier:
            return
        try:
            intent, dest, tool_call = await self._fast_classifier.classify(
                full_text, PreIntent.LIKELY_NAV, timeout=3.0,
            )
            if intent == "navigation":
                logger.info(
                    f"🔍 确认分类: NAV ✅ (与 partial 一致) | "
                    f"\"{full_text}\" dest=\"{dest}\""
                )
                # 如果 final 提取到更好的目的地, 更新日志
                if dest and dest != "目的地" and dest != self._partial_nav_dest:
                    logger.info(f"🔍 目的地更新: \"{self._partial_nav_dest}\" → \"{dest}\"")
                    await self._write_nav_log(dest, full_text)
            else:
                logger.warning(
                    f"🔍 确认分类: {intent} ⚠️ "
                    f"(与 partial=NAV 不一致!) | "
                    f"\"{full_text}\""
                )
                # TODO: 上报不一致事件, 用于调优 prompt
        except Exception as e:
            logger.warning(f"🔍 确认分类失败: {e}")

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
            state = self._audio_gate.state
            if state == GateState.PENDING:
                logger.warning(f"⏰ 超时! gate={state.value} → 放行")
                self._audio_gate.decide_play()
                if self._client:
                    self._client.unmute_speaker()
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
    # 分类 + 决策
    # ═════════════════════════════════════════

    async def _classify_and_decide(self, full_text, pre_intent):
        client = self._client
        if not client:
            self._audio_gate.decide_play()
            return

        # 如果本地正在播报, 放行 (不打断自己的 TTS)
        if self._coordinator and self._coordinator.is_local_playing:
            self._audio_gate.decide_play()
            client.unmute_speaker()
            return

        gate = self._audio_gate.state
        if gate != GateState.PENDING:
            logger.info(f"⚠️ 分类启动时 gate={gate.value}, 跳过")
            return

        intent, dest, tool_call = "chat", "", None

        if self._fast_classifier:
            try:
                budget = self._audio_gate.get_remaining_budget()
                logger.info(
                    f"🔍 Final 分类 (budget={budget:.2f}s, "
                    f"pre={pre_intent.value}, gate={gate.value})"
                )
                intent, dest, tool_call = await self._fast_classifier.classify(
                    full_text, pre_intent, timeout=budget,
                )
                logger.info(f"🏷️ 分类结果: intent={intent}, dest=\"{dest}\"")
            except asyncio.CancelledError:
                # 被取消 (用户重新说话) → 恢复
                self._audio_gate.reset()
                client.unmute_speaker()
                return
            except Exception as e:
                logger.error(f"分类异常: {e}")

        # 取消超时守护
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()

        # 执行决策
        if intent == "navigation":
            await self._act_navigation(client, full_text, dest)
        else:
            # chat / unknown_tool / 异常 → 放行
            await self._act_chat(client)

    async def _act_navigation(self, client, full_text, dest):
        """final=nav → cancel + TTS"""
        logger.info(f"🧭 Final=NAV → cancel + TTS | dest=\"{dest}\"")

        self._already_cancelled = True
        self._audio_gate.decide_discard()

        # 打断服务端
        try:
            await client.cancel_response()
            await asyncio.sleep(0.05)
        except Exception as e:
            logger.warning(f"打断失败: {e}")

        # 恢复扬声器 (给 TTS 用)
        client.unmute_speaker()

        self._server_response_active = False
        if self._coordinator:
            self._coordinator.notify_server_audio_done()

        # 记录
        await self._write_nav_log(dest, full_text)

        # TTS
        if self._coordinator:
            tts = (
                f"好的，正在为您导航到{dest}，请稍候。"
                if dest and dest != "目的地"
                else "好的，正在为您规划路线，请稍候。"
            )
            await self._coordinator._queue.put(PlaybackTask(
                priority=2, text=tts,
                source=PlaybackSource.NAVIGATION_INTENT,
                conflict_policy=ConflictPolicy.INTERRUPT,
                metadata={"destination": dest, "original_text": full_text,
                          "trigger": "final"},
            ))

        self._audio_gate.reset()

    async def _act_chat(self, client):
        """final=chat → unmute, 放行 queue 中积累的音频"""
        self._audio_gate.decide_play()
        client.unmute_speaker()
        logger.info("💬 Final=CHAT → unmute_speaker (放行)")

    async def _write_nav_log(self, dest, text):
        d = Path("./nav_logs"); d.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(d/"navigation.jsonl", "a", encoding="utf-8") as f:
            await f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "type": "navigation", "destination": dest, "original_text": text,
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
    
    # 添加命令行参数
    parser.add_argument("--ws-url", help="WebSocket 服务器地址")
    parser.add_argument("--user-id", help="用户 ID")
    parser.add_argument("--token", help="认证 Token")
    parser.add_argument("--agent-id", help="Agent ID")
    parser.add_argument("--tone-id", help="音色 ID")
    parser.add_argument("--config", "-c", help="配置文件路径（JSON 格式）")
    parser.add_argument("--sample-rate", type=int, help="采样率（默认 16000）")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="日志级别")
    parser.add_argument("--generate-config", metavar="PATH", help="生成示例配置文件到指定路径")
    parser.add_argument("--robot-action", action="store_true", help="启用机器人动作")
    
    args = parser.parse_args()
    
    # --- 1. 生成配置文件模式 ---
    if args.generate_config:
        config = ClientConfig()
        config.save(args.generate_config)
        print(f"✓ 已生成配置文件: {args.generate_config}")
        return
    
    # --- 2. 加载与覆盖配置 ---
    # 先从文件加载（如果提供），否则使用默认
    config = ClientConfig.load(args.config) if args.config else ClientConfig()
    
    # 命令行参数优先级最高，覆盖已有配置
    if args.ws_url: config.ws_url = args.ws_url
    if args.token: config.token = args.token
    if args.agent_id: config.agent_id = args.agent_id
    if args.tone_id: config.tone_id = args.tone_id
    if args.sample_rate: config.sample_rate = args.sample_rate
    if args.robot_action is not None: config.robot_action = args.robot_action
    
    # 特殊处理 user_id: 如果没传，生成一个 UUID
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
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    try:
        asyncio.run(async_main(config))
    except KeyboardInterrupt:
        print("\n✅ 已退出")

if __name__ == "__main__":
    main()