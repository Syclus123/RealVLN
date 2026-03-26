"""
多任务意图路由器 — 可扩展的任务系统。

架构:
  ASR → Intent LLM (tool calling) → Intent Router
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              Passthrough          Inline Task         Background Task
              (chat: 服务端        (nav, vqa:          (follow, guard:
               音频直接放行)        本地处理后 TTS)      子进程管理)

任务冲突管理:
  每个任务声明所需资源 (locomotion, camera)。
  启动新任务前，自动取消占用冲突资源的后台任务。
  chat 不占用资源，可与任何后台任务并行。

添加新意图只需:
  1. 在 TASK_REGISTRY 添加 TaskDefinition
  2. 在 MULTI_TASK_TOOLS 添加 LLM tool
  3. 更新 intent_llm_system_prompt
  4. (如果是 inline 任务) 在 IntentEventHandler 注册 handler
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
from typing import Optional, Callable
import uuid
import argparse
import signal
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
from .functions import ACTION_TOOLS

ACTION_FUNC_NAMES = {t["function"]["name"] for t in ACTION_TOOLS}

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


# ═══════════════════════════════════════════════════════════════════════════
# Task System — 任务定义与管理
# ═══════════════════════════════════════════════════════════════════════════

RESOURCE_LOCOMOTION = "locomotion"
RESOURCE_CAMERA = "camera"

# 快速停止关键词: 有后台任务运行时，跳过 LLM 直接触发 stop_task
_STOP_FAST_RE = re.compile(
    r"^[，。,.\s]*(停[止下一]|取消|别[跟追动]|住手|安静|不[要用]了)"
)


def resolve_conda_python(env_name: str) -> str:
    """根据 conda env 名称找到对应的 python 解释器路径。"""
    import subprocess as _sp

    # 1. conda info --base
    try:
        result = _sp.run(
            ["conda", "info", "--base"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            base = result.stdout.strip()
            candidate = os.path.join(base, "envs", env_name, "bin", "python")
            if os.path.isfile(candidate):
                return candidate
    except Exception:
        pass

    # 2. fallback: 常见安装位置
    for prefix in [
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniforge3"),
        "/opt/conda",
        "/usr/local/conda",
    ]:
        candidate = os.path.join(prefix, "envs", env_name, "bin", "python")
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"找不到 conda 环境 '{env_name}' 的 python。"
        f"请确认环境已创建，或改用 python_path 直接指定解释器路径。"
    )


# ── 子进程安全: 父进程死亡时自动终止子进程 (Linux 专用) ──────────────

def _set_pdeathsig():
    """preexec_fn: 通过 prctl(PR_SET_PDEATHSIG) 让内核在父进程死亡时
    自动向子进程发送 SIGTERM，无论父进程是正常退出还是 crash/abort/SIGKILL。"""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG = 1
    except Exception:
        pass


_child_preexec = _set_pdeathsig if sys.platform == "linux" else None


@dataclass
class TaskDefinition:
    """
    任务元数据。添加新意图只需在 TASK_REGISTRY 中增加一条。

    环境配置 (按优先级，互斥使用):
      script_cmd   — 完整命令列表，直接传给 subprocess (最高优先级，向后兼容)
      shell_setup  — 需要 source 脚本或复杂环境变量时，用 shell 包装，通过 exec 保证信号正确传递
      conda_env    — 指定 conda 环境名，自动解析 python 路径
      python_path  — 显式指定 python 解释器路径 (conda/virtualenv/system)
      (都不设)     — 默认使用当前 sys.executable

    脚本配置:
      script_path  — 脚本路径 (搭配上面的环境配置使用)
      script_args  — 脚本额外参数

    子进程配置:
      env_vars     — 子进程额外环境变量 (合并到 os.environ)
      working_dir  — 子进程工作目录
    """
    name: str
    display_name: str
    feedback_text: str
    resources: frozenset[str] = field(default_factory=frozenset)
    is_background: bool = False
    preemptible: bool = True
    # 环境 & 脚本
    script_cmd: Optional[list[str]] = None
    script_path: Optional[str] = None
    script_args: list[str] = field(default_factory=list)
    conda_env: Optional[str] = None
    python_path: Optional[str] = None
    shell_setup: Optional[str] = None
    # 子进程环境
    env_vars: Optional[dict[str, str]] = None
    working_dir: Optional[str] = None


@dataclass
class RunningTask:
    """一个正在运行的后台任务实例。"""
    id: str
    definition: TaskDefinition
    user_text: str
    process: Optional[asyncio.subprocess.Process] = None
    asyncio_task: Optional[asyncio.Task] = None
    started_at: float = field(default_factory=time.time)


class TaskManager:
    """管理后台任务 (子进程) 的生命周期与资源冲突。"""

    def __init__(self, coordinator=None):
        self._coordinator = coordinator
        self._definitions: dict[str, TaskDefinition] = {}
        self._handlers: dict[str, Callable] = {}
        self._running: dict[str, RunningTask] = {}

    def set_coordinator(self, c):
        self._coordinator = c

    def register_task(self, defn: TaskDefinition):
        self._definitions[defn.name] = defn

    def register_handler(self, name: str, handler: Callable):
        """注册 inline 任务的 async handler: handler(user_text) -> None"""
        self._handlers[name] = handler

    def get_definition(self, name: str) -> Optional[TaskDefinition]:
        return self._definitions.get(name)

    def get_running_tasks(self) -> list[RunningTask]:
        return list(self._running.values())

    def find_conflicts(self, defn: TaskDefinition) -> list[RunningTask]:
        """找出与给定任务存在资源冲突的正在运行的后台任务。"""
        if not defn.resources:
            return []
        return [
            t for t in self._running.values()
            if t.definition.resources & defn.resources
        ]

    async def execute(self, intent_name: str, user_text: str) -> Optional[asyncio.Task]:
        """
        执行一个意图任务:
        1. 取消冲突的后台任务
        2. 如果有注册的 inline handler → 运行它
        3. 如果有 script_cmd → 启动子进程
        返回 asyncio.Task (caller 可选择跟踪)。
        """
        defn = self._definitions.get(intent_name)
        if not defn:
            logger.warning(f"未知意图: {intent_name}")
            return None

        conflicts = self.find_conflicts(defn)
        for c in conflicts:
            if c.definition.preemptible:
                logger.info(f"🔄 取消冲突任务: {c.definition.display_name} "
                            f"(资源冲突: {c.definition.resources & defn.resources})")
                await self._cancel_instance(c)

        handler = self._handlers.get(intent_name)
        if handler:
            task = asyncio.create_task(self._run_handler(defn, handler, user_text))
            if defn.is_background:
                task_id = str(uuid.uuid4())[:8]
                rt = RunningTask(
                    id=task_id, definition=defn,
                    user_text=user_text, asyncio_task=task,
                )
                self._running[task_id] = rt
                task.add_done_callback(lambda _, tid=task_id: self._running.pop(tid, None))
            return task

        if defn.script_cmd or defn.script_path:
            return await self._start_script(defn, user_text)

        logger.warning(f"意图 {intent_name} 没有注册 handler 或 script_path/script_cmd")
        return None

    async def _run_handler(self, defn: TaskDefinition, handler: Callable, user_text: str):
        try:
            await handler(user_text)
        except asyncio.CancelledError:
            logger.info(f"任务 {defn.display_name} 被取消")
        except Exception as e:
            logger.error(f"任务 {defn.display_name} 异常: {e}")

    def _resolve_cmd(self, defn: TaskDefinition) -> tuple[list[str], bool]:
        """
        根据 TaskDefinition 的环境配置解析实际命令。
        返回 (cmd_list, use_shell):
          use_shell=True  → cmd_list 是单元素 [shell_string]
          use_shell=False → cmd_list 是 exec 参数列表
        """
        # 最高优先级: 完整自定义命令
        if defn.script_cmd:
            return list(defn.script_cmd), False

        if not defn.script_path:
            raise ValueError(f"任务 {defn.name}: 必须设置 script_cmd 或 script_path")

        args_str = " ".join(defn.script_args) if defn.script_args else ""

        # shell_setup: 需要 source 脚本等复杂操作，通过 exec 确保信号直接送到 python
        if defn.shell_setup:
            python = defn.python_path or "python"
            if defn.conda_env:
                shell_cmd = (
                    f"{defn.shell_setup} && "
                    f"conda run --no-capture-output -n {defn.conda_env} "
                    f"python {defn.script_path} {args_str}"
                )
            else:
                shell_cmd = (
                    f"{defn.shell_setup} && "
                    f"exec {python} {defn.script_path} {args_str}"
                )
            return [shell_cmd], True

        # conda_env: 自动解析 python 路径
        if defn.conda_env:
            try:
                python = resolve_conda_python(defn.conda_env)
            except FileNotFoundError as e:
                logger.error(str(e))
                python = "python"
            return [python, defn.script_path] + list(defn.script_args), False

        # python_path: 显式指定解释器
        if defn.python_path:
            return [defn.python_path, defn.script_path] + list(defn.script_args), False

        # 默认: 当前解释器
        return [sys.executable, defn.script_path] + list(defn.script_args), False

    async def _start_script(self, defn: TaskDefinition, user_text: str) -> asyncio.Task:
        task_id = str(uuid.uuid4())[:8]
        task = asyncio.create_task(self._script_lifecycle(task_id, defn, user_text))
        rt = RunningTask(
            id=task_id, definition=defn,
            user_text=user_text, asyncio_task=task,
        )
        self._running[task_id] = rt
        return task

    async def _script_lifecycle(self, task_id: str, defn: TaskDefinition, user_text: str):
        """启动外部脚本子进程，监控 stdout 协议行:
        STATUS:msg  → TTS 播报
        LOG:LEVEL:msg → 汇入主日志 (保留原始级别)
        其他         → logger.info
        """
        proc = None
        try:
            cmd, use_shell = self._resolve_cmd(defn)

            env = {**os.environ, **(defn.env_vars or {})}
            env["VOICE_AGENT_SUBPROCESS"] = "1"

            cwd = defn.working_dir

            if use_shell:
                proc = await asyncio.create_subprocess_shell(
                    cmd[0],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env, cwd=cwd,
                    preexec_fn=_child_preexec,
                )
                cmd_display = cmd[0][:80]
            else:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env, cwd=cwd,
                    preexec_fn=_child_preexec,
                )
                cmd_display = " ".join(cmd)

            self._running[task_id].process = proc
            logger.info(f"🚀 后台脚本已启动: {defn.display_name} "
                        f"(pid={proc.pid}, cmd={cmd_display})")

            if proc.stdout:
                async for line_bytes in proc.stdout:
                    text = line_bytes.decode(errors="replace").strip()
                    if not text:
                        continue
                    if text.startswith("LOG:"):
                        parts = text.split(":", 2)
                        if len(parts) >= 3:
                            lvl = getattr(logging, parts[1], logging.INFO)
                            logger.log(lvl, f"[{defn.name}] {parts[2]}")
                        else:
                            logger.info(f"[{defn.name}] {text}")
                    elif text.startswith("STATUS:"):
                        status_msg = text[7:].strip()
                        logger.info(f"[{defn.name}] STATUS: {status_msg}")
                        if status_msg and self._coordinator:
                            await self._coordinator._queue.put(PlaybackTask(
                                priority=4, text=status_msg,
                                source=PlaybackSource.TASK_STATUS,
                                conflict_policy=ConflictPolicy.WAIT,
                            ))
                    else:
                        logger.info(f"📋 [{defn.name}] {text}")

            await proc.wait()
            rc = proc.returncode
            logger.info(f"✅ 后台脚本完成: {defn.display_name} (exit={rc})")

        except asyncio.CancelledError:
            if proc and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            logger.info(f"🛑 后台脚本已终止: {defn.display_name}")
        except Exception as e:
            logger.error(f"❌ 后台脚本异常: {defn.display_name}: {e}")
        finally:
            self._running.pop(task_id, None)

    async def _cancel_instance(self, rt: RunningTask):
        if rt.asyncio_task and not rt.asyncio_task.done():
            rt.asyncio_task.cancel()
            try:
                await rt.asyncio_task
            except asyncio.CancelledError:
                pass
        self._running.pop(rt.id, None)

    async def stop_all(self) -> list[str]:
        """停止所有后台任务，返回被停止的任务显示名列表。"""
        stopped = []
        for rt in list(self._running.values()):
            stopped.append(rt.definition.display_name)
            await self._cancel_instance(rt)
        return stopped

    async def cleanup(self):
        await self.stop_all()

    def force_kill_children(self):
        """同步、信号安全: 向所有子进程发送 SIGTERM。
        专为 crash signal handler 设计，不做任何 async 操作。"""
        for rt in list(self._running.values()):
            if rt.process and rt.process.returncode is None:
                try:
                    rt.process.terminate()
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════════════════
# Task Registry — 在此添加新意图
# ═══════════════════════════════════════════════════════════════════════════

TASK_REGISTRY: dict[str, TaskDefinition] = {
    # ── inline 任务 (有注册 handler，不需要 script) ──
    "nav": TaskDefinition(
        name="nav",
        display_name="导航",
        feedback_text="好的，正在为您导航，请稍候。",
        resources=frozenset({RESOURCE_LOCOMOTION}),
    ),
    "vqa": TaskDefinition(
        name="vqa",
        display_name="视觉问答",
        feedback_text="好的，让我看看。",
        resources=frozenset({RESOURCE_CAMERA}),
    ),

    # ── 后台脚本任务 ──
    "follow_person": TaskDefinition(
        name="follow_person",
        display_name="跟随模式",
        feedback_text="好的，开始跟随模式。",
        resources=frozenset({RESOURCE_LOCOMOTION, RESOURCE_CAMERA}),
        is_background=True,
        script_path="/home/unitree/RealVLN/openclaw/skills/go2-follow-person/scripts/follow_fast_yolo_kevin.py",
        python_path="/usr/bin/python3",
        script_args=["eth0"],  # network_interface，会被 async_main 覆盖
    ),

    "guard_dog": TaskDefinition(
        name="guard_dog",
        display_name="看门狗模式",
        feedback_text="好的，进入看门狗模式。",
        resources=frozenset({RESOURCE_LOCOMOTION, RESOURCE_CAMERA}),
        is_background=True,
        script_path="/home/unitree/RealVLN/openclaw/skills/go2-guard-dog/scripts/guard_dog_kevin.py",
        python_path="/usr/bin/python3",
        script_args=["eth0"],  # network_interface，会被 async_main 覆盖
    ),

    # 方式 3: shell_setup — 最灵活，需要 source 脚本/设置复杂环境
    # "ros_patrol": TaskDefinition(
    #     name="ros_patrol",
    #     display_name="ROS 巡逻",
    #     feedback_text="好的，开始巡逻。",
    #     resources=frozenset({RESOURCE_LOCOMOTION}),
    #     is_background=True,
    #     script_path="scripts/ros_patrol.py",
    #     shell_setup="source /opt/ros/humble/setup.bash && source ~/catkin_ws/install/setup.bash",
    #     conda_env="ros_env",
    # ),

    # 方式 4: script_cmd — 完全自定义命令 (向后兼容)
    # "custom_task": TaskDefinition(
    #     name="custom_task",
    #     display_name="自定义任务",
    #     feedback_text="好的，开始执行。",
    #     is_background=True,
    #     script_cmd=["bash", "-c", "conda activate my_env && python scripts/custom.py"],
    # ),

    # 声源定位 — 用系统 python (依赖 pyusb 等系统级包)
    "audio_track": TaskDefinition(
        name="audio_track",
        display_name="声源定位",
        feedback_text="好的，开始声源定位。",
        resources=frozenset({RESOURCE_LOCOMOTION}),
        is_background=True,
        script_path="/home/unitree/RealVLN/usb_4_mic_array/audio_track_kevin.py",
        python_path="/usr/bin/python3",
        script_args=["eth0"],  # network_interface，会被 async_main 覆盖
    ),

    # ── 特殊意图 ──
    "stop_task": TaskDefinition(
        name="stop_task",
        display_name="停止任务",
        feedback_text="好的，马上停止。",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# LLM Tools & Prompt — 在此添加新意图的 tool
# ═══════════════════════════════════════════════════════════════════════════

MULTI_TASK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "chat",
            "description": "日常闲聊、打招呼、或者回答不需要视觉和导航的通用知识问题时调用。",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nav",
            "description": "导航指令，到哪儿去，帮我找个东西，看看某个东西在哪儿，"
                           "当用户要求去某地、找某物、询问某物在哪里时调用。",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vqa",
            "description": "图像问答，看看这个是什么，看看这里有什么，"
                           "当用户要求看看这是什么、描述当前场景、或者询问眼前物品的具体细节时调用。",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "follow_person",
            "description": "跟随用户移动，当用户说跟着我、跟上来、跟我走时调用。",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "guard_dog",
            "description": "进入看门狗/巡逻模式，当用户说看门、守着、巡逻、站岗时调用。",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "audio_track",
            "description": "声源定位，转向声音来源方向，"
                           "当用户说声音从哪来、听听声音在哪、定位声源时调用。",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_task",
            "description": "停止当前正在执行的后台任务（如跟随、看门、声源定位等），"
                           "当用户说别跟了、取消任务时调用。注意：如果没有后台任务运行，"
                           "用户说'停'或'停下来'应调用动作控制的 stop 工具而不是此工具。",
        },
    },
]

MULTI_TASK_TOOLS.extend(ACTION_TOOLS)

intent_llm_system_prompt = '''
你是一个搭载了视觉感知、移动导航、动作控制和智能任务模块的智能机器狗。你必须将用户的每一句话解析为对应的工具调用，不能输出任何工具调用之外的自然语言废话。

你拥有以下工具：
1. 【动作控制工具】：负责机器狗的即时动作控制，包括站立(stand_up)、趴下(stand_down)、平衡站立(balance)、恢复站立(recovery)、阻尼模式(damp)、停止(stop)、向前走(move_forward)、后退(move_backward)、横向移动(move_lateral)、左转(turn_left)、右转(turn_right)、左空翻(left_flip)、后空翻(back_flip)、自由行走(free_walk)、倒立(handstand)、跳跃(free_jump)、蹦跳(free_bound)、自动避障(free_avoid)、直立行走(walk_upright)、交叉步(cross_step)、坐下(sit)、打招呼(hello)、伸展(stretch)。当用户要求机器狗执行具体动作时，直接调用对应的动作工具。
2. 【nav (导航与寻物)】：负责空间移动到具体目的地、带路、寻找物品的位置。（关键词：去某个地方、找某个东西、带我到、哪儿）
3. 【vqa (视觉问答)】：负责分析当前视野内的图像，识别眼前物品。（关键词：这是什么、看看、描述眼前）
4. 【chat (闲聊与百科)】：负责日常寒暄、情感交流、回答通用百科知识。（场景：不需要移动、不需要看图的纯对话）
5. 【follow_person (跟随)】：跟随用户移动。（关键词：跟着我、跟上来、跟我走）
6. 【guard_dog (看门狗)】：进入巡逻看守模式。（关键词：看门、守着、巡逻、站岗）
7. 【audio_track (声源定位)】：定位声音来源方向并转向。（关键词：声音从哪来、听听、定位声源、哪个方向的声音）
8. 【stop_task (停止后台任务)】：停止正在执行的后台任务（如跟随、看门、声源定位）。（关键词：别跟了、取消任务）

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
- 用户："跟着我走。" -> 跟随用户，调用 `follow_person`。
- 用户："帮我看着门。" -> 看守模式，调用 `guard_dog`。
- 用户："声音从哪来的？" -> 声源定位，调用 `audio_track`。
- 用户："别跟了。" -> 停止后台任务，调用 `stop_task`。

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
    NAVIGATION_STATUS = "navigation_status"
    TASK_FEEDBACK     = "task_feedback"
    TASK_RESPONSE     = "task_response"
    TASK_STATUS       = "task_status"

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
# PartialIntentPredictor — 通用化，单回调
# ═══════════════════════════════════════════════════════════════════════════

class PartialIntentPredictor:
    """
    用 partial ASR 文本调用 intent LLM 做提前预判。
    检测到意图后通过单一回调 on_intent_callback(intent_name) 通知上层。
    """
    MIN_CHARS = 2
    DEBOUNCE_SEC = 0.15
    PREDICT_TIMEOUT = 2.0
    MIN_DELTA_CHARS = 2

    PASSTHROUGH_INTENTS = frozenset({"chat"})

    def __init__(self, llm: QwenLLM):
        self._llm = llm
        self._lock = threading.Lock()
        self._version: int = 0
        self._last_submitted_text: str = ""
        self._debounce_task: Optional[asyncio.Task] = None
        self._predict_task: Optional[asyncio.Task] = None
        self._detected_intent: Optional[str] = None
        self._last_tool_call: Optional[dict] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._on_intent_callback: Optional[Callable[[str], None]] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_intent_callback(self, cb: Callable[[str], None]) -> None:
        self._on_intent_callback = cb

    @property
    def detected_intent(self) -> Optional[str]:
        with self._lock:
            return self._detected_intent

    @property
    def last_tool_call(self) -> Optional[dict]:
        with self._lock:
            return self._last_tool_call

    @property
    def is_local_intent(self) -> bool:
        with self._lock:
            return (self._detected_intent is not None
                    and self._detected_intent not in self.PASSTHROUGH_INTENTS)

    def reset(self) -> None:
        with self._lock:
            self._version += 1
            self._detected_intent = None
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
            if self._detected_intent and self._detected_intent not in self.PASSTHROUGH_INTENTS:
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
                    tools=MULTI_TASK_TOOLS,
                ),
                timeout=self.PREDICT_TIMEOUT,
            )
            with self._lock:
                if version != self._version:
                    return
                self._last_tool_call = tool_call
            intent = self._parse_intent(result_text, tool_call)
            if intent:
                with self._lock:
                    self._detected_intent = intent
                logger.info(f"🔮 Partial 预判结果: v{version} \"{text}\" → {intent}")
                if self._on_intent_callback:
                    self._on_intent_callback(intent)
        except asyncio.TimeoutError:
            logger.warning(f"🔮 预判超时: v{version}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"🔮 预判异常: {e}")

    @staticmethod
    def _parse_intent(result_text, tool_call) -> Optional[str]:
        if result_text is None and tool_call:
            return tool_call.get("function", {}).get("name") or None
        return None

    def cancel_pending(self) -> None:
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        if self._predict_task and not self._predict_task.done():
            self._predict_task.cancel()

    def stop(self) -> None:
        self.reset()


# ═══════════════════════════════════════════════════════════════════════════
# FastIntentClassifier — 仅用于提取目的地
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

    def cancel_current_playback(self):
        """中断当前正在播放的本地 TTS 并清空已缓冲的音频。"""
        if not self._local_playing:
            return
        self._cancel_current.set()
        if self._client:
            while not self._client.output_queue.empty():
                try:
                    self._client.output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        logger.info("🎛️  本地 TTS 已被打断")

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
# IntentEventHandler — 通用意图路由
# ═══════════════════════════════════════════════════════════════════════════

class IntentEventHandler(EventHandler):
    """
    通用意图处理器，决策逻辑:

    speech_started   → OPEN, unmute, reset (不影响后台任务)
    partial=chat     → unmute_speaker() 放行
    partial=其他     → cancel_response() + TTS 即时反馈
    speech_stopped   → 已决策? 跳过 : mute_speaker() 封堵 + timeout_guard
    transcript_done  →
      chat           → 跳过 (已放行)
      stop_task      → 跳过 (已即时处理)
      其他 local     → execute via TaskManager (用完整文本)
      未决策         → 等待预判完成或超时兜底
    """

    def __init__(self, llm=None, coordinator=None, task_manager=None,
                 camera=None, vlm=None):
        super().__init__()
        self._client: Optional[AudioWebSocketClient] = None
        self._llm = llm
        self._coordinator = coordinator
        self._task_manager: Optional[TaskManager] = task_manager
        self._camera: Optional[Go2Camera] = camera
        self._vlm: Optional[QwenVLM] = vlm
        self._audio_gate = AudioGate()

        self._already_decided = False
        self._already_cancelled = False
        self._decided_intent: Optional[str] = None
        self._partial_control_func: str = ""
        self._pending_final_text: Optional[str] = None
        self._current_inline_task: Optional[asyncio.Task] = None

        self._partial_predictor = None
        if llm:
            self._partial_predictor = PartialIntentPredictor(llm)
            self._partial_predictor.set_intent_callback(self._on_intent_predicted)

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
    #  统一的预判回调
    # ─────────────────────────────────────────

    def _on_intent_predicted(self, intent_name: str):
        """由 PartialIntentPredictor 回调，intent_name 为 LLM tool call 的 function name。"""
        if self._already_decided:
            return

        if intent_name == "chat":
            self._handle_passthrough()
        elif intent_name == "stop_task":
            self._handle_stop_task()
        elif intent_name in ACTION_FUNC_NAMES:
            self._handle_control_intent(intent_name)
        else:
            self._handle_local_intent(intent_name)

    def _handle_passthrough(self):
        """chat 意图 → 直接放行服务端音频。"""
        logger.info("💬 Partial=CHAT → 立即 unmute 放行!")
        self._already_decided = True
        self._decided_intent = "chat"
        self._audio_gate.decide_play()
        if self._client:
            self._client.unmute_speaker()

    def _handle_stop_task(self):
        """stop_task 意图 → 立即停止所有后台任务（不等 final text）。"""
        logger.info("🛑 Partial=STOP_TASK → 停止后台任务!")
        self._already_decided = True
        self._already_cancelled = True
        self._decided_intent = "stop_task"
        self._audio_gate.decide_discard()

        if self._client:
            self._client.cancel_response()

        # 同步立即发 SIGTERM，不等 event loop 调度
        if self._task_manager:
            self._task_manager.force_kill_children()

        self._run_async_nowait(self._execute_stop_task())

    def _handle_local_intent(self, intent_name: str):
        """
        所有非 chat/stop 的意图走这里:
        1. cancel_response 拦截服务端回复
        2. TTS 即时反馈 (从 TaskDefinition.feedback_text 获取)
        3. 等 final text 到达后执行实际任务
        """
        defn = self._task_manager.get_definition(intent_name) if self._task_manager else None
        display = defn.display_name if defn else intent_name
        logger.info(f"🎯 Partial={intent_name} ({display}) → cancel + TTS 反馈!")

        self._already_decided = True
        self._already_cancelled = True
        self._decided_intent = intent_name
        self._audio_gate.decide_discard()

        if self._client:
            self._client.cancel_response()

        if defn and defn.feedback_text and self._coordinator:
            self._run_async_nowait(
                self._coordinator._queue.put(PlaybackTask(
                    priority=2, text=defn.feedback_text,
                    source=PlaybackSource.TASK_FEEDBACK,
                    conflict_policy=ConflictPolicy.INTERRUPT,
                    metadata={"intent": intent_name, "trigger": "partial"},
                ))
            )

        if self._pending_final_text:
            text = self._pending_final_text
            self._pending_final_text = None
            self._run_async_nowait(self._execute_intent(intent_name, text))

    def _handle_control_intent(self, func_name: str):
        """
        动作控制意图 → 立即 cancel_response + 写 control log + TTS 反馈。
        与 nav 类似：立即止损，将动作函数名写入 navigation.jsonl。
        """
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

        logger.info(f"🎮 Partial=CONTROL → 动作: {func_name}, 参数: {func_args}")

        self._already_decided = True
        self._already_cancelled = True
        self._decided_intent = func_name
        self._partial_control_func = func_name
        self._audio_gate.decide_discard()

        if self._client:
            self._client.cancel_response()

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
                    source=PlaybackSource.TASK_FEEDBACK,
                    conflict_policy=ConflictPolicy.INTERRUPT,
                    metadata={"action": func_name, "original_text": current_text,
                              "trigger": "partial_control"},
                ))
            )
            logger.info(f"🎮 TTS 已入队 (partial control): action=\"{func_name}\"")

    @staticmethod
    def _get_action_description(func_name: str) -> str:
        _ACTION_DESC = {
            "stand_up": "站立", "stand_down": "趴下",
            "balance": "平衡站立", "recovery": "恢复站立",
            "damp": "关节放松", "stop": "停止",
            "move_forward": "向前走", "move_backward": "后退",
            "move_lateral": "横向移动",
            "turn_left": "左转", "turn_right": "右转",
            "left_flip": "左空翻", "back_flip": "后空翻",
            "free_walk": "自由行走", "handstand": "倒立",
            "free_jump": "跳跃", "free_bound": "蹦跳",
            "free_avoid": "自动避障", "walk_upright": "直立行走",
            "cross_step": "交叉步",
            "sit": "坐下", "hello": "打招呼", "stretch": "伸展",
        }
        return _ACTION_DESC.get(func_name, func_name)

    # ─────────────────────────────────────────
    #  意图执行
    # ─────────────────────────────────────────

    async def _execute_intent(self, intent_name: str, user_text: str):
        """拿到完整文本后执行具体任务。"""
        logger.info(f"⚡ 执行意图: {intent_name}, text=\"{user_text}\"")

        if not self._task_manager:
            logger.warning("TaskManager 未设置")
            return

        task = await self._task_manager.execute(intent_name, user_text)

        if task:
            defn = self._task_manager.get_definition(intent_name)
            if defn and not defn.is_background:
                self._current_inline_task = task

    async def _execute_stop_task(self):
        """停止所有后台任务并 TTS 通知用户。"""
        if not self._task_manager:
            return
        stopped = await self._task_manager.stop_all()
        if stopped:
            names = "、".join(stopped)
            text = f"好的，已停止{names}。"
        else:
            text = "当前没有正在执行的后台任务。"
        if self._coordinator:
            await self._coordinator._queue.put(PlaybackTask(
                priority=2, text=text,
                source=PlaybackSource.TASK_FEEDBACK,
                conflict_policy=ConflictPolicy.INTERRUPT,
            ))

    # ─────────────────────────────────────────
    #  Inline 任务 handlers (注册到 TaskManager)
    # ─────────────────────────────────────────

    async def _handle_nav_task(self, user_text: str):
        dest = FastIntentClassifier._extract_dest(user_text) if user_text else "目的地"
        logger.info(f"🧭 导航任务: dest=\"{dest}\", text=\"{user_text}\"")
        await self._write_nav_log(dest, user_text)

    async def _handle_vqa_task(self, user_text: str):
        """拍照 → VLM → TTS 播报。"""
        query = user_text or "请描述你看到的画面"
        logger.info(f"📷 VQA pipeline 启动, query=\"{query}\"")

        if not self._camera:
            logger.error("📷 VQA: 摄像头未初始化")
            return

        image_path = await asyncio.to_thread(self._camera.capture)
        if not image_path:
            logger.error("📷 VQA: 拍照失败")
            if self._coordinator:
                await self._coordinator._queue.put(PlaybackTask(
                    priority=3, text="抱歉，摄像头拍照失败了，无法查看。",
                    source=PlaybackSource.TASK_RESPONSE,
                    conflict_policy=ConflictPolicy.WAIT,
                ))
            return

        if not self._vlm:
            logger.error("🧠 VQA: VLM 未初始化")
            return

        logger.info(f"🧠 VQA: 调用 VLM, image={image_path}")
        vlm_response = await call_vlm_api(
            logger=logger,
            vlm=self._vlm,
            user_image_path=image_path,
            user_text=query,
            conversation_history=[],
        )

        if not vlm_response:
            logger.error("🧠 VQA: VLM 返回为空")
            if self._coordinator:
                await self._coordinator._queue.put(PlaybackTask(
                    priority=3, text="抱歉，视觉分析失败了，请稍后再试。",
                    source=PlaybackSource.TASK_RESPONSE,
                    conflict_policy=ConflictPolicy.WAIT,
                ))
            return

        logger.info(f"🗣️ VQA: VLM 回答 ({len(vlm_response)} chars): "
                    f"\"{vlm_response[:100]}{'...' if len(vlm_response) > 100 else ''}\"")

        if self._coordinator:
            await self._coordinator._queue.put(PlaybackTask(
                priority=3, text=vlm_response,
                source=PlaybackSource.TASK_RESPONSE,
                conflict_policy=ConflictPolicy.WAIT,
                metadata={"user_query": query, "image_path": image_path},
            ))

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
        self._partial_control_func = ""
        self._pending_final_text = None

        # 取消当前 inline 任务 (不影响后台脚本任务)
        if self._current_inline_task and not self._current_inline_task.done():
            self._current_inline_task.cancel()
            logger.info("🔄 取消上一轮 inline 任务")
        self._current_inline_task = None

        # 打断正在播放的本地 TTS (如 VQA 回答)
        if self._coordinator:
            self._coordinator.cancel_current_playback()

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

        # 快速停止: 有后台任务时，关键词匹配直接触发，跳过 LLM
        if (self._task_manager
                and self._task_manager.get_running_tasks()
                and _STOP_FAST_RE.search(accumulated)):
            logger.info(f"⚡ 快速停止关键词命中: \"{accumulated}\"")
            self._on_intent_predicted("stop_task")
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
    # speech_stopped
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

        detected = self._partial_predictor.detected_intent if self._partial_predictor else None

        logger.info(
            f"📝 Final ASR: \"{full_text}\" | "
            f"detected={detected} | "
            f"decided={self._already_decided} | "
            f"intent={self._decided_intent} | "
            f"cancelled={self._already_cancelled}"
        )

        # ═══ 已决策 = chat ═══
        if self._already_decided and self._decided_intent == "chat":
            if self._partial_predictor:
                self._partial_predictor.cancel_pending()
            logger.info("✅ partial=CHAT 已放行, 跳过")
            return

        # ═══ 已决策 = stop_task (已即时执行) ═══
        if self._already_decided and self._decided_intent == "stop_task":
            if self._partial_predictor:
                self._partial_predictor.cancel_pending()
            logger.info("✅ stop_task 已执行, 跳过")
            return

        # ═══ 已决策 = CONTROL (partial 已 cancel + TTS) ═══
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

        # ═══ 已决策 = 其他 local intent → 执行 ═══
        if self._already_cancelled and self._decided_intent:
            if self._partial_predictor:
                self._partial_predictor.cancel_pending()
            text = full_text or ""
            logger.info(f"⚡ 已决策={self._decided_intent}, 执行任务 | text=\"{text}\"")
            self._run_async_nowait(self._execute_intent(self._decided_intent, text))
            return

        # ═══ 预判未决 → 等待预判结果或超时兜底 ═══
        logger.info("⚠️ 预判未决，等待预判结果或超时后再兜底放行")
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

async def async_main(config: ClientConfig, network_interface: str = "eth0"):
    intent_llm = QwenLLM(logger)
    vlm = QwenVLM(logger)
    tts_client = TencentTTS()

    camera = Go2Camera(network_interface)
    camera.initialize()

    coordinator = PlaybackCoordinator(tts_client=tts_client)

    task_manager = TaskManager(coordinator=coordinator)
    for defn in TASK_REGISTRY.values():
        if defn.script_args and defn.script_args == ["eth0"]:
            defn.script_args = [network_interface]
        task_manager.register_task(defn)

    # ── 注册 crash 信号处理: 主进程异常退出时尽力终止子进程 ──
    _prev_handlers: dict[int, object] = {}

    def _crash_handler(signum, frame):
        logger.error(f"收到致命信号 {signum}，正在终止所有子进程...")
        task_manager.force_kill_children()
        # 恢复默认处理并重新发送，让系统产生 core dump 等默认行为
        signal.signal(signum, _prev_handlers.get(signum, signal.SIG_DFL))
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGABRT,):
        _prev_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _crash_handler)

    handler = IntentEventHandler(
        llm=intent_llm, coordinator=coordinator,
        task_manager=task_manager,
        camera=camera, vlm=vlm,
    )

    task_manager.register_handler("nav", handler._handle_nav_task)
    task_manager.register_handler("vqa", handler._handle_vqa_task)

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
        if handler._current_inline_task and not handler._current_inline_task.done():
            handler._current_inline_task.cancel()
        await task_manager.cleanup()
        await poller.stop()
        await coordinator.stop()
        # 恢复默认信号处理
        for sig, prev in _prev_handlers.items():
            signal.signal(sig, prev)
        logger.info("✅ 清理完成")


def main():
    """命令行入口."""
    parser = argparse.ArgumentParser(
        description="Audio WebSocket Client — Multi-Task Intent Router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python -m src.main_client_fast_multi_task

  # 指定服务器地址和网络接口
  python -m src.main_client_fast_multi_task --ws-url ws://localhost:18790/ws --network-interface eth0
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

    if args.generate_config:
        config = ClientConfig()
        config.save(args.generate_config)
        print(f"✓ 已生成配置文件: {args.generate_config}")
        return

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

    try:
        config.validate()
    except ValueError as e:
        print(f"❌ 配置错误: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 50)
    print("Audio WebSocket Client (Multi-Task)")
    print("=" * 50)
    print(f"  服务器: {config.ws_url}")
    print(f"  用户ID: {config.user_id}")
    print(f"  采样率: {config.sample_rate} Hz")
    print(f"  摄像头: {args.network_interface}")
    bg_tasks = [d.display_name for d in TASK_REGISTRY.values() if d.is_background]
    print(f"  后台任务: {', '.join(bg_tasks)}")
    print(f"  日志级别: {args.log_level}")
    print("=" * 50)
    print("按 Ctrl+C 退出")

    setup_logging(console_level="INFO", log_file="audio_client.log")

    try:
        asyncio.run(async_main(config, args.network_interface))
    except KeyboardInterrupt:
        print("\n✅ 已退出")


if __name__ == "__main__":
    main()
