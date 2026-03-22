"""语音通话 —— 异步调用示例.

使用场景：
    当程序本身已经运行在 asyncio 事件循环中时使用，例如：
    - 集成到 FastAPI / aiohttp 等异步 Web 框架
    - 需要和其他异步任务并发运行（如同时跑多个客户端）
    - 在已有的 asyncio.run() 中组合多个协程

使用方式：
    python example_async.py
"""

import asyncio
import logging

from audio_client import AudioWebSocketClient, ClientConfig, EventHandler, setup_logging


# ---------------------------------------------------------------------------
# 用户日志配置 —— 独立于 SDK 内部日志，专门记录用户回调信息
# ---------------------------------------------------------------------------

def _setup_user_logger() -> logging.Logger:
    """配置用户日志器，输出到 user_async.log 文件."""
    logger = logging.getLogger("user_async")
    logger.setLevel(logging.DEBUG)
    # 避免重复添加 handler
    if not logger.handlers:
        file_handler = logging.FileHandler("user_async.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


user_logger = _setup_user_logger()


# ---------------------------------------------------------------------------
# 自定义事件处理器 —— 和同步示例完全一样，按需重写回调
# ---------------------------------------------------------------------------

class VoiceChatHandler(EventHandler):
    """语音通话事件处理器."""

    def __init__(self):
        self._ai_text_buffer = ""
        self.client: "AudioWebSocketClient | None" = None
        self._cancel_task: asyncio.Task | None = None
        self._audio_count = 0  # 记录收到的音频帧数，方便日志观察

    def on_connected(self) -> None:
        msg = "✅ 已连接到服务器"
        print(msg)
        user_logger.info(msg)

    def on_session_created(self, session: dict) -> None:
        msg = "✅ 会话已创建"
        print(msg)
        user_logger.info(msg)
        user_logger.debug("session_created payload: %s", session)
        self.client.mute_microphone()

    def on_session_updated(self, session: dict) -> None:
        msg = "✅ 会话配置已更新，可以开始说话了"
        print(msg)
        print()
        user_logger.info(msg)
        user_logger.debug("session_updated payload: %s", session)
        # 会话就绪后，立即发送一个文本查询
        # 注意：回调在线程池中执行，不能使用 asyncio.get_event_loop()，
        # 应使用 SDK 提供的线程安全同步方法 send_text_query_sync()
        if self.client:
            query_msg = "📝 正在发送文本查询: 讲个故事"
            print(query_msg)
            user_logger.info(query_msg)
            self.client.send_text_query_sync("讲个故事")

    def on_speech_started(self, event: dict) -> None:
        msg = "🎤 [检测到说话...]"
        print(msg)
        user_logger.info(msg)
        # speech_started 会解除静音，重置状态以便下一轮回复重新计时
        self._audio_count = 0
        # 取消还未触发的定时器（如果有）
        if self._cancel_task and not self._cancel_task.done():
            self._cancel_task.cancel()
        self._cancel_task = None

    def on_speech_stopped(self, event: dict) -> None:
        msg = "🎤 [说话结束]"
        print(msg)
        user_logger.info(msg)

    def on_ai_transcript_delta(self, text: str) -> None:
        self._ai_text_buffer += text
        display = f"🤖 AI: {self._ai_text_buffer}"
        print(f"\r{display}", end="", flush=True)
        user_logger.info("AI 增量文本: %s", text)

    def on_ai_transcript_done(self, event: dict) -> None:
        print()
        msg = f"🤖 AI 完整回复: {self._ai_text_buffer}"
        user_logger.info(msg)
        self._ai_text_buffer = ""

    def on_audio_playing(self, pcm_bytes: bytes) -> None:
        """收到下行音频：仅在每轮第一帧时启动 2 秒 cancel 定时器."""
        self._audio_count += 1
        if self._cancel_task is None or self._cancel_task.done():
            # 本轮还没启动过定时器，启动一个
            msg = f"🔊 收到第 {self._audio_count} 帧音频，启动 2s cancel 定时器"
            print(msg)
            user_logger.info(msg)
            # loop = asyncio.get_event_loop()
            # self._cancel_task = loop.create_task(self._do_cancel())
        else:
            # 定时器已在运行，只计数
            msg = f"🔊 收到第 {self._audio_count} 帧音频"
            print(msg)
            user_logger.debug(msg)

    async def _do_cancel(self) -> None:
        """从第一帧音频开始等 2 秒，然后主动取消 AI 响应."""
        try:
            await asyncio.sleep(2)
            if self.client:
                print(f"⏱️  收到首帧音频 2 秒已到（共收到 {self._audio_count} 帧），正在发送 cancel_response...")
                await self.client.cancel_response()
                print("✅ cancel_response 已发送，进入静音模式，等待下次 speech_started 恢复")
        except asyncio.CancelledError:
            # 被 speech_started 取消了，正常行为
            pass

    def on_query_completed(self, item_id: str) -> None:
        msg = f"✅ 服务端已确认收到文本查询, item_id={item_id}"
        print(msg)
        user_logger.info(msg)

    def on_user_transcript(self, text: str) -> None:
        msg = f"👤 用户: {text}"
        print(msg)
        user_logger.info(msg)

    def on_error(self, code: str, message: str) -> None:
        msg = f"❌ 错误 [{code}]: {message}"
        print(msg)
        user_logger.error(msg)

    def on_disconnected(self, reason: str) -> None:
        msg = f"🔌 已断开连接: {reason}"
        print(f"\n{msg}")
        user_logger.info(msg)


# ---------------------------------------------------------------------------
# 异步主函数
# ---------------------------------------------------------------------------

async def main():
    # 配置日志（可选）
    setup_logging(console_level="INFO", log_file="audio_client.log")

    config = ClientConfig(
        ws_url="ws://183.47.116.239:13000/realtime",
        token="R5NXpdgOFyAjyclSR6206NqB7z1DOFZr",
        user_id="python_client_004",
        agent_id="hunyuan-turbos-latest",
        tone_id="chenwenqingnian",
        # scene_type="hunyuan-o",
        disable_greeting=True,
        subtitle=True,
        business_id=8,
        sample_rate=16000,
    )

    print("=" * 50)
    print("🎙️  语音通话客户端（异步模式 - Cancel 测试）")
    print("=" * 50)
    print(f"  服务器:  {config.ws_url}")
    print(f"  用户ID:  {config.user_id}")
    print(f"  Agent:   {config.agent_id}")
    print(f"  采样率:  {config.sample_rate} Hz")
    print("=" * 50)
    print("按 Ctrl+C 退出")
    print()

    # 先创建 handler，再创建 client，然后将 client 引用赋给 handler
    handler = VoiceChatHandler()
    client = AudioWebSocketClient(config, event_handler=handler)
    handler.client = client

    await client.run_async()

    print("✅ 客户端已退出")


if __name__ == "__main__":
    # asyncio.run() 会创建事件循环并运行 main() 协程
    asyncio.run(main())
