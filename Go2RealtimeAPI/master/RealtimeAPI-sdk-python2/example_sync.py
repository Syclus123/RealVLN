"""语音通话示例.

直接运行即可开始语音通话：
    python example.py
"""

from audio_client import AudioWebSocketClient, ClientConfig, EventHandler, setup_logging


# ---------------------------------------------------------------------------
# 自定义事件处理器 —— 根据需要重写感兴趣的回调方法
# ---------------------------------------------------------------------------

class VoiceChatHandler(EventHandler):
    """语音通话事件处理器."""

    def __init__(self):
        self._ai_text_buffer = ""  # 用于累积当前轮 AI 回复的完整文本

    def on_connected(self) -> None:
        print("✅ 已连接到服务器")

    def on_session_created(self, session: dict) -> None:
        print("✅ 会话已创建")

    def on_session_updated(self, session: dict) -> None:
        print("✅ 会话配置已更新，可以开始说话了")
        print()

    def on_speech_started(self, event: dict) -> None:
        print("🎤 [检测到说话...]")

    def on_speech_stopped(self, event: dict) -> None:
        print("🎤 [说话结束]")

    def on_ai_transcript_delta(self, text: str) -> None:
        """AI 回复增量输出."""
        self._ai_text_buffer += text
        # 实时输出到同一行
        print(f"\r🤖 AI: {self._ai_text_buffer}", end="", flush=True)

    def on_ai_transcript_done(self, event: dict) -> None:
        """AI 回复结束."""
        print()  # 换行
        self._ai_text_buffer = ""  # 重置缓冲区，准备下一轮

    def on_user_transcript(self, text: str) -> None:
        """用户语音识别结果."""
        print(f"👤 用户: {text}")

    def on_error(self, code: str, message: str) -> None:
        print(f"❌ 错误 [{code}]: {message}")

    def on_disconnected(self, reason: str) -> None:
        print(f"\n🔌 已断开连接: {reason}")


# ---------------------------------------------------------------------------
# 配置 & 启动
# ---------------------------------------------------------------------------

def main():
    # 配置日志（可选，注释掉则不输出 SDK 内部日志）
    setup_logging(console_level="INFO", log_file="audio_client.log")

    # 直接在代码中配置参数
    config = ClientConfig(
        ws_url="ws://183.47.116.239:13000/realtime",
        token="R5NXpdgOFyAjyclSR6206NqB7z1DOFZr",
        user_id="python_client_001",
        agent_id="hunyuan-turbos-latest",
        tone_id="607240090",
        scene_type="hunyuan-o",
        subtitle=True,
        stream_subtitle=True,
        business_id=1,
        disable_greeting=True, # 关闭打招呼
        sample_rate=16000,
    )

    # 打印配置摘要
    print("=" * 50)
    print("🎙️  语音通话客户端")
    print("=" * 50)
    print(f"  服务器:  {config.ws_url}")
    print(f"  用户ID:  {config.user_id}")
    print(f"  Agent:   {config.agent_id}")
    print(f"  采样率:  {config.sample_rate} Hz")
    print("=" * 50)
    print("按 Ctrl+C 退出")
    print()

    # 创建客户端并启动（run() 是同步阻塞方法，内部会管理事件循环）
    client = AudioWebSocketClient(config, event_handler=VoiceChatHandler())
    client.run()

    print("✅ 客户端已退出")


if __name__ == "__main__":
    main()
