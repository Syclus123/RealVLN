"""客户端配置模块."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ClientConfig:
    """音频 WebSocket 客户端配置.
    
    支持从环境变量、配置文件或代码直接配置。
    环境变量优先级最高，其次是配置文件，最后是默认值。
    """
    
    # WebSocket 连接配置
    ws_url: str = "ws://36.151.181.166/realtime"
    user_id: str = "python_client_001"
    token: str = "R5NXpdgOFyAjyclSR6206NqB7z1DOFZr"
    # 业务配置
    subtitle: bool = True
    stream_subtitle: bool = True
    business_id: int = 1
    agent_id: str = "hunyuan-turbos-latest"
    tone_id: str = "607240090"
    scene_type: str = ""
    disable_greeting: bool = False  # 是否禁用问候语
    
    # 音频参数
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16
    frame_duration_ms: int = 20  # 每帧时长（毫秒）
    
    # 心跳配置
    ping_interval_sec: float = 1.0
    
    # 日志配置
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "ClientConfig":
        """从环境变量加载配置.
        
        支持的环境变量：
        - AUDIO_CLIENT_WS_URL
        - AUDIO_CLIENT_USER_ID
        - AUDIO_CLIENT_TOKEN
        - AUDIO_CLIENT_AGENT_ID
        - AUDIO_CLIENT_TONE_ID
        - AUDIO_CLIENT_SCENE_TYPE
        - AUDIO_CLIENT_SUBTITLE
        - AUDIO_CLIENT_BUSINESS_ID
        - AUDIO_CLIENT_SAMPLE_RATE
        - AUDIO_CLIENT_CHANNELS
        - AUDIO_CLIENT_LOG_LEVEL
        - AUDIO_CLIENT_DISABLE_GREETING
        """
        return cls(
            ws_url=os.getenv("AUDIO_CLIENT_WS_URL", cls.ws_url),
            user_id=os.getenv("AUDIO_CLIENT_USER_ID", cls.user_id),
            token=os.getenv("AUDIO_CLIENT_TOKEN", cls.token),
            subtitle=os.getenv("AUDIO_CLIENT_SUBTITLE", "true").lower() == "true",
            business_id=int(os.getenv("AUDIO_CLIENT_BUSINESS_ID", str(cls.business_id))),
            agent_id=os.getenv("AUDIO_CLIENT_AGENT_ID", cls.agent_id),
            tone_id=os.getenv("AUDIO_CLIENT_TONE_ID", cls.tone_id),
            scene_type=os.getenv("AUDIO_CLIENT_SCENE_TYPE", cls.scene_type),
            sample_rate=int(os.getenv("AUDIO_CLIENT_SAMPLE_RATE", str(cls.sample_rate))),
            channels=int(os.getenv("AUDIO_CLIENT_CHANNELS", str(cls.channels))),
            log_level=os.getenv("AUDIO_CLIENT_LOG_LEVEL", cls.log_level),
            disable_greeting=os.getenv("AUDIO_CLIENT_DISABLE_GREETING", "false").lower() == "true",
        )
    
    @classmethod
    def from_file(cls, config_path: str | Path) -> "ClientConfig":
        """从 JSON 配置文件加载配置."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(**data)
    
    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "ClientConfig":
        """智能加载配置.
        
        优先级：环境变量 > 配置文件 > 默认值
        """
        # 先加载默认配置
        config = cls()
        
        # 尝试从配置文件加载
        if config_path:
            try:
                config = cls.from_file(config_path)
            except FileNotFoundError:
                pass
        
        # 环境变量覆盖
        env_config = cls.from_env()
        for field_name in ["ws_url", "user_id", "token", "agent_id", "tone_id", "scene_type",
                           "subtitle", "business_id", "sample_rate", 
                           "channels", "log_level", "disable_greeting"]:
            env_val = getattr(env_config, field_name)
            default_val = getattr(cls, field_name)
            if env_val != default_val:
                setattr(config, field_name, env_val)
        
        return config
    
    def validate(self) -> None:
        """验证配置有效性."""
        if not self.ws_url:
            raise ValueError("ws_url 不能为空")
        if not self.user_id:
            raise ValueError("user_id 不能为空")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate 必须为正整数")
        if self.channels not in (1, 2):
            raise ValueError("channels 仅支持 1 或 2")
        if self.bits_per_sample != 16:
            raise ValueError("当前仅支持 16bit PCM")
        if self.ping_interval_sec <= 0:
            raise ValueError("ping_interval_sec 必须为正数")
    
    def to_dict(self) -> dict:
        """转换为字典."""
        return {
            "ws_url": self.ws_url,
            "user_id": self.user_id,
            "token": self.token,
            "agent_id": self.agent_id,
            "tone_id": self.tone_id,
            "scene_type": self.scene_type,
            "subtitle": self.subtitle,
            "business_id": self.business_id,
            "disable_greeting": self.disable_greeting,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bits_per_sample": self.bits_per_sample,
            "frame_duration_ms": self.frame_duration_ms,
            "ping_interval_sec": self.ping_interval_sec,
            "log_level": self.log_level,
        }
    
    def save(self, config_path: str | Path) -> None:
        """保存配置到文件."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
