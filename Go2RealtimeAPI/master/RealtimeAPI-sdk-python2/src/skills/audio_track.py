import os
import time
import math
import logging
import signal
import sys
import usb.core
import usb.util
from tuning import Tuning
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

logger = logging.getLogger("audio_track")


class _PipeLogHandler(logging.Handler):
    """子进程 → 父进程日志协议: 通过 stdout 输出 LOG:LEVEL:message"""
    def emit(self, record):
        try:
            print(f"LOG:{record.levelname}:{self.format(record)}", flush=True)
        except Exception:
            pass

_shutdown = False


def _on_signal(signum, frame):
    global _shutdown
    _shutdown = True


def _sleep(duration):
    """可被 _shutdown 标志中断的 sleep，最大响应延迟 ~20ms。"""
    end = time.time() + duration
    while not _shutdown and time.time() < end:
        time.sleep(min(0.02, end - time.time()))


def main():
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    network_interface = sys.argv[1] if len(sys.argv) > 1 else "eth0"

    ChannelFactoryInitialize(0, network_interface)
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if not dev:
        print("STATUS: 未找到 ReSpeaker 设备，声源定位启动失败", flush=True)
        return
    Mic_tuning = Tuning(dev)

    DEAD_ZONE = 15
    FIXED_YAW_SPEED = 0.8
    DEG_PER_SEC = math.degrees(FIXED_YAW_SPEED)

    print("STATUS: 声源定位已启动，请说话测试", flush=True)

    try:
        while not _shutdown:
            if Mic_tuning.is_voice() == 1:
                doa_angle = Mic_tuning.direction

                if doa_angle > 180:
                    error_angle = doa_angle - 360
                else:
                    error_angle = doa_angle

                if abs(error_angle) > DEAD_ZONE:
                    turn_duration = abs(error_angle) / DEG_PER_SEC

                    if error_angle > 0:
                        yaw_cmd = FIXED_YAW_SPEED
                    else:
                        yaw_cmd = -FIXED_YAW_SPEED

                    logger.info(f"检测到声源，方位 {doa_angle} 度，正在转向")

                    start_time = time.time()
                    while time.time() - start_time < turn_duration and not _shutdown:
                        sport_client.Move(0.0, 0.0, float(yaw_cmd))
                        time.sleep(0.02)

                    sport_client.Move(0.0, 0.0, 0.0)
                    _sleep(0.5)

            else:
                sport_client.Move(0.0, 0.0, 0.0)

            _sleep(0.05)
    finally:
        sport_client.Move(0.0, 0.0, 0.0)
        print("STATUS: 声源定位已停止", flush=True)


if __name__ == '__main__':
    if os.environ.get("VOICE_AGENT_SUBPROCESS"):
        h = _PipeLogHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("audio_track.log"),
            ],
        )
    main()
