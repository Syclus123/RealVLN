#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import signal
import cv2
import numpy as np
import threading
import argparse
from ultralytics import YOLO

# 导入宇树 SDK
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

# 动态导入语音技能
sys.path.insert(0, '/home/unitree/openclaw/skills/go2-audio-play/scripts')
from play import play_sound

logger = logging.getLogger("guard_dog")


class _PipeLogHandler(logging.Handler):
    """子进程 → 父进程日志协议: 通过 stdout 输出 LOG:LEVEL:message"""
    def emit(self, record):
        try:
            print(f"LOG:{record.levelname}:{self.format(record)}", flush=True)
        except Exception:
            pass


# ==========================================
# 核心配置与阈值
# ==========================================
MODEL_PATH = '/home/unitree/openclaw/skills/unitree-go2/yolov8n.pt'
SOUND_PATH = '/home/unitree/openclaw/skills/go2-audio-play/sounds/dog_barking.wav'

WARN_DISTANCE = 2.0
DANGER_DISTANCE = 1.0
BARK_INTERVAL = 1.8

cmd_state = {
    "vx": 0.0,
    "vyaw": 0.0,
    "force_step": False,
    "last_update": time.time(),
    "active": True
}

bark_event = threading.Event()


def _on_signal(signum, frame):
    cmd_state["active"] = False


# ==========================================
# 1. 音频守护线程
# ==========================================
def _sleep(duration):
    """可被 cmd_state["active"]=False 中断的 sleep，最大响应延迟 ~20ms。"""
    end = time.time() + duration
    while cmd_state["active"] and time.time() < end:
        time.sleep(min(0.02, end - time.time()))


def audio_worker_thread():
    while cmd_state["active"]:
        if bark_event.is_set():
            play_sound(SOUND_PATH)
            _sleep(BARK_INTERVAL)
            bark_event.clear()
        else:
            _sleep(0.05)

# ==========================================
# 2. 运动控制线程 (极简倒脚踏步版)
# ==========================================
def control_thread_task(sport_client):
    real_vx, real_vy, real_vyaw = 0.0, 0.0, 0.0
    SMOOTH_FACTOR = 0.25

    try:
        while cmd_state["active"]:
            if time.time() - cmd_state["last_update"] > 0.5:
                target_vx, target_vyaw = 0.0, 0.0
                cmd_state["force_step"] = False
            else:
                target_vx = cmd_state["vx"]
                target_vyaw = cmd_state["vyaw"]

            if cmd_state["force_step"]:
                target_vy = 0.03 if int(time.time() * 5) % 2 == 0 else -0.03
            else:
                target_vy = 0.0

            real_vx += SMOOTH_FACTOR * (target_vx - real_vx)
            real_vy += SMOOTH_FACTOR * (target_vy - real_vy)
            real_vyaw += SMOOTH_FACTOR * (target_vyaw - real_vyaw)

            if abs(real_vx) < 0.001 and abs(real_vy) < 0.001 and abs(real_vyaw) < 0.001:
                sport_client.StopMove()
            else:
                sport_client.Move(real_vx, real_vy, real_vyaw)

            time.sleep(0.05)
    except Exception as e:
        logger.error(f"[控制线程] 异常: {e}")
        print(f"STATUS: 控制线程异常: {e}", flush=True)
    finally:
        sport_client.StopMove()

# ==========================================
# 3. 决策大脑
# ==========================================
def estimate_distance(y2):
    return np.interp(y2, [0.70, 0.80, 0.90, 0.95], [3.0, 2.0, 1.0, 0.5])

def main():
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    parser = argparse.ArgumentParser()
    parser.add_argument("interface", nargs="?", default="eth0",
                        help="network interface (default: eth0)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.interface)
    video_client = VideoClient()
    video_client.Init()

    sport_client = None
    if not args.test:
        sport_client = SportClient()
        sport_client.SetTimeout(3.0)
        sport_client.Init()
        code, ver = sport_client.GetServerApiVersion()
        if code == 0:
            logger.info(f"运动服务器连接成功 (API: {ver})")

    model = YOLO(MODEL_PATH)
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    threading.Thread(target=audio_worker_thread, daemon=True).start()
    if sport_client:
        threading.Thread(target=control_thread_task, args=(sport_client,), daemon=True).start()

    last_action = ""
    last_person_time = 0.0

    print("STATUS: 看门狗模式已启动，开始监控", flush=True)

    try:
        while cmd_state["active"]:
            code, data = video_client.GetImageSample()
            if code != 0:
                continue

            img = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            results = model.track(img, classes=[0], conf=0.45, persist=True,
                                  tracker="botsort.yaml", verbose=False)
            person_detected = len(results[0].boxes) > 0
            action = ""

            if person_detected:
                last_person_time = time.time()

                boxes = results[0].boxes.xyxy.cpu().numpy()
                idx = np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in boxes])
                x1, y1, x2, y2 = boxes[idx]

                dist = estimate_distance(y2 / img.shape[0])

                tracking_yaw = -(((x1 + x2) / (2 * img.shape[1]) - 0.5) * 2) * 2.5
                tracking_yaw = max(-1.0, min(1.0, tracking_yaw))

                if dist > WARN_DISTANCE:
                    action = f"远距盯防 [距 {dist:.1f}米]"
                    cmd_state["vx"] = 0.0
                    cmd_state["vyaw"] = tracking_yaw if abs(tracking_yaw) > 0.1 else 0.0
                    cmd_state["force_step"] = False
                elif dist > DANGER_DISTANCE:
                    action = f"警戒踏步 [距 {dist:.1f}米]"
                    cmd_state["vx"] = 0.0
                    cmd_state["vyaw"] = tracking_yaw
                    cmd_state["force_step"] = True
                else:
                    action = f"狂吠警告 [距 {dist:.1f}米]"
                    bark_event.set()
                    cmd_state["vx"] = 0.0
                    cmd_state["vyaw"] = tracking_yaw
                    cmd_state["force_step"] = True

                cmd_state["last_update"] = time.time()

            else:
                if time.time() - last_person_time < 1.5:
                    pass
                else:
                    action = "安全"
                    cmd_state["vx"] = 0.0
                    cmd_state["vyaw"] = 0.0
                    cmd_state["force_step"] = False
                    cmd_state["last_update"] = time.time()

            if action and action != last_action:
                logger.info(f"状态切换: {action}")
                last_action = action

    except KeyboardInterrupt:
        pass
    finally:
        cmd_state["active"] = False
        cmd_state["vx"] = 0.0
        cmd_state["vyaw"] = 0.0
        cmd_state["force_step"] = False
        cmd_state["last_update"] = 0.0
        time.sleep(0.5)
        print("STATUS: 看门狗模式已停止", flush=True)

if __name__ == "__main__":
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
                logging.FileHandler("guard_dog.log"),
            ],
        )
    main()

