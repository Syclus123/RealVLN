#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
walk_the_dog.py — 自主漫步避障
机器狗以慢速持续前进，YOLO 检测到前方障碍物时随机选择左转90°、右转90°或掉头180°，
避开后继续前进，循环往复。

注意：YOLO 基于 COCO 80 类目标检测，能识别常见物体（家具、垃圾桶、行人等），
但无法直接检测纯墙面。如需墙壁检测可考虑结合深度传感器或边缘检测。
"""

import sys
import time
import random
import cv2
import numpy as np
import threading
from ultralytics import YOLO

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

# ==========================================
# 配置参数
# ==========================================
NETWORK_INTERFACE = "eth0"
MODEL_PATH = '/home/unitree/openclaw/skills/unitree-go2/yolov8n.pt'

WALK_SPEED = 0.3            # 巡航前进速度（m/s）
TURN_SPEED = 0.8            # 原地转向角速度（rad/s）

# 障碍检测阈值
OBSTACLE_Y2_THRESH = 0.70   # 障碍底边超过画面 70% → 距离过近
OBSTACLE_CENTER_BAND = 0.6  # 只关心画面中央 60% 宽度内的障碍（两侧忽略）
OBSTACLE_MIN_WIDTH = 0.12   # 障碍宽度至少占画面 12% 才视为威胁

# 转向时长（秒）— EMA 平滑会吃掉一部分响应，需要比理论值更长
TURN_90_DURATION = 2.5
TURN_180_DURATION = 5.0

# ==========================================
# 全局控制状态（视觉线程与控制线程共享）
# ==========================================
STATE_WALKING = "walking"
STATE_TURNING = "turning"

cmd_state = {
    "vx": 0.0,
    "vy": 0.0,
    "vyaw": 0.0,
    "last_update": 0.0,
    "mode": STATE_WALKING,
    "turn_end_time": 0.0,
}
CMD_TIMEOUT = 0.5


def control_thread_task(sport_client):
    """
    独立控制线程（20Hz）：读取目标速度，EMA 平滑后发送给电机。
    """
    print("🟢 [控制线程] 运动控制循环已启动 (20Hz)...")

    real_vx = 0.0
    real_vyaw = 0.0
    SMOOTH_FACTOR = 0.20  # 比跟随模式稍高，让避障刹车更快

    try:
        while True:
            if time.time() - cmd_state["last_update"] > CMD_TIMEOUT:
                target_vx, target_vy, target_vyaw = 0.0, 0.0, 0.0
            else:
                target_vx = cmd_state["vx"]
                target_vy = cmd_state["vy"]
                target_vyaw = cmd_state["vyaw"]

            real_vx += SMOOTH_FACTOR * (target_vx - real_vx)
            real_vyaw += SMOOTH_FACTOR * (target_vyaw - real_vyaw)

            if abs(real_vx) < 0.02: real_vx = 0.0
            if abs(real_vyaw) < 0.05: real_vyaw = 0.0

            if real_vx == 0.0 and target_vy == 0.0 and real_vyaw == 0.0:
                sport_client.StopMove()
            else:
                sport_client.Move(real_vx, target_vy, real_vyaw)

            time.sleep(0.05)

    except Exception as e:
        print(f"❌ [控制线程] 异常: {e}")
    finally:
        sport_client.StopMove()


def main():
    # ==========================================
    # 初始化
    # ==========================================
    print("🔌 正在初始化底层通道 (仅执行一次)...")
    ChannelFactoryInitialize(0, NETWORK_INTERFACE)

    video_client = VideoClient()
    video_client.SetTimeout(3.0)
    video_client.Init()
    print("✅ 视频客户端初始化完成")

    sport_client = SportClient()
    sport_client.SetTimeout(3.0)
    sport_client.Init()
    code, ver = sport_client.GetServerApiVersion()
    if code != 0:
        print(f"❌ 无法连接到运动服务器, 错误码: {code}")
        sys.exit(1)
    print(f"✅ 运动客户端初始化完成 (API: {ver})")

    print("🚀 正在加载 YOLO 模型并热身...")
    model = YOLO(MODEL_PATH)
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    ctrl_thread = threading.Thread(
        target=control_thread_task, args=(sport_client,), daemon=True
    )
    ctrl_thread.start()

    # ==========================================
    # 开始漫步
    # ==========================================
    cmd_state["vx"] = WALK_SPEED
    cmd_state["vyaw"] = 0.0
    cmd_state["mode"] = STATE_WALKING
    cmd_state["last_update"] = time.time()

    print("\n🐕 自主漫步避障已开启！按 Ctrl+C 停止。\n")

    try:
        while True:
            # --- 转向阶段：等待转完，期间持续刷新时间戳防止超时刹车 ---
            if cmd_state["mode"] == STATE_TURNING:
                if time.time() >= cmd_state["turn_end_time"]:
                    cmd_state["mode"] = STATE_WALKING
                    cmd_state["vx"] = WALK_SPEED
                    cmd_state["vyaw"] = 0.0
                    cmd_state["last_update"] = time.time()
                    print("🚶 转向完成，继续前进")
                else:
                    cmd_state["last_update"] = time.time()
                time.sleep(0.05)
                continue

            # --- 获取图像 ---
            code, data = video_client.GetImageSample()
            if code != 0:
                cmd_state["last_update"] = time.time()
                time.sleep(0.01)
                continue

            np_arr = np.frombuffer(bytes(data), np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                cmd_state["last_update"] = time.time()
                continue

            h, w = img.shape[:2]

            # --- YOLO 检测（所有 COCO 类别均视为潜在障碍） ---
            results = model(img, conf=0.45, verbose=False)

            obstacle_ahead = False
            obstacle_name = ""

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])

                    cx_norm = (x1 + x2) / (2 * w)     # 中心点水平归一化 [0,1]
                    width_norm = (x2 - x1) / w         # 宽度占比
                    y2_norm = y2 / h                    # 底边垂直归一化（越大越近）

                    in_path = abs(cx_norm - 0.5) < OBSTACLE_CENTER_BAND / 2
                    is_close = y2_norm > OBSTACLE_Y2_THRESH
                    is_significant = width_norm > OBSTACLE_MIN_WIDTH

                    if in_path and is_close and is_significant:
                        obstacle_ahead = True
                        obstacle_name = model.names.get(cls_id, f"cls_{cls_id}")
                        break
                if obstacle_ahead:
                    break

            # --- 决策 ---
            if obstacle_ahead:
                maneuver = random.choice(["left_90", "right_90", "turn_180"])

                if maneuver == "left_90":
                    cmd_state["vyaw"] = TURN_SPEED
                    duration = TURN_90_DURATION
                    desc = "左转 90°"
                elif maneuver == "right_90":
                    cmd_state["vyaw"] = -TURN_SPEED
                    duration = TURN_90_DURATION
                    desc = "右转 90°"
                else:
                    cmd_state["vyaw"] = TURN_SPEED
                    duration = TURN_180_DURATION
                    desc = "掉头 180°"

                cmd_state["vx"] = 0.0
                cmd_state["mode"] = STATE_TURNING
                cmd_state["turn_end_time"] = time.time() + duration
                cmd_state["last_update"] = time.time()
                print(f"⚠️ 前方障碍 [{obstacle_name}] → 执行: {desc}")
            else:
                cmd_state["vx"] = WALK_SPEED
                cmd_state["vyaw"] = 0.0
                cmd_state["last_update"] = time.time()

    except KeyboardInterrupt:
        print("\n🛑 接收到退出信号，正在安全停止机器狗...")
        cmd_state["vx"] = 0.0
        cmd_state["vyaw"] = 0.0
        cmd_state["last_update"] = 0.0
        time.sleep(0.5)


if __name__ == "__main__":
    main()
