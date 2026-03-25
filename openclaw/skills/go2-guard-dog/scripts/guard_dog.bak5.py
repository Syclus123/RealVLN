#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
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

# ==========================================
# 配置与阈值
# ==========================================
NETWORK_INTERFACE = "eth0"
MODEL_PATH = '/home/unitree/openclaw/skills/unitree-go2/yolov8n.pt'
SOUND_PATH = '/home/unitree/openclaw/skills/go2-audio-play/sounds/dog_barking.wav'

WARN_DISTANCE = 2.0      
DANGER_DISTANCE = 1.0    
RETREAT_SPEED = 0.1     

# 姿态参数
POUNCE_PITCH = 0.40      # 前压低头
RAISE_PITCH = -0.20      # 后仰抬升
MAX_PITCH_STEP = 0.02    # ⚠️ 关键：每帧最大 Pitch 变化量（限制速度）
BARK_INTERVAL = 1.8      # 稍微拉长间隔，留出动作回位时间

cmd_state = {
    "vx": 0.0,
    "vyaw": 0.0,
    "pitch": 0.0,
    "is_barking_now": False, 
    "last_update": 0.0
}

bark_event = threading.Event()

# ==========================================
# 1. 音频守护线程 - 增加动作保持逻辑
# ==========================================
def audio_worker_thread():
    print("🟢 [音频线程] 柔性同步系统就绪...")
    while True:
        if bark_event.is_set():
            # 抬起动作开始
            cmd_state["is_barking_now"] = True
            play_sound(SOUND_PATH)
            
            # 叫完之后，保持抬高姿态一小会，再允许慢慢压回去
            time.sleep(0.4) 
            cmd_state["is_barking_now"] = False
            
            # 强制冷却，防止连续触发导致身体疯狂抽搐
            time.sleep(BARK_INTERVAL) 
            bark_event.clear()
        else:
            time.sleep(0.05)

# ==========================================
# 2. 运动控制线程 - 引入步进限速器
# ==========================================
# ==========================================
# 2. 运动控制线程 (彻底解决指令打架死锁)
# ==========================================
def control_thread_task(sport_client):
    print("🟢 [控制线程] 互斥锁已激活，解决步态与姿态冲突...")
    real_vx, real_vyaw, real_pitch = 0.0, 0.0, 0.0
    
    try:
        while cmd_state["active"]:
            now = time.time()
            if now - cmd_state["last_update"] > 0.5:
                t_vx, t_vyaw, t_pitch = 0.0, 0.0, 0.0
            else:
                t_vx = cmd_state["vx"]
                t_vyaw = cmd_state["vyaw"]
                t_pitch = RAISE_PITCH if cmd_state["is_barking_now"] else cmd_state["pitch"]

            # 速度与姿态平滑
            real_vx += 0.2 * (t_vx - real_vx)
            real_vyaw += 0.2 * (t_vyaw - real_vyaw)

            diff = t_pitch - real_pitch
            current_step = MAX_PITCH_STEP if t_pitch < real_pitch else (MAX_PITCH_STEP * 0.5)
            if abs(diff) > current_step:
                real_pitch += np.sign(diff) * current_step
            else:
                real_pitch = t_pitch

            # 死区过滤
            if abs(real_vx) < 0.02: real_vx = 0.0
            if abs(real_vyaw) < 0.05: real_vyaw = 0.0
            if abs(real_pitch) < 0.02: real_pitch = 0.0

            # ⚠️ 致命 Bug 修复：互斥锁！
            # 如果需要动腿，优先保运动，绝对不发 Euler 干扰状态机
            if real_vx != 0.0 or real_vyaw != 0.0:
                sport_client.Move(real_vx, 0.0, real_vyaw)
            else:
                # 只有脚步完全静止，才执行压低/抬高姿态
                sport_client.StopMove()
                sport_client.Euler(0.0, real_pitch, 0.0)
            
            time.sleep(0.05) 
    except Exception as e:
        print(f"❌ [控制线程] 异常: {e}")
    finally:
        sport_client.StopMove()
        sport_client.Euler(0,0,0)


# ==========================================
# 3. 决策大脑
# ==========================================
def estimate_distance(y2):
    return np.interp(y2, [0.70, 0.80, 0.90, 0.95], [3.0, 2.0, 1.0, 0.5])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, NETWORK_INTERFACE)
    video_client = VideoClient()
    video_client.Init()

    sport_client = None
    if not args.test:
        sport_client = SportClient()
        sport_client.Init()

    model = YOLO(MODEL_PATH)
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    threading.Thread(target=audio_worker_thread, daemon=True).start()
    if sport_client:
        threading.Thread(target=control_thread_task, args=(sport_client,), daemon=True).start()

    print("\n🐕 Guard Dog V4-Smooth: 柔性威慑模式已激活！")
    
    last_action = ""
    try:
        while True:
            code, data = video_client.GetImageSample()
            if code != 0: continue
            
            img = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
            results = model.track(img, classes=[0], conf=0.45, persist=True, tracker="botsort.yaml", verbose=False)
            
            person_detected = results[0].boxes.id is not None
            v_x, v_yaw, p_pitch = 0.0, 0.0, 0.0
            
            if person_detected:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                idx = np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in boxes])
                x1, y1, x2, y2 = boxes[idx]
                
                dist = estimate_distance(y2 / img.shape[0])
                v_yaw = -(((x1 + x2) / (2 * img.shape[1]) - 0.5) * 2) * 1.5
                v_yaw = max(-0.8, min(0.8, v_yaw))

                if dist > WARN_DISTANCE:
                    action = f"👀 盯着你呢 [距 {dist:.1f}m]"
                    p_pitch = 0.0
                elif dist > DANGER_DISTANCE:
                    action = f"🚨 猛犬前压 [距 {dist:.1f}m]"
                    bark_event.set()
                    p_pitch = POUNCE_PITCH 
                else:
                    action = f"🐕 警戒退防 [距 {dist:.1f}m]"
                    v_x = RETREAT_SPEED
                    bark_event.set()
                    p_pitch = POUNCE_PITCH 
            else:
                action = "😌 安全中..."
            
            if not args.test:
                cmd_state["vx"], cmd_state["vyaw"], cmd_state["pitch"] = v_x, v_yaw, p_pitch
                cmd_state["last_update"] = time.time()

            if action != last_action:
                print(f"[{time.strftime('%H:%M:%S')}] {action}")
                last_action = action

    except KeyboardInterrupt:
        print("\n🛑 安全退出...")

if __name__ == "__main__":
    main()
