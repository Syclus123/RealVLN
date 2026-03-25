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
# 🚀 PRO 版性能参数
# ==========================================
NETWORK_INTERFACE = "eth0"
MODEL_PATH = '/home/unitree/openclaw/skills/unitree-go2/yolov8n.pt'
SOUND_PATH = '/home/unitree/openclaw/skills/go2-audio-play/sounds/dog_barking.wav'

# 追踪参数优化
VYAW_GAIN = 2.5           # 🚀 从 1.5 提升到 2.5，大幅增强转弯速度
MAX_VYAW = 1.2            # 最大旋转角速度
DEADZONE_X = 0.05         # 视觉中心死区，防止画面微小抖动导致狗子频繁踏步

# 姿态与平滑
POUNCE_PITCH = 0.40      
RAISE_PITCH = -0.20      
MAX_PITCH_STEP = 0.015    
SMOOTH_FACTOR_VEL = 0.25  # 🚀 提升速度响应平滑度

# 距离与速度
RETREAT_SPEED = 0.1     

cmd_state = {
    "vx": 0.0,
    "vyaw": 0.0,
    "pitch": 0.0,
    "is_barking_now": False, 
    "last_person_seen": 0.0   # 记录最后一次看到人的时间
}

bark_event = threading.Event()

# ==========================================
# 1. 音频守护线程
# ==========================================
def audio_worker_thread():
    while True:
        if bark_event.is_set():
            cmd_state["is_barking_now"] = True
            play_sound(SOUND_PATH)
            time.sleep(0.4) 
            cmd_state["is_barking_now"] = False
            time.sleep(1.2) 
            bark_event.clear()
        else:
            time.sleep(0.01)

# ==========================================
# 2. 运动控制线程 (心跳保活版)
# ==========================================
def control_thread_task(sport_client):
    print("🟢 [控制线程] 25Hz 高频保活指令已启动...")
    real_vx, real_vyaw, real_pitch = 0.0, 0.0, 0.0
    
    try:
        while True:
            now = time.time()
            # 如果超过 0.8s 没看到人，则平滑归零
            if now - cmd_state["last_person_seen"] > 0.8:
                t_vx, t_vyaw, t_pitch = 0.0, 0.0, 0.0
            else:
                t_vx = cmd_state["vx"]
                t_vyaw = cmd_state["vyaw"]
                t_pitch = RAISE_PITCH if cmd_state["is_barking_now"] else cmd_state["pitch"]

            # 物理模拟平滑
            real_vx += SMOOTH_FACTOR_VEL * (t_vx - real_vx)
            real_vyaw += SMOOTH_FACTOR_VEL * (t_vyaw - real_vyaw)
            
            # Pitch 步进限速
            diff_p = t_pitch - real_pitch
            step_p = MAX_PITCH_STEP if t_pitch < real_pitch else (MAX_PITCH_STEP * 0.5)
            if abs(diff_p) > step_p:
                real_pitch += np.sign(diff_p) * step_p
            else:
                real_pitch = t_pitch

            # 🚀 解决踏步不连续：无论如何都发送指令，维持 SDK 状态机活跃
            if abs(real_vx) < 0.01 and abs(real_vyaw) < 0.02 and abs(real_pitch) < 0.01:
                sport_client.StopMove()
            else:
                sport_client.Move(real_vx, 0.0, real_vyaw)
                sport_client.Euler(0.0, real_pitch, 0.0)
            
            # 25Hz 频率是 Go2 响应最顺滑的区间
            time.sleep(0.04) 
    except Exception as e:
        print(f"❌ [控制线程] 严重异常: {e}")

# ==========================================
# 3. 决策大脑
# ==========================================
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

    print("\n🐕 Guard Dog V5-PRO: 极速追踪+连续步态版激活！")
    
    last_action = ""
    try:
        while True:
            code, data = video_client.GetImageSample()
            if code != 0: continue
            
            img = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
            # 🚀 稍微降低置信度门槛至 0.4，增加复杂光线下的追踪稳定性
            results = model.track(img, classes=[0], conf=0.40, persist=True, tracker="botsort.yaml", verbose=False)
            
            v_x, v_yaw, p_pitch = 0.0, 0.0, 0.0
            action = "😌 安全中..."

            if results[0].boxes.id is not None:
                cmd_state["last_person_seen"] = time.time() # 刷新视觉保活
                
                boxes = results[0].boxes.xyxy.cpu().numpy()
                idx = np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in boxes])
                x1, y1, x2, y2 = boxes[idx]
                
                # 计算水平偏差 (h_err 范围 -1 到 1)
                h_err = ((x1 + x2) / (2 * img.shape[1]) - 0.5) * 2
                
                # 🚀 快速转向逻辑：死区之外全力旋转
                if abs(h_err) > DEADZONE_X:
                    v_yaw = -h_err * VYAW_GAIN
                    v_yaw = np.clip(v_yaw, -MAX_VYAW, MAX_VYAW)
                
                # 距离计算
                dist = np.interp(y2 / img.shape[0], [0.70, 0.80, 0.90, 0.95], [3.0, 2.0, 1.0, 0.5])

                if dist > 3.0:
                    action = f"👀 盯着你呢 [距 {dist:.1f}m]"
                else:
                    bark_event.set()
                    p_pitch = POUNCE_PITCH
                    if dist < 1.5:
                        action = f"🐕 警戒退防 [距 {dist:.1f}m]"
                        v_x = RETREAT_SPEED
                    else:
                        action = f"🚨 猛犬前压 [距 {dist:.1f}m]"
            
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
