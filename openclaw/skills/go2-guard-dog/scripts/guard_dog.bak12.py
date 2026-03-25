#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
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
# 🚀 核心配置与阈值
# ==========================================
NETWORK_INTERFACE = "eth0"
MODEL_PATH = '/home/unitree/openclaw/skills/unitree-go2/yolov8n.pt'
SOUND_PATH = '/home/unitree/openclaw/skills/go2-audio-play/sounds/dog_barking.wav'

WARN_DISTANCE = 2.0      
DANGER_DISTANCE = 1.0    
BARK_INTERVAL = 1.8      

cmd_state = {
    "vx": 0.0,
    "vyaw": 0.0,
    "last_update": time.time(),
    "active": True
}

bark_event = threading.Event()

# ==========================================
# 1. 音频守护线程
# ==========================================
def audio_worker_thread():
    print("🟢 [音频线程] 狂吠系统就绪...")
    while cmd_state["active"]:
        if bark_event.is_set():
            play_sound(SOUND_PATH)
            time.sleep(BARK_INTERVAL) 
            bark_event.clear()
        else:
            time.sleep(0.05)

# ==========================================
# 2. 运动控制线程 (极简还原昨天成功的逻辑)
# ==========================================
def control_thread_task(sport_client):
    print("🟢 [控制线程] 极简动力引擎已激活 (去除所有阻碍踏步的死区)...")
    real_vx = 0.0
    real_vyaw = 0.0
    # 调高平滑因子，让 YOLO 的微小抖动能更直接地传递给电机
    SMOOTH_FACTOR = 0.25 

    try:
        while cmd_state["active"]:
            if time.time() - cmd_state["last_update"] > 0.5:
                target_vx, target_vyaw = 0.0, 0.0
            else:
                target_vx = cmd_state["vx"]
                target_vyaw = cmd_state["vyaw"]

            real_vx += SMOOTH_FACTOR * (target_vx - real_vx)
            real_vyaw += SMOOTH_FACTOR * (target_vyaw - real_vyaw)

            # ⚠️ 致命 Bug 修复处：删除了所有的 abs() < 0.05 的死区过滤！
            # 还原昨天成功的秘诀：允许微小波动直达底层，保持踏步引擎自然活跃！

            if abs(real_vx) < 0.001 and abs(real_vyaw) < 0.001:
                sport_client.StopMove()
            else:
                sport_client.Move(real_vx, 0.0, real_vyaw)

            time.sleep(0.05)
    except Exception as e:
        print(f"❌ [控制线程] 异常: {e}")
    finally:
        sport_client.StopMove()
        print("✅ [控制线程] 已安全锁死底盘")

# ==========================================
# 3. 决策大脑
# ==========================================
def estimate_distance(y2):
    return np.interp(y2, [0.70, 0.80, 0.90, 0.95], [3.0, 2.0, 1.0, 0.5])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("🐕 Guard Dog V12: 极简原味版 (定海神针+自然踏步)")
    print("=" * 60)

    ChannelFactoryInitialize(0, NETWORK_INTERFACE)
    video_client = VideoClient()
    video_client.Init()

    sport_client = None
    if not args.test:
        sport_client = SportClient()
        sport_client.SetTimeout(3.0)
        sport_client.Init()
        code, ver = sport_client.GetServerApiVersion()
        if code == 0:
            print(f"✅ 运动服务器连接成功 (API版本: {ver})")

    model = YOLO(MODEL_PATH)
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    threading.Thread(target=audio_worker_thread, daemon=True).start()
    if sport_client:
        threading.Thread(target=control_thread_task, args=(sport_client,), daemon=True).start()
    
    last_action = ""
    last_person_time = 0.0 
    
    try:
        while True:
            code, data = video_client.GetImageSample()
            if code != 0: continue
            
            img = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            results = model.track(img, classes=[0], conf=0.45, persist=True, tracker="botsort.yaml", verbose=False)
            
            person_detected = len(results[0].boxes) > 0
            
            if person_detected:
                last_person_time = time.time() 
                
                boxes = results[0].boxes.xyxy.cpu().numpy()
                idx = np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in boxes])
                x1, y1, x2, y2 = boxes[idx]
                
                dist = estimate_distance(y2 / img.shape[0])
                
                # 放大转向系数，保证狗子转身果断且必须抬腿
                tracking_yaw = -(((x1 + x2) / (2 * img.shape[1]) - 0.5) * 2) * 2.5
                tracking_yaw = max(-1.0, min(1.0, tracking_yaw))

                if dist > WARN_DISTANCE:
                    action = f"👀 远距盯防 [距 {dist:.1f}m]"
                    cmd_state["vx"] = 0.0             
                    # 远距离给一个极小的脱离阈值，防止在原地瞎晃
                    cmd_state["vyaw"] = tracking_yaw if abs(tracking_yaw) > 0.1 else 0.0
                elif dist > DANGER_DISTANCE:
                    action = f"🚶 警戒！原地踏步+锁定 (不叫) [距 {dist:.1f}m]"
                    cmd_state["vx"] = 0.0             # 绝对不前进
                    cmd_state["vyaw"] = tracking_yaw  # 传递 YOLO 抖动，保持踏步
                else:
                    action = f"💥 死磕！狂吠+原地踏步+锁定 [距 {dist:.1f}m]"
                    bark_event.set()                  # 触发狗叫！
                    cmd_state["vx"] = 0.0             # 绝对不前进
                    cmd_state["vyaw"] = tracking_yaw  # 传递 YOLO 抖动，保持踏步
                
                cmd_state["last_update"] = time.time()
                
            else:
                if time.time() - last_person_time < 1.5:
                    pass 
                else:
                    action = "😌 安全中..."
                    cmd_state["vx"] = 0.0
                    cmd_state["vyaw"] = 0.0
                    cmd_state["last_update"] = time.time()

            if action != last_action:
                print(f"[{time.strftime('%H:%M:%S')}] {action}")
                last_action = action

    except KeyboardInterrupt:
        print("\n🛑 安全退出...")
        cmd_state["active"] = False
        time.sleep(0.5)

if __name__ == "__main__":
    main()
