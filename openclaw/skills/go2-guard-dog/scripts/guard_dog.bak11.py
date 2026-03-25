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

# ⚠️ 能够击穿死区并保持原地踏步的黄金速度
TROT_SPEED = 0.15        
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
# 2. 运动控制线程 (被验证成功的 V9 黄金动力版)
# ==========================================
def control_thread_task(sport_client):
    print("🟢 [控制线程] 运动循环已启动 (完美复刻 V9)...")
    real_vx = 0.0
    real_vyaw = 0.0
    SMOOTH_FACTOR = 0.15

    try:
        while cmd_state["active"]:
            if time.time() - cmd_state["last_update"] > 0.5:
                target_vx, target_vyaw = 0.0, 0.0
            else:
                target_vx = cmd_state["vx"]
                target_vyaw = cmd_state["vyaw"]

            real_vx += SMOOTH_FACTOR * (target_vx - real_vx)
            real_vyaw += SMOOTH_FACTOR * (target_vyaw - real_vyaw)

            if abs(real_vx) < 0.02: real_vx = 0.0
            if abs(real_vyaw) < 0.05: real_vyaw = 0.0

            if real_vx == 0.0 and real_vyaw == 0.0:
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
# 3. 决策大脑 (精准定制距离逻辑)
# ==========================================
def estimate_distance(y2):
    return np.interp(y2, [0.70, 0.80, 0.90, 0.95], [3.0, 2.0, 1.0, 0.5])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("🐕 Guard Dog V10: 完美看门狗版 (精准距离控制)")
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
        else:
            print("❌ 警告：无法连接到运动服务器！请检查狗子状态！")

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
            results = model.track(img, classes=[0], conf=0.45, persist=True, tracker="botsort.yaml", verbose=False)
            
            person_detected = len(results[0].boxes) > 0
            
            if person_detected:
                last_person_time = time.time() 
                
                boxes = results[0].boxes.xyxy.cpu().numpy()
                idx = np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in boxes])
                x1, y1, x2, y2 = boxes[idx]
                
                dist = estimate_distance(y2 / img.shape[0])
                
                # 强化转向系数，不管多远多近，只要人动，狗头就死死咬住
                tracking_yaw = -(((x1 + x2) / (2 * img.shape[1]) - 0.5) * 2) * 2.0
                tracking_yaw = max(-1.0, min(1.0, tracking_yaw))

                if dist > WARN_DISTANCE:
                    action = f"👀 远距盯防 [距 {dist:.1f}m]"
                    cmd_state["vx"] = 0.0             # >2米：不踏步
                    cmd_state["vyaw"] = tracking_yaw  # 保持转向跟踪
                elif dist > DANGER_DISTANCE:
                    action = f"🚶 中距警戒！踏步+转向锁定 [距 {dist:.1f}m]"
                    # ⚠️ 1-2米：踏步 + 转向锁定，但去掉了 bark_event.set() (不叫)
                    cmd_state["vx"] = TROT_SPEED       
                    cmd_state["vyaw"] = tracking_yaw  
                else:
                    action = f"💥 近距死磕！狂吠+踏步+转向锁定 [距 {dist:.1f}m]"
                    bark_event.set()                  # ⚠️ 1米内：触发狗叫！
                    cmd_state["vx"] = TROT_SPEED      # 保持踏步
                    cmd_state["vyaw"] = tracking_yaw  # 保持转向锁定
                
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
