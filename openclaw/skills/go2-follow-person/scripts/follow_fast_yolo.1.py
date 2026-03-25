#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import cv2
import subprocess
import numpy as np
from ultralytics import YOLO

# --- 配置路径 ---
CAPTURE_PATH = os.path.expanduser("~/.openclaw/media/go2_camera.jpg")
CONTROL_SCRIPT = '/home/unitree/openclaw/skills/unitree-go2/scripts/go2_control.py'
MODEL_PATH = '/home/unitree/openclaw/skills/unitree-go2/yolov8n.pt'
CAPTURE_SCRIPT = '/home/unitree/openclaw/skills/go2-vision/scripts/capture.py'

# --- 核心参数调整 ---
CENTER_THRESHOLD = 0.12
# y2 越小人越远。0.70 是个比较灵敏的追击点
FAR_Y2_THRESHOLD = 0.70     
CLOSE_Y2_THRESHOLD = 0.94   
# 动作冷却（由于指令本身运行慢，我们将冷却调至 0.1s，依靠 _is_executing 锁来控制节奏）
MOVE_COOLDOWN = 0.1         

_is_executing = False

def load_model():
    print("1. 正在加载模型...", flush=True)
    model = YOLO(MODEL_PATH)
    print("2. 预热 GPU...", flush=True)
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    return model

def safe_execute(action, angle=None):
    """带锁的异步执行，绝不触发 Timeout 崩溃"""
    global _is_executing
    if _is_executing:
        return
    
    def run():
        global _is_executing
        _is_executing = True
        try:
            cmd = ['python3', CONTROL_SCRIPT, action]
            if angle: cmd.extend(['--angle', str(angle)])
            # 使用 run 确保脚本运行完，但因为它在独立线程里，所以不会卡死主程序
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            _is_executing = False

    import threading
    threading.Thread(target=run, daemon=True).start()

def main():
    os.system("pkill -9 -f capture.py")
    os.system("pkill -9 -f go2_control.py")
    
    model = load_model()
    last_action_time = 0
    print("3. 进入主循环 (异步锁模式)...", flush=True)
    
    try:
        while True:
            # 抓图
            subprocess.run(['python3', CAPTURE_SCRIPT], stdout=subprocess.DEVNULL)
            if not os.path.exists(CAPTURE_PATH):
                continue
            img = cv2.imread(CAPTURE_PATH)
            if img is None: continue

            # 推理
            results = model(img, classes=[0], conf=0.45, verbose=False)
            detected, h_pos, y2_pos = False, 0.0, 0.0
            
            max_area = 0
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2-x1)*(y2-y1)
                    if area > max_area:
                        max_area, detected = area, True
                        h_pos = ((x1 + x2) / (2 * img.shape[1]) - 0.5) * 2
                        y2_pos = y2 / img.shape[0]

            if detected:
                now = time.time()
                if (now - last_action_time) > MOVE_COOLDOWN:
                    # 转弯大步幅：最小20度，确保迈腿
                    angle = max(20, min(40, int(abs(h_pos) * 45)))

                    # --- 复合判定 (不再用 elif) ---
                    # 1. 前后
                    if y2_pos < FAR_Y2_THRESHOLD:
                        print(f"[ACTION] 前进 | y2={y2_pos:.2f}")
                        safe_execute('move_forward')
                        last_action_time = now
                    elif y2_pos > CLOSE_Y2_THRESHOLD:
                        print(f"[ACTION] 后退 | y2={y2_pos:.2f}")
                        safe_execute('move_backward')
                        last_action_time = now
                    
                    # 2. 转向 (如果正在前进，则不重复发转向，除非偏差极大)
                    if h_pos < -CENTER_THRESHOLD:
                        print(f"[ACTION] 左转 {angle}度 | 偏={h_pos:.2f}")
                        safe_execute('turn_left', angle)
                        last_action_time = now
                    elif h_pos > CENTER_THRESHOLD:
                        print(f"[ACTION] 右转 {angle}度 | 偏={h_pos:.2f}")
                        safe_execute('turn_right', angle)
                        last_action_time = now
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        subprocess.run(['python3', CONTROL_SCRIPT, 'stop'])
        print("\n🛑 安全退出")

if __name__ == '__main__':
    main()
