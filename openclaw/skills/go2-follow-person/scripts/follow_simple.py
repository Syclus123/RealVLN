#!/usr/bin/env python3
"""
Go2 Follow Person SIMPLE - 简化版跟随
去掉异步，直接拍照->检测->移动
"""
import os
import sys
import time
import cv2
import numpy as np

# 配置
CAPTURE_PATH = os.path.expanduser("~/.openclaw/media/go2_camera.jpg")
CENTER_THRESHOLD = 0.05  # 中心区域阈值
MOVE_COOLDOWN = 0.5      # 移动冷却时间（秒）
DETECT_INTERVAL = 0.5    # 检测间隔（秒）

# HOG 检测器
_hog = None

def get_hog():
    """获取预加载的 HOG 检测器"""
    global _hog
    if _hog is None:
        print("🔧 加载 HOG 检测器...")
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog


def capture_image():
    """拍照"""
    import subprocess
    result = subprocess.run(
        ['python3', '/home/unitree/openclaw/skills/go2-vision/scripts/capture.py'],
        capture_output=True, text=True, timeout=5
    )
    return result.returncode == 0


def detect_person(image_path):
    """检测人，返回 (detected, h_pos, v_pos)"""
    img = cv2.imread(image_path)
    if img is None:
        return (False, 0, 0)
    
    height, width = img.shape[:2]
    
    # 缩小图片加速检测
    scale = 480 / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    img_resized = cv2.resize(img, (new_width, new_height))
    
    hog = get_hog()
    boxes, weights = hog.detectMultiScale(
        img_resized, winStride=(8, 8), padding=(4, 4), scale=1.05
    )
    
    if len(boxes) > 0:
        # 选择最大的框
        best_idx = np.argmax([w * h for x, y, w, h in boxes])
        x, y, w, h = boxes[best_idx]
        
        # 转换回原始坐标
        x_center = (x + w/2) / new_width
        y_center = (y + h/2) / new_height
        
        h_pos = (x_center - 0.5) * 2  # -1 到 1
        v_pos = y_center
        
        return (True, h_pos, v_pos)
    
    return (False, 0, 0)


def move(action, angle=None):
    """移动"""
    import subprocess
    cmd = ['python3', '/home/unitree/openclaw/skills/unitree-go2/scripts/go2_control.py', action]
    if angle:
        cmd.extend(['--angle', str(angle)])
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  🎮 执行: {action}" + (f" {angle}°" if angle else ""))


def follow_simple():
    """简化跟随主循环"""
    print("🐕 简化跟随模式启动")
    print("=" * 50)
    print(f"阈值: ±{CENTER_THRESHOLD} (0.05)")
    print("=" * 50)
    print("按 Ctrl+C 停止\n")
    
    get_hog()  # 预加载
    last_move_time = 0
    frame = 0
    
    try:
        while True:
            frame += 1
            
            # 1. 拍照
            if not capture_image():
                print(f"\r{frame} | 拍照失败", end="", flush=True)
                time.sleep(0.5)
                continue
            
            # 2. 检测
            detected, h_pos, v_pos = detect_person(CAPTURE_PATH)
            
            # 3. 显示状态
            if detected:
                status = f"✅ 人 位置:{h_pos:+.2f}"
                print(f"\r{frame} | {status}", end="", flush=True)
                
                # 4. 决策移动
                current_time = time.time()
                if current_time - last_move_time > MOVE_COOLDOWN:
                    action = None
                    angle = None
                    
                    if h_pos < -CENTER_THRESHOLD:
                        action = 'turn_left'
                        angle = 20
                    elif h_pos > CENTER_THRESHOLD:
                        action = 'turn_right'
                        angle = 20
                    elif v_pos < 0.4:
                        action = 'move_forward'
                    elif v_pos > 0.7:
                        action = 'move_backward'
                    
                    if action:
                        print()  # 换行
                        move(action, angle)
                        last_move_time = current_time
            else:
                print(f"\r{frame} | ❌ 无人", end="", flush=True)
            
            time.sleep(DETECT_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n🛑 停止")
        move('stop')


if __name__ == '__main__':
    follow_simple()
