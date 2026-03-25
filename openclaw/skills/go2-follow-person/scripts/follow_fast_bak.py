#!/usr/bin/env python3
"""
Go2 Follow Person FAST - 极速跟随版
优化：内联检测 + 并行处理 + 异步移动
"""
import os
import sys
import time
import threading
import cv2
import numpy as np
from datetime import datetime

# 添加依赖路径
sys.path.insert(0, '/home/unitree/openclaw/skills/unitree-go2/scripts')

# 配置
CAPTURE_PATH = os.path.expanduser("~/.openclaw/media/go2_camera.jpg")
CENTER_THRESHOLD = 0.05  # 中心区域阈值（从 0.1 降到 0.05，超敏感）
MOVE_COOLDOWN = 0.5     # 移动冷却时间（秒）
DETECT_INTERVAL = 0.1   # 检测间隔（秒）- 从 0.3s 加快到 0.1s

# HOG 检测器（预加载）
_hog = None
_last_capture_time = 0
_latest_image = None
_capture_lock = threading.Lock()

def get_hog():
    """获取预加载的 HOG 检测器"""
    global _hog
    if _hog is None:
        print("🔧 预加载 HOG 检测器...")
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog


def capture_async():
    """异步拍照（后台线程）"""
    global _last_capture_time, _latest_image
    
    import subprocess
    while True:
        try:
            result = subprocess.run(
                ['python3', '/home/unitree/openclaw/skills/go2-vision/scripts/capture.py'],
                capture_output=True, text=True, timeout=5
            )
            with _capture_lock:
                _last_capture_time = time.time()
                _latest_image = CAPTURE_PATH
        except:
            pass
        time.sleep(0.5)  # 每0.5秒拍一次


def detect_person_fast(image_path):
    """
    极速人检测（内联 + 缩小图片）
    返回: (detected, h_pos, v_pos, x, y, w, h) 目标 <50ms
    """
    start = time.time()
    
    img = cv2.imread(image_path)
    if img is None:
        return (False, 0, 0, 0, 0, 0, 0)
    
    orig_height, orig_width = img.shape[:2]
    
    # 🔧 优化：缩小图片加速检测 (1920x1080 -> 480x270)
    TARGET_WIDTH = 480
    scale = TARGET_WIDTH / orig_width
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    img_resized = cv2.resize(img, (new_width, new_height))
    
    hog = get_hog()
    
    # 在缩小后的图片上检测
    boxes, weights = hog.detectMultiScale(
        img_resized, winStride=(8, 8), padding=(4, 4), scale=1.05
    )
    
    detect_time = (time.time() - start) * 1000
    
    if len(boxes) > 0:
        # 选择最大的框
        best_idx = np.argmax([w * h for x, y, w, h in boxes])
        x, y, w, h = boxes[best_idx]
        
        # 转换回原始图片坐标
        x_orig = x / scale
        y_orig = y / scale
        w_orig = w / scale
        h_orig = h / scale
        
        x_center = (x_orig + w_orig/2) / orig_width
        y_center = (y_orig + h_orig/2) / orig_height
        h_pos = (x_center - 0.5) * 2
        v_pos = y_center
        
        return (True, h_pos, v_pos, int(x_orig), int(y_orig), int(w_orig), int(h_orig))
    
    return (False, 0, 0, 0, 0, 0, 0)


def move_async(action, angle=None):
    """异步移动（不等待完成）"""
    def do_move():
        try:
            import subprocess
            cmd = ['python3', '/home/unitree/openclaw/skills/unitree-go2/scripts/go2_control.py', action]
            if angle:
                cmd.extend(['--angle', str(angle)])
            subprocess.run(cmd, capture_output=True, timeout=3)
        except:
            pass
    
    threading.Thread(target=do_move, daemon=True).start()


def send_image_to_feishu(image_path):
    """发送图片到飞书"""
    try:
        import subprocess
        import os
        # 使用已有的发送图片脚本
        env = os.environ.copy()
        env["FEISHU_CHAT_ID"] = "ou_da6b544f17a5b166a23d6cb4ce4e9b82"
        
        # 调用发送图片
        result = subprocess.run(
            ['python3', '/home/unitree/openclaw/send_image_to_feishu.py', image_path],
            capture_output=True,
            text=True,
            timeout=10,
            env=env
        )
        return result.returncode == 0
    except:
        return False


def follow_fast():
    """极速跟随主循环"""
    print("🐕 极速跟随模式启动")
    print("=" * 50)
    print(f"📷 拍照间隔: 0.5s (后台)")
    print(f"🔍 检测间隔: {DETECT_INTERVAL}s")
    print(f"🎮 移动冷却: {MOVE_COOLDOWN}s")
    print(f"📤 每10秒发送图片到飞书")
    print("=" * 50)
    print("按 Ctrl+C 停止\n")
    
    # 启动异步拍照线程
    capture_thread = threading.Thread(target=capture_async, daemon=True)
    capture_thread.start()
    
    # 预加载 HOG
    get_hog()
    
    last_move_time = 0
    last_detect_time = 0
    last_image_send_time = 0
    frame_count = 0
    
    try:
        while True:
            frame_count += 1
            loop_start = time.time()
            
            # 每10秒发送一张图片到飞书
            current_time = time.time()
            if current_time - last_image_send_time > 10:
                with _capture_lock:
                    image_path = _latest_image
                if image_path and os.path.exists(image_path):
                    threading.Thread(target=send_image_to_feishu, args=(image_path,), daemon=True).start()
                    last_image_send_time = current_time
                    print(f"\n📤 发送图片到飞书...")
            
            # 检查是否有新图片
            with _capture_lock:
                image_path = _latest_image
                capture_time = _last_capture_time
            
            if not image_path or not os.path.exists(image_path):
                time.sleep(0.1)
                continue
            
            # 控制检测频率
            current_time = time.time()
            if current_time - last_detect_time < DETECT_INTERVAL:
                time.sleep(0.05)
                continue
            
            last_detect_time = current_time
            
            # 极速检测
            start_detect = time.time()
            detected, h_pos, v_pos, x, y, w, h = detect_person_fast(image_path)
            detect_ms = (time.time() - start_detect) * 1000
            
            # 显示状态
            status = "✅ 有人" if detected else "❌ 无人"
            pos_str = f"位置:{h_pos:+.2f}" if detected else ""
            print(f"\r{frame_count:3d} | {status} {pos_str} | 检测:{detect_ms:4.1f}ms | 图片:{(current_time-capture_time)*1000:5.0f}ms ago", end="", flush=True)
            
            # 决策移动
            if detected and (current_time - last_move_time) > MOVE_COOLDOWN:
                action = None
                angle = None
                
                if h_pos < -CENTER_THRESHOLD:
                    action = 'turn_left'
                    angle = 20
                    print(f"\n⬅️  左转 (h={h_pos:.2f})")
                elif h_pos > CENTER_THRESHOLD:
                    action = 'turn_right'
                    angle = 20
                    print(f"\n➡️  右转 (h={h_pos:.2f})")
                elif v_pos < 0.4:  # 人太远
                    action = 'move_forward'
                    print(f"\n⬆️  前进 (v={v_pos:.2f})")
                elif v_pos > 0.7:  # 人太近
                    action = 'move_backward'
                    print(f"\n⬇️  后退 (v={v_pos:.2f})")
                
                if action:
                    move_async(action, angle)
                    last_move_time = current_time
            
    except KeyboardInterrupt:
        print("\n\n🛑 停止极速跟随模式")
        move_async('stop')


if __name__ == '__main__':
    follow_fast()
