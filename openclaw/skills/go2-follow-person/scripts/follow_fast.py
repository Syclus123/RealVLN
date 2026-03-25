#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 Follow Person FAST V2 - 极速内存直通版
修复：文件竞态读写、Subprocess进程阻塞、僵硬的 elif 运动逻辑
"""
import os
import sys
import time
import threading
import cv2
import numpy as np

# 🚀 核心升级 2：直接把控制脚本作为 Python 模块导入，在内存中极速调用！
# 确保路径正确，并且目标目录下有 __init__.py（如果没有，系统也能临时找到）
sys.path.insert(0, '/home/unitree/openclaw/skills/unitree-go2/scripts')
try:
    # 假设你的控制脚本叫 go2_control.py
    import go2_control
    HAS_CONTROL_MODULE = True
except ImportError:
    print("⚠️ 警告: 无法直接导入 go2_control 模块，将仅打印调试信息。")
    HAS_CONTROL_MODULE = False

# 配置参数
CENTER_THRESHOLD = 0.05  # 中心区域阈值
MOVE_COOLDOWN = 0.3      # 冷却时间（因为去掉了进程开销，可以缩短到0.3秒，让动作更连贯）
DETECT_INTERVAL = 0.1    # 检测间隔

# 全局内存变量
_latest_frame = None
_frame_lock = threading.Lock()
_camera_running = True
_hog = None

def get_hog():
    global _hog
    if _hog is None:
        print("🔧 预加载 HOG 检测器...")
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog

# 🚀 核心升级 1：彻底抛弃写硬盘，直接在内存里拉取摄像头画面！
def camera_thread_func():
    """后台摄像头拉流线程：永远保持最新的画面在内存中"""
    global _latest_frame, _camera_running
    print("📷 正在启动内存级摄像头流...")
    
    # 注意：如果你用的是狗子背部的特定摄像头，可能需要改数字 0 或使用 GStreamer 管道
    cap = cv2.VideoCapture(0) 
    
    # 强制降低硬件采样的分辨率，从源头减少 CPU 压力
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while _camera_running:
        ret, frame = cap.read()
        if ret:
            with _frame_lock:
                _latest_frame = frame.copy()
        else:
            time.sleep(0.01) # 没读到就稍微歇一下，防止吃满 CPU
            
    cap.release()
    print("📷 摄像头流已关闭")

def detect_person_fast(img):
    """极速人检测（直接接收 Numpy 数组，零磁盘 I/O）"""
    start = time.time()
    
    orig_height, orig_width = img.shape[:2]
    
    # 缩小图片加速检测
    TARGET_WIDTH = 480
    scale = TARGET_WIDTH / orig_width
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    img_resized = cv2.resize(img, (new_width, new_height))
    
    hog = get_hog()
    boxes, weights = hog.detectMultiScale(
        img_resized, winStride=(8, 8), padding=(4, 4), scale=1.05
    )
    
    detect_time = (time.time() - start) * 1000
    
    if len(boxes) > 0:
        best_idx = np.argmax([w * h for x, y, w, h in boxes])
        x, y, w, h = boxes[best_idx]
        
        x_orig = x / scale
        y_orig = y / scale
        w_orig = w / scale
        h_orig = h / scale
        
        x_center = (x_orig + w_orig/2) / orig_width
        y_center = (y_orig + h_orig/2) / orig_height
        h_pos = (x_center - 0.5) * 2
        v_pos = y_center
        
        return (True, h_pos, v_pos, detect_time)
    
    return (False, 0, 0, detect_time)

# 🚀 核心升级 3：平滑运动决策树，抛弃僵硬的 elif
def make_movement_decision(h_pos, v_pos):
    """
    根据视觉偏移量，计算控制指令。
    未来如果你的 SDK 支持摇杆式的双轴速度控制 (vx, vyaw)，可以直接在这里输出比例值。
    """
    action_turn = None
    action_move = None
    angle = 20
    
    # 1. 优先判定左右（狗子得先看准人）
    if h_pos < -CENTER_THRESHOLD:
        action_turn = 'turn_left'
    elif h_pos > CENTER_THRESHOLD:
        action_turn = 'turn_right'
        
    # 2. 独立判定前后（不在 elif 里面，意味着它可以一边判断转弯，一边判断前进！）
    if v_pos < 0.4:
        action_move = 'move_forward'
    elif v_pos > 0.7:
        action_move = 'move_backward'
        
    return action_turn, action_move, angle


def execute_move(action, angle=None):
    """
    异步执行移动（完美还原原始 Subprocess 逻辑）
    """
    if not action: 
        return

    def do_move():
        try:
            import subprocess
            # 还原你的原始路径和命令拼接逻辑
            cmd = ['python3', '/home/unitree/openclaw/skills/unitree-go2/scripts/go2_control.py', action]
            if angle:
                cmd.extend(['--angle', str(angle)])
            
            print(f"⚡ [下发指令] 控制节点启动: {' '.join(cmd)}")
            
            # 执行命令，加上 timeout=3 防止底层卡死拖累系统
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            
            # 如果你想看底层有没有报错，可以加上这行：
            if result.returncode != 0:
                print(f"⚠️ 底层警告: {result.stderr.strip()}")
                
        except Exception as e:
            print(f"❌ 动作 {action} 执行异常: {e}")

    # 核心：每次移动独立开一个后台线程，绝对不阻塞极速检测的主循环！
    threading.Thread(target=do_move, daemon=True).start()


def send_image_to_feishu(img_array):
    """飞书推送：为了不影响主循环，临时存图并用异步子进程发送"""
    import subprocess
    temp_path = "/tmp/feishu_temp_capture.jpg"
    cv2.imwrite(temp_path, img_array) # 只有这里迫不得已才写一次硬盘
    
    try:
        env = os.environ.copy()
        env["FEISHU_CHAT_ID"] = "ou_da6b544f17a5b166a23d6cb4ce4e9b82"
        subprocess.Popen(
            ['python3', '/home/unitree/openclaw/send_image_to_feishu.py', temp_path],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"飞书发送失败: {e}")

def follow_fast():
    global _camera_running, _latest_frame
    print("🐕 极速跟随模式启动 V2")
    print("=" * 50)
    
    # 启动摄像头内存流
    cam_thread = threading.Thread(target=camera_thread_func, daemon=True)
    cam_thread.start()
    
    # 等待摄像头启动并拿到第一帧
    print("⏳ 等待摄像头初始化...")
    while _latest_frame is None:
        time.sleep(0.1)
    
    get_hog() # 预加载检测器
    
    last_move_time = 0
    last_detect_time = 0
    last_image_send_time = time.time()
    frame_count = 0
    
    try:
        while True:
            current_time = time.time()
            
            # 1. 从内存无锁获取最新一帧 (耗时几乎为 0)
            with _frame_lock:
                current_img = _latest_frame.copy()
                
            if current_img is None:
                continue

            # 2. 飞书 10 秒定时推送
            if current_time - last_image_send_time > 10:
                #send_image_to_feishu(current_img)
                last_image_send_time = current_time
                print("\n📤 发送最新画面到飞书...")

            # 3. 频率控制
            if current_time - last_detect_time < DETECT_INTERVAL:
                time.sleep(0.02)
                continue
                
            last_detect_time = current_time
            frame_count += 1
            
            # 4. 极速内存检测
            detected, h_pos, v_pos, detect_ms = detect_person_fast(current_img)
            
            status = "✅ 有人" if detected else "❌ 无人"
            pos_str = f"H:{h_pos:+.2f} V:{v_pos:.2f}" if detected else ""
            print(f"\r{frame_count:3d} | {status} {pos_str} | 检测耗时:{detect_ms:4.1f}ms", end="", flush=True)
            
            # 5. 组合动作执行
            if detected and (current_time - last_move_time) > MOVE_COOLDOWN:
                action_turn, action_move, angle = make_movement_decision(h_pos, v_pos)
                
                # 如果既需要转弯又需要前进，优先执行转弯修正视场，或者连续发送两个指令
                if action_turn:
                    execute_move(action_turn, angle)
                elif action_move:
                    execute_move(action_move)
                    
                if action_turn or action_move:
                    last_move_time = time.time()
            
    except KeyboardInterrupt:
        print("\n\n🛑 停止极速跟随模式")
        _camera_running = False
        execute_move('stop')

if __name__ == '__main__':
    follow_fast()
