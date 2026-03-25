#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 Follow Person FAST - YOLOv8 算力全开版
优化：彻底移除 HOG，接入 Ultralytics YOLOv8 极速模型
修复：文件竞态读写、进程阻塞、僵硬运动逻辑 (保持上个版本优化)
"""
import os
import sys
import time
import threading
import cv2
import numpy as np

# [🔥 YOLOv8 升级 1] 导入 Ultralytics YOLO 库
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    print("❌ 错误: 找不到 ultralytics 库。请运行 'pip3 install ultralytics'")
    HAS_YOLO = False
    sys.exit(1)

# 🚀 [保持 V2 优化] 模块化导入控制脚本
sys.path.insert(0, '/home/unitree/openclaw/skills/unitree-go2/scripts')
try:
    import go2_control
    HAS_CONTROL_MODULE = True
except ImportError:
    print("⚠️ 警告: 无法直接导入 go2_control 模块，将仅打印调试信息。")
    HAS_CONTROL_MODULE = False


# 配置参数
CENTER_THRESHOLD = 0.2  # 中心区域阈值
# 由于 YOLOv8 极快，我们可以再次缩短移动冷却时间，让狗子更“跟手”
MOVE_COOLDOWN = 0.5      # 移动冷却时间 (秒) 
DETECT_INTERVAL = 0.05   # 检测间隔 (秒) - 从 0.1s 加快到 0.05s (20Hz检测)

# 全局内存变量
_latest_frame = None
_frame_lock = threading.Lock()
_camera_running = True
_model = None

# [🔥 YOLOv8 升级 2] 预加载模型 (关键！绝对不能在主循环里临时加载)
def get_model():
    global _model
    if _model is None:
        print("🔧 预加载 YOLOv8n 模型到 Orin NX GPU...")
        # 我们使用最小的 nano 模型以获得极致速度
        # 在 Orin NX 上，系统会自动使用 TensorRT (如果配置好) 或 CUDA 加速
        _model = YOLO('yolov8n.pt') 
        
        # 为了获得极致速度，可以建议将模型编译为 TensorRT: 
        # _model.export(format='engine') # 运行一次即可编译
        # _model = YOLO('yolov8n.engine')
        
        print("🔧 YOLO 模型加载完成。")
    return _model

# 🚀 [保持 V2 优化] 内存直通摄像头流
# 替换原本的 camera_thread_func 函数
def camera_thread_func():
    """后台摄像头拉流线程：使用原有的 capture.py 抓图"""
    global _latest_frame, _camera_running
    print("📷 正在启动 Go2 专属摄像头流...")
    
    import subprocess
    CAPTURE_PATH = os.path.expanduser("~/.openclaw/media/go2_camera.jpg")
    
    while _camera_running:
        try:
            # 调用你原有的抓图脚本
            subprocess.run(
                ['python3', '/home/unitree/openclaw/skills/go2-vision/scripts/capture.py'],
                capture_output=True, timeout=5
            )
            
            # 抓取成功后，立刻读取到内存中供 YOLO 使用
            if os.path.exists(CAPTURE_PATH):
                # 用 cv2 读入内存
                frame = cv2.imread(CAPTURE_PATH)
                if frame is not None:
                    with _frame_lock:
                        _latest_frame = frame.copy()
        except Exception as e:
            print(f"拍照线程报错: {e}")
            
        # 给底层硬件喘口气，0.2秒拍一次 (5fps)
        time.sleep(0.2) 
        
    print("📷 摄像头流已关闭")


# [🔥 YOLOv8 升级 3] 极速检测函数 (彻底替换原 HOG 逻辑)
def detect_person_fast(img):
    """
    基于 YOLOv8 的极速人检测 (内存直通)
    返回: (detected, h_pos, v_pos, detect_time_ms)
    """
    start = time.time()
    
    orig_height, orig_width = img.shape[:2]
    
    # 拿到预加载的模型
    model = get_model()
    
    # 运行推理。verbose=False 禁用日志打印，classes=[0] 仅检测'人'
    # conf=0.5 设定阈值防止误检 (狗把椅子认成人的情况)
    # stream=True 使用迭代器，更节省内存
    results = model(img, classes=[0], conf=0.5, verbose=False, stream=True)
    
    detected = False
    h_pos = 0.0
    size_ratio = 0.0
    max_area = 0
    
    # YOLO 返回的是列表，我们需要遍历它
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            break # 如果这一帧没有检测到任何人，跳过
        
        # 优化：如果我们检测到了多个人，我们需要选择“最大”的那个框作为跟随目标
        # 因为最大的框通常代表离狗子最近的人
        for box in boxes:
            # 拿到 xyxy 格式坐标
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area > max_area:
                max_area = area
                detected = True
                
                # 计算中心点坐标
                #x_center = (x1 + x2) / 2
                #y_center = (y1 + y2) / 2
                
                # 归一化位置 (完全兼容上个版本的决策逻辑)
                #h_pos = (x_center / orig_width - 0.5) * 2 # -1 (最左) 到 1 (最右)
                #v_pos = y_center / orig_height            # 0 (最顶) 到 1 (最底)
		# --- 修改前的这三行 ---
                # x_center = (x1 + x2) / 2
                # y_center = (y1 + y2) / 2
                # v_pos = y_center / orig_height 
                
                # --- 修改后的代码 ---
                x_center = (x1 + x2) / 2
                h_pos = (x_center / orig_width - 0.5) * 2  # 左右依然用中心点
                
                # 【关键核心】用人体的“高度”占整个画面高度的比例，来精准测距！
                size_ratio = h / orig_height  
                
    # 函数最后的 return 也要改一下名字
    detect_time_ms = (time.time() - start) * 1000
    return (detected, h_pos, size_ratio, detect_time_ms)


# 🚀 [保持 V2 优化] 运动决策与组合运动
def make_movement_decision(h_pos, size_ratio):
    """
    智能运动决策：根据偏差大小，动态决定转头幅度和前后移动
    """
    action_turn = None
    action_move = None
    
    # 【动态角度计算】(比例控制 P-Control)
    # 偏得越多，转得越猛；偏得越少，微调即可。
    # h_pos 的绝对值在 0.20 到 1.0 之间。我们乘一个系数来计算角度。
    base_angle = int(abs(h_pos) * 25) 
    # 限制最大转角为 25 度，最小微调为 5 度
    angle = max(5, min(25, base_angle))
    
    # 1. 左右判定 (死区由 0.05 扩大到了 0.20)
    if h_pos < -CENTER_THRESHOLD:
        action_turn = 'turn_left'
        print(f"\n⬅️ 目标在左侧，微调 {angle} 度")
    elif h_pos > CENTER_THRESHOLD:
        action_turn = 'turn_right'
        print(f"\n➡️ 目标在右侧，微调 {angle} 度")
        
    # 2. 前后判定 (同样拉开死区距离，防止频繁前后抽搐)
    # 假设 v_pos 在 0.4 到 0.7 之间是“舒适距离”，狗子保持静止
    if size_ratio < 0.6:     # 人在画面很偏上方（说明人离得远）
        action_move = 'move_forward'
        print("\n⬆️ 目标太远，前进")
    elif size_ratio > 0.85:   # 人在画面很偏下方（说明人快贴脸了）
        action_move = 'move_backward'
        print("\n⬇️ 目标贴脸，后退防守")
        
    return action_turn, action_move, angle



# 🚀 [参考你的逻辑] 稳定的 Subprocess 异步调用
def execute_move(action, angle=None):
    """
    异步移动 (完美还原你的 Subprocess 逻辑，保持容错性)
    """
    if not action: return

    def do_move():
        try:
            import subprocess
            # 还原你的原始路径和命令拼接
            cmd = ['python3', '/home/unitree/openclaw/skills/unitree-go2/scripts/go2_control.py', action]
            if angle:
                cmd.extend(['--angle', str(angle)])
            
            print(f"⚡ [直调] {' '.join(cmd)}")
            
            # 超时增加到 3 秒防止底层卡死
            subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        except Exception as e:
            print(f"❌ 移动异常: {e}")

    # 每次独立开一个守护线程，绝对不阻塞主大脑！
    threading.Thread(target=do_move, daemon=True).start()

# 🚀 [保持 V2 优化] 飞书推送
def send_image_to_feishu(img_array):
    """飞书推送：临时存图异步发送"""
    import subprocess
    temp_path = "/tmp/feishu_temp_capture.jpg"
    cv2.imwrite(temp_path, img_array) 
    
    try:
        env = os.environ.copy()
        env["FEISHU_CHAT_ID"] = "ou_da6b544f17a5b166a23d6cb4ce4e9b82"
        # 飞书也使用 Popen 彻底异步，防止写硬盘阻塞
        subprocess.Popen(
            ['python3', '/home/unitree/openclaw/send_image_to_feishu.py', temp_path],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except:
        pass

def follow_fast():
    global _camera_running, _latest_frame
    print("🐕 极速跟随模式启动 - YOLOv8 算力全开版")
    print("=" * 60)
    print(f"🔍 检测速率: {1/DETECT_INTERVAL:.1f}Hz")
    
    # 启动摄像头内存流
    cam_thread = threading.Thread(target=camera_thread_func, daemon=True)
    cam_thread.start()
    
    # 等待摄像头
    while _latest_frame is None:
        time.sleep(0.1)
    
    # 预加载模型到 GPU (第一次耗时较长)
    get_model() 
    
    last_move_time = 0
    last_detect_time = 0
    last_image_send_time = time.time()
    frame_count = 0
    
    try:
        while True:
            current_time = time.time()
            
            with _frame_lock:
                current_img = _latest_frame.copy()
                
            if current_img is None: continue

            # 定时推送飞书
            if current_time - last_image_send_time > 10:
                send_image_to_feishu(current_img)
                last_image_send_time = current_time
                print("\n📤 发送最新 YOLO 识别画面到飞书...")

            # 频率控制
            if current_time - last_detect_time < DETECT_INTERVAL:
                time.sleep(0.01)
                continue
                
            last_detect_time = current_time
            frame_count += 1
            
            # [🔥 YOLOv8 升级 4] 极速检测
            detected, h_pos, size_ratio, detect_ms = detect_person_fast(current_img)
            
            status = "✅ 有人" if detected else "❌ 无人"
            pos_str = f"H:{h_pos:+.2f} V:{size_ratio:.2f}" if detected else ""
            # 注意：在 Orin NX 上，这个耗时应该只有 10-20ms 左右！
            print(f"\r{frame_count:3d} | {status} {pos_str} | 推理耗时:{detect_ms:4.1f}ms", end="", flush=True)
            
            # 组合动作执行
            if detected and (current_time - last_move_time) > MOVE_COOLDOWN:
                action_turn, action_move, angle = make_movement_decision(h_pos, size_ratio)
                
                # 优先修正转向，以便把人框在视野中心，然后再前进/后退
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
