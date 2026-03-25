#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import time
import signal
import cv2
import numpy as np
import threading
from ultralytics import YOLO

# 导入宇树 SDK
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

# ==========================================
# 配置参数
# ==========================================
NETWORK_INTERFACE = sys.argv[1] if len(sys.argv) > 1 else "eth0"
MODEL_PATH = '/home/unitree/openclaw/skills/unitree-go2/yolov8n.pt'

# 视觉死区阈值（可根据实际灵敏度微调）
CENTER_THRESHOLD = 0.15     
FAR_Y2_THRESHOLD = 0.72     
CLOSE_Y2_THRESHOLD = 0.94   

# ==========================================
# 全局停止标志 (SIGTERM / SIGINT 均可触发)
# ==========================================
_shutdown = False

def _on_signal(signum, frame):
    global _shutdown
    _shutdown = True

def _sleep(duration):
    """可被 _shutdown 标志中断的 sleep，最大响应延迟 ~20ms。"""
    end = time.time() + duration
    while not _shutdown and time.time() < end:
        time.sleep(min(0.02, end - time.time()))

# ==========================================
# 全局控制状态 (在视觉和控制线程间共享)
# ==========================================
cmd_state = {
    "vx": 0.0,
    "vy": 0.0,
    "vyaw": 0.0,
    "last_update": 0.0
}
CMD_TIMEOUT = 0.5  # 如果0.5秒内没检测到人，自动刹车，防止飞车

def control_thread_task(sport_client):
    """
    独立控制线程：加入了一阶低通滤波 (EMA)，实现丝滑的起步和刹车。
    """
    print("🟢 [控制线程] 丝滑运动控制循环已启动 (20Hz)...")
    
    # 记录电机当前的实际速度
    real_vx = 0.0
    real_vyaw = 0.0
    
    # 核心参数：平滑因子 (0.0 到 1.0 之间)
    # 越小：起步刹车越柔和，但响应越迟钝；越大：越跟手，但容易踉跄。
    # 0.15 是一个非常适合机器狗的“油门缓冲”值
    SMOOTH_FACTOR = 0.15 

    try:
        while not _shutdown:
            # 1. 读取视觉给出的"目标速度"
            if time.time() - cmd_state["last_update"] > CMD_TIMEOUT:
                target_vx, target_vy, target_vyaw = 0.0, 0.0, 0.0
            else:
                target_vx = cmd_state["vx"]
                target_vy = cmd_state["vy"]
                target_vyaw = cmd_state["vyaw"]

            # 2. 核心魔法：速度平滑过渡 (当前速度缓慢向目标速度靠近)
            real_vx += SMOOTH_FACTOR * (target_vx - real_vx)
            real_vyaw += SMOOTH_FACTOR * (target_vyaw - real_vyaw)

            # 3. 死区处理（当速度极小时，强制归零，防止电机一直原地微弱震动）
            if abs(real_vx) < 0.02: real_vx = 0.0
            if abs(real_vyaw) < 0.05: real_vyaw = 0.0

            # 4. 发送指令
            if real_vx == 0.0 and target_vy == 0.0 and real_vyaw == 0.0:
                sport_client.StopMove()
            else:
                sport_client.Move(real_vx, target_vy, real_vyaw)
            
            _sleep(0.05)
            
    except Exception as e:
        print(f"❌ [控制线程] 异常: {e}")
    finally:
        sport_client.StopMove()


def main():
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    print("🔌 正在初始化底层通道 (仅执行一次)...", flush=True)
    ChannelFactoryInitialize(0, NETWORK_INTERFACE)

    # 1. 初始化视频客户端
    video_client = VideoClient()
    video_client.SetTimeout(3.0)
    video_client.Init()
    print("✅ 视频客户端初始化完成")

    # 2. 初始化运动客户端
    sport_client = SportClient()
    sport_client.SetTimeout(3.0)
    sport_client.Init()
    code, server_version = sport_client.GetServerApiVersion()
    if code != 0:
        print(f"❌ 无法连接到运动服务器, 错误码: {code}")
        sys.exit(1)
    print(f"✅ 运动客户端初始化完成 (API: {server_version})")

    # 3. 加载 YOLO 模型预热
    print("🚀 正在加载 YOLO 模型并热身...")
    model = YOLO(MODEL_PATH)
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    # 4. 启动后台控制线程
    ctrl_thread = threading.Thread(target=control_thread_task, args=(sport_client,), daemon=True)
    ctrl_thread.start()

    locked_id = None
    last_seen_time = time.time()
    print("STATUS: 跟随模式已启动，正在寻找目标", flush=True)

    try:
        while not _shutdown:
            # --- 极速图像获取 (直接拿内存里的二进制数据) ---
            code, data = video_client.GetImageSample()
            if code != 0:
                _sleep(0.01)
                continue
            
            # 内存级解码，省去存图时间
            np_arr = np.frombuffer(bytes(data), np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None: continue


            # --- GPU 极速追踪 (使用 BoT-SORT 算法分配 ID) ---
            # persist=True 告诉模型要在前后帧之间保持记忆
            # tracker="botsort.yaml" 是一个自带外观特征提取的强力追踪器，适合背影
            results = model.track(img, classes=[0], conf=0.45, persist=True, tracker="botsort.yaml", verbose=False)
            
            detected = False
            h_pos, y2_pos = 0.0, 0.0
            
            # 确保当前帧检测到了人，并且追踪器成功分配了 ID
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                
                # 1. 目标锁定逻辑：如果没有锁定目标，优先锁定画面中面积最大（最近）的人
                if locked_id is None:
                    max_area = 0
                    for box, obj_id in zip(boxes, ids):
                        x1, y1, x2, y2 = box
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            locked_id = obj_id
                    
                    if locked_id is not None:
                        print(f"🔒 [视觉锁定] 已锁定目标主人 ID: {int(locked_id)}")
                        last_seen_time = time.time()
                
                # 2. 目标提取逻辑：在画面中寻找我们锁定的那个 ID
                target_found_this_frame = False
                for box, obj_id in zip(boxes, ids):
                    if obj_id == locked_id:
                        x1, y1, x2, y2 = box
                        detected = True
                        target_found_this_frame = True
                        # 计算被锁定目标的坐标
                        h_pos = ((x1 + x2) / (2 * img.shape[1]) - 0.5) * 2
                        y2_pos = y2 / img.shape[0]
                        break
                
                # 3. 丢失保护机制：如果锁定的目标没在画面中
                if not target_found_this_frame:
                    if time.time() - last_seen_time > 3.0:  # 忍耐 3 秒
                         print("⚠️ [目标丢失] 锁定超时，已解除锁定，等待新目标...")
                         locked_id = None
                else:
                    last_seen_time = time.time() # 只要看到了，就刷新存活时间
            else:
                # 画面里连一个人都没有
                if locked_id is not None and (time.time() - last_seen_time > 3.0):
                     print("⚠️ [视野空白] 目标消失超时，已解除锁定...")
                     locked_id = None



            # --- 决策逻辑 (计算目标速度) ---
            target_vx, target_vyaw = 0.0, 0.0
            action_desc = "停止"
            
            if detected:
                # 1. 动态角速度（人偏离越远，目标角速度越大）
                # 注意方向：h_pos 为负（人在左），我们需要给正的角速度（左转）
                target_vyaw = -h_pos * 1.5
                target_vyaw = max(-0.8, min(0.8, target_vyaw)) # 限制最大转速
                
                if abs(h_pos) > CENTER_THRESHOLD:
                    action_desc = "转向"

                # 2. 动态线速度（允许边转边走，画弧线更自然）
                # 只要人没有偏离到屏幕最边缘（0.3 以内），就允许前进
                if abs(h_pos) < CENTER_THRESHOLD + 0.3:
                    if y2_pos < FAR_Y2_THRESHOLD:
                        target_vx = 1.0
                        action_desc = "前进/弧线"
                    elif y2_pos > CLOSE_Y2_THRESHOLD:
                        target_vx = -0.5
                        action_desc = "后退/弧线"

            # 更新全局状态（目标速度）
            cmd_state["vx"] = target_vx
            cmd_state["vyaw"] = target_vyaw
            cmd_state["last_update"] = time.time()
            
            if detected:
                print(f"👁️ 目标 [y2:{y2_pos:.2f}, h:{h_pos:.2f}] -> 目标指令: {action_desc} (vx:{target_vx:.2f}, vyaw:{target_vyaw:.2f})")
 

    except KeyboardInterrupt:
        pass
    finally:
        cmd_state["vx"] = 0.0
        cmd_state["vy"] = 0.0
        cmd_state["vyaw"] = 0.0
        cmd_state["last_update"] = 0.0
        _sleep(0.5)
        print("STATUS: 跟随模式已停止", flush=True)

if __name__ == "__main__":
    main()



'''
            # --- GPU 极速推理 ---
            results = model(img, classes=[0], conf=0.45, verbose=False)
            detected = False
            h_pos, y2_pos = 0.0, 0.0
            
            max_area = 0
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2-x1)*(y2-y1)
                    if area > max_area:
                        max_area = area
                        detected = True
                        h_pos = ((x1 + x2) / (2 * img.shape[1]) - 0.5) * 2
                        y2_pos = y2 / img.shape[0]
'''

