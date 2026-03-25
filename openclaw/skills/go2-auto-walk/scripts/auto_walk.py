#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import cv2
import torch
import numpy as np
import threading
import argparse

# 导入宇树 SDK
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

# ==========================================
# 🚀 狂暴扫地僧：终极参数配置
# ==========================================
NETWORK_INTERFACE = "eth0"

# 速度参数
MAX_SPEED = 0.35          # 畅通时的最大直线速度
CREEP_SPEED = 0.15        # ⚠️ 强迫抬腿速度！逼迫狗子真正迈开腿转弯
TURN_SPEED = 0.85         # ⚠️ 狂暴转向！坚决逃离障碍物
RECOVERY_SPEED = -0.15    # 死胡同倒车速度

# 深度阈值 (针对 Go2 低矮视角的校准)
OBSTACLE_THRESHOLD = 110  
DANGER_THRESHOLD = 140    

cmd_state = {
    "vx": 0.0,
    "vyaw": 0.0,
    "last_update": time.time(), # 初始时间
    "active": True,
    "ai_ready": False  # <--- 新增这行：用来告诉小脑“我还在加载，先别动”
}
CMD_TIMEOUT = 0.5  # 0.5秒没有视觉更新，自动刹车

# ==========================================
# 🦴 小脑：丝滑运动控制线程
# ==========================================
def control_thread_task(sport_client):
    print("🟢 [底盘小脑] 丝滑运动控制循环已启动 (20Hz)...")
    
    real_vx = 0.0
    real_vyaw = 0.0
    SMOOTH_FACTOR = 0.3  # ⚠️ 调高平滑响应，让动作更干脆
    
    try:
        while cmd_state["active"]:
            # ⚠️ 1. 握手协议：如果大脑还没加载完，小脑就在原地静静等待，绝不发指令！
            if not cmd_state["ai_ready"]:
                time.sleep(0.1)
                continue

            # 2. 读取视觉目标速度 & 超时安全保护
            if time.time() - cmd_state["last_update"] > CMD_TIMEOUT:
                target_vx, target_vyaw = 0.0, 0.0
            else:
                target_vx = cmd_state["vx"]
                target_vyaw = cmd_state["vyaw"]

            # 3. 速度平滑过渡 (EMA)
            real_vx += SMOOTH_FACTOR * (target_vx - real_vx)
            real_vyaw += SMOOTH_FACTOR * (target_vyaw - real_vyaw)

            # 4. 死区处理，防止原地微震
            if abs(real_vx) < 0.02: real_vx = 0.0
            if abs(real_vyaw) < 0.05: real_vyaw = 0.0

            # ⚠️ 5. 核心修复：永远不要在循环里调 StopMove()！
            # 即使速度是 0，也发送 Move(0,0,0)，保持底盘引擎怠速运转！
            sport_client.Move(real_vx, 0.0, real_vyaw)
            
            time.sleep(0.05) # 20Hz

            
    except Exception as e:
        print(f"❌ [控制线程] 异常: {e}")
    finally:
        sport_client.StopMove()
        print("✅ [底盘小脑] 已安全锁死")

# ==========================================
# 🧠 大脑：MiDaS 深度感知线程
# ==========================================
def perception_thread(video_client, show_cv=False):
    print("🟢 [深度大脑] 正在加载 MiDaS 轻量级深度模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform

    print("🟢 [深度大脑] 感知引擎启动就绪！")

    # 👇 新增这两行：正式接管控制权！
    cmd_state["ai_ready"] = True
    cmd_state["last_update"] = time.time() 

    while cmd_state["active"]:
        start_time = time.time()
        
        # 获取图像
        code, data = video_client.GetImageSample()
        if code != 0:
            time.sleep(0.01)
            continue
            
        np_arr = np.frombuffer(bytes(data), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None: continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 深度推理
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False,
            ).squeeze()
        
        depth_map = cv2.normalize(prediction.cpu().numpy(), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # ⚠️ 针对 Go2 的核心视野裁剪：抬高视线，砍掉地板！
        h, w = depth_map.shape
        roi_top = int(h * 0.15)      
        roi_bottom = int(h * 0.60)   
        w_third = w // 3
        
        # 使用 95% 分位数，对细长障碍物（桌腿）更敏感
        left_val = np.percentile(depth_map[roi_top:roi_bottom, :w_third], 95)
        center_val = np.percentile(depth_map[roi_top:roi_bottom, w_third:2*w_third], 95)
        right_val = np.percentile(depth_map[roi_top:roi_bottom, 2*w_third:], 95)

        # 🚀 避障决策大脑
        target_vx, target_vyaw = 0.0, 0.0
        state_str = "停止"

        if center_val > DANGER_THRESHOLD and left_val > DANGER_THRESHOLD and right_val > DANGER_THRESHOLD:
            target_vx = RECOVERY_SPEED
            target_vyaw = TURN_SPEED 
            state_str = "🚨 死胡同! 紧急倒车"
        elif center_val < OBSTACLE_THRESHOLD:
            target_vx = MAX_SPEED
            # ⚠️ 限制居中微调的幅度，防止在开阔地带“画龙”
            centering_yaw = (left_val - right_val) * 0.001
            target_vyaw = np.clip(centering_yaw, -0.2, 0.2)
            state_str = "⬆️ 畅通直行"
        else:
            target_vx = CREEP_SPEED 
            if left_val < right_val:
                target_vyaw = TURN_SPEED
                state_str = "⬅️ 坚决左转"
            else:
                target_vyaw = -TURN_SPEED
                state_str = "➡️ 坚决右转"

        # ⚠️ 绝不能丢的两行：刷新全局指令和心跳包时间
        cmd_state["vx"] = target_vx
        cmd_state["vyaw"] = target_vyaw
        cmd_state["last_update"] = time.time()  # <--- 这就是小脑保持活跃的心跳！

        # 终端打印
        fps = 1.0 / (time.time() - start_time)
        print(f"\r\033[K[感知 {fps:.1f}Hz] {state_str} (vx:{target_vx:.2f}, vyaw:{target_vyaw:.2f}) | 浓度(左:{left_val:.0f} 中:{center_val:.0f} 右:{right_val:.0f})", end="", flush=True)

        if show_cv:
            cv2.imshow("Go2 Cyber Depth", cv2.applyColorMap(depth_map, cv2.COLORMAP_JET))
            if cv2.waitKey(1) & 0xFF == ord('q'): cmd_state["active"] = False

# ==========================================
# 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("🐕 Go2 Auto Roam - 狂暴扫地僧 (终极版)")
    print("=" * 60)

    ChannelFactoryInitialize(0, NETWORK_INTERFACE)
    video_client = VideoClient()
    video_client.Init()
    sport_client = SportClient()
    sport_client.Init()
    
    # 启动双线程
    t_ctrl = threading.Thread(target=control_thread_task, args=(sport_client,), daemon=True)
    t_ctrl.start()
    
    time.sleep(0.5)
    
    t_perc = threading.Thread(target=perception_thread, args=(video_client, args.show), daemon=True)
    t_perc.start()
    
    try:
        while cmd_state["active"]:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 收到退出指令，准备安全降落...")
        cmd_state["active"] = False
        cmd_state["vx"] = 0.0
        cmd_state["vyaw"] = 0.0
        cmd_state["last_update"] = 0.0
        time.sleep(0.5) 

if __name__ == "__main__":
    main()
