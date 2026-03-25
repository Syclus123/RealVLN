import time
import math
import usb.core
import usb.util
from tuning import Tuning # 导入 ReSpeaker 的调节模块

# 导入宇树 SDK2.0 的相关库
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

def main():
    # ==========================================
    # 1. 初始化机器狗运动客户端
    # ==========================================
    # 注意：如果在狗的板载电脑运行，网卡通常是 'eth0' 或 'lo'，请根据实际情况修改
    ChannelFactoryInitialize(0, "eth0") 
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()
    print("[系统] 宇树运动控制节点初始化成功。")

    # ==========================================
    # 2. 初始化 ReSpeaker 麦克风阵列
    # ==========================================
    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if dev:
        Mic_tuning = Tuning(dev)
        print("[系统] ReSpeaker 麦克风阵列连接成功。")
    else:
        print("[错误] 未找到 ReSpeaker 设备，请检查 USB 连接。")
        return

    # ==========================================
    # 3. 追踪控制参数设置
    # ==========================================
    Kp = 0.08             # P 控制器的比例系数，决定转弯有多猛 (建议从小慢慢调大)
    MAX_YAW_SPEED = 0.5    # 最大旋转角速度 (rad/s)，防止狗原地疯狂打转
    DEAD_ZONE = 15         # 角度死区 (度)：误差在正负15度以内就不动了，防止狗头频繁抽搐

    print("[系统] 开始进入听声辨位模式...")

    try:
        while True:
            # 只有当麦克风检测到真实的人声 (Voice Activity Detection) 时才处理
            # 防止风声或狗底盘的电机噪音触发转向
            if Mic_tuning.is_voice() == 1:
                doa_angle = Mic_tuning.direction # 获取 0-359 的角度

                # 步骤 A：将 0~359 度的坐标系映射为 -180 ~ 180 度的偏差角
                # 假设 0 度是正前方。声音在右边是 1~179度，在左边是 181~359度。
                if doa_angle > 180:
                    error_angle = doa_angle - 360 # 转换为负数角 (左边)
                else:
                    error_angle = doa_angle       # 正数角 (右边)

                # 步骤 B & C：死区判断与增强型 P 控制计算
                if abs(error_angle) > DEAD_ZONE:
                    # 1. 基础 P 控制计算
                    target_yaw_speed = -error_angle * Kp 
                    
                    # 2. 突破底层死区：设置一个最小起步速度 (例如 0.2 rad/s)
                    MIN_YAW_SPEED = 0.25 
                    if target_yaw_speed > 0 and target_yaw_speed < MIN_YAW_SPEED:
                        target_yaw_speed = MIN_YAW_SPEED
                    elif target_yaw_speed < 0 and target_yaw_speed > -MIN_YAW_SPEED:
                        target_yaw_speed = -MIN_YAW_SPEED

                    # 3. 安全限幅，防止转得太猛把狗甩飞
                    target_yaw_speed = max(min(target_yaw_speed, MAX_YAW_SPEED), -MAX_YAW_SPEED)

                    # 4. 强制转换为 float 类型，确保 SDK 能够正确解析
                    target_yaw_speed = float(target_yaw_speed)

                    # 发送运动指令: Move(vx, vy, vyaw)
                    sport_client.Move(0.0, 0.0, target_yaw_speed)
                    print(f"🎤 听到声音！偏差: {error_angle:4d}° | 狗转向速度: {target_yaw_speed: .2f} rad/s")

                else:
                    # 已经在正前方死区内，刹车停止
                    sport_client.Move(0.0, 0.0, 0.0)
                    print(f"🎯 已经正对声源 (角度 {doa_angle}°)，保持不动。")
            else:
                # 没人说话时，保持静止
                # 注意：如果狗在走路，这里可以不发指令，或者发 (0,0,0) 刹车
                sport_client.Move(0.0, 0.0, 0.0)
            
            # 控制频率，20Hz 左右足够了
            time.sleep(0.05) 

    except KeyboardInterrupt:
        # 按 Ctrl+C 退出时，务必发指令让狗停下
        sport_client.Move(0.0, 0.0, 0.0)
        print("\n[系统] 已退出听声辨位模式，机器人停止。")

if __name__ == '__main__':
    main()
