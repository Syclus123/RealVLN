import time
import math
import usb.core
import usb.util
from tuning import Tuning
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

def main():
    ChannelFactoryInitialize(0, "eth0") 
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if not dev:
        print("[错误] 未找到 ReSpeaker 设备！")
        return
    Mic_tuning = Tuning(dev)

    DEAD_ZONE = 15         
    # 设置一个固定的旋转速度 (0.8 rad/s 是一个很舒服、平稳的转身速度)
    FIXED_YAW_SPEED = 0.8  
    # 计算这个速度对应每秒转多少度 (0.8 * 180 / π ≈ 45.8度/秒)
    DEG_PER_SEC = math.degrees(FIXED_YAW_SPEED) 

    print("\n✅ 具备【动作记忆】的听声辨位已启动！请喊它一声测试。")

    try:
        while True:
            if Mic_tuning.is_voice() == 1:
                doa_angle = Mic_tuning.direction 

                # 1. 角度换算 (-180 到 180)
                if doa_angle > 180:
                    error_angle = doa_angle - 360 
                else:
                    error_angle = doa_angle       

                if abs(error_angle) > DEAD_ZONE:
                    # 2. 核心魔法：计算转完这个角度需要几秒钟？
                    # 时间 = 角度距离 / 旋转速度
                    turn_duration = abs(error_angle) / DEG_PER_SEC
                    
                    # 3. 确定旋转方向 (右手坐标系)
                    if error_angle > 0:
                        yaw_cmd = FIXED_YAW_SPEED  # 声音在右，往右转
                    else:
                        yaw_cmd = -FIXED_YAW_SPEED   # 声音在左，往左转

                    print(f"🎤 听到声音！方位: {doa_angle:03d}° | 预计闭眼旋转: {turn_duration:.2f} 秒")

                    # 4. 执行“死命令”：锁死在这个循环里，坚定地转完指定时间
                    start_time = time.time()
                    while time.time() - start_time < turn_duration:
                        sport_client.Move(0.0, 0.0, float(yaw_cmd))
                        time.sleep(0.02) # 高频维持底层通讯

                    # 5. 转完之后，精准刹车
                    sport_client.Move(0.0, 0.0, 0.0)
                    print("🎯 转身完毕，已面向声源！等待下一次呼唤...\n")
                    
                    # 💡 可选：转完后冷却 0.5 秒，防止麦克风收到自己刹车的震动噪音
                    time.sleep(0.5) 
                
            else:
                # 没人说话且不在执行旋转任务时，保持静止
                sport_client.Move(0.0, 0.0, 0.0)
            
            time.sleep(0.05) 

    except KeyboardInterrupt:
        print("\n[系统] 紧急制动...")
        sport_client.Move(0.0, 0.0, 0.0)

if __name__ == '__main__':
    main()
