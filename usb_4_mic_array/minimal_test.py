import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

def main():
    # 1. 初始化底层通信通道 (如果你在狗的板载电脑上运行，通常是 eth0 或 lo)
    ChannelFactoryInitialize(0, "eth0")
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    print("========================================")
    print("✅ 节点初始化成功！")
    print("⚠️  请拿起遥控器，按两次 Start 键，确保狗处于【站立且可运动状态】")
    print("⏳ 3 秒后将开始下发运动指令，请注意安全...")
    print("========================================")
    
    time.sleep(3)

    print("🚀 发送指令：向左匀速旋转 (0.3 rad/s)")
    
    # 宇树底盘的特点：必须“持续不断”地高频发送运动指令，否则看门狗会立刻刹车
    start_time = time.time()
    while time.time() - start_time < 3.0: # 持续发 3 秒
        # Move 参数：vx (前后), vy (左右), vyaw (旋转角速度)
        sport_client.Move(0.0, 0.0, 0.3)
        time.sleep(0.02) # 以 50Hz 的频率发送指令

    print("🛑 发送指令：刹车停止")
    sport_client.Move(0.0, 0.0, 0.0)
    print("🏁 测试结束。")

if __name__ == '__main__':
    main()
