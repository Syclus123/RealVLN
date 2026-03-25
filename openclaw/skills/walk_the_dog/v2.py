#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
walk_the_dog.py — 自主漫步避障

行走：ObstaclesAvoidClient.Move() → 前进有碰撞保护，遇墙自动停
扫描旋转：SportClient.Move() → 绕过避障层，保证能转动
扫描试探：ObstaclesAvoidClient.Move() → 试走，被挡说明路不通

循环：前进 → 卡住 → 左旋一段 → 试走 → 不通继续旋 → 通了就走
"""

import sys
import time
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

# ==========================================
# 配置参数
# ==========================================
NETWORK_INTERFACE = "eth0"

WALK_SPEED = 0.5
TURN_SPEED = 0.8

# 卡住检测
STUCK_TIMEOUT = 1.5
STUCK_VX_THRESHOLD = 0.03

# 扫描参数
ROTATE_DURATION = 1.2       # 每次旋转多久（秒），约转 ~55°
PROBE_DURATION = 1.0        # 每次试探前进多久（秒）
PROBE_VX_THRESHOLD = 0.10   # 试探期间 vx 超过此值 = 路通了
MAX_SCAN_CYCLES = 8         # 最多转+试探几轮（覆盖 360°+）

# ==========================================
# 全局状态
# ==========================================
STATE_WALKING = "walking"
STATE_SCAN_ROTATE = "scan_rotate"
STATE_SCAN_PROBE = "scan_probe"

cmd_state = {
    "vx": 0.0,
    "vyaw": 0.0,
    "last_update": 0.0,
    "mode": STATE_WALKING,
    "phase_start": 0.0,
    "scan_cycles": 0,
    "use_sport": False,
}

actual_vel = {
    "vx": 0.0,
    "last_update": 0.0,
}

CMD_TIMEOUT = 0.5


def sport_state_handler(msg: SportModeState_):
    actual_vel["vx"] = msg.velocity[0]
    actual_vel["last_update"] = time.time()


# ==========================================
# 控制线程：根据 use_sport 标志选择客户端
# ==========================================
def control_thread_task(avoid_client, sport_client):
    print("🟢 [控制线程] 已启动 (20Hz)")

    real_vx = 0.0
    real_vyaw = 0.0
    SMOOTH_FACTOR = 0.30

    try:
        while True:
            if time.time() - cmd_state["last_update"] > CMD_TIMEOUT:
                target_vx, target_vyaw = 0.0, 0.0
            else:
                target_vx = cmd_state["vx"]
                target_vyaw = cmd_state["vyaw"]

            real_vx += SMOOTH_FACTOR * (target_vx - real_vx)
            real_vyaw += SMOOTH_FACTOR * (target_vyaw - real_vyaw)

            if abs(real_vx) < 0.02: real_vx = 0.0
            if abs(real_vyaw) < 0.05: real_vyaw = 0.0

            if cmd_state["use_sport"]:
                sport_client.Move(real_vx, 0.0, real_vyaw)
            else:
                avoid_client.Move(real_vx, 0.0, real_vyaw)

            time.sleep(0.05)

    except Exception as e:
        print(f"❌ [控制线程] 异常: {e}")
    finally:
        sport_client.StopMove()
        avoid_client.Move(0.0, 0.0, 0.0)


# ==========================================
# 主函数
# ==========================================
def main():
    print("🔌 正在初始化...")
    ChannelFactoryInitialize(0, NETWORK_INTERFACE)

    sport_client = SportClient()
    sport_client.SetTimeout(3.0)
    sport_client.Init()
    code, ver = sport_client.GetServerApiVersion()
    if code != 0:
        print(f"❌ 运动服务器连接失败: {code}")
        sys.exit(1)
    print(f"✅ SportClient 初始化完成 (API: {ver})")

    avoid_client = ObstaclesAvoidClient()
    avoid_client.SetTimeout(3.0)
    avoid_client.Init()

    print("🛡️ 正在开启 SDK 内置避障...")
    for _ in range(30):
        code, enabled = avoid_client.SwitchGet()
        if code == 0 and enabled:
            break
        avoid_client.SwitchSet(True)
        time.sleep(0.1)
    else:
        print("⚠️ SDK 避障开关未确认")

    avoid_client.UseRemoteCommandFromApi(True)
    time.sleep(0.3)
    code, enabled = avoid_client.SwitchGet()
    print(f"🛡️ SDK 避障: {'已开启' if enabled else '未开启'}")

    print("📡 正在订阅速度反馈...")
    state_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    state_sub.Init(sport_state_handler, 10)
    time.sleep(0.5)
    if (time.time() - actual_vel["last_update"]) >= 1.0:
        print("❌ 速度反馈不可用，退出")
        sys.exit(1)
    print("📡 速度反馈: 在线")

    ctrl_thread = threading.Thread(
        target=control_thread_task, args=(avoid_client, sport_client), daemon=True
    )
    ctrl_thread.start()

    cmd_state["vx"] = WALK_SPEED
    cmd_state["vyaw"] = 0.0
    cmd_state["mode"] = STATE_WALKING
    cmd_state["use_sport"] = False
    cmd_state["last_update"] = time.time()

    print(f"\n🐕 自主漫步已开启！速度 {WALK_SPEED}m/s")
    print("   前进(避障保护) → 卡住 → 左旋(SportClient) → 试走(避障保护) → 循环")
    print("   按 Ctrl+C 停止。\n")

    stuck_since = 0.0

    try:
        while True:
            now = time.time()
            vel_fresh = (now - actual_vel["last_update"]) < 1.0

            # ======== 行走模式 ========
            if cmd_state["mode"] == STATE_WALKING:
                cmd_state["vx"] = WALK_SPEED
                cmd_state["vyaw"] = 0.0
                cmd_state["use_sport"] = False
                cmd_state["last_update"] = now

                if vel_fresh and abs(actual_vel["vx"]) < STUCK_VX_THRESHOLD:
                    if stuck_since == 0.0:
                        stuck_since = now
                    elif now - stuck_since > STUCK_TIMEOUT:
                        cmd_state["mode"] = STATE_SCAN_ROTATE
                        cmd_state["phase_start"] = now
                        cmd_state["scan_cycles"] = 0
                        stuck_since = 0.0
                        print(f"🚧 卡住 → 开始左旋扫描")
                else:
                    stuck_since = 0.0

            # ======== 扫描-旋转阶段：用 SportClient 左旋 ========
            elif cmd_state["mode"] == STATE_SCAN_ROTATE:
                cmd_state["vx"] = 0.0
                cmd_state["vyaw"] = TURN_SPEED
                cmd_state["use_sport"] = True
                cmd_state["last_update"] = now

                if now - cmd_state["phase_start"] > ROTATE_DURATION:
                    cmd_state["mode"] = STATE_SCAN_PROBE
                    cmd_state["phase_start"] = now
                    cmd_state["scan_cycles"] += 1
                    print(f"   🔄 旋转完毕(第{cmd_state['scan_cycles']}轮) → 试探前进...")

            # ======== 扫描-试探阶段：用 ObstaclesAvoidClient 试走 ========
            elif cmd_state["mode"] == STATE_SCAN_PROBE:
                cmd_state["vx"] = WALK_SPEED
                cmd_state["vyaw"] = 0.0
                cmd_state["use_sport"] = False
                cmd_state["last_update"] = now

                elapsed = now - cmd_state["phase_start"]

                if elapsed > PROBE_DURATION:
                    if vel_fresh and actual_vel["vx"] > PROBE_VX_THRESHOLD:
                        cmd_state["mode"] = STATE_WALKING
                        stuck_since = 0.0
                        print(f"   ✅ 路通了！(vx: {actual_vel['vx']:.2f}) 恢复直行")
                    elif cmd_state["scan_cycles"] >= MAX_SCAN_CYCLES:
                        cmd_state["mode"] = STATE_WALKING
                        stuck_since = 0.0
                        print(f"   ⏱️ 已扫描 {MAX_SCAN_CYCLES} 轮，强制前进")
                    else:
                        cmd_state["mode"] = STATE_SCAN_ROTATE
                        cmd_state["phase_start"] = now
                        print(f"   ❌ 还是不通 (vx: {actual_vel['vx']:.3f}) → 继续旋转")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 正在安全停止...")
        cmd_state["vx"] = 0.0
        cmd_state["vyaw"] = 0.0
        cmd_state["use_sport"] = False
        cmd_state["last_update"] = 0.0
        time.sleep(0.5)
        sport_client.StopMove()
        avoid_client.Move(0.0, 0.0, 0.0)
        avoid_client.UseRemoteCommandFromApi(False)
        print("✅ 已停止")


if __name__ == "__main__":
    main()
