#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
walk_the_dog.py — 自主漫步避障

行走：ObstaclesAvoidClient.Move() → 前进有碰撞保护
卡住：检测到 actual_vx ≈ 0
后退：SportClient 后退 0.5s 拉开距离（防止旋转时身体蹭墙）
旋转：SportClient 低速左旋（平滑无抖动）
试探：ObstaclesAvoidClient 尝试前进，检查速度
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

WALK_SPEED = 0.4            # 前进速度（稍慢，给 SDK 更多刹车余量）
TURN_SPEED = 0.5            # 旋转速度（降低，减少抖动）
BACKUP_SPEED = -0.2         # 后退速度（轻柔后退）

# 卡住检测
STUCK_TIMEOUT = 1.0         # 检测到卡住的时间缩短（更早反应）
STUCK_VX_THRESHOLD = 0.03

# 扫描参数
BACKUP_DURATION = 0.6       # 后退时长（拉开 ~12cm 间距，够机器狗安全旋转）
ROTATE_DURATION = 1.2       # 每次旋转时长（~35°）
PROBE_DURATION = 1.0        # 试探前进时长
PROBE_VX_THRESHOLD = 0.10   # 试探期间 vx 超过此值 = 路通了
MAX_SCAN_CYCLES = 12        # 最多试几轮（覆盖 360°+）

# ==========================================
# 全局状态
# ==========================================
STATE_WALKING = "walking"
STATE_SCAN_BACKUP = "scan_backup"
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
start_time = 0.0


def sport_state_handler(msg: SportModeState_):
    actual_vel["vx"] = msg.velocity[0]
    actual_vel["last_update"] = time.time()


def log(msg):
    elapsed = time.time() - start_time
    vx = actual_vel["vx"]
    mode = cmd_state["mode"]
    cvx = cmd_state["vx"]
    cvyaw = cmd_state["vyaw"]
    client = "sport" if cmd_state["use_sport"] else "avoid"
    print(f"[{elapsed:7.1f}s] [{mode:<12}] [{client}] cmd(vx={cvx:+.2f} vyaw={cvyaw:+.2f}) actual_vx={vx:+.3f} | {msg}")


# ==========================================
# 控制线程
# ==========================================
def control_thread_task(avoid_client, sport_client):
    real_vx = 0.0
    real_vyaw = 0.0
    SMOOTH_FACTOR = 0.25
    last_use_sport = None

    try:
        while True:
            if time.time() - cmd_state["last_update"] > CMD_TIMEOUT:
                target_vx, target_vyaw = 0.0, 0.0
            else:
                target_vx = cmd_state["vx"]
                target_vyaw = cmd_state["vyaw"]

            use_sport = cmd_state["use_sport"]

            # 切换客户端时重置 EMA，避免旧速度残留导致抖动
            if use_sport != last_use_sport:
                real_vx = 0.0
                real_vyaw = 0.0
                last_use_sport = use_sport

            real_vx += SMOOTH_FACTOR * (target_vx - real_vx)
            real_vyaw += SMOOTH_FACTOR * (target_vyaw - real_vyaw)

            if abs(real_vx) < 0.02: real_vx = 0.0
            if abs(real_vyaw) < 0.05: real_vyaw = 0.0

            if use_sport:
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
    global start_time
    start_time = time.time()

    print("🔌 正在初始化...")
    ChannelFactoryInitialize(0, NETWORK_INTERFACE)

    sport_client = SportClient()
    sport_client.SetTimeout(3.0)
    sport_client.Init()
    code, ver = sport_client.GetServerApiVersion()
    if code != 0:
        print(f"❌ 运动服务器连接失败: {code}")
        sys.exit(1)
    print(f"✅ SportClient (API: {ver})")

    avoid_client = ObstaclesAvoidClient()
    avoid_client.SetTimeout(3.0)
    avoid_client.Init()

    print("🛡️ 开启 SDK 避障...")
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

    print("📡 订阅速度反馈...")
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

    print(f"\n🐕 配置: 行走={WALK_SPEED}m/s 旋转={TURN_SPEED}rad/s 后退={BACKUP_SPEED}m/s")
    print(f"   卡住检测={STUCK_TIMEOUT}s 后退={BACKUP_DURATION}s 旋转={ROTATE_DURATION}s 试探={PROBE_DURATION}s")
    print(f"   机器狗尺寸: 70cm×30cm, 旋转半径 ~35cm")
    print(f"   按 Ctrl+C 停止。\n")

    log("开始行走")

    stuck_since = 0.0
    last_status_log = 0.0

    try:
        while True:
            now = time.time()
            vel_fresh = (now - actual_vel["last_update"]) < 1.0

            # 定期状态日志（行走中每 3 秒一次）
            if cmd_state["mode"] == STATE_WALKING and now - last_status_log > 3.0:
                stuck_dur = f" 卡住计时={now - stuck_since:.1f}s" if stuck_since > 0 else ""
                log(f"正常行走中{stuck_dur}")
                last_status_log = now

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
                        cmd_state["mode"] = STATE_SCAN_BACKUP
                        cmd_state["phase_start"] = now
                        cmd_state["scan_cycles"] = 0
                        stuck_since = 0.0
                        log(f"卡住 {STUCK_TIMEOUT}s! → 先后退拉开距离")
                else:
                    if stuck_since > 0.0:
                        log(f"卡住计时重置（vx 恢复: {actual_vel['vx']:.3f}）")
                    stuck_since = 0.0

            # ======== 扫描-后退阶段：拉开距离防止旋转蹭墙 ========
            elif cmd_state["mode"] == STATE_SCAN_BACKUP:
                cmd_state["vx"] = BACKUP_SPEED
                cmd_state["vyaw"] = 0.0
                cmd_state["use_sport"] = True
                cmd_state["last_update"] = now

                if now - cmd_state["phase_start"] > BACKUP_DURATION:
                    cmd_state["mode"] = STATE_SCAN_ROTATE
                    cmd_state["phase_start"] = now
                    log(f"后退完毕 → 开始左旋")

            # ======== 扫描-旋转阶段 ========
            elif cmd_state["mode"] == STATE_SCAN_ROTATE:
                cmd_state["vx"] = 0.0
                cmd_state["vyaw"] = TURN_SPEED
                cmd_state["use_sport"] = True
                cmd_state["last_update"] = now

                if now - cmd_state["phase_start"] > ROTATE_DURATION:
                    cmd_state["mode"] = STATE_SCAN_PROBE
                    cmd_state["phase_start"] = now
                    cmd_state["scan_cycles"] += 1
                    log(f"旋转完毕(第{cmd_state['scan_cycles']}/{MAX_SCAN_CYCLES}轮) → 试探前进")

            # ======== 扫描-试探阶段 ========
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
                        log(f"✅ 路通了! 恢复直行")
                    elif cmd_state["scan_cycles"] >= MAX_SCAN_CYCLES:
                        cmd_state["mode"] = STATE_WALKING
                        stuck_since = 0.0
                        log(f"⏱️ 已扫描 {MAX_SCAN_CYCLES} 轮, 强制前进")
                    else:
                        cmd_state["mode"] = STATE_SCAN_BACKUP
                        cmd_state["phase_start"] = now
                        log(f"❌ 不通 → 后退+继续旋转")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print(f"\n🛑 正在安全停止... (运行了 {time.time() - start_time:.0f}s)")
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
