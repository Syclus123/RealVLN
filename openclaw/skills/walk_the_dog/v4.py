#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
walk_the_dog.py — 自主漫步避障

全程只用 ObstaclesAvoidClient，不再用 SportClient 做运动。
避障系统始终开启，所有运动指令都通过避障层，不会产生冲突和颤抖。

行走：avoid_client.Move(vx, 0, 0)
后退弧线：avoid_client.Move(-vx, 0, vyaw) — 边后退边转，避障不拦后退方向
试探：avoid_client.Move(vx, 0, 0) — 测试新方向

关键发现：
- SwitchSet(False) 会让 SportClient 指令也失效
- SportClient + 避障开启 → 两个系统打架 → 颤抖
- 只用 avoid_client → 单一控制源 → 无冲突
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

WALK_SPEED = 0.4
ARC_BACK_SPEED = -0.15      # 弧线后退速度
ARC_TURN_SPEED = 0.5        # 弧线转弯速度
ARC_DURATION = 2.0           # 每次弧线时长（后退~30cm，转~57°）

STUCK_TIMEOUT = 1.0
STUCK_VX_THRESHOLD = 0.03

PROBE_DURATION = 1.0
PROBE_VX_THRESHOLD = 0.10
MAX_SCAN_CYCLES = 8          # 最多 8 轮弧线（覆盖 ~456°）

# ==========================================
# 全局状态
# ==========================================
STATE_WALKING = "walking"
STATE_SCAN_ARC = "scan_arc"
STATE_SCAN_PROBE = "scan_probe"

cmd_state = {
    "vx": 0.0,
    "vyaw": 0.0,
    "last_update": 0.0,
    "mode": STATE_WALKING,
    "phase_start": 0.0,
    "scan_cycles": 0,
}

actual_vel = {
    "vx": 0.0,
    "vyaw": 0.0,
    "last_update": 0.0,
}

CMD_TIMEOUT = 0.5
start_time = 0.0
last_detail_log = 0.0


def sport_state_handler(msg: SportModeState_):
    actual_vel["vx"] = msg.velocity[0]
    actual_vel["vyaw"] = msg.velocity[2]
    actual_vel["last_update"] = time.time()


def log(msg):
    elapsed = time.time() - start_time
    avx = actual_vel["vx"]
    avyaw = actual_vel["vyaw"]
    mode = cmd_state["mode"]
    cvx = cmd_state["vx"]
    cvyaw = cmd_state["vyaw"]
    print(
        f"[{elapsed:7.1f}s] [{mode:<12}] "
        f"cmd(vx={cvx:+.2f} vyaw={cvyaw:+.2f}) "
        f"actual(vx={avx:+.3f} vyaw={avyaw:+.3f}) "
        f"| {msg}"
    )


# ==========================================
# 控制线程：只用 avoid_client
# ==========================================
def control_thread_task(avoid_client):
    real_vx = 0.0
    real_vyaw = 0.0
    SMOOTH = 0.30

    try:
        while True:
            if time.time() - cmd_state["last_update"] > CMD_TIMEOUT:
                tvx, tvyaw = 0.0, 0.0
            else:
                tvx = cmd_state["vx"]
                tvyaw = cmd_state["vyaw"]

            real_vx += SMOOTH * (tvx - real_vx)
            real_vyaw += SMOOTH * (tvyaw - real_vyaw)

            if abs(real_vx) < 0.01:
                real_vx = 0.0
            if abs(real_vyaw) < 0.03:
                real_vyaw = 0.0

            avoid_client.Move(real_vx, 0.0, real_vyaw)
            time.sleep(0.05)

    except Exception as e:
        print(f"❌ [控制线程] 异常: {e}")
    finally:
        avoid_client.Move(0.0, 0.0, 0.0)


# ==========================================
# 主函数
# ==========================================
def main():
    global start_time, last_detail_log
    start_time = time.time()

    print("=" * 60)
    print("  walk_the_dog.py — 自主漫步避障 (纯 avoid_client)")
    print("=" * 60)
    print()

    print("🔌 初始化 SDK...")
    ChannelFactoryInitialize(0, NETWORK_INTERFACE)

    sport_client = SportClient()
    sport_client.SetTimeout(3.0)
    sport_client.Init()
    code, ver = sport_client.GetServerApiVersion()
    if code != 0:
        print(f"❌ 运动服务器连接失败: {code}")
        sys.exit(1)
    print(f"  ✅ SportClient (API: {ver}) — 仅用于初始化和停止")

    avoid_client = ObstaclesAvoidClient()
    avoid_client.SetTimeout(3.0)
    avoid_client.Init()

    print("  🛡️ 开启 SDK 避障...")
    for i in range(30):
        code, enabled = avoid_client.SwitchGet()
        if code == 0 and enabled:
            print(f"  🛡️ SDK 避障: 已开启 (尝试 {i+1} 次)")
            break
        avoid_client.SwitchSet(True)
        time.sleep(0.1)
    else:
        print("  ⚠️ SDK 避障开关未确认，继续运行")

    avoid_client.UseRemoteCommandFromApi(True)
    time.sleep(0.3)

    print("  📡 订阅速度反馈...")
    state_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    state_sub.Init(sport_state_handler, 10)
    time.sleep(0.5)
    if (time.time() - actual_vel["last_update"]) >= 1.0:
        print("  ❌ 速度反馈不可用，退出")
        sys.exit(1)
    print(f"  📡 速度反馈: 在线 (vx={actual_vel['vx']:.3f} vyaw={actual_vel['vyaw']:.3f})")

    ctrl_thread = threading.Thread(
        target=control_thread_task, args=(avoid_client,), daemon=True
    )
    ctrl_thread.start()
    print("  🔄 控制线程: 已启动 (20Hz, 纯 avoid_client)")

    cmd_state["vx"] = WALK_SPEED
    cmd_state["vyaw"] = 0.0
    cmd_state["mode"] = STATE_WALKING
    cmd_state["last_update"] = time.time()

    print()
    print(f"🐕 配置:")
    print(f"  行走速度       = {WALK_SPEED} m/s")
    print(f"  弧线后退速度   = {ARC_BACK_SPEED} m/s")
    print(f"  弧线转弯速度   = {ARC_TURN_SPEED} rad/s")
    print(f"  弧线时长       = {ARC_DURATION}s/轮 (后退~{abs(ARC_BACK_SPEED)*ARC_DURATION*100:.0f}cm 转~{ARC_TURN_SPEED*ARC_DURATION*57.3:.0f}°)")
    print(f"  卡住检测       = {STUCK_TIMEOUT}s (vx<{STUCK_VX_THRESHOLD})")
    print(f"  试探时长       = {PROBE_DURATION}s (vx>{PROBE_VX_THRESHOLD})")
    print(f"  最大扫描轮数   = {MAX_SCAN_CYCLES}")
    print(f"  控制方式       = 纯 avoid_client（无 sport_client 冲突）")
    print()
    print("  流程: 行走 → 卡住 → 后退弧线(边退边转) → 试探 → ...")
    print("  按 Ctrl+C 停止。")
    print()

    log("🚀 开始行走")

    stuck_since = 0.0
    last_status_log = 0.0

    try:
        while True:
            now = time.time()
            vel_fresh = (now - actual_vel["last_update"]) < 1.0

            # 行走状态每 3 秒输出一次
            if cmd_state["mode"] == STATE_WALKING and now - last_status_log > 3.0:
                stuck_dur = f" 卡住计时={now - stuck_since:.1f}s" if stuck_since > 0 else ""
                log(f"正常行走中{stuck_dur}")
                last_status_log = now

            # 弧线/试探阶段每 0.5 秒输出一次详细数据
            if cmd_state["mode"] in (STATE_SCAN_ARC, STATE_SCAN_PROBE):
                if now - last_detail_log > 0.5:
                    phase_elapsed = now - cmd_state["phase_start"]
                    log(f"阶段进行中 ({phase_elapsed:.1f}s)")
                    last_detail_log = now

            # ======== 行走模式 ========
            if cmd_state["mode"] == STATE_WALKING:
                cmd_state["vx"] = WALK_SPEED
                cmd_state["vyaw"] = 0.0
                cmd_state["last_update"] = now

                if vel_fresh and abs(actual_vel["vx"]) < STUCK_VX_THRESHOLD:
                    if stuck_since == 0.0:
                        stuck_since = now
                    elif now - stuck_since > STUCK_TIMEOUT:
                        log(f"🚧 卡住 {now - stuck_since:.1f}s → 开始后退弧线")
                        cmd_state["mode"] = STATE_SCAN_ARC
                        cmd_state["phase_start"] = now
                        cmd_state["scan_cycles"] = 0
                        stuck_since = 0.0
                        last_detail_log = now
                else:
                    if stuck_since > 0.0 and now - stuck_since > 0.3:
                        log(f"卡住计时重置 (vx={actual_vel['vx']:.3f})")
                    stuck_since = 0.0

            # ======== 后退弧线：边退边左转 ========
            elif cmd_state["mode"] == STATE_SCAN_ARC:
                cmd_state["vx"] = ARC_BACK_SPEED
                cmd_state["vyaw"] = ARC_TURN_SPEED
                cmd_state["last_update"] = now

                if now - cmd_state["phase_start"] > ARC_DURATION:
                    cmd_state["scan_cycles"] += 1
                    log(f"🔄 弧线完毕 (第{cmd_state['scan_cycles']}/{MAX_SCAN_CYCLES}轮) → 试探前进")
                    cmd_state["mode"] = STATE_SCAN_PROBE
                    cmd_state["phase_start"] = now
                    last_detail_log = now

            # ======== 试探阶段 ========
            elif cmd_state["mode"] == STATE_SCAN_PROBE:
                cmd_state["vx"] = WALK_SPEED
                cmd_state["vyaw"] = 0.0
                cmd_state["last_update"] = now

                elapsed = now - cmd_state["phase_start"]

                if elapsed > PROBE_DURATION:
                    cur_vx = actual_vel["vx"]
                    if vel_fresh and cur_vx > PROBE_VX_THRESHOLD:
                        log(f"✅ 路通了! (vx={cur_vx:.3f}) → 恢复直行")
                        cmd_state["mode"] = STATE_WALKING
                        stuck_since = 0.0
                    elif cmd_state["scan_cycles"] >= MAX_SCAN_CYCLES:
                        log(f"⏱️ 已扫描 {MAX_SCAN_CYCLES} 轮, 强制前进")
                        cmd_state["mode"] = STATE_WALKING
                        stuck_since = 0.0
                    else:
                        log(f"❌ 不通 (vx={cur_vx:.3f}) → 继续后退弧线")
                        cmd_state["mode"] = STATE_SCAN_ARC
                        cmd_state["phase_start"] = now
                        last_detail_log = now

            time.sleep(0.05)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n🛑 正在安全停止... (运行了 {elapsed:.0f}s)")
        cmd_state["vx"] = 0.0
        cmd_state["vyaw"] = 0.0
        cmd_state["last_update"] = 0.0
        time.sleep(0.5)
        sport_client.StopMove()
        avoid_client.Move(0.0, 0.0, 0.0)
        avoid_client.UseRemoteCommandFromApi(False)
        print("✅ 已停止")


if __name__ == "__main__":
    main()
