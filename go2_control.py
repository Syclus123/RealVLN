#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 Sport Controller - 控制Unitree Go2机器狗的运动
"""
from __future__ import annotations

import sys
import time

# 全局变量，用于存储转向角度参数
_turn_angle_degrees = 90  # 默认90度

# 动作映射表
ACTIONS = {
    "damp": {"id": 0, "desc": "阻尼模式", "fn": lambda c: c.Damp()},
    "stand_up": {"id": 1, "desc": "站立", "fn": lambda c: c.StandUp()},
    "stand_down": {"id": 2, "desc": "趴下", "fn": lambda c: c.StandDown()},
    "move_forward": {"id": 3, "desc": "前进", "fn": lambda c: _move_forward(c)},
    "move_backward": {"id": 23, "desc": "后退", "fn": lambda c: _move_backward(c)},    
    "move_lateral": {"id": 4, "desc": "横向移动", "fn": lambda c: c.Move(0, 0.3, 0)},
    "turn_left": {"id": 5, "desc": "左转", "fn": lambda c: _turn_left(c)},
    "turn_right": {"id": 25, "desc": "右转", "fn": lambda c: _turn_right(c)},
    "stop": {"id": 6, "desc": "停止", "fn": lambda c: c.StopMove()},
    "handstand": {"id": 7, "desc": "倒立", "fn": lambda c: _handstand(c)},
    "balance": {"id": 9, "desc": "平衡站立", "fn": lambda c: c.BalanceStand()},
    "recovery": {"id": 10, "desc": "恢复站立", "fn": lambda c: c.RecoveryStand()},
    "left_flip": {"id": 11, "desc": "左侧翻", "fn": lambda c: c.LeftFlip()},
    "back_flip": {"id": 12, "desc": "后空翻", "fn": lambda c: c.BackFlip()},
    "free_walk": {"id": 13, "desc": "自由行走", "fn": lambda c: c.FreeWalk()},
    "free_bound": {"id": 14, "desc": "自由跳跃", "fn": lambda c: _free_bound(c)},
    "free_avoid": {"id": 15, "desc": "自由避障", "fn": lambda c: _free_avoid(c)},
    "walk_upright": {"id": 17, "desc": "直立行走", "fn": lambda c: _walk_upright(c)},
    "cross_step": {"id": 18, "desc": "交叉步", "fn": lambda c: _cross_step(c)},
    "free_jump": {"id": 19, "desc": "自由跳跃", "fn": lambda c: _free_jump(c)},
    "sit": {"id": 20, "desc": "坐下", "fn": lambda c: c.Sit()},
    "hello": {"id": 21, "desc": "打招呼", "fn": lambda c: c.Hello()},
    "stretch": {"id": 22, "desc": "伸展", "fn": lambda c: c.Stretch()},
}

def _handstand(c):
    c.HandStand(True)
    time.sleep(4)
    return c.HandStand(False)

def _free_bound(c):
    c.FreeBound(True)
    time.sleep(2)
    return c.FreeBound(False)

def _free_avoid(c):
    c.FreeAvoid(True)
    time.sleep(2)
    return c.FreeAvoid(False)

def _walk_upright(c):
    c.WalkUpright(True)
    time.sleep(4)
    return c.WalkUpright(False)

def _cross_step(c):
    c.CrossStep(True)
    time.sleep(4)
    return c.CrossStep(False)

def _free_jump(c):
    c.FreeJump(True)
    time.sleep(4)
    return c.FreeJump(False)

def _move_forward(c, duration=2.0, vx=0.3):
    """持续前进一段时间"""
    print(f"开始前进 {duration} 秒...")
    start_time = time.time()
    while time.time() - start_time < duration:
        c.Move(vx, 0, 0)
        time.sleep(0.1)  # 每100ms发送一次移动命令
    print("前进结束，准备停止")
    return c.StopMove()


def _move_backward(c, duration=2.0, vx=-0.3):
    """持续前进一段时间"""
    print(f"开始前进 {duration} 秒...")
    start_time = time.time()
    while time.time() - start_time < duration:
        c.Move(vx, 0, 0)
        time.sleep(0.1)  # 每100ms发送一次移动命令
    print("前进结束，准备停止")
    return c.StopMove()


# 全局变量，用于存储转向角度参数
_turn_angle_degrees = 90  # 默认90度

def _turn_left(c, angle=None):
    """向左旋转（逆时针）指定角度"""
    global _turn_angle_degrees
    target_angle = angle if angle is not None else _turn_angle_degrees
    # 修正系数：90度约需2.5秒
    duration = target_angle / 90 * 2.5
    vyaw = 0.8
    
    print(f"开始左转 {target_angle} 度（约 {duration:.1f} 秒）...")
    start_time = time.time()
    while time.time() - start_time < duration:
        c.Move(0, 0, vyaw)
        time.sleep(0.1)
    print("左转结束，准备停止")
    return c.StopMove()


def _turn_right(c, angle=None):
    """向右旋转（顺时针）指定角度"""
    global _turn_angle_degrees
    target_angle = angle if angle is not None else _turn_angle_degrees
    # 修正系数：90度约需2.5秒
    duration = target_angle / 90 * 2.5
    vyaw = -0.8
    
    print(f"开始右转 {target_angle} 度（约 {duration:.1f} 秒）...")
    start_time = time.time()
    while time.time() - start_time < duration:
        c.Move(0, 0, vyaw)
        time.sleep(0.1)
    print("右转结束，准备停止")
    return c.StopMove()


def print_actions():
    print("可用动作列表:")
    for name, info in ACTIONS.items():
        print(f"  {name:15} - {info['desc']}")


def get_action_list() -> list[dict]:
    """返回所有可用动作的简要信息列表（供 WebUI 等外部模块使用）。"""
    return [
        {"name": name, "id": info["id"], "desc": info["desc"]}
        for name, info in ACTIONS.items()
    ]


class Go2Controller:
    """
    通过子进程调用 go2_control.py 执行动作，避免 Unitree SDK 的 DDS
    与 ROS2 DDS 在同一进程中冲突导致 std::bad_alloc。
    """

    def __init__(self, network_interface: str = "eth0"):
        self.network_interface = network_interface
        self._lock = __import__("threading").Lock()
        self._script_path = __import__("os").path.join(
            __import__("os").path.dirname(__import__("os").path.abspath(__file__)),
            "go2_control.py",
        )

    def init(self) -> None:
        """验证脚本存在即可，实际 SDK 初始化在子进程中完成。"""
        if not __import__("os").path.isfile(self._script_path):
            raise FileNotFoundError(f"go2_control.py 未找到: {self._script_path}")
        print(f"[Go2] 子进程模式就绪，脚本: {self._script_path}，接口: {self.network_interface}")

    @property
    def is_ready(self) -> bool:
        return __import__("os").path.isfile(self._script_path)

    def execute(self, action_name: str, angle: float | None = None) -> tuple[bool, str]:
        """
        通过子进程执行指定动作，返回 (success, message)。
        线程安全：同一时刻只允许一个动作在执行。
        """
        import subprocess, os

        if action_name not in ACTIONS:
            return False, f"未知动作 '{action_name}'，可用: {', '.join(ACTIONS.keys())}"

        with self._lock:
            desc = ACTIONS[action_name]["desc"]
            cmd = [
                sys.executable, self._script_path,
                action_name, self.network_interface,
            ]
            if angle is not None:
                cmd += ["--angle", str(angle)]

            print(f"[Go2] 子进程执行: {' '.join(cmd)}")
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=os.path.dirname(self._script_path),
                )
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()
                if stdout:
                    for line in stdout.splitlines():
                        print(f"[Go2]   {line}")
                if result.returncode != 0:
                    err_msg = stderr or stdout or f"退出码 {result.returncode}"
                    return False, f"动作执行失败: {err_msg}"
                return True, f"动作 '{action_name}' ({desc}) 执行完成"
            except subprocess.TimeoutExpired:
                return False, f"动作 '{action_name}' 执行超时（30s）"
            except Exception as e:
                return False, f"动作执行异常: {e}"


def execute_action(action_name, network_interface="eth0"):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    if action_name not in ACTIONS:
        print(f"错误: 未知动作 '{action_name}'")
        print_actions()
        sys.exit(1)
    
    action = ACTIONS[action_name]
    print(f"执行动作: {action_name} (ID: {action['id']}) - {action['desc']}")
    
    if network_interface:
        print(f"使用网络接口: {network_interface}")
        ChannelFactoryInitialize(0, network_interface)
    else:
        print("使用默认网络配置")
        ChannelFactoryInitialize(0)
    
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    
    print("正在初始化 SportClient...")
    sport_client.Init()
    
    code, server_version = sport_client.GetServerApiVersion()
    if code != 0:
        print(f"错误: 无法连接到机器狗服务器，错误码: {code}")
        sys.exit(1)
    print(f"服务器API版本: {server_version}")
    print("已连接到机器狗")
    
    time.sleep(0.5)
    
    print(f"正在执行: {action['desc']}...")
    code = action['fn'](sport_client)
    
    if code is not None and code != 0:
        print(f"错误: 动作执行失败，错误码: {code}")
        sys.exit(1)
    
    print(f"✓ 动作 '{action_name}' 执行完成")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python go2_control.py <动作名称> [网络接口] [--angle 角度]")
        print("示例:")
        print("  python go2_control.py stand_up eth0")
        print("  python go2_control.py turn_left eth0 --angle 45")
        print("  python go2_control.py turn_right eth0 --angle 90")
        print_actions()
        sys.exit(1)
    
    action = sys.argv[1]
    network_interface = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "eth0"
    
    # 解析角度参数
    if "--angle" in sys.argv:
        angle_idx = sys.argv.index("--angle")
        if angle_idx + 1 < len(sys.argv):
            try:
                _turn_angle_degrees = float(sys.argv[angle_idx + 1])
                print(f"设置转向角度: {_turn_angle_degrees} 度")
            except ValueError:
                print(f"警告: 无效的角度值 '{sys.argv[angle_idx + 1]}'，使用默认值 90 度")
    
    if action in ["list", "-l", "--list", "help", "-h", "--help"]:
        print_actions()
        sys.exit(0)

    # 动作前准备：机器狗可能处于坐姿（刚执行过 sit），此时 BalanceStand() 会报错 -1。
    # - balance / stand_up / recovery：无需前置，直接执行。
    # - sit：先 balance 再 sit（假定当前为站立）。
    # - 其余动作（转向、移动等）：先 stand_up 再 balance，确保站立后再执行。
    if action == "sit":
        execute_action("balance", network_interface)
    elif action not in ("balance", "stand_up", "recovery"):
        execute_action("stand_up", network_interface)
        execute_action("balance", network_interface)

    execute_action(action, network_interface)