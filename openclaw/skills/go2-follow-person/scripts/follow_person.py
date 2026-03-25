#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 Follow Person - 机器狗自动跟随人

功能：
1. 持续拍照
2. 使用VLM或简单检测识别人
3. 计算人在画面中的位置
4. 控制机器狗移动跟随

工作流程：
- 如果人在画面左边 → 左转
- 如果人在画面右边 → 右转  
- 如果人在画面中间但远 → 前进
- 如果人在画面中间且近 → 停止
"""

import sys
import os
import time
import subprocess
import json

# 添加依赖路径
sys.path.insert(0, '/home/unitree/openclaw/skills/go2-vision/scripts')
sys.path.insert(0, '/home/unitree/openclaw/skills/unitree-go2/scripts')

# 媒体保存路径
MEDIA_DIR = os.path.expanduser("~/.openclaw/media")
CAPTURE_PATH = os.path.join(MEDIA_DIR, "go2_camera.jpg")

# 跟随参数
CENTER_THRESHOLD = 0.2  # 中心区域阈值（画面宽度的20%）
FOLLOW_DISTANCE = 0.3   # 理想跟随距离（画面高度的30%）
MOVE_INTERVAL = 2.0     # 移动间隔（秒）
CAPTURE_INTERVAL = 1.0  # 拍照间隔（秒）


def capture_image():
    """拍照"""
    try:
        result = subprocess.run(
            ['python3', '/home/unitree/openclaw/skills/go2-vision/scripts/capture.py'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return CAPTURE_PATH
        return None
    except Exception as e:
        print(f"拍照失败: {e}")
        return None


def analyze_person_position(image_path):
    """
    分析人在画面中的位置 (使用 YOLO 检测器)
    返回: (person_detected, horizontal_position, vertical_position)
    - person_detected: bool 是否检测到人
    - horizontal_position: -1(左) 0(中) 1(右)
    - vertical_position: 0(远) 1(近) 
    """
    try:
        import json
        import subprocess
        
        print(f"分析图片: {image_path}")
        
        # 调用混合检测器 (HOG 快速 + VLM 精确)
        result = subprocess.run(
            ['python3', '/home/unitree/openclaw/skills/go2-follow-person/scripts/hybrid_detector.py', image_path],
            capture_output=True,
            text=True,
            timeout=35
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout.strip().split('\n')[-1])
            detected = data.get('detected', False)
            h_pos = data.get('horizontal', 0)
            v_pos = data.get('vertical', 0)
            method = data.get('method', 'unknown')
            return (detected, h_pos, v_pos)
        else:
            print(f"检测器错误: {result.stderr}")
            return (False, 0, 0)
        
    except Exception as e:
        print(f"分析失败: {e}")
        return (False, 0, 0)


def control_robot(action, angle=None):
    """控制机器狗移动"""
    try:
        cmd = ['python3', '/home/unitree/openclaw/skills/unitree-go2/scripts/go2_control.py', action]
        if angle:
            cmd.extend(['--angle', str(angle)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print(f"执行动作: {action} {'角度:' + str(angle) if angle else ''}")
        return result.returncode == 0
    except Exception as e:
        print(f"控制失败: {e}")
        return False


def follow_person():
    """
    主跟随逻辑
    """
    print("🐕 启动自动跟随模式")
    print("=" * 50)
    print("跟随逻辑:")
    print("  - 人在左边 → 左转")
    print("  - 人在右边 → 右转")
    print("  - 人在中间且远 → 前进")
    print("  - 人在中间且近 → 停止")
    print("=" * 50)
    print("按 Ctrl+C 停止")
    print()
    
    last_move_time = 0
    
    try:
        while True:
            # 1. 拍照
            print("📷 拍照...", end=" ", flush=True)
            image_path = capture_image()
            if not image_path:
                print("❌ 拍照失败")
                time.sleep(CAPTURE_INTERVAL)
                continue
            print("✅")
            
            # 2. 分析人位置
            print("🔍 分析人位置...", end=" ", flush=True)
            detected, h_pos, v_pos = analyze_person_position(image_path)
            
            if not detected:
                print("❌ 未检测到人")
                # 原地转圈搜索
                if time.time() - last_move_time > MOVE_INTERVAL:
                    print("🔄 搜索目标...")
                    control_robot('turn_left', 30)
                    last_move_time = time.time()
                time.sleep(CAPTURE_INTERVAL)
                continue
            
            print(f"✅ 位置: {'左' if h_pos < 0 else '右' if h_pos > 0 else '中'}, "
                  f"距离: {'近' if v_pos > 0.7 else '远' if v_pos < 0.3 else '中'}")
            
            # 3. 决策并移动
            current_time = time.time()
            if current_time - last_move_time < MOVE_INTERVAL:
                time.sleep(0.5)
                continue
            
            # 水平方向调整
            if h_pos < -CENTER_THRESHOLD:
                # 人在左边，左转
                print("⬅️  左转")
                control_robot('turn_left', 15)
                last_move_time = current_time
                
            elif h_pos > CENTER_THRESHOLD:
                # 人在右边，右转
                print("➡️  右转")
                control_robot('turn_right', 15)
                last_move_time = current_time
                
            else:
                # 人在中间，调整距离
                if v_pos < FOLLOW_DISTANCE - 0.1:
                    # 人太远，前进
                    print("⬆️  前进")
                    control_robot('move_forward')
                    last_move_time = current_time
                    
                elif v_pos > FOLLOW_DISTANCE + 0.1:
                    # 人太近，后退（可选）
                    print("⬇️  后退")
                    control_robot('move_backward')
                    last_move_time = current_time
                    
                else:
                    # 距离合适，停止
                    print("⏹️  停止（距离合适）")
                    control_robot('stop')
            
            print()
            time.sleep(CAPTURE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n🛑 停止跟随模式")
        control_robot('stop')
        print("机器狗已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Go2 自动跟随人')
    parser.add_argument('--duration', '-d', type=int, default=0,
                        help='跟随持续时间（秒），0表示无限')
    parser.add_argument('--test', '-t', action='store_true',
                        help='测试模式：只拍照分析，不移动')
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 测试模式")
        image_path = capture_image()
        if image_path:
            detected, h_pos, v_pos = analyze_person_position(image_path)
            print(f"检测结果: 人={'是' if detected else '否'}, 水平={h_pos}, 垂直={v_pos}")
    else:
        follow_person()


if __name__ == '__main__':
    main()
