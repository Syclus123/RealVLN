#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 Follow Person - Advanced Version
使用 OpenCV HOG 人检测器
"""

import sys
import os
import time
import subprocess
import json

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("⚠️  OpenCV 未安装，将使用基础模式")

# 媒体保存路径
MEDIA_DIR = os.path.expanduser("~/.openclaw/media")
CAPTURE_PATH = os.path.join(MEDIA_DIR, "go2_camera.jpg")

# 跟随参数
CENTER_THRESHOLD = 0.15  # 中心区域阈值
FOLLOW_DISTANCE = 0.4    # 理想跟随距离（人占画面高度比例）
MOVE_INTERVAL = 1.5      # 移动间隔
CAPTURE_INTERVAL = 0.8   # 拍照间隔
MIN_CONFIDENCE = 0.3     # 最小检测置信度


class PersonDetector:
    """人检测器"""
    
    def __init__(self):
        self.detector = None
        if OPENCV_AVAILABLE:
            try:
                # 使用 OpenCV 的 HOG 检测器
                self.detector = cv2.HOGDescriptor()
                self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                print("✅ HOG 人检测器初始化成功")
            except Exception as e:
                print(f"⚠️  HOG 初始化失败: {e}")
                self.detector = None
    
    def detect(self, image_path):
        """
        检测人并返回位置
        返回: (detected, center_x, center_y, width, height)
        """
        if not OPENCV_AVAILABLE or self.detector is None:
            # 基础模式：模拟检测
            return self._mock_detect(image_path)
        
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return (False, 0, 0, 0, 0)
            
            img_height, img_width = img.shape[:2]
            
            # HOG 检测
            boxes, weights = self.detector.detectMultiScale(
                img, 
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05
            )
            
            # 找到最大的人（最近的人）
            if len(boxes) > 0:
                # 选择面积最大的框（通常是最近的人）
                largest_idx = np.argmax([w * h for (x, y, w, h) in boxes])
                x, y, w, h = boxes[largest_idx]
                confidence = weights[largest_idx] if len(weights) > largest_idx else 1.0
                
                if confidence < MIN_CONFIDENCE:
                    return (False, 0, 0, 0, 0)
                
                # 计算中心点（归一化到 0-1）
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                return (True, center_x, center_y, norm_width, norm_height)
            
            return (False, 0, 0, 0, 0)
            
        except Exception as e:
            print(f"检测错误: {e}")
            return (False, 0, 0, 0, 0)
    
    def _mock_detect(self, image_path):
        """模拟检测（用于测试）"""
        # 这里可以添加简单的图像分析
        # 比如检测肤色区域、运动等
        return (False, 0, 0, 0, 0)


def capture_image():
    """拍照"""
    try:
        result = subprocess.run(
            ['python3', '/home/unitree/openclaw/skills/go2-vision/scripts/capture.py'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return CAPTURE_PATH if result.returncode == 0 else None
    except Exception as e:
        print(f"❌ 拍照失败: {e}")
        return None


def control_robot(action, angle=None):
    """控制机器狗"""
    try:
        cmd = ['python3', '/home/unitree/openclaw/skills/unitree-go2/scripts/go2_control.py', action]
        if angle:
            cmd.extend(['--angle', str(angle)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 控制失败: {e}")
        return False


def follow_person_advanced(duration=0):
    """
    高级跟随模式
    """
    print("\n" + "=" * 60)
    print("🐕 Go2 自动跟随人 - 高级版")
    print("=" * 60)
    
    if OPENCV_AVAILABLE:
        print("✅ 使用 OpenCV HOG 检测器")
    else:
        print("⚠️  使用基础检测模式（建议安装 OpenCV）")
    print("-" * 60)
    print("操作说明：")
    print("  • 人出现在画面中 → 机器狗自动跟随")
    print("  • 人消失 → 机器狗原地搜索")
    print("  • 按 Ctrl+C 停止")
    print("=" * 60 + "\n")
    
    # 初始化检测器
    detector = PersonDetector()
    
    last_move_time = 0
    search_direction = 1  # 1=左转, -1=右转
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            # 检查持续时间
            if duration > 0 and time.time() - start_time > duration:
                print("\n⏰ 达到设定时长，停止跟随")
                break
            
            frame_count += 1
            print(f"\n📷 Frame #{frame_count}")
            
            # 1. 拍照
            image_path = capture_image()
            if not image_path:
                time.sleep(CAPTURE_INTERVAL)
                continue
            
            # 2. 检测人
            detected, center_x, center_y, width, height = detector.detect(image_path)
            
            if not detected:
                print("🔍 未检测到人，搜索中...")
                if time.time() - last_move_time > MOVE_INTERVAL:
                    # 交替左右搜索
                    action = 'turn_left' if search_direction > 0 else 'turn_right'
                    control_robot(action, 30)
                    search_direction *= -1
                    last_move_time = time.time()
                time.sleep(CAPTURE_INTERVAL)
                continue
            
            # 3. 计算位置偏差
            h_offset = center_x - 0.5  # -0.5 ~ +0.5，负=左，正=右
            v_size = height  # 人占画面高度的比例，越大=越近
            
            print(f"👤 检测到: 水平偏移={h_offset:+.2f}, 距离比例={v_size:.2f}")
            
            # 4. 决策
            current_time = time.time()
            if current_time - last_move_time < MOVE_INTERVAL:
                time.sleep(0.3)
                continue
            
            # 水平调整
            if abs(h_offset) > CENTER_THRESHOLD:
                if h_offset < 0:
                    print("⬅️  左转")
                    control_robot('turn_left', 15)
                else:
                    print("➡️  右转")
                    control_robot('turn_right', 15)
                last_move_time = current_time
            
            # 距离调整
            else:
                if v_size < FOLLOW_DISTANCE - 0.05:
                    print("⬆️  前进（人太远）")
                    control_robot('move_forward')
                    last_move_time = current_time
                    
                elif v_size > FOLLOW_DISTANCE + 0.05:
                    print("⬇️  后退（人太近）")
                    control_robot('move_backward')
                    last_move_time = current_time
                    
                else:
                    print("✅ 位置理想，停止")
                    control_robot('stop')
            
            time.sleep(CAPTURE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n🛑 用户停止")
    finally:
        control_robot('stop')
        print("🐕 机器狗已停止")
        print(f"📊 共处理 {frame_count} 帧")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Go2 自动跟随人 - 高级版')
    parser.add_argument('--duration', '-d', type=int, default=0,
                        help='跟随持续时间（秒），默认无限')
    parser.add_argument('--install', action='store_true',
                        help='安装 OpenCV 依赖')
    
    args = parser.parse_args()
    
    if args.install:
        print("安装 OpenCV...")
        os.system('pip3 install opencv-python numpy')
        return
    
    follow_person_advanced(args.duration)


if __name__ == '__main__':
    main()
