#!/usr/bin/env python3
"""测试跟随模式的时间消耗"""
import time
import subprocess
import json

def test_detection_timing():
    """测试每个步骤的时间"""
    image_path = "/home/unitree/.openclaw/media/go2_camera.jpg"
    
    print("⏱️  跟随模式时间分析")
    print("=" * 50)
    
    # 1. 拍照时间
    print("\n1️⃣  拍照时间:")
    start = time.time()
    result = subprocess.run(
        ['python3', '/home/unitree/openclaw/skills/go2-vision/scripts/capture.py'],
        capture_output=True, text=True, timeout=10
    )
    capture_time = time.time() - start
    print(f"   耗时: {capture_time:.2f} 秒")
    
    # 2. HOG 检测时间
    print("\n2️⃣  HOG 检测时间:")
    start = time.time()
    result = subprocess.run(
        ['python3', '-c', f'''
import cv2
import numpy as np
img = cv2.imread("{image_path}")
if img is not None:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(img, winStride=(8,8), padding=(8,8), scale=1.05)
    print(f"检测到 {{len(boxes)}} 个人")
'''],
        capture_output=True, text=True, timeout=10
    )
    hog_time = time.time() - start
    print(f"   耗时: {hog_time:.2f} 秒")
    
    # 3. VLM 检测时间
    print("\n3️⃣  VLM 检测时间:")
    start = time.time()
    result = subprocess.run(
        ['python3', '/home/unitree/openclaw/skills/go2-follow-person/scripts/vlm_detector.py', image_path],
        capture_output=True, text=True, timeout=35
    )
    vlm_time = time.time() - start
    print(f"   耗时: {vlm_time:.2f} 秒")
    
    # 4. 混合检测时间
    print("\n4️⃣  混合检测时间 (HOG + 失败时VLM):")
    start = time.time()
    result = subprocess.run(
        ['python3', '/home/unitree/openclaw/skills/go2-follow-person/scripts/hybrid_detector.py', image_path],
        capture_output=True, text=True, timeout=35
    )
    hybrid_time = time.time() - start
    print(f"   耗时: {hybrid_time:.2f} 秒")
    
    # 5. 移动命令时间
    print("\n5️⃣  移动命令时间:")
    start = time.time()
    result = subprocess.run(
        ['python3', '/home/unitree/openclaw/skills/unitree-go2/scripts/go2_control.py', 'turn_left', '--angle', '15'],
        capture_output=True, text=True, timeout=10
    )
    move_time = time.time() - start
    print(f"   耗时: {move_time:.2f} 秒")
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 时间分析总结:")
    print(f"   拍照:     {capture_time:.2f}s")
    print(f"   HOG检测:  {hog_time:.2f}s")
    print(f"   VLM检测:  {vlm_time:.2f}s")
    print(f"   混合检测: {hybrid_time:.2f}s")
    print(f"   移动命令: {move_time:.2f}s")
    print(f"   单周期总计: ~{capture_time + hybrid_time + move_time:.2f}s")
    print("\n💡 优化建议:")
    if vlm_time > 2:
        print("   - VLM 检测太慢 (2-3s)，建议优先使用 HOG")
    if capture_time > 1:
        print("   - 拍照时间较长，检查网络连接")

if __name__ == '__main__':
    test_detection_timing()
