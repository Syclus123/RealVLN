#!/usr/bin/env python3
"""
Person Detector - 使用 OpenCV HOG 行人检测器
不需要下载模型，开箱即用
"""
import os
import sys
import json
import cv2
import numpy as np

# 全局检测器
_hog_detector = None

def get_hog_detector():
    """获取 HOG 检测器（单例）"""
    global _hog_detector
    if _hog_detector is None:
        print("初始化 HOG 行人检测器...")
        _hog_detector = cv2.HOGDescriptor()
        _hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("✅ HOG 检测器初始化完成")
    return _hog_detector


def detect_person(image_path):
    """
    检测图片中的人
    
    返回: (detected, h_pos, v_pos)
    - detected: bool
    - h_pos: -1(左) to 1(右), 0是中间
    - v_pos: 0(远/上) to 1(近/下)
    """
    try:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return (False, 0, 0)
        
        height, width = img.shape[:2]
        
        # 获取检测器
        hog = get_hog_detector()
        
        # 检测行人
        # winStride: 滑动窗口步长
        # padding: 边缘填充
        # scale: 图像金字塔缩放比例
        boxes, weights = hog.detectMultiScale(
            img, 
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )
        
        # 过滤低置信度的检测
        min_confidence = 0.5
        valid_detections = []
        
        for i, (x, y, w, h) in enumerate(boxes):
            confidence = weights[i] if i < len(weights) else 0
            if confidence > min_confidence or confidence == 0:  # confidence=0 是 OpenCV HOG 的特殊情况
                valid_detections.append((x, y, w, h, confidence))
        
        if len(valid_detections) > 0:
            # 选择最大的人（通常是最近的）
            best = max(valid_detections, key=lambda d: d[2] * d[3])  # 按面积排序
            x, y, w, h, conf = best
            
            # 计算中心位置
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            
            # 转换到 -1~1 范围
            h_pos = (x_center - 0.5) * 2
            v_pos = y_center
            
            print(f"检测到行人: 位置=({x_center:.2f}, {y_center:.2f}), 置信度={conf:.2f}")
            return (True, h_pos, v_pos)
        else:
            print("未检测到行人")
            return (False, 0, 0)
            
    except Exception as e:
        print(f"检测错误: {e}")
        import traceback
        traceback.print_exc()
        return (False, 0, 0)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='HOG 行人检测')
    parser.add_argument('image', help='图片路径')
    args = parser.parse_args()
    
    detected, h_pos, v_pos = detect_person(args.image)
    print(json.dumps({
        'detected': detected,
        'horizontal': h_pos,
        'vertical': v_pos
    }))
