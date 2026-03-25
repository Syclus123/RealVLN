#!/usr/bin/env python3
"""
YOLO Person Detector - 使用 YOLO 检测图片中的人
"""
import os
import sys
import json

def detect_person_with_vlm(image_path):
    """
    使用 VLM 分析图片，检测人的位置和大小
    
    返回: (detected, x_center, y_center, width_ratio, height_ratio)
    - detected: bool 是否检测到人
    - x_center: 0-1 人在画面中的水平中心位置
    - y_center: 0-1 人在画面中的垂直中心位置  
    - width_ratio: 0-1 人占画面宽度比例
    - height_ratio: 0-1 人占画面高度比例
    """
    try:
        # 使用简单的图像分析 - 基于亮度/颜色检测
        # 实际应该调用 VLM 或 YOLO
        
        # 这里我们用一个简单启发式：检查图片下方区域（通常人站在那里）
        from PIL import Image
        import numpy as np
        
        img = Image.open(image_path)
        img_array = np.array(img)
        
        height, width = img_array.shape[:2]
        
        # 简单的基于颜色方差检测
        # 人通常和背景有不同颜色特征
        bottom_half = img_array[height//2:, :, :]
        
        # 计算颜色方差
        variance = np.var(bottom_half)
        
        # 如果方差大，可能有物体（包括人）
        # 这是一个非常简化的检测
        if variance > 1000:  # 阈值
            # 假设人在中间
            return (True, 0.5, 0.75, 0.3, 0.5)
        else:
            return (False, 0, 0, 0, 0)
            
    except Exception as e:
        print(f"检测错误: {e}")
        return (False, 0, 0, 0, 0)


def detect_person_simple(image_path):
    """
    简单的人检测 - 返回人在画面中的相对位置
    
    返回: (detected, h_pos, v_pos)
    - detected: bool
    - h_pos: -1(左) to 1(右), 0是中间
    - v_pos: 0(远/上) to 1(近/下)
    """
    try:
        # 这里应该使用真正的 YOLO 模型
        # 暂时返回模拟数据用于测试框架
        
        # 检测文件是否存在
        if not os.path.exists(image_path):
            return (False, 0, 0)
        
        # TODO: 集成真正的 YOLO 检测
        # 例如: model = YOLO('yolov8n.pt')
        #       results = model(image_path)
        #       person_boxes = [r for r in results if r.cls == 0]  # class 0 = person
        
        # 临时：返回随机位置用于测试
        import random
        if random.random() > 0.3:  # 70%概率检测到人
            h_pos = random.uniform(-0.8, 0.8)
            v_pos = random.uniform(0.2, 0.9)
            return (True, h_pos, v_pos)
        else:
            return (False, 0, 0)
            
    except Exception as e:
        print(f"检测失败: {e}")
        return (False, 0, 0)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='检测图片中的人')
    parser.add_argument('image', help='图片路径')
    parser.add_argument('--method', choices=['simple', 'vlm'], default='simple',
                       help='检测方法')
    
    args = parser.parse_args()
    
    if args.method == 'simple':
        detected, h_pos, v_pos = detect_person_simple(args.image)
        print(json.dumps({
            'detected': detected,
            'horizontal': h_pos,
            'vertical': v_pos
        }))
    else:
        detected, x, y, w, h = detect_person_with_vlm(args.image)
        print(json.dumps({
            'detected': detected,
            'x_center': x,
            'y_center': y,
            'width': w,
            'height': h
        }))
