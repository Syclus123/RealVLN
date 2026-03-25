#!/usr/bin/env python3
"""
Hybrid Person Detector - 混合检测器
先用 HOG 快速检测，检测不到再用 VLM
兼顾速度和准确性
"""
import os
import sys
import json
import cv2
import numpy as np
import base64
import requests

# API Key
DASHSCOPE_API_KEY = "sk-5a03c62b9a1549f8bec8fba654f3fd52"

# HOG 检测器（单例）
_hog_detector = None

def get_hog_detector():
    """获取 HOG 检测器"""
    global _hog_detector
    if _hog_detector is None:
        _hog_detector = cv2.HOGDescriptor()
        _hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog_detector


def detect_with_hog(image_path, output_path=None):
    """
    使用 HOG 快速检测
    返回: (detected, x_center, y_center, output_path) 或 (False, 0, 0, None)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return (False, 0, 0, None)
        
        height, width = img.shape[:2]
        hog = get_hog_detector()
        
        # 检测
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
        
        if len(boxes) > 0:
            # 选择最大的框
            best_idx = np.argmax([w * h for x, y, w, h in boxes])
            x, y, w, h = boxes[best_idx]
            
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            
            # 保存带标注的图片
            if output_path:
                img_copy = img.copy()
                cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(img_copy, (int(x+w/2), int(y+h/2)), 5, (0, 0, 255), -1)
                cv2.putText(img_copy, f"Person ({x_center:.2f}, {y_center:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(output_path, img_copy)
            
            return (True, x_center, y_center, output_path)
        
        return (False, 0, 0, None)
        
    except Exception as e:
        return (False, 0, 0, None)


def detect_with_vlm(image_path):
    """
    使用 VLM 精确检测
    返回: (detected, x_center, y_center) 或 (False, 0, 0)
    """
    try:
        with open(image_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        response = requests.post(
            'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation',
            headers={
                'Authorization': f'Bearer {DASHSCOPE_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'qwen-vl-max',
                'input': {
                    'messages': [
                        {
                            'role': 'system',
                            'content': [{'text': '你是一个图像分析助手。分析图片中是否有人，以及人的位置。只返回JSON格式结果。'}]
                        },
                        {
                            'role': 'user',
                            'content': [
                                {'image': f'data:image/jpeg;base64,{img_base64}'},
                                {'text': '图片中有人吗？如果有，估计人在画面中的中心位置（x: 0-1左到右, y: 0-1上到下）。返回JSON：{"has_person": true/false, "center_x": 0.5, "center_y": 0.5}'}  
                            ]
                        }
                    ]
                }
            },
            timeout=30
        )
        
        result = response.json()
        
        if 'output' in result and 'choices' in result['output']:
            content = result['output']['choices'][0]['message']['content'][0]['text']
            
            import re
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                data = json.loads(json_match.group())
                has_person = data.get('has_person', False)
                center_x = data.get('center_x', 0.5)
                center_y = data.get('center_y', 0.5)
                
                if has_person:
                    return (True, center_x, center_y)
        
        return (False, 0, 0)
        
    except Exception as e:
        return (False, 0, 0)


def detect_person_hybrid(image_path, use_vlm=True, annotate=True):
    """
    混合检测：先用 HOG 快速检测，检测不到再用 VLM
    
    返回: (detected, h_pos, v_pos, method, annotated_path)
    - detected: bool
    - h_pos: -1(左) to 1(右)
    - v_pos: 0(远) to 1(近)
    - method: 'hog' 或 'vlm'
    - annotated_path: 带标注的图片路径
    """
    # 生成输出路径
    annotated_path = None
    if annotate:
        base, ext = os.path.splitext(image_path)
        annotated_path = f"{base}_detected{ext}"
    
    # 第一步：HOG 快速检测
    detected, x, y, _ = detect_with_hog(image_path, annotated_path if annotate else None)
    
    if detected:
        h_pos = (x - 0.5) * 2
        v_pos = y
        print(f"[HOG] 检测到行人: 位置=({x:.2f}, {y:.2f})")
        return (True, h_pos, v_pos, 'hog', annotated_path)
    
    # 第二步：VLM 精确检测
    if use_vlm:
        print("[HOG] 未检测到，使用 VLM...")
        detected, x, y = detect_with_vlm(image_path)
        
        if detected:
            h_pos = (x - 0.5) * 2
            v_pos = y
            print(f"[VLM] 检测到行人: 位置=({x:.2f}, {y:.2f})")
            # VLM 也保存标注图
            if annotate:
                img = cv2.imread(image_path)
                if img is not None:
                    height, width = img.shape[:2]
                    cx, cy = int(x * width), int(y * height)
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)
                    cv2.putText(img, f"Person VLM ({x:.2f}, {y:.2f})", 
                               (cx-50, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imwrite(annotated_path, img)
            return (True, h_pos, v_pos, 'vlm', annotated_path)
    
    return (False, 0, 0, 'none', None)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='混合行人检测 (HOG + VLM)')
    parser.add_argument('image', help='图片路径')
    parser.add_argument('--no-vlm', action='store_true', help='只使用 HOG，不使用 VLM')
    
    args = parser.parse_args()
    
    detected, h_pos, v_pos, method, annotated_path = detect_person_hybrid(args.image, use_vlm=not args.no_vlm)
    
    result = {
        'detected': detected,
        'horizontal': h_pos,
        'vertical': v_pos,
        'method': method
    }
    if annotated_path:
        result['annotated_image'] = annotated_path
    
    print(json.dumps(result))
