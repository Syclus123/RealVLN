#!/usr/bin/env python3
"""
Person Detector - 使用 VLM (视觉语言模型) 检测人
比 HOG/YOLO 更准，能检测部分身体
"""
import os
import sys
import json
import base64
import requests

DASHSCOPE_API_KEY = "sk-5a03c62b9a1549f8bec8fba654f3fd52"


def detect_person_with_vlm(image_path):
    """
    使用 VLM 检测图片中的人（带重试）
    
    返回: (detected, h_pos, v_pos)
    - detected: bool
    - h_pos: -1(左) to 1(右), 0是中间  
    - v_pos: 0(远/上) to 1(近/下)
    """
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # 读取图片
            with open(image_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # 调用 DashScope VLM
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
                
                # 解析 JSON
                import re
                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                    has_person = data.get('has_person', False)
                    center_x = data.get('center_x', 0.5)
                    center_y = data.get('center_y', 0.5)
                    
                    if has_person:
                        # 转换到 -1~1 范围
                        h_pos = (center_x - 0.5) * 2
                        v_pos = center_y
                        print(f"VLM检测到行人: 位置=({center_x:.2f}, {center_y:.2f})")
                        return (True, h_pos, v_pos)
                
                return (False, 0, 0)
            else:
                print(f"VLM返回异常: {result}")
                if attempt < max_retries - 1:
                    continue
                return (False, 0, 0)
            
        except Exception as e:
            print(f"VLM检测错误 (尝试{attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
                continue
            return (False, 0, 0)
    
    return (False, 0, 0)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='VLM 行人检测')
    parser.add_argument('image', help='图片路径')
    args = parser.parse_args()
    
    detected, h_pos, v_pos = detect_person_with_vlm(args.image)
    print(json.dumps({
        'detected': detected,
        'horizontal': h_pos,
        'vertical': v_pos
    }))
