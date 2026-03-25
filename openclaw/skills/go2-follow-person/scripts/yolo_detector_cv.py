#!/usr/bin/env python3
"""
YOLO Person Detector - 使用 OpenCV DNN 加载 YOLO 模型
不需要安装 ultralytics，直接用 opencv
"""
import os
import sys
import cv2
import numpy as np

# YOLO 模型路径
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_CFG = os.path.join(MODEL_DIR, "yolov3-tiny.cfg")
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov3-tiny.weights")
YOLO_NAMES = os.path.join(MODEL_DIR, "coco.names")

# 人对应的类别ID
PERSON_CLASS_ID = 0


def download_model():
    """下载 YOLO 模型文件（如果不存在）"""
    import urllib.request
    
    # 模型文件 URLs
    files = {
        YOLO_CFG: "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
        YOLO_WEIGHTS: "https://pjreddie.com/media/files/yolov3-tiny.weights",
        YOLO_NAMES: "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    
    for filepath, url in files.items():
        if not os.path.exists(filepath):
            print(f"下载 {os.path.basename(filepath)}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"✅ 下载完成: {filepath}")
            except Exception as e:
                print(f"❌ 下载失败: {e}")
                return False
    
    return True


class YoloPersonDetector:
    """YOLO 人检测器"""
    
    def __init__(self):
        self.net = None
        self.classes = None
        self.initialized = False
        
    def init(self):
        """初始化模型"""
        try:
            # 下载模型（如果不存在）
            if not download_model():
                print("模型下载失败，使用模拟检测")
                return False
            
            # 加载类别名称
            with open(YOLO_NAMES, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # 加载网络
            print("加载 YOLO 模型...")
            self.net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
            
            # 使用 CPU
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self.initialized = True
            print("✅ YOLO 模型加载完成")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def detect(self, image_path, conf_threshold=0.5, nms_threshold=0.4):
        """
        检测图片中的人
        
        返回: (detected, x_center, y_center, width, height)
        - detected: bool
        - x_center, y_center: 人在画面中的相对位置 (0-1)
        - width, height: 人的框相对大小 (0-1)
        """
        if not self.initialized:
            # 如果模型未加载，返回模拟数据
            return self._mock_detect(image_path)
        
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return (False, 0, 0, 0, 0)
            
            height, width = img.shape[:2]
            
            # 创建 blob
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            
            # 获取输出层
            layer_names = self.net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # 前向传播
            outputs = self.net.forward(output_layers)
            
            # 解析检测结果
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # 只检测人
                    if class_id == PERSON_CLASS_ID and confidence > conf_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # 非极大值抑制
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            
            if len(indexes) > 0:
                # 选择置信度最高的人
                best_idx = indexes[0]
                if isinstance(best_idx, (list, tuple, np.ndarray)):
                    best_idx = best_idx[0] if len(best_idx) > 0 else 0
                
                x, y, w, h = boxes[best_idx]
                
                # 计算相对位置
                x_center = (x + w/2) / width
                y_center = (y + h/2) / height
                w_rel = w / width
                h_rel = h / height
                
                return (True, x_center, y_center, w_rel, h_rel)
            else:
                return (False, 0, 0, 0, 0)
                
        except Exception as e:
            print(f"检测错误: {e}")
            return (False, 0, 0, 0, 0)
    
    def _mock_detect(self, image_path):
        """模拟检测（备用）"""
        import random
        if random.random() > 0.5:
            return (True, random.uniform(0.2, 0.8), random.uniform(0.3, 0.9), 0.2, 0.4)
        return (False, 0, 0, 0, 0)


# 全局检测器实例
_detector = None

def get_detector():
    """获取检测器实例（单例）"""
    global _detector
    if _detector is None:
        _detector = YoloPersonDetector()
        _detector.init()
    return _detector


def detect_person(image_path):
    """
    检测图片中的人（简化接口）
    
    返回: (detected, h_pos, v_pos)
    - detected: bool
    - h_pos: -1(左) to 1(右), 0是中间
    - v_pos: 0(远/上) to 1(近/下)
    """
    detector = get_detector()
    detected, x, y, w, h = detector.detect(image_path)
    
    if detected:
        # 转换到 -1~1 范围
        h_pos = (x - 0.5) * 2  # 0~1 -> -1~1
        v_pos = y  # 保持 0~1
        return (True, h_pos, v_pos)
    else:
        return (False, 0, 0)


if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='YOLO 人检测')
    parser.add_argument('image', help='图片路径')
    args = parser.parse_args()
    
    detected, h_pos, v_pos = detect_person(args.image)
    print(json.dumps({
        'detected': detected,
        'horizontal': h_pos,
        'vertical': v_pos
    }))
