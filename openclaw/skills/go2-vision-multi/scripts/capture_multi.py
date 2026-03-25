#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 Vision Multi - 转圈拍照（改进版）
改为：转一点 → 停下来 → 拍照 → 继续转
避免照片抖动模糊
"""
import sys
import os
import time
import argparse
from datetime import datetime

# SDK imports
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient


class Go2CaptureController:
    """Go2 拍摄控制器 - 转-停-拍模式"""
    
    def __init__(self, network_interface="eth0"):
        self.network_interface = network_interface
        self.folder_path = None
        self.image_paths = []
        
        # Initialize channel
        ChannelFactoryInitialize(0, network_interface)
        
        # 创建客户端
        self.video_client = VideoClient()
        self.video_client.SetTimeout(3.0)
        self.video_client.Init()
        
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        
    def create_folder(self):
        """创建带时间戳的文件夹"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.expanduser("~/.openclaw/media")
        self.folder_path = os.path.join(base_dir, f"go2_capture_{timestamp}")
        os.makedirs(self.folder_path, exist_ok=True)
        return self.folder_path
    
    def capture_image(self, index):
        """
        拍摄单张照片
        
        Args:
            index: 照片序号
            
        Returns:
            成功返回文件路径，失败返回 None
        """
        try:
            print(f"📷 拍摄第 {index} 张...")
            
            code, data = self.video_client.GetImageSample()
            
            if code != 0:
                print(f"❌ 第 {index} 张获取失败，错误码: {code}", file=sys.stderr)
                return None
            
            # 生成文件名
            filename = f"go2_{index:03d}.jpg"
            image_path = os.path.join(self.folder_path, filename)
            
            # 保存图像
            with open(image_path, "wb") as f:
                f.write(bytes(data))
            
            size = os.path.getsize(image_path)
            print(f"✅ 已保存: {filename} ({size} bytes)")
            
            return image_path
            
        except Exception as e:
            print(f"❌ 拍照错误: {e}", file=sys.stderr)
            return None
    
    def rotate_step(self, angle_per_step, direction=1):
        """
        旋转指定角度（单步）
        
        Args:
            angle_per_step: 每次旋转的角度
            direction: 1=左转, -1=右转
        """
        # 计算旋转时间：90度约需2.5秒
        duration = abs(angle_per_step) / 90.0 * 2.5
        vyaw = 0.8 * direction
        
        print(f"🔄 旋转 {angle_per_step:.1f} 度...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            self.sport_client.Move(0, 0, vyaw)
            time.sleep(0.05)  # 每50ms发送一次移动命令
        
        # 停止
        self.sport_client.StopMove()
        print(f"✓ 旋转完成，已停止")
        
        # 等待稳定（防止抖动）
        time.sleep(0.3)
    
    def capture_by_steps(self, count=8, angle=360):
        """
        转圈拍照（改进版）：转一点 → 停 → 拍 → 继续
        
        Args:
            count: 拍摄照片数量
            angle: 总旋转角度（默认360度）
        
        Returns:
            folder_path, image_paths
        """
        # 创建文件夹
        self.create_folder()
        
        direction = 1 if angle > 0 else -1
        direction_name = "左转" if angle > 0 else "右转"
        angle_per_step = abs(angle) / count  # 每步旋转的角度
        
        print(f"\n🎯 转圈拍照模式（转-停-拍）")
        print(f"   拍摄数量: {count} 张")
        print(f"   总旋转: {abs(angle)} 度（{direction_name}）")
        print(f"   每步旋转: {angle_per_step:.1f} 度")
        print(f"   保存目录: {self.folder_path}")
        print()
        
        for i in range(count):
            print(f"\n========== 第 {i+1}/{count} 张 ==========")
            
            # 1. 旋转到下一个位置（第一步不旋转）
            if i > 0:
                self.rotate_step(angle_per_step, direction)
            else:
                print(f"📍 起始位置，无需旋转")
                time.sleep(0.3)  # 等待稳定
            
            # 2. 拍照（已停止，照片清晰）
            image_path = self.capture_image(i + 1)
            
            if image_path:
                self.image_paths.append(image_path)
                print(f"📸 第 {i+1} 张拍摄完成")
            else:
                print(f"⚠️ 第 {i+1} 张拍摄失败")
        
        # 最后转回起始方向（可选）
        print(f"\n🔄 转回起始方向...")
        for i in range(count):
            self.rotate_step(angle_per_step, -direction)
        print(f"✓ 已回到起始位置")
        
        # 输出结果
        print()
        print(f"🎉 拍摄完成！共保存 {len(self.image_paths)} 张照片")
        print(f"📁 文件夹: {self.folder_path}")
        
        # 输出 MEDIA 标签
        for path in self.image_paths:
            print(f"MEDIA: {path}")
        
        return self.folder_path, self.image_paths


def main():
    parser = argparse.ArgumentParser(description="Go2 转圈拍照（转-停-拍模式，照片清晰）")
    parser.add_argument("--count", "-c", type=int, default=8, help="拍摄照片数量（默认8张）")
    parser.add_argument("--angle", "-a", type=float, default=360, help="旋转角度，正数左转负数右转（默认360度）")
    parser.add_argument("--network", "-n", default="eth0", help="网络接口（默认eth0）")
    
    args = parser.parse_args()
    
    # 创建控制器
    controller = Go2CaptureController(network_interface=args.network)
    
    # 执行转圈拍照
    folder_path, image_paths = controller.capture_by_steps(
        count=args.count,
        angle=args.angle
    )
    
    if image_paths:
        print(folder_path)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
