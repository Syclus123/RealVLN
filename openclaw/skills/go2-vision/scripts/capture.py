#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 Vision - Capture image from robot dog's front camera
"""
import sys
import os
import time

# SDK imports
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

def capture_image(network_interface="eth0"):
    """Capture image from Go2 front camera"""
    
    # Save to ~/.openclaw/media/ for Feishu media access compatibility
    output_image = os.path.expanduser("~/.openclaw/media/go2_camera.jpg")
    
    # Remove old image if exists
    if os.path.exists(output_image):
        os.remove(output_image)
    
    print(f"📷 正在从 Go2 前置摄像头捕获图像...")
    print(f"   网络接口: {network_interface}")
    
    try:
        # Initialize channel
        ChannelFactoryInitialize(0, network_interface)
        
        # Create video client
        client = VideoClient()
        client.SetTimeout(3.0)
        client.Init()
        
        print("##################GetImageSample###################")
        code, data = client.GetImageSample()
        
        if code != 0:
            print(f"❌ 获取图像失败，错误码: {code}", file=sys.stderr)
            return None
        
        # Save image
        with open(output_image, "wb") as f:
            f.write(bytes(data))
        
        # Get file size
        size = os.path.getsize(output_image)
        print(f"✅ 图像已捕获: {output_image} ({size} bytes)")
        
        # Output MEDIA tag for OpenClaw Gateway to capture and send
        print(f"MEDIA: {output_image}")
        
        return output_image
        
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    network_interface = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    
    image_path = capture_image(network_interface)
    
    if image_path:
        print(image_path)
        sys.exit(0)
    else:
        sys.exit(1)
