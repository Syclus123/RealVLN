#!/usr/bin/env python3
"""发送图片到飞书"""
import os
import requests
import json

# 飞书配置（来自 send_audio.py）
APP_ID = "cli_a9232fdff3795cc6"
APP_SECRET = "4YL27fguAn1Wb8pLbdjPngKWWKXfDjNo"
CHAT_ID = "ou_da6b544f17a5b166a23d6cb4ce4e9b82"
IMAGE_PATH = "/home/unitree/.openclaw/media/go2_camera_detected.jpg"

def get_token():
    """获取 tenant_access_token"""
    url = 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal'
    resp = requests.post(url, json={'app_id': APP_ID, 'app_secret': APP_SECRET}, timeout=10)
    data = resp.json()
    if data.get('code') == 0:
        return data['tenant_access_token']
    else:
        print(f"获取 token 失败: {data}")
        return None

def upload_image(token, image_path):
    """上传图片"""
    url = 'https://open.feishu.cn/open-apis/im/v1/files'
    headers = {'Authorization': f'Bearer {token}'}
    
    with open(image_path, 'rb') as f:
        files = {'file': ('detection.jpg', f, 'image/jpeg')}
        data = {'file_type': 'image', 'file_name': 'detection.jpg'}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=30)
    
    result = resp.json()
    if result.get('code') == 0:
        return result['data']['file_key']
    else:
        print(f"上传失败: {result}")
        return None

def send_image(token, chat_id, image_key):
    """发送图片消息"""
    url = 'https://open.feishu.cn/open-apis/im/v1/messages'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    params = {'receive_id_type': 'open_id'}
    content = json.dumps({'image_key': image_key})
    
    data = {
        'receive_id': chat_id,
        'msg_type': 'image',
        'content': content
    }
    
    resp = requests.post(url, headers=headers, params=params, json=data, timeout=10)
    return resp.json()

# 执行
token = get_token()
if token:
    print(f"✅ 获取 token 成功")
    
    image_key = upload_image(token, IMAGE_PATH)
    if image_key:
        print(f"✅ 上传图片成功: {image_key}")
        
        result = send_image(token, CHAT_ID, image_key)
        if result.get('code') == 0:
            print(f"✅ 图片发送成功!")
        else:
            print(f"❌ 发送失败: {result}")
    else:
        print("❌ 上传失败")
else:
    print("❌ 无法获取 token")
