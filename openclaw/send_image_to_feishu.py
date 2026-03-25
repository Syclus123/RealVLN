#!/usr/bin/env python3
"""发送图片到飞书 - 供 follow_fast 调用"""
import sys
import os
import requests
import json

APP_ID = "cli_a9232fdff3795cc6"
APP_SECRET = "4YL27fguAn1Wb8pLbdjPngKWWKXfDjNo"
CHAT_ID = os.environ.get("FEISHU_CHAT_ID", "ou_da6b544f17a5b166a23d6cb4ce4e9b82")

def get_token():
    url = 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal'
    resp = requests.post(url, json={'app_id': APP_ID, 'app_secret': APP_SECRET}, timeout=10)
    if resp.json().get('code') == 0:
        return resp.json()['tenant_access_token']
    return None

def upload_image(token, image_path):
    url = 'https://open.feishu.cn/open-apis/im/v1/images'
    headers = {'Authorization': f'Bearer {token}'}
    with open(image_path, 'rb') as f:
        files = {'image': ('image.jpg', f, 'image/jpeg')}
        data = {'image_type': 'message'}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=30)
    result = resp.json()
    if result.get('code') == 0:
        return result['data']['image_key']
    return None

def send_image(token, chat_id, image_key):
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
    return resp.json().get('code') == 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 send_image_to_feishu.py <图片路径>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    token = get_token()
    if token:
        image_key = upload_image(token, image_path)
        if image_key:
            if send_image(token, CHAT_ID, image_key):
                print("✅ 图片发送成功")
                sys.exit(0)
    
    print("❌ 发送失败")
    sys.exit(1)
