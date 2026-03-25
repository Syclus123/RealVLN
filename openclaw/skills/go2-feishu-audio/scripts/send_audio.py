#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 Feishu Audio - 发送飞书语音消息
将音频文件作为语音消息发送到飞书聊天（可直接播放）
"""
import sys
import os
import argparse
import requests
import json
import time

# 飞书机器人配置
# 这些值需要从飞书开发者后台获取：https://open.feishu.cn/app
FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID", "cli_a9232fdff3795cc6")
FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "4YL27fguAn1Wb8pLbdjPngKWWKXfDjNo")

# API 基础 URL
FEISHU_API_BASE = "https://open.feishu.cn/open-apis"


class FeishuAudioSender:
    """飞书语音消息发送器"""
    
    def __init__(self, app_id=None, app_secret=None):
        self.app_id = app_id or FEISHU_APP_ID
        self.app_secret = app_secret or FEISHU_APP_SECRET
        self.tenant_access_token = None
        self.token_expire_time = 0
        
    def get_tenant_access_token(self):
        """获取 tenant_access_token"""
        # 如果 token 还有效，直接返回
        if self.tenant_access_token and time.time() < self.token_expire_time:
            return self.tenant_access_token
        
        url = f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal"
        
        headers = {
            "Content-Type": "application/json; charset=utf-8"
        }
        
        data = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }
        
        try:
            print(f"🔑 正在获取 tenant_access_token...")
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ 获取 token 失败: HTTP {response.status_code}", file=sys.stderr)
                print(f"响应: {response.text}", file=sys.stderr)
                return None
            
            result = response.json()
            
            if result.get("code") != 0:
                print(f"❌ 获取 token 失败: {result.get('msg')}", file=sys.stderr)
                return None
            
            self.tenant_access_token = result.get("tenant_access_token")
            # token 有效期 2 小时，提前 5 分钟刷新
            self.token_expire_time = time.time() + result.get("expire", 7200) - 300
            
            print(f"✅ 获取 token 成功")
            return self.tenant_access_token
            
        except Exception as e:
            print(f"❌ 获取 token 错误: {e}", file=sys.stderr)
            return None
    
    def upload_audio(self, audio_path):
        """
        上传音频文件到飞书
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            file_key: 文件标识，失败返回 None
        """
        token = self.get_tenant_access_token()
        if not token:
            return None
        
        url = f"{FEISHU_API_BASE}/im/v1/files"
        
        headers = {
            "Authorization": f"Bearer {token}"
        }
        
        # 准备文件
        audio_path = os.path.expanduser(audio_path)
        if not os.path.exists(audio_path):
            print(f"❌ 音频文件不存在: {audio_path}", file=sys.stderr)
            return None
        
        # 获取文件名和大小
        filename = os.path.basename(audio_path)
        file_size = os.path.getsize(audio_path)
        
        # 准备表单数据
        data = {
            "file_type": "opus",  # 飞书语音消息只支持 opus 格式
            "file_name": filename
        }
        
        try:
            print(f"📤 正在上传音频文件...")
            print(f"   文件: {filename}")
            print(f"   大小: {file_size} bytes")
            
            with open(audio_path, "rb") as f:
                files = {"file": (filename, f, "audio/opus")}
                response = requests.post(url, headers=headers, data=data, files=files, timeout=60)
            
            if response.status_code != 200:
                print(f"❌ 上传失败: HTTP {response.status_code}", file=sys.stderr)
                print(f"响应: {response.text}", file=sys.stderr)
                return None
            
            result = response.json()
            
            if result.get("code") != 0:
                print(f"❌ 上传失败: {result.get('msg')}", file=sys.stderr)
                return None
            
            file_key = result.get("data", {}).get("file_key")
            print(f"✅ 上传成功，file_key: {file_key}")
            return file_key
            
        except Exception as e:
            print(f"❌ 上传错误: {e}", file=sys.stderr)
            return None
    
    def send_audio_message(self, chat_id, file_key, duration=0):
        """
        发送语音消息
        
        Args:
            chat_id: 聊天 ID（用户的 open_id）
            file_key: 音频文件的 file_key
            duration: 音频时长（秒）
        
        Returns:
            成功返回 True，失败返回 False
        """
        token = self.get_tenant_access_token()
        if not token:
            return False
        
        url = f"{FEISHU_API_BASE}/im/v1/messages?receive_id_type=open_id"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        # 构建音频消息内容
        content = {
            "file_key": file_key,
            "duration": duration
        }
        
        data = {
            "receive_id": chat_id,
            "msg_type": "audio",
            "content": json.dumps(content)
        }
        
        try:
            print(f"📨 正在发送语音消息...")
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ 发送失败: HTTP {response.status_code}", file=sys.stderr)
                print(f"响应: {response.text}", file=sys.stderr)
                return False
            
            result = response.json()
            
            if result.get("code") != 0:
                print(f"❌ 发送失败: {result.get('msg')}", file=sys.stderr)
                return False
            
            print(f"✅ 语音消息发送成功！")
            return True
            
        except Exception as e:
            print(f"❌ 发送错误: {e}", file=sys.stderr)
            return False
    
    def send_audio(self, audio_path, chat_id, duration=0):
        """
        完整的音频发送流程：上传 + 发送
        
        Args:
            audio_path: 音频文件路径
            chat_id: 聊天 ID
            duration: 音频时长（秒）
        
        Returns:
            成功返回 True，失败返回 False
        """
        print(f"🎤 开始发送语音消息...")
        print(f"   文件: {audio_path}")
        print(f"   目标: {chat_id}")
        
        # 1. 上传音频文件
        file_key = self.upload_audio(audio_path)
        if not file_key:
            return False
        
        # 2. 发送语音消息
        return self.send_audio_message(chat_id, file_key, duration)


def get_chat_id_from_env():
    """从环境变量获取当前聊天 ID"""
    # 尝试多种可能的环境变量名
    for key in ["FEISHU_CHAT_ID", "FEISHU_USER_ID", "FEISHU_OPEN_ID", "OPENCLAW_CHAT_ID"]:
        value = os.environ.get(key)
        if value:
            return value
    return None


def main():
    parser = argparse.ArgumentParser(description="发送飞书语音消息")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("--chat-id", "-c", help="目标聊天 ID（默认从环境变量获取）")
    parser.add_argument("--duration", "-d", type=int, default=0, help="音频时长（秒）")
    parser.add_argument("--app-id", help="飞书 App ID")
    parser.add_argument("--app-secret", help="飞书 App Secret")
    
    args = parser.parse_args()
    
    # 获取 chat_id
    chat_id = args.chat_id or get_chat_id_from_env()
    if not chat_id:
        print("❌ 请提供 chat_id 或设置环境变量 FEISHU_CHAT_ID", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # 创建发送器
    sender = FeishuAudioSender(app_id=args.app_id, app_secret=args.app_secret)
    
    # 发送音频
    success = sender.send_audio(args.audio, chat_id, args.duration)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
