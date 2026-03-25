#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 ASR - 语音识别技能
使用 DashScope qwen3-asr-flash 模型进行语音识别
"""
import sys
import os
import argparse
import requests
import base64
from pathlib import Path

# DashScope API 配置
DASHSCOPE_API_KEY = "sk-5a03c62b9a1549f8bec8fba654f3fd52"
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


def transcribe_audio(audio_path_or_url, api_key=DASHSCOPE_API_KEY):
    """
    识别音频文件内容
    
    Args:
        audio_path_or_url: 音频文件路径 (本地文件) 或 URL
        api_key: DashScope API Key
    
    Returns:
        识别的文本内容，失败返回 None
    """
    
    # 判断是本地文件还是 URL
    if audio_path_or_url.startswith("http://") or audio_path_or_url.startswith("https://"):
        # 直接使用 URL
        audio_content = audio_path_or_url
        is_url = True
    else:
        # 本地文件，读取并转为 base64
        audio_path = os.path.expanduser(audio_path_or_url)
        if not os.path.exists(audio_path):
            print(f"❌ 音频文件不存在: {audio_path}", file=sys.stderr)
            return None
        
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        # 转换为 base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        audio_content = f"data:audio/ogg;base64,{audio_base64}"
        is_url = False
    
    # 构建请求体
    payload = {
        "model": "qwen3-asr-flash",
        "input": {
            "messages": [
                {
                    "content": [{"text": ""}],
                    "role": "system"
                },
                {
                    "content": [{"audio": audio_content}],
                    "role": "user"
                }
            ]
        },
        "parameters": {
            "asr_options": {
                "enable_itn": False
            }
        }
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"🎤 正在识别音频...")
        response = requests.post(DASHSCOPE_API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ API 请求失败: HTTP {response.status_code}", file=sys.stderr)
            print(f"响应: {response.text}", file=sys.stderr)
            return None
        
        result = response.json()
        
        # 解析响应
        if "output" in result and "choices" in result["output"]:
            choices = result["output"]["choices"]
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                content = message.get("content", [])
                if content and len(content) > 0:
                    text = content[0].get("text", "")
                    print(f"✅ 识别成功!")
                    print(f"📝 识别结果: {text}")
                    return text
        
        # 如果没找到预期的结构，打印完整响应用于调试
        print(f"⚠️ 无法解析响应: {result}", file=sys.stderr)
        return None
        
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时", file=sys.stderr)
        return None
    except Exception as e:
        print(f"❌ 请求错误: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Go2 语音识别 - DashScope ASR")
    parser.add_argument("audio", help="音频文件路径或 URL")
    parser.add_argument("--api-key", "-k", default=DASHSCOPE_API_KEY, help="DashScope API Key")
    
    args = parser.parse_args()
    
    result = transcribe_audio(args.audio, args.api_key)
    
    if result:
        print(result)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
