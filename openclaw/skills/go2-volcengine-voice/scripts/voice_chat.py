#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 Volcengine Voice - 火山引擎实时语音对话
支持语音输入、大模型对话、语音输出的完整流程
"""
import sys
import os
import argparse
import asyncio
import json
import base64
import time
import wave
import tempfile
from pathlib import Path

import requests
import sounddevice as sd
import numpy as np
import websockets
from websockets.client import WebSocketClientProtocol

# ================= 火山引擎配置 =================
# 请在火山引擎控制台获取: https://console.volcengine.com/
VOLCENGINE_API_KEY = os.environ.get("VOLCENGINE_API_KEY", "")
VOLCENGINE_APP_ID = os.environ.get("VOLCENGINE_APP_ID", "")
VOLCENGINE_ACCESS_TOKEN = os.environ.get("VOLCENGINE_ACCESS_TOKEN", "")

# API 地址
ASR_API_URL = "https://openspeech.bytedance.com/api/v1/auc"
TTS_API_URL = "https://openspeech.bytedance.com/api/v1/tts"
CHAT_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

# 机器狗音频配置
AUDIO_INPUT_DEVICE_ID = 24   # 麦克风
AUDIO_OUTPUT_DEVICE_ID = 24  # 扬声器
HW_SAMPLE_RATE = 48000
WS_SAMPLE_RATE = 16000
FRAME_DURATION_MS = 20


class VolcengineVoiceClient:
    """火山引擎语音对话客户端"""
    
    def __init__(self, api_key=None, app_id=None, access_token=None):
        self.api_key = api_key or VOLCENGINE_API_KEY
        self.app_id = app_id or VOLCENGINE_APP_ID
        self.access_token = access_token or VOLCENGINE_ACCESS_TOKEN
        
        if not self.api_key:
            raise ValueError("请设置 VOLCENGINE_API_KEY 环境变量或在初始化时提供")
        
    # ================= ASR 语音识别 =================
    
    def speech_to_text(self, audio_path):
        """
        语音转文字 (ASR)
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            识别文本
        """
        print(f"🎤 正在识别语音: {audio_path}")
        
        # 读取音频文件
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        # 构建请求
        headers = {
            "Authorization": f"Bearer; {self.access_token}",
            "Content-Type": "application/json",
            "Resource-Id": "volc.engine_asr_common"
        }
        
        # 将音频转为 base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        payload = {
            "appid": self.app_id,
            "token": "access_token",
            "cluster": "volcengine_common",
            "audio_format": "wav",
            "sample_rate": 16000,
            "audio_data": audio_base64
        }
        
        try:
            response = requests.post(ASR_API_URL, json=payload, headers=headers, timeout=30)
            result = response.json()
            
            if result.get("code") == 1000:
                text = result.get("result", [{}])[0].get("text", "")
                print(f"✅ 识别成功: {text}")
                return text
            else:
                print(f"❌ 识别失败: {result}")
                return None
                
        except Exception as e:
            print(f"❌ ASR 请求错误: {e}")
            return None
    
    # ================= Chat 大模型对话 =================
    
    def chat(self, message, model="doubao-lite-4k"):
        """
        大模型对话
        
        Args:
            message: 用户消息
            model: 模型名称
        
        Returns:
            AI 回复文本
        """
        print(f"💬 正在对话: {message[:50]}...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个 helpful 的 AI 助手"},
                {"role": "user", "content": message}
            ]
        }
        
        try:
            response = requests.post(CHAT_API_URL, json=payload, headers=headers, timeout=30)
            result = response.json()
            
            if "choices" in result:
                reply = result["choices"][0]["message"]["content"]
                print(f"✅ 对话成功: {reply[:50]}...")
                return reply
            else:
                print(f"❌ 对话失败: {result}")
                return None
                
        except Exception as e:
            print(f"❌ Chat 请求错误: {e}")
            return None
    
    # ================= TTS 语音合成 =================
    
    def text_to_speech(self, text, output_path, voice_type="BV001_streaming"):
        """
        文字转语音 (TTS)
        
        Args:
            text: 要合成的文本
            output_path: 输出音频文件路径
            voice_type: 音色类型
        
        Returns:
            成功返回 True
        """
        print(f"🔊 正在合成语音: {text[:50]}...")
        
        headers = {
            "Authorization": f"Bearer; {self.access_token}",
            "Content-Type": "application/json",
            "Resource-Id": "volc.engine_tts_common"
        }
        
        payload = {
            "appid": self.app_id,
            "token": "access_token",
            "cluster": "volcengine_common",
            "audio_type": "wav",
            "voice_type": voice_type,
            "text": text
        }
        
        try:
            response = requests.post(TTS_API_URL, json=payload, headers=headers, timeout=30)
            result = response.json()
            
            if result.get("code") == 1000:
                # 解码音频数据
                audio_base64 = result.get("data")
                if audio_base64:
                    audio_data = base64.b64decode(audio_base64)
                    
                    # 保存音频文件
                    with open(output_path, 'wb') as f:
                        f.write(audio_data)
                    
                    print(f"✅ 合成成功: {output_path}")
                    return True
            
            print(f"❌ 合成失败: {result}")
            return False
            
        except Exception as e:
            print(f"❌ TTS 请求错误: {e}")
            return False
    
    # ================= 完整对话流程 =================
    
    def voice_chat(self, input_audio_path, output_audio_path):
        """
        语音对话完整流程：语音输入 -> ASR -> Chat -> TTS -> 语音输出
        
        Args:
            input_audio_path: 输入语音文件路径
            output_audio_path: 输出语音文件路径
        
        Returns:
            (识别文本, AI回复文本, 是否成功)
        """
        print("\n" + "="*50)
        print("🎙️ 开始语音对话")
        print("="*50)
        
        # 1. ASR 语音识别
        user_text = self.speech_to_text(input_audio_path)
        if not user_text:
            return None, None, False
        
        # 2. Chat 大模型对话
        ai_reply = self.chat(user_text)
        if not ai_reply:
            return user_text, None, False
        
        # 3. TTS 语音合成
        success = self.text_to_speech(ai_reply, output_audio_path)
        
        print("="*50)
        
        return user_text, ai_reply, success


def play_audio(file_path, device_id=AUDIO_OUTPUT_DEVICE_ID):
    """播放音频文件"""
    try:
        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 重采样到 48kHz
            if sample_rate == 16000:
                audio_array = np.repeat(audio_array, 3)
            
            stream = sd.OutputStream(
                device=device_id,
                samplerate=HW_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=960
            )
            stream.start()
            stream.write(audio_array)
            sd.wait()
            stream.stop()
            stream.close()
            
    except Exception as e:
        print(f"⚠️ 播放错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="火山引擎语音对话")
    parser.add_argument("input", nargs="?", help="输入语音文件路径")
    parser.add_argument("--output", "-o", help="输出语音文件路径")
    parser.add_argument("--api-key", help="火山引擎 API Key")
    parser.add_argument("--app-id", help="火山引擎 App ID")
    parser.add_argument("--access-token", help="火山引擎 Access Token")
    parser.add_argument("--play", "-p", action="store_true", help="播放输出语音")
    
    args = parser.parse_args()
    
    # 检查 API 配置
    api_key = args.api_key or VOLCENGINE_API_KEY
    app_id = args.app_id or VOLCENGINE_APP_ID
    access_token = args.access_token or VOLCENGINE_ACCESS_TOKEN
    
    if not api_key:
        print("❌ 请设置环境变量 VOLCENGINE_API_KEY 或使用 --api-key 参数")
        print("   获取地址: https://console.volcengine.com/")
        sys.exit(1)
    
    # 创建客户端
    client = VolcengineVoiceClient(api_key, app_id, access_token)
    
    if args.input:
        # 语音对话模式
        output_path = args.output or "/tmp/volcengine_reply.wav"
        user_text, ai_reply, success = client.voice_chat(args.input, output_path)
        
        if success:
            print(f"\n👤 你说: {user_text}")
            print(f"🤖 AI 回复: {ai_reply}")
            print(f"🔊 输出音频: {output_path}")
            
            if args.play:
                print("🎵 正在播放...")
                play_audio(output_path)
        else:
            print("❌ 语音对话失败")
            sys.exit(1)
    else:
        print("请提供输入语音文件路径")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
