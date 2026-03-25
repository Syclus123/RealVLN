#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 TTS Tencent - 腾讯云语音合成技能
使用腾讯云实时语音合成 WebSocket API
"""
import sys
import os
import argparse
import websocket
import time
import hmac
import hashlib
import base64
import urllib.parse
import json
import uuid
import ssl
from datetime import datetime

# ================= 腾讯云配置 =================
# 在腾讯云控制台获取：https://console.cloud.tencent.com/cam/capi
APPID = "1325896784"
SECRET_ID = "AKIDCsKYYIn46tRlriKhpibGgrmTzHK8h16M"
SECRET_KEY = "4sSowpTt77tSq2zjMmzzmMjsnbCYfYDg"

# 默认音色配置
DEFAULT_VOICE_TYPE = 101016  # 智瑞 - 沉稳男声 (用户指定默认)
DEFAULT_CODEC = "mp3"        # 默认格式 mp3
DEFAULT_SAMPLE_RATE = 16000


class TencentTTS:
    """腾讯云语音合成客户端"""
    
    def __init__(self, appid=APPID, secret_id=SECRET_ID, secret_key=SECRET_KEY):
        self.appid = appid
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.finished = False
        self.output_file = ""
        self.audio_data = b""
        
    def generate_url(self, text, voice_type=DEFAULT_VOICE_TYPE, codec=DEFAULT_CODEC, sample_rate=DEFAULT_SAMPLE_RATE):
        """生成带签名的 wss 请求地址"""
        # 基础请求参数
        params = {
            "Action": "TextToStreamAudioWS",
            "AppId": int(self.appid),
            "SecretId": self.secret_id,
            "Timestamp": int(time.time()),
            "Expired": int(time.time()) + 86400,
            "SessionId": str(uuid.uuid1()),
            "Text": text,
            "VoiceType": voice_type,
            "Codec": codec,
            "SampleRate": sample_rate,
            "EnableSubtitle": "True",
            "Volume": 0,
            "Speed": 0,
        }

        # 1. 对参数按字典序排序
        sorted_params = sorted(params.items(), key=lambda d: d[0])
        
        # 2. 拼接签名原文串
        query_str = "&".join([f"{k}={v}" for k, v in sorted_params])
        sign_source = f"GETtts.cloud.tencent.com/stream_ws?{query_str}"

        # 3. HMAC-SHA1 加密并 Base64 编码
        hmac_code = hmac.new(self.secret_key.encode('utf-8'), sign_source.encode('utf-8'), hashlib.sha1).digest()
        signature = base64.b64encode(hmac_code).decode('utf-8')

        # 4. 将所有参数（包括签名）进行 URL 编码并构建最终 URL
        params["Signature"] = signature
        final_query = urllib.parse.urlencode(params)
        return f"wss://tts.cloud.tencent.com/stream_ws?{final_query}"

    def on_message(self, ws, message):
        # 文本帧：返回状态、字幕时间戳等
        if isinstance(message, str):
            resp = json.loads(message)
            if resp.get("code") != 0:
                print(f"\n❌ 错误: {resp.get('message')}", file=sys.stderr)
                ws.close()
                return

            if resp.get("final") == 1:
                print(f"\n✅ 合成完毕，文件已保存: {self.output_file}")
                self.finished = True
                ws.close()
            else:
                # 显示正在合成的文字
                res = resp.get("result", {})
                if res and res.get("subtitles"):
                    for s in res["subtitles"]:
                        print(f"🎤 正在合成: {s['Text']}", end='\r')

        # 二进制帧：音频数据片段
        else:
            self.audio_data += message
            with open(self.output_file, "ab") as f:
                f.write(message)

    def on_error(self, ws, error):
        print(f"\n❌ WebSocket 错误: {error}", file=sys.stderr)

    def on_close(self, ws, close_status_code, close_msg):
        if not self.finished:
            print(f"\n⚠️ WebSocket 连接关闭 (code: {close_status_code})")

    def synthesize(self, text, output_file, voice_type=DEFAULT_VOICE_TYPE, codec=DEFAULT_CODEC):
        """
        合成语音
        
        Args:
            text: 要合成的文本
            output_file: 输出文件路径
            voice_type: 音色ID
            codec: 音频格式 (pcm 或 mp3)
        
        Returns:
            成功返回 output_file，失败返回 None
        """
        self.output_file = output_file
        self.finished = False
        self.audio_data = b""

        # 清空旧文件
        with open(self.output_file, "wb") as f:
            pass
        
        request_url = self.generate_url(text, voice_type, codec)
        ws = websocket.WebSocketApp(
            request_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        print(f"🎤 正在合成语音...")
        print(f"   文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"   音色: {voice_type}")
        print(f"   格式: {codec}")
        
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        
        if self.finished and os.path.exists(self.output_file):
            size = os.path.getsize(self.output_file)
            print(f"📊 文件大小: {size} bytes")
            print(f"MEDIA: {self.output_file}")
            return self.output_file
        else:
            return None


def text_to_speech(text, output_path=None, voice_type=DEFAULT_VOICE_TYPE, codec=DEFAULT_CODEC):
    """
    将文本转为语音（便捷函数）
    
    Args:
        text: 要合成的文本
        output_path: 输出音频文件路径（默认自动生成）
        voice_type: 音色ID
        codec: 音频格式 (pcm 或 mp3)
    
    Returns:
        音频文件路径，失败返回 None
    """
    
    if not text or not text.strip():
        print("❌ 文本不能为空", file=sys.stderr)
        return None
    
    # 自动生成输出路径
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.expanduser("~/.openclaw/media")
        os.makedirs(base_dir, exist_ok=True)
        output_path = os.path.join(base_dir, f"tts_tencent_{timestamp}.{codec}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 创建 TTS 客户端并合成
    tts = TencentTTS()
    return tts.synthesize(text, output_path, voice_type, codec)


def list_voices():
    """列出腾讯云常用的音色"""
    voices = {
        101001: "智瑜 - 温柔女声",
        101002: "智聆 - 标准女声",
        101003: "智美 - 活泼女声",
        101004: "智云 - 沉稳男声",
        101005: "智莉 - 甜美女声",
        101006: "智言 - 知性女声",
        101007: "智娜 - 方言女声",
        101008: "智琪 - 新闻女声",
        101009: "智芸 - 可爱女声",
        101010: "智华 - 磁性男声",
        101011: "智燕 - 严厉女声",
        101012: "智丹 - 柔和女声",
        101013: "智霞 - 自然女声",
        101014: "智婷 - 活力女声",
        101015: "智刚 - 硬朗男声",
        101016: "智瑞 - 沉稳男声",
        101017: "智萌 - 童声",
        101018: "智栋 - 青年男声",
        101019: "智杰 - 成熟男声",
        101020: "智浩 - 亲切男声",
    }
    
    print("🎙️ 腾讯云常用音色列表：")
    print()
    for voice_id, desc in voices.items():
        marker = " ⭐" if voice_id == DEFAULT_VOICE_TYPE else ""
        print(f"  {voice_id:6} - {desc}{marker}")
    print()
    print(f"使用 --voice 参数指定音色，如：--voice 101010")


def main():
    parser = argparse.ArgumentParser(description="Go2 腾讯云语音合成 (TTS)")
    parser.add_argument("text", nargs="?", help="要合成的文本")
    parser.add_argument("--output", "-o", help="输出音频文件路径")
    parser.add_argument("--voice", "-v", type=int, default=DEFAULT_VOICE_TYPE, 
                        help=f"音色ID（默认: {DEFAULT_VOICE_TYPE}）")
    parser.add_argument("--codec", "-c", default=DEFAULT_CODEC, 
                        choices=["mp3", "pcm"], help=f"音频格式（默认: {DEFAULT_CODEC}）")
    parser.add_argument("--list-voices", "-l", action="store_true", help="列出可用音色")
    
    args = parser.parse_args()
    
    if args.list_voices:
        list_voices()
        sys.exit(0)
    
    if not args.text:
        print("❌ 请提供要合成的文本，或使用 --list-voices 查看音色列表", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    result = text_to_speech(
        text=args.text,
        output_path=args.output,
        voice_type=args.voice,
        codec=args.codec
    )
    
    if result:
        print(result)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
