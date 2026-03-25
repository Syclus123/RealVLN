#!/usr/bin/env python3
"""批量合成音色样本"""
import subprocess
import sys

# 腾讯云 TTS 音色列表
voices = [
    (101001, "智瑜 - 温柔女声"),
    (101002, "智聆 - 标准女声"),
    (101003, "智美 - 活泼女声"),
    (101004, "智云 - 沉稳男声"),
    (101005, "智莉 - 甜美女声"),
    (101006, "智言 - 知性女声"),
    (101007, "智娜 - 方言女声"),
    (101008, "智琪 - 新闻女声"),
    (101009, "智芸 - 可爱女声"),
    (101010, "智华 - 磁性男声"),
    (101011, "智燕 - 严厉女声"),
    (101012, "智丹 - 柔和女声"),
    (101013, "智霞 - 自然女声"),
    (101014, "智婷 - 活力女声"),
    (101015, "智刚 - 硬朗男声"),
    (101016, "智瑞 - 沉稳男声"),
    (101017, "智萌 - 童声"),
    (101018, "智栋 - 青年男声"),
    (101019, "智杰 - 成熟男声"),
    (101020, "智浩 - 亲切男声"),
]

text = "你好，我是勾勾，一只机器狗。"

for voice_id, name in voices:
    print(f"\n🎤 正在合成: {voice_id} - {name}")
    result = subprocess.run([
        "python3", "/home/unitree/openclaw/skills/go2-tts/scripts/synthesize.py",
        text,
        "--voice", str(voice_id)
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        # 提取文件路径
        for line in result.stdout.split("\n"):
            if line.startswith("/home/unitree"):
                print(f"   文件: {line}")
                break
    else:
        print(f"   ❌ 失败: {result.stderr}")
