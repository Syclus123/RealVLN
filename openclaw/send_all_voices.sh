#!/bin/bash
# 批量发送音色样本

CHAT_ID="ou_da6b544f17a5b166a23d6cb4ce4e9b82"
VOICE_DIR="/home/unitree/.openclaw/media"

# 音色列表
voices=(
    "195810.mp3:101001 - 智瑜（温柔女声）"
    "195836.mp3:101002 - 智聆（标准女声）"
    "195837.mp3:101003 - 智美（活泼女声）"
    "195838.mp3:101004 - 智云（沉稳男声）"
    "195839.mp3:101005 - 智莉（甜美女声）"
    "195840.mp3:101006 - 智言（知性女声）"
    "195841.mp3:101007 - 智娜（方言女声）"
    "195843.mp3:101008 - 智琪（新闻女声）"
    "195845.mp3:101009 - 智芸（可爱女声）"
    "195846.mp3:101010 - 智华（磁性男声）"
    "195846_2.mp3:101011 - 智燕（严厉女声）"
    "195847.mp3:101012 - 智丹（柔和女声）"
    "195848.mp3:101013 - 智霞（自然女声）"
    "195849.mp3:101014 - 智婷（活力女声）"
    "195850.mp3:101015 - 智刚（硬朗男声）"
    "195851.mp3:101016 - 智瑞（沉稳男声）"
    "195852.mp3:101017 - 智萌（童声）"
    "195853.mp3:101018 - 智栋（青年男声）"
    "195854.mp3:101019 - 智杰（成熟男声）"
    "195855.mp3:101020 - 智浩（亲切男声）"
)

count=1
total=20

for item in "${voices[@]}"; do
    IFS=':' read -r file name <<< "$item"
    file_path="$VOICE_DIR/tts_tencent_20260306_$file"
    
    echo "[$count/$total] 发送: $name"
    
    # 发送语音
    python3 /home/unitree/openclaw/skills/go2-feishu-audio/scripts/send_voice.py \
        "$file_path" \
        --chat-id "$CHAT_ID" \
        --duration 3
    
    sleep 1
    ((count++))
done

echo "全部发送完成！"
