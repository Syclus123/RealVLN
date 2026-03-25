# Voice-Agent

支持意图、打断、聊天等简易的voice-agent

## 模式说明

目前支持两种模式：
1. 语音对话实时链路(thinker-talker): 通过事件回调的方式 获取asr结果经过llm进行意图路由，执行(chat/nav/vqa/follow_person/guard_dog/audio_track/stop_task)等意图的设定行为
2. 本地简易模式: 本地server部署实现vad + asr的功能，然后通过事件回调的方式，获取asr结果经过llm进行意图路由，执行(nav/vqa/follow_person/guard_dog/audio_track/stop_task)等意图的设定行为

两种方式的区别点在于模式1走云端语音对话实时链路，本身就支持chat, asr和vad均由云端服务来做，模式2走本地部署的vad和asr, 灵活可控性较高，延时更低。

```
模式1: bash run_go2_realtime.sh
模式2: bash run_go2_fast.sh
```

## 环境说明

### python环境
整个repo 环境依赖都基于uv进行管理，在run_go2_realtime.sh 和 run_go2_fast.sh 都增加了环境自动检测和安装，执行一下即可。

### 账号资源
因整个链路需要调用一些模型，账号ak: https://git.woa.com/kevinkhu/RealtimeAPI-sdk-python/blob/main/.env#L3

目前llm调用qwen3-30b-a3b-instruct-2507, vlm调用qwen3.5-plus, asr调用qwen3-asr-flash, 以上三个都使用的是阿里云百炼的我个人账户的免费额度, tts调用的是腾讯云账号ak

请注意⚠️隐私和权限管理，如后续需更换模型或额度消耗完，修改ak和model_name即可


#### 模型调用修改路径
- llm: https://git.woa.com/kevinkhu/RealtimeAPI-sdk-python/blob/main/src/intentllm.py#L9
- vlm: https://git.woa.com/kevinkhu/RealtimeAPI-sdk-python/blob/main/src/vlm.py
- asr: https://git.woa.com/kevinkhu/RealtimeAPI-sdk-python/blob/main/src/audio_server/asr.py#L78
- tts: https://git.woa.com/kevinkhu/RealtimeAPI-sdk-python/blob/main/src/tts_client.py#L30


## 开机自启配置

把voice-agent变成systemd的一个服务是尝试下来最稳定，不需要过多考虑环境网络处理的方式，具体步骤见下：

sudo vim /etc/systemd/system/voice-agent.service

```
[Unit]
Description=Unitree Go2 Realtime Voice Agent
After=network.target sound.target

[Service]
User=unitree
Group=unitree
WorkingDirectory=/home/unitree/VoiceAgent/master

Environment="PATH=/home/unitree/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ExecStart=/bin/bash run_go2_realtime.sh

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```
sudo systemctl daemon-reload

sudo systemctl restart voice-agent.service


kevinkhu

2026.03.23