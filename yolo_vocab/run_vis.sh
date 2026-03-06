#!/usr/bin/env bash
set -euo pipefail

# 直接给出两条可运行命令（复制到两个终端执行）


===== 复制下面两条命令分别在两个终端运行 =====

[Server]
python3 -u ./yolo_detect_server_realtime_vis.py \
    --host 127.0.0.1 \
    --port 5802 \
    --model yolov8x-worldv2.pt \
    --cam-json ./cam_params.json \
    --vocab-file ./yolo_vocab/obj365v1_class_texts.json  \
    --conf 0.35 \
    --show-window \
    --show-depth
  #  --vocab box,chair,person \
  #  --vocab-file ./yolo_vocab/obj365v1_class_texts.json   ./yolo_vocab/coco_class_texts.json
[Client]
python3 -u ./yolo_detect_client.py \
    --server-url http://127.0.0.1:5802/detect \
    --cam-json ./cam_params.json \
    --rgb-topic /camera/camera/color/image_raw \
    --depth-topic /camera/camera/aligned_depth_to_color/image_raw \
    --odom-topic /odom_bridge \
    --frame-stride 10 \
    --rate 0.3

