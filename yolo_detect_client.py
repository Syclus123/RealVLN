#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 目标检测 HTTP 客户端（ROS2 版），含交互式语义导航目标查询 + Web UI。

功能：
  - ROS2 订阅 RGB + Depth（时间同步）+ Odometry，逐帧发送给 yolo_detect_server
  - 后台 detection_thread 持续检测，结果发布到 /yolo_detect/results
  - interactive_thread 支持在终端交互式输入目标类别，自动：
      1. 查询 server 中已追踪到的对应物体
      2. 用 TF (map→base_link) 获取机器人当前位置（失败则回退到 odom）
      3. 选取最近的 track，按 standoff 距离计算导航目标点
      4. 发布 PoseStamped 到 /goal_pose（Nav2 兼容）
  - 自然语言意图解析模式（需 --deepseek-api-key）：
      输入中文或自然语言描述（如"请寻找一把椅子"），
      自动调用 DeepSeek API 从当前 YOLO 检测类别列表中匹配最合适的物体名，
      再执行查询与导航目标发布。
  - Web UI（--webui-port 指定端口，默认 8080）：
      浏览器访问 http://<host>:<port> 即可看到实时画面、检测信息和 query 输入框。

启动：
    python -u yolo_detect_client.py \
        --server-url http://127.0.0.1:5802/detect \
        --cam-json cam_params.json \
        --frame-stride 10 \
        --goal-standoff 0.6 \
        --map-frame map \
        --base-link-frame base_link \
        --deepseek-api-key <YOUR_API_KEY> \
        --webui-port 8080

交互命令（运行后在终端输入）：
    chair                  直接查询 "chair" 类别，找到后发布导航目标
    请帮我找一把椅子       自然语言模式：DeepSeek 解析后匹配类别，再发布导航目标
    list                   列出 server 中所有已追踪物体类别
    q / quit               退出交互线程（检测仍继续）
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import math
import sys
import threading
import time
from collections import deque

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import cv2
import numpy as np
import rclpy
import requests
from rclpy.action import ActionClient
from cv_bridge import CvBridge
from flask import Flask, Response, jsonify, request as flask_request
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from PIL import Image as PIL_Image
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from action_msgs.msg import GoalStatus
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

try:
    import sensor_msgs_py.point_cloud2 as pc2
    _HAS_PC2 = True
except ImportError:
    _HAS_PC2 = False
    print("[Warn] sensor_msgs_py 未安装，PointCloud2 订阅无效")

from thread_utils import ReadWriteLock


# -------------------------------------------
# HTTP helper
# -------------------------------------------
def call_detect(
    image_bytes: io.BytesIO,
    depth_bytes: io.BytesIO,
    pose: np.ndarray,
    reset: bool,
    url: str,
    timeout: float = 100.0,
    lidar_points_cam: np.ndarray | None = None,
) -> dict:
    """
    向 yolo_detect_server 发送一帧数据并获取检测结果。

    Parameters
    ----------
    image_bytes      : io.BytesIO  (JPEG)
    depth_bytes      : io.BytesIO  (PNG, uint16)
    pose             : np.ndarray  shape (4,4) T_wc
    reset            : bool
    url              : str         /detect endpoint
    timeout          : float
    lidar_points_cam : np.ndarray | None  shape (N, 3) 相机坐标系下的激光雷达点云

    Returns
    -------
    dict
    """
    payload: dict = {
        "reset": reset,
        "pose": pose.flatten().tolist(),
    }

    json_payload = json.dumps(payload)

    files = {
        "image": ("rgb_image.jpg", image_bytes, "image/jpeg"),
        "depth": ("depth_image.png", depth_bytes, "image/png"),
    }

    if lidar_points_cam is not None and lidar_points_cam.shape[0] > 0:
        pts_f32 = lidar_points_cam.astype(np.float32)
        lidar_buf = io.BytesIO(pts_f32.tobytes())
        lidar_buf.seek(0)
        files["lidar"] = ("lidar_points.bin", lidar_buf, "application/octet-stream")

    t0 = time.time()
    response = requests.post(
        url, files=files, data={"json": json_payload}, timeout=timeout
    )
    elapsed = time.time() - t0
    lidar_info = f"  lidar_pts={lidar_points_cam.shape[0]}" if lidar_points_cam is not None else ""
    # print(f"[Client] HTTP {response.status_code}, cost {elapsed:.3f}s{lidar_info}")

    if response.status_code == 413:
        raise RuntimeError(
            f"HTTP 413 Request Entity Too Large：请求体过大（"
            f"RGB+Depth 数据超出 server MAX_CONTENT_LENGTH），请重启 server"
        )
    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code}：{response.text[:200]}"
        )
    return json.loads(response.text)


# -------------------------------------------
# Query / list HTTP helpers
# -------------------------------------------
def call_query(
    base_url: str,
    class_name: str,
    robot_x: float,
    robot_y: float,
    timeout: float = 100.0,
) -> dict:
    """
    查询 server 中指定类别的所有 track，按距机器人距离排序返回。
    base_url 为去掉 /detect 后缀的根地址，如 http://127.0.0.1:5802
    """
    resp = requests.post(
        f"{base_url}/query",
        json={"class_name": class_name, "robot_x": robot_x, "robot_y": robot_y},
        timeout=timeout,
    )
    return resp.json()


def call_list_classes(base_url: str, timeout: float = 5.0) -> dict:
    """查询 server 当前已追踪到的所有物体类别。"""
    resp = requests.get(f"{base_url}/list_classes", timeout=timeout)
    return resp.json()


# -------------------------------------------
# DeepSeek 意图解析
# -------------------------------------------
def parse_intent_with_deepseek(
    user_query: str,
    candidate_classes: list[str],
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    timeout: float = 15.0,
) -> str | None:
    """
    调用 DeepSeek API，将用户自然语言描述映射到 YOLO 检测类别列表中最匹配的一个。

    Parameters
    ----------
    user_query        : 用户输入，如 "请帮我找一把椅子"
    candidate_classes : YOLO 当前已追踪到的物体类别列表，如 ["chair", "potted plant", "box"]
    api_key           : DeepSeek API Key
    base_url          : DeepSeek API 根地址（默认 https://api.deepseek.com）
    model             : 使用的模型名（默认 deepseek-chat）
    timeout           : HTTP 请求超时（秒）

    Returns
    -------
    str | None : 最匹配的类别名（来自 candidate_classes），解析失败时返回 None
    """
    if not candidate_classes:
        return None

    candidates_str = "\n".join(f"- {c}" for c in candidate_classes)
    system_prompt = (
        "你是一个机器人语义导航助手。"
        "用户会用自然语言描述想要寻找的物体，"
        "你需要从给定的候选类别列表中选出最匹配的一个类别名称，"
        "直接输出该类别名称，不要输出任何其他内容。"
        "如果没有任何匹配的类别，输出 NONE。"
    )
    user_prompt = (
        f"用户描述：{user_query}\n\n"
        f"候选类别列表：\n{candidates_str}\n\n"
        "请从上面的候选类别中选出最匹配的一个，直接输出类别名称："
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        answer = result["choices"][0]["message"]["content"].strip()
        if answer.upper() == "NONE" or not answer:
            return None
        # 校验返回值是否在候选列表中（大小写不敏感）
        answer_lower = answer.lower()
        for cls in candidate_classes:
            if cls.lower() == answer_lower:
                return cls
        # 若不完全匹配，尝试子串匹配（防止模型输出带引号或多余空格）
        for cls in candidate_classes:
            if cls.lower() in answer_lower or answer_lower in cls.lower():
                return cls
        # 返回原始答案，交由调用方处理
        return answer
    except Exception as e:
        print(f"[DeepSeek] API 调用失败: {e}")
        return None


# -------------------------------------------
# Camera offset helpers
# -------------------------------------------
def load_cam_offset(cam_json_path: str | None) -> tuple[float, float, float]:
    """
    从 cam_params.json 读取相机在 base_link 下的平移偏移量 (x, y, z)。
    若未提供路径，返回 (0, 0, 0)。

    cam_params.json 字段名：camera.cam-offset-x / y / z
    """
    if cam_json_path is None:
        return 0.0, 0.0, 0.0
    with open(cam_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cam = data.get("camera", data)
    ox = float(cam.get("cam-offset-x", 0.0))
    oy = float(cam.get("cam-offset-y", 0.0))
    oz = float(cam.get("cam-offset-z", 0.0))
    return ox, oy, oz


def build_T_base_cam(offset_x: float, offset_y: float, offset_z: float) -> np.ndarray:
    """
    构造 T_base_cam：相机在 base_link 坐标系下的齐次变换矩阵。
    假设相机与机器人同朝向（无旋转），仅有平移偏移。

      T_base_cam = | I  t |
                   | 0  1 |
    其中 t = [offset_x, offset_y, offset_z]^T
    """
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = offset_x
    T[1, 3] = offset_y
    T[2, 3] = offset_z
    return T


# -------------------------------------------
# Odom -> 4x4 camera pose in world frame
# ----------f---------------------------------
def odom_to_pose(
    x: float,
    y: float,
    yaw: float,
    T_base_cam: np.ndarray | None = None,
) -> np.ndarray:
    """
    将里程计 (x, y, yaw) 转换为相机在世界坐标系下的位姿矩阵 T_w_cam。

    里程计描述的是 base_link 在世界系下的位姿 T_w_base，
    相机位姿通过链式变换得到：
        T_w_cam = T_w_base @ T_base_cam

    Parameters
    ----------
    x, y, yaw   : base_link 在世界系下的位置和朝向
    T_base_cam  : 4x4 相机相对 base_link 的变换矩阵；
                  若为 None 则不做偏移修正，直接返回 T_w_base
    """
    c, s = np.cos(yaw), np.sin(yaw)
    T_w_base = np.array([
        [c, -s, 0.0, x],
        [s,  c, 0.0, y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    if T_base_cam is None:
        return T_w_base

    # T_w_cam = T_w_base @ T_base_cam
    return T_w_base @ T_base_cam


# -------------------------------------------
# Global state (same pattern as the original)
# -------------------------------------------
rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
lidar_rw_lock = ReadWriteLock()   # 对  manager.lidar_pts / lidar_time / lidar_frame 的读写保护
manager = None  # will be set to YoloDetectNode instance

# -------------------------------------------
# Web UI 全局状态
# -------------------------------------------
_webui_lock = threading.Lock()
_webui_latest_frame_jpg = None      # bytes | None，带检测框标注的 JPEG 帧
_webui_latest_detections = []       # list[dict]，最新检测结果列表
_webui_query_log = []               # list[dict]，query 交互日志（最多保留 50 条）
_webui_server_base_url = ""         # str，server 根地址，供 Web UI 查询用
_webui_goal_standoff = 0.5          # float
_webui_z_offset = 0.0               # float
_webui_map_frame = "map"            # str
_webui_deepseek_api_key = None      # str | None
_webui_deepseek_base_url = "https://api.deepseek.com"  # str
_webui_deepseek_model = "deepseek-chat"                # str

# 导航状态：由 nav2_status_callback 写入，前端轮询读取
_nav_status_lock = threading.Lock()
_nav_succeeded = False          # 是否刚刚导航成功（前端读取后重置）
_nav_succeeded_target = ""      # 本次导航成功的目标类别名
_webui_last_nav_target = ""     # 最近一次发布导航目标的类别名

# YOLO 检测框颜色表（BGR → RGB 转换后用于 cv2）
_BBOX_COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 128),
    (128, 255, 0), (0, 255, 128), (255, 0, 0), (0, 0, 255),
]

# -------------------------------------------
# Web UI HTML 模板
# -------------------------------------------
_WEBUI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YOLO 检测 · 实时监控</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border-bottom: 1px solid #2d3748;
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  header .logo {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  header h1 { font-size: 18px; font-weight: 600; color: #f7fafc; }
  header .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #48bb78; margin-left: auto;
    box-shadow: 0 0 6px #48bb78;
    animation: pulse 2s infinite;
  }
  header .status-dot.offline { background: #fc8181; box-shadow: 0 0 6px #fc8181; animation: none; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .main-layout {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 340px;
    grid-template-rows: 1fr auto;
    gap: 0;
    height: calc(100vh - 57px);
  }

  /* 左侧：视频区 */
  .video-panel {
    background: #0a0d14;
    display: flex;
    flex-direction: column;
    border-right: 1px solid #2d3748;
  }
  .video-header {
    padding: 10px 16px;
    background: #161b27;
    border-bottom: 1px solid #2d3748;
    font-size: 12px;
    color: #718096;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .video-header span { color: #a0aec0; font-weight: 500; }
  #fps-badge {
    margin-left: auto;
    background: #2d3748;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    color: #68d391;
  }
  .video-container {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
    overflow: hidden;
  }
  #live-stream {
    max-width: 100%;
    max-height: 100%;
    border-radius: 8px;
    border: 1px solid #2d3748;
    object-fit: contain;
  }
  .no-signal {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    color: #4a5568;
  }
  .no-signal .icon { font-size: 48px; }

  /* 右侧面板 */
  .side-panel {
    display: flex;
    flex-direction: column;
    background: #13192a;
    overflow: hidden;
  }

  /* 检测信息区 */
  .detections-section {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    border-bottom: 1px solid #2d3748;
  }
  .section-title {
    padding: 10px 16px;
    font-size: 12px;
    font-weight: 600;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: #161b27;
    border-bottom: 1px solid #2d3748;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .section-title .count-badge {
    background: #2b6cb0;
    color: #bee3f8;
    padding: 1px 7px;
    border-radius: 10px;
    font-size: 11px;
    margin-left: auto;
  }
  #detections-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }
  #detections-list::-webkit-scrollbar { width: 4px; }
  #detections-list::-webkit-scrollbar-track { background: transparent; }
  #detections-list::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 2px; }

  .det-item {
    background: #1a2035;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .det-item:hover { border-color: #4a90d9; background: #1e2a45; }
  .det-item .det-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }
  .det-item .class-dot {
    width: 8px; height: 8px; border-radius: 50%;
    flex-shrink: 0;
  }
  .det-item .class-name {
    font-size: 13px;
    font-weight: 600;
    color: #e2e8f0;
    flex: 1;
  }
  .det-item .conf-badge {
    font-size: 11px;
    padding: 1px 6px;
    border-radius: 8px;
    background: #276749;
    color: #9ae6b4;
  }
  .det-item .conf-badge.low { background: #744210; color: #fbd38d; }
  .det-item .det-pos {
    font-size: 11px;
    color: #718096;
    font-family: 'Courier New', monospace;
  }
  .det-item .track-id {
    font-size: 10px;
    color: #4a5568;
    margin-top: 2px;
  }
  .det-item .det-caption {
    font-size: 11px;
    color: #a0aec0;
    margin-top: 3px;
    line-height: 1.4;
    font-style: italic;
  }
  .empty-hint {
    text-align: center;
    color: #4a5568;
    font-size: 13px;
    padding: 32px 16px;
  }

  /* 对话/查询区 */
  .query-section {
    background: #161b27;
    border-top: 1px solid #2d3748;
  }
  .query-log {
    height: 180px;
    overflow-y: auto;
    padding: 8px 12px;
    font-size: 12px;
    line-height: 1.6;
  }
  .query-log::-webkit-scrollbar { width: 4px; }
  .query-log::-webkit-scrollbar-track { background: transparent; }
  .query-log::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 2px; }
  .log-entry { margin-bottom: 6px; }
  .log-entry .log-time { color: #4a5568; margin-right: 6px; }
  .log-entry.user .log-text { color: #90cdf4; }
  .log-entry.system .log-text { color: #68d391; }
  .log-entry.error .log-text { color: #fc8181; }
  .log-entry.info .log-text { color: #a0aec0; }

  .query-input-row {
    display: flex;
    gap: 8px;
    padding: 10px 12px;
    border-top: 1px solid #2d3748;
  }
  #query-input {
    flex: 1;
    background: #0f1117;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 8px 12px;
    color: #e2e8f0;
    font-size: 13px;
    outline: none;
    transition: border-color 0.15s;
  }
  #query-input:focus { border-color: #4a90d9; }
  #query-input::placeholder { color: #4a5568; }
  #query-btn {
    background: linear-gradient(135deg, #4a90d9, #357abd);
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    color: white;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s;
    white-space: nowrap;
  }
  #query-btn:hover { opacity: 0.85; }
  #query-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .quick-btns {
    display: flex;
    gap: 6px;
    padding: 0 12px 8px;
    flex-wrap: wrap;
  }
  .quick-btn {
    background: #1a2035;
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    color: #a0aec0;
    cursor: pointer;
    transition: all 0.15s;
  }
  .quick-btn:hover { border-color: #4a90d9; color: #90cdf4; }

  /* 导航成功横幅 */
  #nav-success-banner {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    background: linear-gradient(135deg, #1a4731, #22543d);
    border-bottom: 2px solid #48bb78;
    padding: 0;
    box-shadow: 0 4px 24px rgba(72, 187, 120, 0.35);
  }
  #nav-success-banner.show {
    display: flex;
    animation: slideDown 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
  }
  @keyframes slideDown {
    from { transform: translateY(-100%); opacity: 0; }
    to   { transform: translateY(0);     opacity: 1; }
  }
  @keyframes slideUp {
    from { transform: translateY(0);     opacity: 1; }
    to   { transform: translateY(-100%); opacity: 0; }
  }
  #nav-success-banner.hide {
    animation: slideUp 0.35s ease-in forwards;
  }
  .banner-inner {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    padding: 14px 24px;
  }
  .banner-icon {
    font-size: 28px;
    animation: bounceIn 0.6s ease;
  }
  @keyframes bounceIn {
    0%   { transform: scale(0.3); opacity: 0; }
    60%  { transform: scale(1.15); }
    100% { transform: scale(1); opacity: 1; }
  }
  .banner-text {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .banner-title {
    font-size: 18px;
    font-weight: 700;
    color: #68d391;
    letter-spacing: 0.03em;
  }
  .banner-sub {
    font-size: 13px;
    color: #9ae6b4;
    opacity: 0.85;
  }
  .banner-close {
    margin-left: auto;
    background: none;
    border: 1px solid #276749;
    border-radius: 6px;
    color: #68d391;
    padding: 4px 12px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .banner-close:hover { background: #276749; }

  /* 视频边框成功高亮 */
  #live-stream.nav-success {
    border-color: #48bb78;
    box-shadow: 0 0 0 3px rgba(72, 187, 120, 0.4);
    transition: border-color 0.3s, box-shadow 0.3s;
  }
</style>
</head>
<body>

<!-- 导航成功横幅（全屏顶部浮层） -->
<div id="nav-success-banner">
  <div class="banner-inner">
    <div class="banner-icon">&#x2705;</div>
    <div class="banner-text">
      <div class="banner-title">导航成功！</div>
      <div class="banner-sub" id="banner-sub-text">已到达目标位置</div>
    </div>
    <button class="banner-close" onclick="dismissBanner()">关闭</button>
  </div>
</div>

<header>
  <div class="logo">&#x1F916;</div>
  <h1>RealVLN 实时检测监控</h1>
  <div id="status-dot" class="status-dot offline"></div>
</header>

<div class="main-layout">
  <!-- 左侧视频 -->
  <div class="video-panel">
    <div class="video-header">
      &#x1F4F9; <span>实时画面</span>
      <span id="fps-badge">-- FPS</span>
    </div>
    <div class="video-container">
      <img id="live-stream" alt="视频流" style="display:none">
      <div class="no-signal" id="no-signal">
        <div class="icon">&#x1F4F7;</div>
        <div>等待视频流...</div>
      </div>
    </div>
  </div>

  <!-- 右侧面板 -->
  <div class="side-panel">
    <!-- 检测信息 -->
    <div class="detections-section">
      <div class="section-title">
        &#x1F50D; 检测结果
        <span class="count-badge" id="det-count">0</span>
      </div>
      <div id="detections-list">
        <div class="empty-hint">暂无检测结果</div>
      </div>
    </div>

    <!-- 查询对话框 -->
    <div class="query-section">
      <div class="section-title">&#x1F4AC; 语义导航查询</div>
      <div class="query-log" id="query-log">
        <div class="log-entry info">
          <span class="log-text">系统已就绪，输入目标类别或自然语言描述发布导航目标</span>
        </div>
      </div>
      <div class="quick-btns">
        <button class="quick-btn" onclick="quickQuery('list')">list 列出物体</button>
        <button class="quick-btn" onclick="quickQuery('chair')">chair</button>
        <button class="quick-btn" onclick="quickQuery('box')">box</button>
        <button class="quick-btn" onclick="quickQuery('person')">person</button>
      </div>
      <div class="query-input-row">
        <input id="query-input" type="text" placeholder="输入目标类别或自然语言（如：请找一把椅子）" />
        <button id="query-btn" onclick="submitQuery()">发送</button>
      </div>
    </div>
  </div>
</div>

<script>
const COLORS = [
  '#48bb78','#ed8936','#4299e1','#ed64a6',
  '#a0aec0','#68d391','#fc8181','#f6e05e'
];

let lastFrameTime = Date.now();
let frameCount = 0;
let fps = 0;

// ── 视频流轮询模式 ──────────────────────────────────────────────
const img = document.getElementById('live-stream');
const noSignal = document.getElementById('no-signal');
const statusDot = document.getElementById('status-dot');
let polling = false;
let hasFrame = false;

function pollSnapshot() {
  if (polling) return;
  polling = true;
  const t0 = Date.now();
  fetch('/snapshot')
    .then(r => { if (!r.ok) throw new Error(r.status); return r.blob(); })
    .then(blob => {
      const url = URL.createObjectURL(blob);
      img.onload = function() {
        URL.revokeObjectURL(url);
        if (!hasFrame) { hasFrame = true; noSignal.style.display = 'none'; img.style.display = 'block'; }
        statusDot.className = 'status-dot';
        frameCount++;
        const now = Date.now();
        if (now - lastFrameTime >= 1000) {
          fps = Math.round(frameCount * 1000 / (now - lastFrameTime));
          document.getElementById('fps-badge').textContent = fps + ' FPS';
          frameCount = 0;
          lastFrameTime = now;
        }
        polling = false;
      };
      img.onerror = function() { URL.revokeObjectURL(url); polling = false; };
      img.src = url;
    })
    .catch(() => {
      statusDot.className = 'status-dot offline';
      polling = false;
    });
}
setInterval(pollSnapshot, 100);

// ── 检测信息轮询 ────────────────────────────────────────────────
let latestCaptions = {};

async function fetchDetections() {
  try {
    const resp = await fetch('/api/detections');
    if (!resp.ok) return;
    const data = await resp.json();
    renderDetections(data.detections || []);
  } catch(e) {}
}

async function fetchCaptions() {
  try {
    const resp = await fetch('/api/captions');
    if (!resp.ok) return;
    const data = await resp.json();
    latestCaptions = data.captions || {};
  } catch(e) {}
}

function renderDetections(dets) {
  const list = document.getElementById('detections-list');
  const badge = document.getElementById('det-count');
  badge.textContent = dets.length;

  if (dets.length === 0) {
    list.innerHTML = '<div class="empty-hint">暂无检测结果</div>';
    return;
  }

  list.innerHTML = dets.map((d, i) => {
    const color = COLORS[i % COLORS.length];
    const conf = d.confidence || 0;
    const confClass = conf < 0.5 ? 'low' : '';
    const pos = d.world_fused
      ? `(${d.world_fused[0].toFixed(2)}, ${d.world_fused[1].toFixed(2)}, ${d.world_fused[2].toFixed(2)})`
      : '位置未知';
    const cap = d.caption || latestCaptions[String(d.track_id)] || '';
    const capHtml = cap ? `<div class="det-caption">${cap}</div>` : '';
    return `<div class="det-item" onclick="quickQuery('${d.class_name}')">
      <div class="det-header">
        <div class="class-dot" style="background:${color}"></div>
        <span class="class-name">${d.class_name}</span>
        <span class="conf-badge ${confClass}">${(conf*100).toFixed(0)}%</span>
      </div>
      <div class="det-pos">世界坐标: ${pos}</div>
      <div class="track-id">Track #${d.track_id}</div>
      ${capHtml}
    </div>`;
  }).join('');
}

setInterval(fetchDetections, 500);
setInterval(fetchCaptions, 3000);

// ── 查询提交 ────────────────────────────────────────────────────
function addLog(text, type='info') {
  const log = document.getElementById('query-log');
  const now = new Date().toLocaleTimeString('zh-CN', {hour12:false});
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  entry.innerHTML = `<span class="log-time">[${now}]</span><span class="log-text">${text}</span>`;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
  // 保留最多 100 条
  while (log.children.length > 100) log.removeChild(log.firstChild);
}

async function submitQuery() {
  const input = document.getElementById('query-input');
  const btn = document.getElementById('query-btn');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  btn.disabled = true;
  addLog(text, 'user');

  try {
    const resp = await fetch('/api/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: text})
    });
    const data = await resp.json();
    if (data.error) {
      addLog('错误: ' + data.error, 'error');
    } else {
      for (const line of (data.messages || [])) {
        addLog(line, 'system');
      }
    }
  } catch(e) {
    addLog('请求失败: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    input.focus();
  }
}

function quickQuery(text) {
  document.getElementById('query-input').value = text;
  submitQuery();
}

document.getElementById('query-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') submitQuery();
});

// ── 导航成功状态轮询 ────────────────────────────────────────────
let bannerTimer = null;
let navPolling = false;

function showNavSuccessBanner(target) {
  const banner = document.getElementById('nav-success-banner');
  const sub = document.getElementById('banner-sub-text');
  sub.textContent = target ? `已到达目标：${target}` : '已到达目标位置';

  // 先强制隐藏再显示，触发 slideDown 动画重放
  banner.classList.remove('show', 'hide');
  // 强制重排，让浏览器重新计算动画起始状态
  void banner.offsetWidth;
  banner.classList.add('show');

  // 视频边框绿色高亮
  const liveImg = document.getElementById('live-stream');
  liveImg.classList.add('nav-success');

  // 在查询日志中也追加一条
  addLog(`✅ 导航成功！已到达 "${target || '目标'}"`, 'system');

  // 5 秒后自动消失
  if (bannerTimer) clearTimeout(bannerTimer);
  bannerTimer = setTimeout(dismissBanner, 5000);
}

function dismissBanner() {
  const banner = document.getElementById('nav-success-banner');
  banner.classList.add('hide');
  setTimeout(() => {
    banner.classList.remove('show', 'hide');
    document.getElementById('live-stream').classList.remove('nav-success');
  }, 380);
  if (bannerTimer) { clearTimeout(bannerTimer); bannerTimer = null; }
}

async function pollNavStatus() {
  if (navPolling) return;
  navPolling = true;
  try {
    const resp = await fetch('/api/nav_status');
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.succeeded) {
      showNavSuccessBanner(data.target || '');
      // 确认收到后通知后端重置，避免重复弹出
      fetch('/api/nav_status/ack', {method: 'POST'}).catch(() => {});
    }
  } catch(e) {}
  finally { navPolling = false; }
}
setInterval(pollNavStatus, 800);
</script>
</body>
</html>"""


# -------------------------------------------
# Web UI：帧标注工具
# -------------------------------------------
def _annotate_frame(rgb_image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    在 RGB 图像上绘制检测框和标签，返回标注后的 BGR 图像（供 cv2 编码）。
    """
    img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    for i, det in enumerate(detections):
        color = _BBOX_COLORS[i % len(_BBOX_COLORS)]
        bbox = det.get("bbox_xyxy")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det.get('class_name', '?')} {det.get('confidence', 0):.2f}"
            label_y = max(y1 - 8, 16)
            cv2.putText(img, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return img


def _encode_frame_to_jpg(bgr_image: np.ndarray, quality: int = 80) -> bytes:
    """将 BGR 图像编码为 JPEG bytes。"""
    _, buf = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# -------------------------------------------
# Web UI：Flask 应用
# -------------------------------------------
webui_app = Flask(__name__)
webui_app.logger.disabled = True
import logging
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


@webui_app.route("/")
def webui_index():
    return _WEBUI_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


@webui_app.route("/stream")
def webui_stream():
    """MJPEG 视频流端点（保留兼容）。"""
    def generate():
        while True:
            with _webui_lock:
                frame_jpg = _webui_latest_frame_jpg
            if frame_jpg is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for camera...", (120, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
                frame_jpg = _encode_frame_to_jpg(placeholder)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_jpg + b"\r\n"
            )
            time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@webui_app.route("/snapshot")
def webui_snapshot():
    """返回最新一帧 JPEG 快照（轮询模式用）。"""
    with _webui_lock:
        frame_jpg = _webui_latest_frame_jpg
    if frame_jpg is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for camera...", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
        frame_jpg = _encode_frame_to_jpg(placeholder)
    return Response(frame_jpg, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store"})


@webui_app.route("/api/detections")
def webui_detections():
    """返回最新检测结果列表。"""
    with _webui_lock:
        dets = list(_webui_latest_detections)
    return jsonify({"detections": dets, "count": len(dets)})


@webui_app.route("/api/nav_status")
def webui_nav_status():
    """返回导航成功状态；前端需调用 /api/nav_status/ack 确认后才重置。"""
    global _nav_succeeded, _nav_succeeded_target
    with _nav_status_lock:
        succeeded = _nav_succeeded
        target = _nav_succeeded_target
    return jsonify({"succeeded": succeeded, "target": target})


@webui_app.route("/api/nav_status/ack", methods=["POST"])
def webui_nav_status_ack():
    """前端收到导航成功通知后调用此接口，重置状态。"""
    global _nav_succeeded, _nav_succeeded_target
    with _nav_status_lock:
        _nav_succeeded = False
        _nav_succeeded_target = ""
    return jsonify({"ok": True})


@webui_app.route("/api/captions")
def webui_captions():
    """代理 server 的 /captions 接口，返回所有 root_id → caption 映射。"""
    try:
        resp = requests.get(f"{_webui_server_base_url}/captions", timeout=3)
        return resp.json()
    except Exception as e:
        return jsonify({"captions": {}, "error": str(e)})


@webui_app.route("/api/query", methods=["POST"])
def webui_query():
    """
    接收 Web UI 的查询请求，执行与 interactive_thread 相同的逻辑，
    返回操作结果消息列表。
    """
    global _webui_query_log

    data = flask_request.get_json(force=True, silent=True) or {}
    user_input = str(data.get("query", "")).strip()
    if not user_input:
        return jsonify({"error": "query 不能为空"}), 400

    messages: list[str] = []

    # ── list 命令 ─────────────────────────────────────────────────
    if user_input.lower() == "list":
        try:
            resp_data = call_list_classes(_webui_server_base_url)
            classes = resp_data.get("classes", {})
            total = resp_data.get("total_tracks", 0)
            frames = resp_data.get("frame_count", 0)
            messages.append(f"已处理 {frames} 帧，共追踪 {total} 个 track")
            if classes:
                for cn, cnt in sorted(classes.items()):
                    messages.append(f"  {cn}  ×{cnt}")
            else:
                messages.append("（暂未追踪到任何物体）")
        except Exception as e:
            return jsonify({"error": f"list 查询失败: {e}"}), 500
        return jsonify({"messages": messages})

    # ── 自然语言解析 ──────────────────────────────────────────────
    query_class = user_input
    if _is_natural_language(user_input) and _webui_deepseek_api_key:
        try:
            classes_data = call_list_classes(_webui_server_base_url)
            candidate_classes = list(classes_data.get("classes", {}).keys())
        except Exception:
            candidate_classes = []

        if candidate_classes:
            messages.append(f"正在解析意图（候选类别：{', '.join(candidate_classes[:8])}...）")
            parsed = parse_intent_with_deepseek(
                user_input, candidate_classes,
                _webui_deepseek_api_key,
                _webui_deepseek_base_url,
                _webui_deepseek_model,
            )
            if parsed:
                query_class = parsed
                messages.append(f"意图解析：'{user_input}' → '{query_class}'")
            else:
                messages.append("意图解析失败，将直接使用原始输入")
        else:
            messages.append("暂无已追踪物体，无法进行意图解析")

    # ── 获取机器人位置 ────────────────────────────────────────────
    robot_x, robot_y = 0.0, 0.0
    if manager is not None:
        odom_rw_lock.acquire_read()
        odom_snapshot = copy.deepcopy(manager.odom)
        odom_rw_lock.release_read()
        if odom_snapshot is not None:
            robot_x, robot_y = odom_snapshot[0], odom_snapshot[1]

    # ── 查询 server ───────────────────────────────────────────────
    try:
        resp_data = call_query(_webui_server_base_url, query_class, robot_x, robot_y)
    except Exception as e:
        return jsonify({"error": f"查询失败: {e}"}), 500

    matches = resp_data.get("matches", [])
    if not matches:
        messages.append(f"未找到类别 '{query_class}'，请输入 list 查看已追踪物体")
        return jsonify({"messages": messages})

    messages.append(f"找到 {len(matches)} 个 '{query_class}' track：")
    for i, m in enumerate(matches):
        pos = m["position"]
        marker = " ← 已选" if i == 0 else ""
        messages.append(
            f"  Track#{m['track_id']} conf={m['mean_conf']:.2f}"
            f" dist={m['distance_to_robot']:.2f}m"
            f" pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}){marker}"
        )

    # ── 计算并发布导航目标 ────────────────────────────────────────
    chosen = matches[0]
    pos = chosen["position"]
    obj_x, obj_y, obj_z = pos[0], pos[1], pos[2]
    dx = obj_x - robot_x
    dy = obj_y - robot_y
    dist = math.hypot(dx, dy)

    if dist < 1e-6:
        messages.append("机器人与目标几乎重合，跳过发布")
        return jsonify({"messages": messages})

    goal_x = obj_x
    goal_y = obj_y
    goal_z = obj_z + _webui_z_offset
    yaw = math.atan2(dy, dx)

    if manager is not None:
        global _webui_last_nav_target
        _webui_last_nav_target = query_class
        manager.publish_goal(goal_x, goal_y, goal_z, yaw, frame_id=_webui_map_frame)
        messages.append(
            f"✓ 已发布导航目标 '{query_class}' track#{chosen['track_id']}"
        )
        messages.append(
            f"  goal=({goal_x:.3f},{goal_y:.3f},{goal_z:.3f})"
            f" yaw={math.degrees(yaw):.1f}°"
        )
    else:
        messages.append("ROS 节点未就绪，无法发布目标")

    return jsonify({"messages": messages})


def webui_thread(port: int) -> None:
    """在独立线程中运行 Web UI Flask 服务。"""
    print(f"[WebUI] 启动 Web UI，访问 http://0.0.0.0:{port}")
    webui_app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)


# -------------------------------------------
# Detection thread
# -------------------------------------------
def detection_thread(
    server_url: str,
    desired_interval: float,
    T_base_cam: np.ndarray | None = None,
    frame_stride: int = 1,
    camera_frame: str | None = None,
    lidar_max_points: int = 2000,
    marker_frame: str = "map",
):
    """
    frame_stride : 每隔多少帧处理一次。
      1  = 每帧都处理（默认）
      10 = 每 10 帧处理 1 帧（跳过中间 9 帧）
    camera_frame : 保留参数（兼容旧命令行），当前不用于 TF 变换。
                   为 None 时不发送激光雷达数据。
    lidar_max_points : 发送给 server 的最大点数，超出则随机下采样。
    """
    global _webui_latest_frame_jpg, _webui_latest_detections
    policy_init = True
    http_idx = 0
    arrived_count = 0   # 统计 ROS 回调触发的帧总数

    while True:
        time.sleep(0.05)
        if manager is None or not manager.new_image_arrived:
            time.sleep(0.01)
            continue

        t_start = time.time()
        manager.new_image_arrived = False
        arrived_count += 1

        # 帧采样：只处理第 1、stride+1、2*stride+1 … 帧
        if (arrived_count - 1) % frame_stride != 0:
            continue

        rgb_depth_rw_lock.acquire_read()
        rgb_bytes = copy.deepcopy(manager.rgb_bytes)
        depth_bytes = copy.deepcopy(manager.depth_bytes)
        rgb_image_snapshot = manager.rgb_image.copy() if manager.rgb_image is not None else None
        rgb_time = manager.rgb_time
        rgb_depth_rw_lock.release_read()

        odom_rw_lock.acquire_read()
        odom_infer = None
        min_diff = 1e10
        for odom_entry in manager.odom_queue:
            diff = abs(odom_entry[0] - rgb_time)
            if diff < min_diff:
                min_diff = diff
                odom_infer = copy.deepcopy(odom_entry[1])
        odom_rw_lock.release_read()

        if rgb_bytes is None or depth_bytes is None:
            print("[Client] skip: image data is None")
            continue

        if odom_infer is not None:
            pose = odom_to_pose(
                odom_infer[0], odom_infer[1], odom_infer[2],
                T_base_cam=T_base_cam,
            )
        else:
            pose = np.eye(4, dtype=np.float64)
            print("[Client] warn: no odom available, handheld-camera test mode, using identity pose")

        # ── 激光雷达数据（可选）─────────────────────────────
        lidar_pts_cam: np.ndarray | None = None
        if camera_frame and manager is not None:
            lidar_rw_lock.acquire_read()
            lidar_pts_raw = copy.deepcopy(manager.lidar_pts)
            lidar_frame_id = manager.lidar_frame_id
            lidar_rw_lock.release_read()

            if lidar_pts_raw is not None and lidar_frame_id:
                _t_prepare0 = time.time()
                pts_send = lidar_pts_raw

                # 随机下采样限制点数
                if pts_send.shape[0] > lidar_max_points:
                    idx = np.random.choice(pts_send.shape[0], lidar_max_points, replace=False)
                    pts_send = pts_send[idx]
                lidar_pts_cam = pts_send.astype(np.float32)

                _t_prepare1 = time.time()
                _payload_kb = lidar_pts_cam.nbytes / 1024.0
                # print(
                #     f"[DEBUG][LiDAR] raw={lidar_pts_raw.shape[0]}pts"
                #     f"  no_tf_transform"
                #     f"  send={lidar_pts_cam.shape[0]}pts({_payload_kb:.1f}KB)"
                #     f"  prepare={(_t_prepare1 - _t_prepare0)*1000:.2f}ms"
                # )

        try:
            _t_http0 = time.time()
            result = call_detect(
                image_bytes=rgb_bytes,
                depth_bytes=depth_bytes,
                pose=pose,
                reset=policy_init,
                url=server_url,
                lidar_points_cam=lidar_pts_cam,
            )
            _t_http1 = time.time()
            if lidar_pts_cam is not None:
                _payload_kb = lidar_pts_cam.nbytes / 1024.0
                print(
                    f"[DEBUG][HTTP] upload+inference={(_t_http1 - _t_http0)*1000:.2f}ms"
                    f"  lidar_payload={_payload_kb:.1f}KB"
                    f"  server_infer={result.get('inference_time', 0)*1000:.2f}ms"
                )

            policy_init = False
            http_idx += 1

            n_det = result.get("num_detections", 0)
            inf_time = result.get("inference_time", 0)
            # print(
            #     f"[Client] frame {http_idx} (arrived={arrived_count}): "
            #     f"{n_det} detections, server cost {inf_time:.3f}s"
            # )

            for det in result.get("detections", []):
                wf = det.get("world_fused", [0, 0, 0])
                # print(
                #     f"  Track {det['track_id']}: {det['class_name']} "
                #     f"conf={det['confidence']:.2f} "
                #     f"world=({wf[0]:.2f}, {wf[1]:.2f}, {wf[2]:.2f})"
                # )

            if manager is not None:
                msg = String()
                msg.data = json.dumps(result)
                manager.result_pub.publish(msg)

                objects_world = result.get("objects_world", None)
                if objects_world is None:
                    objects_world = []
                    for det in result.get("detections", []):
                        wf = det.get("world_fused")
                        if wf is None or len(wf) != 3:
                            continue
                        objects_world.append({
                            "track_id": det.get("track_id", -1),
                            "class_name": det.get("class_name", "unknown"),
                            "world": wf,
                            "confidence": det.get("confidence", 0.0),
                        })
                manager.publish_object_markers(objects_world, frame_id=marker_frame)

            # ── 更新 Web UI：用发送到 server 的同一帧标注检测框 ──
            detections = result.get("detections", [])
            with _webui_lock:
                _webui_latest_detections = detections
            if rgb_image_snapshot is not None:
                try:
                    annotated = _annotate_frame(rgb_image_snapshot, detections)
                    frame_jpg = _encode_frame_to_jpg(annotated)
                    with _webui_lock:
                        _webui_latest_frame_jpg = frame_jpg
                except Exception as e:
                    print(f"[WebUI] 帧标注失败: {e}")

        except Exception as e:
            print(f"[Client] HTTP error: {e}")

        elapsed = time.time() - t_start
        time.sleep(max(0, desired_interval - elapsed))


# -------------------------------------------
# Interactive semantic navigation thread
# -------------------------------------------
def _is_natural_language(text: str) -> bool:
    """
    判断输入是否为自然语言描述（而非直接的类别名称）。
    若包含中文字符、或包含空格且长度 > 6，则视为自然语言。
    """
    if any('\u4e00' <= ch <= '\u9fff' for ch in text):
        return True
    if ' ' in text and len(text) > 6:
        return True
    return False


def interactive_thread(
    base_url: str,
    goal_standoff: float,
    z_offset: float,
    map_frame: str,
    base_link_frame: str,
    deepseek_api_key: str | None = None,
    deepseek_base_url: str = "https://api.deepseek.com",
    deepseek_model: str = "deepseek-chat",
) -> None:
    """
    从终端读取用户输入，查询 server 中对应类别的目标，
    选取距机器人最近的 track，计算 standoff 目标点后发布 PoseStamped。

    命令：
      <类别名>          直接查询并发布导航目标（如 box、chair）
      <自然语言描述>    调用 DeepSeek API 解析意图，匹配最近类别后发布目标
                        （如 "请寻找一把椅子"、"帮我找盆栽"）
      list              列出当前 server 中已追踪到的所有类别
      q / quit          退出本线程（检测继续运行）

    自然语言模式触发条件：
      - 输入包含中文字符，或
      - 输入包含空格且长度 > 6
    需要通过 --deepseek-api-key 提供 DeepSeek API Key。
    """
    has_deepseek = bool(deepseek_api_key)

    # 检测是否有可用的 stdin（tmux/非交互式环境下 stdin 可能不是 tty）
    import sys as _sys
    if not _sys.stdin.isatty():
        print("[Interactive] 检测到非交互式终端（tmux/重定向），终端交互模式已禁用。请使用 Web UI 进行查询。")
        return

    print("\n" + "=" * 60)
    print("  语义导航交互模式已启动")
    print("  输入目标类别后按 Enter 发布导航目标")
    if has_deepseek:
        print("  支持自然语言描述（如：请寻找一把椅子）")
    else:
        print("  [提示] 未配置 --deepseek-api-key，自然语言模式不可用")
    print("  输入 'list' 查看已追踪物体，'q' 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n目标 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("[Interactive] 退出交互线程")
            break

        if not user_input:
            continue

        if user_input.lower() in ("q", "quit", "exit"):
            print("[Interactive] 退出交互线程")
            break

        # ── list：列出已追踪类别 ────────────────────────────────
        if user_input.lower() == "list":
            try:
                data = call_list_classes(base_url)
                classes = data.get("classes", {})
                total = data.get("total_tracks", 0)
                frames = data.get("frame_count", 0)
                print(f"[Interactive] 已处理 {frames} 帧，共追踪 {total} 个 track")
                if classes:
                    for cn, cnt in sorted(classes.items()):
                        print(f"  {cn:20s}  {cnt} track(s)")
                else:
                    print("  （暂未追踪到任何物体）")
            except Exception as e:
                print(f"[Interactive] list 查询失败: {e}")
            continue
        query_class = user_input

        # ── 获取机器人当前位置：仅使用 odom（不依赖 TF） ─────────
        robot_x, robot_y = 0.0, 0.0
        pos_source = "default(0,0)"
        if manager is not None:
            odom_rw_lock.acquire_read()
            odom_snapshot = copy.deepcopy(manager.odom)
            odom_rw_lock.release_read()
            if odom_snapshot is not None:
                robot_x, robot_y = odom_snapshot[0], odom_snapshot[1]
                pos_source = "odom"

        print(f"[Interactive] 机器人位置: ({robot_x:.3f}, {robot_y:.3f})  来源={pos_source}")

        # ── 查询 server ──────────────────────────────────────────
        try:
            data = call_query(base_url, query_class, robot_x, robot_y)
        except Exception as e:
            print(f"[Interactive] 查询失败: {e}")
            continue

        matches = data.get("matches", [])
        if not matches:
            print(f"[Interactive] 未找到类别 '{query_class}'，输入 'list' 查看已追踪物体")
            continue

        print(f"[Interactive] 查询 '{query_class}'，找到 {len(matches)} 个 track：")
        for i, m in enumerate(matches):
            pos = m["position"]
            marker = "  ← 最近，已选" if i == 0 else ""
            print(
                f"  Track {m['class_name']} {m['track_id']:3d}: conf={m['mean_conf']:.2f}"
                f"  dist={m['distance_to_robot']:.2f}m"
                f"  pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                f"{marker}"
            )

        # ── 计算 standoff 目标点 ─────────────────────────────────
        chosen = matches[0]
        pos = chosen["position"]
        obj_x, obj_y, obj_z = pos[0], pos[1], pos[2]

        dx = obj_x - robot_x
        dy = obj_y - robot_y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            print("[Interactive] 机器人与目标几乎重合，跳过")
            continue

        goal_x = obj_x
        goal_y = obj_y
        goal_z = obj_z + z_offset

        # 朝向使用“目标点 -> 物体点”，而不是“机器人出发点 -> 目标点”
        yaw = math.atan2(dy, dx)

        # ── 发布 PoseStamped ─────────────────────────────────────
        if manager is not None:
            manager.publish_goal(goal_x, goal_y, goal_z, yaw, frame_id=map_frame)
            display_label = (
                f"{user_input} → {query_class}"
                if query_class != user_input.lower()
                else query_class
            )
            print(
                f"[Interactive] ✓ 已发布目标 '{display_label}' track={chosen['track_id']}\n"
                f"              goal=({goal_x:.3f}, {goal_y:.3f}, {goal_z:.3f})"
                f"  yaw={math.degrees(yaw):.1f}°"
                f"  mode=object_pose  dist={dist:.2f}m"
            )
        else:
            print("[Interactive] ROS 节点未就绪，无法发布目标")


# -------------------------------------------
# ROS2 Node
# -------------------------------------------
class YoloDetectNode(Node):
    """
    ROS2 node: subscribe RGB + Depth (time-synced) + Odometry,
    encode to bytes and hand off to detection_thread.
    Also acts as goal publisher for interactive semantic navigation.
    """

    def __init__(
        self,
        rgb_topic: str,
        depth_topic: str,
        odom_topic: str,
        goal_topic: str = "/goal_pose",
        marker_topic: str = "/yolo_detect/markers",
        nav2_status_topic: str = "/navigate_to_pose/_action/status",
        lidar_topic: str | None = None,
        lidar_type: str = "laserscan",
    ):
        super().__init__("yolo_detect_client")

        rgb_sub = Subscriber(self, Image, rgb_topic)
        depth_sub = Subscriber(self, Image, depth_topic)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.synchronizer = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=1, slop=0.1
        )
        self.synchronizer.registerCallback(self.rgb_depth_callback)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, qos_profile
        )

        # ── 激光雷达订阅 ─────────────────────────────────────────
        self.lidar_pts: np.ndarray | None = None   # N×3 在 lidar 坐标系下
        self.lidar_frame_id: str | None = None
        self.lidar_time: float = 0.0
        self._T_cam_lidar_cache: np.ndarray | None = None
        self._T_cam_lidar_lock = threading.Lock()

        if lidar_topic:
            lidar_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
            )
            if lidar_type == "pointcloud2":
                if _HAS_PC2:
                    self.lidar_sub = self.create_subscription(
                        PointCloud2, lidar_topic, self.lidar_pc2_callback, lidar_qos
                    )
                    self.get_logger().info(f"激光雷达订阅 PointCloud2: {lidar_topic}")
                else:
                    self.get_logger().error(
                        "sensor_msgs_py 未安装，无法订阅 PointCloud2"
                    )
            else:  # laserscan (default)
                self.lidar_sub = self.create_subscription(
                    LaserScan, lidar_topic, self.lidar_scan_callback, lidar_qos
                )
                self.get_logger().info(f"激光雷达订阅 LaserScan: {lidar_topic}")
        else:
            self.get_logger().info("未配置 --lidar-topic，不订阅激光雷达")
        # ────────────────────────────────────────────────────────

        self.result_pub = self.create_publisher(String, "/yolo_detect/results", 10)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)

        self.nav2_action_name = "navigate_to_pose"
        self.nav2_action_client = ActionClient(self, NavigateToPose, self.nav2_action_name)
        self._goal_handle_lock = threading.Lock()
        self._current_goal_handle = None

        self.cv_bridge = CvBridge()
        self.rgb_bytes = None
        self.depth_bytes = None
        self.rgb_image = None
        self.depth_image = None
        self.new_image_arrived = False
        self.rgb_time = 0.0

        self.odom = None
        self.odom_queue = deque(maxlen=50)
        self.odom_timestamp = 0.0

        self.get_logger().info(
            f"YoloDetectNode 已启动，goal_topic={goal_topic}, marker_topic={marker_topic}, "
            f"nav2_action={self.nav2_action_name}"
        )

    # ------------------------------------------------------------------
    # 激光雷达回调
    # ------------------------------------------------------------------
    def lidar_scan_callback(self, msg: LaserScan) -> None:
        """LaserScan → N×3 numpy（在雷达坐标系下），带写锁存储。"""
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = np.linspace(
            msg.angle_min,
            msg.angle_min + msg.angle_increment * (len(ranges) - 1),
            len(ranges),
            dtype=np.float32,
        )
        valid = (
            np.isfinite(ranges)
            & (ranges >= msg.range_min)
            & (ranges <= msg.range_max)
        )
        r = ranges[valid]
        a = angles[valid]
        x = r * np.cos(a)
        y = r * np.sin(a)
        z = np.zeros_like(x)
        pts = np.stack([x, y, z], axis=1)  # N×3

        lidar_rw_lock.acquire_write()
        self.lidar_pts = pts
        self.lidar_frame_id = msg.header.frame_id
        self.lidar_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        lidar_rw_lock.release_write()

    def lidar_pc2_callback(self, msg: PointCloud2) -> None:
        """PointCloud2 → N×3 numpy（在雷达坐标系下），带写锁存储。"""
        if not _HAS_PC2:
            return
        try:
            pts_list = [
                [p[0], p[1], p[2]]
                for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            ]
            if not pts_list:
                return
            pts = np.array(pts_list, dtype=np.float32)
        except Exception as e:
            self.get_logger().warning(f"PointCloud2 解析失败: {e}")
            return

        lidar_rw_lock.acquire_write()
        self.lidar_pts = pts
        self.lidar_frame_id = msg.header.frame_id
        self.lidar_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        lidar_rw_lock.release_write()

    # ------------------------------------------------------------------
    # 导航 / 里程计 / 姿态
    # ------------------------------------------------------------------

    def publish_goal(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        frame_id: str = "map",
    ) -> None:
        """通过 Nav2 NavigateToPose Action 发送导航目标。"""
        half = 0.5 * yaw
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(half)
        pose.pose.orientation.w = math.cos(half)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        if not self.nav2_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error(
                f"Nav2 Action server '{self.nav2_action_name}' 不可用，目标未发送"
            )
            return

        send_future = self.nav2_action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._on_nav_goal_response)

    def _on_nav_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"导航目标发送异常: {exc}")
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warning("导航目标被拒绝")
            return

        with self._goal_handle_lock:
            self._current_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_nav_result)

    def _on_nav_result(self, future) -> None:
        global _nav_succeeded, _nav_succeeded_target
        try:
            result_wrap = future.result()
        except Exception as exc:
            self.get_logger().error(f"导航结果获取异常: {exc}")
            return

        status = result_wrap.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            print("导航成功")
            with _nav_status_lock:
                _nav_succeeded = True
                _nav_succeeded_target = _webui_last_nav_target
        else:
            self.get_logger().warning(f"导航结束，status={status}")

    def publish_object_markers(
        self,
        objects_world: list[dict],
        frame_id: str = "map",
    ) -> None:
        """发布目标可视化 MarkerArray：球体表示位置，文字显示类别与坐标。"""
        now = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        clear_marker.header.stamp = now
        clear_marker.ns = "yolo_objects"
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        for idx, obj in enumerate(objects_world):
            world = obj.get("world", [0.0, 0.0, 0.0])
            if world is None or len(world) != 3:
                continue

            track_id = int(obj.get("track_id", idx + 1))
            class_name = str(obj.get("class_name", "unknown"))
            conf = float(obj.get("confidence", 0.0))
            x, y, z = float(world[0]), float(world[1]), float(world[2])

            pos_marker = Marker()
            pos_marker.header.frame_id = frame_id
            pos_marker.header.stamp = now
            pos_marker.ns = "yolo_object_pos"
            pos_marker.id = track_id
            pos_marker.type = Marker.SPHERE
            pos_marker.action = Marker.ADD
            pos_marker.pose.position.x = x
            pos_marker.pose.position.y = y
            pos_marker.pose.position.z = z
            pos_marker.pose.orientation.w = 1.0
            pos_marker.scale.x = 0.20
            pos_marker.scale.y = 0.20
            pos_marker.scale.z = 0.20
            pos_marker.color.r = 0.1
            pos_marker.color.g = 0.9
            pos_marker.color.b = 0.1
            pos_marker.color.a = 0.9
            pos_marker.lifetime = Duration(seconds=3.0).to_msg()
            marker_array.markers.append(pos_marker)

            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = now
            text_marker.ns = "yolo_object_text"
            text_marker.id = track_id + 100000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = z + 0.28
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.16
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.95
            text_marker.text = (
                f"{class_name}#{track_id} c={conf:.2f}\n"
                f"({x:.2f}, {y:.2f}, {z:.2f})"
            )
            text_marker.lifetime = Duration(seconds=3.0).to_msg()
            marker_array.markers.append(text_marker)

        self.marker_pub.publish(marker_array)

    def rgb_depth_callback(self, rgb_msg, depth_msg):
        raw_rgb = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        pil_rgb = PIL_Image.fromarray(raw_rgb)
        rgb_buf = io.BytesIO()
        pil_rgb.save(rgb_buf, format="JPEG")
        rgb_buf.seek(0)

        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        raw_depth = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
        pil_depth = PIL_Image.fromarray(raw_depth.astype(np.uint16))
        depth_buf = io.BytesIO()
        pil_depth.save(depth_buf, format="PNG")
        depth_buf.seek(0)

        rgb_depth_rw_lock.acquire_write()
        self.rgb_bytes = rgb_buf
        self.depth_bytes = depth_buf
        self.rgb_image = raw_rgb
        self.depth_image = raw_depth
        self.rgb_time = (
            rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1.0e9
        )
        rgb_depth_rw_lock.release_write()

        self.new_image_arrived = True

    def odom_callback(self, msg):
        odom_rw_lock.acquire_write()
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
        self.odom_timestamp = time.time()
        odom_rw_lock.release_write()

# -------------------------------------------
# Main
# -------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO detect HTTP client (ROS2)"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:5802/detect",
        help="yolo_detect_server /detect endpoint",
    )
    parser.add_argument(
        "--rgb-topic",
        type=str,
        default="/camera/camera/color/image_raw",
    )
    parser.add_argument(
        "--depth-topic",
        type=str,
        default="/camera/camera/aligned_depth_to_color/image_raw",
    )
    parser.add_argument(
        "--odom-topic",
        type=str,
        default="/odom_bridge",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.3,
        help="Min interval (sec) between detect requests",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=15,
        help="帧采样间隔：每隔 N 帧处理一帧（默认 1 即每帧都处理，设为 10 则每 10 帧处理 1 帧）",
    )
    parser.add_argument(
        "--cam-json",
        type=str,
        default=None,
        help=(
            "相机参数 JSON 文件路径（如 cam_params.json），"
            "用于读取 cam-offset-x/y/z 将里程计从 base_link 转换到相机坐标系"
        ),
    )
    parser.add_argument(
        "--cam-offset-x", type=float, default=None,
        help="相机在 base_link 下 x 方向偏移(米)，优先级高于 --cam-json"
    )
    parser.add_argument(
        "--cam-offset-y", type=float, default=None,
        help="相机在 base_link 下 y 方向偏移(米)"
    )
    parser.add_argument(
        "--cam-offset-z", type=float, default=None,
        help="相机在 base_link 下 z 方向偏移(米)"
    )
    # ── 语义导航参数 ─────────────────────────────────────────────
    parser.add_argument(
        "--goal-topic",
        type=str,
        default="/goal_pose",
        help="发布导航目标的 ROS2 topic（默认 /goal_pose，Nav2 兼容）",
    )
    parser.add_argument(
        "--marker-topic",
        type=str,
        default="/yolo_detect/markers",
        help="发布检测结果可视化 MarkerArray 的话题（RViz2 订阅）",
    )
    parser.add_argument(
        "--nav2-status-topic",
        type=str,
        default="/navigate_to_pose/_action/status",
        help="已弃用参数（保留兼容），当前使用 NavigateToPose Action 返回状态判定导航结果",
    )
    parser.add_argument(
        "--map-frame",
        type=str,
        default="map",
        help="地图坐标系 frame id（TF 查询用，默认 map）",
    )
    parser.add_argument(
        "--base-link-frame",
        type=str,
        default="base_link",
        help="机器人底盘 frame id（TF 查询用，默认 base_link）",
    )
    parser.add_argument(
        "--goal-standoff",
        type=float,
        default=0.5,
        help="导航目标距物体的安全停留距离（米，默认 0.6）",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="导航目标 z 方向额外偏移（米，默认 0.0）",
    )
    # ── 激光雷达参数 ───────────────────────────────────────────────
    parser.add_argument(
        "--lidar-topic",
        type=str,
        default=None,
        help="激光雷达话题，如 /scan 或 /livox/lidar；不填则不订阅",
    )
    parser.add_argument(
        "--lidar-type",
        type=str,
        default="pointcloud2",
        choices=["laserscan", "pointcloud2"],
        help="激光雷达消息类型：laserscan（2D）或 pointcloud2（3D）（默认 pointcloud2）",
    )
    parser.add_argument(
        "--camera-frame",
        type=str,
        default="camera_color_optical_frame",
        help=(
            "相机光学坐标系的 TF frame id，用于查询 T_cam_lidar。"
            "（默认 camera_color_optical_frame）"
        ),
    )
    parser.add_argument(
        "--lidar-max-points",
        type=int,
        default=32000,
        help="每帧发送给 server 的最大激光雷达点数（默认 32000）",
    )
    # ── DeepSeek 意图解析参数 ─────────────────────────────────
    parser.add_argument(
        "--deepseek-api-key",
        type=str,
        default=None,
        help="DeepSeek API Key，用于自然语言意图解析（不填则禁用自然语言模式）",
    )
    parser.add_argument(
        "--deepseek-base-url",
        type=str,
        default="https://api.deepseek.com",
        help="DeepSeek API 根地址（默认 https://api.deepseek.com）",
    )
    parser.add_argument(
        "--deepseek-model",
        type=str,
        default="deepseek-chat",
        help="DeepSeek 模型名（默认 deepseek-chat）",
    )
    # ── Web UI 参数 ───────────────────────────────────────────────
    parser.add_argument(
        "--webui-port",
        type=int,
        default=8080,
        help="Web UI 监听端口（默认 8080），浏览器访问 http://<host>:<port> 查看实时画面",
    )
    parser.add_argument(
        "--no-webui",
        action="store_true",
        help="禁用 Web UI（仅使用终端交互模式）",
    )
    args = parser.parse_args()

    # 构建 T_base_cam：优先使用命令行偏移，其次从 cam_params.json 读取
    if args.cam_offset_x is not None:
        ox, oy, oz = (
            args.cam_offset_x,
            args.cam_offset_y or 0.0,
            args.cam_offset_z or 0.0,
        )
    else:
        ox, oy, oz = load_cam_offset(args.cam_json)

    T_base_cam = build_T_base_cam(ox, oy, oz)
    print(
        f"[Client] T_base_cam offset: x={ox:.4f}, y={oy:.4f}, z={oz:.4f}\n"
        f"         T_base_cam =\n{T_base_cam}"
    )

    frame_stride = max(1, args.frame_stride)
    print(f"[Client] frame_stride={frame_stride}（每 {frame_stride} 帧处理 1 帧）")
    print(
        f"[Client] 语义导航: goal_topic={args.goal_topic}"
        f"  standoff={args.goal_standoff}m  z_offset={args.z_offset}m"
        f"  TF: {args.map_frame}→{args.base_link_frame}"
    )
    print("[Client] Nav2 成功判定: 使用 navigate_to_pose Action result.status")
    print(f"[Client] RViz 标记发布: marker_topic={args.marker_topic}  frame={args.map_frame}")
    if args.lidar_topic:
        print(
            f"[Client] 激光雷达: topic={args.lidar_topic}"
            f"  type={args.lidar_type}"
            f"  camera_frame={args.camera_frame}"
            f"  max_pts={args.lidar_max_points}"
        )
    else:
        print("[Client] 未配置激光雷达，仅使用 RGBD 深度")

    # 从 /detect URL 提取根地址，用于 /query、/list_classes 请求
    server_base_url = args.server_url.rstrip("/")
    if server_base_url.endswith("/detect"):
        server_base_url = server_base_url[: -len("/detect")]

    # ── 初始化 Web UI 全局配置 ────────────────────────────────────
    _webui_server_base_url = server_base_url
    _webui_goal_standoff = args.goal_standoff
    _webui_z_offset = args.z_offset
    _webui_map_frame = args.map_frame
    _webui_deepseek_api_key = args.deepseek_api_key
    _webui_deepseek_base_url = args.deepseek_base_url
    _webui_deepseek_model = args.deepseek_model

    det_thread = threading.Thread(
        target=detection_thread,
        args=(
            args.server_url,
            args.rate,
            T_base_cam,
            frame_stride,
            args.camera_frame if args.lidar_topic else None,
            args.lidar_max_points,
            args.map_frame,
        ),
        daemon=True,
    )
    iac_thread = threading.Thread(
        target=interactive_thread,
        args=(
            server_base_url,
            args.goal_standoff,
            args.z_offset,
            args.map_frame,
            args.base_link_frame,
            args.deepseek_api_key,
            args.deepseek_base_url,
            args.deepseek_model,
        ),
        daemon=True,
    )

    rclpy.init()
    try:
        manager = YoloDetectNode(
            rgb_topic=args.rgb_topic,
            depth_topic=args.depth_topic,
            odom_topic=args.odom_topic,
            goal_topic=args.goal_topic,
            marker_topic=args.marker_topic,
            nav2_status_topic=args.nav2_status_topic,
            lidar_topic=args.lidar_topic,
            lidar_type=args.lidar_type,
        )
        det_thread.start()
        iac_thread.start()

        # ── 启动 Web UI ───────────────────────────────────────────
        if not args.no_webui:
            ui_thread = threading.Thread(
                target=webui_thread,
                args=(args.webui_port,),
                daemon=True,
            )
            ui_thread.start()
            print(f"[WebUI] 浏览器访问: http://0.0.0.0:{args.webui_port}")

        print(f"[Client] started, server: {args.server_url}")
        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        manager.destroy_node()
        rclpy.shutdown()