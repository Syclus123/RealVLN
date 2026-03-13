# -*- coding: utf-8 -*-
"""
Web UI 共享状态与业务逻辑
"""

from __future__ import annotations

import copy
import math
import threading
from typing import Any

import cv2
import numpy as np
import requests


# -------------------------------------------
# Web UI 全局状态
# -------------------------------------------
webui_lock = threading.Lock()

latest_frame_jpg: bytes | None = None
latest_detections: list[dict] = []

server_base_url: str = ""
goal_standoff: float = 0.5
z_offset: float = 0.0
map_frame: str = "map"

deepseek_api_key: str | None = None
deepseek_base_url: str = "https://api.deepseek.com"
deepseek_model: str = "deepseek-chat"

nav_status_lock = threading.Lock()
nav_succeeded: bool = False
nav_succeeded_target: str = ""
nav_last_status_text: str = ""

BBOX_COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (255, 0, 128),
    (128, 255, 0),
    (0, 255, 128),
    (255, 0, 0),
    (0, 0, 255),
]


# -------------------------------------------
# 初始化配置
# -------------------------------------------
def init_webui_config(
    *,
    server_base_url_: str,
    goal_standoff_: float,
    z_offset_: float,
    map_frame_: str,
    deepseek_api_key_: str | None,
    deepseek_base_url_: str,
    deepseek_model_: str,
) -> None:
    global server_base_url, goal_standoff, z_offset, map_frame
    global deepseek_api_key, deepseek_base_url, deepseek_model

    server_base_url = server_base_url_
    goal_standoff = goal_standoff_
    z_offset = z_offset_
    map_frame = map_frame_

    deepseek_api_key = deepseek_api_key_
    deepseek_base_url = deepseek_base_url_
    deepseek_model = deepseek_model_


# -------------------------------------------
# 导航状态
# -------------------------------------------
def set_nav_success(target: str) -> None:
    global nav_succeeded, nav_succeeded_target
    with nav_status_lock:
        nav_succeeded = True
        nav_succeeded_target = target


def set_nav_status_text(text: str) -> None:
    global nav_last_status_text
    with nav_status_lock:
        nav_last_status_text = text


def get_nav_status() -> dict[str, Any]:
    with nav_status_lock:
        return {
            "succeeded": nav_succeeded,
            "target": nav_succeeded_target,
            "status_text": nav_last_status_text,
        }


def ack_nav_status() -> None:
    global nav_succeeded, nav_succeeded_target
    with nav_status_lock:
        nav_succeeded = False
        nav_succeeded_target = ""


# -------------------------------------------
# 视频帧 / 检测结果
# -------------------------------------------
def set_latest_detections(dets: list[dict]) -> None:
    global latest_detections
    with webui_lock:
        latest_detections = list(dets)


def get_latest_detections() -> list[dict]:
    with webui_lock:
        return list(latest_detections)


def set_latest_frame_jpg(frame_jpg: bytes) -> None:
    global latest_frame_jpg
    with webui_lock:
        latest_frame_jpg = frame_jpg


def get_latest_frame_jpg() -> bytes | None:
    with webui_lock:
        return latest_frame_jpg


# -------------------------------------------
# 工具函数
# -------------------------------------------
def annotate_frame(rgb_image: np.ndarray, detections: list[dict]) -> np.ndarray:
    img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    for i, det in enumerate(detections):
        color = BBOX_COLORS[i % len(BBOX_COLORS)]
        bbox = det.get("bbox_xyxy")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det.get('class_name', '?')} {det.get('confidence', 0):.2f}"
            label_y = max(y1 - 8, 16)
            cv2.putText(
                img,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return img


def encode_frame_to_jpg(bgr_image: np.ndarray, quality: int = 80) -> bytes:
    _, buf = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# -------------------------------------------
# 依赖主程序的回调注入
# -------------------------------------------
_manager_getter = None
_odom_lock_getter = None
_call_list_classes = None
_call_query = None
_parse_intent_with_deepseek = None
_is_natural_language = None
_go2_controller = None


def bind_runtime_dependencies(
    *,
    manager_getter,
    odom_lock_getter,
    call_list_classes,
    call_query,
    parse_intent_with_deepseek,
    is_natural_language,
    go2_controller=None,
) -> None:
    global _manager_getter, _odom_lock_getter
    global _call_list_classes, _call_query
    global _parse_intent_with_deepseek, _is_natural_language
    global _go2_controller

    _manager_getter = manager_getter
    _odom_lock_getter = odom_lock_getter
    _call_list_classes = call_list_classes
    _call_query = call_query
    _parse_intent_with_deepseek = parse_intent_with_deepseek
    _is_natural_language = is_natural_language
    _go2_controller = go2_controller


# -------------------------------------------
# query 业务逻辑
# -------------------------------------------
def resolve_query_class_for_text(user_input: str) -> tuple[str, list[str]]:
    messages: list[str] = []
    query_class = user_input

    if _is_natural_language and _is_natural_language(user_input) and deepseek_api_key:
        try:
            classes_data = _call_list_classes(server_base_url)
            candidate_classes = list(classes_data.get("classes", {}).keys())
        except Exception:
            candidate_classes = []

        if candidate_classes:
            messages.append(
                f"正在解析意图（候选类别：{', '.join(candidate_classes[:8])}...）"
            )
            parsed = _parse_intent_with_deepseek(
                user_input,
                candidate_classes,
                deepseek_api_key,
                deepseek_base_url,
                deepseek_model,
            )
            if parsed:
                query_class = parsed
                messages.append(f"意图解析：'{user_input}' → '{query_class}'")
            else:
                messages.append("意图解析失败，将直接使用原始输入")
        else:
            messages.append("暂无已追踪物体，无法进行意图解析")

    return query_class, messages


def handle_query_request(user_input: str) -> tuple[list[str], str | None]:
    """
    返回:
      messages, error
    """
    if not user_input:
        return [], "query 不能为空"

    messages: list[str] = []

    if user_input.lower() == "list":
        try:
            resp_data = _call_list_classes(server_base_url)
            classes = resp_data.get("classes", {})
            total = resp_data.get("total_tracks", 0)
            frames = resp_data.get("frame_count", 0)
            messages.append(f"已处理 {frames} 帧，共追踪 {total} 个 track")
            if classes:
                for cn, cnt in sorted(classes.items()):
                    messages.append(f"  {cn}  ×{cnt}")
            else:
                messages.append("（暂未追踪到任何物体）")
            return messages, None
        except Exception as e:
            return [], f"list 查询失败: {e}"

    query_class, parse_msgs = resolve_query_class_for_text(user_input)
    messages.extend(parse_msgs)

    robot_x, robot_y = 0.0, 0.0
    manager = _manager_getter() if _manager_getter else None
    odom_lock = _odom_lock_getter() if _odom_lock_getter else None

    if manager is not None and odom_lock is not None:
        odom_lock.acquire_read()
        odom_snapshot = copy.deepcopy(manager.odom)
        odom_lock.release_read()
        if odom_snapshot is not None:
            robot_x, robot_y = odom_snapshot[0], odom_snapshot[1]

    try:
        resp_data = _call_query(server_base_url, query_class, robot_x, robot_y)
    except Exception as e:
        return [], f"查询失败: {e}"

    matches = resp_data.get("matches", [])
    if not matches:
        messages.append(f"未找到类别 '{query_class}'，请输入 list 查看已追踪物体")
        return messages, None

    messages.append(f"找到 {len(matches)} 个 '{query_class}' track：")
    for i, m in enumerate(matches):
        pos = m["position"]
        marker = " ← 已选" if i == 0 else ""
        messages.append(
            f"  Track#{m['track_id']} conf={m['mean_conf']:.2f}"
            f" dist={m['distance_to_robot']:.2f}m"
            f" pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}){marker}"
        )

    chosen = matches[0]
    pos = chosen["position"]
    obj_x, obj_y, obj_z = pos[0], pos[1], pos[2]

    dx = obj_x - robot_x
    dy = obj_y - robot_y
    dist = math.hypot(dx, dy)

    if dist < 1e-6:
        messages.append("机器人与目标几乎重合，跳过发布")
        return messages, None

    effective_standoff = max(0.0, float(goal_standoff))
    if dist > effective_standoff:
        target_dist = dist - effective_standoff
    else:
        target_dist = max(dist * 0.5, 0.1)

    scale = target_dist / dist
    goal_x = robot_x + dx * scale
    goal_y = robot_y + dy * scale
    goal_z = obj_z + z_offset
    yaw = math.atan2(dy, dx)

    if manager is not None:
        target_name = query_class
        manager.publish_goal(
            goal_x,
            goal_y,
            goal_z,
            yaw,
            frame_id=map_frame,
            target_name=target_name,
        )
        messages.append(f"✓ 已发布导航目标 '{target_name}' track#{chosen['track_id']}")
        messages.append(
            f"  goal=({goal_x:.3f},{goal_y:.3f},{goal_z:.3f})"
            f" yaw={math.degrees(yaw):.1f}°"
        )
    else:
        messages.append("ROS 节点未就绪，无法发布目标")

    return messages, None


def fetch_captions() -> dict:
    try:
        resp = requests.get(f"{server_base_url}/captions", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"captions": {}, "error": str(e)}


# -------------------------------------------
# Go2 运动控制
# -------------------------------------------
def get_go2_controller():
    return _go2_controller


def handle_go2_action(action_name: str, angle: float | None = None) -> tuple[list[str], str | None]:
    """
    执行 Go2 运动控制动作。
    返回: (messages, error)
    """
    if _go2_controller is None:
        return [], "Go2 控制器未初始化（启动时未启用 --go2-interface）"

    if not _go2_controller.is_ready:
        return [], "Go2 控制器尚未就绪"

    success, msg = _go2_controller.execute(action_name, angle=angle)
    if success:
        return [f"✓ {msg}"], None
    else:
        return [], msg


def is_go2_action(text: str) -> bool:
    """判断用户输入是否为 Go2 动作指令。"""
    from go2_control import ACTIONS
    return text.lower().strip() in ACTIONS
