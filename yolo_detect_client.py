#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 目标检测 HTTP 客户端（ROS2 版）

说明：
  - Web UI 相关逻辑已拆分到：
      1. webui_state.py
      2. webui_server.py
  - 本文件仅保留：
      * ROS2 节点
      * 检测线程
      * 终端交互线程
      * 与 Web UI 状态模块的对接

新增：
  - 支持两种导航目标发送 / 到达判定模式：
      1. action:
         - 通过 Nav2 NavigateToPose Action 下发目标
         - 通过 Action result.status 判断是否到达
      2. goal_pose:
         - 通过 /goal_pose 发布 PoseStamped
         - 通过当前 odom 与目标点距离 < arrival_threshold 判断是否到达
"""

from __future__ import annotations

import argparse
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

import numpy as np
import rclpy
import requests
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from PIL import Image as PIL_Image
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

try:
    import sensor_msgs_py.point_cloud2 as pc2

    _HAS_PC2 = True
except ImportError:
    _HAS_PC2 = False
    print("[Warn] sensor_msgs_py 未安装，PointCloud2 订阅无效")

from thread_utils import ReadWriteLock
import webui_state as ws
from webui_server import start_webui


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

    response = requests.post(
        url, files=files, data={"json": json_payload}, timeout=timeout
    )

    if response.status_code == 413:
        raise RuntimeError(
            "HTTP 413 Request Entity Too Large：请求体过大（RGB+Depth 数据超出 server MAX_CONTENT_LENGTH）"
        )
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}：{response.text[:200]}")
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
    resp = requests.post(
        f"{base_url}/query",
        json={"class_name": class_name, "robot_x": robot_x, "robot_y": robot_y},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def call_list_classes(base_url: str, timeout: float = 5.0) -> dict:
    resp = requests.get(f"{base_url}/list_classes", timeout=timeout)
    resp.raise_for_status()
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

        answer_lower = answer.lower()
        for cls in candidate_classes:
            if cls.lower() == answer_lower:
                return cls
        for cls in candidate_classes:
            if cls.lower() in answer_lower or answer_lower in cls.lower():
                return cls
        return answer
    except Exception as e:
        print(f"[DeepSeek] API 调用失败: {e}")
        return None


# -------------------------------------------
# Camera offset helpers
# -------------------------------------------
def load_cam_offset(cam_json_path: str | None) -> tuple[float, float, float]:
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
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = offset_x
    T[1, 3] = offset_y
    T[2, 3] = offset_z
    return T


# -------------------------------------------
# Odom -> 4x4 camera pose in world frame
# -------------------------------------------
def odom_to_pose(
    x: float,
    y: float,
    yaw: float,
    T_base_cam: np.ndarray | None = None,
) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    T_w_base = np.array(
        [
            [c, -s, 0.0, x],
            [s, c, 0.0, y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    if T_base_cam is None:
        return T_w_base

    return T_w_base @ T_base_cam


# -------------------------------------------
# Global state
# -------------------------------------------
rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
lidar_rw_lock = ReadWriteLock()
manager = None


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
    policy_init = True
    arrived_count = 0

    while True:
        time.sleep(0.05)
        if manager is None or not manager.new_image_arrived:
            time.sleep(0.01)
            continue

        t_start = time.time()
        manager.new_image_arrived = False
        arrived_count += 1

        if (arrived_count - 1) % frame_stride != 0:
            continue

        rgb_depth_rw_lock.acquire_read()
        rgb_bytes = copy.deepcopy(manager.rgb_bytes)
        depth_bytes = copy.deepcopy(manager.depth_bytes)
        rgb_image_snapshot = (
            manager.rgb_image.copy() if manager.rgb_image is not None else None
        )
        rgb_time = manager.rgb_time
        rgb_depth_rw_lock.release_read()

        odom_rw_lock.acquire_read()
        odom_infer = None
        odom_infer_is_map = False
        odom_infer_frame = "odom"
        min_diff = 1e10
        for odom_entry in manager.odom_queue:
            diff = abs(odom_entry[0] - rgb_time)
            if diff < min_diff:
                min_diff = diff
                odom_infer = copy.deepcopy(odom_entry[1])
            odom_infer_is_map = bool(odom_entry[2]) if len(odom_entry) >= 3 else False
            odom_infer_frame = odom_entry[3] if len(odom_entry) >= 4 else "odom"
        odom_rw_lock.release_read()

        if rgb_bytes is None or depth_bytes is None:
            print("[Client] skip: image data is None")
            continue

        if odom_infer is not None:
            pose_base = odom_infer
            if not odom_infer_is_map and manager is not None:
                converted = manager._odom_pose_to_map(
                    odom_infer[0], odom_infer[1], odom_infer[2], odom_infer_frame
                )
                if converted is not None:
                    pose_base = [converted[0], converted[1], converted[2]]

            pose = odom_to_pose(
                pose_base[0],
                pose_base[1],
                pose_base[2],
                T_base_cam=T_base_cam,
            )
        else:
            pose = np.eye(4, dtype=np.float64)
            print("[Client] warn: no odom available, using identity pose")

        lidar_pts_cam: np.ndarray | None = None
        if camera_frame and manager is not None:
            lidar_rw_lock.acquire_read()
            lidar_pts_raw = copy.deepcopy(manager.lidar_pts)
            lidar_frame_id = manager.lidar_frame_id
            lidar_rw_lock.release_read()

            if lidar_pts_raw is not None and lidar_frame_id:
                pts_send = lidar_pts_raw
                if pts_send.shape[0] > lidar_max_points:
                    idx = np.random.choice(
                        pts_send.shape[0], lidar_max_points, replace=False
                    )
                    pts_send = pts_send[idx]
                lidar_pts_cam = pts_send.astype(np.float32)

        try:
            result = call_detect(
                image_bytes=rgb_bytes,
                depth_bytes=depth_bytes,
                pose=pose,
                reset=policy_init,
                url=server_url,
                lidar_points_cam=lidar_pts_cam,
            )
            policy_init = False

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
                        objects_world.append(
                            {
                                "track_id": det.get("track_id", -1),
                                "class_name": det.get("class_name", "unknown"),
                                "world": wf,
                                "confidence": det.get("confidence", 0.0),
                            }
                        )
                manager.publish_object_markers(objects_world, frame_id=marker_frame)

            detections = result.get("detections", [])
            ws.set_latest_detections(detections)

            if rgb_image_snapshot is not None:
                try:
                    annotated = ws.annotate_frame(rgb_image_snapshot, detections)
                    frame_jpg = ws.encode_frame_to_jpg(annotated)
                    ws.set_latest_frame_jpg(frame_jpg)
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
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return True
    if " " in text and len(text) > 6:
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
    has_deepseek = bool(deepseek_api_key)

    import sys as _sys

    if not _sys.stdin.isatty():
        print("[Interactive] 非交互式终端，终端交互模式已禁用，请使用 Web UI。")
        return

    print("\n" + "=" * 60)
    print("  语义导航交互模式已启动")
    print("  输入目标类别后按 Enter 发送导航目标")
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

        if _is_natural_language(user_input):
            if not has_deepseek:
                print("[Interactive] 未配置 DeepSeek API Key，无法解析自然语言")
                continue
            try:
                classes_data = call_list_classes(base_url)
                candidate_classes = list(classes_data.get("classes", {}).keys())
            except Exception as e:
                print(f"[Interactive] 获取类别列表失败: {e}")
                candidate_classes = []

            if not candidate_classes:
                print("[Interactive] 暂无已追踪类别，无法进行意图解析")
                continue

            print(
                f"[Interactive] 正在解析意图，候选类别: {', '.join(candidate_classes[:8])}..."
            )
            parsed = parse_intent_with_deepseek(
                user_input,
                candidate_classes,
                deepseek_api_key,
                deepseek_base_url,
                deepseek_model,
            )
            if not parsed:
                print("[Interactive] 意图解析失败")
                continue
            query_class = parsed
            print(f"[Interactive] 意图解析: '{user_input}' -> '{query_class}'")

        robot_x, robot_y = 0.0, 0.0
        pos_source = "default(0,0)"
        if manager is not None:
            odom_rw_lock.acquire_read()
            odom_snapshot = copy.deepcopy(manager.odom)
            odom_rw_lock.release_read()
            if odom_snapshot is not None:
                robot_x, robot_y = odom_snapshot[0], odom_snapshot[1]
                pos_source = "odom"

        print(
            f"[Interactive] 机器人位置: ({robot_x:.3f}, {robot_y:.3f}) 来源={pos_source}"
        )

        try:
            data = call_query(base_url, query_class, robot_x, robot_y)
        except Exception as e:
            print(f"[Interactive] 查询失败: {e}")
            continue

        matches = data.get("matches", [])
        if not matches:
            print(
                f"[Interactive] 未找到类别 '{query_class}'，输入 'list' 查看已追踪物体"
            )
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

        chosen = matches[0]
        pos = chosen["position"]
        obj_x, obj_y, obj_z = pos[0], pos[1], pos[2]

        dx = obj_x - robot_x
        dy = obj_y - robot_y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            print("[Interactive] 机器人与目标几乎重合，跳过")
            continue

        effective_standoff = max(0.0, float(goal_standoff))
        if dist > effective_standoff:
            target_dist = dist - effective_standoff
        else:
            target_dist = max(dist * 0.5, 0.1)

        scale = target_dist / dist
        goal_x = robot_x + dx * scale
        goal_y = robot_y + dy * scale
        goal_z = obj_z
        yaw = math.atan2(dy, dx)

        if manager is not None:
            display_label = (
                f"{user_input} → {query_class}"
                if query_class != user_input
                else query_class
            )
            manager.publish_goal(
                goal_x,
                goal_y,
                goal_z,
                yaw,
                frame_id=map_frame,
                target_name=display_label,
            )
            print(
                f"[Interactive] ✓ 已发送目标 '{display_label}' track={chosen['track_id']}\n"
                f"              goal=({goal_x:.3f}, {goal_y:.3f}, {goal_z:.3f})"
                f" yaw={math.degrees(yaw):.1f}°"
                f" dist={dist:.2f}m"
            )
        else:
            print("[Interactive] ROS 节点未就绪，无法发布目标")


# -------------------------------------------
# ROS2 Node
# -------------------------------------------
class YoloDetectNode(Node):
    def __init__(
        self,
        rgb_topic: str,
        depth_topic: str,
        odom_topic: str,
        map_frame: str = "map",
        odom_frame: str | None = None,
        goal_topic: str = "/goal_pose",
        marker_topic: str = "/yolo_detect/markers",
        lidar_topic: str | None = None,
        lidar_type: str = "laserscan",
        publish_goal_pose_topic: bool = True,
        goal_send_mode: str = "action",
        arrival_threshold: float = 0.3,
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

        self.lidar_pts: np.ndarray | None = None
        self.lidar_frame_id: str | None = None
        self.lidar_time: float = 0.0

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
            else:
                self.lidar_sub = self.create_subscription(
                    LaserScan, lidar_topic, self.lidar_scan_callback, lidar_qos
                )
                self.get_logger().info(f"激光雷达订阅 LaserScan: {lidar_topic}")
        else:
            self.get_logger().info("未配置 --lidar-topic，不订阅激光雷达")

        self.result_pub = self.create_publisher(String, "/yolo_detect/results", 10)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)

        self.map_frame = map_frame
        self.odom_frame = odom_frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_tf_warn_time = 0.0

        self.goal_topic = goal_topic
        self.publish_goal_pose_topic = publish_goal_pose_topic
        self.goal_send_mode = goal_send_mode
        self.arrival_threshold = float(arrival_threshold)

        self.goal_pose_pub = None
        if self.publish_goal_pose_topic or self.goal_send_mode == "goal_pose":
            self.goal_pose_pub = self.create_publisher(PoseStamped, goal_topic, 10)

        self.nav2_action_name = "navigate_to_pose"
        self.nav2_action_client = ActionClient(
            self, NavigateToPose, self.nav2_action_name
        )

        self._goal_handle_lock = threading.Lock()
        self._current_goal_handle = None
        self._current_goal_name = ""
        self._server_ready_logged = False

        # goal_pose 模式下的目标到达检测状态
        self._goal_arrival_lock = threading.Lock()
        self._active_goal_pose = (
            None  # dict | None: {x, y, z, yaw, frame_id, target_name}
        )
        self._goal_arrived_reported = False

        self._arrival_timer = self.create_timer(0.2, self._check_goal_arrival)

        self.cv_bridge = CvBridge()
        self.rgb_bytes = None
        self.depth_bytes = None
        self.rgb_image = None
        self.depth_image = None
        self.new_image_arrived = False
        self.rgb_time = 0.0

        self.odom = None
        self.odom_in_map = False
        self.odom_queue = deque(maxlen=50)
        self.odom_timestamp = 0.0

        self.get_logger().info(
            f"YoloDetectNode 已启动，goal_topic={goal_topic}, marker_topic={marker_topic}, "
            f"map_frame={map_frame}, odom_frame={odom_frame or 'auto'}, "
            f"goal_send_mode={goal_send_mode}, arrival_threshold={self.arrival_threshold:.3f}m, "
            f"nav2_action={self.nav2_action_name}, publish_goal_pose_topic={publish_goal_pose_topic}"
        )

    # ------------------------------------------------------------------
    # 激光雷达回调
    # ------------------------------------------------------------------
    def lidar_scan_callback(self, msg: LaserScan) -> None:
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = np.linspace(
            msg.angle_min,
            msg.angle_min + msg.angle_increment * (len(ranges) - 1),
            len(ranges),
            dtype=np.float32,
        )
        valid = (
            np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        )
        r = ranges[valid]
        a = angles[valid]
        x = r * np.cos(a)
        y = r * np.sin(a)
        z = np.zeros_like(x)
        pts = np.stack([x, y, z], axis=1)

        lidar_rw_lock.acquire_write()
        self.lidar_pts = pts
        self.lidar_frame_id = msg.header.frame_id
        self.lidar_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        lidar_rw_lock.release_write()

    def lidar_pc2_callback(self, msg: PointCloud2) -> None:
        if not _HAS_PC2:
            return
        try:
            pts_list = [
                [p[0], p[1], p[2]]
                for p in pc2.read_points(
                    msg, field_names=("x", "y", "z"), skip_nans=True
                )
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
    # 导航 / 里程计
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def _odom_pose_to_map(
        self,
        x_odom: float,
        y_odom: float,
        yaw_odom: float,
        odom_frame_id: str,
    ) -> tuple[float, float, float] | None:
        source_frame = (odom_frame_id or self.odom_frame or "odom").strip()
        if source_frame == self.map_frame:
            return x_odom, y_odom, yaw_odom

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.map_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as exc:
            now = time.time()
            if now - self._last_tf_warn_time > 2.0:
                self.get_logger().warning(
                    f"TF转换失败: {source_frame} -> {self.map_frame}: {exc}"
                )
                self._last_tf_warn_time = now
            return None

        tr = tf_msg.transform.translation
        rot = tf_msg.transform.rotation
        tf_yaw = self._quat_to_yaw(rot.x, rot.y, rot.z, rot.w)
        c, s = math.cos(tf_yaw), math.sin(tf_yaw)

        x_map = c * x_odom - s * y_odom + tr.x
        y_map = s * x_odom + c * y_odom + tr.y
        yaw_map = self._normalize_angle(yaw_odom + tf_yaw)
        return x_map, y_map, yaw_map

    @staticmethod
    def _goal_status_to_text(status: int) -> str:
        mapping = {
            GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
            GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
            GoalStatus.STATUS_EXECUTING: "EXECUTING",
            GoalStatus.STATUS_CANCELING: "CANCELING",
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_CANCELED: "CANCELED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
        }
        return mapping.get(status, f"UNDEFINED({status})")

    def _make_pose_stamped(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        frame_id: str,
    ) -> PoseStamped:
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
        return pose

    def publish_goal(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        frame_id: str = "map",
        target_name: str = "",
    ) -> None:
        pose = self._make_pose_stamped(x, y, z, yaw, frame_id)

        # 新模式：通过 /goal_pose 下发，并用距离阈值判定到达
        if self.goal_send_mode == "goal_pose":
            if self.goal_pose_pub is None:
                self.get_logger().error("goal_pose publisher 未初始化，目标未发送")
                return

            self.goal_pose_pub.publish(pose)

            with self._goal_arrival_lock:
                self._active_goal_pose = {
                    "x": x,
                    "y": y,
                    "z": z,
                    "yaw": yaw,
                    "frame_id": frame_id,
                    "target_name": target_name,
                }
                self._goal_arrived_reported = False

            self.get_logger().info(
                f"已通过 /goal_pose 发布目标: {target_name} "
                f"goal=({x:.3f}, {y:.3f}, {z:.3f}), "
                f"arrival_threshold={self.arrival_threshold:.3f}m"
            )
            ws.set_nav_status_text(f"{target_name}: GOAL_SENT")
            return

        if self.goal_pose_pub is not None and self.publish_goal_pose_topic:
            self.goal_pose_pub.publish(pose)

        # 旧模式：Action 下发
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        if not self.nav2_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error(
                f"Nav2 Action server '{self.nav2_action_name}' 不可用，目标未发送"
            )
            return

        if not self._server_ready_logged:
            self.get_logger().info(
                f"Nav2 Action server '{self.nav2_action_name}' 已连接"
            )
            self._server_ready_logged = True

        with self._goal_handle_lock:
            old_handle = self._current_goal_handle
            old_name = self._current_goal_name

        if old_handle is not None:
            self.get_logger().info(f"发送新目标前取消旧目标: {old_name}")
            try:
                cancel_future = old_handle.cancel_goal_async()
                cancel_future.add_done_callback(
                    lambda fut, name=old_name: self._on_cancel_done(
                        fut, name
                    )
                )
            except Exception as exc:
                self.get_logger().warning(f"取消旧目标失败: {exc}")

        send_future = self.nav2_action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(
            lambda fut, name=target_name: self._on_nav_goal_response(fut, name)
        )

    def _check_goal_arrival(self) -> None:
        # 仅在 goal_pose 模式下启用距离判定
        if self.goal_send_mode != "goal_pose":
            return

        with self._goal_arrival_lock:
            goal = copy.deepcopy(self._active_goal_pose)
            already_reported = self._goal_arrived_reported

        if goal is None or already_reported:
            return

        odom_rw_lock.acquire_read()
        odom_snapshot = copy.deepcopy(self.odom)
        odom_rw_lock.release_read()
        if odom_snapshot is None:
            return

        robot_x, robot_y = odom_snapshot[0], odom_snapshot[1]
        dx = goal["x"] - robot_x
        dy = goal["y"] - robot_y
        dist = math.hypot(dx, dy)

        ws.set_nav_status_text(
            f'{goal["target_name"]}: DIST={dist:.3f}m / TH={self.arrival_threshold:.3f}m'
        )

        if dist < self.arrival_threshold:
            with self._goal_arrival_lock:
                if self._goal_arrived_reported:
                    return
                self._goal_arrived_reported = True

            self.get_logger().info(
                f'到达目标: {goal["target_name"]}, '
                f"dist={dist:.3f}m < threshold={self.arrival_threshold:.3f}m"
            )
            ws.set_nav_success(goal["target_name"])
            ws.set_nav_status_text(
                f'{goal["target_name"]}: ARRIVED_BY_DISTANCE '
                f"(dist={dist:.3f}m, th={self.arrival_threshold:.3f}m)"
            )

    def _on_cancel_done(self, future, target_name: str) -> None:
        try:
            _ = future.result()
            self.get_logger().info(f"旧目标取消请求已完成: {target_name}")
        except Exception as exc:
            self.get_logger().warning(f"旧目标取消异常 [{target_name}]: {exc}")

    def _on_nav_goal_response(self, future, target_name: str) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"导航目标发送异常 [{target_name}]: {exc}")
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warning(f"导航目标被拒绝: {target_name}")
            return

        with self._goal_handle_lock:
            self._current_goal_handle = goal_handle
            self._current_goal_name = target_name

        self.get_logger().info(f"导航目标已接受: {target_name}")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda fut, name=target_name: self._on_nav_result(fut, name)
        )

    def _on_nav_result(self, future, target_name: str) -> None:
        try:
            result_wrap = future.result()
        except Exception as exc:
            self.get_logger().error(f"导航结果获取异常 [{target_name}]: {exc}")
            return

        status = result_wrap.status
        status_text = self._goal_status_to_text(status)

        with self._goal_handle_lock:
            if self._current_goal_name == target_name:
                self._current_goal_handle = None
                self._current_goal_name = ""

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"导航成功: {target_name}")
            ws.set_nav_success(target_name)
            ws.set_nav_status_text(status_text)
        else:
            self.get_logger().warning(f"导航结束 [{target_name}]，status={status_text}")
            ws.set_nav_status_text(f"{target_name}: {status_text}")

    def publish_object_markers(
        self,
        objects_world: list[dict],
        frame_id: str = "map",
    ) -> None:
        now = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        for idx, obj in enumerate(objects_world):
            world = obj.get("world", [0.0, 0.0, 0.0])
            if world is None or len(world) != 3:
                continue

            track_id = int(obj.get("track_id", idx + 1))
            class_name = str(obj.get("class_name", "unknown"))
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
            pos_marker.lifetime = Duration(seconds=0.0).to_msg()
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
            text_marker.scale.z = 0.32
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.95
            text_marker.text = f"{class_name}"
            text_marker.lifetime = Duration(seconds=0.0).to_msg()
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
        self.rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1.0e9
        rgb_depth_rw_lock.release_write()

        self.new_image_arrived = True

    def odom_callback(self, msg):
        odom_rw_lock.acquire_write()
        ori = msg.pose.pose.orientation
        yaw = self._quat_to_yaw(ori.x, ori.y, ori.z, ori.w)

        x_odom = msg.pose.pose.position.x
        y_odom = msg.pose.pose.position.y
        odom_frame_id = msg.header.frame_id or self.odom_frame or "odom"

        converted = self._odom_pose_to_map(x_odom, y_odom, yaw, odom_frame_id)
        if converted is not None:
            x_use, y_use, yaw_use = converted
            odom_is_map = True
        else:
            x_use, y_use, yaw_use = x_odom, y_odom, yaw
            odom_is_map = False

        odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1.0e9
        self.odom = [x_use, y_use, yaw_use]
        self.odom_in_map = odom_is_map
        self.odom_queue.append(
            (odom_stamp, copy.deepcopy(self.odom), odom_is_map, odom_frame_id)
        )
        self.odom_timestamp = odom_stamp
        odom_rw_lock.release_write()


# -------------------------------------------
# Main
# -------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO detect HTTP client (ROS2)")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:5802/detect",
        help="yolo_detect_server /detect endpoint",
    )
    parser.add_argument(
        "--rgb-topic", type=str, default="/camera/camera/color/image_raw"
    )
    parser.add_argument(
        "--depth-topic",
        type=str,
        default="/camera/camera/aligned_depth_to_color/image_raw",
    )
    parser.add_argument("--odom-topic", type=str, default="/odom_bridge")
    parser.add_argument(
        "--odom-frame",
        type=str,
        default=None,
        help="里程计坐标系名称（默认自动使用 odom 消息 header.frame_id）",
    )
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=15,
        help="每隔 N 帧处理 1 帧",
    )
    parser.add_argument(
        "--cam-json",
        type=str,
        default=None,
        help="相机参数 JSON 文件路径",
    )
    parser.add_argument("--cam-offset-x", type=float, default=None)
    parser.add_argument("--cam-offset-y", type=float, default=None)
    parser.add_argument("--cam-offset-z", type=float, default=None)

    parser.add_argument(
        "--goal-topic",
        type=str,
        default="/goal_pose",
        help="发布 PoseStamped 的 topic",
    )
    parser.add_argument(
        "--no-goal-topic-pub",
        action="store_true",
        help="action 模式下不额外发布 PoseStamped 到 --goal-topic；goal_pose 模式下此参数无效",
    )
    parser.add_argument(
        "--goal-send-mode",
        type=str,
        default="action",
        choices=["action", "goal_pose"],
        help="目标发送/到达判定模式：action 或 goal_pose",
    )
    parser.add_argument(
        "--arrival-threshold",
        type=float,
        default=0.30,
        help="goal_pose 模式下的到达距离阈值（米）",
    )
    parser.add_argument(
        "--marker-topic",
        type=str,
        default="/yolo_detect/markers",
    )
    parser.add_argument(
        "--nav2-status-topic",
        type=str,
        default="/navigate_to_pose/_action/status",
        help="保留兼容参数，action 模式下成功判定以 NavigateToPose Action result 为准",
    )
    parser.add_argument("--map-frame", type=str, default="map")
    parser.add_argument("--base-link-frame", type=str, default="base_link")
    parser.add_argument("--goal-standoff", type=float, default=0.5)
    parser.add_argument("--z-offset", type=float, default=0.0)

    parser.add_argument("--lidar-topic", type=str, default=None)
    parser.add_argument(
        "--lidar-type",
        type=str,
        default="pointcloud2",
        choices=["laserscan", "pointcloud2"],
    )
    parser.add_argument(
        "--camera-frame",
        type=str,
        default="camera_color_optical_frame",
    )
    parser.add_argument("--lidar-max-points", type=int, default=32000)

    parser.add_argument("--deepseek-api-key", type=str, default=None)
    parser.add_argument(
        "--deepseek-base-url",
        type=str,
        default="https://api.deepseek.com",
    )
    parser.add_argument(
        "--deepseek-model",
        type=str,
        default="deepseek-chat",
    )

    parser.add_argument("--webui-port", type=int, default=8080)
    parser.add_argument("--no-webui", action="store_true")
    args = parser.parse_args()

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
        f"[Client] 语义导航: mode={args.goal_send_mode}"
        f" goal_topic={args.goal_topic}"
        f" standoff={args.goal_standoff}m z_offset={args.z_offset}m"
        f" arrival_threshold={args.arrival_threshold:.3f}m"
    )

    if args.goal_send_mode == "action":
        print("[Client] 到达判定: 使用 navigate_to_pose Action result.status")
    else:
        print("[Client] 到达判定: 使用当前位置与目标位置距离阈值判断")

    if args.lidar_topic:
        print(
            f"[Client] 激光雷达: topic={args.lidar_topic}"
            f" type={args.lidar_type}"
            f" camera_frame={args.camera_frame}"
            f" max_pts={args.lidar_max_points}"
        )
    else:
        print("[Client] 未配置激光雷达，仅使用 RGBD 深度")

    server_base_url = args.server_url.rstrip("/")
    if server_base_url.endswith("/detect"):
        server_base_url = server_base_url[: -len("/detect")]

    ws.init_webui_config(
        server_base_url_=server_base_url,
        goal_standoff_=args.goal_standoff,
        z_offset_=args.z_offset,
        map_frame_=args.map_frame,
        deepseek_api_key_=args.deepseek_api_key,
        deepseek_base_url_=args.deepseek_base_url,
        deepseek_model_=args.deepseek_model,
    )

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
            map_frame=args.map_frame,
            odom_frame=args.odom_frame,
            goal_topic=args.goal_topic,
            marker_topic=args.marker_topic,
            lidar_topic=args.lidar_topic,
            lidar_type=args.lidar_type,
            publish_goal_pose_topic=not args.no_goal_topic_pub,
            goal_send_mode=args.goal_send_mode,
            arrival_threshold=args.arrival_threshold,
        )

        ws.bind_runtime_dependencies(
            manager_getter=lambda: manager,
            odom_lock_getter=lambda: odom_rw_lock,
            call_list_classes=call_list_classes,
            call_query=call_query,
            parse_intent_with_deepseek=parse_intent_with_deepseek,
            is_natural_language=_is_natural_language,
        )

        det_thread.start()
        iac_thread.start()

        if not args.no_webui:
            ui_thread = start_webui(args.webui_port)
            print(f"[WebUI] 浏览器访问: http://0.0.0.0:{args.webui_port}")

        print(f"[Client] started, server: {args.server_url}")
        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        if manager is not None:
            manager.destroy_node()
        rclpy.shutdown()
