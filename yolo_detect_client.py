#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 目标检测 HTTP 客户端（ROS2 版），含交互式语义导航目标查询。

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

启动：
    python -u yolo_detect_client.py \
        --server-url http://127.0.0.1:5802/detect \
        --cam-json cam_params.json \
        --frame-stride 10 \
        --goal-standoff 0.6 \
        --map-frame map \
        --base-link-frame base_link \
        --deepseek-api-key <YOUR_API_KEY>

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

import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from PIL import Image as PIL_Image
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from action_msgs.msg import GoalStatusArray, GoalStatus
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

    if lidar_points_cam is not None and lidar_points_cam.shape[0] > 0:
        pts_f32 = lidar_points_cam.astype(np.float32)
        payload["lidar_num_points"] = int(pts_f32.shape[0])
        payload["lidar_points_b64"] = base64.b64encode(pts_f32.tobytes()).decode("ascii")

    json_payload = json.dumps(payload)

    files = {
        "image": ("rgb_image.jpg", image_bytes, "image/jpeg"),
        "depth": ("depth_image.png", depth_bytes, "image/png"),
    }

    t0 = time.time()
    response = requests.post(
        url, files=files, data={"json": json_payload}, timeout=timeout
    )
    elapsed = time.time() - t0
    lidar_info = f"  lidar_pts={lidar_points_cam.shape[0]}" if lidar_points_cam is not None else ""
    # print(f"[Client] HTTP {response.status_code}, cost {elapsed:.3f}s{lidar_info}")

    return json.loads(response.text)


# -------------------------------------------
# Query / list HTTP helpers
# -------------------------------------------
def call_query(
    base_url: str,
    class_name: str,
    robot_x: float,
    robot_y: float,
    timeout: float = 10.0,
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
                print(
                    f"[DEBUG][LiDAR] raw={lidar_pts_raw.shape[0]}pts"
                    f"  no_tf_transform"
                    f"  send={lidar_pts_cam.shape[0]}pts({_payload_kb:.1f}KB)"
                    f"  prepare={(_t_prepare1 - _t_prepare0)*1000:.2f}ms"
                )

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
        self.goal_pub = self.create_publisher(PoseStamped, goal_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.nav2_status_sub = self.create_subscription(
            GoalStatusArray,
            nav2_status_topic,
            self.nav2_status_callback,
            qos_profile,
        )
        self._succeeded_goal_ids: set[tuple[int, ...]] = set()

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
            f"nav2_status_topic={nav2_status_topic}"
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
        """发布 PoseStamped 导航目标到 /goal_pose（Nav2 兼容）。"""
        half = 0.5 * yaw
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = frame_id
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = math.sin(half)
        goal.pose.orientation.w = math.cos(half)
        self.goal_pub.publish(goal)

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

    def nav2_status_callback(self, msg: GoalStatusArray) -> None:
        """监听 Nav2 导航状态；若导航成功则打印“导航成功”。"""
        for status_item in msg.status_list:
            if status_item.status == GoalStatus.STATUS_SUCCEEDED:
                goal_id = tuple(status_item.goal_info.goal_id.uuid)
                if goal_id not in self._succeeded_goal_ids:
                    self._succeeded_goal_ids.add(goal_id)
                    print("导航成功")


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
        default=10,
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
        help="Nav2 导航状态话题，监听成功状态（默认 /navigate_to_pose/_action/status）",
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
        default="/xt16_cloud",
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
    print(f"[Client] Nav2 状态监听: topic={args.nav2_status_topic}")
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
        print(f"[Client] started, server: {args.server_url}")
        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        manager.destroy_node()
        rclpy.shutdown()