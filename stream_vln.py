#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + RGBD + Pose: 输出目标中心点的相机/世界坐标，并可选保存可视化与俯视轨迹图。

支持两种使用模式：

1. 批处理模式（命令行）：
   python stream_vln.py --input-dir <dir> ...

2. 流式模式（API 调用）：
   from stream_vln import StreamProcessor
   proc = StreamProcessor(model="yolov8s-world.pt", fx=..., fy=..., cx=..., cy=...)
   for rgb, depth, pose in data_source:
       results = proc.process_frame(rgb, depth, pose)
       for r in results:
           print(r.track_id, r.class_name, r.world_fused)

输入目录结构（批处理模式）：
  <input-dir>/
    traj.txt                  # 每行16个数，4x4位姿矩阵（与frame/depth索引对齐）
    results/                  # 或者图像直接在input-dir下
      frameXXXX.jpg
      depthXXXX.png           # 单通道深度（常见为uint16，单位mm或由cam-json/--depth-scale指定）

输出：
  - CSV：每帧每个track的中心点坐标（相机/世界 raw/fused）
  - 可选：rgb/depth 可视化标注图
  - 可选：世界坐标系XY俯视散点轨迹图（含相机轨迹）
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# 重要：在无GUI服务器上保存图像需要Agg后端
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ultralytics import YOLO


# -----------------------------
# Data classes for streaming results
# -----------------------------
@dataclass
class FrameDetectionResult:
    """单个检测目标在一帧中的结构化输出。"""

    track_id: int
    cls_id: int
    class_name: str
    confidence: float
    u: int
    v: int
    bbox_xyxy: tuple[float, float, float, float]
    depth_m: float
    cam_xyz: np.ndarray        # shape (3,) 相机坐标
    world_raw: np.ndarray      # shape (3,) 世界坐标（未融合）
    world_fused: np.ndarray    # shape (3,) 世界坐标（融合后）

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "cls_id": self.cls_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "u": self.u,
            "v": self.v,
            "bbox_xyxy": self.bbox_xyxy,
            "depth_m": self.depth_m,
            "cam_xyz": self.cam_xyz.tolist(),
            "world_raw": self.world_raw.tolist(),
            "world_fused": self.world_fused.tolist(),
        }


# -----------------------------
# IO / Utils
# -----------------------------
def get_nested(d: dict, dotted_key: str):
    cur = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_cam_json(cam_json: Path | None) -> dict | None:
    if cam_json is None:
        return None
    with cam_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_traj(traj_path: Path) -> np.ndarray:
    poses = []
    with traj_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            values = np.fromstring(text, sep=" ")
            if values.size != 16:
                raise ValueError(
                    f"traj.txt 第 {line_no} 行应为16个数，实际为 {values.size}: {traj_path}"
                )
            poses.append(values.reshape(4, 4))

    if not poses:
        raise ValueError(f"traj.txt 为空: {traj_path}")

    return np.stack(poses, axis=0)


def load_intrinsics(
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
    cam_data: dict | None,
) -> tuple[float, float, float, float]:
    if fx is not None and fy is not None and cx is not None and cy is not None:
        return fx, fy, cx, cy

    if cam_data is None:
        raise ValueError("请通过 --fx --fy --cx --cy 或 --cam-json 提供相机内参")

    data = cam_data

    # 1) 兼容 K = [fx,0,cx, 0,fy,cy, 0,0,1]
    if (
        isinstance(data, dict)
        and "K" in data
        and isinstance(data["K"], (list, tuple))
        and len(data["K"]) >= 9
    ):
        k = data["K"]
        return float(k[0]), float(k[4]), float(k[2]), float(k[5])

    # 2) 兼容常见嵌套字段
    key_sets = [
        ("fx", "fy", "cx", "cy"),
        ("camera.fx", "camera.fy", "camera.cx", "camera.cy"),
        ("intrinsics.fx", "intrinsics.fy", "intrinsics.cx", "intrinsics.cy"),
    ]

    for ks in key_sets:
        vals = [get_nested(data, k) for k in ks]
        if all(v is not None for v in vals):
            return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])

    raise ValueError("无法从 --cam-json 解析内参，请改用 --fx --fy --cx --cy")


def resolve_depth_scale(arg_depth_scale: float | None, cam_data: dict | None) -> float:
    """
    depth_scale: 将 depth 图像数值转换到米的比例
      - 若传了 --depth-scale，直接用
      - 若 cam-json 里有 camera.scale 或 scale（常见含义为：depth单位=1/scale米）
        则返回 1/scale
      - 否则默认 0.001（即 mm -> m）
    """
    if arg_depth_scale is not None:
        return arg_depth_scale

    if cam_data is not None:
        scale = get_nested(cam_data, "camera.scale")
        if scale is None:
            scale = get_nested(cam_data, "scale")
        if scale is not None:
            scale_val = float(scale)
            if scale_val > 0:
                return 1.0 / scale_val

    return 0.001


def parse_classes(names: dict[int, str], class_filter: str | None) -> set[int] | None:
    if not class_filter:
        return None

    wanted = {x.strip().lower() for x in class_filter.split(",") if x.strip()}
    if not wanted:
        return None

    valid_ids = {idx for idx, name in names.items() if name.lower() in wanted}
    return valid_ids if valid_ids else set()


def parse_vocab_arg(vocab_arg: str | None) -> list[str] | None:
    if not vocab_arg:
        return None

    path = Path(vocab_arg)
    if path.exists() and path.is_file():
        tokens = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return tokens if tokens else None

    tokens = [token.strip() for token in vocab_arg.split(",") if token.strip()]
    return tokens if tokens else None


def apply_model_vocabulary(model: YOLO, vocab_list: list[str] | None) -> None:
    if not vocab_list:
        return

    if hasattr(model, "set_classes"):
        model.set_classes(vocab_list)
        print(f"[Info] 已设置查询词表({len(vocab_list)}): {vocab_list}")
    else:
        print("[Warn] 当前模型不支持 set_classes，已忽略 --vocab")


def list_frames(results_dir: Path) -> tuple[list[Path], list[Path]]:
    frames = sorted(results_dir.glob("frame*.jpg"))
    depths = sorted(results_dir.glob("depth*.png"))
    if not frames:
        raise FileNotFoundError(f"未找到 RGB 图像: {results_dir}/frame*.jpg")
    if not depths:
        raise FileNotFoundError(f"未找到 深度图像: {results_dir}/depth*.png")
    return frames, depths


def parse_frame_index(path: Path, prefix: str) -> int | None:
    m = re.match(rf"^{prefix}(\d+)\.[^.]+$", path.name)
    if m is None:
        return None
    return int(m.group(1))


def build_aligned_samples(
    frames: list[Path],
    depths: list[Path],
    pose_count: int,
) -> list[tuple[Path, Path, int]]:
    """
    优先使用 frameXXXX / depthXXXX 的索引进行对齐；若解析不到索引，则退化为按序对齐（取最短长度）。
    """
    frame_indexed: list[tuple[int, Path]] = []
    for fp in frames:
        idx = parse_frame_index(fp, "frame")
        if idx is not None:
            frame_indexed.append((idx, fp))

    depth_by_idx: dict[int, Path] = {}
    for dp in depths:
        idx = parse_frame_index(dp, "depth")
        if idx is not None:
            depth_by_idx[idx] = dp

    samples: list[tuple[Path, Path, int]] = []
    if frame_indexed and depth_by_idx:
        for idx, fp in sorted(frame_indexed, key=lambda x: x[0]):
            dp = depth_by_idx.get(idx)
            if dp is None:
                continue
            if idx < 0 or idx >= pose_count:
                continue
            samples.append((fp, dp, idx))
        if samples:
            return samples

    total = min(len(frames), len(depths), pose_count)
    return [(frames[i], depths[i], i) for i in range(total)]


def resolve_traj_path(input_dir: Path, explicit_traj: Path | None) -> Path:
    if explicit_traj is not None:
        return explicit_traj

    direct = input_dir / "traj.txt"
    if direct.exists():
        return direct

    # 兼容 xxx_sampled_iN 目录（回退到父目录同名序列的traj.txt）
    sampled_match = re.match(r"^(.*)_sampled_i\d+$", input_dir.name)
    if sampled_match:
        base = sampled_match.group(1)
        fallback = input_dir.parent / base / "traj.txt"
        if fallback.exists():
            print(f"[Info] 当前目录缺少traj.txt，自动回退到: {fallback}")
            return fallback

    raise FileNotFoundError(f"未找到traj.txt: {direct}")


def resolve_results_dir(input_dir: Path, explicit_results_dir: Path | None) -> Path:
    if explicit_results_dir is not None:
        return explicit_results_dir

    default_results = input_dir / "results"
    if default_results.exists():
        return default_results

    # 否则认为图像就放在input_dir下
    return input_dir


# -----------------------------
# Geometry
# -----------------------------
def depth_patch_bounds(h: int, w: int, u: int, v: int, radius: int) -> tuple[int, int, int, int]:
    u0 = max(0, u - radius)
    u1 = min(w, u + radius + 1)
    v0 = max(0, v - radius)
    v1 = min(h, v + radius + 1)
    return u0, u1, v0, v1


def depth_at_center(depth_mm: np.ndarray, u: int, v: int, radius: int) -> float:
    h, w = depth_mm.shape
    u0, u1, v0, v1 = depth_patch_bounds(h=h, w=w, u=u, v=v, radius=radius)
    patch = depth_mm[v0:v1, u0:u1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def camera_to_world(T_wc: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
    """
    将光学相机坐标系(右,下,前)的点(x,y,z)映射到世界坐标：
      - 这里假设里程计/世界坐标使用(前,左,上)
      - 因此做一个轴变换：x_fwd=z, y_left=-x, z_up=-y
      - 再用 T_wc 变换到世界系
    """
    x_fwd = z
    y_left = -x
    z_up = -y
    p_c = np.array([x_fwd, y_left, z_up, 1.0], dtype=np.float64)
    p_w = T_wc @ p_c
    return p_w[:3]


# -----------------------------
# Visualization Helpers
# -----------------------------
def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        gray = np.zeros_like(depth_mm, dtype=np.uint8)
    else:
        max_v = float(np.percentile(valid, 98))
        max_v = max(max_v, 1.0)
        gray = np.clip(depth_mm.astype(np.float32) / max_v * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


@dataclass
class DetectionCandidate:
    cls_id: int
    class_name: str
    conf: float
    u: int
    v: int
    bbox_xyxy: np.ndarray
    bbox_xyxy_int: tuple[int, int, int, int]
    depth_patch_uvxy: tuple[int, int, int, int]
    cam_xyz: np.ndarray
    world_xyz_raw: np.ndarray


def draw_detection_on_rgb(
    image: np.ndarray,
    det: DetectionCandidate,
    track_id: int,
    world_xyz: np.ndarray,
) -> np.ndarray:
    canvas = image.copy()
    x1, y1, x2, y2 = det.bbox_xyxy_int
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(canvas, (det.u, det.v), 4, (0, 0, 255), -1)
    text = (
        f"id={track_id} {det.class_name} {det.conf:.2f} | "
        f"CenterW=({world_xyz[0]:.2f},{world_xyz[1]:.2f},{world_xyz[2]:.2f})"
    )
    cv2.putText(
        canvas,
        text,
        (max(5, det.u - 120), max(20, det.v - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
    )
    return canvas


def draw_detection_on_depth(
    depth_color: np.ndarray,
    det: DetectionCandidate,
    track_id: int,
) -> np.ndarray:
    canvas = depth_color.copy()
    x1, y1, x2, y2 = det.bbox_xyxy_int
    u0, u1, v0, v1 = det.depth_patch_uvxy
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 显示用：当采样框太小时放大显示，便于肉眼观察（不影响实际深度计算）
    pw = max(1, u1 - u0)
    ph = max(1, v1 - v0)
    min_show = 12
    cx = det.u
    cy = det.v
    if pw < min_show or ph < min_show:
        half = max(min_show // 2, 1)
        su0 = max(0, cx - half)
        sv0 = max(0, cy - half)
        su1 = min(canvas.shape[1], cx + half + 1)
        sv1 = min(canvas.shape[0], cy + half + 1)
    else:
        su0, su1, sv0, sv1 = u0, u1, v0, v1

    # 黑色描边 + 黄色内框
    cv2.rectangle(canvas, (su0, sv0), (max(su0, su1 - 1), max(sv0, sv1 - 1)), (0, 0, 0), 3)
    cv2.rectangle(canvas, (su0, sv0), (max(su0, su1 - 1), max(sv0, sv1 - 1)), (0, 255, 255), 2)
    cv2.circle(canvas, (det.u, det.v), 4, (0, 0, 255), -1)
    cv2.putText(
        canvas,
        f"id={track_id} {det.class_name}",
        (max(5, det.u - 100), max(20, det.v - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    return canvas


# -----------------------------
# Tracking / Fusion
# -----------------------------
@dataclass
class TrackState:
    track_id: int
    cls_id: int
    last_bbox: np.ndarray
    history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=5))
    miss_count: int = 0
    kf_x: np.ndarray | None = None
    kf_P: np.ndarray | None = None
    latest_world_raw: np.ndarray | None = None
    latest_world_fused: np.ndarray | None = None


@dataclass
class FrameObservation:
    track_id: int
    det: DetectionCandidate
    fused_world: np.ndarray


def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    a1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    a2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union = a1 + a2 - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def match_tracks(
    tracks: dict[int, TrackState],
    detections: list[DetectionCandidate],
    iou_thres: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    pairs: list[tuple[float, int, int]] = []
    track_ids = list(tracks.keys())

    for t_id in track_ids:
        track = tracks[t_id]
        for d_idx, det in enumerate(detections):
            if det.cls_id != track.cls_id:
                continue
            iou = bbox_iou(track.last_bbox, det.bbox_xyxy)
            if iou >= iou_thres:
                pairs.append((iou, t_id, d_idx))

    pairs.sort(key=lambda x: x[0], reverse=True)
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    matches: list[tuple[int, int]] = []

    for _, t_id, d_idx in pairs:
        if t_id in used_tracks or d_idx in used_dets:
            continue
        used_tracks.add(t_id)
        used_dets.add(d_idx)
        matches.append((t_id, d_idx))

    unmatched_tracks = [t_id for t_id in track_ids if t_id not in used_tracks]
    unmatched_dets = [idx for idx in range(len(detections)) if idx not in used_dets]
    return matches, unmatched_tracks, unmatched_dets


def kalman_update(track: TrackState, measurement: np.ndarray) -> np.ndarray:
    if track.kf_x is None or track.kf_P is None:
        track.kf_x = measurement.astype(np.float64).copy()
        track.kf_P = np.eye(3, dtype=np.float64) * 0.05
        return track.kf_x.copy()

    x = track.kf_x
    P = track.kf_P

    Q = np.eye(3, dtype=np.float64) * 0.002
    R = np.eye(3, dtype=np.float64) * 0.01

    x_pred = x
    P_pred = P + Q

    y = measurement - x_pred
    S = P_pred + R
    K = P_pred @ np.linalg.inv(S)

    x_new = x_pred + K @ y
    P_new = (np.eye(3, dtype=np.float64) - K) @ P_pred

    track.kf_x = x_new
    track.kf_P = P_new
    return x_new.copy()


def fuse_world_xyz(track: TrackState, world_xyz_raw: np.ndarray, fusion_mode: str) -> np.ndarray:
    if fusion_mode == "none":
        return world_xyz_raw

    if fusion_mode == "moving_average":
        track.history.append(world_xyz_raw)
        stacked = np.stack(list(track.history), axis=0)
        return stacked.mean(axis=0)

    if fusion_mode == "kalman":
        return kalman_update(track, world_xyz_raw)

    raise ValueError(f"未知 fusion_mode: {fusion_mode}")


def build_track_merge_map(
    tracks: dict[int, TrackState],
    merge_distance_thres: float,
) -> dict[int, int]:
    """
    同类别track若最新fused位置距离<=阈值，则合并（并查集）。
    返回：track_id -> root_id
    """
    if merge_distance_thres <= 0 or len(tracks) <= 1:
        return {tid: tid for tid in tracks.keys()}

    parent: dict[int, int] = {tid: tid for tid in tracks.keys()}

    def find(tid: int) -> int:
        root = tid
        while parent[root] != root:
            root = parent[root]
        while parent[tid] != tid:
            nxt = parent[tid]
            parent[tid] = root
            tid = nxt
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        parent[max(ra, rb)] = min(ra, rb)

    valid_ids = [tid for tid, t in tracks.items() if t.latest_world_fused is not None]
    for i in range(len(valid_ids)):
        tid_i = valid_ids[i]
        ti = tracks[tid_i]
        pi = ti.latest_world_fused
        if pi is None:
            continue
        for j in range(i + 1, len(valid_ids)):
            tid_j = valid_ids[j]
            tj = tracks[tid_j]
            if ti.cls_id != tj.cls_id:
                continue
            pj = tj.latest_world_fused
            if pj is None:
                continue
            dist = float(np.linalg.norm(pi - pj))
            if dist <= merge_distance_thres:
                union(tid_i, tid_j)

    return {tid: find(tid) for tid in tracks.keys()}


def apply_track_merges(
    tracks: dict[int, TrackState],
    merge_map: dict[int, int],
    moving_avg_window: int,
) -> None:
    grouped: dict[int, list[int]] = {}
    for tid, root in merge_map.items():
        grouped.setdefault(root, []).append(tid)

    for root_id, members in grouped.items():
        if root_id not in tracks:
            continue
        root_track = tracks[root_id]
        for mid in members:
            if mid == root_id or mid not in tracks:
                continue
            member = tracks[mid]

            merged_hist = list(root_track.history) + list(member.history)
            if merged_hist:
                root_track.history = deque(merged_hist[-moving_avg_window:], maxlen=moving_avg_window)

            if root_track.latest_world_raw is None:
                root_track.latest_world_raw = member.latest_world_raw
            elif member.latest_world_raw is not None:
                root_track.latest_world_raw = 0.5 * (root_track.latest_world_raw + member.latest_world_raw)

            if root_track.latest_world_fused is None:
                root_track.latest_world_fused = member.latest_world_fused
            elif member.latest_world_fused is not None:
                root_track.latest_world_fused = 0.5 * (root_track.latest_world_fused + member.latest_world_fused)

            if root_track.kf_x is None:
                root_track.kf_x = member.kf_x
            elif member.kf_x is not None:
                root_track.kf_x = 0.5 * (root_track.kf_x + member.kf_x)

            if root_track.kf_P is None:
                root_track.kf_P = member.kf_P
            elif member.kf_P is not None:
                root_track.kf_P = 0.5 * (root_track.kf_P + member.kf_P)

            root_track.miss_count = min(root_track.miss_count, member.miss_count)
            del tracks[mid]


# -----------------------------
# Plot
# -----------------------------
def set_equal_aspect_2d(ax, points_xy: np.ndarray) -> None:
    mins = points_xy.min(axis=0)
    maxs = points_xy.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)


def save_world_track_plot(
    track_points: dict[int, list[np.ndarray]],
    track_class_names: dict[int, str],
    track_confidences: dict[int, list[float]],
    camera_traj_xyz: np.ndarray,
    out_path: Path,
    min_points: int,
) -> None:
    valid_items: list[tuple[int, np.ndarray]] = []
    for tid, pts in track_points.items():
        if len(pts) < min_points:
            continue
        valid_items.append((tid, np.stack(pts, axis=0)))

    if not valid_items:
        print(f"[Warn] 无可绘制轨迹（min_points={min_points}），跳过世界坐标图导出")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    all_xy = []
    for tid, points in valid_items:
        conf_list = track_confidences.get(tid, [])
        if conf_list:
            conf_avg = float(np.mean(conf_list))
            conf_last = float(conf_list[-1])
            label = f"id={tid}:{track_class_names.get(tid, 'obj')} conf={conf_avg:.2f}/{conf_last:.2f}"
        else:
            label = f"id={tid}:{track_class_names.get(tid, 'obj')} conf=NA"

        xy = points[:, :2]
        ax.scatter(xy[:, 0], xy[:, 1], s=14, alpha=0.85, label=label)
        all_xy.append(xy)

    if camera_traj_xyz.shape[0] > 0:
        cam_xy = camera_traj_xyz[:, :2]
        ax.plot(cam_xy[:, 0], cam_xy[:, 1], "k--", linewidth=1.2, label="camera_traj")
        ax.scatter(cam_xy[0, 0], cam_xy[0, 1], c="k", s=30, marker="^")
        ax.scatter(cam_xy[-1, 0], cam_xy[-1, 1], c="k", s=30, marker="v")
        all_xy.append(cam_xy)

    stacked_xy = np.concatenate(all_xy, axis=0)
    set_equal_aspect_2d(ax, stacked_xy)
    ax.set_xlabel("X_world")
    ax.set_ylabel("Y_world")
    ax.set_title("Top-down object center points in world coordinates")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"World-coordinate plot saved to: {out_path}")


# -----------------------------
# StreamProcessor: 流式逐帧处理 API
# -----------------------------
class StreamProcessor:
    """
    流式 YOLO + RGBD + Pose 处理器。

    使用方法::

        proc = StreamProcessor(
            model="yolov8s-world.pt",
            fx=615.0, fy=615.0, cx=320.0, cy=240.0,
            depth_scale=0.001,
        )
        # 可选：设置开放词表
        proc.set_vocabulary(["chair", "table", "bottle"])

        for rgb, depth, pose in my_data_stream():
            results = proc.process_frame(rgb, depth, pose)
            for r in results:
                print(f"Track {r.track_id}: {r.class_name} @ world {r.world_fused}")

        # 可选：最终输出汇总图
        proc.save_world_plot("world_tracks.png")
    """

    def __init__(
        self,
        model: str | Path | YOLO = "yolov8s-world.pt",
        fx: float = 615.0,
        fy: float = 615.0,
        cx: float = 320.0,
        cy: float = 240.0,
        depth_scale: float = 0.001,
        cam_json: Path | str | None = None,
        conf_thres: float = 0.4,
        class_filter: str | None = None,
        vocab: str | list[str] | None = None,
        center_patch_radius: int = 2,
        fusion_mode: str = "moving_average",
        moving_avg_window: int = 5,
        track_iou_thres: float = 0.3,
        track_max_miss: int = 8,
        merge_distance_thres: float = 0.35,
        depth_source: str = "rgbd",
        lidar_min_pts_in_bbox: int = 1,
    ):
        if isinstance(model, YOLO):
            self.model = model
        else:
            self.model = YOLO(str(model))

        cam_data = load_cam_json(Path(cam_json)) if cam_json else None

        if cam_data is not None:
            try:
                fx, fy, cx, cy = load_intrinsics(None, None, None, None, cam_data)
            except ValueError:
                pass
            depth_scale = resolve_depth_scale(None, cam_data)

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale
        self.conf_thres = conf_thres
        self.center_patch_radius = max(0, center_patch_radius)
        self.fusion_mode = fusion_mode
        self.moving_avg_window = max(1, moving_avg_window)
        self.track_iou_thres = max(0.0, min(1.0, track_iou_thres))
        self.track_max_miss = max(0, track_max_miss)
        self.merge_distance_thres = max(0.0, merge_distance_thres)
        self.depth_source = depth_source          # "rgbd" | "lidar"
        self.lidar_min_pts_in_bbox = max(1, lidar_min_pts_in_bbox)

        if isinstance(vocab, list):
            apply_model_vocabulary(self.model, vocab)
        elif isinstance(vocab, str):
            apply_model_vocabulary(self.model, parse_vocab_arg(vocab))

        self.class_ids = parse_classes(self.model.names, class_filter)

        self._tracks: dict[int, TrackState] = {}
        self._next_track_id = 1
        self._frame_count = 0

        self._track_points_raw: dict[int, list[np.ndarray]] = {}
        self._track_points_fused: dict[int, list[np.ndarray]] = {}
        self._track_class_names: dict[int, str] = {}
        self._track_confidences: dict[int, list[float]] = {}
        self._camera_positions: list[np.ndarray] = []

        self._last_rgb_vis: np.ndarray | None = None
        self._last_depth_vis: np.ndarray | None = None

    def reset(self) -> None:
        """清除所有跟踪状态，重新开始。"""
        self._tracks.clear()
        self._next_track_id = 1
        self._frame_count = 0
        self._track_points_raw.clear()
        self._track_points_fused.clear()
        self._track_class_names.clear()
        self._track_confidences.clear()
        self._camera_positions.clear()
        self._last_rgb_vis = None
        self._last_depth_vis = None

    def set_vocabulary(self, vocab: list[str]) -> None:
        """动态设置 YOLO-World 开放词表。"""
        apply_model_vocabulary(self.model, vocab)

    @staticmethod
    def _ensure_depth(depth: np.ndarray) -> np.ndarray | None:
        if depth is None:
            return None
        if depth.ndim == 2:
            return depth
        if depth.ndim == 3 and depth.shape[2] >= 1:
            return depth[:, :, 0]
        return None

    @staticmethod
    def _lidar_depth_for_bbox(
        lidar_pts_cam: np.ndarray,
        bbox_xyxy: tuple,
        fx: float,
        fy: float,
        cx_: float,
        cy_: float,
        min_pts: int = 1,
    ) -> float | None:
        """
        从相机坐标系下的激光雷达点云中，为指定检测框估计深度。

        Parameters
        ----------
        lidar_pts_cam : np.ndarray  shape (N, 3)
            相机光学坐标系下的点云 (x_right, y_down, z_forward)。
        bbox_xyxy : tuple  (x1, y1, x2, y2)
            像素检测框。
        fx, fy, cx_, cy_ : float
            相机内参。
        min_pts : int
            框内有效点数下限，不足则返回 None（触发回退到 RGBD）。

        Returns
        -------
        float | None
            框内所有激光雷达点 z（深度）的中位数，单位米；点数不足返回 None。
        """
        if lidar_pts_cam is None or lidar_pts_cam.shape[0] == 0:
            return None

        # 仅保留前方点（z > 0）
        valid_z = lidar_pts_cam[:, 2] > 0
        pts = lidar_pts_cam[valid_z]
        if pts.shape[0] == 0:
            return None

        # 投影到图像平面
        u_proj = fx * pts[:, 0] / pts[:, 2] + cx_
        v_proj = fy * pts[:, 1] / pts[:, 2] + cy_

        x1, y1, x2, y2 = bbox_xyxy
        in_box = (u_proj >= x1) & (u_proj <= x2) & (v_proj >= y1) & (v_proj <= y2)
        in_pts = pts[in_box]

        if in_pts.shape[0] < min_pts:
            return None

        return float(np.median(in_pts[:, 2]))

    @staticmethod
    def _lidar_to_depth_map(
        lidar_pts_cam: np.ndarray,
        h: int,
        w: int,
        fx: float,
        fy: float,
        cx_: float,
        cy_: float,
    ) -> np.ndarray:
        """
        将相机坐标系下的激光雷达点云投影为稀疏深度图（单位：米）。

        Parameters
        ----------
        lidar_pts_cam : np.ndarray  shape (N, 3)
            相机光学坐标系下的点云 (x_right, y_down, z_forward)，单位米。
        h, w : int
            目标图像的高度和宽度（与 RGB 图像一致）。
        fx, fy, cx_, cy_ : float
            相机内参。

        Returns
        -------
        np.ndarray  shape (H, W), dtype=float32
            稀疏深度图，单位米；0 表示该像素无有效激光雷达深度。
            多点投影到同一像素时保留最近（最小 z）的深度。
        """
        depth_map = np.zeros((h, w), dtype=np.float32)
        if lidar_pts_cam is None or lidar_pts_cam.shape[0] == 0:
            return depth_map

        # 仅保留前方点（z > 0）
        valid_z = lidar_pts_cam[:, 2] > 0
        pts = lidar_pts_cam[valid_z]
        if pts.shape[0] == 0:
            return depth_map

        # 投影到图像平面
        u_proj = np.round(fx * pts[:, 0] / pts[:, 2] + cx_).astype(np.int32)
        v_proj = np.round(fy * pts[:, 1] / pts[:, 2] + cy_).astype(np.int32)

        in_image = (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h)
        u_proj = u_proj[in_image]
        v_proj = v_proj[in_image]
        z_vals = pts[in_image, 2]

        # 按 z 从远到近排序，近处点后写覆盖远处点，确保保留最近深度
        order = np.argsort(z_vals)[::-1]
        depth_map[v_proj[order], u_proj[order]] = z_vals[order]
        return depth_map

    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        *,
        frame_idx: int | None = None,
        lidar_points_cam: np.ndarray | None = None,
    ) -> list[FrameDetectionResult]:
        """
        处理单帧数据，返回检测结果列表。

        Parameters
        ----------
        rgb : np.ndarray
            BGR 彩色图像，shape (H, W, 3)，uint8。
        depth : np.ndarray
            深度图，shape (H, W)，原始数值（乘以 depth_scale 后为米）。
        pose : np.ndarray
            4x4 相机位姿矩阵 T_wc（世界←相机）。
        frame_idx : int, optional
            帧编号（仅用于日志，不影响逻辑）。若不传则自动递增。
        lidar_points_cam : np.ndarray | None, shape (N, 3)
            相机光学坐标系下的激光雷达点云（由 client 转换后传入）。
            当 self.depth_source == "lidar" 时优先使用，失败则回退到 RGBD。

        Returns
        -------
        list[FrameDetectionResult]
            该帧中所有被跟踪目标的结构化检测结果。
        """
        if frame_idx is None:
            frame_idx = self._frame_count
        self._frame_count += 1

        depth_mm = self._ensure_depth(depth)
        if depth_mm is None:
            print(f"[Skip] frame {frame_idx}: 深度图维度异常 {depth.shape}")
            return []

        # ── 激光雷达逐像素深度图（可选）────────────────────────────
        # 当 depth_source=="lidar" 且点云有效时，将点云投影为稀疏深度图（单位米）。
        # 后续检测框深度估计优先从该图取值（等效于逐像素深度），失败则回退 RGBD。
        lidar_depth_map: np.ndarray | None = None
        h_img, w_img = depth_mm.shape
        if self.depth_source == "lidar" and lidar_points_cam is not None and lidar_points_cam.shape[0] > 0:
            lidar_depth_map = self._lidar_to_depth_map(
                lidar_points_cam,
                h_img, w_img,
                self.fx, self.fy, self.cx, self.cy,
            )
            valid_px = int((lidar_depth_map > 0).sum())
            print(f"[LiDAR] frame {frame_idx}: 投影有效像素={valid_px}/{h_img*w_img}")

        self._camera_positions.append(pose[:3, 3].copy())

        result = self.model.predict(source=rgb, conf=self.conf_thres, verbose=False)[0]
        candidates: list[DetectionCandidate] = []

        for box in result.boxes:
            cls_id = int(box.cls.item())
            if self.class_ids is not None and cls_id not in self.class_ids:
                continue

            conf = float(box.conf.item())
            if conf < self.conf_thres:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            u = int(round((x1 + x2) * 0.5))
            v = int(round((y1 + y2) * 0.5))

            h, w = depth_mm.shape
            if u < 0 or u >= w or v < 0 or v >= h:
                continue

            # ── 深度估计：激光雷达逐像素深度图 → per-bbox → RGBD 三级回退 ──
            z: float | None = None
            depth_from_lidar = False

            if lidar_depth_map is not None:
                # 优先：从激光雷达深度图中心块取中位数（逐像素，单位已为米）
                z_lidar_px = depth_at_center(lidar_depth_map, u, v, self.center_patch_radius)
                if z_lidar_px > 0:
                    z = float(z_lidar_px)   # lidar_depth_map 单位已是米，scale=1.0
                    depth_from_lidar = True
                else:
                    # 逐像素失败 → 尝试 per-bbox 方法（框内所有投影点中位数）
                    z_lidar_bbox = self._lidar_depth_for_bbox(
                        lidar_points_cam,
                        (x1, y1, x2, y2),
                        self.fx, self.fy, self.cx, self.cy,
                        min_pts=self.lidar_min_pts_in_bbox,
                    )
                    if z_lidar_bbox is not None and z_lidar_bbox > 0:
                        z = z_lidar_bbox
                        depth_from_lidar = True

            if z is None:
                # 最终回退：使用 RGBD 深度图
                depth_center = depth_at_center(depth_mm, u, v, self.center_patch_radius)
                if depth_center <= 0:
                    continue
                z = float(depth_center) * float(self.depth_scale)

            patch_u0, patch_u1, patch_v0, patch_v1 = depth_patch_bounds(
                h=h, w=w, u=u, v=v, radius=self.center_patch_radius
            )

            x_c = (u - self.cx) * z / self.fx
            y_c = (v - self.cy) * z / self.fy

            world_xyz = camera_to_world(pose, x_c, y_c, z)
            class_name = self.model.names.get(cls_id, str(cls_id))

            candidates.append(
                DetectionCandidate(
                    cls_id=cls_id,
                    class_name=class_name,
                    conf=conf,
                    u=u,
                    v=v,
                    bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float64),
                    bbox_xyxy_int=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                    depth_patch_uvxy=(patch_u0, patch_u1, patch_v0, patch_v1),
                    cam_xyz=np.array([x_c, y_c, z], dtype=np.float64),
                    world_xyz_raw=world_xyz,
                )
            )

        matches, unmatched_tracks, unmatched_dets = match_tracks(
            tracks=self._tracks,
            detections=candidates,
            iou_thres=self.track_iou_thres,
        )

        for tid in unmatched_tracks:
            self._tracks[tid].miss_count += 1

        stale_ids = [tid for tid, t in self._tracks.items() if t.miss_count > self.track_max_miss]
        for tid in stale_ids:
            del self._tracks[tid]

        frame_observations: list[FrameObservation] = []

        for tid, det_idx in matches:
            det = candidates[det_idx]
            track = self._tracks[tid]
            track.last_bbox = det.bbox_xyxy
            track.miss_count = 0

            if self.fusion_mode == "moving_average" and track.history.maxlen != self.moving_avg_window:
                existing = list(track.history)
                track.history = deque(existing[-self.moving_avg_window:], maxlen=self.moving_avg_window)

            fused_world = fuse_world_xyz(track, det.world_xyz_raw, self.fusion_mode)
            track.latest_world_raw = det.world_xyz_raw.copy()
            track.latest_world_fused = fused_world.copy()
            frame_observations.append(
                FrameObservation(track_id=tid, det=det, fused_world=fused_world.copy())
            )

        for det_idx in unmatched_dets:
            det = candidates[det_idx]
            tid = self._next_track_id
            self._next_track_id += 1

            track = TrackState(
                track_id=tid,
                cls_id=det.cls_id,
                last_bbox=det.bbox_xyxy,
                history=deque(maxlen=self.moving_avg_window),
            )
            self._tracks[tid] = track

            fused_world = fuse_world_xyz(track, det.world_xyz_raw, self.fusion_mode)
            track.latest_world_raw = det.world_xyz_raw.copy()
            track.latest_world_fused = fused_world.copy()
            frame_observations.append(
                FrameObservation(track_id=tid, det=det, fused_world=fused_world.copy())
            )

        merge_map = build_track_merge_map(
            tracks=self._tracks, merge_distance_thres=self.merge_distance_thres
        )
        apply_track_merges(
            tracks=self._tracks, merge_map=merge_map, moving_avg_window=self.moving_avg_window
        )

        merged_observations: dict[int, FrameObservation] = {}
        for obs in frame_observations:
            root_id = merge_map.get(obs.track_id, obs.track_id)
            root_track = self._tracks.get(root_id)
            fused = obs.fused_world
            if root_track is not None and root_track.latest_world_fused is not None:
                fused = root_track.latest_world_fused.copy()

            new_obs = FrameObservation(track_id=root_id, det=obs.det, fused_world=fused)
            old = merged_observations.get(root_id)
            if old is None or new_obs.det.conf > old.det.conf:
                merged_observations[root_id] = new_obs

        output: list[FrameDetectionResult] = []
        vis = rgb.copy()
        depth_vis = colorize_depth(depth_mm)

        for root_id in sorted(merged_observations.keys()):
            obs = merged_observations[root_id]
            det = obs.det
            fused_world = obs.fused_world

            self._track_points_raw.setdefault(root_id, []).append(det.world_xyz_raw.copy())
            self._track_points_fused.setdefault(root_id, []).append(fused_world.copy())
            self._track_class_names[root_id] = det.class_name
            self._track_confidences.setdefault(root_id, []).append(det.conf)

            vis = draw_detection_on_rgb(vis, det, root_id, fused_world)
            depth_vis = draw_detection_on_depth(depth_vis, det, root_id)

            output.append(
                FrameDetectionResult(
                    track_id=root_id,
                    cls_id=det.cls_id,
                    class_name=det.class_name,
                    confidence=det.conf,
                    u=det.u,
                    v=det.v,
                    bbox_xyxy=(float(det.bbox_xyxy[0]), float(det.bbox_xyxy[1]),
                               float(det.bbox_xyxy[2]), float(det.bbox_xyxy[3])),
                    depth_m=float(det.cam_xyz[2]),
                    cam_xyz=det.cam_xyz.copy(),
                    world_raw=det.world_xyz_raw.copy(),
                    world_fused=fused_world.copy(),
                )
            )

        self._last_rgb_vis = vis
        self._last_depth_vis = depth_vis
        return output

    @property
    def last_rgb_vis(self) -> np.ndarray | None:
        """最近一帧的标注后 RGB 图像。"""
        return self._last_rgb_vis

    @property
    def last_depth_vis(self) -> np.ndarray | None:
        """最近一帧的标注后彩色深度图像。"""
        return self._last_depth_vis

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def active_tracks(self) -> dict[int, TrackState]:
        """当前活跃的跟踪对象（只读引用）。"""
        return self._tracks

    def get_all_track_summaries(self, source: str = "fused") -> dict[int, dict]:
        """
        返回所有 track 的汇总信息。

        Returns
        -------
        dict[int, dict]
            track_id -> {"class_name", "num_points", "mean_conf", "last_position"}
        """
        points = self._track_points_fused if source == "fused" else self._track_points_raw
        summaries: dict[int, dict] = {}
        for tid, pts in points.items():
            confs = self._track_confidences.get(tid, [])
            summaries[tid] = {
                "class_name": self._track_class_names.get(tid, "unknown"),
                "num_points": len(pts),
                "mean_conf": float(np.mean(confs)) if confs else 0.0,
                "last_position": pts[-1].tolist() if pts else None,
            }
        return summaries

    def save_world_plot(
        self,
        out_path: str | Path,
        source: str = "fused",
        min_points: int = 3,
    ) -> None:
        """保存世界坐标系俯视 2D 轨迹图。"""
        points = self._track_points_fused if source == "fused" else self._track_points_raw
        camera_traj = np.stack(self._camera_positions, axis=0) if self._camera_positions else np.empty((0, 3))
        save_world_track_plot(
            track_points=points,
            track_class_names=self._track_class_names,
            track_confidences=self._track_confidences,
            camera_traj_xyz=camera_traj,
            out_path=Path(out_path),
            min_points=max(1, min_points),
        )


# -----------------------------
# Main batch inference loop
# -----------------------------
def _ensure_depth_single_channel(depth: np.ndarray, depth_path: Path) -> np.ndarray | None:
    """
    兼容一些png被读成多通道的情况（例如错误编码）。
    正常应为 HxW 的 uint16 / uint32 / float 等单通道。
    """
    if depth is None:
        return None
    if depth.ndim == 2:
        return depth
    # 某些情况下会读成 HxWx3，尝试取第一个通道
    if depth.ndim == 3 and depth.shape[2] >= 1:
        return depth[:, :, 0]
    print(f"[Skip] 深度图维度异常: {depth.shape} -> {depth_path}")
    return None


def infer(
    model: YOLO,
    frames: Iterable[Path],
    depths: list[Path],
    poses: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
    conf_thres: float,
    class_ids: set[int] | None,
    center_patch_radius: int,
    fusion_mode: str,
    moving_avg_window: int,
    track_iou_thres: float,
    track_max_miss: int,
    save_vis_dir: Path | None,
    csv_out: Path,
    save_world_plot: Path | None,
    world_plot_source: str,
    world_plot_min_points: int,
    merge_distance_thres: float,
) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    if save_vis_dir is not None:
        save_vis_dir.mkdir(parents=True, exist_ok=True)
        vis_rgb_dir = save_vis_dir / "rgb"
        vis_depth_dir = save_vis_dir / "depth"
        vis_rgb_dir.mkdir(parents=True, exist_ok=True)
        vis_depth_dir.mkdir(parents=True, exist_ok=True)
    else:
        vis_rgb_dir = None
        vis_depth_dir = None

    frame_list = list(frames)
    aligned_samples = build_aligned_samples(frame_list, depths, poses.shape[0])
    total = len(aligned_samples)
    if total == 0:
        raise RuntimeError("没有可处理的数据")

    if len(frame_list) != len(depths) or len(frame_list) != poses.shape[0]:
        print(
            f"[Warn] 数量不一致: rgb={len(frame_list)}, depth={len(depths)}, poses={poses.shape[0]}，"
            f"将按最小对齐长度 {total} 处理"
        )
    first_idx = aligned_samples[0][2]
    last_idx = aligned_samples[-1][2]
    print(f"[Info] 轨迹索引对齐范围: {first_idx} -> {last_idx} (samples={total})")

    tracks: dict[int, TrackState] = {}
    next_track_id = 1

    track_points_raw: dict[int, list[np.ndarray]] = {}
    track_points_fused: dict[int, list[np.ndarray]] = {}
    track_class_names: dict[int, str] = {}
    track_confidences: dict[int, list[float]] = {}

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_idx",
                "rgb_name",
                "depth_name",
                "track_id",
                "class_id",
                "class_name",
                "confidence",
                "u",
                "v",
                "depth_m",
                "x_cam",
                "y_cam",
                "z_cam",
                "x_world_raw",
                "y_world_raw",
                "z_world_raw",
                "x_world_fused",
                "y_world_fused",
                "z_world_fused",
                "center_world_x",
                "center_world_y",
                "center_world_z",
            ]
        )

        for i, (rgb_path, depth_path, pose_idx) in enumerate(aligned_samples):
            image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

            if image is None:
                print(f"[Skip] 读取RGB失败: {rgb_path}")
                continue
            depth_mm = _ensure_depth_single_channel(depth_raw, depth_path)
            if depth_mm is None:
                print(f"[Skip] 读取深度失败: {depth_path}")
                continue

            # YOLO推理
            result = model.predict(source=image, conf=conf_thres, verbose=False)[0]
            vis = image
            depth_vis = colorize_depth(depth_mm)
            candidates: list[DetectionCandidate] = []

            # 收集检测候选
            for box in result.boxes:
                cls_id = int(box.cls.item())
                if class_ids is not None and cls_id not in class_ids:
                    continue

                conf = float(box.conf.item())
                if conf < conf_thres:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                u = int(round((x1 + x2) * 0.5))
                v = int(round((y1 + y2) * 0.5))

                h, w = depth_mm.shape
                if u < 0 or u >= w or v < 0 or v >= h:
                    continue

                depth_center = depth_at_center(depth_mm, u, v, center_patch_radius)
                if depth_center <= 0:
                    continue

                patch_u0, patch_u1, patch_v0, patch_v1 = depth_patch_bounds(
                    h=h, w=w, u=u, v=v, radius=center_patch_radius
                )

                z = float(depth_center) * float(depth_scale)
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                world_xyz = camera_to_world(poses[pose_idx], x, y, z)
                class_name = model.names.get(cls_id, str(cls_id))

                candidates.append(
                    DetectionCandidate(
                        cls_id=cls_id,
                        class_name=class_name,
                        conf=conf,
                        u=u,
                        v=v,
                        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float64),
                        bbox_xyxy_int=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                        depth_patch_uvxy=(patch_u0, patch_u1, patch_v0, patch_v1),
                        cam_xyz=np.array([x, y, z], dtype=np.float64),
                        world_xyz_raw=world_xyz,
                    )
                )

            # 跟踪匹配
            matches, unmatched_tracks, unmatched_dets = match_tracks(
                tracks=tracks,
                detections=candidates,
                iou_thres=track_iou_thres,
            )

            # 未匹配track计数+清理
            for tid in unmatched_tracks:
                tracks[tid].miss_count += 1

            stale_ids = [tid for tid, t in tracks.items() if t.miss_count > track_max_miss]
            for tid in stale_ids:
                del tracks[tid]

            frame_observations: list[FrameObservation] = []

            # 更新匹配到的track
            for tid, det_idx in matches:
                det = candidates[det_idx]
                track = tracks[tid]
                track.last_bbox = det.bbox_xyxy
                track.miss_count = 0

                # 若窗口变化，调整deque长度
                if fusion_mode == "moving_average" and track.history.maxlen != moving_avg_window:
                    existing = list(track.history)
                    track.history = deque(existing[-moving_avg_window:], maxlen=moving_avg_window)

                fused_world = fuse_world_xyz(track, det.world_xyz_raw, fusion_mode)
                track.latest_world_raw = det.world_xyz_raw.copy()
                track.latest_world_fused = fused_world.copy()

                frame_observations.append(
                    FrameObservation(track_id=tid, det=det, fused_world=fused_world.copy())
                )

            # 为未匹配检测创建新track
            for det_idx in unmatched_dets:
                det = candidates[det_idx]
                tid = next_track_id
                next_track_id += 1

                track = TrackState(
                    track_id=tid,
                    cls_id=det.cls_id,
                    last_bbox=det.bbox_xyxy,
                    history=deque(maxlen=moving_avg_window),
                )
                tracks[tid] = track

                fused_world = fuse_world_xyz(track, det.world_xyz_raw, fusion_mode)
                track.latest_world_raw = det.world_xyz_raw.copy()
                track.latest_world_fused = fused_world.copy()

                frame_observations.append(
                    FrameObservation(track_id=tid, det=det, fused_world=fused_world.copy())
                )

            # 同类track基于世界坐标合并（可选）
            merge_map = build_track_merge_map(tracks=tracks, merge_distance_thres=merge_distance_thres)
            apply_track_merges(tracks=tracks, merge_map=merge_map, moving_avg_window=moving_avg_window)

            # 合并后：每个root track只输出置信度最高的那条观测
            merged_observations: dict[int, FrameObservation] = {}
            for obs in frame_observations:
                root_id = merge_map.get(obs.track_id, obs.track_id)
                root_track = tracks.get(root_id)
                fused = obs.fused_world
                if root_track is not None and root_track.latest_world_fused is not None:
                    fused = root_track.latest_world_fused.copy()

                new_obs = FrameObservation(track_id=root_id, det=obs.det, fused_world=fused)
                old = merged_observations.get(root_id)
                if old is None or new_obs.det.conf > old.det.conf:
                    merged_observations[root_id] = new_obs

            # 写CSV/保存轨迹/可视化
            for root_id in sorted(merged_observations.keys()):
                obs = merged_observations[root_id]
                det = obs.det
                fused_world = obs.fused_world

                x_cam, y_cam, z_cam = det.cam_xyz.tolist()

                writer.writerow(
                    [
                        i,
                        rgb_path.name,
                        depth_path.name,
                        root_id,
                        det.cls_id,
                        det.class_name,
                        det.conf,
                        det.u,
                        det.v,
                        z_cam,      # depth_m
                        x_cam,
                        y_cam,
                        z_cam,      # z_cam
                        det.world_xyz_raw[0],
                        det.world_xyz_raw[1],
                        det.world_xyz_raw[2],
                        fused_world[0],
                        fused_world[1],
                        fused_world[2],
                        fused_world[0],  # center_world_*（这里与fused保持一致）
                        fused_world[1],
                        fused_world[2],
                    ]
                )

                track_points_raw.setdefault(root_id, []).append(det.world_xyz_raw.copy())
                track_points_fused.setdefault(root_id, []).append(fused_world.copy())
                track_class_names[root_id] = det.class_name
                track_confidences.setdefault(root_id, []).append(det.conf)

                if save_vis_dir is not None:
                    vis = draw_detection_on_rgb(vis, det, root_id, fused_world)
                    depth_vis = draw_detection_on_depth(depth_vis, det, root_id)

            if save_vis_dir is not None:
                cv2.imwrite(str(vis_rgb_dir / rgb_path.name), vis)
                cv2.imwrite(str(vis_depth_dir / depth_path.name), depth_vis)

            if (i + 1) % 50 == 0 or i + 1 == total:
                print(f"Processed {i + 1}/{total}")

    print(f"Done. CSV saved to: {csv_out}")
    if save_vis_dir is not None:
        print(f"Annotated images saved to: {save_vis_dir}")

    if save_world_plot is not None:
        points = track_points_fused if world_plot_source == "fused" else track_points_raw
        camera_traj_xyz = np.stack([poses[pidx, :3, 3] for _, _, pidx in aligned_samples], axis=0)
        save_world_track_plot(
            track_points=points,
            track_class_names=track_class_names,
            track_confidences=track_confidences,
            camera_traj_xyz=camera_traj_xyz,
            out_path=save_world_plot,
            min_points=max(1, world_plot_min_points),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO + RGBD + Pose: 输出目标中心的相机/世界坐标")
    parser.add_argument("--input-dir", type=Path, required=True, help="序列目录，如 /home/joker/Desktop/real/717_12")
    parser.add_argument("--traj", type=Path, default=None, help="traj.txt 路径，默认 <input-dir>/traj.txt")
    parser.add_argument("--results-dir", type=Path, default=None, help="图像目录，默认 <input-dir>/results（否则用input-dir）")

    parser.add_argument("--model", type=str, default="yolov8s-world.pt", help="YOLO/YOLO-World 权重")
    parser.add_argument("--conf", type=float, default=0.4, help="检测置信度阈值")
    parser.add_argument("--vocab", type=str, default=None, help="YOLO-World查询词表：逗号分隔字符串或txt文件路径")
    parser.add_argument("--classes", type=str, default=None, help="类别过滤，逗号分隔，如 person,bottle")

    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--cam-json", type=Path, default=None, help="相机参数JSON（可选）")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="深度单位到米的比例；若不传且cam-json含scale，则自动用1/scale，否则默认0.001",
    )

    parser.add_argument("--center-patch-radius", type=int, default=2, help="中心深度取中值的邻域半径")
    parser.add_argument(
        "--fusion",
        type=str,
        default="moving_average",
        choices=["none", "moving_average", "kalman"],
        help="3D融合方式：none/moving_average/kalman",
    )
    parser.add_argument("--moving-avg-window", type=int, default=5, help="滑动平均窗口")
    parser.add_argument("--track-iou-thres", type=float, default=0.3, help="跟踪匹配IoU阈值")
    parser.add_argument("--track-max-miss", type=int, default=8, help="track最大丢失帧数")
    parser.add_argument(
        "--merge-distance-thres",
        type=float,
        default=0.35,
        help="同类track在世界坐标距离小于该阈值(米)时合并；<=0表示关闭",
    )

    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("./object_centers.csv"),
        help="输出CSV路径",
    )
    parser.add_argument("--save-vis-dir", type=Path, default=None, help="可视化输出目录（可选）")
    parser.add_argument(
        "--save-world-plot",
        type=Path,
        default=None,
        help="保存世界坐标系俯视2D轨迹图（含相机轨迹），如 ./world_tracks_topdown.png",
    )
    parser.add_argument(
        "--world-plot-source",
        type=str,
        default="fused",
        choices=["raw", "fused"],
        help="世界坐标图使用raw或fused坐标",
    )
    parser.add_argument(
        "--world-plot-min-points",
        type=int,
        default=3,
        help="绘图最小轨迹点数，小于该值的track不画",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    traj_path = resolve_traj_path(input_dir=input_dir, explicit_traj=args.traj)
    results_dir = resolve_results_dir(input_dir=input_dir, explicit_results_dir=args.results_dir)

    poses = load_traj(traj_path)
    frames, depths = list_frames(results_dir)

    cam_data = load_cam_json(args.cam_json)
    model = YOLO(args.model)

    vocab_list = parse_vocab_arg(args.vocab)
    apply_model_vocabulary(model, vocab_list)

    fx, fy, cx, cy = load_intrinsics(args.fx, args.fy, args.cx, args.cy, cam_data)
    depth_scale = resolve_depth_scale(args.depth_scale, cam_data)
    class_ids = parse_classes(model.names, args.classes)

    print(f"Using intrinsics: fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}; depth_scale={depth_scale:.9f}")
    print(
        "Tracking/Fusion: "
        f"fusion={args.fusion}, moving_avg_window={args.moving_avg_window}, "
        f"track_iou_thres={args.track_iou_thres}, track_max_miss={args.track_max_miss}, "
        f"merge_distance_thres={args.merge_distance_thres}"
    )

    infer(
        model=model,
        frames=frames,
        depths=depths,
        poses=poses,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        depth_scale=depth_scale,
        conf_thres=args.conf,
        class_ids=class_ids if class_ids is not None and len(class_ids) > 0 else None,
        center_patch_radius=max(0, args.center_patch_radius),
        fusion_mode=args.fusion,
        moving_avg_window=max(1, args.moving_avg_window),
        track_iou_thres=max(0.0, min(1.0, args.track_iou_thres)),
        track_max_miss=max(0, args.track_max_miss),
        save_vis_dir=args.save_vis_dir,
        csv_out=args.csv_out,
        save_world_plot=args.save_world_plot,
        world_plot_source=args.world_plot_source,
        world_plot_min_points=max(1, args.world_plot_min_points),
        merge_distance_thres=max(0.0, args.merge_distance_thres),
    )


if __name__ == "__main__":
    main()