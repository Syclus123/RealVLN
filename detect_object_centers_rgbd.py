#!/usr/bin/env python3
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
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


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
    with cam_json.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_traj(traj_path: Path) -> np.ndarray:
    poses = []
    with traj_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
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

    if isinstance(data, dict) and "K" in data and isinstance(data["K"], (list, tuple)) and len(data["K"]) >= 9:
        k = data["K"]
        return float(k[0]), float(k[4]), float(k[2]), float(k[5])

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
    return valid_ids


def parse_vocab_arg(vocab_arg: str | None) -> list[str] | None:
    if not vocab_arg:
        return None

    path = Path(vocab_arg)
    if path.exists() and path.is_file():
        tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
    match = re.match(rf"^{prefix}(\d+)\.[^.]+$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def build_aligned_samples(
    frames: list[Path],
    depths: list[Path],
    pose_count: int,
) -> list[tuple[Path, Path, int]]:
    frame_indexed: list[tuple[int, Path]] = []
    for frame_path in frames:
        idx = parse_frame_index(frame_path, "frame")
        if idx is not None:
            frame_indexed.append((idx, frame_path))

    depth_by_idx: dict[int, Path] = {}
    for depth_path in depths:
        idx = parse_frame_index(depth_path, "depth")
        if idx is not None:
            depth_by_idx[idx] = depth_path

    samples: list[tuple[Path, Path, int]] = []
    if frame_indexed and depth_by_idx:
        for idx, frame_path in sorted(frame_indexed, key=lambda x: x[0]):
            depth_path = depth_by_idx.get(idx)
            if depth_path is None:
                continue
            if idx < 0 or idx >= pose_count:
                continue
            samples.append((frame_path, depth_path, idx))

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

    sampled_match = re.match(r"^(.*)_sampled_i\d+$", input_dir.name)
    if sampled_match:
        base_name = sampled_match.group(1)
        fallback = input_dir.parent / base_name / "traj.txt"
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

    return input_dir


def depth_at_center(depth_mm: np.ndarray, u: int, v: int, radius: int) -> float:
    h, w = depth_mm.shape
    u0, u1, v0, v1 = depth_patch_bounds(h=h, w=w, u=u, v=v, radius=radius)
    patch = depth_mm[v0:v1, u0:u1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def depth_patch_bounds(h: int, w: int, u: int, v: int, radius: int) -> tuple[int, int, int, int]:
    u0 = max(0, u - radius)
    u1 = min(w, u + radius + 1)
    v0 = max(0, v - radius)
    v1 = min(h, v + radius + 1)
    return u0, u1, v0, v1


def camera_to_world(T_wc: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
    # 光学相机坐标系(右,下,前) -> 里程计使用坐标系(前,左,上)
    x_fwd = z
    y_left = -x
    z_up = -y
    p_c = np.array([x_fwd, y_left, z_up, 1.0], dtype=np.float64)
    p_w = T_wc @ p_c
    return p_w[:3]


def draw_detection(
    image: np.ndarray,
    label: str,
    conf: float,
    u: int,
    v: int,
    world_xyz: np.ndarray,
    track_id: int,
) -> np.ndarray:
    canvas = image.copy()
    cv2.circle(canvas, (u, v), 4, (0, 0, 255), -1)
    text = (
        f"id={track_id} {label} {conf:.2f} | "
        f"W=({world_xyz[0]:.2f},{world_xyz[1]:.2f},{world_xyz[2]:.2f})"
    )
    cv2.putText(canvas, text, (max(5, u - 120), max(20, v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return canvas


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
    cv2.putText(canvas, text, (max(5, det.u - 120), max(20, det.v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return canvas


def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        gray = np.zeros_like(depth_mm, dtype=np.uint8)
    else:
        max_v = float(np.percentile(valid, 98))
        max_v = max(max_v, 1.0)
        gray = np.clip(depth_mm.astype(np.float32) / max_v * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def draw_detection_on_depth(
    depth_color: np.ndarray,
    det: DetectionCandidate,
    track_id: int,
) -> np.ndarray:
    canvas = depth_color.copy()
    x1, y1, x2, y2 = det.bbox_xyxy_int
    u0, u1, v0, v1 = det.depth_patch_uvxy
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 仅用于显示：当采样框太小时放大显示，便于肉眼观察（不影响实际深度计算）
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

    # 黑色描边 + 黄色内框，提升在伪彩深度图上的可见性
    cv2.rectangle(canvas, (su0, sv0), (max(su0, su1 - 1), max(sv0, sv1 - 1)), (0, 0, 0), 3)
    cv2.rectangle(canvas, (su0, sv0), (max(su0, su1 - 1), max(sv0, sv1 - 1)), (0, 255, 255), 2)
    cv2.circle(canvas, (det.u, det.v), 4, (0, 0, 255), -1)
    cv2.putText(canvas, f"id={track_id} {det.class_name}", (max(5, det.u - 100), max(20, det.v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return canvas


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
    if merge_distance_thres <= 0 or len(tracks) <= 1:
        return {track_id: track_id for track_id in tracks.keys()}

    parent: dict[int, int] = {track_id: track_id for track_id in tracks.keys()}

    def find(track_id: int) -> int:
        root = track_id
        while parent[root] != root:
            root = parent[root]
        while parent[track_id] != track_id:
            nxt = parent[track_id]
            parent[track_id] = root
            track_id = nxt
        return root

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    valid_track_ids = [
        track_id
        for track_id, track in tracks.items()
        if track.latest_world_fused is not None
    ]

    for i in range(len(valid_track_ids)):
        tid_i = valid_track_ids[i]
        track_i = tracks[tid_i]
        pos_i = track_i.latest_world_fused
        if pos_i is None:
            continue
        for j in range(i + 1, len(valid_track_ids)):
            tid_j = valid_track_ids[j]
            track_j = tracks[tid_j]
            if track_i.cls_id != track_j.cls_id:
                continue
            pos_j = track_j.latest_world_fused
            if pos_j is None:
                continue
            dist = float(np.linalg.norm(pos_i - pos_j))
            if dist <= merge_distance_thres:
                union(tid_i, tid_j)

    return {track_id: find(track_id) for track_id in tracks.keys()}


def apply_track_merges(
    tracks: dict[int, TrackState],
    merge_map: dict[int, int],
    moving_avg_window: int,
) -> None:
    grouped: dict[int, list[int]] = {}
    for track_id, root_id in merge_map.items():
        grouped.setdefault(root_id, []).append(track_id)

    for root_id, members in grouped.items():
        if root_id not in tracks:
            continue
        root_track = tracks[root_id]
        for member_id in members:
            if member_id == root_id or member_id not in tracks:
                continue
            member_track = tracks[member_id]

            merged_hist = list(root_track.history) + list(member_track.history)
            if merged_hist:
                root_track.history = deque(merged_hist[-moving_avg_window:], maxlen=moving_avg_window)

            if root_track.latest_world_raw is None:
                root_track.latest_world_raw = member_track.latest_world_raw
            elif member_track.latest_world_raw is not None:
                root_track.latest_world_raw = 0.5 * (root_track.latest_world_raw + member_track.latest_world_raw)

            if root_track.latest_world_fused is None:
                root_track.latest_world_fused = member_track.latest_world_fused
            elif member_track.latest_world_fused is not None:
                root_track.latest_world_fused = 0.5 * (root_track.latest_world_fused + member_track.latest_world_fused)

            if root_track.kf_x is None:
                root_track.kf_x = member_track.kf_x
            elif member_track.kf_x is not None:
                root_track.kf_x = 0.5 * (root_track.kf_x + member_track.kf_x)

            if root_track.kf_P is None:
                root_track.kf_P = member_track.kf_P
            elif member_track.kf_P is not None:
                root_track.kf_P = 0.5 * (root_track.kf_P + member_track.kf_P)

            root_track.miss_count = min(root_track.miss_count, member_track.miss_count)
            del tracks[member_id]


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
    for track_id, points_list in track_points.items():
        if len(points_list) < min_points:
            continue
        valid_items.append((track_id, np.stack(points_list, axis=0)))

    if not valid_items:
        print(f"[Warn] 无可绘制轨迹（min_points={min_points}），跳过世界坐标图导出")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    all_xy = []
    for track_id, points in valid_items:
        conf_list = track_confidences.get(track_id, [])
        if conf_list:
            conf_avg = float(np.mean(conf_list))
            conf_last = float(conf_list[-1])
            label = f"id={track_id}:{track_class_names.get(track_id, 'obj')} conf={conf_avg:.2f}/{conf_last:.2f}"
        else:
            label = f"id={track_id}:{track_class_names.get(track_id, 'obj')} conf=NA"
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
            f"[Warn] 数量不一致: rgb={len(frame_list)}, depth={len(depths)}, poses={poses.shape[0]}，将按最小长度 {total} 处理"
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

    with csv_out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
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
            depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"[Skip] 读取RGB失败: {rgb_path}")
                continue
            if depth_mm is None:
                print(f"[Skip] 读取深度失败: {depth_path}")
                continue
            if depth_mm.ndim != 2:
                print(f"[Skip] 深度图不是单通道: {depth_path}")
                continue

            result = model.predict(source=image, conf=conf_thres, verbose=False)[0]
            vis = image
            depth_vis = colorize_depth(depth_mm)
            candidates: list[DetectionCandidate] = []

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

                depth_mm_center = depth_at_center(depth_mm, u, v, center_patch_radius)
                if depth_mm_center <= 0:
                    continue
                patch_u0, patch_u1, patch_v0, patch_v1 = depth_patch_bounds(
                    h=h, w=w, u=u, v=v, radius=center_patch_radius
                )

                z = depth_mm_center * depth_scale
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
                        bbox_xyxy_int=(
                            int(round(x1)),
                            int(round(y1)),
                            int(round(x2)),
                            int(round(y2)),
                        ),
                        depth_patch_uvxy=(patch_u0, patch_u1, patch_v0, patch_v1),
                        cam_xyz=np.array([x, y, z], dtype=np.float64),
                        world_xyz_raw=world_xyz,
                    )
                )

            matches, unmatched_tracks, unmatched_dets = match_tracks(
                tracks=tracks,
                detections=candidates,
                iou_thres=track_iou_thres,
            )

            for track_id in unmatched_tracks:
                tracks[track_id].miss_count += 1

            stale_track_ids = [
                track_id
                for track_id, track in tracks.items()
                if track.miss_count > track_max_miss
            ]
            for track_id in stale_track_ids:
                del tracks[track_id]

            frame_observations: list[FrameObservation] = []

            for track_id, det_idx in matches:
                det = candidates[det_idx]
                track = tracks[track_id]
                track.last_bbox = det.bbox_xyxy
                track.miss_count = 0
                if fusion_mode == "moving_average" and track.history.maxlen != moving_avg_window:
                    existing = list(track.history)
                    track.history = deque(existing[-moving_avg_window:], maxlen=moving_avg_window)

                fused_world = fuse_world_xyz(track, det.world_xyz_raw, fusion_mode)
                track.latest_world_raw = det.world_xyz_raw.copy()
                track.latest_world_fused = fused_world.copy()
                frame_observations.append(
                    FrameObservation(
                        track_id=track_id,
                        det=det,
                        fused_world=fused_world.copy(),
                    )
                )

            for det_idx in unmatched_dets:
                det = candidates[det_idx]
                track_id = next_track_id
                next_track_id += 1

                history = deque(maxlen=moving_avg_window)
                track = TrackState(
                    track_id=track_id,
                    cls_id=det.cls_id,
                    last_bbox=det.bbox_xyxy,
                    history=history,
                )
                tracks[track_id] = track

                fused_world = fuse_world_xyz(track, det.world_xyz_raw, fusion_mode)

                track.latest_world_raw = det.world_xyz_raw.copy()
                track.latest_world_fused = fused_world.copy()
                frame_observations.append(
                    FrameObservation(
                        track_id=track_id,
                        det=det,
                        fused_world=fused_world.copy(),
                    )
                )

            merge_map = build_track_merge_map(
                tracks=tracks,
                merge_distance_thres=merge_distance_thres,
            )
            apply_track_merges(
                tracks=tracks,
                merge_map=merge_map,
                moving_avg_window=moving_avg_window,
            )

            merged_observations: dict[int, FrameObservation] = {}
            for observation in frame_observations:
                merged_track_id = merge_map.get(observation.track_id, observation.track_id)
                merged_track = tracks.get(merged_track_id)
                merged_fused = observation.fused_world
                if merged_track is not None and merged_track.latest_world_fused is not None:
                    merged_fused = merged_track.latest_world_fused.copy()

                existing = merged_observations.get(merged_track_id)
                new_observation = FrameObservation(
                    track_id=merged_track_id,
                    det=observation.det,
                    fused_world=merged_fused,
                )
                if existing is None or new_observation.det.conf > existing.det.conf:
                    merged_observations[merged_track_id] = new_observation

            for merged_track_id in sorted(merged_observations.keys()):
                observation = merged_observations[merged_track_id]
                det = observation.det
                fused_world = observation.fused_world

                x_cam, y_cam, z_cam = det.cam_xyz.tolist()

                writer.writerow(
                    [
                        i,
                        rgb_path.name,
                        depth_path.name,
                        merged_track_id,
                        det.cls_id,
                        det.class_name,
                        det.conf,
                        det.u,
                        det.v,
                        z_cam,
                        x_cam,
                        y_cam,
                        z_cam,
                        det.world_xyz_raw[0],
                        det.world_xyz_raw[1],
                        det.world_xyz_raw[2],
                        fused_world[0],
                        fused_world[1],
                        fused_world[2],
                        fused_world[0],
                        fused_world[1],
                        fused_world[2],
                    ]
                )

                track_points_raw.setdefault(merged_track_id, []).append(det.world_xyz_raw.copy())
                track_points_fused.setdefault(merged_track_id, []).append(fused_world.copy())
                track_class_names[merged_track_id] = det.class_name
                track_confidences.setdefault(merged_track_id, []).append(det.conf)

                if save_vis_dir is not None:
                    vis = draw_detection_on_rgb(vis, det, merged_track_id, fused_world)
                    depth_vis = draw_detection_on_depth(depth_vis, det, merged_track_id)

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
        camera_traj_xyz = np.stack([poses[pose_idx, :3, 3] for _, _, pose_idx in aligned_samples], axis=0)
        save_world_track_plot(
            track_points=points,
            track_class_names=track_class_names,
            track_confidences=track_confidences,
            camera_traj_xyz=camera_traj_xyz,
            out_path=save_world_plot,
            min_points=max(1, world_plot_min_points),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO + RGBD + Pose: 输出目标中心的相机/世界坐标"
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="序列目录，如 /home/joker/Desktop/real/717_12")
    parser.add_argument("--traj", type=Path, default=None, help="traj.txt 路径，默认 <input-dir>/traj.txt")
    parser.add_argument("--results-dir", type=Path, default=None, help="图像目录，默认 <input-dir>/results")

    parser.add_argument("--model", type=str, default="yolov8s-world.pt", help="YOLO/YOLO-World 权重，如 yolov8s-world.pt")
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
        default=Path("/home/joker/Desktop/yolo_rgbd_object_center/object_centers.csv"),
        help="输出CSV",
    )
    parser.add_argument("--save-vis-dir", type=Path, default=None, help="可视化输出目录（可选）")
    parser.add_argument(
        "--save-world-plot",
        type=Path,
        default=None,
        help="保存世界坐标系俯视2D轨迹图（含相机轨迹），如 /path/world_tracks_topdown.png",
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

    print(
        f"Using intrinsics: fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}; depth_scale={depth_scale:.9f}"
    )
    print(
        f"Tracking/Fusion: fusion={args.fusion}, moving_avg_window={args.moving_avg_window}, track_iou_thres={args.track_iou_thres}, track_max_miss={args.track_max_miss}, merge_distance_thres={args.merge_distance_thres}"
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
        class_ids=class_ids,
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
