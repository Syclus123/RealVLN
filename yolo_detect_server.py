#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 目标检测 HTTP 服务端。

仿照 http_internvla_server.py 的写法，基于 Flask 接收 RGB/Depth 图像 + Pose，
内部使用 StreamProcessor 进行逐帧流式推理，返回 JSON 检测结果。

启动（基础模式）：
    python -u yolo_detect_server.py \
        --model yolov8s-world.pt \
        --cam-json cam_params.json \
        --vocab box --conf 0.35 \
        --port 5802

启动（保存模式）：
    python -u yolo_detect_server.py \
        --model yolov8s-world.pt \
        --cam-json cam_params.json \
        --vocab box --conf 0.35 \
        --output-dir ./output_realtime \
        --save-vis \
        --save-world-plot \
        --port 5802

保存内容（--output-dir 下）：
    detections.csv          每帧每个track的检测结果
    track_frame_ids.csv     每个track出现过的frame索引与frame名称汇总
    raw/rgb/frameXXXX.jpg   原始RGB图（无标注）
    vis/rgb/frameXXXX.jpg   标注后RGB图（需 --save-vis）
    vis/depth/frameXXXX.jpg 标注后深度图（需 --save-vis）
    world_tracks.png        世界坐标轨迹图（需 --save-world-plot，Ctrl+C 时生成）

客户端每帧 POST /detect：
    files:  image (JPEG)、depth (PNG)
    form:   json = {"reset": false, "pose": [16 floats], "frame_name": "frame0001"}
    返回:   {"detections": [...], "frame_idx": N, "inference_time": float}
"""

import argparse
import atexit
import base64
import csv
import io
import json
import math
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import threading

# 强制行缓冲，确保 conda run / 管道环境下 print 立即可见
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("[Server] 正在加载依赖（numpy, cv2, ultralytics, flask ...）")

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image

print("[Server] 正在加载 StreamProcessor（含 YOLO 模型定义）...")
from stream_vln import StreamProcessor, parse_vocab_arg

print("[Server] 依赖加载完成")

app = Flask(__name__)

# ── 全局运行时状态 ──────────────────────────────────────────────
processor: StreamProcessor | None = None
frame_idx: int = 0
start_time: float = time.time()

# ── 保存模式配置（由 main 写入） ─────────────────────────────────
output_dir: Path | None = None
save_vis: bool = False
save_world_plot_flag: bool = False 
world_plot_min_points: int = 3

# ── Caption：新增 root_id 时选一张图生成描述（可选）
_caption_enabled: bool = False
_caption_client = None  # OpenAI 兼容客户端（如豆包），仅 --enable-caption 时初始化

# 保存模式文件句柄 / 路径
_csv_file = None
_csv_writer = None
_raw_rgb_dir: Path | None = None
_vis_rgb_dir: Path | None = None
_vis_depth_dir: Path | None = None
_latest_objects_lock = threading.Lock()
_latest_objects: list[dict] = []
_latest_frame_name: str | None = None
_latest_frame_idx: int = -1
_track_frames_lock = threading.Lock()
_track_frames: dict[int, set[int]] = {}
_track_frame_names: dict[int, set[str]] = {}


# ── 保存模式初始化 ───────────────────────────────────────────────
def _init_save_mode(out_dir: Path, do_vis: bool) -> None:
    global _csv_file, _csv_writer, _raw_rgb_dir, _vis_rgb_dir, _vis_depth_dir, _track_frames, _track_frame_names

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "detections.csv"
    _csv_file = csv_path.open("w", newline="", encoding="utf-8")
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow([
        "frame_idx", "frame_name",
        "track_id", "cls_id", "class_name", "confidence",
        "u", "v", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "depth_m",
        "x_cam", "y_cam", "z_cam",
        "x_world_raw", "y_world_raw", "z_world_raw",
        "x_world_fused", "y_world_fused", "z_world_fused",
    ])
    _csv_file.flush()
    print(f"[Server] CSV 保存至: {csv_path}")

    with _track_frames_lock:
        _track_frames = {}
        _track_frame_names = {}

    # 原始 RGB 图像目录（无标注）
    _raw_rgb_dir = out_dir / "raw" / "rgb"
    _raw_rgb_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Server] 原始 RGB 图像保存至: {_raw_rgb_dir}")

    if do_vis:
        _vis_rgb_dir = out_dir / "vis" / "rgb"
        _vis_depth_dir = out_dir / "vis" / "depth"
        _vis_rgb_dir.mkdir(parents=True, exist_ok=True)
        _vis_depth_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Server] 可视化图像保存至: {out_dir / 'vis'}")


def _flush_csv() -> None:
    if _csv_file is not None and not _csv_file.closed:
        _csv_file.flush()


def _close_save_mode() -> None:
    """Server 退出时调用：关闭 CSV，可选生成轨迹图。"""
    if output_dir is not None:
        with _track_frames_lock:
            snapshot = {tid: sorted(list(frames)) for tid, frames in _track_frames.items()}
            snapshot_names = {tid: sorted(list(names)) for tid, names in _track_frame_names.items()}
        if snapshot:
            tf_csv_path = output_dir / "track_frame_ids.csv"
            with tf_csv_path.open("w", newline="", encoding="utf-8") as tf_csv:
                writer = csv.writer(tf_csv)
                writer.writerow(["track_id", "num_frames", "frame_indices", "frame_names"])
                for tid in sorted(snapshot.keys()):
                    frames = snapshot[tid]
                    frame_names = snapshot_names.get(tid, [])
                    writer.writerow([
                        tid,
                        len(frames),
                        " ".join(str(x) for x in frames),
                        " ".join(frame_names),
                    ])
            print(f"[Server] Track-Frame 汇总保存至: {tf_csv_path}")

    if _csv_file is not None and not _csv_file.closed:
        _csv_file.close()
        print("[Server] CSV 已关闭")

    if save_world_plot_flag and processor is not None and output_dir is not None:
        plot_path = output_dir / "world_tracks.png"
        print(f"[Server] 正在生成世界坐标轨迹图: {plot_path}")
        try:
            processor.save_world_plot(
                str(plot_path), source="fused", min_points=world_plot_min_points
            )
        except Exception as e:
            print(f"[Server] 轨迹图生成失败: {e}")


# ── Caption 生成（新增 root_id 时在完整帧上只标注该目标的 bbox）───
def _draw_single_bbox(rgb_bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float], class_name: str) -> np.ndarray:
    """在原始 RGB 上只画一个目标的黄色 bbox + 类名标签，返回副本。"""
    canvas = rgb_bgr.copy()
    x1, y1, x2, y2 = int(round(bbox_xyxy[0])), int(round(bbox_xyxy[1])), int(round(bbox_xyxy[2])), int(round(bbox_xyxy[3]))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(canvas, class_name, (max(5, x1), max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return canvas


def _generate_caption_for_root(rgb_bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float], class_name: str) -> str | None:
    """
    在完整帧上只标注该 root 的 bbox，调用多模态 API 生成一句与视角无关的物体描述。
    保留完整场景以提供环境上下文，单一标注确保模型知道描述哪个目标。
    """
    global _caption_client
    if _caption_client is None:
        return None
    annotated = _draw_single_bbox(rgb_bgr, bbox_xyxy, class_name)
    _, buf = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    prompt = (
        "黄色边界框标出的是目标物体。"
        "请用100个字左右描述该物体：类型、主要颜色、形状，以及它与周围物体或环境特征的空间关系（如放在桌上、靠近墙壁、旁边有货架等）。"
        "禁止使用与视角相关的表述（如图片左侧、前景中），"
        "只使用物理空间关系词（如：上方、旁边、靠近）。"
    )
    try:
        resp = _caption_client.chat.completions.create(
            model=os.environ.get("ARK_CAPTION_MODEL", "doubao-seed-2-0-mini-260215"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return text if text else None
    except Exception as e:
        print(f"[Server] Caption API 调用失败: {e}")
        return None


# ── /detect 路由 ─────────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    """
    接收一帧 RGB + Depth + Pose，执行 YOLO 检测 + 跟踪融合，返回结果。

    请求 form 字段 json 支持：
        reset      (bool)          : 是否重置跟踪状态
        pose       (list[float],16): 4x4 相机位姿矩阵 T_wc，行优先
        frame_name (str, 可选)     : 帧文件名，用于保存图像命名
    """
    global frame_idx, start_time

    t0 = time.time()

    json_data = request.form.get("json", "{}")
    data = json.loads(json_data)

    policy_reset = data.get("reset", False)
    if policy_reset:
        processor.reset()
        frame_idx = 0
        start_time = time.time()
        print("[Server] 跟踪状态已重置")

    frame_name = data.get("frame_name", f"frame{frame_idx:06d}")

    image_file = request.files["image"]
    depth_file = request.files["depth"]

    image = Image.open(image_file.stream).convert("RGB")

    # 保存原始 RGB 图像（无标注）
    if output_dir is not None and _raw_rgb_dir is not None:
        raw_path = _raw_rgb_dir / f"{frame_name}.jpg"
        try:
            image.save(raw_path, format="JPEG", quality=95)
        except Exception as e:
            print(f"[Server] 保存原始 RGB 图像失败: {e}")

    rgb = np.asarray(image)[:, :, ::-1].copy()  # RGB -> BGR

    depth_pil = Image.open(depth_file.stream)
    depth = np.asarray(depth_pil, dtype=np.float64)

    pose_flat = data.get("pose", None)
    if pose_flat is not None and len(pose_flat) == 16:
        pose = np.array(pose_flat, dtype=np.float64).reshape(4, 4)
    else:
        pose = np.eye(4, dtype=np.float64)

    results = processor.process_frame(rgb, depth, pose, frame_idx=frame_idx)

    # 流式 caption：记录 root_id 新增/删除，对新增的用本帧该 root 的一张图生成 caption
    current_root_ids = {r.track_id for r in results}
    added_root_ids, removed_root_ids = processor.get_root_id_changes(current_root_ids)
    if added_root_ids and _caption_enabled:
        for root_id in added_root_ids:
            match = next((r for r in results if r.track_id == root_id), None)
            if match is None:
                continue
            caption = _generate_caption_for_root(rgb, match.bbox_xyxy, match.class_name)
            if caption:
                processor.set_caption(root_id, caption)
                short = caption[:100] + "..." if len(caption) > 100 else caption
                print(f"[Server] root_id={root_id} ({match.class_name}) caption: {short}")
    for root_id in removed_root_ids:
        if processor.get_caption(root_id) is not None:
            processor.remove_caption(root_id)
            print(f"[Server] root_id={root_id} 已消失，caption 已清除")

    detections = []
    for r in results:
        d = r.to_dict()
        cap = processor.get_caption(r.track_id)
        if cap is not None:
            d["caption"] = cap
        detections.append(d)
    objects_world = []
    for det in detections:
        wf = det.get("world_fused", None)
        if wf is None or len(wf) != 3:
            continue
        objects_world.append({
            "track_id": int(det.get("track_id", -1)),
            "class_name": str(det.get("class_name", "")),
            "world": [float(wf[0]), float(wf[1]), float(wf[2])],
            "confidence": float(det.get("confidence", 0.0)),
        })

    with _latest_objects_lock:
        global _latest_objects, _latest_frame_name, _latest_frame_idx
        _latest_objects = objects_world
        _latest_frame_name = frame_name
        _latest_frame_idx = frame_idx

    inference_time = time.time() - t0

    # 打印每个检测到物体的类别名称
    for det in detections:
        try:
            cname = det.get("class_name", None)
        except AttributeError:
            cname = None
        if cname is not None:
            print(f"[Server]   object: {cname}")

    print(
        f"[Server] frame {frame_idx} ({frame_name}): "
        f"{len(results)} detections, {inference_time:.3f}s"
    )

    # ── 保存模式写盘 ─────────────────────────────────────────────
    if output_dir is not None:
        # 写 CSV
        if _csv_writer is not None:
            for det in detections:
                bbox = det["bbox_xyxy"]
                cam = det["cam_xyz"]
                wr = det["world_raw"]
                wf = det["world_fused"]
                tid = int(det["track_id"])
                _csv_writer.writerow([
                    frame_idx, frame_name,
                    tid, det["cls_id"], det["class_name"], det["confidence"],
                    det["u"], det["v"],
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    det["depth_m"],
                    cam[0], cam[1], cam[2],
                    wr[0], wr[1], wr[2],
                    wf[0], wf[1], wf[2],
                ])
                with _track_frames_lock:
                    if tid not in _track_frames:
                        _track_frames[tid] = set()
                    _track_frames[tid].add(frame_idx)
                    if tid not in _track_frame_names:
                        _track_frame_names[tid] = set()
                    _track_frame_names[tid].add(frame_name)
            _flush_csv()

        # 写可视化图像
        if save_vis and _vis_rgb_dir is not None:
            vis_rgb = processor.last_rgb_vis
            vis_dep = processor.last_depth_vis
            if vis_rgb is not None:
                cv2.imwrite(str(_vis_rgb_dir / f"{frame_name}.jpg"), vis_rgb)
            if vis_dep is not None:
                cv2.imwrite(str(_vis_depth_dir / f"{frame_name}.jpg"), vis_dep)

    frame_idx += 1

    return jsonify({
        "frame_idx": frame_idx - 1,
        "frame_name": frame_name,
        "num_detections": len(detections),
        "detections": detections,
        "objects_world": objects_world,
        "inference_time": inference_time,
    })


# ── 其他路由 ─────────────────────────────────────────────────────
@app.route("/reset", methods=["POST"])
def reset():
    global frame_idx
    processor.reset()
    frame_idx = 0
    print("[Server] 跟踪状态已重置")
    return jsonify({"status": "ok", "message": "tracking state reset"})


@app.route("/tracks", methods=["GET"])
def tracks():
    source = request.args.get("source", "fused")
    summaries = processor.get_all_track_summaries(source=source)
    captions = processor.get_all_captions()
    # 为每个 track 附带 caption（若有）
    tracks_out = {}
    for k, v in summaries.items():
        out = dict(v)
        if k in captions:
            out["caption"] = captions[k]
        tracks_out[str(k)] = out
    return jsonify({
        "frame_count": processor.frame_count,
        "num_tracks": len(summaries),
        "tracks": tracks_out,
    })


@app.route("/captions", methods=["GET"])
def get_captions():
    """返回所有 root_id -> caption 的映射。"""
    captions = processor.get_all_captions()
    return jsonify({"captions": {str(k): v for k, v in captions.items()}})


@app.route("/save_plot", methods=["POST"])
def save_plot():
    """手动触发保存世界坐标轨迹图（不依赖 --save-world-plot 开关）。"""
    if output_dir is None:
        return jsonify({"error": "output_dir not set, use --output-dir"}), 400
    plot_path = output_dir / "world_tracks.png"
    try:
        processor.save_world_plot(str(plot_path), source="fused", min_points=world_plot_min_points)
        return jsonify({"status": "ok", "path": str(plot_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/world_plot", methods=["GET"])
def world_plot():
    """生成并返回世界坐标系俯视轨迹图 (PNG)，实时下载。"""
    import tempfile
    source = request.args.get("source", "fused")
    min_pts = int(request.args.get("min_points", str(world_plot_min_points)))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    processor.save_world_plot(tmp_path, source=source, min_points=min_pts)
    return send_file(tmp_path, mimetype="image/png")


@app.route("/vis_rgb", methods=["GET"])
def vis_rgb():
    vis = processor.last_rgb_vis
    if vis is None:
        return jsonify({"error": "no frame processed yet"}), 404
    img = Image.fromarray(vis[:, :, ::-1])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/vis_depth", methods=["GET"])
def vis_depth():
    vis = processor.last_depth_vis
    if vis is None:
        return jsonify({"error": "no frame processed yet"}), 404
    img = Image.fromarray(vis[:, :, ::-1])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


def _match_query_by_caption(user_query: str, captions: dict[int, str]) -> list[int]:
    """
    用多模态 API（文本模式）从已有 caption 中找出与用户 query 最匹配的 root_id 列表。
    返回匹配的 root_id（按相关性排序），无匹配返回空列表。
    """
    if not _caption_client or not captions:
        return []

    candidates = "\n".join(f"- root_id={rid}: {cap}" for rid, cap in captions.items())
    prompt = (
        f"用户想要找到的目标：{user_query}\n\n"
        f"以下是当前场景中已追踪到的物体及其描述：\n{candidates}\n\n"
        "请从上面的列表中选出与用户描述最匹配的物体。"
        "只输出匹配物体的 root_id，用逗号分隔，按匹配度从高到低排列。"
        "如果没有任何匹配，输出 NONE。"
        "只输出 root_id 数字，不要输出任何其他内容。"
    )
    try:
        resp = _caption_client.chat.completions.create(
            model=os.environ.get("ARK_CAPTION_MODEL", "doubao-seed-2-0-pro-260215"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        answer = (resp.choices[0].message.content or "").strip()
        if answer.upper() == "NONE" or not answer:
            return []
        result = []
        for token in answer.replace(" ", "").split(","):
            try:
                rid = int(token)
                if rid in captions:
                    result.append(rid)
            except ValueError:
                continue
        return result
    except Exception as e:
        print(f"[Server] Caption 语义匹配失败: {e}")
        return []


@app.route("/query", methods=["POST"])
def query():
    """
    查询已追踪的目标，返回按距离排序的匹配列表。
    优先按类别名精确匹配；无结果时回退到基于 caption 的语义匹配。

    请求 JSON：
        class_name (str)           : 用户查询（类别名或自然语言描述）
        robot_x    (float, 可选)   : 机器人当前 x 坐标（世界系），用于排距离
        robot_y    (float, 可选)   : 机器人当前 y 坐标

    返回 JSON：
        {
            "class_name": str,
            "match_mode": "exact" | "caption",
            "num_matches": int,
            "matches": [
                {
                    "track_id": int,
                    "class_name": str,
                    "caption": str | null,
                    "num_points": int,
                    "mean_conf": float,
                    "position": [x, y, z],
                    "distance_to_robot": float
                }, ...
            ]  ← 按 distance_to_robot 升序
        }
    """
    data = request.get_json(force=True, silent=True) or {}
    user_query = str(data.get("class_name", "")).strip()
    robot_x = float(data.get("robot_x", 0.0))
    robot_y = float(data.get("robot_y", 0.0))

    if not user_query:
        return jsonify({"error": "class_name is required"}), 400

    summaries = processor.get_all_track_summaries(source="fused")
    captions = processor.get_all_captions()

    def _build_match(tid: int, info: dict) -> dict | None:
        pos = info.get("last_position")
        if pos is None:
            return None
        dx = pos[0] - robot_x
        dy = pos[1] - robot_y
        dist = math.sqrt(dx * dx + dy * dy)
        return {
            "track_id": int(tid),
            "class_name": info["class_name"],
            "caption": captions.get(tid),
            "num_points": info["num_points"],
            "mean_conf": round(info["mean_conf"], 4),
            "position": pos,
            "distance_to_robot": round(dist, 4),
        }

    # 1) 精确类别匹配
    matches = []
    query_lower = user_query.lower()
    for tid, info in summaries.items():
        if info["class_name"].lower() == query_lower:
            m = _build_match(tid, info)
            if m:
                matches.append(m)

    if matches:
        matches.sort(key=lambda m: m["distance_to_robot"])
        return jsonify({
            "class_name": user_query,
            "match_mode": "exact",
            "num_matches": len(matches),
            "matches": matches,
        })

    # 2) 基于 caption 的语义匹配（需 --enable-caption 且有 caption 数据）
    if _caption_enabled and captions:
        matched_ids = _match_query_by_caption(user_query, captions)
        for rid in matched_ids:
            info = summaries.get(rid)
            if info is None:
                continue
            m = _build_match(rid, info)
            if m:
                matches.append(m)

    if matches:
        matches.sort(key=lambda m: m["distance_to_robot"])

    return jsonify({
        "class_name": user_query,
        "match_mode": "caption" if matches else "none",
        "num_matches": len(matches),
        "matches": matches,
    })


@app.route("/list_classes", methods=["GET"])
def list_classes():
    """列出当前所有已追踪到的目标类别及其 track 数量。"""
    summaries = processor.get_all_track_summaries(source="fused")
    class_counts: dict[str, int] = {}
    for info in summaries.values():
        cn = info["class_name"]
        class_counts[cn] = class_counts.get(cn, 0) + 1
    return jsonify({
        "frame_count": processor.frame_count,
        "total_tracks": len(summaries),
        "classes": class_counts,
    })


@app.route("/latest_objects", methods=["GET"])
def latest_objects():
    """返回最近一帧检测到的对象名称与世界坐标。"""
    with _latest_objects_lock:
        objs = list(_latest_objects)
        fidx = _latest_frame_idx
        fname = _latest_frame_name

    return jsonify({
        "frame_idx": fidx,
        "frame_name": fname,
        "num_objects": len(objs),
        "objects_world": objs,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "frame_count": processor.frame_count,
        "output_dir": str(output_dir) if output_dir else None,
        "save_vis": save_vis,
        "save_world_plot": save_world_plot_flag,
    })


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 目标检测 HTTP 服务端")

    # 模型
    parser.add_argument("--model", type=str, default="yolov8s-world.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument(
        "--vocab", type=str, default=None,
        help="YOLO-World 词表：逗号分隔字符串、txt 文件路径、或 JSON 文件路径（如 yolo_vocab/obj365v1_class_texts.json）",
    )
    parser.add_argument("--classes", type=str, default=None)

    # 相机
    parser.add_argument("--fx", type=float, default=386.5)
    parser.add_argument("--fy", type=float, default=386.5)
    parser.add_argument("--cx", type=float, default=328.9)
    parser.add_argument("--cy", type=float, default=244.0)
    parser.add_argument("--cam-json", type=str, default=None)
    parser.add_argument("--depth-scale", type=float, default=0.0001)

    # 跟踪
    parser.add_argument("--fusion", type=str, default="moving_average",
                        choices=["none", "moving_average", "kalman"])
    parser.add_argument("--moving-avg-window", type=int, default=5)
    parser.add_argument("--track-iou-thres", type=float, default=0.3)
    parser.add_argument("--track-max-miss", type=int, default=8)
    parser.add_argument("--merge-distance-thres", type=float, default=0.35)
    parser.add_argument("--center-patch-radius", type=int, default=2)

    # 服务
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5802)

    # ── 保存模式 ─────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=(
            "开启保存模式，指定输出目录（自动用时间戳命名子目录）。"
            "将持续写入 detections.csv；"
            "配合 --save-vis 逐帧保存标注图；"
            "配合 --save-world-plot 在退出时保存轨迹图。"
        ),
    )
    parser.add_argument(
        "--save-vis", action="store_true",
        help="逐帧保存标注后 RGB 和深度图（需 --output-dir）",
    )
    parser.add_argument(
        "--save-world-plot", action="store_true",
        help="Ctrl+C 退出时自动保存世界坐标俯视轨迹图（需 --output-dir）",
    )
    parser.add_argument(
        "--world-plot-min-points", type=int, default=3,
        help="轨迹图最少点数阈值",
    )
    parser.add_argument(
        "--enable-caption", action="store_true",
        help="对新增的 root_id 用本帧裁剪图调用多模态 API 生成 caption；需环境变量 ARK_API_KEY（可选 ARK_BASE_URL、ARK_CAPTION_MODEL）",
    )

    args = parser.parse_args()

    # ── 初始化 StreamProcessor ───────────────────────────────────
    vocab_list = parse_vocab_arg(args.vocab)

    processor = StreamProcessor(
        model=args.model,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        depth_scale=args.depth_scale,
        cam_json=args.cam_json,
        conf_thres=args.conf,
        class_filter=args.classes,
        vocab=vocab_list,
        center_patch_radius=args.center_patch_radius,
        fusion_mode=args.fusion,
        moving_avg_window=args.moving_avg_window,
        track_iou_thres=args.track_iou_thres,
        track_max_miss=args.track_max_miss,
        merge_distance_thres=args.merge_distance_thres,
    )

    # ── Caption（新增 root 时选一张图生成描述）────────────────────
    _caption_enabled = args.enable_caption
    if _caption_enabled:
        ark_key = os.environ.get("ARK_API_KEY")
        if ark_key:
            from openai import OpenAI
            _caption_client = OpenAI(
                base_url=os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
                api_key=ark_key,
            )
            print("[Server] Caption 已开启（豆包/方舟多模态 API）")
        else:
            _caption_enabled = False
            print("[Server] --enable-caption 已忽略：未设置环境变量 ARK_API_KEY")

    # ── 初始化保存模式 ───────────────────────────────────────────
    save_vis = args.save_vis
    save_world_plot_flag = args.save_world_plot
    world_plot_min_points = args.world_plot_min_points

    if args.output_dir is not None:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / timestamp
        _init_save_mode(output_dir, save_vis)
    else:
        if save_vis:
            print("[Warn] --save-vis 需要同时指定 --output-dir，已忽略")
        if save_world_plot_flag:
            print("[Warn] --save-world-plot 需要同时指定 --output-dir，已忽略")
            save_world_plot_flag = False

    # ── 退出时清理（Ctrl+C 或正常退出均触发）────────────────────
    atexit.register(_close_save_mode)

    def _sigint_handler(sig, frame):
        print("\n[Server] 收到中断信号，正在保存并退出...")
        sys.exit(0)  # 触发 atexit

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    # ── 模型预热 ─────────────────────────────────────────────────
    print("[Server] 模型预热中...")
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_depth = np.zeros((480, 640), dtype=np.uint16)
    processor.process_frame(dummy_rgb, dummy_depth, np.eye(4))
    processor.reset()
    print("[Server] 预热完成，等待请求")
    if output_dir:
        print(f"[Server] 保存模式已开启，输出目录: {output_dir}")

    app.run(host=args.host, port=args.port)
