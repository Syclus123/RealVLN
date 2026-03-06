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

启动（实时窗口监控 + 可选深度窗口）：
    python -u yolo_detect_server.py \
        --model yolov8s-world.pt \
        --cam-json cam_params.json \
        --vocab box --conf 0.35 \
        --live-vis --live-vis-depth --live-vis-scale 1.0 \
        --port 5802

浏览器实时看（MJPEG）：
    http://<server_ip>:5802/stream_rgb

保存内容（--output-dir 下）：
    detections.csv          每帧每个track的检测结果
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
import csv
import io
import json
import math
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# 强制行缓冲，确保 conda run / 管道环境下 print 立即可见
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("[Server] 正在加载依赖（numpy, cv2, ultralytics, flask ...）")

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image

print("[Server] 正在加载 StreamProcessor（含 YOLO 模型定义）...")
from stream_vln import StreamProcessor

print("[Server] 依赖加载完成")

app = Flask(__name__)

# ── 全局运行时状态 ──────────────────────────────────────────────
processor: StreamProcessor | None = None
frame_idx: int = 0
start_time: float = time.time()

# 原始 RGB(BGR) 保存
save_raw_rgb: bool = False
_raw_rgb_dir: Path | None = None

# ── 保存模式配置（由 main 写入） ─────────────────────────────────
output_dir: Path | None = None
save_vis: bool = False
save_world_plot_flag: bool = False
world_plot_min_points: int = 3

# 保存模式文件句柄 / 路径
_csv_file = None
_csv_writer = None
_vis_rgb_dir: Path | None = None
_vis_depth_dir: Path | None = None

# ── Live 可视化（窗口 / MJPEG） ──────────────────────────────────
live_vis_enabled: bool = False
live_vis_depth: bool = False
live_vis_scale: float = 1.0

_live_lock = threading.Lock()
_live_event = threading.Event()
_live_stop = threading.Event()

_live_rgb_bgr: np.ndarray | None = None
_live_depth_bgr: np.ndarray | None = None
_live_meta: dict = {}


def _overlay_info(img_bgr: np.ndarray, meta: dict) -> np.ndarray:
    """在可视化图上叠加帧号、检测数、耗时等信息。"""
    if img_bgr is None:
        return img_bgr
    out = img_bgr
    txt1 = f"frame={meta.get('frame_idx', -1)}  det={meta.get('num_det', 0)}"
    txt2 = f"t={meta.get('infer', 0.0):.3f}s"
    cv2.putText(out, txt1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, txt2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def _safe_resize(img_bgr: np.ndarray, scale: float) -> np.ndarray:
    if img_bgr is None or abs(scale - 1.0) < 1e-6:
        return img_bgr
    h, w = img_bgr.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    nh = max(nh, 1)
    nw = max(nw, 1)
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)


def _live_visualizer_worker():
    """后台线程：用 OpenCV 窗口实时显示最新标注帧。"""
    # 没有 display 的环境直接提示并退出（避免报错卡死）
    if os.environ.get("DISPLAY", "") == "":
        print("[LiveVis] 未检测到 DISPLAY（可能是无头环境），窗口实时显示已禁用。可用 /stream_rgb 网页流替代。")
        return

    win_rgb = "YOLO Live (RGB)"
    win_dep = "YOLO Live (Depth)"

    try:
        cv2.namedWindow(win_rgb, cv2.WINDOW_NORMAL)
        if live_vis_depth:
            cv2.namedWindow(win_dep, cv2.WINDOW_NORMAL)
    except Exception as e:
        print(f"[LiveVis] OpenCV 窗口创建失败（可能是 headless cv2）：{e}。可用 /stream_rgb 网页流替代。")
        return

    print("[LiveVis] 窗口实时显示已开启：按 'q' 退出服务；按 'd' 切换深度窗显示。")

    show_depth = live_vis_depth
    while not _live_stop.is_set():
        # 等待新帧（超时为了能响应 stop）
        _live_event.wait(timeout=0.2)
        _live_event.clear()

        with _live_lock:
            rgb = None if _live_rgb_bgr is None else _live_rgb_bgr.copy()
            dep = None if _live_depth_bgr is None else _live_depth_bgr.copy()
            meta = dict(_live_meta)

        if rgb is not None:
            rgb = _overlay_info(rgb, meta)
            rgb = _safe_resize(rgb, live_vis_scale)
            cv2.imshow(win_rgb, rgb)

        if show_depth and dep is not None:
            dep = _safe_resize(dep, live_vis_scale)
            cv2.imshow(win_dep, dep)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[LiveVis] 收到 'q'，发送 SIGINT 退出...")
            os.kill(os.getpid(), signal.SIGINT)
            break
        if key == ord("d"):
            show_depth = not show_depth
            if show_depth and live_vis_depth:
                try:
                    cv2.namedWindow(win_dep, cv2.WINDOW_NORMAL)
                except Exception:
                    pass
            else:
                try:
                    cv2.destroyWindow(win_dep)
                except Exception:
                    pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


# ── 保存模式初始化 ───────────────────────────────────────────────
def _init_save_mode(out_dir: Path, do_vis: bool, do_raw_rgb: bool) -> None:
    global _csv_file, _csv_writer, _vis_rgb_dir, _vis_depth_dir, _raw_rgb_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "detections.csv"
    _csv_file = csv_path.open("w", newline="", encoding="utf-8")
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow(
        [
            "frame_idx",
            "frame_name",
            "track_id",
            "cls_id",
            "class_name",
            "confidence",
            "u",
            "v",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
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
        ]
    )
    _csv_file.flush()
    print(f"[Server] CSV 保存至: {csv_path}")

    if do_vis:
        _vis_rgb_dir = out_dir / "vis" / "rgb"
        _vis_depth_dir = out_dir / "vis" / "depth"
        _vis_rgb_dir.mkdir(parents=True, exist_ok=True)
        _vis_depth_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Server] 可视化图像保存至: {out_dir / 'vis'}")
        
    if do_raw_rgb:
        _raw_rgb_dir = out_dir / "raw" / "rgb"
        _raw_rgb_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Server] 原始RGB保存至: {_raw_rgb_dir}")

def _flush_csv() -> None:
    if _csv_file is not None and not _csv_file.closed:
        _csv_file.flush()


def _close_save_mode() -> None:
    """Server 退出时调用：关闭 CSV，可选生成轨迹图。"""
    if _csv_file is not None and not _csv_file.closed:
        _csv_file.close()
        print("[Server] CSV 已关闭")

    if save_world_plot_flag and processor is not None and output_dir is not None:
        plot_path = output_dir / "world_tracks.png"
        print(f"[Server] 正在生成世界坐标轨迹图: {plot_path}")
        try:
            processor.save_world_plot(str(plot_path), source="fused", min_points=world_plot_min_points)
        except Exception as e:
            print(f"[Server] 轨迹图生成失败: {e}")


def _close_live_vis() -> None:
    """退出时关闭 live 可视化线程和窗口。"""
    _live_stop.set()
    _live_event.set()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


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
        assert processor is not None
        processor.reset()
        frame_idx = 0
        start_time = time.time()
        print("[Server] 跟踪状态已重置")

    frame_name = data.get("frame_name", f"frame{frame_idx:06d}")

    image_file = request.files["image"]
    depth_file = request.files["depth"]

    image = Image.open(image_file.stream).convert("RGB")
    rgb = np.asarray(image)[:, :, ::-1].copy()  # RGB -> BGR
    
    raw_bgr = rgb.copy()  # 备份：保证后面即使被 processor 原地改了也不影响原始保存
    if output_dir is not None and save_raw_rgb and _raw_rgb_dir is not None:
        cv2.imwrite(str(_raw_rgb_dir / f"{frame_name}.jpg"), raw_bgr)

    results = processor.process_frame(rgb, depth, pose, frame_idx=frame_idx)
    
    depth_pil = Image.open(depth_file.stream)
    depth = np.asarray(depth_pil, dtype=np.float64)

    pose_flat = data.get("pose", None)
    if pose_flat is not None and len(pose_flat) == 16:
        pose = np.array(pose_flat, dtype=np.float64).reshape(4, 4)
    else:
        pose = np.eye(4, dtype=np.float64)

    assert processor is not None
    results = processor.process_frame(rgb, depth, pose, frame_idx=frame_idx)

    detections = [r.to_dict() for r in results]
    inference_time = time.time() - t0

    print(f"[Server] frame {frame_idx} ({frame_name}): {len(results)} detections, {inference_time:.3f}s")

    # ── 保存模式写盘 ─────────────────────────────────────────────
    if output_dir is not None:
        # 写 CSV
        if _csv_writer is not None:
            for det in detections:
                bbox = det["bbox_xyxy"]
                cam = det["cam_xyz"]
                wr = det["world_raw"]
                wf = det["world_fused"]
                _csv_writer.writerow(
                    [
                        frame_idx,
                        frame_name,
                        det["track_id"],
                        det["cls_id"],
                        det["class_name"],
                        det["confidence"],
                        det["u"],
                        det["v"],
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3],
                        det["depth_m"],
                        cam[0],
                        cam[1],
                        cam[2],
                        wr[0],
                        wr[1],
                        wr[2],
                        wf[0],
                        wf[1],
                        wf[2],
                    ]
                )
            _flush_csv()

        # 写可视化图像
        if save_vis and _vis_rgb_dir is not None:
            vis_rgb = processor.last_rgb_vis
            vis_dep = processor.last_depth_vis
            if vis_rgb is not None:
                cv2.imwrite(str(_vis_rgb_dir / f"{frame_name}.jpg"), vis_rgb)
            if vis_dep is not None:
                cv2.imwrite(str(_vis_depth_dir / f"{frame_name}.jpg"), vis_dep)

    # ── Live 可视化：把最新标注帧推给窗口线程 / MJPEG ─────────────
    if live_vis_enabled:
        with _live_lock:
            _live_meta.clear()
            _live_meta.update(
                {
                    "frame_idx": frame_idx,
                    "num_det": len(detections),
                    "infer": inference_time,
                    "frame_name": frame_name,
                }
            )
            if processor.last_rgb_vis is not None:
                _live_rgb_bgr = processor.last_rgb_vis
            if processor.last_depth_vis is not None:
                _live_depth_bgr = processor.last_depth_vis
        _live_event.set()

    frame_idx += 1

    return jsonify(
        {
            "frame_idx": frame_idx - 1,
            "frame_name": frame_name,
            "num_detections": len(detections),
            "detections": detections,
            "inference_time": inference_time,
        }
    )


# ── 其他路由 ─────────────────────────────────────────────────────
@app.route("/reset", methods=["POST"])
def reset():
    global frame_idx
    assert processor is not None
    processor.reset()
    frame_idx = 0
    print("[Server] 跟踪状态已重置")
    return jsonify({"status": "ok", "message": "tracking state reset"})


@app.route("/tracks", methods=["GET"])
def tracks():
    source = request.args.get("source", "fused")
    assert processor is not None
    summaries = processor.get_all_track_summaries(source=source)
    return jsonify(
        {
            "frame_count": processor.frame_count,
            "num_tracks": len(summaries),
            "tracks": {str(k): v for k, v in summaries.items()},
        }
    )


@app.route("/save_plot", methods=["POST"])
def save_plot():
    """手动触发保存世界坐标轨迹图（不依赖 --save-world-plot 开关）。"""
    if output_dir is None:
        return jsonify({"error": "output_dir not set, use --output-dir"}), 400
    assert processor is not None
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
    assert processor is not None
    processor.save_world_plot(tmp_path, source=source, min_points=min_pts)
    return send_file(tmp_path, mimetype="image/png")


@app.route("/vis_rgb", methods=["GET"])
def vis_rgb():
    assert processor is not None
    vis = processor.last_rgb_vis
    if vis is None:
        return jsonify({"error": "no frame processed yet"}), 404
    img = Image.fromarray(vis[:, :, ::-1])  # BGR->RGB
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/vis_depth", methods=["GET"])
def vis_depth():
    assert processor is not None
    vis = processor.last_depth_vis
    if vis is None:
        return jsonify({"error": "no frame processed yet"}), 404
    img = Image.fromarray(vis[:, :, ::-1])  # BGR->RGB
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/stream_rgb", methods=["GET"])
def stream_rgb():
    """MJPEG 流：浏览器打开 http://ip:port/stream_rgb 即可实时看标注结果（需 --live-vis 才会持续更新缓存）。"""

    def gen():
        while True:
            with _live_lock:
                frame = None if _live_rgb_bgr is None else _live_rgb_bgr.copy()
                meta = dict(_live_meta)

            if frame is None:
                time.sleep(0.05)
                continue

            frame = _overlay_info(frame, meta)

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                time.sleep(0.02)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            )
            time.sleep(0.02)  # 限速（避免占满 CPU/带宽）

    return app.response_class(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/query", methods=["POST"])
def query():
    """
    按类别名称查询已追踪的目标，返回按距离排序的匹配列表。

    请求 JSON：
        class_name (str)           : 目标类别（如 "box"），大小写不敏感
        robot_x    (float, 可选)   : 机器人当前 x 坐标（世界系），用于排距离
        robot_y    (float, 可选)   : 机器人当前 y 坐标

    返回 JSON：
        {
            "class_name": str,
            "num_matches": int,
            "matches": [
                {
                    "track_id": int,
                    "class_name": str,
                    "num_points": int,
                    "mean_conf": float,
                    "position": [x, y, z],          ← world_fused 均值位置
                    "distance_to_robot": float
                }, ...
            ]  ← 按 distance_to_robot 升序
        }
    """
    data = request.get_json(force=True, silent=True) or {}
    class_name = str(data.get("class_name", "")).lower().strip()
    robot_x = float(data.get("robot_x", 0.0))
    robot_y = float(data.get("robot_y", 0.0))

    if not class_name:
        return jsonify({"error": "class_name is required"}), 400

    assert processor is not None
    summaries = processor.get_all_track_summaries(source="fused")

    matches = []
    for tid, info in summaries.items():
        if info["class_name"].lower() != class_name:
            continue
        pos = info.get("last_position")
        if pos is None:
            continue
        dx = pos[0] - robot_x
        dy = pos[1] - robot_y
        dist = math.sqrt(dx * dx + dy * dy)
        matches.append(
            {
                "track_id": int(tid),
                "class_name": info["class_name"],
                "num_points": info["num_points"],
                "mean_conf": round(info["mean_conf"], 4),
                "position": pos,
                "distance_to_robot": round(dist, 4),
            }
        )

    matches.sort(key=lambda m: m["distance_to_robot"])
    return jsonify({"class_name": class_name, "num_matches": len(matches), "matches": matches})


@app.route("/list_classes", methods=["GET"])
def list_classes():
    """列出当前所有已追踪到的目标类别及其 track 数量。"""
    assert processor is not None
    summaries = processor.get_all_track_summaries(source="fused")
    class_counts: dict[str, int] = {}
    for info in summaries.values():
        cn = info["class_name"]
        class_counts[cn] = class_counts.get(cn, 0) + 1
    return jsonify(
        {
            "frame_count": processor.frame_count,
            "total_tracks": len(summaries),
            "classes": class_counts,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    assert processor is not None
    return jsonify(
        {
            "status": "ok",
            "frame_count": processor.frame_count,
            "output_dir": str(output_dir) if output_dir else None,
            "save_vis": save_vis,
            "save_world_plot": save_world_plot_flag,
            "live_vis": live_vis_enabled,
            "live_vis_depth": live_vis_depth,
            "live_vis_scale": live_vis_scale,
        }
    )


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 目标检测 HTTP 服务端")

    # 模型
    parser.add_argument("--model", type=str, default="yolov8s-world.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--vocab", type=str, default=None)
    parser.add_argument("--classes", type=str, default=None)

    # 相机
    parser.add_argument("--fx", type=float, default=386.5)
    parser.add_argument("--fy", type=float, default=386.5)
    parser.add_argument("--cx", type=float, default=328.9)
    parser.add_argument("--cy", type=float, default=244.0)
    parser.add_argument("--cam-json", type=str, default=None)
    parser.add_argument("--depth-scale", type=float, default=0.0001)

    # 跟踪
    parser.add_argument("--fusion", type=str, default="moving_average", choices=["none", "moving_average", "kalman"])
    parser.add_argument("--moving-avg-window", type=int, default=5)
    parser.add_argument("--track-iou-thres", type=float, default=0.3)
    parser.add_argument("--track-max-miss", type=int, default=8)
    parser.add_argument("--merge-distance-thres", type=float, default=0.35)
    parser.add_argument("--center-patch-radius", type=int, default=2)

    # 服务
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5802)

    parser.add_argument(
        "--save-raw-rgb", action="store_true",
        help="逐帧保存原始RGB图（实际写盘为BGR jpg），需 --output-dir"
    )
    
    # ── 保存模式 ─────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "开启保存模式，指定输出目录（自动用时间戳命名子目录）。"
            "将持续写入 detections.csv；"
            "配合 --save-vis 逐帧保存标注图；"
            "配合 --save-world-plot 在退出时保存轨迹图。"
        ),
    )
    parser.add_argument("--save-vis", action="store_true", help="逐帧保存标注后 RGB 和深度图（需 --output-dir）")
    parser.add_argument("--save-world-plot", action="store_true", help="Ctrl+C 退出时自动保存世界坐标俯视轨迹图（需 --output-dir）")
    parser.add_argument("--world-plot-min-points", type=int, default=3, help="轨迹图最少点数阈值")

    # ── Live 可视化（窗口 / MJPEG） ─────────────────────────────
    parser.add_argument("--live-vis", action="store_true", help="开启 OpenCV 窗口实时显示标注 RGB（需要有 DISPLAY / GUI）")
    parser.add_argument("--live-vis-depth", action="store_true", help="同时允许显示深度可视化窗口（按 d 切换）")
    parser.add_argument("--live-vis-scale", type=float, default=1.0, help="窗口显示缩放比例，例如 0.75 / 1.0 / 1.5")

    args = parser.parse_args()

    # ── 初始化 StreamProcessor ───────────────────────────────────
    vocab_list = None
    if args.vocab:
        vocab_list = [v.strip() for v in args.vocab.split(",") if v.strip()]

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

    # ── 初始化保存模式 ───────────────────────────────────────────
    save_vis = args.save_vis
    save_world_plot_flag = args.save_world_plot
    world_plot_min_points = args.world_plot_min_points
    save_raw_rgb = args.save_raw_rgb
    
    if args.output_dir is not None:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / timestamp
        _init_save_mode(output_dir, save_vis, save_raw_rgb)
    else:
        if save_vis:
            print("[Warn] --save-vis 需要同时指定 --output-dir，已忽略")
        if save_world_plot_flag:
            print("[Warn] --save-world-plot 需要同时指定 --output-dir，已忽略")
            save_world_plot_flag = False
        if args.output_dir is None and save_raw_rgb:
            print("[Warn] --save-raw-rgb 需要同时指定 --output-dir，已忽略")
            save_raw_rgb = False

    # ── Live 可视化开关 ──────────────────────────────────────────
    live_vis_enabled = args.live_vis
    live_vis_depth = args.live_vis_depth
    live_vis_scale = args.live_vis_scale

    if live_vis_enabled:
        t = threading.Thread(target=_live_visualizer_worker, daemon=True)
        t.start()

    # ── 退出时清理（Ctrl+C 或正常退出均触发）────────────────────
    atexit.register(_close_save_mode)
    atexit.register(_close_live_vis)

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
    if live_vis_enabled:
        print("[Server] Live 可视化已开启：窗口（如可用）+ /stream_rgb")

    app.run(host=args.host, port=args.port)