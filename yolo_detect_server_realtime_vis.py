#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 目标检测 HTTP 服务端（实时窗口可视化版）。

在 yolo_detect_server.py 基础上新增：
  - 本地窗口实时显示最新检测结果（RGB 可视化，支持拼接深度可视化）
  - 按 q 或 ESC 可关闭窗口显示（不影响 HTTP 服务继续运行）

启动示例：
    python -u yolo_detect_server_realtime_vis.py \
        --model yolov8s-world.pt \
        --cam-json cam_params.json \
        --vocab box --conf 0.35 \
        --port 5802 \
        --show-window \
        --show-depth
"""

import argparse
import atexit
import csv
import glob
import io
import json
import math
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("[ServerVis] 正在加载依赖（numpy, cv2, ultralytics, flask ...）")

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image

print("[ServerVis] 正在加载 StreamProcessor（含 YOLO 模型定义）...")
from stream_vln import StreamProcessor

print("[ServerVis] 依赖加载完成")

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

# 保存模式文件句柄 / 路径
_csv_file = None
_csv_writer = None
_raw_rgb_dir: Path | None = None
_vis_rgb_dir: Path | None = None
_vis_depth_dir: Path | None = None


# ── 实时窗口显示配置 ─────────────────────────────────────────────
_show_window: bool = False
_show_depth: bool = False
_window_name: str = "YOLO Realtime Detection"
_display_fps: float = 20.0

_display_enabled: bool = False
_display_lock = threading.Lock()
_display_latest_frame: np.ndarray | None = None
_display_stop_event = threading.Event()
_display_thread: threading.Thread | None = None


class RealtimeDisplay:
    """后台窗口显示线程：持续渲染最近一帧可视化图像。"""

    def __init__(self, window_name: str, fps: float = 20.0):
        self.window_name = window_name
        self.fps = max(1.0, fps)
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

    def update(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        with self._lock:
            self._latest = frame.copy()

    def _loop(self) -> None:
        period = 1.0 / self.fps
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._enabled = True
            print(f"[ServerVis] 实时窗口已开启: {self.window_name}（按 q/ESC 关闭窗口）")
        except Exception as e:
            self._enabled = False
            print(f"[ServerVis] 无法创建显示窗口，已禁用实时显示: {e}")
            return

        while not self._stopped.is_set():
            t0 = time.time()
            frame = None
            with self._lock:
                if self._latest is not None:
                    frame = self._latest.copy()

            if frame is not None:
                try:
                    cv2.imshow(self.window_name, frame)
                except Exception as e:
                    print(f"[ServerVis] 显示失败，停止实时窗口: {e}")
                    break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[ServerVis] 收到键盘退出，关闭实时窗口（HTTP 服务继续）")
                break

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

        self._enabled = False
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


display: RealtimeDisplay | None = None


def _flatten_vocab_items(obj) -> list[str]:
    """将词表 JSON 展平为字符串列表。支持 list[str] / list[list[str]] / 嵌套 list。"""
    out: list[str] = []
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            out.append(s)
        return out
    if isinstance(obj, list):
        for item in obj:
            out.extend(_flatten_vocab_items(item))
    return out


def load_vocab_from_files(vocab_file_args: list[str] | None) -> list[str]:
    """
    从 --vocab-file 加载词表。
    支持：
      - 单文件：/path/coco_class_texts.json
      - 通配符：/path/*_class_texts.json
      - 多个参数：--vocab-file a.json b.json
    """
    if not vocab_file_args:
        return []

    all_paths: list[str] = []
    for pattern in vocab_file_args:
        expanded = glob.glob(str(Path(pattern).expanduser()))
        if expanded:
            all_paths.extend(sorted(expanded))
        else:
            all_paths.append(str(Path(pattern).expanduser()))

    vocab_items: list[str] = []
    for path_str in all_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"词表文件不存在: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        items = _flatten_vocab_items(data)
        if not items:
            print(f"[ServerVis][Warn] 词表文件为空或格式不支持: {path}")
            continue
        vocab_items.extend(items)
        print(f"[ServerVis] 已加载词表文件: {path}  ({len(items)} items)")

    deduped = list(dict.fromkeys(vocab_items))
    return deduped


def _build_display_frame(rgb_vis: np.ndarray | None, depth_vis: np.ndarray | None) -> np.ndarray | None:
    if rgb_vis is None:
        return None

    frame = rgb_vis
    if _show_depth and depth_vis is not None:
        if depth_vis.shape[:2] != rgb_vis.shape[:2]:
            depth_vis = cv2.resize(depth_vis, (rgb_vis.shape[1], rgb_vis.shape[0]))
        frame = np.hstack([rgb_vis, depth_vis])

    stamp = time.strftime("%H:%M:%S")
    cv2.putText(
        frame,
        f"{stamp}  frame={frame_idx}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


# ── 保存模式初始化 ───────────────────────────────────────────────
def _init_save_mode(out_dir: Path, do_vis: bool) -> None:
    global _csv_file, _csv_writer, _raw_rgb_dir, _vis_rgb_dir, _vis_depth_dir

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
    print(f"[ServerVis] CSV 保存至: {csv_path}")

    _raw_rgb_dir = out_dir / "raw" / "rgb"
    _raw_rgb_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ServerVis] 原始 RGB 图像保存至: {_raw_rgb_dir}")

    if do_vis:
        _vis_rgb_dir = out_dir / "vis" / "rgb"
        _vis_depth_dir = out_dir / "vis" / "depth"
        _vis_rgb_dir.mkdir(parents=True, exist_ok=True)
        _vis_depth_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ServerVis] 可视化图像保存至: {out_dir / 'vis'}")


def _flush_csv() -> None:
    if _csv_file is not None and not _csv_file.closed:
        _csv_file.flush()


def _close_save_mode() -> None:
    if _csv_file is not None and not _csv_file.closed:
        _csv_file.close()
        print("[ServerVis] CSV 已关闭")

    if save_world_plot_flag and processor is not None and output_dir is not None:
        plot_path = output_dir / "world_tracks.png"
        print(f"[ServerVis] 正在生成世界坐标轨迹图: {plot_path}")
        try:
            processor.save_world_plot(
                str(plot_path), source="fused", min_points=world_plot_min_points
            )
        except Exception as e:
            print(f"[ServerVis] 轨迹图生成失败: {e}")


# ── /detect 路由 ─────────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    global frame_idx, start_time

    t0 = time.time()

    json_data = request.form.get("json", "{}")
    data = json.loads(json_data)

    policy_reset = data.get("reset", False)
    if policy_reset:
        processor.reset()
        frame_idx = 0
        start_time = time.time()
        print("[ServerVis] 跟踪状态已重置")

    frame_name = data.get("frame_name", f"frame{frame_idx:06d}")

    image_file = request.files["image"]
    depth_file = request.files["depth"]

    image = Image.open(image_file.stream).convert("RGB")

    if output_dir is not None and _raw_rgb_dir is not None:
        raw_path = _raw_rgb_dir / f"{frame_name}.jpg"
        try:
            image.save(raw_path, format="JPEG", quality=95)
        except Exception as e:
            print(f"[ServerVis] 保存原始 RGB 图像失败: {e}")

    rgb = np.asarray(image)[:, :, ::-1].copy()

    depth_pil = Image.open(depth_file.stream)
    depth = np.asarray(depth_pil, dtype=np.float64)

    pose_flat = data.get("pose", None)
    if pose_flat is not None and len(pose_flat) == 16:
        pose = np.array(pose_flat, dtype=np.float64).reshape(4, 4)
    else:
        pose = np.eye(4, dtype=np.float64)

    results = processor.process_frame(rgb, depth, pose, frame_idx=frame_idx)

    detections = [r.to_dict() for r in results]
    inference_time = time.time() - t0

    for det in detections:
        cname = det.get("class_name", None)
        if cname is not None:
            print(f"[ServerVis]   object: {cname}")

    print(
        f"[ServerVis] frame {frame_idx} ({frame_name}): "
        f"{len(results)} detections, {inference_time:.3f}s"
    )

    # 实时窗口帧更新
    if display is not None and display.enabled:
        vis_rgb = processor.last_rgb_vis
        vis_depth = processor.last_depth_vis
        vis_frame = _build_display_frame(vis_rgb, vis_depth)
        display.update(vis_frame)

    if output_dir is not None:
        if _csv_writer is not None:
            for det in detections:
                bbox = det["bbox_xyxy"]
                cam = det["cam_xyz"]
                wr = det["world_raw"]
                wf = det["world_fused"]
                _csv_writer.writerow([
                    frame_idx, frame_name,
                    det["track_id"], det["cls_id"], det["class_name"], det["confidence"],
                    det["u"], det["v"],
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    det["depth_m"],
                    cam[0], cam[1], cam[2],
                    wr[0], wr[1], wr[2],
                    wf[0], wf[1], wf[2],
                ])
            _flush_csv()

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
        "inference_time": inference_time,
    })


@app.route("/reset", methods=["POST"])
def reset():
    global frame_idx
    processor.reset()
    frame_idx = 0
    print("[ServerVis] 跟踪状态已重置")
    return jsonify({"status": "ok", "message": "tracking state reset"})


@app.route("/tracks", methods=["GET"])
def tracks():
    source = request.args.get("source", "fused")
    summaries = processor.get_all_track_summaries(source=source)
    return jsonify({
        "frame_count": processor.frame_count,
        "num_tracks": len(summaries),
        "tracks": {str(k): v for k, v in summaries.items()},
    })


@app.route("/save_plot", methods=["POST"])
def save_plot():
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


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True, silent=True) or {}
    class_name = str(data.get("class_name", "")).lower().strip()
    robot_x = float(data.get("robot_x", 0.0))
    robot_y = float(data.get("robot_y", 0.0))

    if not class_name:
        return jsonify({"error": "class_name is required"}), 400

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
        matches.append({
            "track_id": int(tid),
            "class_name": info["class_name"],
            "num_points": info["num_points"],
            "mean_conf": round(info["mean_conf"], 4),
            "position": pos,
            "distance_to_robot": round(dist, 4),
        })

    matches.sort(key=lambda m: m["distance_to_robot"])
    return jsonify({
        "class_name": class_name,
        "num_matches": len(matches),
        "matches": matches,
    })


@app.route("/list_classes", methods=["GET"])
def list_classes():
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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "frame_count": processor.frame_count,
        "output_dir": str(output_dir) if output_dir else None,
        "save_vis": save_vis,
        "save_world_plot": save_world_plot_flag,
        "show_window": _show_window,
        "show_depth": _show_depth,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 目标检测 HTTP 服务端（实时窗口可视化版）")

    parser.add_argument("--model", type=str, default="yolov8s-world.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--vocab", type=str, default=None)
    parser.add_argument(
        "--vocab-file",
        type=str,
        nargs="+",
        default=None,
        help=(
            "词表 JSON 文件路径（支持多个或通配符），如: "
            "--vocab-file ./coco_class_texts.json "
            "或 --vocab-file '/data/*_class_texts.json'"
        ),
    )
    parser.add_argument("--classes", type=str, default=None)

    parser.add_argument("--fx", type=float, default=386.5)
    parser.add_argument("--fy", type=float, default=386.5)
    parser.add_argument("--cx", type=float, default=328.9)
    parser.add_argument("--cy", type=float, default=244.0)
    parser.add_argument("--cam-json", type=str, default=None)
    parser.add_argument("--depth-scale", type=float, default=0.0001)

    parser.add_argument("--fusion", type=str, default="moving_average",
                        choices=["none", "moving_average", "kalman"])
    parser.add_argument("--moving-avg-window", type=int, default=5)
    parser.add_argument("--track-iou-thres", type=float, default=0.3)
    parser.add_argument("--track-max-miss", type=int, default=8)
    parser.add_argument("--merge-distance-thres", type=float, default=0.35)
    parser.add_argument("--center-patch-radius", type=int, default=2)

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5802)

    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="开启保存模式，指定输出目录（自动用时间戳命名子目录）",
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
        "--show-window", action="store_true",
        help="开启本地实时窗口可视化（需要图形环境）",
    )
    parser.add_argument(
        "--show-depth", action="store_true",
        help="窗口中同时显示深度可视化（与 RGB 横向拼接）",
    )
    parser.add_argument(
        "--window-name", type=str, default="YOLO Realtime Detection",
        help="实时显示窗口名称",
    )
    parser.add_argument(
        "--display-fps", type=float, default=20.0,
        help="窗口刷新帧率上限（默认 20）",
    )

    args = parser.parse_args()

    _show_window = args.show_window
    _show_depth = args.show_depth
    _window_name = args.window_name
    _display_fps = args.display_fps

    vocab_list: list[str] = []
    if args.vocab:
        vocab_list.extend([v.strip() for v in args.vocab.split(",") if v.strip()])
    if args.vocab_file:
        vocab_list.extend(load_vocab_from_files(args.vocab_file))
    if vocab_list:
        vocab_list = list(dict.fromkeys(vocab_list))
        print(f"[ServerVis] 词表总数: {len(vocab_list)}")
    else:
        vocab_list = None

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

    if _show_window:
        display = RealtimeDisplay(window_name=_window_name, fps=_display_fps)
        display.start()
    else:
        print("[ServerVis] 未开启实时窗口（可加 --show-window）")

    def _cleanup_all() -> None:
        _close_save_mode()
        if display is not None:
            display.stop()

    atexit.register(_cleanup_all)

    def _sigint_handler(sig, frame):
        print("\n[ServerVis] 收到中断信号，正在保存并退出...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    print("[ServerVis] 模型预热中...")
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_depth = np.zeros((480, 640), dtype=np.uint16)
    processor.process_frame(dummy_rgb, dummy_depth, np.eye(4))
    processor.reset()
    print("[ServerVis] 预热完成，等待请求")
    if output_dir:
        print(f"[ServerVis] 保存模式已开启，输出目录: {output_dir}")

    app.run(host=args.host, port=args.port)
