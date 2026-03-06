#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass
class ObjectAnchor:
    class_name: str
    track_id: int
    x: float
    y: float
    z: float


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


class SemanticGoalNode(Node):
    def __init__(
        self,
        object_csv_default: str,
        cam_offset_x_default: float,
        cam_offset_y_default: float,
        cam_offset_z_default: float,
        goal_object_standoff_default: float,
    ) -> None:
        super().__init__("semantic_goal_node")

        self.declare_parameter("object_csv", object_csv_default)
        self.declare_parameter("target_topic", "/semantic_target")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_link_frame", "base_link")
        self.declare_parameter("tf_timeout_sec", 0.2)
        self.declare_parameter("z_offset", 0.0)
        self.declare_parameter("cam_offset_x", cam_offset_x_default)
        self.declare_parameter("cam_offset_y", cam_offset_y_default)
        self.declare_parameter("cam_offset_z", cam_offset_z_default)
        self.declare_parameter("goal_object_standoff", goal_object_standoff_default)

        csv_path = Path(str(self.get_parameter("object_csv").value))
        target_topic = str(self.get_parameter("target_topic").value)
        goal_topic = str(self.get_parameter("goal_topic").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_link_frame = str(self.get_parameter("base_link_frame").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.z_offset = float(self.get_parameter("z_offset").value)
        self.cam_offset_x = float(self.get_parameter("cam_offset_x").value)
        self.cam_offset_y = float(self.get_parameter("cam_offset_y").value)
        self.cam_offset_z = float(self.get_parameter("cam_offset_z").value)
        self.goal_object_standoff = max(0.0, float(self.get_parameter("goal_object_standoff").value))

        self.object_anchors = self._load_object_anchors(csv_path)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.goal_pub = self.create_publisher(PoseStamped, goal_topic, 10)
        self.target_sub = self.create_subscription(String, target_topic, self._on_target, 10)

        self.get_logger().info(
            f"semantic_goal_node started. anchors={len(self.object_anchors)}, "
            f"object_csv={csv_path}, "
            f"target_topic={target_topic}, goal_topic={goal_topic}, "
            f"tf=({self.map_frame}->{self.base_link_frame}, timeout={self.tf_timeout_sec:.2f}s), "
            f"cam_offset=({self.cam_offset_x:.3f},{self.cam_offset_y:.3f},{self.cam_offset_z:.3f}), "
            f"goal_object_standoff={self.goal_object_standoff:.3f}m"
        )

    def _load_object_anchors(self, csv_path: Path) -> list[ObjectAnchor]:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        grouped: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            required = {"class_name", "track_id", "center_world_x", "center_world_y", "center_world_z"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"CSV缺少必要字段: {required}")

            for row in reader:
                try:
                    class_name = str(row["class_name"]).strip().lower()
                    track_id = int(row["track_id"])
                    x_cam_origin = float(row["center_world_x"])
                    y_cam_origin = float(row["center_world_y"])
                    z_cam_origin = float(row["center_world_z"])
                except (ValueError, TypeError, KeyError):
                    continue

                # object_csv坐标默认以“机器人出发时相机位置”为原点；
                # Nav2地图通常以“机器人出发时base_link”为原点，因此加上相机在base_link下的平移偏置。
                x = x_cam_origin + self.cam_offset_x
                y = y_cam_origin + self.cam_offset_y
                z = z_cam_origin + self.cam_offset_z
                grouped.setdefault((class_name, track_id), []).append((x, y, z))

        anchors: list[ObjectAnchor] = []
        for (class_name, track_id), points in grouped.items():
            if not points:
                continue
            sx = sum(p[0] for p in points) / len(points)
            sy = sum(p[1] for p in points) / len(points)
            sz = sum(p[2] for p in points) / len(points)
            anchors.append(ObjectAnchor(class_name=class_name, track_id=track_id, x=sx, y=sy, z=sz))

        return anchors

    def _get_robot_xy_from_tf(self) -> tuple[float, float] | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_link_frame,
                Time(),
                timeout=Duration(seconds=max(0.0, self.tf_timeout_sec)),
            )
        except TransformException as exc:
            self.get_logger().warning(
                f"TF查询失败: {self.map_frame}->{self.base_link_frame}: {exc}"
            )
            return None

        robot_x = float(transform.transform.translation.x)
        robot_y = float(transform.transform.translation.y)
        return robot_x, robot_y

    def _on_target(self, msg: String) -> None:
        target_name = msg.data.strip().lower()
        if not target_name:
            self.get_logger().warning("收到空目标名，忽略")
            return

        robot_xy = self._get_robot_xy_from_tf()
        if robot_xy is None:
            return
        robot_x, robot_y = robot_xy

        candidates = [anchor for anchor in self.object_anchors if anchor.class_name == target_name]
        if not candidates:
            self.get_logger().warning(f"未找到目标类别: {target_name}")
            return

        def dist2(anchor: ObjectAnchor) -> float:
            dx = anchor.x - robot_x
            dy = anchor.y - robot_y
            return dx * dx + dy * dy

        chosen = min(candidates, key=dist2)
        dx = chosen.x - robot_x
        dy = chosen.y - robot_y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            self.get_logger().warning("机器人与目标点几乎重合，忽略")
            return

        yaw = math.atan2(dy, dx)
        qx, qy, qz, qw = yaw_to_quaternion(yaw)

        move_dist = max(0.0, dist - self.goal_object_standoff)
        ux = dx / dist
        uy = dy / dist
        goal_x = robot_x + ux * move_dist
        goal_y = robot_y + uy * move_dist

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.map_frame
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.position.z = chosen.z + self.z_offset
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        self.goal_pub.publish(goal)
        self.get_logger().info(
            f"发布目标 {target_name} -> track_id={chosen.track_id}, "
            f"obj=({chosen.x:.3f}, {chosen.y:.3f}), "
            f"goal=({goal.pose.position.x:.3f}, {goal.pose.position.y:.3f}, {goal.pose.position.z:.3f}), "
            f"dist={dist:.3f}, standoff={self.goal_object_standoff:.3f}, yaw={yaw:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 semantic navigation goal publisher")
    parser.add_argument(
        "pack_name",
        nargs="?",
        default=None,
        help="包名（如 667_14_sampled_i30）。传入后自动使用 output_<pack>/centers_<pack>.csv",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to object centers CSV file（优先级高于包名自动解析）",
    )
    parser.add_argument(
        "--pack-name",
        dest="pack_name_opt",
        type=str,
        default=None,
        help="包名（与位置参数等价，二选一即可）",
    )
    parser.add_argument(
        "--cam-offset-x",
        type=float,
        default=0.33,
        help="Camera X offset from base_link (meters)",
    )
    parser.add_argument(
        "--cam-offset-y",
        type=float,
        default=0.0,
        help="Camera Y offset from base_link (meters)",
    )
    parser.add_argument(
        "--cam-offset-z",
        type=float,
        default=0.075,
        help="Camera Z offset from base_link (meters)",
    )
    parser.add_argument(
        "--goal-object-standoff",
        type=float,
        default=0.6,
        help="与物体保持的最小距离（米）",
    )
    args, ros_args = parser.parse_known_args()

    work_root = Path("/home/joker/Desktop/yolo_rgbd_object_center")

    pack_pos = args.pack_name.strip() if isinstance(args.pack_name, str) and args.pack_name.strip() else None
    pack_opt = args.pack_name_opt.strip() if isinstance(args.pack_name_opt, str) and args.pack_name_opt.strip() else None
    if pack_pos is not None and pack_opt is not None and pack_pos != pack_opt:
        raise ValueError("位置参数 pack_name 与 --pack-name 不一致，请仅保留一个或保证一致")
    chosen_pack = pack_pos if pack_pos is not None else pack_opt

    if args.csv_path:
        resolved_csv = Path(args.csv_path)
    else:
        if chosen_pack:
            resolved_csv = work_root / f"output_{chosen_pack}" / f"centers_{chosen_pack}.csv"
        else:
            resolved_csv = work_root / "object_centers.csv"

    if not resolved_csv.exists():
        raise FileNotFoundError(
            f"CSV not found: {resolved_csv}. 请传 --csv-path 或 pack_name/--pack-name。"
        )

    rclpy.init(args=ros_args)

    node = SemanticGoalNode(
        object_csv_default=str(resolved_csv),
        cam_offset_x_default=args.cam_offset_x,
        cam_offset_y_default=args.cam_offset_y,
        cam_offset_z_default=args.cam_offset_z,
        goal_object_standoff_default=args.goal_object_standoff,
    )
    
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
