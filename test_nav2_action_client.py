#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import argparse

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose


class Nav2ActionTestClient(Node):
    def __init__(self, action_name: str = "navigate_to_pose"):
        super().__init__("nav2_action_test_client")
        self._action_name = action_name
        self._client = ActionClient(self, NavigateToPose, self._action_name)
        self._goal_handle = None
        self._done = False

    @staticmethod
    def yaw_to_quaternion(yaw: float):
        half = yaw * 0.5
        qz = math.sin(half)
        qw = math.cos(half)
        return qz, qw

    def build_goal_pose(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        frame_id: str = "map",
    ) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)

        qz, qw = self.yaw_to_quaternion(yaw)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    @staticmethod
    def goal_status_to_text(status: int) -> str:
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

    def send_goal(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        frame_id: str = "map",
        wait_server_timeout: float = 5.0,
    ) -> None:
        self.get_logger().info(f"等待 Nav2 action server: {self._action_name} ...")
        ok = self._client.wait_for_server(timeout_sec=wait_server_timeout)
        if not ok:
            self.get_logger().error(f"Nav2 action server 不可用: {self._action_name}")
            self._done = True
            return

        pose = self.build_goal_pose(x, y, z, yaw, frame_id)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(
            f"发送目标: frame={frame_id}, "
            f"x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={math.degrees(yaw):.1f} deg"
        )

        send_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        send_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        parts = []

        if hasattr(fb, "distance_remaining"):
            parts.append(f"distance_remaining={fb.distance_remaining:.3f}")

        if hasattr(fb, "navigation_time"):
            nav_time = fb.navigation_time.sec + fb.navigation_time.nanosec / 1e9
            parts.append(f"navigation_time={nav_time:.2f}s")

        if hasattr(fb, "estimated_time_remaining"):
            eta = (
                fb.estimated_time_remaining.sec
                + fb.estimated_time_remaining.nanosec / 1e9
            )
            parts.append(f"eta={eta:.2f}s")

        if hasattr(fb, "number_of_recoveries"):
            parts.append(f"recoveries={fb.number_of_recoveries}")

        if hasattr(fb, "current_pose"):
            cp = fb.current_pose.pose.position
            parts.append(f"current_pose=({cp.x:.3f}, {cp.y:.3f}, {cp.z:.3f})")

        if parts:
            self.get_logger().info("[Feedback] " + ", ".join(parts))

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"发送目标异常: {e}")
            self._done = True
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warning("目标被 action server 拒绝")
            self._done = True
            return

        self._goal_handle = goal_handle
        self.get_logger().info("目标已被接受，等待执行结果...")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        try:
            result_wrap = future.result()
        except Exception as e:
            self.get_logger().error(f"获取结果异常: {e}")
            self._done = True
            return

        status = result_wrap.status
        status_text = self.goal_status_to_text(status)

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"导航成功: {status_text}")
        else:
            self.get_logger().warning(f"导航结束: {status_text}")

        self._done = True


def main():
    parser = argparse.ArgumentParser(
        description="Nav2 NavigateToPose action client test"
    )
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0, help="弧度")
    parser.add_argument("--yaw-deg", type=float, default=None, help="角度")
    parser.add_argument("--frame-id", type=str, default="map")
    parser.add_argument("--action-name", type=str, default="navigate_to_pose")
    parser.add_argument("--wait-server-timeout", type=float, default=5.0)
    args = parser.parse_args()

    yaw = math.radians(args.yaw_deg) if args.yaw_deg is not None else args.yaw

    rclpy.init()
    node = Nav2ActionTestClient(action_name=args.action_name)

    try:
        node.send_goal(
            x=args.x,
            y=args.y,
            z=args.z,
            yaw=yaw,
            frame_id=args.frame_id,
            wait_server_timeout=args.wait_server_timeout,
        )

        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        node.get_logger().info("收到 Ctrl+C，退出")
    finally:
        try:
            if node._goal_handle is not None:
                node._goal_handle = None
            if hasattr(node, "_client") and node._client is not None:
                node._client.destroy()
                node._client = None
        except Exception as e:
            print(f"[Warn] ActionClient destroy 异常: {e}")

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
