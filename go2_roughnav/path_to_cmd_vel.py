#!/usr/bin/env python3
"""Convert a nav_msgs/Path into simple velocity commands for Go2 policy."""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class PathToCmdVel(Node):
    def __init__(self):
        super().__init__("path_to_cmd_vel")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("odom_topic", "/Odometry")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("lookahead_m", 0.8)
        self.declare_parameter("max_lin_x", 0.35)
        self.declare_parameter("max_lin_y", 0.20)
        self.declare_parameter("max_ang_z", 0.7)
        self.path = []
        self.odom = None
        self.lookahead = float(self.get_parameter("lookahead_m").value)
        self.max_x = float(self.get_parameter("max_lin_x").value)
        self.max_y = float(self.get_parameter("max_lin_y").value)
        self.max_z = float(self.get_parameter("max_ang_z").value)
        self.pub = self.create_publisher(Twist, str(self.get_parameter("cmd_vel_topic").value), 10)
        self.create_subscription(Path, str(self.get_parameter("path_topic").value), self._path_cb, 1)
        self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value), self._odom_cb, 10)
        self.create_timer(0.05, self._tick)

    def _path_cb(self, msg: Path) -> None:
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def _odom_cb(self, msg: Odometry) -> None:
        self.odom = msg

    def _tick(self) -> None:
        cmd = Twist()
        if self.odom is None or not self.path:
            self.pub.publish(cmd)
            return
        p = self.odom.pose.pose.position
        yaw = _yaw(self.odom.pose.pose.orientation)
        dists = [math.hypot(x - p.x, y - p.y) for x, y in self.path]
        nearest = int(np.argmin(dists))
        target = self.path[-1]
        for candidate in self.path[nearest:]:
            if math.hypot(candidate[0] - p.x, candidate[1] - p.y) >= self.lookahead:
                target = candidate
                break
        dx = target[0] - p.x
        dy = target[1] - p.y
        c, s = math.cos(-yaw), math.sin(-yaw)
        bx = c * dx - s * dy
        by = s * dx + c * dy
        heading_err = _wrap(math.atan2(dy, dx) - yaw)
        cmd.linear.x = float(np.clip(0.6 * bx, -0.15, self.max_x))
        cmd.linear.y = float(np.clip(0.5 * by, -self.max_y, self.max_y))
        cmd.angular.z = float(np.clip(1.5 * heading_err, -self.max_z, self.max_z))
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PathToCmdVel()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
