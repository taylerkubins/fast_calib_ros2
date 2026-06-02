#!/usr/bin/env python3

import argparse
import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class OneShotImageSaver(Node):
    def __init__(self, topic: str, output_path: str):
        super().__init__("capture_one_image")
        self._topic = topic
        self._output_path = output_path
        self._bridge = CvBridge()
        self._saved = False
        self._sub = self.create_subscription(Image, self._topic, self._callback, 10)
        self.get_logger().info(f"Waiting for one image on topic: {self._topic}")

    @property
    def saved(self) -> bool:
        return self._saved

    def _callback(self, msg: Image) -> None:
        if self._saved:
            return
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
            ok = cv2.imwrite(self._output_path, cv_image)
            if not ok:
                self.get_logger().error(f"Failed to save image to: {self._output_path}")
                return
            self._saved = True
            self.get_logger().info(f"Saved image to: {self._output_path}")
        except Exception as exc:
            self.get_logger().error(f"Image conversion/saving failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one ROS image message and save it to disk.")
    parser.add_argument("--topic", required=True, help="Image topic, e.g. /camera/color/image_raw")
    parser.add_argument("--output", required=True, help="Output png/jpg path")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait before giving up")
    args = parser.parse_args()

    rclpy.init()
    node = OneShotImageSaver(args.topic, args.output)

    start = time.time()
    try:
        while rclpy.ok() and not node.saved:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.time() - start > args.timeout:
                node.get_logger().error(f"Timeout after {args.timeout:.1f}s waiting for image on {args.topic}")
                break
    finally:
        ok = node.saved
        node.destroy_node()
        rclpy.shutdown()

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
