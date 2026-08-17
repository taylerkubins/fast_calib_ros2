#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从当前目录的 rosbag2 中提取第一帧 LiDAR 点云（PCD）和第一帧相机图像，
用于 lidar-相机标定（标定流程第三部分：得到一帧 lidar 的 PCD 和一帧图像）。

- 输入：当前目录下的 rosbag2（含 metadata.yaml 与 .db3/.mcap）
- 输出：与脚本同目录下的 frame_1.pcd、image_1.png

用法（在 left 目录下执行）：
    cd /path/to/left
    python3 extract_first_frame.py

或指定 bag 目录：
    python3 extract_first_frame.py /path/to/left

依赖：ROS2 环境（source setup.bash）、rosbag2_py、sensor_msgs、opencv-python、numpy<2
"""

from __future__ import print_function

import os
import sys

try:
    import cv2
    import numpy as np
except ImportError as e:
    print("[ERROR] 需要 opencv-python 和 numpy: pip3 install 'opencv-python<4.12' 'numpy<2'", file=sys.stderr)
    sys.exit(1)

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
    from sensor_msgs_py.point_cloud2 import read_points as read_points_pc2
except ImportError as e:
    print("[ERROR] 请先 source ROS2 环境: source /opt/ros/<distro>/setup.bash", file=sys.stderr)
    print("  详细: {}".format(e), file=sys.stderr)
    sys.exit(1)


# --------------- 配置：输出文件名与默认 topic ---------------
OUT_PCD = "lidar_1.pcd"
OUT_IMAGE = "image_1.png"
DEFAULT_LIDAR_TOPIC = "/livox/lidar"
DEFAULT_IMAGE_TOPIC = "/camera/camera/color/image_raw"


def imgmsg_to_bgr(msg):
    """将 sensor_msgs/Image 转为 BGR numpy 数组（不依赖 cv_bridge，避免 NumPy 2.x ABI 冲突）。"""
    h, w = msg.height, msg.width
    enc = (msg.encoding or "bgr8").strip().lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("bgr8", "rgb8"):
        step = msg.step if msg.step else w * 3
        arr = data.reshape((h, step))[:, : w * 3].reshape((h, w, 3))
        return arr if enc == "bgr8" else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if enc in ("bgra8", "rgba8"):
        step = msg.step if msg.step else w * 4
        arr = data.reshape((h, step))[:, : w * 4].reshape((h, w, 4))
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR if enc == "bgra8" else cv2.COLOR_RGBA2BGR)
    if enc in ("mono8", "8uc1"):
        step = msg.step if msg.step else w
        arr = data.reshape((h, step))[:, :w]
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if enc in ("16uc1", "mono16"):
        data16 = np.frombuffer(msg.data, dtype=np.uint16)
        step = (msg.step // 2) if msg.step else w
        arr = data16.reshape((h, step))[:, :w]
        arr8 = cv2.convertScaleAbs(arr, alpha=255.0 / 65535.0)
        return cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)
    raise ValueError("不支持的图像编码: {}".format(msg.encoding))


def get_bag_uri_and_storage(bag_path):
    """解析 bag 路径，返回 (uri, storage_id)。"""
    bag_path = os.path.abspath(bag_path)
    if os.path.isfile(bag_path):
        if bag_path.lower().endswith(".mcap"):
            return os.path.dirname(bag_path), "mcap"
        if bag_path.lower().endswith(".db3"):
            return os.path.dirname(bag_path), "sqlite3"
        return os.path.dirname(bag_path), "mcap"
    if os.path.isdir(bag_path):
        for f in os.listdir(bag_path):
            if f.endswith(".mcap"):
                return bag_path, "mcap"
            if f.endswith(".db3"):
                return bag_path, "sqlite3"
        return bag_path, "mcap"
    return bag_path, "mcap"


def find_intensity_field(msg):
    """PointCloud2 中强度字段名。"""
    for field in msg.fields:
        if field.name.lower() in ("intensity", "reflectivity", "i", "ref"):
            return field.name
    return None


def save_pcd_one_frame(points, intensities, output_path):
    """保存一帧点云为带 intensity 的 PCD（ASCII）。"""
    n = len(points)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        "WIDTH {}\n"
        "HEIGHT 1\n"
        "POINTS {}\n"
        "DATA ascii\n"
    ).format(n, n)
    with open(output_path, "w") as f:
        f.write(header)
        for (x, y, z), i in zip(points, intensities):
            f.write("{} {} {} {}\n".format(x, y, z, i))
    print("[PCD] 已保存第一帧点云: {}".format(output_path))


def main():
    # 脚本所在目录即 left 目录，作为默认 bag 路径与输出路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bag_path = script_dir
    if len(sys.argv) > 1:
        bag_path = os.path.abspath(sys.argv[1])
    output_dir = bag_path if os.path.isdir(bag_path) else os.path.dirname(bag_path)

    uri, storage_id = get_bag_uri_and_storage(bag_path)
    if not os.path.isdir(uri) and not os.path.isfile(bag_path):
        print("[ERROR] 无效的 bag 路径: {}".format(bag_path), file=sys.stderr)
        sys.exit(1)

    print("[Bag] 打开: uri={}, storage_id={}".format(uri, storage_id))
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="",
        output_serialization_format="",
    )
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    lidar_topic = None
    image_topic = None
    for name, typ in topic_types.items():
        if typ == "sensor_msgs/msg/PointCloud2":
            lidar_topic = name
            break
    for name, typ in topic_types.items():
        if typ == "sensor_msgs/msg/Image":
            image_topic = name
            break
    if not lidar_topic:
        lidar_topic = DEFAULT_LIDAR_TOPIC
        if lidar_topic not in topic_types:
            print("[ERROR] 未找到 PointCloud2 topic，可用: {}".format(list(topic_types.keys())), file=sys.stderr)
            sys.exit(1)
    if not image_topic:
        image_topic = DEFAULT_IMAGE_TOPIC
        if image_topic not in topic_types:
            print("[ERROR] 未找到 Image topic，可用: {}".format(list(topic_types.keys())), file=sys.stderr)
            sys.exit(1)

    print("[Bag] LiDAR topic: {}".format(lidar_topic))
    print("[Bag] Image topic: {}".format(image_topic))

    lidar_done = False
    image_done = False
    lidar_type = topic_types[lidar_topic]
    image_type = topic_types[image_topic]
    msg_lidar_class = get_message(lidar_type)
    msg_image_class = get_message(image_type)

    while reader.has_next() and (not lidar_done or not image_done):
        topic, data, timestamp = reader.read_next()
        if topic == lidar_topic and not lidar_done:
            try:
                msg = deserialize_message(data, msg_lidar_class)
                if lidar_type == "sensor_msgs/msg/PointCloud2":
                    intensity_field = find_intensity_field(msg)
                    if not intensity_field:
                        field_names = ["x", "y", "z"]
                        points = []
                        for p in read_points_pc2(msg, field_names=field_names, skip_nans=True):
                            points.append([p[0], p[1], p[2]])
                        intensities = [0.0] * len(points)
                    else:
                        field_names = ["x", "y", "z", intensity_field]
                        points = []
                        intensities = []
                        for p in read_points_pc2(msg, field_names=field_names, skip_nans=True):
                            points.append([p[0], p[1], p[2]])
                            intensities.append(p[3])
                    if points:
                        pcd_path = os.path.join(output_dir, OUT_PCD)
                        save_pcd_one_frame(points, intensities, pcd_path)
                        lidar_done = True
            except Exception as e:
                print("[WARN] 解析第一帧点云失败: {}".format(e), file=sys.stderr)

        if topic == image_topic and not image_done:
            try:
                msg = deserialize_message(data, msg_image_class)
                cv_image = imgmsg_to_bgr(msg)
                img_path = os.path.join(output_dir, OUT_IMAGE)
                if cv2.imwrite(img_path, cv_image):
                    print("[Image] 已保存第一帧图像: {}".format(img_path))
                    image_done = True
                else:
                    print("[WARN] 写入图像失败: {}".format(img_path), file=sys.stderr)
            except Exception as e:
                print("[WARN] 解析第一帧图像失败: {}".format(e), file=sys.stderr)

    if not lidar_done:
        print("[ERROR] 未找到可用的 LiDAR 第一帧", file=sys.stderr)
        sys.exit(1)
    if not image_done:
        print("[ERROR] 未找到可用的 Image 第一帧", file=sys.stderr)
        sys.exit(1)
    print("[OK] 第一帧 PCD 与图像已提取到目录: {}".format(output_dir))


if __name__ == "__main__":
    main()
