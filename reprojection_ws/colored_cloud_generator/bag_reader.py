"""
ROS2 bag 数据读取：LiDAR 点云与相机图像。

依赖 rosbags 库，无需 source ROS2 环境。

主要功能:
    - load_pointcloud_from_bag: 合并多帧 PointCloud2 为 Nx3 点云
    - load_image_from_bag: 读取指定帧 sensor_msgs/Image 或 CompressedImage
    - find_image_topic: 自动检测 bag 中的图像话题
"""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# 支持的图像消息类型
IMAGE_MSG_TYPES = (
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
)
# 常见默认话题（仅作文档参考，实际以 bag 内话题为准）
DEFAULT_IMAGE_TOPIC = "/camera/camera/color/image_raw"

# sensor_msgs/PointField 枚举值 -> NumPy dtype
# 用于零拷贝解析 PointCloud2 原始字节
POINTFIELD_TO_DTYPE = {
    1: np.dtype(np.int8),
    2: np.dtype(np.uint8),
    3: np.dtype(np.int16),
    4: np.dtype(np.uint16),
    5: np.dtype(np.int32),
    6: np.dtype(np.uint32),
    7: np.dtype(np.float32),
    8: np.dtype(np.float64),
}


def _pointcloud2_msg_to_xyz(msg) -> np.ndarray:
    """
    将 rosbags 反序列化后的 PointCloud2 消息转为 (N, 3) float32 数组。

    实现思路（与 FAST-Calib data_preprocess 及 PCL fromROSMsg 等价）:
        1. 根据 fields 里 x/y/z 的 offset、datatype 构造结构化 dtype
        2. np.frombuffer 直接解析 msg.data，避免 Python 循环
        3. 兼容 row_step != point_step * width 的非紧凑布局
        4. 过滤 NaN/Inf

    Args:
        msg: rosbags 反序列化后的 PointCloud2 对象

    Returns:
        shape (N, 3) 的 float32 数组；无效时返回空数组
    """
    if msg.height == 0 or msg.width == 0:
        return np.zeros((0, 3), dtype=np.float32)

    field_map = {field.name: field for field in msg.fields}
    if not all(name in field_map for name in ("x", "y", "z")):
        return np.zeros((0, 3), dtype=np.float32)

    # 按消息字节序构造 x/y/z 的结构化 dtype
    endian = ">" if msg.is_bigendian else "<"
    formats = []
    offsets = []
    names = []
    for axis in ("x", "y", "z"):
        field = field_map[axis]
        base_dtype = POINTFIELD_TO_DTYPE.get(field.datatype)
        if base_dtype is None:
            return np.zeros((0, 3), dtype=np.float32)
        names.append(axis)
        formats.append(base_dtype.newbyteorder(endian))
        offsets.append(field.offset)

    point_dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": msg.point_step,
        }
    )

    total_points = msg.width * msg.height
    expected_len = msg.row_step * msg.height
    if len(msg.data) < expected_len:
        return np.zeros((0, 3), dtype=np.float32)

    # 紧凑布局：整包一次 frombuffer
    if msg.row_step == msg.point_step * msg.width:
        raw_data = np.frombuffer(msg.data, dtype=point_dtype, count=total_points)
    else:
        # 非紧凑布局：每行跳过 row_step 与有效数据之间的填充字节
        rows = []
        data_view = memoryview(msg.data)
        row_nbytes = msg.point_step * msg.width
        for row_idx in range(msg.height):
            start = row_idx * msg.row_step
            row_buffer = data_view[start : start + row_nbytes]
            rows.append(np.frombuffer(row_buffer, dtype=point_dtype, count=msg.width))
        raw_data = np.concatenate(rows) if rows else np.array([], dtype=point_dtype)

    points = np.empty((raw_data.shape[0], 3), dtype=np.float32)
    points[:, 0] = raw_data["x"].astype(np.float32, copy=False)
    points[:, 1] = raw_data["y"].astype(np.float32, copy=False)
    points[:, 2] = raw_data["z"].astype(np.float32, copy=False)
    points = points[np.isfinite(points).all(axis=1)]
    return points


def _get_typestore():
    """获取 rosbags 类型仓库，优先 LATEST，失败则回退 HUMBLE。"""
    try:
        return get_typestore(Stores.LATEST)
    except Exception:
        return get_typestore(Stores.ROS2_HUMBLE)


def _resolve_bag_path(bag_path: str) -> str:
    """校验 bag 路径存在并返回绝对路径。"""
    bag_path = os.path.abspath(os.path.expanduser(bag_path))
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"bag 路径不存在: {bag_path}")
    return bag_path


def find_image_topic(bag_path: str, preferred_topic: str = "") -> str:
    """
    在 bag 中查找图像话题。

    查找策略:
        1. preferred_topic 非空且存在于 bag 中 -> 直接使用
        2. 否则返回第一个 Image/CompressedImage 类型话题

    Args:
        bag_path: bag 目录或 .db3 文件
        preferred_topic: 配置文件中指定的 image_topic（可为空）

    Raises:
        ValueError: 指定话题不存在，或 bag 中无任何图像话题
    """
    bag_path = _resolve_bag_path(bag_path)
    with Reader(bag_path) as reader:
        topics = reader.topics
        if preferred_topic:
            if preferred_topic not in topics:
                raise ValueError(f"图像话题 {preferred_topic} 不在 bag 中，可用: {list(topics.keys())}")
            if topics[preferred_topic].msgtype not in IMAGE_MSG_TYPES:
                raise ValueError(f"话题 {preferred_topic} 不是图像类型: {topics[preferred_topic].msgtype}")
            return preferred_topic

        for topic, info in topics.items():
            if info.msgtype in IMAGE_MSG_TYPES:
                return topic

    raise ValueError(f"bag 中未找到 Image/CompressedImage 话题: {bag_path}")


def _image_message_to_bgr(msg, msg_type_name: str) -> np.ndarray:
    """
    将 ROS 图像消息解码为 OpenCV BGR 格式 (H, W, 3) uint8。

    支持:
        - sensor_msgs/msg/CompressedImage: cv2.imdecode
        - sensor_msgs/msg/Image: bgr8 / rgb8 / 单通道灰度

    Args:
        msg: 反序列化后的图像消息
        msg_type_name: 消息类型全名

    Returns:
        BGR 图像数组
    """
    if msg_type_name == "sensor_msgs/msg/CompressedImage":
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("CompressedImage 解码失败")
        return img

    if msg_type_name == "sensor_msgs/msg/Image":
        h, w = msg.height, msg.width
        step = msg.step if msg.step else w * 3
        enc = (msg.encoding or "bgr8").strip().lower()
        ch = 3 if ("rgb" in enc or "bgr" in enc) else 1
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, step))
        arr = arr[:, : w * ch]
        if ch == 1:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        img = arr.reshape((h, w, ch))
        if "rgb" in enc:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    raise ValueError(f"不支持的图像消息类型: {msg_type_name}")


def load_image_from_bag(
    bag_path: str,
    topic_name: str = "",
    frame_index: int = 0,
) -> Tuple[np.ndarray, str, int]:
    """
    从 bag 顺序读取并返回指定帧的相机图像。

    按 bag 内消息时间顺序计数，frame_index=0 表示该话题的第一条消息。

    Args:
        bag_path: bag 目录或 .db3 文件
        topic_name: 图像话题；空字符串则自动检测
        frame_index: 使用第几帧（0 表示第一帧）

    Returns:
        (bgr_image, topic_used, frame_index_used)

    Raises:
        RuntimeError: 指定帧不存在或图像解码失败
    """
    bag_path = _resolve_bag_path(bag_path)
    topic = find_image_topic(bag_path, topic_name.strip())
    typestore = _get_typestore()

    seen = 0
    with Reader(bag_path) as reader:
        msg_type = reader.topics[topic].msgtype
        for connection, _timestamp, rawdata in reader.messages():
            if connection.topic != topic:
                continue
            # 跳过前 frame_index 帧，取目标帧
            if seen < frame_index:
                seen += 1
                continue

            msg = typestore.deserialize_cdr(rawdata, msg_type)
            image = _image_message_to_bgr(msg, msg_type)
            if image.size == 0:
                raise RuntimeError(f"bag 图像为空: topic={topic}, frame={frame_index}")
            return image, topic, frame_index

    raise RuntimeError(
        f"bag 中未读到第 {frame_index + 1} 帧图像: {bag_path}, topic={topic}"
    )


def load_pointcloud_from_bag(
    bag_path: str,
    topic_name: str,
    max_frames: int = 0,
    skip_rate: int = 1,
) -> np.ndarray:
    """
    从 ROS2 bag 读取 LiDAR 点云并按帧累加。

    行为与 FAST-Calib data_preprocess.hpp::loadPointCloudFromBag 一致:
        遍历 bag 中所有匹配 topic 的 PointCloud2，逐帧 append 后 vstack。

    Args:
        bag_path: bag 目录或 .db3 文件
        topic_name: LiDAR 话题名（如 /livox/lidar）
        max_frames: 最多合并帧数；0 表示不限制，合并全部帧
        skip_rate: 合并完成后均匀降采样步长（1=不降采样）

    Returns:
        shape (N, 3) 的 float32 LiDAR 系点云

    Raises:
        ValueError: 话题不存在
        RuntimeError: 未读到任何有效点
    """
    bag_path = _resolve_bag_path(bag_path)
    typestore = _get_typestore()

    chunks = []
    frame_count = 0

    with Reader(bag_path) as reader:
        topics = list(reader.topics.keys())
        if topic_name not in topics:
            raise ValueError(f"话题 {topic_name} 不在 bag 中，可用: {topics}")

        for connection, _timestamp, rawdata in reader.messages():
            if connection.topic != topic_name:
                continue

            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            xyz = _pointcloud2_msg_to_xyz(msg)
            if xyz.size == 0:
                continue
            chunks.append(xyz)
            frame_count += 1
            if max_frames > 0 and frame_count >= max_frames:
                break

    if not chunks:
        raise RuntimeError(f"bag 中未读到有效点云: {bag_path}, topic={topic_name}")

    cloud = np.vstack(chunks)
    if skip_rate > 1:
        cloud = cloud[::skip_rate]
    return cloud
