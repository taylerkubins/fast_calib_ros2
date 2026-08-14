#!/usr/bin/env python3
"""
LiDAR-相机点云投影可视化节点。

功能概述：
    订阅同步的相机图像与 LiDAR 点云，利用外参将点云从 LiDAR 坐标系变换到相机坐标系，
    再通过相机内参投影到图像平面，以彩色圆点叠加在图像上并发布。

依赖配置（由 ros2_camera_lidar_fusion 包提供）：
    - general: 外参/内参 YAML 路径、话题名、队列参数等
    - projection: 投影可视化专用参数（深度过滤、降采样、着色等）

坐标系约定：
    - 外参矩阵 T_lidar_to_cam：将 LiDAR 系下的齐次坐标变换到相机系
    - 相机系 Z 轴指向前方，仅保留 Z > min_depth_m 的点参与投影
"""

import os
from typing import Dict

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import yaml

from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from ros2_camera_lidar_fusion.read_yaml import extract_configuration


def _as_bool(value: object, default: bool = False) -> bool:
    """将 YAML/字符串等配置值解析为布尔量。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def resolve_config_path(config_folder: str, filename: str, package_config_dir: str) -> str:
    """
    解析配置文件绝对路径。

    查找顺序：
        1. filename 本身为绝对路径时直接返回
        2. config_folder（绝对或相对）+ filename
        3. 包内 config 目录 + filename
    若均不存在，返回第一个候选路径（由调用方在读取时触发 FileNotFoundError）。
    """
    if os.path.isabs(filename):
        return filename

    candidates = []
    if config_folder:
        if os.path.isabs(config_folder):
            candidates.append(config_folder)
        else:
            candidates.append(os.path.join(package_config_dir, config_folder))
            candidates.append(os.path.abspath(config_folder))
    candidates.append(package_config_dir)

    for base in candidates:
        resolved = os.path.join(base, filename)
        if os.path.isfile(resolved):
            return resolved

    return os.path.join(candidates[0], filename)


def load_extrinsic_matrix(yaml_path: str, invert: bool = False) -> np.ndarray:
    """
    从 YAML 加载 4x4 外参矩阵（LiDAR -> Camera）。

    支持的键名（按优先级）：
        extrinsic_matrix / T_lidar_to_cam / transform_lidar_to_camera

    Args:
        yaml_path: 外参 YAML 文件路径
        invert: 为 True 时对矩阵求逆（用于配置文件中存储的是 cam->lidar 的情况）

    Returns:
        shape (4, 4) 的 float64 齐次变换矩阵
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No extrinsic file found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    matrix_list = None
    for key in ('extrinsic_matrix', 'T_lidar_to_cam', 'transform_lidar_to_camera'):
        if key in data:
            matrix_list = data[key]
            break
    if matrix_list is None:
        raise KeyError(
            f"YAML {yaml_path} has none of supported keys: "
            "'extrinsic_matrix', 'T_lidar_to_cam', 'transform_lidar_to_camera'."
        )

    T = np.array(matrix_list, dtype=np.float64)
    if T.size == 16:
        T = T.reshape(4, 4)
    if T.shape != (4, 4):
        raise ValueError(f"Extrinsic matrix in {yaml_path} is not 4x4.")
    if invert:
        T = np.linalg.inv(T)
    return T


def load_camera_calibration(yaml_path: str) -> (np.ndarray, np.ndarray):
    """
    从 ROS 相机标定 YAML 加载内参与畸变系数。

    期望格式（与 camera_calibration 输出一致）：
        camera_matrix: { data: [fx, 0, cx, 0, fy, cy, 0, 0, 1] }
        distortion_coefficients: { data: [k1, k2, p1, p2, ...] }

    Returns:
        camera_matrix: (3, 3) 内参矩阵 K
        dist_coeffs: (1, N) 畸变系数，供 cv2.projectPoints 使用
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No camera calibration file: {yaml_path}")

    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)

    cam_mat_data = calib_data['camera_matrix']['data']
    camera_matrix = np.array(cam_mat_data, dtype=np.float64)
    if camera_matrix.size == 9:
        camera_matrix = camera_matrix.reshape(3, 3)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"Camera matrix in {yaml_path} is not 3x3.")

    dist_data = calib_data['distortion_coefficients']['data']
    dist_coeffs = np.array(dist_data, dtype=np.float64).reshape(1, -1)

    return camera_matrix, dist_coeffs


# PointField 枚举值 -> NumPy dtype，用于零拷贝解析 PointCloud2 原始字节
POINTFIELD_TO_DTYPE: Dict[int, np.dtype] = {
    PointField.INT8: np.dtype(np.int8),
    PointField.UINT8: np.dtype(np.uint8),
    PointField.INT16: np.dtype(np.int16),
    PointField.UINT16: np.dtype(np.uint16),
    PointField.INT32: np.dtype(np.int32),
    PointField.UINT32: np.dtype(np.uint32),
    PointField.FLOAT32: np.dtype(np.float32),
    PointField.FLOAT64: np.dtype(np.float64),
}


def pointcloud2_to_xyz_array_fast(
    cloud_msg: PointCloud2,
    skip_rate: int = 1,
    logger=None
) -> np.ndarray:
    """
    将 sensor_msgs/PointCloud2 高效解析为 (N, 3) 的 XYZ 数组。

    实现要点：
        - 根据 fields 中 x/y/z 的 offset 与 datatype 构造结构化 dtype，直接 frombuffer 解析
        - 兼容 row_step != point_step * width 的非紧凑布局（逐行读取）
        - 过滤 NaN/Inf
        - skip_rate > 1 时均匀降采样，减轻投影与绘制的计算量

    Args:
        cloud_msg: ROS2 点云消息
        skip_rate: 降采样步长，1 表示保留全部有效点
        logger: 可选 ROS logger，解析失败时输出警告

    Returns:
        float32 数组，shape (N, 3)，每行为 [x, y, z]（LiDAR 坐标系）
    """
    if cloud_msg.height == 0 or cloud_msg.width == 0:
        return np.zeros((0, 3), dtype=np.float32)

    field_map = {field.name: field for field in cloud_msg.fields}
    if not all(name in field_map for name in ('x', 'y', 'z')):
        if logger is not None:
            logger.warn("PointCloud2 does not contain x/y/z fields.")
        return np.zeros((0, 3), dtype=np.float32)

    # 按消息字节序构造 x/y/z 的结构化 dtype
    endian = '>' if cloud_msg.is_bigendian else '<'
    formats = []
    offsets = []
    names = []
    for axis in ('x', 'y', 'z'):
        field = field_map[axis]
        base_dtype = POINTFIELD_TO_DTYPE.get(field.datatype)
        if base_dtype is None:
            if logger is not None:
                logger.warn(f"Unsupported datatype for field '{axis}': {field.datatype}")
            return np.zeros((0, 3), dtype=np.float32)
        names.append(axis)
        formats.append(base_dtype.newbyteorder(endian))
        offsets.append(field.offset)

    point_dtype = np.dtype(
        {
            'names': names,
            'formats': formats,
            'offsets': offsets,
            'itemsize': cloud_msg.point_step
        }
    )

    total_points = cloud_msg.width * cloud_msg.height
    expected_len = cloud_msg.row_step * cloud_msg.height
    if len(cloud_msg.data) < expected_len:
        if logger is not None:
            logger.warn("PointCloud2 data buffer is smaller than expected from row_step/height.")
        return np.zeros((0, 3), dtype=np.float32)

    # 紧凑布局：整包一次 frombuffer；否则按行跳过 row_step 填充字节
    if cloud_msg.row_step == cloud_msg.point_step * cloud_msg.width:
        raw_data = np.frombuffer(cloud_msg.data, dtype=point_dtype, count=total_points)
    else:
        rows = []
        data_view = memoryview(cloud_msg.data)
        row_nbytes = cloud_msg.point_step * cloud_msg.width
        for row_idx in range(cloud_msg.height):
            start = row_idx * cloud_msg.row_step
            row_buffer = data_view[start:start + row_nbytes]
            rows.append(np.frombuffer(row_buffer, dtype=point_dtype, count=cloud_msg.width))
        raw_data = np.concatenate(rows) if rows else np.array([], dtype=point_dtype)

    points = np.empty((raw_data.shape[0], 3), dtype=np.float32)
    points[:, 0] = raw_data['x'].astype(np.float32, copy=False)
    points[:, 1] = raw_data['y'].astype(np.float32, copy=False)
    points[:, 2] = raw_data['z'].astype(np.float32, copy=False)
    # 剔除无效坐标
    points = points[np.isfinite(points).all(axis=1)]

    if skip_rate > 1:
        points = points[::skip_rate]

    return points


class LidarCameraProjectionNode(Node):
    """
    LiDAR 点云投影到相机图像的 ROS2 节点。

    数据流：
        Image + PointCloud2 (时间同步)
            -> 点云解析与坐标变换 (LiDAR -> Camera)
            -> 深度过滤 + cv2.projectPoints 投影
            -> 在图像上绘制彩色点并发布
    """

    def __init__(self):
        super().__init__('lidar_camera_projection_node')

        # ---------- 加载融合包总配置 ----------
        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return

        general_cfg = config_file.get('general', {})
        projection_cfg = config_file.get('projection', {})

        package_config_dir = os.path.join(get_package_share_directory('ros2_camera_lidar_fusion'), 'config')
        config_folder = str(general_cfg.get('config_folder', ''))

        # ---------- 外参：LiDAR -> Camera ----------
        invert_extrinsic = _as_bool(projection_cfg.get('invert_extrinsic', False))
        extrinsic_yaml = resolve_config_path(
            config_folder,
            general_cfg['camera_extrinsic_calibration'],
            package_config_dir
        )
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml, invert=invert_extrinsic)

        # ---------- 内参：相机 K 与畸变 ----------
        camera_yaml = resolve_config_path(
            config_folder,
            general_cfg['camera_intrinsic_calibration'],
            package_config_dir
        )
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)

        self.get_logger().info(f"Loaded extrinsic from: {extrinsic_yaml}")
        self.get_logger().info("Loaded extrinsic matrix:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info(f"Loaded camera calibration from: {camera_yaml}")
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))
        self.get_logger().info("Distortion coeffs:\n{}".format(self.dist_coeffs))

        # ---------- 订阅：图像 + 点云（近似时间同步） ----------
        lidar_topic = config_file['lidar']['lidar_topic']
        image_topic = config_file['camera']['image_topic']
        self.get_logger().info(f"Subscribing to lidar topic: {lidar_topic}")
        self.get_logger().info(f"Subscribing to image topic: {image_topic}")

        self.image_sub = Subscriber(self, Image, image_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic)

        # slop：允许的时间戳偏差（秒）；queue_size：同步队列长度
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub],
            queue_size=max(1, int(projection_cfg.get('queue_size', general_cfg.get('queue_size', 5)))),
            slop=float(projection_cfg.get('slop', general_cfg.get('slop', 0.07)))
        )
        self.ts.registerCallback(self.sync_callback)

        # ---------- 发布：叠加点云后的图像 ----------
        projected_topic = config_file['camera']['projected_topic']
        self.pub_image = self.create_publisher(Image, projected_topic, 1)
        self.bridge = CvBridge()

        # ---------- 投影与可视化参数 ----------
        self.skip_rate = max(1, int(projection_cfg.get('skip_rate', general_cfg.get('skip_rate', 1))))
        self.min_depth_m = float(projection_cfg.get('min_depth_m', 0.0))
        max_depth_m = projection_cfg.get('max_depth_m', None)
        self.max_depth_m = float(max_depth_m) if max_depth_m is not None else None
        self.point_radius = max(1, int(projection_cfg.get('point_radius', 2)))
        self.use_depth_colormap = _as_bool(projection_cfg.get('use_depth_colormap', True), default=True)
        self.show_overlay_stats = _as_bool(projection_cfg.get('show_overlay_stats', True), default=True)

    def sync_callback(self, image_msg: Image, lidar_msg: PointCloud2):
        """
        同步回调：将一帧 LiDAR 点云投影到对应相机图像并发布。

        处理步骤：
            1. 解析点云为 Nx3
            2. 齐次坐标左乘外参，变换到相机系
            3. 按 Z 深度过滤（仅保留相机前方的点）
            4. cv2.projectPoints 投影（点已在相机系，rvec/tvec 为零）
            5. 在图像范围内绘制圆点，可选深度伪彩色
        """
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # 1) LiDAR 系点云 -> (N, 3)
        xyz_lidar = pointcloud2_to_xyz_array_fast(
            lidar_msg,
            skip_rate=self.skip_rate,
            logger=self.get_logger()
        )
        n_points = xyz_lidar.shape[0]
        if n_points == 0:
            self.get_logger().warn("Empty cloud. Nothing to project.")
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)
            return

        # 2) LiDAR 齐次坐标 -> 相机系：P_cam = T @ P_lidar
        xyz_lidar_f64 = xyz_lidar.astype(np.float64)
        ones = np.ones((n_points, 1), dtype=np.float64)
        xyz_lidar_h = np.hstack((xyz_lidar_f64, ones))

        xyz_cam_h = xyz_lidar_h @ self.T_lidar_to_cam.T
        xyz_cam = xyz_cam_h[:, :3]

        # 3) 深度过滤：相机系 Z 为前向深度
        mask_in_front = xyz_cam[:, 2] > self.min_depth_m
        if self.max_depth_m is not None:
            mask_in_front &= xyz_cam[:, 2] < self.max_depth_m
        xyz_cam_front = xyz_cam[mask_in_front]
        n_front = xyz_cam_front.shape[0]
        if n_front == 0:
            self.get_logger().info("No valid points in front of camera after depth filtering.")
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)
            return

        # 4) 投影到像素平面（点已在相机坐标系，外参已应用，故 rvec/tvec=0）
        rvec = np.zeros((3,1), dtype=np.float64)
        tvec = np.zeros((3,1), dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            xyz_cam_front,
            rvec, tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        image_points = image_points.reshape(-1, 2)

        # 5) 按深度生成 TURBO 伪彩色（近->蓝，远->红）
        point_colors = None
        if self.use_depth_colormap and n_front > 0:
            depths = xyz_cam_front[:, 2]
            depth_min = float(np.min(depths))
            depth_max = float(np.max(depths))
            depth_scale = max(depth_max - depth_min, 1e-6)
            color_map_input = (((depths - depth_min) / depth_scale) * 255.0).astype(np.uint8).reshape(-1, 1)
            point_colors = cv2.applyColorMap(color_map_input, cv2.COLORMAP_TURBO).reshape(-1, 3)

        # 6) 在图像有效范围内绘制投影点
        h, w = cv_image.shape[:2]
        in_frame = 0
        for idx, (u, v) in enumerate(image_points):
            u_int = int(u + 0.5)
            v_int = int(v + 0.5)
            if 0 <= u_int < w and 0 <= v_int < h:
                if point_colors is not None:
                    color = tuple(int(c) for c in point_colors[idx])
                else:
                    color = (0, 255, 0)  # 未启用深度着色时使用绿色
                cv2.circle(cv_image, (u_int, v_int), self.point_radius, color, -1)
                in_frame += 1

        # 7) 左上角叠加统计信息，便于在线检查标定质量
        if self.show_overlay_stats:
            summary = f"LiDAR pts: {n_points}  front: {n_front}  in-frame: {in_frame}"
            cv2.putText(
                cv_image,
                summary,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            if self.max_depth_m is None:
                depth_text = f"Depth filter: z > {self.min_depth_m:.2f} m"
            else:
                depth_text = f"Depth filter: {self.min_depth_m:.2f} < z < {self.max_depth_m:.2f} m"
            cv2.putText(
                cv_image,
                depth_text,
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        out_msg.header = image_msg.header
        self.pub_image.publish(out_msg)


def main(args=None):
    """节点入口：初始化 ROS2，创建节点并 spin。"""
    rclpy.init(args=args)
    node = LidarCameraProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
