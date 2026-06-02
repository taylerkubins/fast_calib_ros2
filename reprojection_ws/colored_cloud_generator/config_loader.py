"""
YAML 参数加载与数据结构定义。

配置文件示例见 reprojection_ws/config/reprojection_params.yaml。
所有路径字段在加载时会展开 ~ 并转为绝对路径。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class CameraIntrinsics:
    """
    针孔相机内参与畸变系数。

    对应 OpenCV 使用的 K 矩阵与 distCoeffs（k1,k2,p1,p2,k3）。
    """

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @property
    def camera_matrix(self):
        """3x3 相机内参矩阵 K。"""
        import numpy as np

        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist_coeffs(self):
        """(1,5) 畸变系数，供 cv2.undistort / projectPoints 使用。"""
        import numpy as np

        return np.array([[self.k1, self.k2, self.p1, self.p2, 0.0]], dtype=np.float64)


@dataclass
class ReprojectionConfig:
    """
    colored_cloud 生成所需的全部运行参数。

    Attributes:
        bag_path: ROS2 bag 目录或 .db3 文件路径
        lidar_topic: LiDAR PointCloud2 话题名
        image_topic: 相机 Image 话题；空字符串表示自动检测
        extrinsic_calib_path: FAST-LIVO2 calib_result.txt（含 Rcl/Pcl）
        output_path: 输出目录
        camera: YAML 中配置的相机内参（可覆盖 calib_result 内参）
        image_frame_index: 使用 bag 中第几帧图像（0=第一帧）
        max_lidar_frames: 最多合并多少帧点云，0=全部
        skip_rate: 点云均匀降采样步长
        save_pcd: 是否额外输出 colored_cloud.pcd
    """

    bag_path: str
    lidar_topic: str
    image_topic: str
    extrinsic_calib_path: str
    output_path: str
    camera: CameraIntrinsics
    image_frame_index: int = 0
    max_lidar_frames: int = 0
    skip_rate: int = 1
    save_pcd: bool = True


def _require_str(data: Dict[str, Any], key: str) -> str:
    """读取必填字符串参数并转为绝对路径。"""
    value = data.get(key)
    if not value or not str(value).strip():
        raise ValueError(f"参数 '{key}' 不能为空")
    return os.path.abspath(os.path.expanduser(str(value).strip()))


def load_config(config_path: str) -> ReprojectionConfig:
    """
    从 YAML 文件加载并校验配置。

    Args:
        config_path: reprojection_params.yaml 路径

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 必填项缺失或数值非法
    """
    config_path = os.path.abspath(os.path.expanduser(config_path))
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # ---------- 必填路径 ----------
    bag_path = _require_str(raw, "bag_path")
    extrinsic_calib_path = _require_str(raw, "extrinsic_calib_path")
    output_path = _require_str(raw, "output_path")

    # ---------- 话题与帧索引 ----------
    lidar_topic = str(raw.get("lidar_topic", "/livox/lidar")).strip()
    if not lidar_topic:
        raise ValueError("参数 'lidar_topic' 不能为空")

    image_topic = str(raw.get("image_topic", "")).strip()
    image_frame_index = int(raw.get("image_frame_index", 0))
    if image_frame_index < 0:
        raise ValueError("参数 'image_frame_index' 不能为负数")

    # ---------- 相机内参（YAML camera 段） ----------
    cam_raw = raw.get("camera") or {}
    camera = CameraIntrinsics(
        fx=float(cam_raw.get("fx", 0.0)),
        fy=float(cam_raw.get("fy", 0.0)),
        cx=float(cam_raw.get("cx", 0.0)),
        cy=float(cam_raw.get("cy", 0.0)),
        k1=float(cam_raw.get("k1", 0.0)),
        k2=float(cam_raw.get("k2", 0.0)),
        p1=float(cam_raw.get("p1", 0.0)),
        p2=float(cam_raw.get("p2", 0.0)),
    )

    return ReprojectionConfig(
        bag_path=bag_path,
        lidar_topic=lidar_topic,
        image_topic=image_topic,
        extrinsic_calib_path=extrinsic_calib_path,
        output_path=output_path,
        camera=camera,
        image_frame_index=image_frame_index,
        max_lidar_frames=int(raw.get("max_lidar_frames", 0)),
        skip_rate=max(1, int(raw.get("skip_rate", 1))),
        save_pcd=bool(raw.get("save_pcd", True)),
    )
