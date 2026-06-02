"""
点云投影着色逻辑。

复现 FAST-Calib_Ros2/include/common_lib.h::projectPointCloudToImage 的完整流程:

    LiDAR 点 (x,y,z)
        -> T_cam_lidar 变换到相机系
        -> 丢弃 Z < 0（相机后方）
        -> cv2.projectPoints 投影到像素 (u,v)
        -> 在 undistort 后的图像上取 BGR 颜色
        -> 输出: 相机系 XYZ + RGB

注意:
    - 输出点坐标是**相机坐标系**，不是 LiDAR 系
    - 图像先 undistort，投影时使用零畸变系数
    - PLY 中存储的 r/g/b 与 PCL PointXYZRGB 一致（OpenCV BGR 转 RGB）
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def project_point_cloud_to_image(
    cloud_xyz: np.ndarray,
    T_cam_lidar: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_bgr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    向量化实现：将 LiDAR 点云投影到图像并着色（默认使用此函数，速度更快）。

    Args:
        cloud_xyz: (N, 3) LiDAR 系点云
        T_cam_lidar: (4, 4) LiDAR -> Camera 外参
        camera_matrix: (3, 3) 内参 K
        dist_coeffs: (1, 5) 畸变系数（仅用于 undistort）
        image_bgr: (H, W, 3) 原始 BGR 图像

    Returns:
        points_cam: (M, 3) 相机系下成功着色的点
        colors_rgb: (M, 3) uint8 RGB 颜色
    """
    if cloud_xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    # 1) 整图去畸变（与 C++ cv::undistort 一致）
    undistorted = cv2.undistort(image_bgr, camera_matrix, dist_coeffs)
    h, w = undistorted.shape[:2]

    # 2) LiDAR 齐次坐标 -> 相机系: P_cam = T @ P_lidar
    T = T_cam_lidar.astype(np.float64)
    pts = cloud_xyz.astype(np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack((pts, ones))
    pts_cam = (T @ pts_h.T).T[:, :3]

    # 3) 仅保留相机前方的点（Z >= 0，与 C++ transformed_point(2) < 0 跳过对应）
    front_mask = pts_cam[:, 2] >= 0.0
    pts_cam = pts_cam[front_mask]
    if pts_cam.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    # 4) 投影到像素平面
    # 点已在相机系，外参已应用，故 rvec/tvec=0；图像已 undistort，畸变系数置零
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    zero_dist = np.zeros((5, 1), dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        pts_cam.reshape(-1, 1, 3),
        rvec,
        tvec,
        camera_matrix,
        zero_dist,
    )
    image_points = image_points.reshape(-1, 2)

    # 5) 像素坐标取整并过滤越界点（与 C++ static_cast<int> 向零截断一致）
    u = image_points[:, 0].astype(np.int64)
    v = image_points[:, 1].astype(np.int64)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    pts_cam = pts_cam[in_bounds]
    u = u[in_bounds]
    v = v[in_bounds]

    # 6) 从 undistort 图像取色：OpenCV BGR -> 输出 RGB（PLY 存储格式）
    colors_bgr = undistorted[v, u]
    colors_rgb = colors_bgr[:, ::-1].astype(np.uint8)

    return pts_cam.astype(np.float32), colors_rgb


def project_point_cloud_to_image_loop(
    cloud_xyz: np.ndarray,
    T_cam_lidar: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_bgr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    逐点循环版本，与 C++ 源码逻辑一一对应。

    用途:
        - 对照调试，验证向量化实现与 C++ 结果一致
        - 命令行加 --loop 时 pipeline 会调用此函数

    算法与 project_point_cloud_to_image 相同，仅实现方式不同。
    """
    if cloud_xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    undistorted = cv2.undistort(image_bgr, camera_matrix, dist_coeffs)
    h, w = undistorted.shape[:2]

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    zero_dist = np.zeros((5, 1), dtype=np.float64)

    points_out = []
    colors_out = []

    T = T_cam_lidar.astype(np.float64)
    for point in cloud_xyz:
        # 单点齐次变换
        p_h = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
        p_cam = T @ p_h
        if p_cam[2] < 0.0:
            continue

        img_pt, _ = cv2.projectPoints(
            np.array([[p_cam[0], p_cam[1], p_cam[2]]], dtype=np.float64),
            rvec,
            tvec,
            camera_matrix,
            zero_dist,
        )
        u = int(img_pt[0, 0, 0])
        v = int(img_pt[0, 0, 1])
        if u < 0 or u >= w or v < 0 or v >= h:
            continue

        b, g, r = undistorted[v, u]
        points_out.append([p_cam[0], p_cam[1], p_cam[2]])
        colors_out.append([r, g, b])

    if not points_out:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return np.asarray(points_out, dtype=np.float32), np.asarray(colors_out, dtype=np.uint8)
