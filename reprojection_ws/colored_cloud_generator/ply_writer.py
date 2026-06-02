"""
彩色点云文件写出。

输出格式与 FAST-Calib saveCalibrationResults 中 PCL 写出一致:
    - colored_cloud.ply: ASCII PLY，字段 x y z red green blue (uchar)
    - colored_cloud.pcd: ASCII PCD，字段 x y z rgb（可选）

可用 CloudCompare / MeshLab / PCL 直接打开。
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def save_colored_ply(
    filepath: str,
    points: np.ndarray,
    colors_rgb: np.ndarray,
) -> None:
    """
    写入 PCL 兼容的 ASCII PLY 文件。

    文件头格式与 pcl::io::savePLYFile 输出一致::
        property float x/y/z
        property uchar red/green/blue

    Args:
        filepath: 输出 .ply 路径
        points: (N, 3) 相机系 XYZ（float）
        colors_rgb: (N, 3) uint8 RGB

    Raises:
        ValueError: 点云为空
    """
    if points.shape[0] == 0:
        raise ValueError("彩色点云为空，无法保存 PLY")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
    n = points.shape[0]
    colors = colors_rgb.astype(np.uint8)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment PCL generated\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("element face 0\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def save_colored_pcd_ascii(
    filepath: str,
    points: np.ndarray,
    colors_rgb: np.ndarray,
) -> None:
    """
    写入 PCL 兼容的 ASCII PCD 文件（FIELDS x y z rgb）。

    rgb 字段为 32 位打包整数: (R << 16) | (G << 8) | B

    Args:
        filepath: 输出 .pcd 路径
        points: (N, 3) 相机系 XYZ
        colors_rgb: (N, 3) uint8 RGB
    """
    if points.shape[0] == 0:
        raise ValueError("彩色点云为空，无法保存 PCD")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
    colors = colors_rgb.astype(np.uint8)
    n = points.shape[0]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F U\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {n}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")
        for i in range(n):
            x, y, z = points[i]
            r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
            rgb_packed = (r << 16) | (g << 8) | b
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {rgb_packed}\n")


def save_colored_cloud(
    output_dir: str,
    points: np.ndarray,
    colors_rgb: np.ndarray,
    save_pcd: bool = True,
) -> Tuple[str, str | None]:
    """
    将彩色点云写入 output_dir，文件名与 FAST-Calib 保持一致。

    Args:
        output_dir: 输出目录（不存在则自动创建）
        points: (N, 3) 相机系彩色点坐标
        colors_rgb: (N, 3) RGB 颜色
        save_pcd: 是否额外写出 colored_cloud.pcd

    Returns:
        (ply_path, pcd_path)，未写 pcd 时 pcd_path 为 None
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ply_path = os.path.join(output_dir, "colored_cloud.ply")
    save_colored_ply(ply_path, points, colors_rgb)

    pcd_path = None
    if save_pcd:
        pcd_path = os.path.join(output_dir, "colored_cloud.pcd")
        save_colored_pcd_ascii(pcd_path, points, colors_rgb)

    return ply_path, pcd_path
