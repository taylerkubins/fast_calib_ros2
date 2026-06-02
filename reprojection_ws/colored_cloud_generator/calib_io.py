"""
标定文件 I/O：解析 FAST-LIVO2 格式的 calib_result.txt。

外参约定:
    T_cam_lidar (4x4): 将 LiDAR 系齐次坐标变换到相机系
    P_cam = Rcl @ P_lidar + Pcl

与 FAST-Calib_Ros2/include/common_lib.h 中 loadExtrinsicFromCalibResult 行为一致。
"""

from __future__ import annotations

import re
from typing import Tuple

import numpy as np

from .config_loader import CameraIntrinsics


def _parse_numbers(text: str):
    """从文本行中提取浮点数（支持科学计数法）。"""
    return [float(x) for x in re.findall(r"-?[\d.]+(?:[eE][+-]?\d+)?", text)]


def load_extrinsic_from_calib_result(calib_path: str) -> np.ndarray:
    """
    解析 calib_result.txt 中的 Rcl、Pcl，构造 4x4 齐次矩阵 T_cam_lidar。

    文件格式示例::
        Rcl: [ r11, r12, r13,
               r21, r22, r23,
               r31, r32, r33]
        Pcl: [ tx, ty, tz]

    Args:
        calib_path: calib_result.txt 绝对/相对路径

    Returns:
        shape (4, 4) 的 float64 变换矩阵

    Raises:
        ValueError: Rcl/Pcl 解析失败
    """
    with open(calib_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    r_vals = []
    p_vals = []
    reading_r = False
    for line in lines:
        if "Rcl:" in line:
            reading_r = True
        # Rcl 可能跨多行，逐行累积直到凑满 9 个元素
        if reading_r and len(r_vals) < 9:
            r_vals.extend(_parse_numbers(line))
            if len(r_vals) >= 9:
                reading_r = False
            continue
        if "Pcl:" in line:
            p_vals = _parse_numbers(line)
            break

    if len(r_vals) < 9 or len(p_vals) < 3:
        raise ValueError(f"无法从 {calib_path} 解析 Rcl/Pcl")

    T = np.eye(4, dtype=np.float64)
    R = np.array(r_vals[:9], dtype=np.float64).reshape(3, 3)
    T[:3, :3] = R
    T[:3, 3] = np.array(p_vals[:3], dtype=np.float64)
    return T


def load_intrinsics_from_calib_result(calib_path: str) -> CameraIntrinsics | None:
    """
    从 calib_result.txt 读取 cam_fx/fy/cx/cy 与 cam_d0~d3。

    Returns:
        解析成功返回 CameraIntrinsics，缺少关键字段时返回 None
    """
    with open(calib_path, "r", encoding="utf-8") as f:
        content = f.read()

    def _find(name: str):
        m = re.search(rf"{name}:\s*([\d.eE+-]+)", content)
        return float(m.group(1)) if m else None

    fx, fy, cx, cy = _find("cam_fx"), _find("cam_fy"), _find("cam_cx"), _find("cam_cy")
    if None in (fx, fy, cx, cy):
        return None

    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        k1=_find("cam_d0") or 0.0,
        k2=_find("cam_d1") or 0.0,
        p1=_find("cam_d2") or 0.0,
        p2=_find("cam_d3") or 0.0,
    )


def load_calibration(
    calib_path: str, fallback_intrinsics: CameraIntrinsics
) -> Tuple[np.ndarray, CameraIntrinsics]:
    """
    同时加载外参矩阵与相机内参。

    内参优先级:
        1. YAML 中 camera.fx/fy > 0 时，使用 YAML 内参（便于单独调参）
        2. 否则使用 calib_result.txt 中的 cam_fx 等
        3. 若 calib_result 也无内参，则回退到 YAML 默认值

    Args:
        calib_path: 外参标定结果文件
        fallback_intrinsics: YAML 中的 camera 段

    Returns:
        (T_cam_lidar, intrinsics)
    """
    T_cam_lidar = load_extrinsic_from_calib_result(calib_path)
    intrinsics = load_intrinsics_from_calib_result(calib_path) or fallback_intrinsics

    # YAML 显式配置了有效焦距时，覆盖 calib_result 内参
    if fallback_intrinsics.fx > 0 and fallback_intrinsics.fy > 0:
        intrinsics = fallback_intrinsics
    return T_cam_lidar, intrinsics
