"""
colored_cloud.ply 生成主流程。

串联各子模块，完成从配置到文件的完整 pipeline::

    load_config
        -> load_image_from_bag      (bag 内相机图像)
        -> load_calibration        (calib_result.txt 外参/内参)
        -> load_pointcloud_from_bag (bag 内 LiDAR 点云)
        -> project_point_cloud_to_image (投影着色)
        -> save_colored_cloud      (写出 ply/pcd)
"""

from __future__ import annotations

import os
from typing import Dict

from .bag_reader import load_image_from_bag, load_pointcloud_from_bag
from .calib_io import load_calibration
from .config_loader import ReprojectionConfig, load_config
from .ply_writer import save_colored_cloud
from .projection import project_point_cloud_to_image


def generate_colored_cloud(
    config_path: str,
    use_loop_projection: bool = False,
) -> Dict[str, object]:
    """
    从 YAML 参数文件生成 colored_cloud.ply（及可选 .pcd）。

    这是本包的核心入口函数，可由命令行或外部脚本直接调用。

    Args:
        config_path: reprojection_params.yaml 路径
        use_loop_projection: True 时使用逐点循环投影（与 C++ 逐行对应，调试用）

    Returns:
        结果字典，主要字段:
            ply_path / pcd_path: 输出文件路径
            input_points: 输入 LiDAR 点数
            colored_points: 成功着色点数
            retention_ratio: 着色保留率
            image_topic / image_frame_index: 实际使用的图像来源

    Raises:
        FileNotFoundError: 外参文件不存在
        RuntimeError: 投影结果为空（外参/话题/图像不匹配）
    """
    # ---------- 1. 加载 YAML 配置 ----------
    cfg = load_config(config_path)

    if not os.path.isfile(cfg.extrinsic_calib_path):
        raise FileNotFoundError(f"外参文件不存在: {cfg.extrinsic_calib_path}")

    # ---------- 2. 从 bag 读取一帧相机图像 ----------
    image, image_topic_used, image_frame_used = load_image_from_bag(
        cfg.bag_path,
        cfg.image_topic,
        frame_index=cfg.image_frame_index,
    )

    # ---------- 3. 加载外参 T_cam_lidar 与相机内参 K/D ----------
    T_cam_lidar, intrinsics = load_calibration(cfg.extrinsic_calib_path, cfg.camera)
    K = intrinsics.camera_matrix
    D = intrinsics.dist_coeffs

    # ---------- 4. 从 bag 读取并合并 LiDAR 点云 ----------
    cloud = load_pointcloud_from_bag(
        cfg.bag_path,
        cfg.lidar_topic,
        max_frames=cfg.max_lidar_frames,
        skip_rate=cfg.skip_rate,
    )

    # ---------- 5. 投影着色 ----------
    if use_loop_projection:
        from .projection import project_point_cloud_to_image_loop

        points_cam, colors_rgb = project_point_cloud_to_image_loop(cloud, T_cam_lidar, K, D, image)
    else:
        points_cam, colors_rgb = project_point_cloud_to_image(cloud, T_cam_lidar, K, D, image)

    if points_cam.shape[0] == 0:
        raise RuntimeError(
            "投影后彩色点云为空。请检查外参、内参、图像与 bag 是否匹配，"
            "以及 LiDAR 话题是否正确。"
        )

    # ---------- 6. 写出 colored_cloud.ply / .pcd ----------
    ply_path, pcd_path = save_colored_cloud(
        cfg.output_path,
        points_cam,
        colors_rgb,
        save_pcd=cfg.save_pcd,
    )

    ratio = points_cam.shape[0] / max(cloud.shape[0], 1)
    result = {
        "output_dir": cfg.output_path,
        "ply_path": ply_path,
        "pcd_path": pcd_path,
        "input_points": int(cloud.shape[0]),
        "colored_points": int(points_cam.shape[0]),
        "retention_ratio": ratio,
        "bag_path": cfg.bag_path,
        "image_topic": image_topic_used,
        "image_frame_index": image_frame_used,
        "extrinsic_calib_path": cfg.extrinsic_calib_path,
    }
    return result


def print_summary(result: Dict[str, object]) -> None:
    """在终端打印生成结果摘要（供命令行脚本调用）。"""
    print("[Result] colored cloud generation finished")
    print(f"  bag:      {result['bag_path']}")
    print(f"  image:    topic={result['image_topic']}  frame={result['image_frame_index']}")
    print(f"  extrinsic:{result['extrinsic_calib_path']}")
    print(f"  output:   {result['ply_path']}")
    if result.get("pcd_path"):
        print(f"  pcd:      {result['pcd_path']}")
    print(
        f"  points:   input={result['input_points']}  "
        f"colored={result['colored_points']}  "
        f"ratio={result['retention_ratio']:.2%}"
    )
