"""
colored_cloud_generator 包入口。

本包独立于 FAST-Calib_Ros2，用于从 ROS2 bag、标定外参/内参生成
与标定工程一致的 colored_cloud.ply 彩色点云。

对外 API:
    generate_colored_cloud(config_path) -> dict
        读取 YAML 配置，完成 bag 解析、投影着色、文件写出。
"""

from .pipeline import generate_colored_cloud

__all__ = ["generate_colored_cloud"]
