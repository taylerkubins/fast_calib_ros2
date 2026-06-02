#!/usr/bin/env python3
"""
独立工具：从 bag + 图像 + 外参/内参生成 colored_cloud.ply。

用法:
  python3 generate_colored_cloud.py --config config/reprojection_params.yaml

与 FAST-Calib_Ros2 的关系:
  复现 common_lib.h::projectPointCloudToImage + saveCalibrationResults 中的
  colored_cloud 生成逻辑，不依赖标定工程编译与运行。
"""

from __future__ import annotations

import argparse
import os
import sys

# 允许直接运行脚本：python3 reprojection_ws/generate_colored_cloud.py
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from colored_cloud_generator.pipeline import generate_colored_cloud, print_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="从 bag 与标定参数生成 colored_cloud.ply")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_ROOT, "config", "reprojection_params.yaml"),
        help="YAML 参数文件路径",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="使用与 C++ 逐点循环完全一致的投影实现（调试用）",
    )
    args = parser.parse_args()

    try:
        result = generate_colored_cloud(args.config, use_loop_projection=args.loop)
        print_summary(result)
        return 0
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
