#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 LiDAR 与相机的安装方式生成轴映射配置，可粘贴到 config/qr_params.yaml 的 ros__parameters 下。

用法：
  python3 set_lidar_camera_axis.py
  python3 set_lidar_camera_axis.py x:z y:-x z:-y
  python3 set_lidar_camera_axis.py --lidar-x z --lidar-y -x --lidar-z -y

格式说明：
  每个 LiDAR 轴对应一个相机轴（含正负）：x, -x, y, -y, z, -z。
  例如 lidar_x 对应 相机的 z 轴 写为 x:z 或 --lidar-x z；
      lidar_y 对应 相机的 -x 轴 写为 y:-x 或 --lidar-y -x。
"""

import argparse
import sys

AXIS_OPTIONS = ["x", "-x", "y", "-y", "z", "-z"]


def parse_pair(s: str):
    """解析 'x:z' 或 'y:-x' 这种格式，返回 (lidar_axis, cam_expr)。"""
    s = s.strip().lower()
    if ":" in s:
        a, b = s.split(":", 1)
        return a.strip(), b.strip()
    return None, None


def validate_cam_axis(v: str) -> str:
    if v in AXIS_OPTIONS:
        return v
    if v in ("+x", "+y", "+z"):
        return v[1:]
    raise ValueError(f"无效的轴描述: {v}，应为 {AXIS_OPTIONS}")


def main():
    parser = argparse.ArgumentParser(
        description="生成 LiDAR-相机轴映射 YAML 配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pairs",
        nargs="*",
        metavar="lidar:cam",
        help="例如 x:z y:-x z:-y 表示 lidar_x->相机z, lidar_y->相机-x, lidar_z->相机-y",
    )
    parser.add_argument("--lidar-x", metavar="AXIS", help="相机轴对应 LiDAR X，如 z 或 -z")
    parser.add_argument("--lidar-y", metavar="AXIS", help="相机轴对应 LiDAR Y")
    parser.add_argument("--lidar-z", metavar="AXIS", help="相机轴对应 LiDAR Z")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="只打印 YAML 片段，不提示粘贴",
    )
    args = parser.parse_args()

    # 从 pairs 或 --lidar-x/y/z 得到映射：cam_x 来自哪条 lidar 轴
    # 配置里写的是：相机 X = ?*LiDAR_?，所以是 lidar_axis_cam_x = "-y" 表示 cam_x = -lidar_y
    # 用户输入的是：lidar_x -> cam z，即 cam_z = lidar_x => lidar_axis_cam_z = "x"
    # 所以需要从 (lidar_axis -> cam_axis) 反成 (cam_axis 由哪条 lidar 得到)
    lidar_to_cam = {}  # lidar_axis -> cam_axis expr, e.g. "x" -> "z", "y" -> "-x"

    if args.pairs:
        for p in args.pairs:
            la, ca = parse_pair(p)
            if la not in ("x", "y", "z"):
                print(f"忽略无效的 LiDAR 轴: {p}", file=sys.stderr)
                continue
            try:
                ca = validate_cam_axis(ca)
            except ValueError as e:
                print(e, file=sys.stderr)
                sys.exit(1)
            lidar_to_cam[la] = ca
    if args.lidar_x is not None:
        lidar_to_cam["x"] = validate_cam_axis(args.lidar_x)
    if args.lidar_y is not None:
        lidar_to_cam["y"] = validate_cam_axis(args.lidar_y)
    if args.lidar_z is not None:
        lidar_to_cam["z"] = validate_cam_axis(args.lidar_z)

    if len(lidar_to_cam) < 3:
        print("交互输入：请依次输入 LiDAR 的 X/Y/Z 轴分别对应相机的哪个轴（x/-x/y/-y/z/-z）")
        for ax in ("x", "y", "z"):
            if ax in lidar_to_cam:
                continue
            prompt = f"  LiDAR {ax.upper()} 轴 → 相机轴 (x/-x/y/-y/z/-z): "
            while True:
                v = input(prompt).strip().lower()
                if not v:
                    v = "z" if ax == "x" else "-x" if ax == "y" else "-y"
                try:
                    lidar_to_cam[ax] = validate_cam_axis(v)
                    break
                except ValueError as e:
                    print(e)

    # 反推：cam_x / cam_y / cam_z 各由哪条 lidar 轴得到（带符号）
    # lidar_axis_cam_x 表示 相机X = ?*LiDAR_?，所以若 lidar_y -> -x，则 cam_x = -lidar_y => lidar_axis_cam_x = "-y"
    cam_from_lidar = {"x": None, "y": None, "z": None}  # cam axis -> "±lidar?"
    for lidar_ax, cam_expr in lidar_to_cam.items():
        sign = "-" if cam_expr.startswith("-") else ""
        cam_ax = cam_expr.lstrip("-+")
        # 相机 cam_ax 轴 = sign + lidar_ax  => 配置项 lidar_axis_cam_<cam_ax> = sign + lidar_ax
        cam_from_lidar[cam_ax] = (sign + lidar_ax) if sign else lidar_ax

    if any(cam_from_lidar[v] is None for v in "xyz"):
        print("错误：必须为 LiDAR 的 x/y/z 各指定一个相机轴（且三个相机轴各出现一次）", file=sys.stderr)
        sys.exit(1)

    yaml = f"""    # LiDAR 与相机轴安装映射（用于圆心排序时的坐标对齐）
    # 格式：相机各轴由哪条 LiDAR 轴及符号得到。可选: "x", "-x", "y", "-y", "z", "-z"
    lidar_axis_cam_x: "{cam_from_lidar["x"]}"   # 相机 X = {"-" if cam_from_lidar["x"].startswith("-") else ""}LiDAR {cam_from_lidar["x"].strip("-").upper()}
    lidar_axis_cam_y: "{cam_from_lidar["y"]}"   # 相机 Y = {"-" if cam_from_lidar["y"].startswith("-") else ""}LiDAR {cam_from_lidar["y"].strip("-").upper()}
    lidar_axis_cam_z: "{cam_from_lidar["z"]}"   # 相机 Z = {"-" if cam_from_lidar["z"].startswith("-") else ""}LiDAR {cam_from_lidar["z"].strip("-").upper()}
"""

    print(yaml)
    if not args.print_only:
        print("请将以上内容复制到 config/qr_params.yaml 的 fast_calib -> ros__parameters 下（可覆盖已有 lidar_axis_cam_* 项）。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
