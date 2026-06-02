#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能（ROS2 版本）：
1) 自动检测 rosbag2 中雷达点云类型：
   - sensor_msgs/msg/PointCloud2  (如 /hesai/pandar)
   - livox_ros_driver/CustomMsg (如 /livox/lidar)
2) 按各自的解析方式把点云导出成一个带 intensity 的 PCD 文件 (x y z intensity, ASCII)
3) 使用 Open3D 对该 PCD 进行交互选点（至少 4 个点），并根据 4 个点计算包围范围，
   保存为同名 .txt 文件。

依赖：
    - rosbag2_py, rclpy, rosidl_runtime_py
    - sensor_msgs, sensor_msgs_py
    - open3d, numpy
    - （若使用 Livox CustomMsg）livox_ros_driver2 或 livox_ros_driver

用法示例：
    python3 distance_filter_tool.py
    python3 distance_filter_tool.py /path/to/rosbag2_dir
    # rosbag2 目录为包含 metadata.yaml 和 .mcap 或 .db3 的文件夹
    # PCD 与范围 .txt 将保存到与输入路径相同的目录
"""

import os
import sys
import numpy as np

try:
    import open3d as o3d
except ModuleNotFoundError:
    print("[ERROR] 未安装 open3d。请执行: pip3 install open3d --user", file=sys.stderr)
    sys.exit(1)

# ROS2 相关导入（离线工具，仅在需要读 bag 时使用）
try:
    import rclpy
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
    from sensor_msgs_py.point_cloud2 import read_points as read_points_pc2
except ImportError as e:
    print("[ERROR] ROS2 环境未 source 或缺少依赖。请先执行: source /opt/ros/<distro>/setup.bash", file=sys.stderr)
    print(f"  详细: {e}", file=sys.stderr)
    sys.exit(1)

# ===================== 通用：保存 PCD =====================

def save_pcd_with_intensity(points, intensities, output_path):
    """
    保存点云为带 intensity 字段的 PCD 文件 (ASCII 格式)
    points: list/ndarray of [x, y, z]
    intensities: list/ndarray of intensity
    """
    N = len(points)
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {N}
HEIGHT 1
POINTS {N}
DATA ascii
"""
    with open(output_path, 'w') as f:
        f.write(header)
        for (x, y, z), inten in zip(points, intensities):
            f.write(f"{x} {y} {z} {inten}\n")
    print(f"[PCD] 保存带 intensity 字段的点云到: {output_path}")


# ===================== ROS2 bag 路径与存储格式 =====================

def get_bag_uri_and_storage(bag_path):
    """
    解析用户输入的 bag 路径，返回 (uri, storage_id)。
    rosbag2 的 bag 是一个目录，包含 metadata.yaml 和 .mcap 或 .db3 文件。
    若用户传入的是 .mcap 或 .db3 文件路径，则使用其所在目录为 uri。
    """
    bag_path = os.path.abspath(bag_path)
    if os.path.isfile(bag_path):
        lower = bag_path.lower()
        if lower.endswith('.mcap'):
            return os.path.dirname(bag_path), 'mcap'
        if lower.endswith('.db3'):
            return os.path.dirname(bag_path), 'sqlite3'
        print(f"[WARN] 传入的是文件但非 .mcap/.db3，尝试作为目录名失败: {bag_path}", file=sys.stderr)
        return bag_path, 'mcap'  # 默认尝试 mcap
    if os.path.isdir(bag_path):
        # 自动检测目录内是 mcap 还是 sqlite3
        for f in os.listdir(bag_path):
            if f.endswith('.mcap'):
                return bag_path, 'mcap'
            if f.endswith('.db3'):
                return bag_path, 'sqlite3'
        return bag_path, 'mcap'
    return bag_path, 'mcap'


# ===================== 情况 1：PointCloud2 =====================

def find_intensity_field(msg):
    """在 PointCloud2 的 fields 中自动检测强度字段名称"""
    candidates = ["intensity", "reflectivity", "i", "ref"]
    for field in msg.fields:
        if field.name.lower() in candidates:
            return field.name
    return None


def convert_pointcloud2_bag_to_pcd(
    bag_uri,
    storage_id,
    output_dir,
    topic_name="/hesai/pandar",
    pcd_name="sensor_PointCloud2_inten_ascii.pcd",
):
    """
    将 rosbag2 中 PointCloud2 类型的点云合并导出为一个 PCD 文件。
    保持原始雷达坐标，不做坐标变换。
    """
    print(f"[Bag] 打开 rosbag2: uri={bag_uri}, storage_id={storage_id}")
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='',
        output_serialization_format=''
    )
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic_name not in topic_types:
        print(f"[ERROR] Topic '{topic_name}' 不在 bag 中。可用: {list(topic_types.keys())}", file=sys.stderr)
        return None

    type_name = topic_types[topic_name]
    if type_name != 'sensor_msgs/msg/PointCloud2':
        print(f"[ERROR] Topic '{topic_name}' 类型为 {type_name}，非 PointCloud2。", file=sys.stderr)
        return None

    msg_class = get_message(type_name)
    intensity_field = None
    all_points = []
    all_intensities = []

    print(f"[Bag] 从 topic '{topic_name}' 读取 PointCloud2 点云...")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic != topic_name:
            continue
        try:
            msg = deserialize_message(data, msg_class)
            if intensity_field is None:
                intensity_field = find_intensity_field(msg)
                if not intensity_field:
                    print("[ERROR] 未找到强度字段! 退出 PointCloud2 转换。", file=sys.stderr)
                    return None
                print(f"[Bag] 检测到 intensity 字段: {intensity_field}")
            field_names = ["x", "y", "z", intensity_field]
            for point in read_points_pc2(msg, field_names=field_names, skip_nans=True):
                all_points.append([point[0], point[1], point[2]])
                all_intensities.append(point[3])
        except Exception as e:
            print(f"[ERROR] 读取错误: {str(e)}", file=sys.stderr)
            continue

    if not all_points:
        print("[ERROR] 未找到 PointCloud2 点云数据！", file=sys.stderr)
        return None

    output_path = os.path.join(output_dir, pcd_name)
    save_pcd_with_intensity(all_points, all_intensities, output_path)
    return output_path


# ===================== 情况 2：Livox CustomMsg =====================

def parse_livox_custom_msg(msg):
    """
    从 livox_ros_driver CustomMsg 中解析 x, y, z, reflectivity。
    假设 msg.points 是 CustomPoint 对象列表，字段为 x, y, z, reflectivity。
    """
    points = []
    intensities = []
    for pt in msg.points:
        points.append([pt.x, pt.y, pt.z])
        intensities.append(pt.reflectivity)
    return points, intensities


def convert_livox_custom_bag_to_pcd(
    bag_uri,
    storage_id,
    output_dir,
    topic_name="/livox/lidar",
    pcd_name="livox_CustomMsg_inten_ascii.pcd",
):
    """
    将 rosbag2 中 Livox CustomMsg 类型的点云合并导出为一个 PCD 文件。
    """
    print(f"[Bag] 打开 rosbag2: uri={bag_uri}, storage_id={storage_id}")
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='',
        output_serialization_format=''
    )
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic_name not in topic_types:
        print(f"[ERROR] Topic '{topic_name}' 不在 bag 中。可用: {list(topic_types.keys())}", file=sys.stderr)
        return None

    type_name = topic_types[topic_name]
    if 'CustomMsg' not in type_name or 'livox' not in type_name.lower():
        print(f"[ERROR] Topic '{topic_name}' 类型为 {type_name}，非 Livox CustomMsg。", file=sys.stderr)
        return None

    try:
        msg_class = get_message(type_name)
    except Exception as e:
        print(f"[ERROR] 无法加载消息类型 {type_name}，请确认已安装 livox_ros_driver2 或 livox_ros_driver: {e}", file=sys.stderr)
        return None

    all_points = []
    all_intensities = []

    print(f"[Bag] 从 topic '{topic_name}' 读取 CustomMsg 点云...")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic != topic_name:
            continue
        try:
            msg = deserialize_message(data, msg_class)
            pts, intens = parse_livox_custom_msg(msg)
            all_points.extend(pts)
            all_intensities.extend(intens)
        except Exception as e:
            print(f"[ERROR] 读取错误: {str(e)}", file=sys.stderr)
            continue

    if not all_points:
        print("[ERROR] 未找到 Livox CustomMsg 点云数据!", file=sys.stderr)
        return None

    output_path = os.path.join(output_dir, pcd_name)
    intensities = np.array(all_intensities, dtype=np.float32)
    save_pcd_with_intensity(all_points, intensities, output_path)
    return output_path


# ===================== 自动检测：这个 bag 用哪种方式 =====================

def detect_lidar_msg_type(bag_uri, storage_id):
    """
    在 bag 里扫一圈，检测是否有 PointCloud2 或 Livox CustomMsg。
    返回：("PointCloud2", topic_name) 或 ("CustomMsg", topic_name) 或 (None, None)。
    若两种都有，默认优先 PointCloud2。
    """
    print(f"[Detect] 扫描 bag: uri={bag_uri}, storage_id={storage_id}")
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='',
        output_serialization_format=''
    )
    reader.open(storage_options, converter_options)

    pc2_topics = []
    livox_topics = []
    for t in reader.get_all_topics_and_types():
        if t.type == 'sensor_msgs/msg/PointCloud2':
            pc2_topics.append(t.name)
        elif 'livox' in t.type.lower() and 'CustomMsg' in t.type:
            livox_topics.append(t.name)

    if pc2_topics and livox_topics:
        print("[Detect] 同时检测到 PointCloud2 和 Livox CustomMsg，默认使用 PointCloud2。")
        return "PointCloud2", pc2_topics[0]
    elif pc2_topics:
        print("[Detect] 检测到 PointCloud2 点云。")
        return "PointCloud2", pc2_topics[0]
    elif livox_topics:
        print("[Detect] 检测到 Livox CustomMsg 点云。")
        return "CustomMsg", livox_topics[0]
    else:
        print("[Detect] 未检测到 PointCloud2 或 Livox CustomMsg 点云。")
        return None, None


# ===================== Open3D 交互选点 & 保存范围 =====================

def select_and_save_points(pcd_folder, target_pcd_name):
    """
    在给定目录中读取指定 PCD 文件，用 Open3D 交互式选点并保存范围。
    使用 VisualizerWithVertexSelection，Shift+左键选点，选满后按 Q 关闭。
    """
    pcd_path = os.path.join(pcd_folder, target_pcd_name)
    if not os.path.isfile(pcd_path):
        print(f"[ERROR] 指定的 PCD 文件不存在: {pcd_path}", file=sys.stderr)
        return

    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        print(f"[ERROR] {target_pcd_name} 中没有点云数据，已跳过", file=sys.stderr)
        return

    print(f"\n正在处理: {target_pcd_name}")
    print("操作说明：")
    print("  1) 先按住 Shift 键不松")
    print("  2) 用鼠标左键在点云上点击要选的角点（每点一次选一个，选中会高亮）")
    print("  3) 至少选 4 个点（标定板四角），选满后按 Q 关闭窗口")
    print("  4) 滚轮可大幅放大以便在标定板上精确选点")

    # 使用 VisualizerWithVertexSelection，选点更稳定，返回带坐标的 PickedPoint
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name=f"选择点 - {target_pcd_name}", width=1280, height=720)
    vis.add_geometry(pcd)

    # 允许更大放大倍数：近裁剪平面设得很小，滚轮可放大到极近距离
    ctrl = vis.get_view_control()
    ctrl.set_constant_z_near(0.00001)   # 0.01mm，支持极近距离观察
    ctrl.set_constant_z_far(1e6)
    # 视场角设为最小(约 5°)，等同长焦，同样滚轮位移获得更大放大倍数
    try:
        ctrl.change_field_of_view(-55)  # 从默认约 60° 减到约 5°
    except Exception:
        pass

    # 增大点云显示尺寸，便于在标定板区域精确选点
    opt = vis.get_render_option()
    opt.point_size = 4.0

    vis.run()
    # 必须在 destroy_window 之前获取选点（否则可能被清空）
    picked = vis.get_picked_points()
    vis.destroy_window()

    if not picked:
        print(f"[ERROR] 未选择任何点，{target_pcd_name} 没有保存文件", file=sys.stderr)
        print("  请确认：先按住 Shift 再左键点击点云上的点，选满 4 个后按 Q。", file=sys.stderr)
        return
    if len(picked) < 4:
        print(f"[ERROR] 只选中了 {len(picked)} 个点，少于 4 个，跳过该文件", file=sys.stderr)
        return

    # PickedPoint 的 .coord 为 numpy 数组 [x,y,z]
    selected_points = np.array([np.asarray(p.coord) for p in picked[:4]])

    mins = selected_points.min(axis=0)
    maxs = selected_points.max(axis=0)
    x_min = mins[0] - 0.2
    x_max = maxs[0] + 0.2
    y_min = mins[1] - 0.2
    y_max = maxs[1] + 0.2
    z_min = mins[2] - 0.2
    z_max = maxs[2] + 0.2

    base_name = os.path.splitext(target_pcd_name)[0]
    save_file = os.path.join(pcd_folder, f"{base_name}.txt")
    with open(save_file, 'w') as f:
        f.write("# 4 selected points (x y z)\n")
        for p in selected_points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        f.write("# range values in order:\n")
        f.write(f"x_min: {x_min:.1f}\n")
        f.write(f"x_max: {x_max:.1f}\n")
        f.write(f"y_min: {y_min:.1f}\n")
        f.write(f"y_max: {y_max:.1f}\n")
        f.write(f"z_min: {z_min:.1f}\n")
        f.write(f"z_max: {z_max:.1f}\n")
    print(f"[Save] 已保存选点与范围到: {save_file}")
    print("点云处理完成。")


# ===================== main =====================

def main():
    # 解析命令行参数（仅需指定输入路径，输出与输入目录一致）
    if len(sys.argv) > 1:
        bag_input = sys.argv[1]
    else:
        bag_input = os.path.join(os.getcwd(), "rosbag2_2025_09_09-13_50_55")
        print(f"未指定 bag 路径，默认使用: {bag_input}")

    bag_uri, storage_id = get_bag_uri_and_storage(bag_input)
    if not os.path.isdir(bag_uri):
        print(f"[ERROR] bag 路径 '{bag_uri}' 不存在或不是目录", file=sys.stderr)
        sys.exit(1)

    output_dir = bag_uri
    print(f"输出目录与输入路径一致: {output_dir}")

    # 初始化 rclpy（用于类型支持/序列化）
    rclpy.init()

    try:
        msg_type, topic_name = detect_lidar_msg_type(bag_uri, storage_id)
        if msg_type is None:
            print("[ERROR] 未检测到支持的雷达消息类型，退出。", file=sys.stderr)
            sys.exit(1)

        if msg_type == "PointCloud2":
            pcd_path = convert_pointcloud2_bag_to_pcd(
                bag_uri=bag_uri,
                storage_id=storage_id,
                output_dir=output_dir,
                topic_name=topic_name,
                pcd_name="sensor_PointCloud2_inten_ascii.pcd",
            )
        else:
            pcd_path = convert_livox_custom_bag_to_pcd(
                bag_uri=bag_uri,
                storage_id=storage_id,
                output_dir=output_dir,
                topic_name=topic_name,
                pcd_name="livox_CustomMsg_inten_ascii.pcd",
            )

        if pcd_path is None:
            print("[ERROR] PCD 生成失败，退出。", file=sys.stderr)
            sys.exit(1)

        select_and_save_points(
            pcd_folder=output_dir,
            target_pcd_name=os.path.basename(pcd_path),
        )
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
